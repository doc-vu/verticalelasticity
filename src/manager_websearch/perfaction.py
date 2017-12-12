from cloudmodel import *
import logging

# class that exectutes performance based actions
class PerfAction(object):

	def __init__(self, container_manager, host, container, min_cores, max_cores, model):
		#self.perfmodel = CloudModel(winSize = 500, gpWinSize = 40 ,featureWinSize=2, sigma=0.8)
		#self.perfmodel = CloudModel(winSize = 4.5, sigma=0.8)
		#self.perfmodel = create_cloud_model('THRESHOLD')

		self.model = model
		self.perfmodel = create_cloud_model(model)
		#self.perfmodel = create_cloud_model('THRESHOLD_LATENCY')
		#self.perfmodel = create_cloud_model('SHS')
		self.host = host
		self.container = container
		self.current_corecount = container_manager.getCurrentCoreCount(host, container)
		self.min_cores = min_cores
		self.max_cores = max_cores
		self.container_manager = container_manager
		self.max_corecount = container_manager.getMaxCoreCount(host)
		#print  self.max_corecount
		self.resetCounter()

	# return the current core count
	def getCurrentCoreCount(self):
		return self.current_corecount

	def resetCounter(self):
		self.counter =  3
		self.latency = 0.0
		self.wt_latency = 0.0
		self.thruput = 0.0
		# modify for selected metrics for model processing
		self.target_stats= [0.0,]
		self.host_stats= [0.0,0.0,0.0,0.0,]

	# invokes the performance model and executes update if anything has changed
	def invokeModel(self, throughput, latency, weighted_latency, utilization_stats):
		#print utilization_stats
		self.latency += latency
		self.wt_latency += weighted_latency
		self.thruput += throughput
	
		# current keys
		# ['web_cpupercent', 'web_disk', 'web_net', 'web_memory', 'web_cpupercent_norm', 'web_disk_norm', 'web_net_norm', 'web_memory_norm', 
		# 'host_disk_io', 'host_net_io', 'host_memory', 'host_cpupercent', 'host_disk_io_norm', 'host_net_io_norm', 'host_memory_norm', 'host_cpupercent_norm', 
		# 'host_llc_bw', 'host_mem_bw', 'host_cache_ref', 'host_IPC', 'host_IPS', 'host_cs', 'host_cache_misses', 'host_kvm_exit',
		# 'host_sched_iowait', 'host_sched_switch', 'host_sched_wait', 'host_mem_bw_norm', 'host_cache_ref_norm',
		# 'host_IPC_norm', 'host_IPS_norm', 'host_cs_norm', 'host_cache_misses_norm', 'host_sched_iowait_norm', 'host_sched_switch_norm', 'host_sched_waiti_norm

		# web_cpupercent_norm
		self.target_stats[0] += utilization_stats[4]

		if len(utilization_stats) < 35:
			logging.warning("Problen with data during invoke")
		else:
			# host_IPSnorm, host_ chache_refnorm, host_ netIO_norm,  host_ schedwait_norm 
			self.host_stats[0] += utilization_stats[29]
			self.host_stats[1] += utilization_stats[28]
			self.host_stats[2] += utilization_stats[13]
			self.host_stats[3] += utilization_stats[34]
		
		
		logging.info('aggregating data set for model processing')
		if self.counter > 1:
			logging.info('wait for more records before model processing')
			self.counter -= 1
			return  (self.current_corecount, )

		self.latency = round(self.latency/3, 3)
		self.wt_latency = round(self.wt_latency/3, 3)
		self.thruput = round(self.thruput/3, 3)

		# throughput is expected to be sum
		#self.thruput = round(self.thruput/3, 3)

		for i in range(0,1):
			self.target_stats[i] = round(self.target_stats[i]/3, 3)
		for i in range(0,4):
			self.host_stats[i] = round(self.host_stats[i]/3, 3)
		
		logging.info("inputs: " + str(self.thruput) + " , " + str(self.target_stats) + " , " + str(self.latency) + " , " + str(self.host_stats))
			
		predicted_core_count = self.perfmodel.sysControl(self.thruput, self.target_stats, self.wt_latency,self.host_stats,
		#predicted_core_count = self.perfmodel.sysControl(self.thruput, self.target_stats, self.latency,self.host_stats,
						self.current_corecount, self.max_cores, self.min_cores)

		print 'predicted_core_count  ' + str(predicted_core_count)
		#to return
		latency = self.latency
		wt_latency = self.wt_latency
                thru = self.thruput
                stat = self.target_stats
		self.resetCounter()
		logging.info('current predicted core count is {0}'.format(predicted_core_count))
		if (predicted_core_count != self.current_corecount and predicted_core_count >= self.min_cores and predicted_core_count <= self.max_cores):
		#if True:
			logging.info('updating current core count.. ')
			#self.container_manager.updateCoreCount(self.host, self.container, predicted_core_count)
			try:
				self.container_manager.updateResources(self.host, self.container, [predicted_core_count,0], self.current_corecount)
			except Exception as e:
				print e.message
				logging.error(e.message)
				
			self.current_corecount = self.container_manager.getCurrentCoreCount(self.host, self.container)
		return  (self.current_corecount,  latency, wt_latency,  thru, stat) 

	# evaluates the performance model
	def evalModel(self):
		result =  self.perfmodel.evalModel()
		logging.info('performance model evaluation result: {0}'.format(result))
		return result
	
	# plots the result for model evaluation
	def plotResults(self):
		logging.info('plotting evaluation chart ...')
		self.perfmodel.plot_results()

	# plots the result for model evaluation
	def exportResults(self):
		logging.info('exporting evaluation chart data...')
		self.perfmodel.export_results()
