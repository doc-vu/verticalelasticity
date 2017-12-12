#from  docker import Client
import docker
from abc import ABCMeta, abstractmethod
from  fabric.api import execute, env, run, settings
import logging
import threading

class FabricException(Exception):
	pass

def run_task(name, host, is_res, manobj):
	with settings(abort_exception = FabricException):
		try:
			if not is_res:
				execute(cp_task, name, hosts=host)
			else: 
				execute(res_task, name, hosts=host)
		except FabricException as e:
			logging.warn("Error during checkpoint/restore action: " + e.message)
	manobj.isprocessing = False

def cp_task(name):
	cmd = "/root/cp_scripts/cptask.sh " + name
	run(cmd)
	logging.info(' done checkpoint: ' + name)
	
def res_task(name):
	logging.info(' start restoring: ' + name)
	cmd = "/root/cp_scripts/restask.sh " + name
	run(cmd)
	logging.info(' done restoring: ' + name)
	
class ContainerManager(object):
	__metaclass__ = ABCMeta	
	
	@abstractmethod
	def getCurrentCoreCount(self, host, container):
		return 
		
	@abstractmethod
	def getMaxCoreCount(self, host):
		return 

class LXCCloudManager(ContainerManager):

	def __init__(self, manager_ip, manager_port):
		logging.info('instantiating docker client for maanger at %s:%s',manager_ip,manager_port)
		self.cli = docker.DockerClient(base_url='tcp://'+manager_ip+':'+manager_port, version='1.24')

		# configure fabric
		env.user = 'root'
		self.manager_host = [manager_ip]
		# hard coding for no
		self.maxCoreCount = 12
		self.period = 100000
		self.target_cores = ''
		self.allowed_cores = '0-' + str(self.maxCoreCount-1)
		self.checkpointed_containers = []
		self.co_containers = []
		self.isprocessing = False

	def getCurrentCoreCount(self, host, container):
		details = self.cli.containers.get(container).attrs
		cpu_string = details["HostConfig"].get("CpusetCpus")
		cpus = cpu_string.split(",")
		cpucount=0
		for cpu in cpus:
			if '-' in cpu:
				vals = cpu.split('-')
				for i in range(int(vals[0]),int(vals[1])+1):
					cpucount += 1
			else:
				cpucount += 1
		logging.info('core count for container %s is %d',container, cpucount)
		return cpucount
			

	def getCurrentCoreCountOld(self, container):
		details = self.cli.containers.get(container).attrs
		quota = details["HostConfig"].get("CpuQuota")
		period = details["HostConfig"].get("CpuPeriod")
		if period == 0:
			return 0
		if period is None:
			period = 100000
		self.period = period
		if quota == -1:
			quota = period*self.maxCoreCount
		core = quota/period
		return core

	def getContainers(self):
                return self.cli.containers.list()

	def getCheckpointedCount(self):
		return len(self.checkpointed_containers)
		
	def getMaxCoreCount(self, host):
		# hard coding for no
		logging.info('returning hard coded max cores for now')
		return self.maxCoreCount

	def checkpointContainer(self):
		cpcount = len(self.checkpointed_containers)
		name = 'cont-' + str(cpcount)
		logging.info(' Starting thread to checkpoint: ' + name)
		threading.Thread(target=run_task, args=[name, self.manager_host, False, self]).start()
		#self.pauseContainer(name)
		self.checkpointed_containers.append(str(cpcount))
		#logging.info(' cp effective count: ' + str(len(self.checkpointed_containers)))
		logging.info("after checkpoint: " + str(self.checkpointed_containers))

	
	def restoreContainer(self):
		cpcount = len(self.checkpointed_containers)
		if cpcount == 0:
			return
		name = 'cont-' + str(cpcount-1)
		logging.info(' Starting thread to restore: ' + name)
		threading.Thread(target=run_task, args=[name, self.manager_host, True, self]).start()
		#self.unpauseContainer(name)
		self.checkpointed_containers.remove(str(cpcount-1))
		#logging.info(' cp effective count: ' + str(len(self.checkpointed_containers)))
		logging.info("after restore: " + str(self.checkpointed_containers))


	def updateResources(self, host_name, container_name, resource, curr_cores=0):
		#logging.info('Updating container %s corecount %d on host %s', container_name, cores, host_name)
		logging.info("checkpoint status: " + str(self.isprocessing))
		if self.isprocessing:
			print "Still Processing last one"
			return
		cores = resource[0]
		memory = resource[1]
		cores = 5

		# need to check if we are increasing or reducing core count
		if curr_cores != 0:
			# increase the number of cores
			logging.info("before cp: " + str(self.checkpointed_containers))
			logging.info("before cp all conts: " + str(self.co_containers))
			total_needed_cores = cores+2*(len(self.co_containers) - len(self.checkpointed_containers))
			print "total_needed_cores: " + str(total_needed_cores)
			logging.info("total_needed_cores: " + str(total_needed_cores))	
			#if cores > curr_cores and total_needed_cores > self.maxCoreCount:
			if total_needed_cores > self.maxCoreCount:
				print "checkpoint now"
				logging.info("checkpoint now")
				self.isprocessing = True
				self.checkpointContainer()
				#cores = 10
			#elif cores <= curr_cores and  len(self.checkpointed_containers)> 0  and total_needed_cores < self.maxCoreCount - 1:
			elif  len(self.checkpointed_containers)> 0  and total_needed_cores < self.maxCoreCount - 1:
				self.isprocessing = True
				self.restoreContainer()
		

		node_mid = self.maxCoreCount/2
		core_mid = cores/2
		target_start = node_mid - core_mid
		target_end = target_start + cores - 1
		target_cores = str(target_start) + '-' + str(target_end)
		
		
		#target_cores = '0-'+str(cores-1)
		target_memory = str(memory) + 'm'
		#other_cores = str(cores) + '-' + str(self.maxCoreCount-1)

		allowed_cores = ''
		if target_start == 1:
			allowed_cores = '0'
		elif target_start != 0:
			allowed_cores = '0-' + str(target_start-1)

		if target_end == (self.maxCoreCount - 2):
			allowed_cores += ',' + str(self.maxCoreCount - 1)
		elif target_end != (self.maxCoreCount - 1):
			allowed_cores += ',' + str(target_end + 1) + '-' + str(self.maxCoreCount - 1)


                containers = self.cli.containers.list(all=True)
		for container in containers:
			info = container.attrs
			cont_name = info.get('Name')[1:]

			# somhow this is not working 
			#if info.get('Node') is not None and info['Node'].get('Name') == host_name:
			if cont_name == container_name:
				logging.info('Updating container %s with corecount  and memory %s %s ', cont_name, target_cores, target_memory)
				
				# hack to not control memory for now
				if memory == 0:
					container.update(cpuset_cpus=target_cores)
				else:
					container.update(cpuset_cpus=target_cores, mem_limit=target_memory)
			#elif info.get('Node') is not None and info['Node'].get('Name') == host_name:
			else:
				logging.info('Updating container %s with allowed cores %s', cont_name, allowed_cores)
				container.update(cpuset_cpus=allowed_cores)

		self.target_cores=target_cores
		self.allowed_cores=allowed_cores


	def updateCoreCountOld(self, container, cores):
                container_obj = self.cli.containers.get(container)
		details =  container_obj.attrs
		period = details["HostConfig"].get("CpuPeriod")
		if period is None:
			period = self.period
                self.cli.containers.get(container).update(cpu_quota=(cores*self.period))

	def startContainer(self, container):
		logging.info('Starting container %s', container)
                container_obj = self.cli.containers.get(container)
		return container_obj.start()

	def removeContainer(self, container):
		name = 'cont-' + container
		if container in self.checkpointed_containers:
			#logging.info("removing CP container: ")
			logging.info("before remove: " + str(self.checkpointed_containers))
			# if using pause instead of checkpoint
			#self.unpauseContainer(name)
			self.checkpointed_containers.remove(container)
			logging.info("after remove: " + str(self.checkpointed_containers))
		self.co_containers.remove(container)
                logging.info('Removing container %s', name)
                container_obj = self.cli.containers.get(name)
                return container_obj.remove(force=True)

	def pauseContainer(self, container):
                container_obj = self.cli.containers.get(container)
                return container_obj.pause()

	def unpauseContainer(self, container):
                container_obj = self.cli.containers.get(container)
                return container_obj.unpause()


	def runContainer(self, container_name, image, command , cpus,  memory, detach=True) :
		self.co_containers.append(container_name)
		name = 'cont-' + container_name
		logging.info('running container %s', name)
		quota = self.period*cpus
		target_memory = str(memory) + 'm'
		print container_name
                container_obj = self.cli.containers.run(image, command=command, name=name, working_dir='/root/',
			  cpu_period=self.period, cpu_quota=quota, mem_limit=target_memory, detach=detach, cpuset_cpus=self.allowed_cores)
		return container_obj
			

	def execContainer(self, container, command, detach=True):
		logging.info('Executing on  container %s command %s', container, command)
                container_obj = self.cli.containers.get(container)
		return container_obj.exec_run(command, detach=detach)

def create_container_manager(manager_type, manager_ip, manager_port):
	if manager_type == 'LINUX_CONTAINER_CLOUD_MANAGER':
		return LXCCloudManager(manager_ip, manager_port)
	else:
		raise NotImplementedError("Cloud manager type has not been implemented...")
