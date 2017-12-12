import threading
import logging
import Queue
import ConfigParser
import os
import time
import datetime
import sys
import traceback
import collections
import csv

from dbutil import DBUtil
from benchmark import *
from containermanager import *
from perfaction import PerfAction

def create_interference(con_manager, container_num, interference):
        container_name = str(container_num)
        image_name = interference['BENCH_NAME']
	cores = 2
	mem = 2500
	command = interference['MEM_COMMAND']
	if (container_num == 0 or container_num == 1):
		image_name = interference['BENCH_NAME_SEC']
	        command = interference['MEM_COMMAND_SEC']
		cores = 2
		mem = 4096

        logging.info('starting  interference container: {0} '.format(container_name))
        print 'starting container .. ', container_name
        name = con_manager.runContainer(container_name, image_name, command, cores, mem)


def run_interference(load, con_manager, interference_data, handler_ins):
    active_containers = []
    curr_interval = 0.0
    for vals in load:
	start = time.time()
	required_containers = vals['count']
	end_interval = vals['end_interval']
	curr_container_count = len(active_containers)
	extra_containers = required_containers - curr_container_count
	print 'need  more containers: ', extra_containers
	logging.info('active containers: ' + str(active_containers))
	if extra_containers > 0:
	    for i in range(curr_container_count, required_containers):
		print 'creating container:  ', i
		create_interference(con_manager, i, interference_data)
		active_containers.append(i)
	elif extra_containers < 0:
	    for i in range(curr_container_count, required_containers, -1):
		print 'removing container: ', str(i-1)
		con_manager.removeContainer(str(i-1))
		active_containers.remove(i-1)

	logging.info('active containers later: ' + str(active_containers))

	handler_ins.loadcount = len(active_containers)
	end = time.time()
	time.sleep((end_interval-curr_interval) - (end-start))
	curr_interval = end_interval
	
		
	    	

# this class is responsible for handling all benchmarking operations
class PerfHandler(object):

    def loadInterferenceData(self, interference_wkload_file, interference):
        interference_load = []
	with open(interference_wkload_file, 'rb') as f:
	    # skip header line
	    f.readline()
	    for line in f:
		values = line.split(',')
		row = {}
		row['end_interval'] = int(values[0])
		row['count'] = int(values[1])
		
		#row['vmid'] = values[0]
		#row['start_interval'] = int(values[3])
		#row['end_interval'] = int(values[4])
		#row['cores'] = int(values[5])
		#row['mem'] = int(values[6])
		#row['cpu_util'] = int(float(values[7]))
		#row['mem_util_percent'] = float(values[9])
		interference_load.append(row)
	threading.Thread(target=run_interference, args=(interference_load, self.container_manager, interference, self)).start()
		

    def __init__(self, interval=1):

	logging.info('initializing performance handler ...')
        self.interval = interval
	self.queue = Queue.Queue()
	self.loadcount = 0

	logging.info('Reading application config ...')
	conf = os.path.join(os.path.dirname(__file__), './config/application.conf')
	config = ConfigParser.ConfigParser();
	config.read(conf);

	host = config.get('DB', 'HOST')
	port = config.get('DB', 'PORT')
	user = config.get('DB', 'USER')
	password = config.get('DB', 'PASSWORD')
	readdb = config.get('DB', 'READDB')
	writedb = config.get('DB', 'WRITEDB')
	self.loopcount = int(config.get('PROCESSING', 'SAMPLE_LOOP_COUNT_IN_SEC'))
	self.waittime = int(config.get('PROCESSING', 'SAMPLE_WAIT_TIME_IN_MS'))
	self.aggregationtime = int(config.get('PROCESSING', 'SAMPLE_AGGREGATION_TIME_IN_SEC'))
	self.rampuptime = int(config.get('PROCESSING', 'RAMP_UP_TIME'))
	containertype = config.get('PROCESSING', 'CONTAINER_TYPE')
	action_model = config.get('PROCESSING', 'ACTION_MODEL')
	
	logging.info('Reading benchmark config ...')
	benchconf = os.path.join(os.path.dirname(__file__), './config/benchmark.conf')
	self.activebench = create_benchmark(benchconf, containertype)

	conf = os.path.join(os.path.dirname(__file__), './config/interference.info')
	interference_config = ConfigParser.ConfigParser();
	interference_config.read(conf);

	containermanager_port = config.get('CONTAINER_MANAGER', 'PORT')
	containermanager_ip = config.get('CONTAINER_MANAGER', 'HOST_NAME')
	managertype = config.get('CONTAINER_MANAGER', 'TYPE')
	self.container_manager = create_container_manager(managertype, containermanager_ip, containermanager_port)

	dointerference = config.get('PROCESSING', 'PERFORM_INTERFERENCE')  == 'true'
	if dointerference:
		interference_wkload_file = os.path.join(os.path.dirname(__file__), './config/int_load.csv')
		interference = {}
		for key, value in interference_config.items('INTERFERENCE'):
			interference[key.upper()] = value
		self.loadInterferenceData(interference_wkload_file, interference)


	self.reschange = []
	for key, value in interference_config.items('RESOURCE_ALLOC'):
		interval = int(key)
		res = value.split(',')
		core = int(res[0])
		mem = int(res[1])
		self.reschange.append((int(key), (core,mem)))


	details = self.activebench.getTargetDetails()
	self.targethost = details[0]	
	self.targetcontainer = details[2]	
	loadfile = details[5]
	loadfile = os.path.join(os.path.dirname(__file__), loadfile)
	with open(loadfile, 'rb') as f:
	    reader = csv.reader(f)
	    self.load = list(reader)	
	self.perfaction = PerfAction(self.container_manager, details[1], details[2], details[3], details[4], action_model)
	print 'cores: ' + str(self.perfaction.getCurrentCoreCount())

	self.perftime_map = {}
	for server in self.activebench.getPerfServers():
	    self.perftime_map[server] = 0

	self.dbutil = DBUtil(host, int(port), user, password, readdb, writedb)

        thread = threading.Thread(target=self.run, args=())
	logging.info('Starting handler thread for benchmark: ' + ' ' + ' ..')
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    # adds the benchmark input to the processing queue
    def addData(self, perfdata):
	self.queue.put(perfdata)

    def evaluatePredictionModel(self):
	return self.perfaction.evalModel()

    def plotChart(self):
	self.perfaction.plotResults()

    def exportChartData(self):
	self.perfaction.exportResults()

    def finishAndResetExperiment(self):
	self.perfaction.evalModel()
	self.perfaction.exportResults()
	details = self.activebench.getTargetDetails()
        self.perfaction = PerfAction(self.container_manager, details[1], details[2], details[3], details[4])

    # waits till the results have arrived for the analysis duration
    def ensureDataSetWithinTimeBounds(self, benchmarktimestamp):
	loop = self.loopcount
	while (loop > 0):

	    items = self.dbutil.getLastRecordForHosts()
	    istimeinbounds = True
	    for key, gen in items:
		hostname = key[1]['host']
		# this may return any old saved server time diff
		if self.perftime_map.get(hostname) is not None:
		    result =  gen.next()
		    servertimestamp = result['time']/1000000
		    self.perftime_map[hostname] = result['time']
		    if benchmarktimestamp - servertimestamp > self.waittime:
			logging.warning('time not within bounds, differ by {0} ms '.format((benchmarktimestamp-servertimestamp)))
			istimeinbounds = False

	    if istimeinbounds:
		return 
	    time.sleep(1)
	    loop -= 1

    # aggregates data for latency and througput
    def formatDataForModelProcessing(self, formatted_data):
	#print formatted_data
	throughput = sum(formatted_data['thru'])
	# changing throughput definition to input data instead of output
	#throughput = sum(formatted_data['requests'])
	latency = -1.0
	latency90 = -1.0
	weighted_latency = -1.0
	resptime = 0.0
	resptime90 = 0.0
	wt_resptime = 0.0
	count = 0
	for i in range(0, len(formatted_data['resp'])):
	    resp = formatted_data['resp'][i]	
	    resp90 = formatted_data['resp90'][i]	
	    if resp != '-' and resp90 != '-':
	        resptime += resp
	        # bad approximation
	        resptime90 += resp90
	        wt_resptime += resp*formatted_data['thru'][i]
		# changing for now 
	        #wt_resptime += resp*formatted_data['requests'][i]
	        count += 1
	if count != 0:
	    latency = round(resptime / float(count), 3)
	    latency90 = round(resptime90 / float(count), 3)
	    if throughput == 0:
	        weighted_latency = 0.0
	    else:
	        weighted_latency = round(wt_resptime / throughput, 3)
	
	return (throughput, latency, weighted_latency, latency90)


    def getUserCount(self, interval):
	users = 0.0
	for vals in self.load:
		if interval < int(vals[0]):
			break
		users =  int(vals[1])
	return users

    def checkAndPerformConfigChange(self, interval, last_reschange):
	for reschange_interval, resource in self.reschange:
	    if interval >= reschange_interval and last_reschange < reschange_interval:
		print 'updating core count to ' + str(resource[0])
		self.container_manager.updateResources(self.targethost, self.targetcontainer, resource)
		last_reschange = reschange_interval
		return reschange_interval
	return last_reschange




    # run thread processes the queue
    def run(self):
	corecount = 0
	last_interference = 0
	last_reschange = 0
        while True:
	    try:
		data = self.queue.get()
		benchmarktimestamp = data['timestamp']
		benchname = data.get('name')
		#interference_tasks = data.get('interference')
		#interference_tasks_str = ';' + ';'.join(interference_tasks)
		run_grp = data.get('run_grp')
		#print run_grp
		result = None

		# timestamp start 
		starttime = datetime.datetime.now()
		self.ensureDataSetWithinTimeBounds(benchmarktimestamp)
		#timestamp processing after waiting
		intermediatetime = datetime.datetime.now()
		logging.info('waited for {0}s for the times to sync before processing '.format((intermediatetime-starttime).total_seconds()))
		 
		# query the database to get all records
		serverperf_data = self.activebench.getServerData(self.dbutil.getReadClient(), self.perftime_map, benchmarktimestamp - (self.aggregationtime*1000))
		# aggregate and format the records for benchmark specific content
		#print serverperf_data
		formatted_data = self.activebench.formatData(data, serverperf_data)
		logging.debug(formatted_data)
		runid = formatted_data['run_id']
		lastrunid = self.activebench.getRunId()
		
		if (lastrunid is not None) and (lastrunid != runid):
		    logging.info('This is a new experiment run, finish {0}'.format(self.activebench.getRunId()))
		    #self.finishAndResetExperiment()

		interval = formatted_data['interval']
		users = float(self.getUserCount(interval))
		if interval <= self.rampuptime:
		    logging.info('ignore data set as still in warmup')
		    continue

		last_reschange = self.checkAndPerformConfigChange(interval, last_reschange)
	        corecount = self.container_manager.getCurrentCoreCount(self.targethost, self.targetcontainer)

		perfvalues = self.formatDataForModelProcessing(formatted_data)
		throughput = perfvalues[0]
		latency = perfvalues[1]
		weighted_latency = perfvalues[2]
		latency90 = perfvalues[3]
		utilization = self.activebench.getUtilizationStats(formatted_data, (self.targethost, self.perfaction.getCurrentCoreCount()))
	        logging.debug('utilization {0}'.format(utilization))
		if utilization is None:
		    logging.warning('Utilization vector is empty, ignore ..')
		    continue
		utilization_keys = utilization[0]
		#print str(utilization_keys)
		utilization_stats = utilization[1]
		dataprocessed = False

		if weighted_latency != -1.0:

		    # invoke model processing
		    logging.info('Invoking prediction model at interval: ' + str(interval))
		    #result = self.perfaction.invokeModel(users, latency90, weighted_latency, utilization_stats)
		    result = self.perfaction.invokeModel(throughput, latency, weighted_latency, utilization_stats)
		    corecount = result[0]
		    dataprocessed = True
		    logging.info('current core count after model processing: {0}'.format( corecount))

		else:
		    logging.info('no latency to process, ignore')
	

		#timestamp processing at the end
		endtime = datetime.datetime.now()
		totaldiff = (endtime - starttime).total_seconds()*1000
		processingdiff = (endtime - intermediatetime).total_seconds()*1000
		logging.info(' processing time with wait: {0}ms and without wait: {1}ms. '.format(totaldiff, processingdiff))

		perfstoragedata = {}
		#perfstoragedata['interference'] = interference_tasks_str
		perfstoragedata['total_dur_ms'] = totaldiff
		perfstoragedata['process_dur_ms'] = processingdiff
		perfstoragedata['benchtime_ms'] = benchmarktimestamp
		perfstoragedata['cores'] = corecount
		perfstoragedata['load_count'] = self.loadcount
		perfstoragedata['cp_count'] = self.container_manager.getCheckpointedCount()

		perfstoragedata['users'] = users
		#perfstoragedata['users'] = formatted_data['requests']
		# changed the throughput data in processing, so swap for actual storage
		#perfstoragedata['users'] = throughput
		perfstoragedata['interference_change'] = last_interference
		perfstoragedata['interval'] = formatted_data['interval']
		# changed the throughput data in processing, so swap for actual storage
	        perfstoragedata['throughput'] = formatted_data['thru'][0]
		if formatted_data.get('kernel') is None:
			perfstoragedata['kernel'] = 'base'
			perfstoragedata['kernel'] = '0.001'
		else:
			perfstoragedata['kernel'] = formatted_data['kernel']
			perfstoragedata['intensity'] = formatted_data['intensity']
		perfstoragedata['latency'] = latency
		perfstoragedata['latency90'] = latency90
		perfstoragedata['weighted_latency'] = weighted_latency
		for i in range(0, len(utilization_keys)):
		    perfstoragedata[utilization_keys[i]] = utilization_stats[i]

	
		if benchname is None:
			benchname = self.activebench.getName()

		self.dbutil.saveBenchmarkRecord(runid, run_grp, dataprocessed, perfstoragedata, benchname=benchname, 
						 resp=formatted_data['resp'], thru=formatted_data['thru']) 
						#ops=formatted_data['operations'], resp=formatted_data['resp'], thru=formatted_data['thru']) 

		if result is not None and len(result) > 1:
		    perfstoragedata['throughput'] = result[3]
		    perfstoragedata['latency'] = result[1]
		    #perfstoragedata['latency90'] = result[4]
		    perfstoragedata['weighted_latency'] = result[2]
		    #for i in range(0, 4):
		    #for i in range(0, len(utilization_keys)):
		    #	perfstoragedata[utilization_keys[i]] = result[4][i]
	
		    # saving only 1 metrics used in prediction as of now
		    perfstoragedata[utilization_keys[4]] = result[4][0]
		    #perfstoragedata[utilization_keys[2]] = result[4][1]
		    self.dbutil.saveBenchmarkRecord(runid, run_grp, dataprocessed, perfstoragedata, benchname=benchname, isagg=True) 
			
			
		self.activebench.setRunId(runid)
	    except Exception, e:
		logging.error(traceback.print_exc())
		logging.error(e)
		print e
		#raise Exception, "The code is buggy: %s" % e, sys.exc_info()[2]

