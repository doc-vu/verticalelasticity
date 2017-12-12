from abc import ABCMeta, abstractmethod
import ConfigParser
from sets import Set
import time
import pprint
import logging

from dataformatter import *


class Benchmark(object):
    __metaclass__ = ABCMeta

    def __init__(self, issavedata, isconsidermicro, isconsidercontainer, isnormalize, metric_data_formatter, perfservers):
        self.issavedata = issavedata
        self.isconsidermicro = isconsidermicro
        self.isconsidercontainer = isconsidercontainer
	self.isnormalize = isnormalize
        self.metric_data_formatter = metric_data_formatter
        self.perfservers = perfservers
        self.lastrunid = None

    def getPerfServers(self):
        return self.perfservers

    def retrieveInterference(self, client, host, timeval):
        rs = client.query(
                'select * from bench where host=\'' + host + '\' and time>' + timeval + ' group by instance',
                epoch=True)
        return self.formatVMDBData(rs)
	

    def retrieveMeasurement(self, client, measurement, host, timeval, isvm=False):
        if isvm:
            rs = client.query(
                'select * from ' + measurement + ' where host=\'' + host + '\' and time>' + timeval + ' group by instance',
                epoch=True)
            return self.formatVMDBData(rs)

        rs = client.query('select * from ' + measurement + ' where host=\'' + host + '\' and time>' + timeval,
                          epoch=True)
        return self.formatHostDBData(rs)

    def formatHostDBData(self, rs):
        points = rs.get_points()
        values = []
        for point in points:
            values.append(point)
        return values

    def setRunId(self, runid):
        self.lastrunid = runid

    def getRunId(self):
        return self.lastrunid

    def formatVMDBData(self, rs):
        items = rs.items()
        vm_data = {}
        for key, gen in items:
            instance = key[1]['instance']
            vm_data[instance] = []

            for values in gen:
                vm_data[instance].append(values)

        return vm_data

    @abstractmethod
    def formatData(self, benchmark_data, serverperf_data):
        # aggegregate and format data
        return

    def getName(self):
        return self.name

    @abstractmethod
    def getUtilizationStats(self, data):
        # should fetch data for servers from db
        return

    @abstractmethod
    def getTargetDetails(self):
        return

    def getServerData(self, client, hosttime_map, aggregationtimestamp):
        return_map = {}
        timeval = str(aggregationtimestamp * 1000000)
	#print timeval
        for host in hosttime_map.keys():
            return_map[host] = {}
	    
            measurement = 'host_metrics'
            return_map[host][measurement] = self.retrieveMeasurement(client, measurement, host, timeval)
  	    #print return_map
            if self.isconsidermicro:
                measurement = 'host_metrics_micro'
                return_map[host][measurement] = self.retrieveMeasurement(client, measurement, host, timeval)

            if self.isconsidercontainer:
                measurement = 'vm_metrics'
                # measurement = 'container_metrics'
                result = self.retrieveMeasurement(client, measurement, host, timeval, isvm=True)
                return_map[host][measurement] = result

                if self.isconsidermicro:
                    # measurement = 'container_metrics_micro'
                    measurement = 'vm_metrics_micro'
                    return_map[host][measurement] = self.retrieveMeasurement(client, measurement, host, timeval,
                                                                             isvm=True)

                # pp = pprint.PrettyPrinter(indent=0)
                # pp.pprint(return_map)
	    return_map[host]['intensity'] = self.retrieveInterference(client, host, timeval)
        return return_map


class CloudSuiteWebSearch(Benchmark):
    def __init__(self, issavedata, isconsidermicro, isconsidercontainer, isnormalize, webservconfig, metric_data_formatter):
        self.targetserver = webservconfig['WEBSERVER'.lower()]
        self.name = webservconfig['NAME'.lower()]
        self.targetserver_host_cores = int(webservconfig['WEBSERVER_HOST_CORES'.lower()])
        self.targetserver_ip = webservconfig['WEBSERVER_IP'.lower()]
        self.targetserver_max_cores = int(webservconfig['WEBSERVER_CONTAINER_MAX_CORES'.lower()])
        self.targetserver_min_cores = int(webservconfig['WEBSERVER_CONTAINER_MIN_CORES'.lower()])
        self.targetserver_container = webservconfig['WEBSERVER_CONTAINER'.lower()]
	print webservconfig
        self.loadfile = webservconfig.get('LOAD_FILE'.lower())
        perfservers = Set([self.targetserver])
        super(CloudSuiteWebSearch, self).__init__(issavedata, isconsidermicro, isconsidercontainer, isnormalize, metric_data_formatter, perfservers)

    def getTargetDetails(self):
        return [self.targetserver, self.targetserver_ip, self.targetserver_container, self.targetserver_min_cores, self.targetserver_max_cores, self.loadfile]

    def getUtilizationStats(self, data, target):
        webdata = data['metrics'].get(self.targetserver)
        if webdata is None:
            return None
        if target[0] != self.targetserver:
            raise Exception('Target does not match')
	corecount = target[1]
        # resultkeys = ['db_cpupercent', 'web_cpupercent', 'db_netin', 'web_netin', 'db_netout', 'web_netout']

        # 'disk_read_kbps','disk_write_kbps', 'net_read_kbps','net_write_kbps','memory_MB','cpu_percent','contextswitch'
        # 'vm_disk_read_kbps','vm_disk_write_kbps', 'vm_net_read_kbps','vm_net_write_kbps','vm_memory_percent','vm_vcpu_percent'
        # metricsnames = ['vm_net_read_kbps','vm_net_write_kbps','vm_memory_percent','vm_vcpu_percent']

	resultkeys = []
        result = []
        webhost = webdata['MACRO']
        webhostmicro = webdata.get('MICRO')

	if self.isconsidercontainer:
		if  webdata.get('VM') is not None and webdata['VM'].get(self.targetserver_container) is not None:
			resultkeys += ['web_cpupercent', 'web_disk','web_net', 'web_memory']
			#print webdata
			webvm = webdata['VM'][self.targetserver_container]['MACRO']
			# presently not using micro for VM
			#webvmmicro = webdata['VM'][self.targetserver_container].get('MICRO')
			result.append(webvm.vm_vcpu_percent)
			# append scaled webserver disk utilization
			result.append(webvm.vm_disk_read_kbps + webvm.vm_disk_write_kbps)
			# append webserver network in utilization
			result.append(webvm.vm_net_read_kbps + webvm.vm_net_read_kbps)
			# append scaled webserver memory utilization
			result.append(webvm.vm_memory_MB)
			if self.isnormalize:
				resultkeys += ['web_cpupercent_norm', 'web_disk_norm','web_net_norm', 'web_memory_norm']
				result.append(webvm.vm_vcpu_percent/corecount)
				result.append((webvm.vm_disk_read_kbps + webvm.vm_disk_write_kbps)*100.0/1200)
				result.append((webvm.vm_disk_read_kbps + webvm.vm_disk_write_kbps)*100.0/100)
				result.append((webvm.vm_memory_MB)*100.0/12000)

        resultkeys += [ 'host_disk_io', 'host_net_io', 'host_memory',
                      'host_cpupercent']
        result.append(webhost.disk_read_kbps + webhost.disk_write_kbps)
        result.append(webhost.net_read_kbps + webhost.net_write_kbps)
        result.append(webhost.memory_MB)
        result.append(webhost.cpu_percent)

	if self.isnormalize:
		resultkeys += [ 'host_disk_io_norm', 'host_net_io_norm', 'host_memory_norm',
			      'host_cpupercent_norm']
		result.append((webhost.disk_read_kbps + webhost.disk_write_kbps)*100.0/32000)
		result.append((webhost.net_read_kbps + webhost.net_write_kbps)*100.0/100)
		result.append(webhost.memory_MB*100.0/3.2e8)
		result.append(webhost.cpu_percent)

        if self.isconsidermicro and webhostmicro is not None:
            resultkeys += ['host_llc_bw', 'host_mem_bw', 'host_cache_ref', 'host_IPC', 'host_IPS', 'host_cs',
                           'host_cache_misses',
			   'host_kvm_exit','host_sched_iowait','host_sched_switch','host_sched_wait']
            result.append(webhostmicro.llc_bw)
            result.append(webhostmicro.mem_bw)
            result.append(webhostmicro.cache_ref)
            result.append(webhostmicro.IPC)
            result.append(webhostmicro.IPS)
            result.append(webhostmicro.CS_micro)
            result.append(webhostmicro.cache_misses)
            result.append(webhostmicro.kvm_exit)
            result.append(webhostmicro.sched_iowait)
            result.append(webhostmicro.sched_switch)
            result.append(webhostmicro.sched_wait)

	    if self.isnormalize:
		    resultkeys += ['host_mem_bw_norm', 'host_cache_ref_norm', 'host_IPC_norm', 'host_IPS_norm', 'host_cs_norm',
				   'host_cache_misses_norm',
				   'host_sched_iowait_norm','host_sched_switch_norm','host_sched_waiti_norm']
		    result.append(webhostmicro.mem_bw*100.0/3.3e9)
		    result.append(webhostmicro.cache_ref*100.0/6.0e9)
		    result.append(webhostmicro.IPC*100.0/2.0)
		    result.append(webhostmicro.IPS*100.0/3.5e10)
		    result.append(webhostmicro.CS_micro*100.0/15000)
		    result.append(webhostmicro.cache_misses*100.0/2.5e7)
		    result.append(webhostmicro.sched_iowait*100.0/9.0e10)
		    result.append(webhostmicro.sched_switch*100.0/18000)
		    result.append(webhostmicro.sched_wait*100.0/2.5e11)

	    
	    if self.isconsidercontainer and webdata.get('VM') is not None and webdata['VM'].get(self.targetserver_container) is not None and webdata['VM'][self.targetserver_container].get('MICRO') is not None:
            	resultkeys += ['web_llc_bw', 'web_mem_bw', 'web_cache_ref', 'web_IPC', 'web_IPS', 'web_cs',
                           'web_cache_misses',
	    		   'web_kvm_exit','web_sched_iowait','web_sched_switch','web_sched_wait']
	    
		result.append(webvmmicro.llc_bw)
		result.append(webvmmicro.mem_bw)
		result.append(webvmmicro.cache_ref)
		result.append(webvmmicro.IPC)
		result.append(webvmmicro.IPS)
		result.append(webvmmicro.CS_micro)
		result.append(webvmmicro.cache_misses)
		result.append(webvmmicro.kvm_exit)
		result.append(webvmmicro.sched_iowait)
		result.append(webvmmicro.sched_switch)
		result.append(webvmmicro.sched_wait)


	#print resultkeys
	#print result
        return (resultkeys, result)

    def formatData(self, benchmark_data, serverperf_data):
        #print benchmark_data
        #print serverperf_data
        data = {}
        data['timestamp'] = benchmark_data['timestamp']
        data['run_id'] = benchmark_data['run_id']
        data['thru'] = benchmark_data['CThru']
        #data['requests'] = benchmark_data['client_count']
        data['resp'] = benchmark_data['CResp']
        data['resp90'] = benchmark_data['C90%Resp']
        data['interval'] = benchmark_data['current_interval']
        data['metrics'] = {}
        for host, perf_data in serverperf_data.iteritems():
            host_data = perf_data.get('host_metrics')
            if host_data is None or not host_data:
		logging.warning('missing host data ')
                continue
            interference_data = perf_data.get('intensity')
	    if interference_data is None or len(interference_data) > 1:
		logging.warning('problem with intereference data. ')
	    for kernel, interf_data in interference_data.iteritems():
		data['kernel'] = kernel
		data['intensity'] = interf_data[0].get('intensity')
		
	    
            data['metrics'][host] = {}
            result = self.metric_data_formatter.formatHostMetricsData(host_data)
            data['metrics'][host]['MACRO'] = result

            if self.isconsidermicro:
                micro_results = self.metric_data_formatter.formatHostMicroMetricsData(perf_data.get('host_metrics_micro'))
		if micro_results is not None:
			data['metrics'][host]['MICRO'] = micro_results

            # vms_data = perf_data.get('container_metrics')
            vms_data = perf_data.get('vm_metrics')
            if vms_data is not None:

                data['metrics'][host]['VM'] = {}
                for vm, vm_data in vms_data.iteritems():
                    #print vm
                    #print vm_data
                    if self.targetserver == host:
                        result =  self.metric_data_formatter.formatContainerMetricsData(vm_data, self.targetserver_max_cores)
                    else:
                        raise Exception('Target server does not match')

                    data['metrics'][host]['VM'][vm] = {}
                    data['metrics'][host]['VM'][vm]['MACRO'] = result

            if self.isconsidermicro:
	
	        # vms_data = perf_data.get('container_metrics')
                vms_data = perf_data.get('vm_metrics_micro')
	        if vms_data is not None:

                    for vm, vm_data in vms_data.iteritems():
                        if self.targetserver == host:
                            result =  self.metric_data_formatter.formatContainerMicroMetricsData(vm_data)
                        else:
                            raise Exception('Target server does not match')

                        data['metrics'][host]['VM'][vm]['MICRO'] = result

	    

        return data




def create_benchmark(configfile, container_type):
    config = ConfigParser.ConfigParser();
    config.read(configfile);
    name = config.get('BENCHMARK', 'ACTIVE')
    issavedata = config.get('BENCHMARK', 'SAVE_DATA')
    isconsidermicro = True if config.get('BENCHMARK', 'CONSIDER_MICRO') == 'true' else False
    isconsidercontainer = True if config.get('BENCHMARK', 'CONSIDER_CONTAINER') == 'true' else False
    isnormalize = True if config.get('BENCHMARK', 'NORMALIZE_DATA') == 'true' else False
    metric_data_formatter = create_formatter(container_type)

    #if name == "CLOUDSUITE_WEBSERVING":
    #    webservconfig = dict(config.items('CLOUDSUITE_WEBSERVING'))
    #    return CloudSuiteWebServing(issavedata, isconsidermicro, isconsidercontainer, webservconfig, metric_data_formatter)
    if name == "CLOUDSUITE_WEBSEARCH":
        webservconfig = dict(config.items('CLOUDSUITE_WEBSEARCH'))
        return CloudSuiteWebSearch(issavedata, isconsidermicro, isconsidercontainer, isnormalize, webservconfig, metric_data_formatter)
    else:
        raise NotImplementedError('benchmark: ' + name + ' not implemented')
