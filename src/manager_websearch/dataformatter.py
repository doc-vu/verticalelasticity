from abc import ABCMeta, abstractmethod

from model import *


class MetricDataFormatter(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        # do nothing
        self.issavedata = ""


    def formatHostMetricsData(self, perf_data):
	metrics = HostMetrics()
        #metricsnames = ['disk_read_kbps', 'disk_write_kbps', 'net_read_kbps', 'net_write_kbps', 'memory_MB',
        #                'cpu_percent', 'contextswitch']
	#print perf_data
        metrics
        first = True
        disk_read = 0
        disk_write = 0
        net_in = 0
        net_out = 0
        memory = 0
        cpu_percent = 0
        interval = 0
        count = 0
        for record in perf_data:
            if record['disk_octets_read'] is None or record['memory'] is None or record['cpu'] is None:
		# or record[ 'if_octets_rx'] is None:

                continue
            if first:
                disk_read = record['disk_octets_read']
                disk_write = record['disk_octets_write']
                net_in = record['if_octets_rx']
                net_out = record['if_octets_tx']
                memory = record['memory']
                cpu_percent = record['cpu']
                interval = record['interval']
                first = False
            else:

                disk_read += record['disk_octets_read']
                disk_write += record['disk_octets_write']
                net_in += record['if_octets_rx']
                net_out += record['if_octets_tx']
                memory += record['memory']
                cpu_percent += record['cpu']
                interval += record['interval']
            count += 1

        if interval == 0:
            raise Exception("interval should not be 0")

	if net_in is None:
		net_in = 0

	if net_out is None:
		net_out = 0

	
        metrics.disk_read_kbps = round(disk_read / (interval * 1024), 3)
        metrics.disk_write_kbps = round(disk_write / (interval * 1024), 3)
        metrics.net_read_kbps = round(net_in / (interval * 1024), 3)
        metrics.net_write_kbps = round(net_out / (interval * 1024), 3)
        metrics.memory_MB = round(memory / (count * 1024 * 1024), 3)
        metrics.cpu_percent = round(cpu_percent / count, 3)
        return metrics

    def formatHostMicroMetricsData(self, perf_data):
        #metricsnames = ['llc-bw', 'mem-bw', 'cache-ref', 'IPC', 'IPS', 'CS-micro', 'page-faults']
        metrics = HostMetricsMicro()
        first = True
        LLC_bandhwidth = 0
        mem_bandwidth = 0
        cache_references = 0
	cache_misses = 0
        instructions = 0
        cycle = 0
        cs = 0
        interval = 0
	kvm_exit = 0
	sched_iowait = 0
	sched_switch = 0
	sched_wait = 0
        count = 0
        for record in perf_data:
            # if record['cache-references'] is None or record['LLC-bandwidth'] is None or record['cycles'] is None  :
            if record['cycles'] is None:
                continue
            if first:
                LLC_bandwidth = record['LLC-bandwidth']
                mem_bandwidth = record['mem-bandwidth']
                cache_references = record['cache-references']
                cache_misses = record['cache-misses']
                instructions = record['instructions']
                cycle = record['cycles']
                cs = record['cs']
		kvm_exit = record['kvm-exit']
		sched_iowait =  record['sched-iowait']
		sched_switch =  record['sched-switch']
		sched_wait =  record['sched-wait']
		
                interval = record['interval']
                first = False
            else:
                LLC_bandwidth += record['LLC-bandwidth']
                mem_bandwidth += record['mem-bandwidth']
                cache_references += record['cache-references']
                cache_misses += record['cache-misses']
                instructions += record['instructions']
                cycle += record['cycles']
                cs += record['cs']
		kvm_exit += record['kvm-exit']
		sched_iowait +=  record['sched-iowait']
		sched_switch +=  record['sched-switch']
		sched_wait +=  record['sched-wait']
                interval += record['interval']
            count += 1

        if interval == 0:
            print perf_data
	    return None
            #raise Exception("interval should not be 0")

        metrics.llc_bw = round(LLC_bandwidth / interval, 3)
        metrics.mem_bw = round(mem_bandwidth / interval, 3)
        metrics.cache_ref = round(cache_references / interval, 3)
        metrics.cache_misses = round(cache_misses / interval, 3)
        metrics.IPC = round(instructions / cycle * 1.0, 3)
        metrics.IPS = round(instructions / interval * 1.0, 3)
        metrics.CS_micro = round(cs / interval, 3)
        metrics.kvm_exit = round(kvm_exit / interval, 3)
	metrics.sched_iowait = round(sched_iowait / interval, 3)
	metrics.sched_switch = round(sched_switch / interval, 3)
	metrics.sched_wait = round(sched_wait / interval, 3)
        return  metrics

    @abstractmethod
    def formatContainerMetricsData(self, perf_data, cores=None):
        return

    @abstractmethod
    def formatContainerMicroMetricsData(self, perf_data):
        return

class LinuxContainerMetricFormatter(MetricDataFormatter):

    def __init__(self):
        super(MetricDataFormatter, self).__init__()

    def formatContainerMetricsData(self, perf_data, cores=None):
	#metricsnames = ['vm_disk_read_kbps', 'vm_disk_write_kbps', 'vm_net_read_kbps', 'vm_net_write_kbps', 'vm_memory_MB', 'vm_vcpu_percent']
        metrics = ContainerMetrics()
        first = True
        disk_read = 0
        disk_write = 0
        net_in = 0
        net_out = 0
        memory = 0
        vcpu_total = 0
        interval = 0
        count = 0

        for record in perf_data:
            if record['cpu.percent'] is None or record['memory.percent'] is None :
                continue
            if first:
		if record['blkio_read'] is not None:
	                disk_read = record['blkio_read']
                disk_write = record['blkio_write']
                net_in = record['network.usage_rx_bytes']
                net_out = record['network.usage_tx_bytes']
                memory = record['memory.usage_total']
                vcpu_total = record['cpu.percent']
                interval = record['interval']
                first = False
            else:
		if record['blkio_read'] is not None:
			disk_read += record['blkio_read']
                disk_write += record['blkio_write']
                net_in += record['network.usage_rx_bytes']
                net_out += record['network.usage_tx_bytes']
                memory += record['memory.usage_total']
                vcpu_total += record['cpu.percent']
                interval += record['interval']
            count += 1

        if interval == 0:
            raise Exception("interval should not be 0")

	if disk_read is None:
		metrics.vm_disk_read_kbps = None
	else:
	        metrics.vm_disk_read_kbps = round(disk_read / (interval * 1024), 3)
	if disk_write is None:
		metrics.vm_disk_write_kbps = None
	else:
	        metrics.vm_disk_write_kbps = round(disk_write / (interval * 1024), 3)
        # metrics['round(disk_write/(interval*1024),3))
        metrics.vm_net_read_kbps = round(net_in / (interval * 1024), 3)
        metrics.vm_net_write_kbps = round(net_out / (interval * 1024), 3)
        metrics.vm_memory_MB = round(memory / (count * 1024 * 1024), 3)
        metrics.vm_vcpu_percent = round(vcpu_total / count, 3)

        return metrics

    def formatContainerMicroMetricsData(self, perf_data):
        #metricsnames = ['llc-bw', 'mem-bw', 'cache-ref', 'IPC', 'IPS', 'CS-micro', 'page-faults']
        metrics = ContainerMetricsMicro()
	return metrics



class VMMetricFormatter(MetricDataFormatter):

    def __init__(self):
        super(MetricDataFormatter, self).__init__()


    def formatContainerMetricsData(self, perf_data, cores=None):
	if cores == None or cores == 0:
            raise Exception("Invalid value for core count")

        #metricsnames = ['vm_disk_read_kbps', 'vm_disk_write_kbps', 'vm_net_read_kbps', 'vm_net_write_kbps', 'vm_memory_MB', 'vm_vcpu_percent']
        metrics = ContainerMetrics()
        first = True
        disk_read = 0
        disk_write = 0
        net_in = 0
        net_out = 0
        memory = 0
        vcpu_total = 0
        interval = 0
        count = 0
        # printcpu = []
        # printnewcpu = []
        # printcpuavg = []
        # printcpusum = []
        # printcputotal = []
        for record in perf_data:
            if record.get('disk_octets_read') is None or record.get('virt_cpu_total') is None :
                continue
	   
            if first:
                disk_read = record['disk_octets_read']
                disk_write = record['disk_octets_write']
                net_in = record['if_octets_rx']
                net_out = record['if_octets_tx']
                memory = record['memory_actual_balloon']
                #vcpu = record['vcpu_count']
                vcpu_total = record['virt_cpu_total']
                interval = record['interval']
                first = False
            else:
                disk_read += record['disk_octets_read']
                disk_write += record['disk_octets_write']
                net_in += record['if_octets_rx']
                net_out += record['if_octets_tx']
                memory += record['memory_actual_balloon']
                vcpu_total += record['virt_cpu_total']
                interval += record['interval']
            count += 1
        # if record['virt_cpu_total'] != 0:
        #	printcpuavg.append(record['vcpu_avg'])
        #	printcpusum.append(record['vcpu_sum'])
        #	printcputotal.append(record['virt_cpu_total'])
        #	printcpu.append(round( record['vcpu_sum']*100/record['virt_cpu_total'],3))
        #	printnewcpu.append(round( record['vcpu_avg']*cores*100/record['virt_cpu_total'],3))

        if interval == 0:
            raise Exception("interval should not be 0")

        metrics.vm_disk_read_kbps = round(disk_read / (interval * 1024), 3)
        metrics.vm_disk_write_kbps = round(disk_write / (interval * 1024), 3)
        metrics.vm_net_read_kbps = round(net_in / (interval * 1024), 3)
        metrics.vm_net_write_kbps = round(net_out / (interval * 1024), 3)
        metrics.vm_memory_MB = round(memory / (count * 1024 * 1024), 3)
	metrics.vm_vcpu_percent = round(vcpu_total * 100.0 / (count*cores*1e9), 3)
        # print str(printcpusum)
        # print str(printcpuavg)
        # print str(printcputotal)
        # print str(printcpu)
        # print str(printnewcpu)
        return metrics


    def formatContainerMicroMetricsData(self, perf_data):
        #metricsnames = ['llc-bw', 'mem-bw', 'cache-ref', 'IPC', 'IPS', 'CS-micro', 'page-faults']
        metrics = ContainerMetricsMicro()
        first = True
        LLC_bandhwidth = 0
        mem_bandwidth = 0
        cache_references = 0
	cache_misses = 0
        instructions = 0
        cycle = 0
        cs = 0
	kvm_exit = 0
        sched_iowait = 0
        sched_switch = 0
        sched_wait = 0

        interval = 0
        count = 0
        for record in perf_data:
            # if record['cache-references'] is None or record['LLC-bandwidth'] is None or record['cycles'] is None  :
            if record['cycles'] is None:
                continue
            if first:
                LLC_bandwidth = record['LLC-bandwidth']
                mem_bandwidth = record['mem-bandwidth']
                cache_references = record['cache-references']
                cache_misses = record['cache-misses']
                instructions = record['instructions']
                cycle = record['cycles']
                cs = record['cs']
		kvm_exit = record['kvm-exit']
                sched_iowait =  record['sched-iowait']
                sched_switch =  record['sched-switch']
                sched_wait =  record['sched-wait']

                interval = record['interval']
                first = False
            else:
                LLC_bandwidth += record['LLC-bandwidth']
                mem_bandwidth += record['mem-bandwidth']
                cache_references += record['cache-references']
                cache_misses += record['cache-misses']
                instructions += record['instructions']
                cycle += record['cycles']
                cs += record['cs']
		kvm_exit += record['kvm-exit']
                sched_iowait +=  record['sched-iowait']
                sched_switch +=  record['sched-switch']
                sched_wait +=  record['sched-wait']

                interval += record['interval']
            count += 1

        if interval == 0:
            print perf_data
            raise Exception("interval should not be 0")

	if mem_bandwidth is None:
		mem_bandwidth = 0.0

	if LLC_bandwidth is None:
		LLC_bandwidth = 0.0

	if cache_references is None:
		cache_references = 0.0

	if cache_misses is None:
		cache_misses = 0.0

	if instructions is None:
		instructions = 0.0

        metrics.llc_bw = round(LLC_bandwidth / interval, 3)
        metrics.mem_bw = round(mem_bandwidth / interval, 3)
        metrics.cache_ref = round(cache_references / interval, 3)
        metrics.cache_misses = round(cache_misses / interval, 3)
        metrics.IPC = round(instructions / cycle * 1.0, 3)
        metrics.IPS = round(instructions / interval * 1.0, 3)
        metrics.CS_micro = round(cs / interval, 3)
	metrics.kvm_exit = round(kvm_exit / interval, 3)
        metrics.sched_iowait = round(sched_iowait / interval, 3)
        metrics.sched_switch = round(sched_switch / interval, 3)
        metrics.sched_wait = round(sched_wait / interval, 3)

        return  metrics



def create_formatter(container_type):
    if container_type == "VM":
        return VMMetricFormatter()
    elif container_type == "DOCKER":
        return LinuxContainerMetricFormatter()
    else:
        raise NotImplementedError('benchmark: ' + container_type + ' not implemented')
