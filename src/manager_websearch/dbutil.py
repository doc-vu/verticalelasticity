from influxdb import InfluxDBClient
import json
import logging

SELECT_LAST_RECORD_QUERY = "select last(disk_octets_write) from host_metrics where time > now() - 1h group by host"
#SELECT_RECORD_QUERY = "select * from system_metrics where "

# class for performing database operations currently mapped to influxdb
class DBUtil(object):

	_instance = None
	_initialized = False
	
	def __new__(cls, *args, **kwargs):
		if not cls._instance:
        		cls._instance = super(DBUtil, cls).__new__(cls, *args, **kwargs)
		return cls._instance
        	
	def __init__(self, host, port, user, password, readdb, writedb):
		if(self._initialized):
			return
		logging.info("Initializing db client ...")
		self.dbreadclient = InfluxDBClient(host, port, user, password, readdb)
		self.dbwriteclient = InfluxDBClient(host, port, user, password, writedb)

		self._initialized = True

	# return the handle to read db client
        def getReadClient(self):
        	return self.dbreadclient

	# query to fetch recent database record time
	def getLastRecordForHosts(self):
		rs = self.dbreadclient.query(SELECT_LAST_RECORD_QUERY, epoch=True)
                items = rs.items()
		return items
	
	#records the data point for the current run
	def saveBenchmarkRecord(self, benchid, bench_grp, dataprocessed, perfstoragedata, 
					benchname='DEFAULT', isagg=False, ops=None, resp=None, thru=None):
		datapoint = {}
		#print str(perfstoragedata)
		datapoint['fields'] = perfstoragedata
		if isagg:
			datapoint['measurement'] = benchname + '_AGG'
		else:
			datapoint['measurement'] = benchname
			#for i in range (0,len(thru)):
			#	if resp[i] == '-':
			#		resp[i] = -1.0
			#	if thru[i] == '-':
			#		thru[i] = -1.0
			#	pre = ""
			#	if ops is not None:
			#		pre = ops[i] + '_'
			#	datapoint['fields'][pre + 'resp'] = resp[i]
			#	datapoint['fields'][pre +'thru'] = thru[i]

		datapoint['tags'] = { 'benchmark_id' : benchid, 'dataprocessed' : dataprocessed, 'benchmark_grp'  : bench_grp}


		logging.debug("Writing points: {0}".format([datapoint]))
		self.dbwriteclient.write_points([datapoint])
		
