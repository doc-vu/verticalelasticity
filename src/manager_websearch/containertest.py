import sys, os, logging
import ConfigParser
from containermanager import *

def main(argv):
	conf = os.path.join(os.path.dirname(__file__), './config/application.conf')
        Config = ConfigParser.ConfigParser();
        Config.read(conf);

        host = Config.get('SERVER', 'HOST')
        port = Config.get('SERVER', 'PORT')
        logfile = Config.get('LOGGING', 'LOG_FILE')
        loglevel = Config.get('LOGGING', 'LEVEL')
        logging.basicConfig(format='%(asctime)s - %(levelname)s  - %(threadName)s - %(module)s - %(funcName)s -line %(lineno)d - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename="TEST.log", level=loglevel)
	logging.info("ASSSS")
	manager =  create_container_manager('LINUX_CONTAINER_CLOUD_MANAGER', '129.59.234.215', '2375')
	#cons =  manager.getContainers()
	#cores = manager.getMaxCoreCount()
	#cons = manager.getContainerCount()
	#for con in cons:
	#	print str(con.attrs)
	#manager.updateCoreCount('isislab5', 'server',6)
	manager.updateResources('isislab5', 'server',[5,12288],5)
	#manager.startContainer('benchmark1')
	#manager.execContainer('benchmark1','nohup python /root/indices/benchmark/bench-collection/local_bench/runtest.py &')
	


if __name__ == "__main__":
	main(sys.argv)
