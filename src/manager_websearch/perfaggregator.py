from flask import Flask, request, jsonify
import logging
import os
import ConfigParser

#from requesthandler import *
from perfhandler import PerfHandler

app = Flask(__name__)

@app.route('/api/runtime_perf', methods=['GET', 'POST'])
def runtime_perf():
	content = request.json
	logging.info("received: " + str(content))
	print content
	handler.addData(content)
	return 'OK'


@app.route('/api/evaluate_results', methods=['GET'])
def evaluate_results():
	return str(handler.evaluatePredictionModel())

@app.route('/api/plot_results', methods=['GET'])
def plot_results():
	handler.plotChart()
	return 'OK'

@app.route('/api/export_results', methods=['GET'])
def export_results():
	handler.exportChartData()
	return 'OK'


@app.route('/api/finish_experiment', methods=['GET'])
def finish_experiment():
	handler.finishAndResetExperiment()
	return 'OK'

if __name__ == '__main__':

	conf = os.path.join(os.path.dirname(__file__), './config/application.conf')
	Config = ConfigParser.ConfigParser();
	Config.read(conf);
	
	host = Config.get('SERVER', 'HOST')
	port = Config.get('SERVER', 'PORT')
	logfile = Config.get('LOGGING', 'LOG_FILE')
	loglevel = Config.get('LOGGING', 'LEVEL')
	logging.basicConfig(format='%(asctime)s - %(levelname)s  - %(threadName)s - %(module)s - %(funcName)s -line %(lineno)d - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=logfile, level=loglevel)
	#logging.getLogger('requests.packages.urllib3.connectionpool').propagate = False
	logging.getLogger('requests.packages.urllib3').propagate = False
	logging.info('#############################################################################');

	#initialize handler
	handler = PerfHandler()
	print 'server started ...'
	app.run(host= host,port=port)

