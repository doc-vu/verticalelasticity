from celery import Celery
import os
import ConfigParser


conf = os.path.join(os.path.dirname(__file__), './config/application.conf')
Config = ConfigParser.ConfigParser();
Config.read(conf);

 # Celery configuration
broker_url = Config.get('MESSAGE_QUEUE', 'URL')

celery = Celery(broker=broker_url)

@celery.task
def handle_benchmark_data(content):
	print str(content)
