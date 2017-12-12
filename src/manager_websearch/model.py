#DTO
class BenchmarkResult:

	def __init__(self, **kwargs):
        	self.__dict__ = kwargs
        	
	def __repr__(self):
		return str(self.__dict__)

class HostMetrics:

	def __init__(self, **kwargs):
        	self.__dict__ = kwargs
        	
	def __repr__(self):
		return str(self.__dict__)

class HostMetricsMicro:

	def __init__(self, **kwargs):
        	self.__dict__ = kwargs
        	
	def __repr__(self):
		return str(self.__dict__)

class ContainerMetrics:

	def __init__(self, **kwargs):
        	self.__dict__ = kwargs
        	
	def __repr__(self):
		return str(self.__dict__)

class ContainerMetricsMicro:

	def __init__(self, **kwargs):
        	self.__dict__ = kwargs
        	
	def __repr__(self):
		return str(self.__dict__)

