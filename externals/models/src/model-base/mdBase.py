import os
import paths
import sys

sys.path.append(os.path.join(paths.prj_path, '/utils/src/'));

from fnAddOnesToX import *;

def fnCostFunc(theta,x,y):

	print("ERROR: Method 'fnCostFunc_' must be implemented in derived model")

def fnGradient(theta,x,y):

	print("ERROR: Method 'fnGradient_' must be implemented in derived model")
	
def fnPredict(theta,x):

	XOnes = fnAddOnesToX(x);
	y = np.dot(XOnes, np.transpose(theta));

	return y;

def fnSolveModel(theta0,x,y):

	print("ERROR: Method 'fnSolveModel_' must be implemented in derived model")

class ModelBase:
	def __init__(self,name):
		self.name = name

	def displayName_(self):
		print "Model Base"
		
	def fnCostFunc_(self,theta,x,y):
		"""
		Returns cost value 
		"""
		return fnCostFunc(theta,x,y)  	

	def fnGradient_(self,theta,x,y):
		"""
		Returns gradient
		"""
		return fnGradient(theta,x,y)  	

	def fnPredict_(self,theta,x):
		"""
		Returns prediction 
		"""
		return fnPredict(theta,x)
		
	def fnSolveModel_(self,theta0,x,y):
		"""
		Solves model
		"""
		return fnSolveModel(theta0,x,y)  	