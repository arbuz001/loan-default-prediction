import numpy as np
import os
import paths
import scipy.optimize as op
import sys

sys.path.append(os.path.join(paths.prj_path, '/utils/src'));
sys.path.append(os.path.join(paths.prj_path, '/models/src/model-base/'));

from mdBase import ModelBase;

def fnCostFunc(theta,x,y):

	m = len(y);
	h = np.dot(x,theta)-y;
	J = 0.5/m*np.dot(np.transpose(h),h);
	
	return J;
	
def fnGradient(theta,x,y):

	# m = len(y);
	# z = (fnSigmoid(np.dot(x, np.transpose(theta))) - y);
	# J = 1.0/m*np.dot(x,z);
	
	return J;

def fnSolveModelOpMinimize(theta0,x,y):

	Result = op.minimize(fun = fnCostFunc, x0 = theta0, args = (x, y), method = 'TNC');
	theta = Result.x;
	
	return theta;

def fnSolveModelOpMinimize(theta0,x,y):

	Result = op.minimize(fun = fnCostFunc, x0 = theta0, args = (x, y), method = 'TNC');
	theta = Result.x;
	
	return theta;

def fnSolveModelGradientDescent(x, y, theta, alpha, nIters):

	m = len(y)
	J_history = np.array([])
	
	for iter in range(1, nIters):
		theta = theta - alpha/m*np.dot(np.transpose(x) , np.dot(x,theta) - y)
		
		# Save the cost J in every iteration    
		J_history = np.append(J_history,fnCostFunc_(theta,x,y))

	return theta, J_history

	
class RegularRegression(ModelBase):
	def __init__(self,name):
		self.name = name

	def displayName_(self):
		print "Regular Regression"
		
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
		
	def fnSolveModel_(self,theta0,x,y):
		"""
		Solves model
		"""
		return fnSolveModelOpMinimize(theta0,x,y)
		# return fnSolveModelGradientDescent(x, y, theta0, alpha, nIters) 		