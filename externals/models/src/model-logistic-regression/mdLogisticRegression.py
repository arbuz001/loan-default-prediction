import numpy as np
import os
import paths
import scipy.optimize as op
import sys

sys.path.append(os.path.join(paths.prj_path, '/utils/src'));
sys.path.append(os.path.join(paths.prj_path, '/models/src/model-base/'));

from fnSigmoid import *;
from fnAddOnesToX import *;
from mdBase import ModelBase;

def fnCostFunc(theta,x,y):

	m = len(y);
	h = fnSigmoid(np.dot(x, np.transpose(theta)));
	J = 1.0/m*np.sum(-y*np.log(h) - (1-y)*np.log(1 - h));
	
	return J;

def fnGradient(theta,x,y):

	m = len(y);
	z = (fnSigmoid(np.dot(x, np.transpose(theta))) - y);
	J = 1.0/m*np.dot(x,z);
	
	return J;
	
def fnPredict(theta,x):

	XOnes = fnAddOnesToX(x);
	y = np.dot(XOnes, np.transpose(theta));

	idx = np.where((y >= 0.5));
	y = np.zeros_like(y);
	y[idx] = 1.0;

	return y;

def fnSolveModel(theta0,x,y):

	Result = op.minimize(fun = fnCostFunc, x0 = theta0, args = (x, y), method = 'TNC');
	theta = Result.x;
	
	return theta;

class LogisticRegression(ModelBase):
	def __init__(self,name):
		self.name = name

	def displayName_(self):
		print "Logistic Regression"
		
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