import sys
import os
import numpy as np
import paths

sys.path.append(os.path.join(paths.prj_path, '/utils/src'))

from fnCostFunction import *

def fnGradientDescent(X, y, theta, alpha, nIters):
	# GRADIENTDESCENT Performs gradient descent to learn theta
		# theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
		# taking num_iters gradient steps with learning rate alpha

	m = len(y)
	J_history = np.array([])
	
	for iter in range(1, nIters):
		theta = theta - alpha/m*np.dot(np.transpose(X) , np.dot(X,theta) - y)
		
		# Save the cost J in every iteration    
		J_history = np.append(J_history,fnCostFunction(X, y, theta))

	return theta, J_history