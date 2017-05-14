import sys
import os
import numpy as np
import paths
import scipy.optimize as op

sys.path.append(os.path.join(paths.prj_path, '/utils/src'))

from fnCostFunction import *

def fnNumOptimization(X, y, theta):
	# fnNumOptimization performs numerical optimization using pyhton built-in algorithm 
		# theta = fnNumOptimization(X, y, theta, alpha, num_iters) updates theta by 
		# taking num_iters gradient steps with learning rate alpha

	# m = len(y)
	# J_history = np.array([])
	
	# for iter in range(1, nIters):
		# theta = theta - alpha/m*np.dot(np.transpose(X) , np.dot(X,theta) - y)
		
		# # Save the cost J in every iteration    
		# J_history = np.append(J_history,fnCostFunction(X, y, theta))
	
	# Result = op.minimize(fun = fnCostFunction, x0 = theta, args = (X, y), method = 'TNC', jac = Gradient)
	Result = op.minimize(fun = fnCostFunction, x0 = theta, args = (X, y), method = 'TNC')
	
	print Result

	return Result.x