import numpy as np
from scipy import linalg

def fnPredictFunction(x0, theta):
	# fnPredictFunction Predicts value y for a given x0
	#
	
	# y = (np.transpose(x0).T).dot(theta)
	y = np.dot(np.transpose(x0),theta)
	
	return y