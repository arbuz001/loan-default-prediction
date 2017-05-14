import numpy as np
import pandas as pd
import os

def fnAddOnesToX(X, bOutputToFile = False):
	#	
	# fnAddOnesToX Adds vector of ones to X matrix
	#

	m , n = X.shape;
	
	XX = np.hstack((np.ones((m, 1), dtype=X.dtype) , X));

	if bOutputToFile == True:
		filename = './intermediate_log/out_fnAddOnesToX.csv'
		
		dir = os.path.dirname(filename)
		try:
			os.stat(dir)
		except:
			os.mkdir(dir) 
			
		# XX.to_csv(filename, index = False)

	return XX
	
	
