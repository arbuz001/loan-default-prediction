import numpy as np
import pandas as pd
import os

def fnExtractFeatures(Z, features, bOutputToFile = False):
	#	
	# fnExtractFeatures Extracts features from the list 'features'
	#
	
	zz = Z[features]
	if bOutputToFile == True:
		filename = './intermediate_log/out_fnExtractFeatures_' + str(features[0]) + '.csv'
		
		dir = os.path.dirname(filename)
		try:
			os.stat(dir)
		except:
			os.mkdir(dir) 
			
		zz.to_csv(filename, index = False)

	return zz.values