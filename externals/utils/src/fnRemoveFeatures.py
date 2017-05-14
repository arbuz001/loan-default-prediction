import numpy as np
import pandas as pd
import os

def fnRemoveFeatures(Z, filter_out, bOutputToFile = False):
	#	
	# fnRemoveFeatures Removes features from the list 'features'
	#
	
	features = list(f for f in Z.columns if f not in filter_out)
	zz = Z[features]
	
	if bOutputToFile == True:
		filename = './intermediate_log/out_fnRemoveFeatures_'+ str(features[0]) + '.csv'
		
		dir = os.path.dirname(filename)
		try:
			os.stat(dir)
		except:
			os.mkdir(dir) 
		
		zz.to_csv(filename, index = False)

	return zz.values