import sys;
import os;
import numpy as np;
import pandas as pd;
import paths;
import matplotlib.pyplot as plt;

sys.path.append(os.path.join(paths.prj_path, '/utils/src/'));
sys.path.append(os.path.join(paths.prj_path, '/models/src/model-regular-regression/'));
sys.path.append(os.path.join(paths.prj_path, '/models/src/model-base/'));

# from mdRegularRegression import RegularRegression;
from fnAddOnesToX import *;
from fnExtractFeatures import *;
from fnRemoveFeatures import *;

tol = 1e-3;

# # --------- test case-1 --------- 

# # data in: 
# #	X = [[1,2],[3,4]] 
# #	y = [1.49,3.13]
# # expected out:
# #   theta = [0.15, 0.67]
	
# X = np.array([[1,2],[3,4]]);
# y = np.array([1.49,3.13]);

# m , n = X.shape;
# theta0 = np.zeros(n);

# modelZ = RegularRegression("modelZ");

# cost0 = modelZ.fnCostFunc_(theta0,X,y)

# theta = modelZ.fnSolveModel_(theta0, X, y)

# y = modelZ.fnPredict_(theta,X)

# out1 = np.array([0.15, 0.67]);
# assert max(abs(theta - out1)) < tol, "FAILURE: test case-1 failed!"


# --------- test case-2 --------- 

# data in: 
#	X = [[1,2],[3,4]] 
#	y = [1.49,3.13]
# expected out:
#   theta = [0.15, 0.67]
	
strFileIn = paths.prj_path + '/models/test/data-cases/data-fx-historical.csv';
data = pd.read_csv(strFileIn, sep=',', header=0);

y = fnExtractFeatures(data,'RUB-USD');

XOnes = fnAddOnesToX(X);

m , nFeatures = XOnes.shape;
theta0 = np.zeros(nFeatures);

modelX = LogisticRegression("modelX");

theta = modelX.fnSolveModel_(theta0,XOnes, y)
yy = modelX.fnPredict_(theta,X)

out1 = np.array([0.15, 0.67]);
assert max(abs(theta - out1)) < tol, "FAILURE: test case-2 failed!"

print "DONE: test '" + "mdRegularRegression_test" + "' completed!";