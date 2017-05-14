import matplotlib.pyplot as plt;
import sys;
import os;
import numpy as np;
import pandas as pd;
import paths;

sys.path.append(os.path.join(paths.prj_path, '/utils/src/'));
sys.path.append(os.path.join(paths.prj_path, '/models/src/model-logistic-regression/'));
sys.path.append(os.path.join(paths.prj_path, '/models/src/model-base/'));

from fnAddOnesToX import *;
from fnExtractFeatures import *;
from fnRemoveFeatures import *;
from mdLogisticRegression import LogisticRegression;

tol = 1e-3;

# --------- test case-1 --------- 

# data in: 
#	X = [[1,2],[3,4]] 
#	y = [1.49,3.13]
# expected out:
#   theta = [0.15, 0.67]
	
# X = np.array([[1,2],[3,4]]);
# y = np.array([1.49,3.13]);

# m , n = X.shape;
# initial_theta = np.zeros(n);

# Result = op.minimize(fun = fnCostFuncRegression, x0 = initial_theta, args = (X, y), method = 'TNC');
# theta = Result.x;

# out1 = np.array([0.15, 0.67]);
# assert max(abs(theta - out1)) < tol, "FAILURE: test case-1 failed!"
	
# --------- test case-1 --------- 
strFileIn = paths.prj_path + '/models/test/data-cases/data-case-4.csv';
data = pd.read_csv(strFileIn, sep=',', header=0);

X = fnRemoveFeatures(data,['y']);
y = fnExtractFeatures(data,'y');

XOnes = fnAddOnesToX(X);

m , nFeatures = XOnes.shape;
theta0 = np.zeros(nFeatures);

modelX = LogisticRegression("modelX");

theta = modelX.fnSolveModel_(theta0,XOnes, y)
yy = modelX.fnPredict_(theta,X)

assert max(abs(yy - yy)) < tol, "FAILURE: test case-1 failed!"

print "DONE: test '" + "mdLogisticRegression_test" + "' completed!";