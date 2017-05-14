import numpy as np
import os
import pandas as pd
import paths
import sys

sys.path.append(os.path.join(paths.prj_path, 'utils/src'))

from fnCostFunction import *
from fnExtractFeatures import *
from fnRemoveFeatures import *

tol = 1e-7

# --------- test case-1 --------- 
X = np.array([1.0,2.0,3.0])
y = np.array([0.0,0.0,0.0])
theta = 1.0

out1 = 0.5*14.0/3
z = fnCostFunction(X, y, theta)
assert abs(z - out1) < tol, "FAILURE: test case-1 failed!"

# --------- test case-2 ---------
X = np.array([1.0,2.0,3.0,4.0])
y = np.array([1.0,2.0,3.0,4.0])
theta = 1.0

out2 = 0.0
z = fnCostFunction(X, y, theta)
assert abs(z - out2) < tol, "FAILURE: test case-2 failed!"

# --------- test case-3 ---------
t = np.arange(0.0*np.pi,1.0*np.pi,1/2000.0)
X = np.cos(t)
y = np.sin(t)
theta = 1.0

out3 = 0.5
z = fnCostFunction(X, y, theta)
assert abs(z - out3) < tol, "FAILURE: test case-3 failed!"

# --------- test case-5 --------- 
X = np.array([[1.0,4.0],[2.0,5.0],[3.0,6.0]])
y = np.array([0.5,0.6,0.0])
theta = np.array([1.0,2.0])

out5 = 0.5*427.21/3
z = fnCostFunction(X, y, theta)
assert abs(z - out5) < tol, "FAILURE: test case-1 failed!"

# --------- test case-5 ---------
strFileIn = paths.prj_path + '/utils/test/data-cases/data-case-5.csv'

data = pd.read_csv(strFileIn, sep=',',header=0)

X = fnRemoveFeatures(data,['y'])
y = fnExtractFeatures(data,'y')

nFeatures = (X[:1].values).shape[1]
theta = np.zeros(nFeatures) + 0.1

out5 = 0.040983598
z = fnCostFunction(X.values, y.values, theta)
assert abs(z - out5) < tol, "FAILURE: test case-3 failed!"