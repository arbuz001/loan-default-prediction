import numpy as np
import os
import pandas as pd
import paths
import sys

sys.path.append(os.path.join(paths.prj_path, 'utils/src'))
sys.path.append(os.path.join(paths.prj_path, 'models/src'))

from fnAddOnesToX import *
from fnExtractFeatures import *
from fnGradientDescent import *
from fnPredictFunction import *
from fnRemoveFeatures import *
from fnScipyLinalgLstsq import *


# --------- test case-1 --------- 
tol = 1e-10
x0 = np.array([1.0,2.0,3.0])
theta = np.array([0.1,0.2,0.3])

out1 = 1.40
z = fnPredictFunction(x0, theta)
assert abs(z - out1) < tol, "FAILURE in 'fnPredictFunction_test': test case-1 failed!"

# --------- test case-2 ---------
tol = 1e-5

strFileIn = paths.prj_path + '/utils/test/data-cases/data-case-6.csv'

data = pd.read_csv(strFileIn, sep=',',header=0)

X = fnRemoveFeatures(data,['y'])
y = fnExtractFeatures(data,'y')

X_with_ones = fnAddOnesToX(X,y)

nFeatures = (X_with_ones[:1].values).shape[1]
theta_init = np.zeros(nFeatures) + 0.1

iterations = 1500
alpha = 0.01

Z = fnGradientDescent(X_with_ones.values, y.values, theta_init, alpha, iterations)
theta = Z[0]

x0 = np.array([1.0, 0.400232238, -2.091310307, -1.57585723])
y0 = fnPredictFunction(x0, theta)

out2 = 0.519177478

assert abs(y0 - out2) < tol, "FAILURE: test case-2 failed!"