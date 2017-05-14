import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import pandas as pd
import paths

sys.path.append(os.path.join(paths.prj_path, 'utils/src'))
sys.path.append(os.path.join(paths.prj_path, 'models/src'))

from fnAddOnesToX import *
from fnExtractFeatures import *
from fnGradientDescent import *
from fnRemoveFeatures import *

tol = 1e-7
	
# --------- test case-1 --------- 

X_with_ones = np.array([[1.0,1.1],[1.0,2.1],[1.0,3.1]])
y = np.array([-2.67, -5.37, -8.07])

theta_init = np.array([0.0,0.0])

iterations = 5000
alpha = 0.03

Z = fnGradientDescent(X_with_ones, y, theta_init, alpha, iterations)
theta = Z[0]

# err = Z[1]
# plt.plot(Z[1])
# plt.xlabel('iteration')
# plt.ylabel('error value')
# plt.title('error decay')
# plt.show()

out1 = np.array([0.3,-2.7])
assert max(abs(theta - out1)) < tol, "FAILURE: test case-1 failed!"

# --------- test case-2 --------- 
strFileIn = paths.prj_path + '/models/test/data-cases/data-case-2.csv'

data = pd.read_csv(strFileIn, sep=',',header=0)

X = fnRemoveFeatures(data,['y'])
y = fnExtractFeatures(data,'y')

X_with_ones = fnAddOnesToX(X,y)

nFeatures = (X_with_ones[:1].values).shape[1]
theta_init = np.zeros(nFeatures)

iterations = 2000
alpha = 0.02

Z = fnGradientDescent(X_with_ones.values, y.values, theta_init, alpha, iterations)
theta = Z[0]

# err = Z[1]
# plt.plot(Z[1])
# plt.xlabel('iteration')
# plt.ylabel('error value')
# plt.title('error decay')
# plt.show()

out2 = np.array([0.0,0.2,-0.1])
assert max(abs(theta - out2)) < tol, "FAILURE: test case-2 failed!"

# # --------- test case-3 --------- 
# strFileIn = paths.prj_path + '/models/test/data-cases/data-case-3.csv'

# data = pd.read_csv(strFileIn, sep=',',header=0)

# X = fnRemoveFeatures(data,['y'])
# y = fnExtractFeatures(data,'y')

# X_with_ones = fnAddOnesToX(X,y,True)

# nFeatures = (X_with_ones[:1].values).shape[1]
# theta_init = np.zeros(nFeatures)

# iterations = 4000
# alpha = 0.02

# Z = fnGradientDescent(X_with_ones.values, y.values, theta_init, alpha, iterations)
# theta = Z[0]

# # err = Z[1]
# # plt.plot(Z[1])
# # plt.xlabel('iteration')
# # plt.ylabel('error value')
# # plt.title('error decay')
# # plt.show()

# out3 = np.array([0.0,0.2])
# assert max(abs(theta - out3)) < tol, "FAILURE: test case-3 failed!"