import numpy as np
import os
import pandas as pd
import paths
import sys

sys.path.append(os.path.join(paths.prj_path, 'utils/src'))
sys.path.append(os.path.join(paths.prj_path, 'models/src'))

from fnSigmoid import *
from fnRemoveFeatures import *

# --------- test case-1 --------- 
tol = 1e-8
z = np.array([0.8381620741857530,0.1925116063970390,0.3866884921473850])

g = fnSigmoid(z)

out1 = np.array([0.698077986444712,0.547979812519007,0.595485267672453])

assert max(abs(g - out1)) < tol, "FAILURE in 'fnPredictFunction_test': test case-1 failed!"

# --------- test case-2 ---------
tol = 1e-5

strFileIn = paths.prj_path + '/utils/test/data-cases/data-case-6.csv'

data = pd.read_csv(strFileIn, sep=',',header=0)

X = fnRemoveFeatures(data,['y'])

g = fnSigmoid(X)

strFileIn = paths.prj_path + '/utils/test/data-cases/data-case-7.csv'

out2 = pd.read_csv(strFileIn, sep=',',header=0)

assert np.amax(abs(g - out2.values)) < tol, "FAILURE in 'fnPredictFunction_test': test case-2 failed!"

print "DONE: test '" + "fnSigmoid_test" + "' completed!";