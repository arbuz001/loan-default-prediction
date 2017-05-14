import unittest
import sys
import os
import paths
import compileall

# re-compile all modules before running unit tests
compileall.compile_dir(paths.prj_path, force = True, quiet = 1)

sys.path.append(os.path.join(paths.prj_path,'models/src/model-base/'))
sys.path.append(os.path.join(paths.prj_path,'models/src/model-logistic-regression/'))
sys.path.append(os.path.join(paths.prj_path,'models/test/model-base/'))
sys.path.append(os.path.join(paths.prj_path,'models/test/model-logistic-regression/'))
sys.path.append(os.path.join(paths.prj_path,'utils/test'))
sys.path.append(os.path.join(paths.prj_path,'models/test/model-regular-regression/'))
sys.path.append(os.path.join(paths.prj_path,'models/src/model-regular-regression/'))

testmodules = [
	'mdRegularRegression_test',
	'fnSigmoid_test',
	'mdLogisticRegression_test',
	'mdBase_test'
    ]

suite = unittest.TestSuite()

for t in testmodules:
    try:
        # If the module defines a suite() function, call it to get the suite.
        mod = __import__(t, globals(), locals(), ['suite'])
        suitefn = getattr(mod, 'suite')
        suite.addTest(suitefn())
    except (ImportError, AttributeError):
        # else, just load all the test cases from the module.
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

unittest.TextTestRunner().run(suite)