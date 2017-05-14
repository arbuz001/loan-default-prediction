import sys;
import os;
import paths;

sys.path.append(os.path.join(paths.prj_path, '/models/src/model-base/'));

from mdBase import ModelBase;

# --------- test case-1 --------- 
modelX = ModelBase("modelX");
print modelX.displayName_();

print "DONE: test '" + "mdBase_test" + "' completed!";