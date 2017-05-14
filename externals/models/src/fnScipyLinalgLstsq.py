import numpy as np
from scipy import linalg

def fnScipyLinalgLstsq(A, y):
	# Performs linear least squares using efficient scipy implementation

	linalg.lstsq(A, y)

	return 1.0