import numpy as np


def fnSigmoid(z):
    # fnSigmoid Computes sigmoid function

    return 1.0 / (1.0 + np.exp(-z));
