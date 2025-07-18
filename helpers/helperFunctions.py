import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from env.pitch import X_MIN, X_MAX, Y_MIN, Y_MAX

def normalize(x, y):
    """
    Normalize field coordinates from meters to [0, 1] range
    """
    x_norm = (x - X_MIN) / (X_MAX - X_MIN)
    y_norm = (y - Y_MIN) / (Y_MAX - Y_MIN)
    return x_norm, y_norm


def distance(a, b):
    """
    Compute Euclidean distance between two points
    """
    return np.linalg.norm(np.array(a) - np.array(b))