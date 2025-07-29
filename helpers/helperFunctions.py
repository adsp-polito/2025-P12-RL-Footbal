import numpy as np


# These constants must match the Pitch object you are using
X_MIN, Y_MIN = -5, -5
X_MAX, Y_MAX = 125, 85

def normalize(x, y):
    """
    Normalize field coordinates from meters to [0, 1] range.

    Parameters:
        x (float): x coordinate in meters.
        y (float): y coordinate in meters.

    Returns:
        tuple: Normalized coordinates (x_norm, y_norm) in range [0, 1].
    """
    x_norm = (x - X_MIN) / (X_MAX - X_MIN)
    y_norm = (y - Y_MIN) / (Y_MAX - Y_MIN)
    return [x_norm, y_norm]


def denormalize(x_norm, y_norm):
    """
    Convert normalized field coordinates [0, 1] back to meters.

    Parameters:
        x_norm (float): Normalized x coordinate in range [0, 1].
        y_norm (float): Normalized y coordinate in range [0, 1].

    Returns:
        tuple: Field coordinates (x, y) in meters.
    """
    x = x_norm * (X_MAX - X_MIN) + X_MIN
    y = y_norm * (Y_MAX - Y_MIN) + Y_MIN
    return [x, y]


def distance(a, b):
    """
    Compute Euclidean distance between two points.

    Parameters:
        a (tuple): First point (x, y).
        b (tuple): Second point (x, y).

    Returns:
        float: Euclidean distance between points a and b.
    """
    return np.linalg.norm(np.array(a) - np.array(b))
