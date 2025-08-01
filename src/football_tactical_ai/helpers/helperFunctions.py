import numpy as np
from football_tactical_ai.configs import pitch_settings as PS

def normalize(x, y):
    """
    Normalize field coordinates from meters to [0, 1] range.

    Parameters:
        x (float): x coordinate in meters.
        y (float): y coordinate in meters.

    Returns:
        tuple: Normalized coordinates (x_norm, y_norm) in range [0, 1].
    """
    x_norm = (x - PS.X_MIN) / (PS.X_MAX - PS.X_MIN)
    y_norm = (y - PS.Y_MIN) / (PS.Y_MAX - PS.Y_MIN)
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
    x = x_norm * (PS.X_MAX - PS.X_MIN) + PS.X_MIN
    y = y_norm * (PS.Y_MAX - PS.Y_MIN) + PS.Y_MIN
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
