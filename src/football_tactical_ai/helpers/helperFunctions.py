import numpy as np
import os
from football_tactical_ai.configs import pitchSettings as PS

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

def ensure_dirs(path_or_paths):
    """
    Ensure that one or more directories exist.

    Args:
        path_or_paths (str | list[str] | dict): Path(s) to ensure exist.
    """
    if isinstance(path_or_paths, dict):
        paths = path_or_paths.values()
    elif isinstance(path_or_paths, str):
        paths = [path_or_paths]
    else:
        paths = path_or_paths

    for path in paths:
        dir_path = os.path.dirname(path) if os.path.splitext(path)[1] else path
        os.makedirs(dir_path, exist_ok=True)