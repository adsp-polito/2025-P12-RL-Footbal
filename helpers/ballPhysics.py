
import numpy as np

def apply_friction(velocity, friction_factor=0.2):
    """
    Applies friction to reduce ball velocity over time.
    
    Args:
        - velocity (np.ndarray): current velocity vector [vx, vy]
        - friction_factor (float): multiplicative decay per frame (between 0 and 1)
    
    Returns:
        np.ndarray: updated velocity vector after applying friction
    """
    return velocity * (1-friction_factor)