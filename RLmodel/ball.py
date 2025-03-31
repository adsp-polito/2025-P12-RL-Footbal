import numpy as np

class Ball:
    """
    The Ball class represents the football used in the environment
    It includes its position and velocity on the pitch, and updates
    its motion at each time step.

    Attributes:
        - position (np.ndarray): Current position of the ball [x, y]
        - velocity (np.ndarray): Current velocity vector [vx, vy]
    """

    def __init__(self):

        # Initial position (center of the field)
        self.init_position = np.array([0.5, 0.5], dtype=np.float32)

        # Current position and velocity
        self.position = self.init_position.copy()
        self.velocity = np.zeros(2, dtype=np.float32)

    def reset(self):
        """Reset the ball to the initial state"""
        self.position = self.init_position.copy()
        self.velocity = np.zeros(2, dtype=np.float32)

    def update(self):
        """
        Update the ball's position based on its current velocity
        This version uses a simple linear model without friction
        Future versions can include collisions, possession, and drag
        """
        self.position += self.velocity

    def get_position(self):
        """Returns the current position of the ball"""
        return self.position