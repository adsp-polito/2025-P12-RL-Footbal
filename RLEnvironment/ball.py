import numpy as np

class Ball:
    """
    A class representing a ball in a simplified football environment.
    The ball's position is represented in a normalized 2D space, typically [0,1] x [0,1].
    The ball can be reset to its initial position and can be set to a specific position.
    The ball can also be checked for out-of-bounds conditions.
    """

    def __init__(self):
        # Initial (normalized) position, default to center
        self.init_position = np.array([0.5, 0.5], dtype=np.float32)
        self.position = self.init_position.copy()
        self.owner_id = None  # ID of possessing player, or None if free

    def reset(self):
        """Reset the ball to the initial state."""
        self.position = self.init_position.copy()
        self.owner_id = None

    def set_position(self, new_pos):
        """
        Explicitly set the ball's position (e.g., to match the player).

        Args:
            - new_pos (tuple): New (x, y) coordinates in normalized units.
        """
        self.position = np.array(new_pos, dtype=np.float32)

    def get_position(self):
        """Returns the current position of the ball as a NumPy array."""
        return self.position

    def is_out_of_bounds(self):
        """
        Checks if the ball has gone beyond the [0,1] x [0,1] normalized pitch limits.
        This can be used for reward or termination logic.
        """
        x, y = self.position
        return x < 0 or x > 1 or y < 0 or y > 1