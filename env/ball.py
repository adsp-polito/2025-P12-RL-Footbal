import numpy as np

# Normalization parameters
X_MIN, Y_MIN = -5, -5
X_MAX, Y_MAX = 125, 85
PITCH_WIDTH = X_MAX - X_MIN   # 130 meters
PITCH_HEIGHT = Y_MAX - Y_MIN  # 90 meters

class Ball:
    """
    A class representing the football in a simplified environment.

    The ball's position is stored in normalized space ([0, 1] x [0, 1]),
    relative to the extended field dimensions with margins:
    X ∈ [-5, 125], Y ∈ [-5, 85] meters.

    Methods allow resetting, moving, and checking the ball's state.
    """

    def __init__(self):
        """
        Initializes the ball at the center of the pitch (normalized) and sets no owner.
        """
        # Absolute center of the pitch in meters (adjusted for extended bounds)
        abs_cx = 60 + abs(X_MIN)
        abs_cy = 40 + abs(Y_MIN)

        # Normalize to [0, 1]
        cx = abs_cx / PITCH_WIDTH
        cy = abs_cy / PITCH_HEIGHT
        self.init_position = np.array([cx, cy], dtype=np.float32)

        # Current normalized position
        self.position = self.init_position.copy()

        # ID of the player currently in possession (None if free)
        self.owner_id = None

    def reset(self):
        """
        Resets the ball to the initial center position (normalized) and clears possession.
        """
        self.position = self.init_position.copy()
        self.owner_id = None

    def set_position(self, new_pos):
        """
        Sets the ball's position manually.

        Parameters:
            new_pos (tuple or list): New (x, y) position in normalized space [0, 1].
        """
        self.position = np.array(new_pos, dtype=np.float32)

    def get_position(self):
        """
        Returns the current normalized position of the ball.

        Returns:
            np.ndarray: Current ball position as (x, y).
        """
        return self.position

    def is_out_of_bounds(self):
        """
        Checks whether the ball is outside the valid normalized [0, 1] pitch area.

        Returns:
            bool: True if the ball is out of bounds, False otherwise.
        """
        x, y = self.position
        return not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)