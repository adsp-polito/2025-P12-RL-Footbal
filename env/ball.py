import numpy as np

# === Normalization parameters ===
# Field space: X ∈ [-5, 125], Y ∈ [-5, 85] → Normalized [0,1] x [0,1]
X_MIN = -5
Y_MIN = -5
PITCH_WIDTH = 130   # From -5 to 125
PITCH_HEIGHT = 90   # From -5 to 85

class Ball:
    """
    A class representing the football in a simplified environment.
    Position is stored in normalized space ([0,1] × [0,1]), relative to the extended field area:
    X ∈ [-5, 125], Y ∈ [-5, 85] meters.

    The ball can be moved, reset, or checked for boundary violations.
    """

    def __init__(self):
        # Center of the actual pitch in absolute meters: (60/2 = 30, 80/2 = 40)
        # Adjust to absolute reference frame then normalize
        abs_cx = 30 + abs(X_MIN)
        abs_cy = 40 + abs(Y_MIN)
        cx = abs_cx / PITCH_WIDTH
        cy = abs_cy / PITCH_HEIGHT
        self.init_position = np.array([cx, cy], dtype=np.float32)

        # Current position (normalized)
        self.position = self.init_position.copy()

        # ID of the player currently in possession
        self.owner_id = None

    def reset(self):
        """
        Resets the ball to the center of the pitch (normalized) and clears possession.
        """
        self.position = self.init_position.copy()
        self.owner_id = None

    def set_position(self, new_pos):
        """
        Sets the ball's position manually.

        Args:
            - new_pos (tuple): New (x, y) position in normalized space
        """
        self.position = np.array(new_pos, dtype=np.float32)

    def get_position(self):
        """
        Returns the current normalized position of the ball.

        Returns:
            - np.ndarray: Ball position as (x, y)
        """
        return self.position

    def is_out_of_bounds(self):
        """
        Checks if the ball is outside the normalized range [0, 1] × [0, 1].

        Returns:
            - bool: True if ball has exited the pitch area
        """
        x, y = self.position
        return not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)