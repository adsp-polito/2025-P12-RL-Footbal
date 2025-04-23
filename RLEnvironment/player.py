import numpy as np

class Player:
    """
    A player class for use in simplified football environments.
    Designed for single-agent offensive scenarios (e.g., OffensiveScenarioEnv).
    Models a player's 2D position and movement using discrete actions on a normalized pitch.
    """

    def __init__(self, player_id):
        self.player_id = player_id
        self.position = np.zeros(2, dtype=np.float32)  # Current position (x, y) in normalized [0,1] scale
        self.has_ball = False  # Indicates if the player currently possesses the ball

    def reset(self, start_pos=(0.0, 0.0)):
        """
        Resets the player's state at the beginning of an episode.

        Args:
            - start_pos (tuple): Initial (x, y) position on the field, typically in normalized units [0,1].
        """
        self.position = np.array(start_pos, dtype=np.float32)
        self.has_ball = False

    def get_position(self):
        """Returns the current position of the player as a NumPy array (x, y)."""
        return self.position

    def set_position(self, new_pos):
        """
        Manually sets the player's position on the pitch.

        Args:
            - new_pos (tuple): The target (x, y) coordinates in normalized field units.
        """
        self.position = np.array(new_pos, dtype=np.float32)

    def step(self, action):
        """
        Executes a movement action based on discrete direction indexing.
        Discrete action encoding (0 to 8):
            0 = idle
            1 = move north (+y)
            2 = move south (-y)
            3 = move west (-x)
            4 = move east (+x)
            5 = move northwest (-x, +y)
            6 = move northeast (+x, +y)
            7 = move southwest (-x, -y)
            8 = move southeast (+x, -y)

        All movements occur on a normalized 120x80m pitch, where the step size is scaled accordingly.

        Note: Diagonal directions are normalized to maintain equal step magnitude compared to axial moves.

        Args:
            - action (int): Integer between 0 and 8 indicating the movement direction.
        """
        direction_map = {
            0: (0.0, 0.0),    # idle
            1: (0.0, 1.0),    # N
            2: (0.0, -1.0),   # S
            3: (-1.0, 0.0),   # W
            4: (1.0, 0.0),    # E
            5: (-1.0, 1.0),   # NW
            6: (1.0, 1.0),    # NE
            7: (-1.0, -1.0),  # SW
            8: (1.0, -1.0)    # SE
        }

        dx, dy = direction_map.get(action, (0.0, 0.0))

        # === Step size per frame ===
        # Field width is normalized to 1.0 (i.e., 120m → 1.0 unit)
        # Therefore, one meter = 1/120 unit in this scale.
        # This allows discrete grid movement without breaking normalization.
        step_size = 1.0 / 120.0  # ~1 meter per frame

        movement = np.array([dx, dy], dtype=np.float32)

        # Normalize diagonal movement to ensure uniform travel distance
        norm = np.linalg.norm(movement)
        if norm > 0:
            movement = (movement / norm) * step_size  # Ensures same speed in diagonal as straight

        self.position += movement