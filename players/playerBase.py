import numpy as np
from env.pitch import CELL_SIZE

# === GLOBAL NORMALIZATION CONSTANTS (same as environment) ===
X_MIN = -5
Y_MIN = -5
PITCH_WIDTH = 130  # From -5 to 125
PITCH_HEIGHT = 90  # From -5 to 85
FPS = 24

PLAYER_SPEED_M_S = 5.0  # Constant speed in meters per second

class BasePlayer:
    """
    A player class for use in simplified football environments.
    The player moves smoothly from one grid cell center to another using discrete actions.
    Movement and position are expressed in normalized space based on an extended field grid
    (X ∈ [-5, 125], Y ∈ [-5, 85] meters → [0,1] normalized).
    """

    def __init__(self, player_id):
        self.player_id = player_id
        self.position = np.zeros(2, dtype=np.float32)  # Normalized position
        self.target = None                             # Target position (normalized)
        self.has_ball = False
        self.last_direction = np.zeros(2, dtype=np.float32)

    def reset(self, start_pos=(0.0, 0.0)):
        """Resets the player's state."""
        self.position = np.array(start_pos, dtype=np.float32)
        self.target = None
        self.has_ball = False
        self.last_direction = np.zeros(2, dtype=np.float32)

    def get_position(self):
        """Returns the current normalized player position (x, y)."""
        return self.position

    def set_position(self, new_pos):
        """Sets the player's position (normalized) and stops movement."""
        self.position = np.array(new_pos, dtype=np.float32)
        self.target = None

    def step(self, action):
        """
        Sets the target grid cell center based on the action.

        Args:
            - action (int): Discrete action from 0 to 8 (idle and directional)
        """
        direction_map = {
            0: (0, 0), 1: (0, 1), 2: (0, -1),
            3: (-1, 0), 4: (1, 0), 5: (-1, 1),
            6: (1, 1), 7: (-1, -1), 8: (1, -1)
        }
        dx, dy = direction_map.get(action, (0, 0))

        # Convert current normalized position to absolute meters
        abs_pos = self.position * [PITCH_WIDTH, PITCH_HEIGHT] + [X_MIN, Y_MIN]

        # Find current grid cell
        col = int(abs_pos[0] // CELL_SIZE)
        row = int(abs_pos[1] // CELL_SIZE)

        # Determine target cell center in meters
        target_x = (col + dx) * CELL_SIZE + CELL_SIZE / 2
        target_y = (row + dy) * CELL_SIZE + CELL_SIZE / 2

        # Normalize target position
        self.target = ((np.array([target_x, target_y]) - [X_MIN, Y_MIN]) / 
                       [PITCH_WIDTH, PITCH_HEIGHT]).astype(np.float32)

        # Store direction
        raw = np.array([dx, dy], dtype=np.float32)
        norm = np.linalg.norm(raw)
        self.last_direction = raw / norm if norm > 0 else np.zeros(2)

    def update_position(self):
        """
        Moves the player toward the target cell center.
        Velocity is fixed in real-world meters/second, normalized to [0,1].
        """
        if self.target is None:
            return

        direction = self.target - self.position
        dist = np.linalg.norm(direction)

        if dist > 1e-6:
            self.last_direction = direction / dist

        # Compute normalized step size based on real-world speed
        meters_per_frame = PLAYER_SPEED_M_S / FPS
        step_size = meters_per_frame / PITCH_WIDTH

        if dist <= step_size:
            self.position = self.target
            self.target = None
        else:
            self.position += (direction / dist) * step_size

    def get_ball_offset(self, offset_meters=1.0):
        """
        Return an offset vector ahead of the player, in normalized units.
        Used to place the ball slightly in front of the player while moving.

        Args:
            offset_meters (float): Distance in meters
        """
        return self.last_direction * (offset_meters / PITCH_WIDTH)