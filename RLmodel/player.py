import numpy as np

class Player:
    """
    The Player class models the individual behavior and state of a football player.
    Each player can be either controlled by a reinforcement learning agent or act
    according to a rule-based logic.

    Attributes:
        - player_id (int): Unique identifier for the player
        - team_id (int): Identifier for the team the player belongs to (0 or 1)
        - role (str): Positional role of the player (e.g., 'GK', 'CB', 'CM', 'ST')
        - position (np.ndarray): Current position of the player on the field [x, y]
        - velocity (np.ndarray): Current velocity vector [vx, vy]
        - max_speed (float): Maximum speed the player can reach (in m/s)
        - has_ball (bool): Whether the player is currently in possession of the ball
    """

    def __init__(self, player_id, team_id, role, init_position):
        # Identifiers and positional role
        self.player_id = player_id
        self.team_id = team_id
        self.role = role

        # Initial and current position on the field
        self.init_position = np.array(init_position, dtype=np.float32)
        self.position = np.array(init_position, dtype=np.float32)

        # Velocity vector (vx, vy)
        self.velocity = np.zeros(2, dtype=np.float32)

        # Maximum allowed speed (10 m/s equivalent to 36 km/h)
        self.max_speed = 10.0

        # Possession state
        self.has_ball = False

    def reset(self):
        """
        Resets the player to the initial state (e.g., at the beginning of a new episode)
        This includes resetting the position, velocity, and ball possession status
        """
        self.position = self.init_position.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.has_ball = False

    def step(self, action):
        """
        Updates the player's state based on the action taken.
        The same directional logic applies whether or not the player has possession of the ball.

        Args:
            action (int): Discrete action index
                0 = stay, 1 = up, 2 = down, 3 = left,
                4 = right, 5 = up-left, 6 = up-right, 
                7 = down-left, 8 = down-right
        """

        # Map each discrete action to a direction vector
        direction_map = {
            0: np.array([0.0, 0.0]),       # Stay
            1: np.array([0.0, 1.0]),       # Up
            2: np.array([0.0, -1.0]),      # Down
            3: np.array([-1.0, 0.0]),      # Left
            4: np.array([1.0, 0.0]),       # Right
            5: np.array([-1.0, 1.0]),      # Up-Left
            6: np.array([1.0, 1.0]),       # Up-Right
            7: np.array([-1.0, -1.0]),     # Down-Left
            8: np.array([1.0, -1.0])       # Down-Right
        }

        # Move according to direction
        direction = direction_map.get(action, np.array([0.0, 0.0]))

        # Normalize direction and apply movement scaled by max speed and FPS (24)
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.velocity = (direction / norm) * (self.max_speed / 24) /120 # normalized by pitch length
        else:
            self.velocity = np.zeros(2)

        # Update player position
        self.position += self.velocity

    def get_position(self):
        """
        Returns the current position of the player on the field

        Returns:
            np.ndarray: A 2D vector representing the player's position [x, y]
        """
        return self.position
