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
        - abbr (str): Abbreviation of the role (e.g., 'CM', 'RB') for visual display
        - position (np.ndarray): Current position of the player on the field [x, y]
        - velocity (np.ndarray): Current velocity vector [vx, vy]
        - max_speed (float): Maximum speed the player can reach (in m/s)
        - has_ball (bool): Whether the player is currently in possession of the ball
    """

    def __init__(self, player_id, team_id, role, init_position, abbr=None):
        # Identifiers and positional role
        self.player_id = player_id
        self.team_id = team_id
        self.role = role
        self.abbr = abbr if abbr is not None else role[:2]  # fallback if not provided

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
        Updates the player's state based on the action taken

        Args:
            action (int): Discrete action index
                0 = stay, 1 = up, 2 = down, 3 = left,
                4 = right, 5 = up-left, 6 = up-right, 
                7 = down-left, 8 = down-right
        """
        direction_map = {
            0: np.array([0.0, 0.0]),
            1: np.array([0.0, 1.0]),
            2: np.array([0.0, -1.0]),
            3: np.array([-1.0, 0.0]),
            4: np.array([1.0, 0.0]),
            5: np.array([-1.0, 1.0]),
            6: np.array([1.0, 1.0]),
            7: np.array([-1.0, -1.0]),
            8: np.array([1.0, -1.0])
        }
        # Get the direction vector corresponding to the selected discrete action (e.g., up, down, left, etc.).
        # Defaults to stationary if action is invalid.
        direction = direction_map.get(action, np.array([0.0, 0.0]))  

        # Compute the magnitude (length) of the direction vector.
        # This is used to normalize the vector, ensuring uniform speed across directions 
        # (e.g., diagonals are not faster than vertical/horizontal moves).
        norm = np.linalg.norm(direction)  
       
        if norm > 0:

            # First: normalize the direction vector to ensure unit movement in that direction
            # Then: scale it by (max_speed / 24) to get the movement per frame (since we simulate 24 FPS)
            # Finally: divide by 120 (field length) to normalize the velocity to the coordinate system [0,1]
            # This ensures that 10 m/s corresponds correctly to field-relative movement per frame
            self.velocity = (direction / norm) * (self.max_speed / 24) / 120  #
        
        else:
            self.velocity = np.zeros(2)

        self.position += self.velocity

    def get_position(self):
        """
        Returns the current position of the player on the field

        Returns:
            np.ndarray: A 2D vector representing the player's position [x, y]
        """
        return self.position
