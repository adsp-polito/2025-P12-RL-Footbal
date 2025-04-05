import numpy as np

class Player:
    """
    The Player class models the individual behavior and state of a football player.
    Each player can be either controlled by a reinforcement learning agent or act
    according to a rule-based logic. The player has attributes for identification,
    position, velocity, speed, and ball possession.

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

    def __init__(self, player_id, team_id, role, init_position, abbr=None, movement_mode="discrete"):
        # Identifiers and positional role
        self.player_id = player_id
        self.team_id = team_id
        self.role = role
        
        # Fallback if abbreviation is not provided
        self.abbr = abbr if abbr is not None else role[:2]

        # Initialize position and current position on the field
        self.init_position = np.array(init_position, dtype=np.float32)
        self.position = np.array(init_position, dtype=np.float32)

        # Initialize velocity vector (vx, vy)
        self.velocity = np.zeros(2, dtype=np.float32)

        # Maximum allowed speed (10 m/s equivalent to 36 km/h) and movement mode
        self.max_speed = 10.0
        self.movement_mode = movement_mode  # either 'discrete' or 'continuous'

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
            action (int or list): Discrete action index or continuous movement vector
        """
        if self.movement_mode == "discrete":
            # Map discrete actions to direction vectors
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
            direction = direction_map.get(action, np.array([0.0, 0.0]))  
        elif self.movement_mode == "continuous":
            direction = np.array(action, dtype=np.float32)
            assert direction.shape == (2,), "Action must be a 2D vector in continuous mode."
            assert np.all(np.isfinite(direction)), "Direction vector contains invalid values"
        else:
            raise ValueError(f"Unsupported movement_mode: {self.movement_mode}")

        # Compute the magnitude (length) of the direction vector.
        norm = np.linalg.norm(direction)  
       
        # Normalize the direction vector if its length is greater than 0
        # and scale it to the maximum speed.
        if norm > 0:
            self.velocity = (direction / norm) * (self.max_speed / 24) / 120
        else:
            self.velocity = np.zeros(2)

        # Update player position based on current velocity
        self.position += self.velocity
        
    def pass_to_id(self, receiver_id, all_players, ball, min_speed=5.0, max_speed=35.0, noise_factor=0.05):
        """
        Pass the ball to a teammate identified by receiver_id.
        This version avoids passing a full Player object and instead uses the shared player list.

        Args:
            receiver_id (int): ID of the teammate to receive the ball
            all_players (list[Player]): List of all players in the environment
            ball (Ball): Ball object to be passed
            min_speed (float): Minimum speed to apply to the ball
            max_speed (float): Maximum speed to apply to the ball
            noise_factor (float): Random variation factor for pass realism
        """
        # Calculate distance to the teammate
        teammate = all_players[receiver_id]
        dx = teammate.position[0] - self.position[0]
        dy = teammate.position[1] - self.position[1]
        distance = np.hypot(dx, dy)

        if distance == 0:
            return

        # Calculate speed based on distance and add noise for realism
        direction = np.array([dx, dy]) / distance
        distance_m = distance * 120
        full_range = 75.0
        base_speed = min_speed + (max_speed - min_speed) * min(distance_m / full_range, 1.0)
        variation = np.random.uniform(-noise_factor, noise_factor)
        speed = base_speed * (1 + variation)

        # Assign direction to the ball's velocity for the pass
        ball.velocity = direction * speed
        
        # Track the player who initiated the pass for reward evaluation
        ball.just_passed_by = self.player_id

    def get_position(self):
        """
        Returns the current position of the player on the field

        Returns:
            np.ndarray: A 2D vector representing the player's position [x, y]
        """
        return self.position

    def is_out_of_bounds(self):
        """
        Returns True if the player is outside the normalized pitch boundaries [0,1] x [0,1]
        """
        x, y = self.position
        return x < 0 or x > 1 or y < 0 or y > 1

    def shoot(self, ball, target=np.array([1.0, 0.5]), speed=40.0):
        """
        Shoots the ball toward a specified target location with high velocity
        
        Args:
            - ball (Ball): Ball object to be affected.
            - target (np.ndarray): Target coordinates in normalized field space (default: center of right goal).
            - speed (float): Speed in m/s to apply to the ball.
        """
        # Compute the direction vector from the player to the target
        direction = target - self.position
        norm = np.linalg.norm(direction)

        # Avoid division by zero if target is current position
        if norm == 0:
            return

        # Normalize the direction and apply velocity to the ball
        direction = direction / norm
        ball.velocity = direction * speed

    def tackle(self, ball, target_player, tackle_radius=0.02):
        """
        Attempts to steal the ball from another player if within a defined radius.
        
        Args:
            - ball (Ball): Ball object in play.
            - target_player (Player): The player currently possessing the ball.
            - tackle_radius (float): Maximum distance to attempt a successful tackle.
        
        Returns:
            - bool: True if the tackle was successful and possession was taken.
        """
        # Check if the target player actually has the ball
        if not target_player.has_ball:
            return False

        # Compute the distance between this player and the target
        distance = np.linalg.norm(self.position - target_player.position)

        # If within tackle radius, transfer ball possession
        if distance <= tackle_radius:
            ball.owner_id = self.player_id
            self.has_ball = True
            target_player.has_ball = False
            return True

        # Tackle unsuccessful
        return False
