import numpy as np

class Ball:
    """
    Class representing a ball in a normalized 2D pitch coordinate system

    Attributes:
        position (np.ndarray): Current normalized position on the pitch, with x and y values in [0,1]
        velocity (np.ndarray): Current velocity vector, normalized per second
        owner_id (int or None): ID of the player currently possessing the ball, or None if free
    """

    def __init__(self, position=None):
        """
        Initialize the ball object

        Args:
            position (tuple, list, np.ndarray, optional): Initial normalized position [x, y]
            Defaults to the center of the pitch at [0.5, 0.5] if not specified.
        """
        # Set initial position; default is pitch center
        if position is None:
            position = [0.5, 0.5]
        self.position = np.array(position, dtype=np.float32)

        # Initialize velocity vector as zero
        self.velocity = np.zeros(2, dtype=np.float32)

        # Initialize owner as None
        self.owner = None
        self.radius = 0.11  # Ball radius in meters (standard size 5 ball: 11 cm radius)

    def reset(self, position=None):
        """
        Reset ball state to a given position (or center) and clear velocity and possession

        Args:
            position (tuple, list, np.ndarray, optional): Position to reset to
            Defaults to center [0.5, 0.5] if not provided
        """
        if position is None:
            position = [0.5, 0.5]
        self.position = np.array(position, dtype=np.float32)

        # Reset velocity to zero
        self.velocity.fill(0.0)

        # Clear possession
        self.owner = None

    def set_owner(self, player):
        """
        Set the player who currently possesses the ball
        Args:
            player (Player): The player object who owns the ball
        """
        self.owner = player
        if player is not None:
            self.velocity = np.array([0.0, 0.0], dtype=np.float32)  # Stop ball when possessed

    # Friction model rationale:
    #
    # In this simulation, friction is applied once per frame to the ball's velocity.
    # Supposing that the simulation runs at 24 frames per second, even moderate per-frame
    # friction values (e.g., 0.01 or 1%) would result in a rapid loss of speed over
    # just a few seconds. For example:
    #
    #   - A friction_factor of 0.01 → 99% speed retained per frame
    #   - After 1 second (24 frames): (0.99)^24 ≈ 0.79 → 21% velocity lost
    #   - After 2 seconds: (0.99)^48 ≈ 0.63 → 37% velocity lost
    #
    # While this may be acceptable for certain physics simulations, in football
    # the ball typically travels 10–30 meters in a pass and needs to maintain
    # momentum long enough to realistically simulate this behavior.
    #
    # For this reason, we use a very small friction factor (e.g., 0.001):
    #   - (0.999)^24 ≈ 0.976 → only 2.4% lost after 1 second
    #   - This provides a much smoother and more realistic deceleration
    #     over multiple seconds of simulation.

    def apply_friction(self, friction=0.0015):
        """
        Apply friction to the ball's velocity, reducing its speed slightly
        """
        self.velocity *= (1 - friction)

    def set_position(self, pos):
        """
        Set the ball's position manually
        """
        self.position = np.array(pos, dtype=np.float32)

    def get_position(self):
        """
        Get the current normalized position of the ball
        """
        return self.position

    def set_velocity(self, vel):
        """
        Set the ball's velocity vector manually
        """
        self.velocity = np.array(vel, dtype=np.float32)

    def get_velocity(self):
        """
        Get the current velocity vector of the ball
        """
        return self.velocity

    def get_owner(self):
        """
        Get the ID of the player currently possessing the ball
        """
        return self.owner

    def update(self, time_step):
        """
        Update the ball's position based on its velocity and apply friction to reduce speed
        """
        # Move ball according to velocity scaled by time step
        self.position += self.velocity * time_step

        # Apply friction to the velocity
        self.apply_friction()

        # Stop if velocity is very small 
        # This prevents endless sliding due to friction never reaching zero and avoid numerical instability
        if np.linalg.norm(self.velocity) < 1e-3:
            self.velocity.fill(0.0)

        # Ensure ball position remains within normalized bounds [0,1]
        self.position = np.clip(self.position, 0.0, 1.0)

        return self.velocity

    def is_out_of_bounds(self):
        """
        Check if the ball has left the normalized pitch area
        """
        x, y = self.position
        return not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)

    def copy(self):
        """
        Create a deep copy of this Ball instance, including position, velocity, and owner
        """
        new_ball = Ball()
        new_ball.position = self.position.copy()
        new_ball.velocity = self.velocity.copy()
        new_ball.owner = self.owner
        return new_ball