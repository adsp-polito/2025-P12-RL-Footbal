import numpy as np

class Ball:
    """
    Class representing a football (soccer ball) in a normalized 2D pitch coordinate system.

    Attributes:
        position (np.ndarray): Current normalized position on the pitch, with x and y values in [0,1].
        velocity (np.ndarray): Current velocity vector, normalized per second.
        owner_id (int or None): ID of the player currently possessing the ball, or None if free.
    """

    def __init__(self, position=None):
        """
        Initialize the ball object.

        Args:
            position (tuple, list, np.ndarray, optional): Initial normalized position [x, y].
                Defaults to the center of the pitch at [0.5, 0.5] if not specified.
        """
        # Set initial position; default is pitch center
        if position is None:
            position = [0.5, 0.5]
        self.position = np.array(position, dtype=np.float32)

        # Initialize velocity vector as zero (ball stationary)
        self.velocity = np.zeros(2, dtype=np.float32)

        # Initialize owner as None (ball not possessed)
        self.owner = None

    def reset(self, position=None):
        """
        Reset ball state to a given position (or center) and clear velocity and possession.

        Args:
            position (tuple, list, np.ndarray, optional): Position to reset to.
                Defaults to center [0.5, 0.5] if not provided.
        """
        if position is None:
            position = [0.5, 0.5]
        self.position = np.array(position, dtype=np.float32)

        # Reset velocity to zero
        self.velocity.fill(0.0)

        # Clear possession
        self.owner_id = None

    
    def set_owner(self, player):
        """
        Set the player who currently possesses the ball.
        Args:
            player (Player): The player object who owns the ball.
        """
        self.owner = player
        self.velocity.fill(0)

    def release(self, velocity):
        """
        Release the ball from possession, setting its velocity.
        Args:
            velocity (tuple, list, np.ndarray): Velocity vector to apply when releasing the ball.
        """
        self.owner = None
        self.velocity = np.array(velocity, dtype=np.float32)

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
        Apply friction to the ball's velocity.

        Args:
            friction (float): Friction coefficient to apply.
        """
        self.velocity *= (1 - friction)

    def set_position(self, pos):
        """
        Set the ball's position manually.

        Args:
            pos (tuple, list, np.ndarray): New normalized position [x, y].
        """
        self.position = np.array(pos, dtype=np.float32)

    def get_position(self):
        """
        Get the current normalized position of the ball.

        Returns:
            np.ndarray: Position vector [x, y].
        """
        return self.position

    def set_velocity(self, vel):
        """
        Set the ball's velocity vector manually.

        Args:
            vel (tuple, list, np.ndarray): Velocity vector in normalized units per second.
        """
        self.velocity = np.array(vel, dtype=np.float32)

    def get_velocity(self):
        """
        Get the current velocity vector of the ball.

        Returns:
            np.ndarray: Velocity vector [vx, vy].
        """
        return self.velocity

    def update(self, time_step):
        """
        Update the ball's position based on its velocity and apply friction to reduce speed.

        Args:
            time_step (float): Duration of the simulation step in seconds.
            friction (float): Fractional decay of velocity per time step (default is 0.0015).
        """
        # Move ball according to velocity scaled by time step
        self.position += self.velocity * time_step

        # Apply friction to the velocity
        self.apply_friction()

        # Ensure ball position remains within normalized bounds [0,1]
        self.position = np.clip(self.position, 0.0, 1.0)

        return self.velocity

    def is_out_of_bounds(self):
        """
        Check if the ball has left the normalized pitch area.

        Returns:
            bool: True if ball is out of bounds, False otherwise.
        """
        x, y = self.position
        return not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)

    def copy(self):
        """
        Create a deep copy of this Ball instance, including position, velocity, and owner.

        Returns:
            Ball: A new Ball object with identical state.
        """
        new_ball = Ball()
        new_ball.position = self.position.copy()
        new_ball.velocity = self.velocity.copy()
        new_ball.owner = self.owner.copy() if self.owner else None
        return new_ball
