import numpy as np

class Ball:
    """
    The Ball class represents the football used in the environment
    It includes its position and velocity on the pitch, and updates
    its motion at each time step.

    Attributes:
        - position (np.ndarray): Current position of the ball [x, y]
        - velocity (np.ndarray): Current velocity vector [vx, vy]
    """

    def __init__(self):

        # Initial position (center of the field)
        self.init_position = np.array([0.5, 0.5], dtype=np.float32)

        # Current position and velocity
        self.position = self.init_position.copy()
        self.velocity = np.zeros(2, dtype=np.float32)

        # ID of player currently in possession of the ball
        self.owner_id = None  

    def reset(self):
        """Reset the ball to the initial state"""
        
        # Reset position and velocity
        self.position = self.init_position.copy()
        self.velocity = np.zeros(2, dtype=np.float32)

        # Reset the ball owner
        self.owner_id = None

    def update_position(self):
        """
        Updates the ball's position based on its current velocity.

        This method is used when the ball is free (not in possession).
        It moves the ball linearly on the field using its current velocity.
        Future implementations may include friction, collisions, or other dynamics.
        """
        self.position += self.velocity

    def update_owner(self, players):
        """
        Updates the ball's position to follow the owning player with a slight offset
        in the direction of the player's velocity (movement vector).
        """
        owner = players[self.owner_id]
        player_pos = owner.get_position()
        player_vel = owner.velocity

        # Normalize the velocity vector to get direction
        norm = np.linalg.norm(player_vel)
        if norm > 0:
            direction = player_vel / norm
            offset = direction * 0.01  # Offset scaled to field size (1% of pitch width)
        else:
            offset = np.zeros(2)

        self.position = player_pos + offset

    def get_position(self):
        """Returns the current position of the ball"""
        return self.position