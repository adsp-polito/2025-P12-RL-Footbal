import numpy as np
from helpers.ballPhysics import apply_friction

class Ball:
    """
    This class represents the ball behavior in the environment.
    It includes its position and velocity on the pitch, and updates
    its motion at each time step.

    Attributes:
        - position (np.ndarray): Current position of the ball [x, y]
        - velocity (np.ndarray): Current velocity vector [vx, vy]
        - just_lost_possession (bool): Indicates if the ball was just lost by a player
    """

    def __init__(self):
        # Initial position (center of the field)
        self.init_position = np.array([0.5, 0.5], dtype=np.float32)

        # Current position and velocity
        self.position = self.init_position.copy()
        self.velocity = np.zeros(2, dtype=np.float32)

        # ID of player currently in possession of the ball
        self.owner_id = None  

        # Maximum ball speed in meters per second
        # 30 m/s ≈ 108 km/h → realistic for powerful shots
        # Real-world records: Ronny Heberson (record) ≈ 61 m/s
        # Put it to 50 m/s ≈ 180 km/h
        self.max_speed = 50

        # Indicates if the ball was just lost by a player
        self.just_lost_possession = False

    def reset(self):
        """Reset the ball to the initial state"""
        self.position = np.array([0.5, 0.5], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.owner_id = None
        self.just_lost_possession = False

    def update_position(self, players=None, fps = 24, field_width = 120):
        """
        Updates the ball's position based on its current velocity and applies friction.
        If the ball is owned, it follows the player. Otherwise, it moves according to its velocity.
        """

        if self.owner_id is not None:
            # Ball is in possession: The ball moves with the player's velocity
            owner = players[self.owner_id]
            self.velocity = owner.velocity

             # Update position based on player's velocity
            direction = self.velocity / np.linalg.norm(self.velocity) if np.linalg.norm(self.velocity) > 0 else np.array([0, 0])
            offset = direction * 0.005  # Slight offset to position the ball slightly ahead of the player
            self.position = owner.get_position() + offset

        else:

            # Ball moves freely
            self.position += (self.velocity / fps) / field_width  # Move by actual meters per frame

            # Apply friction
            if np.linalg.norm(self.velocity) > 0:
                self.velocity = apply_friction(self.velocity)

            # Stop the ball if it's too slow
            if np.linalg.norm(self.velocity) < 0.05:
                self.velocity = np.zeros(2)

        if self.owner_id is not None:
            self.just_lost_possession = True
        else:
            self.just_lost_possession = False

    def get_position(self):
        """Returns the current position of the ball"""
        return self.position

    def get_velocity(self):
        """Returns the current velocity of the ball"""
        return self.velocity

    def is_free(self):
        """Returns True if the ball is not possessed by any player"""
        return self.owner_id is None

    def is_out_of_bounds(self):
        """Checks whether the ball has left the normalized field boundaries"""
        x, y = self.position
        return x < 0 or x > 1 or y < 0 or y > 1

        # Future: handle ball bouncing on field boundaries or goalposts