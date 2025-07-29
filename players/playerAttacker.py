import sys
import os

# Add project root to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from helpers.helperFunctions import denormalize

class PlayerAttacker:
    def __init__(self,
             shooting=0.5,
             passing=0.5,
             dribbling=0.5,
             speed=0.5,
             precision=0.7,
             fov_angle=0.5,
             fov_range=0.5,
             max_speed=10.0,
             max_power=15.0,
             max_fov_angle=180.0,
             max_fov_range=90.0):
        """
        Initialize an attacking player with technical, visual, and physical attributes.

        Tunable Parameters (normalized in [0, 1]):
            shooting (float): Shooting skill, overall shot quality and effectiveness.
            passing (float): Passing skill.
            dribbling (float): Dribbling skill.
            speed (float): Current speed factor, multiplier for max_speed.
            precision (float): Shot accuracy, how accurate the shot direction is.
            fov_angle (float): Field of view angle factor [0, 1], scaled by max_fov_angle.
            fov_range (float): Field of view range factor [0, 1], scaled by max_fov_range=90.0):.

        Physical Parameters (real-world max values):
            max_speed (float): Maximum running speed in meters per second.
            max_power (float): Maximum shot power in meters per second (ball velocity).
            max_fov_angle (float): Maximum field of view angle in degrees (e.g., 180).
            max_fov_range=90.0): (float): Maximum vision distance in meters.
        """

        # Technical skills
        self.shooting = shooting
        self.passing = passing
        self.dribbling = dribbling
        self.precision = precision

        # Vision parameters (normalized + physical bounds)
        self.fov_angle = fov_angle
        self.fov_range = fov_range
        self.max_fov_angle = max_fov_angle
        self.max_fov_range = max_fov_range

        # Speed and power
        self.speed = speed
        self.max_speed = max_speed
        self.max_power = max_power

        # Position in normalized coordinates [0, 1]
        self.position = [0.0, 0.0]

        # Last action direction for reward logic
        # This will be used to determine if the player attempted a movement in the wrong direction
        self.last_action_direction = np.array([0.0, 0.0])

    def reset_position(self, position):
        """
        Reset player position using a list of two normalized coordinates.

        Args:
            position (list): [x, y] in normalized coordinates.
        """
        if isinstance(position, list) and len(position) == 2:
            self.position = position
        else:
            raise ValueError("Position must be a list of two elements [x, y].")


    def is_direction_visible(self, desired_direction):
        """
        Check whether the desired direction is within the player's current field of view.

        Args:
            desired_direction (np.array): 2D vector (not necessarily normalized)

        Returns:
            bool: True if within field of view, else False
        """
        # Normalize input direction
        norm = np.linalg.norm(desired_direction)
        if norm == 0:
            return False
        dir_vec = desired_direction / norm

        # Use player's current facing direction (fallback to forward if None)
        forward = getattr(self, "last_action_direction", np.array([1.0, 0.0]))
        forward_norm = np.linalg.norm(forward)
        if forward_norm == 0:
            forward = np.array([1.0, 0.0])
        else:
            forward = forward / forward_norm

        # Compute angle between forward and desired direction
        dot = np.clip(np.dot(forward, dir_vec), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))

        # Compute maximum allowed angle
        max_angle = self.fov_angle * self.max_fov_angle

        return angle_deg <= (max_angle / 2)

    def move(self, delta_position):
        """
        Move the player by a delta in normalized coordinates.
        Used for manual updates such as repositioning or debugging.
        No physics or speed constraints applied here.
        """
        self.position[0] += delta_position[0]
        self.position[1] += delta_position[1]
        # Clamp position inside pitch boundaries
        self.position[0] = max(0.0, min(1.0, self.position[0]))
        self.position[1] = max(0.0, min(1.0, self.position[1]))

    def move_with_action(self, action, time_per_step, x_range, y_range):
        """
        Move player based on continuous action input and visual constraints.

        The player will only move in the intended direction if it lies within
        their field of view (FOV), defined by the player's vision angle.

        Args:
            action (np.array): Array with x and y movement values in [-1, 1].
            time_per_step (float): Duration of simulation step (seconds).
            x_range (float): Width of pitch in meters.
            y_range (float): Height of pitch in meters.
        """

        # Block movement if direction is outside the player's field of view
        # If action is zero vector, allow it
        if np.linalg.norm(action) == 0:
            return # No movement if action is zero vector


        # Check if movement is allowed
        if not self.is_direction_visible(action):
            return


        # Normalize and save last direction
        direction = np.array(action)
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.last_action_direction = direction / norm
        else:
            self.last_action_direction = np.array([1.0, 0.0])

        # Scale the normalized action to movement deltas based on speed and field dimensions
        dx = action[0] * self.speed * self.max_speed * time_per_step / x_range
        dy = action[1] * self.speed * self.max_speed * time_per_step / y_range

        # Apply movement
        self.move([dx, dy])


    def shoot(self, desired_direction, desired_power):
        """
        Attempt to shoot in a given direction, only if it's within the field of view.

        Args:
            desired_direction (np.array): 2D vector representing the shot direction.
            desired_power (float): Desired shot power in range [0, 1].

        Returns:
            tuple: (shot_quality, shot_direction, shot_power)
                - shot_quality (float): Estimated probability of a successful shot (0 to 1).
                - shot_direction (np.array): Actual 2D unit vector of the shot after applying noise.
                - shot_power (float): Final shot power (velocity) in meters per second.
        """

        # Check if the desired direction is within the player's field of view
        # If not, the shot is invalid and returns zeroed output
        if not self.is_direction_visible(desired_direction):
            return 0.0, np.array([0.0, 0.0]), 0.0
        
        print("SHOOOOOTING!")

        # Normalize the desired direction to a unit vector
        norm = np.linalg.norm(desired_direction)
        if norm > 0:
            desired_dir_norm = desired_direction / norm
        else:
            # Default to straight forward if direction is zero
            desired_dir_norm = np.array([1.0, 0.0])

        # Add noise to the shot direction based on the player's precision
        # The less precise, the more deviation from the intended angle
        max_noise_radians = 0.1  # Maximum angular noise (~5.7 degrees)
        noise_strength = (1 - self.precision) * max_noise_radians
        angle_noise = np.random.uniform(-noise_strength, noise_strength)
        cos_a, sin_a = np.cos(angle_noise), np.sin(angle_noise)

        # Apply rotation to simulate directional inaccuracy
        shot_direction = np.array([
            cos_a * desired_dir_norm[0] - sin_a * desired_dir_norm[1],
            sin_a * desired_dir_norm[0] + cos_a * desired_dir_norm[1]
        ])

        # Clamp power input to [0, 1]
        desired_power = np.clip(desired_power, 0.0, 1.0)

        # Scale the shot power based on player shooting skill and max power
        # Add a minimum base power to avoid zero-velocity shots
        shot_power = self.max_power * (0.5 + 0.5 * self.shooting * desired_power)

        # Estimate shot quality based on player position and shooting skill
        x_pos, y_pos = self.position
        x_factor = 0.2 + 0.8 * x_pos         # Better angle as player moves closer to goal (x direction)
        y_dist = abs(y_pos - 0.5)            # Distance from vertical center line
        y_factor = max(0.5, 1 - 2 * y_dist)  # Penalize wide-angle shots near sidelines

        # Final shot quality combines skill and positional advantage
        shot_quality = x_factor * y_factor * self.shooting

        return shot_quality, shot_direction, shot_power


    def get_position(self):
        """Return current position as normalized (x, y) tuple."""
        return tuple(self.position)

    def get_parameters(self):
        """Return technical attributes for logging or debugging."""
        return {
            "shooting": self.shooting,
            "precision": self.precision,
            "passing": self.passing,
            "dribbling": self.dribbling
        }

    def get_role(self):
        """Return the player role name as string."""
        return "ATT"

    def copy(self):
        """
        Create a deep copy of the player instance.
        Useful for rendering snapshots without side effects.
        """
        # Create a new instance with the same parameters
        new_player = PlayerAttacker(
            shooting=self.shooting,
            passing=self.passing,
            dribbling=self.dribbling,
            speed=self.speed,
            precision=self.precision,
            fov_angle=self.fov_angle,
            fov_range=self.fov_range,
            max_speed=self.max_speed,
            max_power=self.max_power,
            max_fov_angle=self.max_fov_angle,
            max_fov_range=self.max_fov_range
        )
        # Copy the position and last action direction
        new_player.position = self.position.copy()
        new_player.last_action_direction = self.last_action_direction.copy()

        new_player.position = self.position.copy()
        return new_player