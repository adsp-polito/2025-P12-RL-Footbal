import numpy as np
from typing import Any
from football_tactical_ai.players.playerBase import BasePlayer

class PlayerAttacker(BasePlayer):
    def __init__(self,
             shooting: float = 0.5,
             passing: float = 0.5,
             dribbling: float = 0.5,
             speed: float = 0.5,
             precision: float = 0.7,
             fov_angle: float = 0.5,
             fov_range: float = 0.5,
             role: str = "ATT",
             **kwargs: Any):
        
        """
        Initialize an attacking player with technical and physical attributes.
        Args:
            shooting (float): Shooting skill [0, 1]
            passing (float): Passing skill [0, 1]
            dribbling (float): Dribbling skill [0, 1]
            speed (float): Current speed factor [0, 1]
            precision (float): Precision of shots [0, 1]
            fov_angle (float): Field of view angle as a fraction of max angle [0, 1].
            fov_range (float): Field of view range as a fraction of max range [0, 1].
            role (str): Player role, default is "ATT".
        Physical maxima are inherited from BasePlayer, but can be overridden.
        """
        super().__init__(agent_id=None, team=None, role=role, **kwargs)

        # Technical skills
        self.shooting  = shooting
        self.passing   = passing
        self.dribbling = dribbling
        self.speed     = speed
        self.precision = precision

        # Vision parameters (normalized + physical bounds)
        self.fov_angle = fov_angle
        self.fov_range = fov_range

        # Last action direction for reward logic
        # This will be used to determine if the player attempted a movement in the wrong direction
        self.last_action_direction = np.array([1.0, 0.0])

    def shoot(self, desired_direction, desired_power, enable_fov=True):
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
                - enable_fov (bool): Whether to check if the shot direction is within the player's field of view.
        """

        # Check if the desired direction is within the player's field of view
        # If not, the shot is invalid and returns zeroed output
        if not self.is_direction_visible(desired_direction) and enable_fov:
            return 0.0, np.array([0.0, 0.0]), 0.0
        
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

    def get_parameters(self):
        """Return technical attributes for logging or debugging."""
        return {
            "role": self.get_role(),
            "speed": self.speed,
            "fov_angle": self.fov_angle,
            "fov_range": self.fov_range,
            "shooting": self.shooting,
            "precision": self.precision,
            "passing": self.passing,
            "dribbling": self.dribbling
        }

    def get_role(self):
        """Return the player role name as string."""
        return "ATT"
    
    def execute_action(self, 
                       action: np.ndarray, 
                       time_step: float, 
                       x_range: float, 
                       y_range: float, 
                       ball_owner: str = None) -> dict:
        """
        Executes a continuous action for an attacking player. The attacker can move 
        and optionally shoot, depending on the action vector.

        Args:
            action (np.ndarray): Action vector of the form:
                [dx, dy, shoot_flag, shot_power, shot_dir_x, shot_dir_y]
            time_step (float): Duration of simulation step in seconds (e.g. 1 / FPS)
            x_range (float): Field width in meters
            y_range (float): Field height in meters
            ball_owner (str): ID of the player who currently owns the ball.
        Returns:
            dict: Contextual information about the action taken.
        """
                
        # Extract movement vector
        dx, dy = action[0], action[1]
        direction = np.array([dx, dy])
        norm = np.linalg.norm(direction)

        # Default visibility check result
        fov_visible = None

        # If agent is moving, update direction and check FOV
        if norm > 1e-6:
            direction /= norm  # Normalize
            self.last_action_direction = direction
            fov_visible = self.is_direction_visible(direction)
        else:
            direction = np.array([0.0, 0.0])
            fov_visible = None  # No movement

        # Movement part (dx, dy)
        super().execute_action(action, time_step, x_range, y_range)

        # Shooting logic
        if len(action) >= 6 and action[2] > 0.5:

            # Extract shot parameters
            desired_power = np.clip(action[3], 0.0, 1.0)
            desired_direction = np.array(action[4:6])

            # FOV visibility check
            is_visible = self.is_direction_visible(desired_direction)
            
            # Perform the shot
            shot_quality, shot_direction, shot_power = self.shoot(
                desired_direction=desired_direction,
                desired_power=desired_power,
                enable_fov=True
            )

            # Compute shot alignment with goal
            goal_direction = np.array([1.0, 0.5]) - np.array(self.position)
            goal_direction /= np.linalg.norm(goal_direction) if np.linalg.norm(goal_direction) > 1e-6 else 1
            alignment = np.dot(shot_direction, goal_direction)
            alignment = (alignment + 1) / 2.0  # [0, 1]

            # Compute positional quality
            x_pos, y_pos = self.position
            x_factor = 0.2 + 0.8 * x_pos
            y_dist = abs(y_pos - 0.5)
            y_factor = max(0.5, 1 - 2 * y_dist)
            positional_quality = x_factor * y_factor

            # Return full context for reward system
            return {
                "shot_attempted": True,
                "shot_quality": shot_quality,
                "shot_power": shot_power,
                "shot_direction": shot_direction,
                "fov_visible": is_visible,
                "invalid_shot_direction": not is_visible,
                "not_owner_shot_attempt": ball_owner != self.agent_id,
                "shot_alignment": alignment,
                "start_shot_bonus": True,
                "shot_positional_quality": positional_quality,
            }

        return {
            "fov_visible": fov_visible,
            "shot_attempted": False,
        }


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
            fov_range=self.fov_range
        )

        # Copy the inherited attributes from BasePlayer
        new_player.max_speed     = self.max_speed
        new_player.max_power     = self.max_power
        new_player.max_fov_angle = self.max_fov_angle
        new_player.max_fov_range = self.max_fov_range

        # Copy the position and last action direction
        new_player.position = self.position.copy()
        new_player.last_action_direction = self.last_action_direction.copy()

        return new_player