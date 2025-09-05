from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Dict, Any


class BasePlayer(ABC):
    """
    Abstract parent for every football role.

    It exposes:
        • fixed physical caps for the *instance*
        • generic state (position)
        • helper methods shared by Attacker / Defender / Goalkeeper

    Role-specific skills (shooting, tackling, reflexes, …) live only
    in the child classes.
    """

    # Physical maxima (real-world units)
    MAX_SPEED: float      = 10.0   # m/s
    MAX_POWER: float      = 15.0   # m/s (ball velocity)
    MAX_FOV_ANGLE: float  = 180.0  # degrees
    MAX_FOV_RANGE: float  = 90.0   # metres

    def __init__(
        self,
        *,
        max_speed: float      = MAX_SPEED,
        max_power: float      = MAX_POWER,
        max_fov_angle: float  = MAX_FOV_ANGLE,
        max_fov_range: float  = MAX_FOV_RANGE,
        **kwargs: Any,
    ) -> None:
        
        # fixed caps for this player
        self.max_speed      = float(max_speed)
        self.max_power      = float(max_power)
        self.max_fov_angle  = float(max_fov_angle)
        self.max_fov_range  = float(max_fov_range)

        # default values for the attributes that are shared by all players
        self.position: List[float] = [0.0, 0.0]   # normalized [0, 1]
        self.last_action_direction: np.ndarray = np.array([1.0, 0.0])  # last action direction

        # attributes for the player instance (for multi agent scenarios)
        self.agent_id = kwargs.get("agent_id", None)  # Example: "att_0"
        self.team = kwargs.get("team", None)          # Example: "A" or "B"
        self.role = kwargs.get("role", "GENERIC")     # Example: "ATT", "DEF", "GK"

        # Current action for observation purposes
        self.current_action = "idle"                  # e.g. "move", "shoot", "tackle", "dive", etc.

    # SHARED METHODS
    def reset_position(self, position: List[float]) -> None:
        """
        Reset player position using a list of two normalized coordinates.

        Args:
            position (list): [x, y] in normalized coordinates.
        """
        if isinstance(position, list) and len(position) == 2:
            self.position[:] = position
        else:
            raise ValueError("Position must be a list of two elements [x, y].")        

    def move(self, delta_position):
        """
        Move the player by a delta in normalized coordinates.
        Used for manual updates such as repositioning or debugging.
        No physics or speed constraints applied here.
        """

        if len(delta_position) != 2:
            raise ValueError("delta_position must contain exactly two values [dx, dy].")

        self.position[0] += delta_position[0]
        self.position[1] += delta_position[1]

        # Clamp position inside pitch boundaries
        self.position[0] = max(0.0, min(1.0, self.position[0]))
        self.position[1] = max(0.0, min(1.0, self.position[1]))

    def get_position(self) -> Tuple[float, float]:
        """
        Get the current position of the player as a tuple of normalized coordinates.
        """
        return tuple(self.position)
    
    def get_agent_id(self) -> str:
        """
        Get the unique identifier for this player instance.
        """
        return self.agent_id
    
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

    def move_with_action(self, action, time_per_step, x_range, y_range, enable_fov=True):
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
        if not self.is_direction_visible(action) and enable_fov == True:
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

    # Execute the action (to be overridden by subclasses)
    # This method is specifically used in multi-agent environments
    def execute_action(self, action: np.ndarray, time_step: float, x_range: float, y_range: float):
        """
        Executes a continuous action vector for the player. This method is called once per frame by the environment.

        The base class handles only movement. Role-specific subclasses (Attacker, Defender, Goalkeeper)
        should override this method to handle actions like shooting, tackling, diving, etc.

        Args:
            action (np.ndarray): Continuous action vector, typically [dx, dy, ...]
            time_step (float): Duration of simulation step in seconds (e.g. 1 / FPS)
            x_range (float): Field width in meters
            y_range (float): Field height in meters
        """

        # Basic movement: interpret dx, dy
        self.move_with_action(
            action=action[:2],
            time_per_step=time_step,
            x_range=x_range,
            y_range=y_range,
            enable_fov=True
        )

        self.current_action = self._infer_action_type(action)

        # This base implementation does not include a ny extra action
        # (e.g. shooting, tackling) – these must be handled by child classes
        return None
    
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
    
    def passage(self, desired_direction, desired_power, enable_fov=True):
        """
        Attempt to pass in a given direction, only if it's within the field of view.

        Args:
            desired_direction (np.array): 2D vector representing the pass direction.
            desired_power (float): Desired pass power in range [0, 1].
            enable_fov (bool): Whether to check if the pass direction is within FOV.

        Returns:
            tuple: (pass_quality, pass_direction, pass_power)
        """

        # Check if the desired direction is within the player's field of view
        # If not, the pass is invalid and returns zeroed output
        if not self.is_direction_visible(desired_direction) and enable_fov:
            return 0.0, np.array([0.0, 0.0]), 0.0

        # Normalize direction
        norm = np.linalg.norm(desired_direction)
        if norm > 0:
            dir_norm = desired_direction / norm
        else:
            dir_norm = np.array([1.0, 0.0])

        # Add noise to the shot direction based on the player's precision
        # The less precise, the more deviation from the intended angle
        max_noise_radians = 0.05  # smaller than shot (~2.9 degrees)
        noise_strength = (1 - self.precision) * max_noise_radians
        angle_noise = np.random.uniform(-noise_strength, noise_strength)
        cos_a, sin_a = np.cos(angle_noise), np.sin(angle_noise)

        # Apply rotation to simulate directional inaccuracy
        pass_direction = np.array([
            cos_a * dir_norm[0] - sin_a * dir_norm[1],
            sin_a * dir_norm[0] + cos_a * dir_norm[1]
        ])

        # Clamp power
        desired_power = np.clip(desired_power, 0.0, 1.0)

        # Scale to ball velocity (lower than shot)
        # based on passing skill
        pass_power = self.max_power * (0.4 + 0.5 * self.passing * desired_power)

        # Pass quality (simplified: skill + closeness to center)
        x_pos, y_pos = self.position
        x_factor = 0.2 + 0.8 * x_pos         # Better angle as player moves closer to goal (x direction)
        y_dist = abs(y_pos - 0.5)
        y_factor = max(0.5, 1 - 2 * y_dist)

        pass_quality = x_factor * y_factor * self.passing

        return pass_quality, pass_direction, pass_power

    
    def _infer_action_type(self, action: np.ndarray) -> str:
        """
        Infer action type from the raw action vector.
        Returns one of: "idle", "move", "pass", "shoot", "tackle", "dive".
        """
        role = self.role

        # Movement
        dx, dy = action[0], action[1]
        if np.linalg.norm([dx, dy]) > 1e-6:
            return "move"

        if role == "ATT":
            if len(action) >= 3 and action[2] > 0.5:  # pass_flag
                return "pass"
            if len(action) >= 4 and action[3] > 0.5:  # shoot_flag
                return "shoot"

        elif role == "DEF":
            if len(action) >= 3 and action[2] > 0.5:  # tackle_flag
                return "tackle"
            if len(action) >= 4 and action[3] > 0.5:  # shoot_flag
                return "shoot"

        elif role == "GK":
            if len(action) >= 3 and (action[2] > 0.5 or action[3] > 0.5):  # dive flags
                return "dive"
            if len(action) >= 5 and action[4] > 0.5:  # shoot_flag
                return "shoot"

        return "idle"


    def get_current_action_code(self):
        """
        Return the integer code for the current action.
        """
        mapping = {"idle": 0, "move": 1, "pass": 2, "shoot": 3, "tackle": 4, "dive": 5}
        return mapping.get(self.current_action, 0)


    # ABSTRACT METHODS

    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """Return the role-specific skill dictionary."""
        ...

    @abstractmethod
    def get_role(self) -> str:
        """Return short tag like 'ATT', 'DEF', 'GK'."""
        ...

    @abstractmethod
    def copy(self):
        """
        Return a deep copy of the player instance.
        Used for cloning in simulations or training.
        """
        ...

