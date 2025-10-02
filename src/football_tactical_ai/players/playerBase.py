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
        self.role = kwargs.get("role", "GENERIC")     # Example: "ATT", "DEF", "GK" or "CF", "LCB", etc.

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
            raise ValueError("Position must be a list of two elements [x, y]")        

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
        Get the current position of the player as a tuple of normalized coordinates
        """
        return tuple(self.position)
    
    def set_position(self, position: Tuple[float, float]) -> None:
        """
        Set the player's position using a tuple of normalized coordinates
        Args:
            position (tuple): (x, y) in normalized coordinates
        """
        if isinstance(position, (list, tuple)) and len(position) == 2:
            self.position[0] = float(position[0])
            self.position[1] = float(position[1])
            # Clamp position inside pitch boundaries
            self.position[0] = max(0.0, min(1.0, self.position[0]))
            self.position[1] = max(0.0, min(1.0, self.position[1]))
        else:
            raise ValueError("Position must be a tuple of two elements (x, y)")
    
    def set_orientation(self, angle_radians: float) -> None:
        """
        Set the player's facing direction using an angle in radians.
        0 radians means facing right (positive x direction), and angles increase counter-clockwise.

        Args:
            angle_radians (float): Angle in radians
        """
        self.last_action_direction = np.array([np.cos(angle_radians), np.sin(angle_radians)])
        norm = np.linalg.norm(self.last_action_direction)
        if norm > 0:
            self.last_action_direction /= norm
        else:
            self.last_action_direction = np.array([1.0, 0.0])  # default facing right if zero vector

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
        Move the player based on a movement vector extracted from the full action array.

        - The input here is only [dx, dy] ∈ [-1, 1].
        - Handles FOV constraints, normalizes direction, and converts displacement
          into meters per step → then back to normalized pitch coordinates.

        Args:
            action (np.array): Array [dx, dy] with movement direction.
            time_per_step (float): Simulation step duration (s).
            x_range (float): Pitch width in meters.
            y_range (float): Pitch height in meters.
            enable_fov (bool): If True, blocks movement outside the player's FOV.
        """


        # If the action is zero (idle), do not move
        if np.linalg.norm(action) < 1e-6:
            return

        # Check if the movement direction is allowed (inside FOV)
        if enable_fov and not self.is_direction_visible(action):
            return

        # Normalize direction and update last facing vector
        direction = np.array(action, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
            self.last_action_direction = direction
        else:
            direction = np.array([0.0, 0.0])
            self.last_action_direction = np.array([1.0, 0.0])  # default facing right

        # Compute how many meters the player should move in this step
        meters_per_step = self.max_speed * self.speed * time_per_step

        # Convert displacement into normalized coordinates [0,1]
        dx = (direction[0] * meters_per_step) / x_range
        dy = (direction[1] * meters_per_step) / y_range

        # Apply the movement update
        self.move([dx, dy])


    # Execute the action (to be overridden by subclasses if needed)
    # This method is specifically used in multi-agent environments
    def execute_action(self, action: np.ndarray, time_step: float, x_range: float, y_range: float):
        """
        Executes a continuous action vector for the player.
        Only movement is handled here; child classes implement pass/shoot/etc.

        Args:
            action (np.ndarray): Flat action vector of shape (7,)
                - action[0], action[1]: movement deltas (dx, dy) in [-1, 1]
                - action[2], action[3]: binary flags (e.g. pass/tackle, shoot/dive)
                - action[4]: power (0-1)
                - action[5], action[6]: direction vector for pass/shot in [-1, 1]
            time_step (float): Duration of simulation step in seconds (e.g. 1/FPS)
            x_range (float): Field width in meters
            y_range (float): Field height in meters
        """

        # Extract movement from the first two values of the action vector
        dx, dy = action[0], action[1]
        move_vec = np.array([dx, dy], dtype=np.float32)

        # Apply movement logic (with FOV constraint)
        self.move_with_action(
            action=move_vec,
            time_per_step=time_step,
            x_range=x_range,
            y_range=y_range,
            enable_fov=True
        )

        # Store the inferred action type (idle, move, pass, shoot, etc.)
        self.current_action = self._infer_action_type(action)

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
    
    def pass_ball(self, desired_direction, desired_power, enable_fov=False):
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
        pass_power = self.max_power * (0.2 + 0.7 * self.passing * desired_power)

        # Pass quality (simplified: skill + closeness to center)
        x_pos, y_pos = self.position
        x_factor = 0.2 + 0.8 * x_pos         # Better angle as player moves closer to goal (x direction)
        y_dist = abs(y_pos - 0.5)
        y_factor = max(0.5, 1 - 2 * y_dist)

        pass_quality = x_factor * y_factor * self.passing

        return pass_quality, pass_direction, pass_power


    def _infer_action_type(self, action: np.ndarray) -> str:
        """
        Infer the action type from the flat action vector.
        Action format = [dx, dy, f0, f1, power, dir_x, dir_y]
        """

        # Pad action if it's shorter (e.g. only [dx, dy])
        if len(action) < 7:
            padded = np.zeros(7, dtype=np.float32)
            padded[:len(action)] = action
            action = padded

        dx, dy, f0, f1, power, dir_x, dir_y = action

        # Movement has priority
        if np.linalg.norm([dx, dy]) > 1e-6:
            return "move"

        # Threshold flags
        f0, f1 = int(f0 > 0.5), int(f1 > 0.5)

        if self.role in {"ATT", "CF", "LW", "RW", "LCF", "RCF", "SS"}:
            if f0: return "pass"
            if f1: return "shoot"

        elif self.role in {"DEF", "LCB", "RCB", "CB"}:
            if f0: return "tackle"
            if f1: return "shoot"

        elif self.role == "GK":
            if f0: return "dive"
            if f1: return "shoot"

        return "idle"


    def get_current_action_code(self) -> int:
        """
        Map the current action string to a discrete integer code.

        Returns:
            int: Encoded action for observations (0 = idle, 1 = move, etc.).
        """
        mapping = {
            "idle": 0,
            "move": 1,
            "pass": 2,
            "shoot": 3,
            "tackle": 4,
            "dive": 5,
        }
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

