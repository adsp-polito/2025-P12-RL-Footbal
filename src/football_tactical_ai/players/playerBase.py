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
        **_: Any,
    ) -> None:
        
        # fixed caps for this player
        self.max_speed      = float(max_speed)
        self.max_power      = float(max_power)
        self.max_fov_angle  = float(max_fov_angle)
        self.max_fov_range  = float(max_fov_range)

        # default values for the attributes that are shared by all players
        self.position: List[float] = [0.0, 0.0]   # normalised [0, 1]
        self.last_action_direction: np.ndarray = np.array([1.0, 0.0])  # last action direction

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
   
