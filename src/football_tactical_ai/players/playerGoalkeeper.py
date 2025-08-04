import numpy as np
from football_tactical_ai.players.playerBase import BasePlayer
from typing import Any

class PlayerGoalkeeper(BasePlayer):
    def __init__(self,
                 reflexes: float = 0.5,
                 reach: float = 0.5,
                 blocking: float = 0.5,
                 speed: float = 0.5,
                 fov_angle: float = 0.5,
                 fov_range: float = 0.5,
                 role: str = "GK",
                 **kwargs: Any):
        """
        Initialize a goalkeeper with goalkeeping-specific attributes.

        Args:
            reflexes (float): Speed of reaction [0, 1].
            reach (float): How far the keeper can dive [0, 1].
            blocking (float): Skill in stopping low/mid shots [0, 1].
            speed (float): Movement speed multiplier [0, 1].
            fov_angle (float): Vision angle [0, 1].
            fov_range (float): Vision range [0, 1].
        """
        super().__init__(agent_id=None, team=None, role=role, **kwargs)

        # Technical attributes
        self.reflexes  = reflexes
        self.reach     = reach
        self.blocking  = blocking
        self.fov_angle = fov_angle
        self.fov_range = fov_range
        self.speed     = speed

        # Position in normalized coordinates [0, 1]
        self.position = [0.0, 0.0]
        self.last_action_direction = np.array([1.0, 0.0])  # last action direction


    def dive(self, direction: str = "left"):
        """
        Simulates a dive attempt. Actual interception logic should be handled externally.

        Args:
            direction (str): "left" or "right"

        Returns:
            float: Effectiveness score of the dive [0, 1]
        """
        if direction not in {"left", "right"}:
            return 0.0
        return 0.5 * self.reflexes + 0.5 * self.reach

    def get_parameters(self):
        """Return technical attributes for logging or debugging."""
        return {
            "role": self.get_role(),
            "speed": self.speed,
            "fov_angle": self.fov_angle,
            "fov_range": self.fov_range,
            "reflexes": self.reflexes,
            "reach": self.reach,
            "blocking": self.blocking
        }
    
    def execute_action(self, action: np.ndarray, time_step: float, x_range: float, y_range: float):
        """
        Executes a continuous action for a goalkeeper.

        Args:
            action (np.ndarray): [dx, dy, dive_left, dive_right]
            time_step (float): Duration of simulation step
            x_range (float): Field width in meters
            y_range (float): Field height in meters

        Returns:
            Optional[float]: Dive effectiveness if attempted, None otherwise
        """
        # Basic movement
        super().execute_action(action, time_step, x_range, y_range)

        # Dive logic
        if len(action) >= 4:
            if action[2] > 0.5:
                return self.dive("left")
            elif action[3] > 0.5:
                return self.dive("right")

        return None

    def get_role(self):
        """Return role name as string for rendering or role-based logic."""
        return "GK"

    def copy(self):
        """
        Create a deep copy of the goalkeeper instance.
        Useful for rendering snapshots or parallel simulations.
        """
        new_player = PlayerGoalkeeper(
            reflexes=self.reflexes,
            reach=self.reach,
            blocking=self.blocking,
            speed=self.speed,
            fov_angle=self.fov_angle,
            fov_range=self.fov_range
        )

        new_player.max_speed     = self.max_speed
        new_player.max_power     = self.max_power
        new_player.max_fov_angle = self.max_fov_angle
        new_player.max_fov_range = self.max_fov_range

        new_player.position = self.position.copy()
        new_player.last_action_direction = self.last_action_direction.copy()

        return new_player
