import numpy as np
from football_tactical_ai.players.playerBase import BasePlayer
from typing import Any

class PlayerDefender(BasePlayer):
    
    def __init__(self,
                 tackling: float = 0.5,
                 marking: float = 0.5,
                 anticipation: float = 0.5,
                 speed: float = 0.5,
                 fov_angle: float = 0.5,
                 fov_range: float = 0.5,
                 role: str = "DEF",
                 **kwargs: Any):  
        """
        Initialize a defending player with technical and physical attributes.
        Args:
            tackling (float): Tackling skill [0, 1]
            marking (float): Marking skill [0, 1]
            anticipation (float): Anticipation skill [0, 1]
            speed (float): Current speed factor [0, 1]
            fov_angle (float): Field of view angle as a fraction of max angle [0, 1].
            fov_range (float): Field of view range as a fraction of max range [0, 1].
            role (str): Player role, default is "DEF".
        Physical maxima are inherited from BasePlayer, but can be overridden.
        """
        super().__init__(agent_id=None, team=None, role=role, **kwargs)

        # Technical skills
        self.tackling      = tackling
        self.marking       = marking
        self.anticipation  = anticipation
        self.speed         = speed

        # Vision parameters (normalized + physical bounds)
        self.fov_angle = fov_angle
        self.fov_range = fov_range

        # Last action direction for reward logic
        # This will be used to determine if the player attempted a movement in the wrong direction
        self.last_action_direction = np.array([1.0, 0.0])

    def tackle(self):
        """
        Attempt a tackle. The logic here is symbolic — real effects
        (e.g. ball possession change) are handled in the environment.
        """
        # In a real system, you'd return success probability
        return self.tackling

    def get_parameters(self):
        """Return technical attributes for logging or debugging."""
        return {
            "role": self.get_role(),
            "speed": self.speed,
            "fov_angle": self.fov_angle,
            "fov_range": self.fov_range,
            "tackling": self.tackling,
            "marking": self.marking,
            "anticipation": self.anticipation
        }

    def get_role(self):
        """Return role name as string for rendering or role-based logic."""
        return "DEF"
    
    def execute_action(self, action: np.ndarray, time_step: float, x_range: float, y_range: float):
        """
        Executes a continuous action for a defensive player.

        Args:
            action (np.ndarray): Action vector of the form:
                [dx, dy, tackle_flag]
            time_step (float): Duration of simulation step (seconds)
            x_range (float): Field width in meters
            y_range (float): Field height in meters

        Returns:
            Optional[float]: If a tackle is attempted, returns its effectiveness.
                             Otherwise, returns None.
        """

        # Movement part
        super().execute_action(action, time_step, x_range, y_range)

        # Tackle logic
        if len(action) >= 3 and action[2] > 0.5:
            return self.tackle()

        return None

    def copy(self):
        """
        Create a deep copy of the player instance, used for rendering state snapshots.
        """
        new_player = PlayerDefender(
            tackling=self.tackling,
            marking=self.marking,
            anticipation=self.anticipation,
            speed=self.speed,
            fov_angle=self.fov_angle,
            fov_range=self.fov_range
        )
        # Copy the inherited attributes from BasePlayer
        new_player.max_speed     = self.max_speed
        new_player.max_power     = self.max_power
        new_player.max_fov_angle = self.max_fov_angle
        new_player.max_fov_range = self.max_fov_range

        # Copy the position
        new_player.position = self.position.copy()
        new_player.last_action_direction = self.last_action_direction.copy()
        
        return new_player
