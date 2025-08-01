import numpy as np
from football_tactical_ai.players.playerBase import BasePlayer

class PlayerGoalkeeper(BasePlayer):
    def __init__(self,
                 reflexes: float = 0.5,
                 diving: float = 0.5,
                 positioning: float = 0.5,
                 speed: float = 0.5):
        """
        Initialize a goalkeeper with technical and physical attributes.

        Args:
            reflexes (float): Reflexes skill [0, 1]
            diving (float): Diving skill [0, 1]
            positioning (float): Positioning skill [0, 1]
            speed (float): Current speed factor [0, 1]

        Physical maxima are inherited from BasePlayer, but can be overridden.
        """
        super().__init__()

        # Technical attributes
        self.reflexes = reflexes
        self.diving = diving
        self.positioning = positioning
        self.speed = speed

        # Position in normalized coordinates [0, 1]
        self.position = [0.0, 0.0]
        self.last_action_direction = np.array([1.0, 0.0])  # last action direction

    def get_parameters(self):
        """Return goalkeeper's technical attributes as dictionary (for logs, debug, stats)."""
        return {
            "reflexes": self.reflexes,
            "diving": self.diving,
            "positioning": self.positioning
        }

    def get_role(self):
        """Return role name as string for rendering or role-based logic."""
        return "GK"

    def copy(self):
        """
        Create a deep copy of the goalkeeper instance.
        This is used to store the current state during rendering.
        """
        new_player = PlayerGoalkeeper(
            reflexes=self.reflexes,
            diving=self.diving,
            positioning=self.positioning,
            speed=self.speed
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
