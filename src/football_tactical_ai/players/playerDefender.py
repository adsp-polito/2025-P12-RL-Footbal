import numpy as np
from football_tactical_ai.players.playerBase import BasePlayer

class PlayerDefender(BasePlayer):
    
    def __init__(self,
                 tackling: float = 0.5,
                 marking: float = 0.5,
                 anticipation: float = 0.5,
                 speed: float = 0.5,
                 fov_angle: float = 0.5,
                 fov_range: float = 0.5):  
        """
        Initialize a defending player with technical and physical attributes.

        Args:
            tackling (float): Tackling skill [0, 1]
            marking (float): Marking skill [0, 1]
            anticipation (float): Anticipation skill [0, 1]
            speed (float): Current speed factor [0, 1]
            fov_angle (float): Field of view angle as a fraction of max angle [0, 1].
            fov_range (float): Field of view range as a fraction of max range [0, 1].
        
        Physical maxima are inherited from BasePlayer, but can be overridden.
        """
        super().__init__()

        # Technical skills
        self.tackling = tackling
        self.marking = marking
        self.anticipation = anticipation
        self.speed = speed

        # Vision parameters (normalized + physical bounds)
        self.fov_angle = fov_angle
        self.fov_range = fov_range

        # Last action direction for reward logic
        # This will be used to determine if the player attempted a movement in the wrong direction
        self.last_action_direction = np.array([1.0, 0.0])

    def get_parameters(self):
        """Return player's technical attributes as dictionary (for logs, debug, stats)."""
        return {
            "tackling": self.tackling,
            "marking": self.marking,
            "anticipation": self.anticipation
        }

    def get_role(self):
        """Return role name as string for rendering or role-based logic."""
        return "DEF"

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
