import numpy as np
from football_tactical_ai.env.objects.ball import Ball
from football_tactical_ai.players.playerBase import BasePlayer
from football_tactical_ai.helpers.helperFunctions import denormalize
from typing import Any

class PlayerDefender(BasePlayer):
    
    def __init__(self,
                 tackling: float = 0.5,
                 shooting: float = 0.5,
                 precision: float = 0.5,
                 speed: float = 0.5,
                 fov_angle: float = 0.5,
                 fov_range: float = 0.5,
                 role: str = "DEF",
                 agent_id: str = "def_0",
                 team: str = "B",
                 **kwargs: Any):  
        """
        Initialize a defending player with technical and physical attributes.
        Args:
            tackling (float): Tackling skill [0, 1]
            shooting (float): Ability to kick the ball [0, 1]
            precision (float): Precision of kicks [0, 1]
            speed (float): Current speed factor [0, 1]
            fov_angle (float): Field of view angle as a fraction of max angle [0, 1].
            fov_range (float): Field of view range as a fraction of max range [0, 1].
            role (str): Player role, default is "DEF".
            agent_id (str): Unique identifier for the defender.
            team (str): Team identifier, default is "B".
            
        Physical maxima are inherited from BasePlayer, but can be overridden.
        """
        super().__init__(agent_id=agent_id, team=team, role=role, **kwargs)

        # Technical skills
        self.tackling      = tackling
        self.shooting      = shooting
        self.precision     = precision
        self.speed         = speed

        # Vision parameters (normalized + physical bounds)
        self.fov_angle = fov_angle
        self.fov_range = fov_range

        # Last action direction for reward logic
        # This will be used to determine if the player attempted a movement in the wrong direction
        self.last_action_direction = np.array([1.0, 0.0])

        # Tackle parameters
        self.tackle_range = 1.0     # meters
        self.tackle_cooldown_time = 1.0  # seconds
        self.tackle_timer = 0.0     # current cooldown state

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
    
    def attempt_tackle(self, ball: Ball, time_step: float) -> bool:
        """
        Attempt a tackle regardless of distance or cooldown.
        Cooldown is tracked but not enforced.

        Args:
            ball (Ball): The current ball object.
            time_step (float): Time elapsed since last step.

        Returns:
            bool: True if tackle was successful, False otherwise.
        """

        # Update internal tackle timer
        self.tackle_timer = max(0.0, self.tackle_timer - time_step)

        # Compute distance to ball (denormalized)
        ball_x, ball_y = denormalize(ball.get_position()[0], ball.get_position()[1])
        self_x, self_y = denormalize(self.get_position()[0], self.get_position()[1])
        dist = np.linalg.norm([ball_x - self_x, ball_y - self_y])

        # Save timestamp of tackle (if useful later)
        self.tackle_timer = self.tackle_cooldown_time  # Always reset cooldown

        # Attempt tackle: success probability based on skill, regardless of distance
        success = np.random.rand() < self.tackling
        return success, dist


    
    def execute_action(self, 
                    action: np.ndarray, 
                    time_step: float, 
                    x_range: float, 
                    y_range: float, 
                    ball: Ball = None) -> dict:
        """
        Executes a continuous action for a defensive player (movement, tackle, optional shooting).

        Args:
            action (np.ndarray): Action vector of shape (7,):
                [dx, dy, tackle_flag, shoot_flag, power, dir_x, dir_y]
            time_step (float): Duration of simulation step in seconds
            x_range (float): Field width in meters
            y_range (float): Field height in meters
            ball (Ball): Current ball object

        Returns:
            dict: Contextual information about the executed actions
        """

        # Initialize context
        context = {
            "tackle_success": False,
            "fake_tackle": False,
            "interception_success": False,   # will be set True if tackle/interception succeeds
            "shot_attempted": False,
            "invalid_shot_attempt": False,
            "invalid_shot_direction": False,
            "fov_visible": None,
            "shot_quality": None,
            "shot_direction": None,
            "shot_power": None
        }

        # Movement setup
        dx, dy = action[0], action[1]
        direction = np.array([dx, dy])
        norm = np.linalg.norm(direction)

        fov_visible = None
        if norm > 1e-6:
            direction /= norm
            self.last_action_direction = direction
            fov_visible = self.is_direction_visible(direction)
        else:
            direction = np.array([0.0, 0.0])
            fov_visible = None

        # Apply movement from BasePlayer
        super().execute_action(action, time_step, x_range, y_range)

        # Save fov info in context
        context["fov_visible"] = fov_visible

        # Tackle attempt
        self.tackle_timer = max(0.0, self.tackle_timer - time_step)  # cooldown update

        if len(action) >= 3 and action[2] > 0.5:  # tackle_flag

            self.current_action = "tackle"

            success, distance = self.attempt_tackle(ball, time_step)
            context["tackle_success"] = success

            # Fake tackle → too far or cooldown active
            if distance > self.tackle_range or self.tackle_timer > 0.0:
                context["fake_tackle"] = True
                context["tackle_success"] = False

            # If tackle was actually successful → mark interception
            if context["tackle_success"]:
                context["interception_success"] = True

        # Shooting attempt (clearance / emergency shot)
        elif len(action) >= 7 and action[3] > 0.5:  # shoot_flag

            self.current_action = "shoot"

            power = np.clip(action[4], 0.0, 1.0)
            direction = np.array(action[5:7])

            shot_quality, shot_direction, shot_power = self.shoot(
                desired_direction=direction,
                desired_power=power,
                enable_fov=True
            )

            context.update({
                "shot_attempted": True,
                "shot_quality": shot_quality,
                "shot_direction": shot_direction,
                "shot_power": shot_power,
                "invalid_shot_attempt": self.agent_id != ball.get_owner(),
                "invalid_shot_direction": np.allclose(shot_direction, [0.0, 0.0]),
            })

        # Default case → movement or idle
        else:
            if norm > 1e-6:
                self.current_action = "move"
            else:
                self.current_action = "idle"

        return context




    def copy(self):
        """
        Create a deep copy of the player instance, used for rendering state snapshots.
        """
        new_player = PlayerDefender(
            tackling=self.tackling,
            shooting=self.shooting,
            precision=self.precision,
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
