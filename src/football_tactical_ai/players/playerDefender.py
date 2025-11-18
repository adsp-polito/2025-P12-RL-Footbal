import numpy as np
from football_tactical_ai.env.objects.ball import Ball
from football_tactical_ai.players.playerBase import BasePlayer
from football_tactical_ai.helpers.helperFunctions import denormalize
from typing import Any

class PlayerDefender(BasePlayer):
    
    
    def __init__(self,
                 tackling: float = 0.15,
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
        Initialize a defending player with technical and physical attributes
        Args:
            tackling (float): Tackling skill [0, 1]
            shooting (float): Ability to kick the ball [0, 1]
            precision (float): Precision of kicks [0, 1]
            speed (float): Current speed factor [0, 1]
            fov_angle (float): Field of view angle as a fraction of max angle [0, 1]
            fov_range (float): Field of view range as a fraction of max range [0, 1]
            role (str): Player role, default is "DEF"
            agent_id (str): Unique identifier for the defender
            team (str): Team identifier, default is "B"
            
        Physical maxima are inherited from BasePlayer, but can be overridden
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
            "tackling": self.tackling
        }

    def get_role(self):
        """Return role name as string for rendering or role-based logic."""
        return "DEF" if self.role is None else self.role
    
    def attempt_tackle(self, ball: Ball) -> tuple[bool, float]:
        """
        Attempt a tackle based on skill.
        Does NOT handle cooldown anymore (only success probability + distance).
        """
        # Compute distance to ball (denormalized)
        ball_x, ball_y = denormalize(*ball.get_position())
        self_x, self_y = denormalize(*self.get_position())
        dist = np.linalg.norm([ball_x - self_x, ball_y - self_y])

        # Success probability depends only on tackling skill
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
            action (np.ndarray): Flat action vector [dx, dy, tackle_flag, shoot_flag, power, dir_x, dir_y]
                - dx, dy ∈ [-1, 1]: movement
                - tackle_flag > 0.5 → attempt tackle
                - shoot_flag  > 0.5 → attempt clearance shot
                - power ∈ [0, 1]: intensity
                - dir_x, dir_y ∈ [-1, 1]: direction for shot/tackle
            time_step (float): Duration of simulation step (s)
            x_range (float): Pitch width (m)
            y_range (float): Pitch height (m)
            ball (Ball): Current ball object

        Returns:
            dict: Context info about the executed actions (for rewards).
        """

        # Context dictionary (used for reward shaping and logging)
        context = {
            "tackle_success": False,
            "fake_tackle": False,
            "interception_success": False,
            "shot_attempted": False,
            "invalid_shot_attempt": False,
            "invalid_shot_direction": False,
            "fov_visible": None,
            "shot_quality": None,
            "shot_direction": None,
            "shot_power": None
        }

        # UNPACK ACTION
        dx, dy, f0, f1, power, dir_x, dir_y = action
        tackle_flag, shoot_flag = int(f0 > 0.5), int(f1 > 0.5)
        desired_power = float(np.clip(power, 0.0, 1.0))
        desired_direction = np.array([dir_x, dir_y], dtype=np.float32)

        # MOVEMENT
        direction = np.array([dx, dy], dtype=np.float32)
        norm = np.linalg.norm(direction)

        fov_visible = None
        if norm > 1e-6:
            direction /= norm
            self.last_action_direction = direction
            fov_visible = self.is_direction_visible(direction)

        # Apply base movement
        super().execute_action(direction, time_step, x_range, y_range)

        # Save FOV information
        context["fov_visible"] = fov_visible

        # Update tackle timer (cooldown)
        self.tackle_timer = max(0.0, self.tackle_timer - time_step)

        # TACKLE attempt
        if tackle_flag == 1:
            self.current_action = "tackle"

            if self.tackle_timer > 0.0:
                # Tackle is on cooldown → cannot succeed
                context["fake_tackle"] = True
            else:
                # Try to execute tackle (probabilistic success)
                success, distance = self.attempt_tackle(ball)

                # Too far from the ball → fake tackle
                if distance > self.tackle_range:
                    context["fake_tackle"] = True
                    context["tackle_success"] = False
                else:
                    context["tackle_success"] = success

                # If tackle attempted (success or fail), start cooldown
                self.tackle_timer = self.tackle_cooldown_time

                # If successful, mark interception
                if context["tackle_success"]:
                    context["interception_success"] = True

        # SHOOTING (CLEARANCE) attempt
        elif shoot_flag == 1:
            self.current_action = "shoot"

            shot_quality, shot_direction, shot_power = self.shoot(
                desired_direction=desired_direction,
                desired_power=desired_power,
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

        # 6. Default case → movement or idle
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
