import numpy as np
from typing import Any
from football_tactical_ai.env.objects.ball import Ball
from football_tactical_ai.players.playerBase import BasePlayer

class PlayerAttacker(BasePlayer):
    def __init__(self,
             shooting: float = 0.5,
             passing: float = 0.5,
             dribbling: float = 0.5,
             speed: float = 0.5,
             precision: float = 0.5,
             fov_angle: float = 0.5,
             fov_range: float = 0.5,
             role: str = "ATT",
             agent_id: str = "att_0",
             team: str = "A",
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
            agent_id (str): Unique identifier for the attacker.
            team (str): Team identifier, default is "A".
            
        Physical maxima are inherited from BasePlayer, but can be overridden.
        """
        super().__init__(agent_id=agent_id, team=team, role=role, **kwargs)

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
        return "CF" if self.role is None else self.role
    
    def execute_action(self, 
                    action: dict, 
                    time_step: float, 
                    x_range: float, 
                    y_range: float, 
                    ball: Ball = None) -> dict:
        """
        Executes a continuous action for an attacking player
        The attacker can move, pass, or shoot depending on the action vector

        Args:
            action (dict): Action dictionary with the following keys:
                - "move": [dx, dy] movement deltas in [-1, 1]
                - "flags": [pass, shoot] binary flags (MultiBinary(2))
                - "power": [p] shot/pass intensity in [0, 1]
                - "direction": [dir_x, dir_y] shot/pass direction in [-1, 1]
                - dir_x, dir_y: direction vector for pass/shot
            time_step (float): Duration of simulation step in seconds (e.g. 1 / FPS)
            x_range (float): Field width in meters
            y_range (float): Field height in meters
            ball (Ball): The ball object representing the current state of the ball

        Returns:
            dict: Contextual information about the action taken (for reward calculation).
        """

        # 1. MOVEMENT
        dx, dy = action["move"]
        direction = np.array([dx, dy])
        norm = np.linalg.norm(direction)

        fov_visible = None
        if norm > 1e-6:
            direction /= norm
            self.last_action_direction = direction
            fov_visible = self.is_direction_visible(direction)
        else:
            direction = np.array([0.0, 0.0])

        # Apply movement via BasePlayer (pass only dx, dy etc.)
        super().execute_action(action, time_step, x_range, y_range)

        # 2. COMMON PARAMETERS
        pass_flag, shoot_flag = action["flags"]   # MultiBinary(2)
        desired_power = float(action["power"][0])  # [0,1]
        desired_direction = np.array(action["direction"])


        context = {
            "fov_visible": fov_visible,   # will be updated for pass/shot
            "shot_attempted": False,
            "pass_attempted": False,
        }


        # If the player does not own the ball, skip shooting/passing
        if ball is None or ball.get_owner() != self.agent_id:
            return context

        # 3. INVALID ACTION CHECK
        if pass_flag == 1 and shoot_flag == 1:
            self.current_action = "idle"
            context.update({
                "invalid_action": True,
                "pass_attempted": False,
                "shot_attempted": False,
                "pass_power": 0.0,
                "shot_power": 0.0
            })
            return context
        
        # 4. PASSING LOGIC
        if pass_flag == 1:
            self.current_action = "pass"

            # Recalculate FOV visibility using pass direction (not movement)
            fov_visible_pass = self.is_direction_visible(desired_direction)

            pass_quality, pass_direction, pass_power = self.pass_ball(
                desired_direction=desired_direction,
                desired_power=desired_power,
                enable_fov=True
            )

            context.update({
                "pass_attempted": True,
                "pass_quality": pass_quality,
                "pass_power": pass_power,
                "pass_direction": pass_direction,
                "invalid_pass_direction": not fov_visible_pass,
                "start_pass_bonus": True,
                "fov_visible": fov_visible_pass,   # override with pass visibility
            })

            return context  

        # 5. SHOOTING LOGIC
        if shoot_flag == 1:
            self.current_action = "shoot"

            # Recalculate FOV visibility using shot direction
            fov_visible_shot = self.is_direction_visible(desired_direction)

            shot_quality, shot_direction, shot_power = self.shoot(
                desired_direction=desired_direction,
                desired_power=desired_power,
                enable_fov=True
            )

            # Compute alignment with goal (simplified target = center of goal)
            goal_direction = np.array([1.0, 0.5]) - np.array(self.position)
            goal_norm = np.linalg.norm(goal_direction)
            if goal_norm > 1e-6:
                goal_direction /= goal_norm
            else:
                goal_direction = np.array([1.0, 0.0])
            alignment = (np.dot(shot_direction, goal_direction) + 1) / 2.0  # ∈ [0, 1]

            # Positional quality factors
            x_pos, y_pos = self.position
            x_factor = 0.2 + 0.8 * x_pos
            y_dist = abs(y_pos - 0.5)
            y_factor = max(0.5, 1 - 2 * y_dist)
            positional_quality = x_factor * y_factor

            context.update({
                "shot_attempted": True,
                "shot_quality": shot_quality,
                "shot_power": shot_power,
                "shot_direction": shot_direction,
                "invalid_shot_direction": not fov_visible_shot,
                "shot_alignment": alignment,
                "start_shot_bonus": True,
                "shot_positional_quality": positional_quality,
                "fov_visible": fov_visible_shot,   # override with shot visibility
            })

            return context

        # 6. DEFAULT CASE → movement only
        self.current_action = "move"
        return context




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