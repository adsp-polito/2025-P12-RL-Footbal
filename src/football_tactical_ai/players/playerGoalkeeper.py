import numpy as np
from football_tactical_ai.env.objects.ball import Ball
from football_tactical_ai.players.playerBase import BasePlayer
from typing import Any

class PlayerGoalkeeper(BasePlayer):
    def __init__(self,
                 reflexes: float = 0.5,
                 reach: float = 0.5,
                 shooting: float = 0.5,
                 precision: float = 0.5,
                 speed: float = 0.5,
                 fov_angle: float = 0.5,
                 fov_range: float = 0.5,
                 catching: float = 0.5,
                 punch_power: float = 0.5,
                 role: str = "GK",
                 **kwargs: Any):
        """
        Initialize a goalkeeper with goalkeeping-specific attributes.

        Args:
            reflexes (float): Speed of reaction [0, 1].
            reach (float): How far the keeper can dive [0, 1].
            shooting (float): Ability to kick the ball [0, 1].
            precision (float): Precision of kicks [0, 1].
            speed (float): Movement speed multiplier [0, 1].
            fov_angle (float): Vision angle [0, 1].
            fov_range (float): Vision range [0, 1].
        """
        super().__init__(agent_id=None, team=None, role=role, **kwargs)

        # Technical attributes
        self.reflexes   = reflexes
        self.reach      = reach
        self.shooting   = shooting
        self.precision  = precision
        self.fov_angle  = fov_angle
        self.fov_range  = fov_range
        self.speed      = speed
        self.catching   = catching
        self.punch_power = punch_power

        # Position in normalized coordinates [0, 1]
        self.position = [0.0, 0.0]
        self.last_action_direction = np.array([1.0, 0.0])  # last action direction


    def dive(self, direction: str, ball: Ball) -> dict:
        """
        Attempt a dive and evaluate interception.

        Args:
            direction (str): 'left' or 'right'
            ball (Ball): The ball object.

        Returns:
            dict: {
                'dive_success': float,
                'save_success': bool,
                'deflected': bool
            }
        """
        if direction not in {"left", "right"}:
            return {"dive_success": 0.0, "save_success": False, "deflected": False}

        # Effectiveness of dive based on reflexes and reach
        effectiveness = 0.5 * self.reflexes + 0.5 * self.reach

        # Distance check
        ball_x, ball_y = ball.get_position(denormalized=True)
        self_x, self_y = self.get_position(denormalized=True)
        distance = np.linalg.norm([ball_x - self_x, ball_y - self_y])
        reachable = distance <= (1.0 + self.reach * 2.0)

        if not reachable:
            return {
                "dive_success": effectiveness,
                "save_success": False,
                "deflected": False
            }

        # Randomized outcome based on effectiveness
        save_success = np.random.rand() < effectiveness

        if save_success:
            # Catch or deflect?
            if np.random.rand() < self.catching:
                ball.set_owner(self.agent_id)
                return {
                    "dive_success": effectiveness,
                    "save_success": True,
                    "deflected": False
                }
            else:
                # Directional deflection based on input
                sign = -1 if direction == "left" else 1
                deflect_velocity = np.array([0.12 * sign, 0.04]) * self.punch_power
                ball.set_velocity(deflect_velocity)
                ball.set_owner(None)
                return {
                    "dive_success": effectiveness,
                    "save_success": True,
                    "deflected": True
                }

        return {
            "dive_success": effectiveness,
            "save_success": False,
            "deflected": False
        }


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
    
    def execute_action(self, 
                       action: np.ndarray, 
                       time_step: float, 
                       x_range: float, 
                       y_range: float, 
                       ball: Ball = None) -> dict:
        """
        Executes a continuous action for a goalkeeper, including movement, dive, and shoot.

        Args:
            action (np.ndarray): Action vector:
                [dx, dy, dive_left, dive_right, shoot_flag, power, dir_x, dir_y]
            time_step (float): Simulation step in seconds
            x_range (float): Field width
            y_range (float): Field height
            ball (Ball): The ball object representing the current state of the ball.

        Returns:
            dict: Contextual information about the action taken.
        """

        context = {
            "dive_success": None,
            "save_success": False,
            "shot_attempted": False,
            "not_owner_shot_attempt": False,
            "invalid_shot_direction": False,
            "fov_visible": None,
            "shot_quality": None,
            "shot_direction": None,
            "shot_power": None,
        }

        # Extract movement vector
        dx, dy = action[0], action[1]
        direction = np.array([dx, dy])
        norm = np.linalg.norm(direction)

        # Default visibility check result
        fov_visible = None

        # If agent is moving, update direction and check FOV
        if norm > 1e-6:
            direction /= norm  # Normalize
            self.last_action_direction = direction
            fov_visible = self.is_direction_visible(direction)
        else:
            direction = np.array([0.0, 0.0])
            fov_visible = None  # No movement

        # Movement
        super().execute_action(action, time_step, x_range, y_range)

        if len(action) >= 4:
            if action[2] > 0.5:
                dive_result = self.dive("left", ball)
            elif action[3] > 0.5:
                dive_result = self.dive("right", ball)
            else:
                dive_result = None

            if dive_result:
                context["dive_success"] = dive_result["dive_success"]
                context["save_success"] = dive_result["save_success"]

        # Shooting logic (e.g., goal kick, pass)
        if len(action) >= 8 and action[4] > 0.5:
            power = np.clip(action[5], 0.0, 1.0)
            direction = np.array(action[6:8])

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
                "not_owner_shot_attempt": self.agent_id != ball.get_owner(),
                "invalid_shot_direction": np.allclose(shot_direction, [0.0, 0.0]),
                "fov_visible": fov_visible
            })

        return context


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
            shooting=self.shooting,
            precision=self.precision,
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
