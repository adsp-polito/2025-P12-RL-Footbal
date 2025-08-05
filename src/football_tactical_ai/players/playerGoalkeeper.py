import numpy as np
from football_tactical_ai.env.objects.ball import Ball
from football_tactical_ai.helpers.helperFunctions import denormalize
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
                 agent_id: str = "gk_0",
                 team: str = "B",
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
            catching (float): Probability of catching the ball [0, 1].
            punch_power (float): Power of a punch to deflect the ball [0, 1].
            role (str): Player role, default is "GK".
            agent_id (str): Unique identifier for the goalkeeper.
            team (str): Team identifier, default is "B".

        Physical maxima are inherited from BasePlayer, but can be overridden.
        """
        super().__init__(agent_id=agent_id, team=team, role=role, **kwargs)

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
        Attempt a dive and evaluate if the shot is blocked or deflected.

        Args:
            direction (str): 'left' or 'right'
            ball (Ball): The ball object.

        Returns:
            dict: {
                'dive_score': float,   # Reflex + reach effectiveness
                'blocked': bool,       # True if shot was caught and stopped
                'deflected': bool      # True if shot was parried away
            }
        """
        if direction not in {"left", "right"}:
            return {"dive_score": 0.0, "blocked": False, "deflected": False}

        # Dive effectiveness based on reflexes and reach
        dive_score = 0.5 * self.reflexes + 0.5 * self.reach

        # Denormalized position and distance check
        ball_x, ball_y = denormalize(*ball.get_position())
        self_x, self_y = denormalize(*self.get_position())
        distance = np.linalg.norm([ball_x - self_x, ball_y - self_y])
        reachable = distance <= (0.5 + self.reach)

        if not reachable:
            return {"dive_score": 0.0, "blocked": False, "deflected": False}

        # Attempt to stop the shot
        dive_successful = np.random.rand() < dive_score

        if not dive_successful:
            return {"dive_score": dive_score, "blocked": False, "deflected": False}

        # CASE 1: Catch and stop
        if np.random.rand() < self.catching:
            ball.set_owner(self.agent_id)
            return {"dive_score": dive_score, "blocked": True, "deflected": False}

        # CASE 2: Deflection
        incoming_velocity = np.array(ball.get_velocity())
        speed = np.linalg.norm(incoming_velocity)
        if speed < 1e-6:
            incoming_dir = np.array([1.0, 0.0])
        else:
            incoming_dir = incoming_velocity / speed

        # Rotate incoming direction slightly
        if direction == "left":
            rot_matrix = np.array([[0.95, -0.3], [0.3, 0.95]])
        else:
            rot_matrix = np.array([[0.95, 0.3], [-0.3, 0.95]])

        deflected_dir = rot_matrix @ incoming_dir

        # Add small random noise for realism
        deflected_dir += np.random.normal(0, 0.05, size=2)
        deflected_dir /= np.linalg.norm(deflected_dir)

        # Final velocity scaled by original speed and punch power
        final_velocity = deflected_dir * speed * self.punch_power

        ball.set_velocity(final_velocity)
        ball.set_owner(None)

        return {"dive_score": dive_score, "blocked": False, "deflected": True}

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

        Action vector format:
            [dx, dy, dive_left, dive_right, shoot_flag, power, dir_x, dir_y]

        Returns:
            dict: Contextual information about the action taken.
        """

        context = {
            "dive_score": None,
            "blocked": False,
            "deflected": False,
            "shot_attempted": False,
            "invalid_shot_attempt": False,
            "invalid_shot_direction": False,
            "fov_visible": None,
            "shot_quality": None,
            "shot_direction": None,
            "shot_power": None,
        }

        # Extract and normalize movement direction
        dx, dy = action[0], action[1]
        direction = np.array([dx, dy])
        norm = np.linalg.norm(direction)

        if norm > 1e-6:
            direction /= norm
            self.last_action_direction = direction
            context["fov_visible"] = self.is_direction_visible(direction)
        else:
            direction = np.array([0.0, 0.0])
            context["fov_visible"] = None

        # Apply movement
        super().execute_action(action, time_step, x_range, y_range)

        # DIVE ACTION
        dive_result = None
        if len(action) >= 4:
            if action[2] > 0.5:
                dive_result = self.dive("left", ball)
            elif action[3] > 0.5:
                dive_result = self.dive("right", ball)

        if dive_result:
            context["dive_score"] = dive_result.get("dive_score", 0.0)
            context["blocked"] = dive_result.get("blocked", False)
            context["deflected"] = dive_result.get("deflected", False)

        # SHOOTING (goal kick or pass)
        if len(action) >= 8 and action[4] > 0.5:
            power = np.clip(action[5], 0.0, 1.0)
            shoot_direction = np.array(action[6:8])

            shot_quality, actual_direction, actual_power = self.shoot(
                desired_direction=shoot_direction,
                desired_power=power,
                enable_fov=True
            )

            context.update({
                "shot_attempted": True,
                "shot_quality": shot_quality,
                "shot_direction": actual_direction,
                "shot_power": actual_power,
                "invalid_shot_attempt": self.agent_id != ball.get_owner(),
                "invalid_shot_direction": np.allclose(actual_direction, [0.0, 0.0])
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
