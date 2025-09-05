import numpy as np
from typing import Dict
from football_tactical_ai.env.objects.ball import Ball
from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.players.playerBase import BasePlayer


def update_ball_state(ball: Ball,
                      players: Dict[str, BasePlayer],
                      pitch: Pitch,
                      actions: Dict[str, np.ndarray],
                      time_step: float,
                      shot_context: Dict[str, bool] = None,):
    """
    Update the ball's position based on possession and action.

    If possessed:
        - Check if a shoot is triggered → set ball velocity and release ownership
        - Else: simulate dribbling (keep ball in front of player)

    If unpossessed:
        - Apply physics update (velocity decay, friction)

    Args:
        ball (Ball): The ball object.
        players (dict): All agents {agent_id: player object}.
        actions (dict): Actions taken by each agent this step.
        time_step (float): Time step duration.
        dribble_offset (float): Distance to place ball in front of owner.
    """
    dribble_offset = 0.01  # Distance to place ball in front of owner
    owner_id = ball.get_owner()

    # Apply shot if provided
    if shot_context and shot_context["shot_by"] and shot_context["direction"] is not None:
        
        pitch_width = pitch.x_max - pitch.x_min
        pitch_height = pitch.y_max - pitch.y_min

        direction = shot_context["direction"]
        power = shot_context["power"]

        velocity_real = direction * power  # m/s
        velocity_norm = np.array([
            velocity_real[0] / pitch_width,
            velocity_real[1] / pitch_height
        ])
        ball.set_velocity(velocity_norm)

        ball.update(time_step)
        return  # skip dribbling/physics this frame since a shot overrides it

    owner_id = ball.get_owner()

    if owner_id in players:
        # Standard dribbling logic
        action = actions.get(owner_id)
        direction = np.array(action[:2]) if action is not None and len(action) >= 2 else np.zeros(2)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.zeros(2)
        new_pos = np.array(players[owner_id].get_position()) + direction * dribble_offset
        ball.set_position(new_pos)
        ball.set_velocity([0.0, 0.0])
        
        ball.update(time_step)
        return

