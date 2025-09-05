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
                      shot_context: Dict[str, bool] = None,
                      pass_context: Dict[str, bool] = None):
    """
    Update the ball's position based on possession and action.

    If possessed:
        - Dribbling logic: ball stays close to owner.
    If released (shot/pass):
        - Initial impulse sets velocity (only first frame).
        - Ball continues with velocity until stopped/owned.
    If free:
        - Apply velocity decay (friction).

    Args:
        ball (Ball): Ball object
        players (dict): All agents
        pitch (Pitch): Pitch instance
        time_step (float): Step duration
        actions (dict): Agent actions
        shot_context (dict): Context of any shot this step
        pass_context (dict): Context of any pass this step
    """

    dribble_offset = 0.01  # distance in front of owner
    owner_id = ball.get_owner()

    # 1. Shot impulse
    if (shot_context and 
        shot_context.get("shot_by") and 
        shot_context.get("direction") is not None and 
        np.linalg.norm(shot_context["direction"]) > 1e-6 and 
        shot_context.get("power", 0.0) > 0.0):

        pitch_width = pitch.x_max - pitch.x_min
        pitch_height = pitch.y_max - pitch.y_min

        direction = shot_context["direction"] / np.linalg.norm(shot_context["direction"])
        power = shot_context["power"]

        velocity_real = direction * power
        velocity_norm = np.array([
            velocity_real[0] / pitch_width,
            velocity_real[1] / pitch_height
        ])

        ball.set_velocity(velocity_norm)
        ball.set_owner(None)  # ensure free

    # 2. Pass
    elif (pass_context and 
          pass_context.get("pass_by") and 
          pass_context.get("direction") is not None and 
          np.linalg.norm(pass_context["direction"]) > 1e-6 and 
          pass_context.get("power", 0.0) > 0.0):
    

        pitch_width = pitch.x_max - pitch.x_min
        pitch_height = pitch.y_max - pitch.y_min

        direction = pass_context["direction"] / np.linalg.norm(pass_context["direction"])
        power = pass_context["power"]

        velocity_real = direction * power
        velocity_norm = np.array([
            velocity_real[0] / pitch_width,
            velocity_real[1] / pitch_height
        ])

        ball.set_velocity(velocity_norm)
        ball.set_owner(None)  # free ball

    # 3. Dribbling
    elif owner_id in players:
        action = actions.get(owner_id)
        direction = np.array(action[:2]) if action is not None and len(action) >= 2 else np.zeros(2)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.zeros(2)

        new_pos = np.array(players[owner_id].get_position()) + direction * dribble_offset
        ball.set_position(new_pos)
        ball.set_velocity([0.0, 0.0])

    # 4. Free physics
    else:
        # Ball has no owner → let velocity continue with decay
        vx, vy = ball.get_velocity()
        ball.set_velocity([vx, vy])

    # Final update
    ball.update(time_step)


