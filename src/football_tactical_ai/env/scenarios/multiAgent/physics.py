import numpy as np
from typing import Dict
from football_tactical_ai.env.objects.ball import Ball
from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.players.playerBase import BasePlayer
from football_tactical_ai.helpers.helperFunctions import normalize, denormalize


def update_ball_state(ball: Ball,
                      players: Dict[str, BasePlayer],
                      pitch: Pitch,
                      actions: Dict[str, np.ndarray],
                      time_step: float,
                      shot_context: Dict[str, bool] = None,
                      pass_context: Dict[str, bool] = None,
                      collision: bool = False):
    """
    Update the ball's position based on possession and action.
    Handles dribbling, passing, shooting, free movement and post collisions.
    """

    dribble_offset = 0.01  # distance in front of the owner when dribbling
    owner_id = ball.get_owner()

    # 1. Shot: ball is released with velocity proportional to shot power
    if (shot_context and 
        shot_context.get("shot_by") and 
        shot_context.get("direction") is not None and 
        np.linalg.norm(shot_context["direction"]) > 1e-6 and 
        shot_context.get("power", 0.0) > 0.0 and
        collision == False):

        pitch_width = pitch.x_max - pitch.x_min
        pitch_height = pitch.y_max - pitch.y_min

        # Normalize direction vector and scale by power
        direction = shot_context["direction"] / np.linalg.norm(shot_context["direction"])
        power = shot_context["power"]

        velocity_real = direction * power
        velocity_norm = np.array([
            velocity_real[0] / pitch_width,
            velocity_real[1] / pitch_height
        ])

        ball.set_velocity(velocity_norm)
        ball.set_owner(None)  # ball is now free

    # 2. Pass: similar to shot, but with different intention
    elif (pass_context and 
          pass_context.get("pass_by") and 
          pass_context.get("direction") is not None and 
          np.linalg.norm(pass_context["direction"]) > 1e-6 and 
          pass_context.get("power", 0.0) > 0.0 and
          collision == False):

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
        ball.set_owner(None)  # ball is now free

    # 3. Dribbling: ball stays close to the owner, slightly in front of their movement
    elif owner_id in players:
        action = actions.get(owner_id)
        direction = np.array(action[:2]) if action is not None and len(action) >= 2 else np.zeros(2)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.zeros(2)

        new_pos = np.array(players[owner_id].get_position()) + direction * dribble_offset
        ball.set_position(new_pos)
        ball.set_velocity([0.0, 0.0])

    # 4. Free physics: ball continues moving according to its velocity
    else:
        vx, vy = ball.get_velocity()
        ball.set_velocity([vx, vy])

    # Final update for ball position
    ball.update(time_step)

    collision = _handle_post_collision(ball, pitch)

    if collision:
        # Reset contexts if collision happens
        if shot_context is not None:
            shot_context.update({"shot_by": None, "direction": None, "power": 0.0})
        if pass_context is not None:
            pass_context.update({"pass_by": None, "direction": None, "power": 0.0})

        ball.update(time_step)  # apply one more step with new velocity



def _handle_post_collision(ball: Ball, pitch: Pitch, restitution: float = 0.8, prob_goal: float = 0.5) -> bool:
    """
    Simplified ball-post collision:
    - On collision, ball can either bounce back or enter as a post-goal
    - Bounce back: random angle towards the field
    - Post-goal: redirected inside the net
    """

    bx, by = denormalize(*ball.get_position())
    vx, vy = denormalize(*ball.get_velocity())

    r_ball, r_post = ball.radius, 0.06  # meters

    posts = [
        (pitch.width, pitch.center_y - pitch.goal_width / 2),  # right bottom
        (pitch.width, pitch.center_y + pitch.goal_width / 2),  # right top
        (0.0, pitch.center_y - pitch.goal_width / 2),          # left bottom
        (0.0, pitch.center_y + pitch.goal_width / 2),          # left top
    ]

    for px, py in posts:
        dx, dy = bx - px, by - py
        dist = np.sqrt(dx**2 + dy**2)

        if dist < r_ball + r_post:
            # Incoming speed
            speed_in = np.linalg.norm([vx, vy])
            if speed_in < 1e-6:
                return False
            speed_out = speed_in * restitution

            # Decide if it's a goal or a bounce
            if np.random.rand() < prob_goal:
                # Force trajectory inside goal
                if px < pitch.width / 2:  # left goal
                    vx_new = speed_out  # goes right into the net
                else:  # right goal
                    vx_new = -speed_out  # goes left into the net
                vy_new = 0.0
            else:
                # Bounce back with random angle towards field
                if px < pitch.width / 2:  # left goal post
                    angle = np.random.uniform(-45, 225) * np.pi / 180
                else:  # right goal post
                    angle = np.random.uniform(-225, 45) * np.pi / 180

                vx_new = speed_out * np.cos(angle)
                vy_new = speed_out * np.sin(angle)

            # Back to normalized
            vx_norm = vx_new / (pitch.x_max - pitch.x_min)
            vy_norm = vy_new / (pitch.y_max - pitch.y_min)
            ball.set_velocity([vx_norm, vy_norm])

            return True

    return False
