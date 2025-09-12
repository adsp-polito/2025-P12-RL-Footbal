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
                      pass_context: Dict[str, bool] = None):
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
        shot_context.get("power", 0.0) > 0.0):

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

    # Handle possible collision with goalposts
    _handle_post_collision(ball, pitch, restitution=0.8)


def _handle_post_collision(ball: Ball, pitch: Pitch, restitution: float = 0.8) -> bool:
    """
    Check if the ball collides with the goalposts (both goals) and handle bounce.

    Args:
        ball (Ball): Ball object
        pitch (Pitch): Pitch object
        restitution (float): Coefficient of restitution (0 = inelastic, 1 = elastic).

    Returns:
        bool: True if a collision occurred, False otherwise.
    """
    # Get ball position in meters (denormalized)
    ball_x, ball_y = denormalize(*ball.get_position())
    vx, vy = ball.get_velocity()

    # Ball and post radii
    r_ball = 0.11  # ball radius in meters
    r_post = 0.06  # post radius in meters

    # Define post centers for both goals (left and right)
    posts = [
        # Right goal (x = pitch.width)
        (pitch.width, pitch.center_y - pitch.goal_width / 2),  # bottom post
        (pitch.width, pitch.center_y + pitch.goal_width / 2),  # top post
        # Left goal (x = 0)
        (0.0, pitch.center_y - pitch.goal_width / 2),  # bottom post
        (0.0, pitch.center_y + pitch.goal_width / 2),  # top post
    ]

    # Check collision with each post
    for px, py in posts:
        dx = ball_x - px
        dy = ball_y - py
        dist = np.sqrt(dx**2 + dy**2)

        if dist < r_ball + r_post:  # collision detected
            # Normal vector from post to ball
            if dist == 0:
                nx, ny = 1.0, 0.0
            else:
                nx, ny = dx / dist, dy / dist

            # Decompose velocity into normal and tangential components
            v = np.array([vx, vy])
            v_normal = np.dot(v, [nx, ny]) * np.array([nx, ny])
            v_tangent = v - v_normal

            # Reflect normal component to simulate bounce
            v_reflected = v_tangent - restitution * v_normal

            # Update ball velocity
            ball.set_velocity(*v_reflected)

            # Push ball outside the post to avoid overlap ("unstick" it)
            overlap = r_ball + r_post - dist
            new_x = ball_x + nx * overlap
            new_y = ball_y + ny * overlap
            ball.set_position(normalize(new_x, new_y))

            return True  # collision handled

    return False  # no collision
