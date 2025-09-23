import os
import numpy as np
from time import time
from football_tactical_ai.env.scenarios.multiAgent.multiAgentEnv import FootballMultiEnv
from football_tactical_ai.helpers.visuals import render_episode_multiAgent
from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.helpers.helperFunctions import normalize


def test_shot_on_post(save_path="test/videoTest/test_shot_post.mp4", post="right"):
    """
    Scripted episode: attacker shoots exactly on the left/right post with perfect precision
    """
    pitch = Pitch()
    env = FootballMultiEnv()
    obs, _ = env.reset()

    # Set attacker precision to perfect
    attacker_id = "att_1"
    if hasattr(env.players[attacker_id], "precision"):
        env.players[attacker_id].precision = 1.0

    # Coordinate in meters of the posts
    if post == "right":
        target_x, target_y = 120, 40 + pitch.goal_width / 2   # ~43.66 m
    else:
        target_x, target_y = 120, 40 - pitch.goal_width / 2   # ~36.34 m

    # Normalize the target
    target_xn, target_yn = normalize(target_x, target_y)

    # Place attacker and ball near the target post
    shot_x = target_xn - 0.05
    shot_y = target_yn
    env.players[attacker_id].set_position((shot_x, shot_y))
    if hasattr(env.players[attacker_id], "set_orientation"):
        env.players[attacker_id].set_orientation(0.0)  # facing the goal
    env.ball.set_position((shot_x, shot_y))
    env.ball.set_owner(attacker_id)


    # Compute shot direction towards the post
    att_x, att_y = env.players[attacker_id].get_position()
    dx, dy = target_xn - att_x, target_yn - att_y
    norm = np.linalg.norm([dx, dy])
    dir_x, dir_y = dx / norm, dy / norm

    states = [env.get_render_state()]

    # Setup actions
    actions = {}
    for agent_id in env.agents:
        if agent_id == attacker_id:
            actions[agent_id] = np.array([
                0.0, 0.0,   # movement
                0.0,        # pass_flag
                1.0,        # shoot_flag
                1.0,        # power
                dir_x, dir_y
            ], dtype=np.float32)
        else:
            actions[agent_id] = np.zeros(env.action_space(agent_id).shape, dtype=np.float32)

    terminated, truncated = False, False
    step_count = 0
    while not terminated and not truncated and step_count < 50:
        obs, rewards, terminations, truncations, infos = env.step(actions)
        states.append(env.get_render_state())
        terminated = terminations["__all__"]
        truncated = truncations["__all__"]
        step_count += 1

    # Save the video
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim = render_episode_multiAgent(
        states, pitch=pitch, fps=env.fps,
        full_pitch=True, show_grid=False, show_heatmap=False, show_rewards=False, reward_grid=None,
        show_fov=False, show_names=True
    )
    anim.save(save_path, writer="ffmpeg", fps=env.fps)
    print(f"Video saved: {save_path}")


if __name__ == "__main__":
    test_shot_on_post(post="right")
