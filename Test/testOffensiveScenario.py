import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path so RLEnvironment can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RLEnvironment.offensiveScenarioEnv import OffensiveScenarioEnv
from RLEnvironment.player import Player
from RLEnvironment.ball import Ball
from helpers.visuals import render_episode

# === AGENT POLICY ===
def agent_policy(obs):
    return 4  # Always move east

# === RUN EPISODE AND RENDER IT ===
def run_episode_with_render():
    env = OffensiveScenarioEnv(render_mode="human")
    obs, _ = env.reset()
    states = []

    total_frames = 48
    action = agent_policy(obs)
    env.player.step(action)  # First target

    for _ in range(total_frames):
        # Trigger new decision as soon as player reaches the center of the target cell
        if env.player.target is None:
            obs = env._get_obs()
            action = agent_policy(obs)
            env.player.step(action)

        # Update motion
        env.player.update_position()
        if env.player.has_ball:
            offset = env.player.get_ball_offset()
            env.ball.set_position(env.player.get_position() + offset)

        # Record frame for animation
        player_copy = Player(player_id=env.player.player_id)
        player_copy.set_position(env.player.get_position())
        player_copy.has_ball = env.player.has_ball
        player_copy.last_direction = env.player.last_direction.copy()

        ball_copy = Ball()
        ball_copy.set_position(env.ball.get_position())
        ball_copy.owner_id = env.ball.owner_id

        states.append({
            "player": player_copy,
            "ball": ball_copy,
            "opponents": []
        })

    anim = render_episode(states, show_grid=True, show_cell_ids=False)
    plt.show()
    env.close()

if __name__ == "__main__":
    run_episode_with_render()