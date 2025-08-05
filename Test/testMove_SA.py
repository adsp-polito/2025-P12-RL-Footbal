import os
from football_tactical_ai.env.scenarios.singleAgent.move import OffensiveScenarioMoveSingleAgent
from football_tactical_ai.helpers.visuals import render_episode_singleAgent
from time import time
from football_tactical_ai.env.objects.pitch import Pitch

# Initialize the pitch
pitch = Pitch()

# Create environment instance
env = OffensiveScenarioMoveSingleAgent(pitch=pitch)
obs, info = env.reset()

# Collect states for rendering later
states = []
rewards = []

# Add initial state before any action is taken (frame 0)
attacker_copy = env.attacker.copy()
defender_copy = env.defender.copy()
ball_copy = env.ball.copy()

states.append({
    "player": attacker_copy,
    "ball": ball_copy,
    "opponents": [defender_copy]
})

rewards.append(0.0)  # No reward yet for frame 0

terminated = truncated = False
while not (terminated or truncated):
    # Sample random action from action space (continuous in [-1, 1])
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Copy current environment state (players and ball)
    attacker_copy = env.attacker.copy()
    defender_copy = env.defender.copy()
    ball_copy = env.ball.copy()

    # Store state for later visualization
    states.append({
        "player": attacker_copy,
        "ball": ball_copy,
        "opponents": [defender_copy]
    })

    # Store rewards for each frame for rendering
    rewards.append(reward)

# Ensure output directory exists for saving animation
os.makedirs('test/videoTest', exist_ok=True)

# Measure rendering time
time_start = time()
print("\nRendering episode...")

# Render the episode and save as .mp4
anim = render_episode_singleAgent(
    states,
    pitch=pitch,
    fps=env.fps,
    full_pitch=True,
    show_grid=False,
    show_heatmap=False,
    show_rewards=False,
    reward_grid=env.reward_grid,
    save_path="test/videoTest/testMove_SA.mp4",
    rewards_per_frame=rewards,
    show_info=True,
    show_fov=False
)

time_end = time()
print("Rendering complete. Animation saved in the 'videoTest' directory.")
print(f"Rendering took {time_end - time_start:.2f} seconds.\n")