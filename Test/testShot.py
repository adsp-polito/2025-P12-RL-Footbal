
import os
from football_tactical_ai.env.scenarios.singleAgent.shot import OffensiveScenarioShotSingleAgent
from football_tactical_ai.helpers.visuals import render_episode
from time import time
from football_tactical_ai.env.objects.pitch import Pitch

# Initialize the pitch object
pitch = Pitch()

# Instantiate the environment with the pitch
env = OffensiveScenarioShotSingleAgent(pitch=pitch)

# Reset the environment and get initial observation and info
obs, info = env.reset()

# Lists to store states and rewards for later rendering
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

# Main interaction loop: run episode until terminated or truncated
while not terminated and not truncated:

    # Sample a random action from the environment's action space
    action = env.action_space.sample()
    # action = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])  # no movement, shot_flag=1, max power direction towards positive x (goal)
    obs, reward, t, truncated, info = env.step(action)

    # Create deep copies of the current attacker, defender, and ball states for visualization
    attacker_copy = env.attacker.copy()
    defender_copy = env.defender.copy()
    ball_copy = env.ball.copy()

    # Store the copies in the states list for rendering the episode later
    states.append({
        "player": attacker_copy,
        "ball": ball_copy,
        "opponents": [defender_copy]
    })

    # Store the reward obtained at this step for plotting or analysis
    rewards.append(reward)

# Ensure the output directory exists for saving the rendered animation
os.makedirs('test/videoTest', exist_ok=True)

# Measure rendering start time
time_start = time()
print("\nRendering episode...")

# Render the entire episode with the collected states and rewards
anim = render_episode(
    states,
    pitch=pitch,
    fps=env.fps,
    full_pitch=True,
    show_grid=False,
    show_heatmap=False,
    show_rewards=False,
    reward_grid=env.reward_grid,
    save_path="test/videoTest/testShot.mp4",  # Output video file path
    rewards_per_frame=rewards,
    show_info=True,
    show_fov=False
)


# Measure rendering end time
time_end = time()
print("Rendering complete. Animation saved in the 'videoTest' directory.")
print(f"Rendering took {time_end - time_start:.2f} seconds.\n")
