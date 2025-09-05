import os
from football_tactical_ai.env.scenarios.singleAgent.view import OffensiveScenarioViewSingleAgent
from football_tactical_ai.helpers.visuals import render_episode_singleAgent
from time import time
from football_tactical_ai.env.objects.pitch import Pitch

# Create the pitch object and verify coordinate system consistency
pitch = Pitch()

# Instantiate the environment with the pitch
env = OffensiveScenarioViewSingleAgent(pitch=pitch)

# Reset the environment and retrieve initial observation
obs, info = env.reset()

# Initialize storage for rendering and reward tracking
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
    obs, reward, terminated, truncated, info = env.step(action)


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
anim = render_episode_singleAgent(
    states,
    pitch=pitch,
    fps=env.fps,
    full_pitch=True,
    show_grid=True,
    show_heatmap=False,
    show_rewards=False,
    reward_grid=env.reward_grid,
    save_path="test/videoTest/testView_SA.mp4",  # Output video file path
    rewards_per_frame=rewards,
    show_info=True,
    show_fov=True
)


# Measure rendering end time
time_end = time()
print("Rendering complete. Animation saved in the 'videoTest' directory.")
print(f"Rendering took {time_end - time_start:.2f} seconds.\n")

