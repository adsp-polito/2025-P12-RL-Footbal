import os
from time import time
from football_tactical_ai.env.scenarios.multiAgent.multiAgentEnv import FootballMultiEnv
from football_tactical_ai.helpers.visuals import render_episode_multiAgent
from football_tactical_ai.env.objects.pitch import Pitch

# Initialize pitch and environment
pitch = Pitch()
env = FootballMultiEnv()
obs = env.reset()[0]  # reset returns (obs, info)

# Storage for rendering
states = []

# Add initial state (frame 0)
frame_0 = {
    "players": {agent_id: env.players[agent_id].copy() for agent_id in env.agents},
    "ball": env.ball.copy()
}

# Store initial state
states.append(frame_0)

# initialize termination and truncation flags
# These will be used to control the loop and check if the episode is done
terminated = {agent: False for agent in env.agents}
truncated = {agent: False for agent in env.agents}

# Main loop for multi-agent environment
while not all(terminated.values()) and not all(truncated.values()):
    # Sample actions for all agents
    actions = {
        agent_id: env.action_space(agent_id).sample()
        for agent_id in env.agents
    }

    # Step the environment with sampled actions
    obs, reward, terminated, truncated, info = env.step(actions)

    # Store the current state for rendering
    frame = {
        "players": {agent_id: env.players[agent_id].copy() for agent_id in env.agents},
        "ball": env.ball.copy()
    }
    states.append(frame)

# Ensure output directory exists
os.makedirs('test/videoTest', exist_ok=True)

# Render episode
time_start = time()
print("\nRendering multi-agent episode...")

# Create the animation object
anim = render_episode_multiAgent(
    states,
    pitch=pitch,
    fps=env.fps,
    full_pitch=True,
    show_grid=False,
    show_heatmap=False,
    show_rewards=False,
    reward_grid=env.reward_grids["ATT"],
    show_fov=True  # Optional: set to False if you don't want vision cones
)

# Save video
anim.save("test/videoTest/testMultiAgent.mp4", writer='ffmpeg', fps=env.fps)

time_end = time()
print("Rendering complete. Animation saved in 'test/videoTest/testMultiAgent.mp4'")
print(f"Rendering took {time_end - time_start:.2f} seconds.\n")
