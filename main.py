from RLEnvironment.footballEnv import FootballEnv
from helpers.visuals import simulate_with_slider, animate_simulation
from helpers.action_strategies import random_strategy

# Choose formations
formation_A = "451"
formation_B = "352"

# Initialize the environment with selected formations
env = FootballEnv(
    render_mode="human",
    teamAColor="red",
    teamBColor="blue",
    teamAFormation=formation_A,
    teamBFormation=formation_B
)

# Reset the environment to get the initial observation
obs, _ = env.reset()

# Choose action strategy
# Set the action strategy (e.g., random movement + pass/shoot)
strategy = random_strategy


# Run full animation
animate_simulation(env, num_frames=24*10, interval_ms=1000/24, action_selector = strategy) # 24 frames * 60 seconds = 1440 frames

# Run slider animation
# simulate_with_slider(env, num_frames=24*10, action_selector=strategy)

# Close environment
env.close()