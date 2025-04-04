from RLmodel.footballEnv import FootballEnv
from helpers.visuals import simulate_with_slider, animate_simulation
from action_logic import random_passing


# Choose formations
formation_A = "3421"
formation_B = "352"

# Initialize the environment with selected formations
env = FootballEnv(
    render_mode="human",
    teamAColor="red",
    teamBColor="blue",
    teamAFormation=formation_A,
    teamBFormation=formation_B
)

# Reset the environment
obs, _ = env.reset()

# Run full animation
# simulate_with_slider(env, num_frames=24*10, action_selector=random_passing)   

animate_simulation(env, num_frames=24*10, interval_ms=1000/24, action_selector = random_passing)  # 24 frames * 60 seconds = 1440 frames

# Close environment
env.close()