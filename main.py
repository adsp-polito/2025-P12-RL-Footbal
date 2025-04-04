from RLmodel.footballEnv import FootballEnv
from helpers.visuals import animate_simulation


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
animate_simulation(env, num_frames=24*10, interval_ms=1000/24)   # 24 frames * 60 seconds = 1440 frames

# Close environment
env.close()