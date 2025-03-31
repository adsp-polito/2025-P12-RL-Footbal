from RLmodel.footballEnv import FootballEnv
from helpers.visuals import animate_simulation

# Initialize the environment
env = FootballEnv(render_mode="human", teamAColor="red", teamBColor="blue")
obs, _ = env.reset()

# Run full animation
animate_simulation(env, num_frames=24*60)   # 24 frames * 60 seconds = 1440 frames

# Close environment
env.close()
