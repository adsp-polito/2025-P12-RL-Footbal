import os
from time import time
from football_tactical_ai.env.scenarios.multiAgent.multiAgentEnv import FootballMultiEnv
from football_tactical_ai.helpers.visuals import render_episode_multiAgent
from football_tactical_ai.env.objects.pitch import Pitch



def test_multiagent_render(save_path="test/videoTest/testMultiAgent.mp4"):
    """
    Run a single episode in FootballMultiEnv with random actions
    and render it to a video file.

    Args:
        save_path (str): Path where the rendered video will be saved.
    """
    # Initialize pitch and environment
    pitch = Pitch()
    env = FootballMultiEnv()
    env.reset()  

    # Storage for rendering
    states = []

    # Add initial state (frame 0)
    frame_0 = {
        "players": {agent_id: env.players[agent_id].copy() for agent_id in env.agents},
        "ball": env.ball.copy()
    }
    states.append(frame_0)

    # Initialize termination and truncation flags
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}

    # Main loop for multi-agent environment
    while not any(terminated.values()) and not any(truncated.values()):
        # Sample random actions for all agents
        actions = {
            agent_id: env.action_space(agent_id).sample()
            for agent_id in env.agents
        }

        # Step the environment
        obs, rewards, terminated, truncated, infos = env.step(actions)

        # Save current state snapshot
        frame = {
            "players": {agent_id: env.players[agent_id].copy() for agent_id in env.agents},
            "ball": env.ball.copy()
        }
        states.append(frame)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Rendering
    print("\nRendering multi-agent episode...")
    time_start = time()

    anim = render_episode_multiAgent(
        states,
        pitch=pitch,
        fps=env.fps,
        full_pitch=True,
        show_grid=False,
        show_heatmap=False,
        show_rewards=False,
        reward_grid=None,   
        show_fov=False,
        show_names=True,
    )

    anim.save(save_path, writer="ffmpeg", fps=env.fps)

    time_end = time()
    print(f"Rendering complete. Animation saved in '{save_path}'")
    print(f"Rendering took {time_end - time_start:.2f} seconds.\n")


if __name__ == "__main__":
    test_multiagent_render()
