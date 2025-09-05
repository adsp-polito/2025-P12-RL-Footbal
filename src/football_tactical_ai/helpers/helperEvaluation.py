from football_tactical_ai.helpers.visuals import render_episode_singleAgent, render_episode_multiAgent
import numpy as np

def evaluate_and_render(model, env, pitch, save_path=None, episode=0, fps=24,
                        show_grid=False, show_heatmap=False,
                        show_rewards=False, full_pitch=True, show_info=True, show_fov=False):
    """
    Evaluate a trained model on a single episode and optionally render it as a video.

    Args:
        model: Trained PPO agent.
        env: Evaluation environment instance.
        pitch: Pitch instance for rendering.
        save_path (str): Optional path to save the video. If None, no rendering is saved.
        episode (int): Current episode number (used for logging).
        fps (int): Frames per second for rendering.
        show_grid (bool): Whether to draw grid lines on the pitch.
        show_heatmap (bool): Whether to color cells based on reward values.
        show_rewards (bool): Whether to display numeric reward values inside cells.
        full_pitch (bool): Whether to render the full pitch or only half.
        show_info (bool): Whether to show cumulative reward and extra info in the video.

    Returns:
        float: The cumulative reward accumulated during this evaluation episode.
    """
    # Reset environment
    obs, _ = env.reset()
    terminated = truncated = False
    states = []
    rewards_per_frame = [] if save_path else None
    cumulative_reward = 0.0

    # Initial state
    attacker_copy = env.attacker.copy()
    defender_copy = env.defender.copy()
    ball_copy = env.ball.copy()

    # Store initial state
    states.append({
        "player": attacker_copy,
        "ball": ball_copy,
        "opponents": [defender_copy]
    })

    # If rendering is enabled, initialize rewards per frame
    if save_path:
        rewards_per_frame.append(0.0)

    # Main episode loop
    while not terminated and not truncated:
        # Get action from the model
        action, _ = model.predict(obs)
        # Step the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        cumulative_reward += reward

        # Store state for rendering
        attacker_copy = env.attacker.copy()
        defender_copy = env.defender.copy()
        ball_copy = env.ball.copy()

        states.append({
            "player": attacker_copy,
            "ball": ball_copy,
            "opponents": [defender_copy]
        })

        # If rendering is enabled, store the reward for this frame
        if save_path:
            rewards_per_frame.append(reward)

    # Final state after episode ends
    if save_path:
        render_episode_singleAgent(
            states,
            pitch=pitch,
            save_path=save_path,
            fps=fps,
            full_pitch=full_pitch,
            show_grid=show_grid,
            show_heatmap=show_heatmap,
            show_rewards=show_rewards,
            reward_grid=env.reward_grid,
            rewards_per_frame=rewards_per_frame,
            show_info=show_info,
            show_fov=show_fov
        )

    return cumulative_reward



def evaluate_and_render_multi(
    model,
    env,
    pitch,
    save_path=None,
    episode=0,
    fps=24,
    show_grid=False,
    show_heatmap=False,
    show_rewards=False,
    full_pitch=True,
    show_info=True,
    show_fov=False,
):
    """
    Evaluate a trained shared-policy PPO model in a multi-agent environment.

    - Each agent receives its own observation.
    - All agents use the same SB3 policy to predict their actions.
    - The episode is rendered to video if save_path is provided.

    Args:
        model: Trained PPO model (shared across agents).
        env: FootballMultiEnv (PettingZoo ParallelEnv).
        pitch: Pitch instance (for rendering).
        save_path (str, optional): Path to save rendered video (mp4).
        episode (int): Current episode index (for logging).
        fps (int): Frames per second for rendering.
        show_grid, show_heatmap, show_rewards: Rendering options.
        full_pitch (bool): Whether to render full pitch or half pitch.
        show_info (bool): Whether to overlay extra info (e.g. reward).
        show_fov (bool): Whether to draw players' field of view.

    Returns:
        float: Cumulative reward summed across all agents.
    """

    # Reset environment
    obs, _ = env.reset()

    # Track rewards and states
    states = []
    cumulative_rewards = {agent: 0.0 for agent in env.agents}
    rewards_per_frame = [] if save_path else None

    # Add initial state (frame 0)
    states.append(env.get_render_state())
    if save_path:
        rewards_per_frame.append(0.0)

    # Termination flags
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}

    # Main evaluation loop
    while not any(terminated.values()) and not any(truncated.values()):
        action_dict = {}

        # Compute one action per agent
        for agent_id in env.agents:
            obs_array = obs[agent_id]
            action, _ = model.predict(obs_array, deterministic=True)
            action_dict[agent_id] = action.squeeze()       # back to 1D

        # Step the environment
        obs, rewards, terminated, truncated, infos = env.step(action_dict)

        # Accumulate rewards
        for agent, r in rewards.items():
            cumulative_rewards[agent] += r

        # Save state snapshot for rendering
        states.append(env.get_render_state())
        if save_path:
            rewards_per_frame.append(sum(rewards.values()))

    # Render the episode if required
    if save_path:
        anim = render_episode_multiAgent(
            states,
            pitch=pitch,
            save_path=save_path,
            fps=fps,
            full_pitch=full_pitch,
            show_grid=show_grid,
            show_heatmap=show_heatmap,
            show_rewards=show_rewards,
            reward_grid=None,
            show_fov=show_fov,
        )
        anim.save(save_path, writer="ffmpeg", fps=fps)

    return sum(cumulative_rewards.values())
