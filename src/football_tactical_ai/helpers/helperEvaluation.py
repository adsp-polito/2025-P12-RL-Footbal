from football_tactical_ai.helpers.visuals import render_episode_singleAgent, render_episode_multiAgent
import torch
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
import torch
import numpy as np

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
    show_fov=False,
    show_names=False,     # if True, shows agent IDs above players
    deterministic=True,   # if True, use greedy actions (mean/argmax)
    debug=False           # if True, print debug info
):
    """
    Evaluate a trained RLlib PPO model in a multi-agent football environment (new RLlib API stack).

    This version assumes a **single shared policy** ("shared_policy")
    across all agents (attackers, defenders, GK). Role-specific behavior
    comes from the one-hot role embedding in the observation space.

    Args:
        model: Trained RLlib Algorithm object (with RLModules).
        env: Environment instance (FootballMultiEnv).
        pitch: Pitch instance for rendering.
        save_path (str, optional): Path to save video (mp4).
        episode (int): Episode number (for logging).
        fps (int): Frames per second for rendering.
        deterministic (bool): If True, actions are greedy (mean/argmax).
        debug (bool): If True, print per-agent action debug info.

    Returns:
        dict: Cumulative rewards per agent for this evaluation episode.
    """

    # Reset environment to initial state
    obs, _ = env.reset()

    # Track rewards and environment states for rendering
    states = [env.get_render_state()]   # first frame
    cumulative_rewards = {agent: 0.0 for agent in env.agents}
    rewards_per_frame = [0.0] if save_path else None

    # Initialize termination flags
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}

    # Main evaluation loop
    while not any(terminated.values()) and not any(truncated.values()):
        action_dict = {}

        for agent_id in env.agents:
            # Convert observation to PyTorch tensor
            obs_array = torch.tensor(obs[agent_id], dtype=torch.float32).unsqueeze(0)

            # Always use the shared policy module
            module = model.get_module("shared_policy")

            # Forward pass (inference mode)
            with torch.no_grad():
                action_out = module.forward_inference({"obs": obs_array})

            # Action distribution inputs (logits for continuous/discrete actions)
            logits = action_out["action_dist_inputs"]

            # Build action distribution from logits
            dist_class = module.get_train_action_dist_cls()
            dist = dist_class.from_logits(logits)

            # Greedy or stochastic action selection
            if deterministic:
                if hasattr(dist, "loc"):   # Gaussian distribution (continuous)
                    action = dist.loc
                else:                      # Discrete distribution
                    action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()

            # Convert to numpy, enforce shape (7,) for continuous actions
            action = np.array(action.cpu().numpy()).squeeze()

            if debug:
                print(f"[{agent_id}] action={action}")

            action_dict[agent_id] = action

        # Step the environment
        obs, rewards, terminated, truncated, infos = env.step(action_dict)

        # Accumulate per-agent rewards
        for agent, r in rewards.items():
            cumulative_rewards[agent] += r

        # Save state for rendering
        states.append(env.get_render_state())
        if save_path:
            rewards_per_frame.append(sum(rewards.values()))

    # Render to video if path provided
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
            show_names=show_names
        )
        anim.save(save_path, writer="ffmpeg", fps=fps)

    return cumulative_rewards
