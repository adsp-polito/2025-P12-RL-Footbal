from football_tactical_ai.helpers.visuals import render_episode_singleAgent, render_episode_multiAgent
import torch
from ray.rllib.utils.numpy import convert_to_numpy

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
    show_fov=False,
    show_names=False,     # if True, shows agent IDs above players
    deterministic=True,   # if True, uses greedy actions
    debug=False           # if True, prints debug info
):
    """
    Evaluate a trained RLlib PPO model in a multi-agent environment.

    - Attaccanti usano attacker_policy
    - Difensori e portiere usano defender_policy
    - Episode is rendered to video if save_path is provided.

    Returns:
        float: Total cumulative reward across all agents.
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

        for agent_id in env.agents:
            obs_array = torch.tensor(obs[agent_id], dtype=torch.float32).unsqueeze(0)

            # Policy: attaccanti vs difensori/portiere
            if agent_id.startswith("att"):
                policy_id = "attacker_policy"
            else:
                policy_id = "defender_policy"

            # RLModule per la policy
            module = model.get_module(policy_id)

            with torch.no_grad():
                action_out = module.forward_inference({"obs": obs_array})

            # Usa logits → distribuzione → sample
            logits = action_out["action_dist_inputs"]
            dist_class = module.get_train_action_dist_cls()
            dist = dist_class.from_logits(logits)

            if deterministic:
                # Use the mean action (greedy)
                action = dist.loc
            else:
                # Sample stochastically
                action = dist.sample()

            action = convert_to_numpy(action).squeeze()   # guarantee shape (7,)

            if debug:
                print(f"[{agent_id}] action shape={action.shape}, action={action}")

            action_dict[agent_id] = action

        # Step the environment
        obs, rewards, terminated, truncated, infos = env.step(action_dict)

        # Accumulate rewards
        for agent, r in rewards.items():
            cumulative_rewards[agent] += r

        # Save state snapshot for rendering
        states.append(env.get_render_state())
        if save_path:
            rewards_per_frame.append(sum(rewards.values()))

    # Render episode
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