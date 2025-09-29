from xml.parsers.expat import model
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
    show_names=False,      # If True, shows agent IDs above players
    deterministic=True,    # If True, agents act greedily
    debug=False,           # If True, print per-agent action debug info
    policy_mapping_fn=None # Optional: mapping from agent_id → policy_id (role-based setup)
):
    """
    Evaluate a trained RLlib PPO model in a multi-agent football environment
    (new API stack, compatible with both one-policy-per-agent and role-based).

    Args:
        model: Trained RLlib Algorithm object (with RLModules).
        env: Environment instance (FootballMultiEnv).
        pitch: Pitch instance for rendering.
        save_path (str, optional): Path to save video (mp4).
        episode (int): Episode number (for logging).
        fps (int): Frames per second for rendering.
        deterministic (bool): If True, actions are greedy (argmax or mean).
        debug (bool): If True, print debug info per step.
        policy_mapping_fn (callable, optional): Maps agent_id → policy_id.
            - If None: assumes one policy per agent (policy_id == agent_id).
            - If provided: role-based (e.g. all attackers share same policy).

    Returns:
        dict: Cumulative rewards per agent for this evaluation episode.
    """

    # Reset environment
    obs, _ = env.reset()

    # Initialize reward and rendering trackers
    states = [env.get_render_state()]
    cumulative_rewards = {agent: 0.0 for agent in env.agents}

    # Termination flags (RLlib new API: dicts per agent + "__all__")
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}

    # Evaluation loop
    while not terminated.get("__all__", False) and not truncated.get("__all__", False):
        action_dict = {}

        for agent_id in env.agents:
            # Select policy_id depending on mapping mode
            if policy_mapping_fn:
                policy_id = policy_mapping_fn(agent_id)
            else:
                policy_id = agent_id  # one-policy-per-agent mode

            # Retrieve the correct RLModule
            module = model.get_module(policy_id)

            # Convert obs to tensor with batch dimension
            obs_array = torch.tensor(
                np.array(obs[agent_id], dtype=np.float32).reshape(1, -1),
                dtype=torch.float32
            )

            # Forward pass through RLModule
            with torch.no_grad():
                out = module.forward_inference({"obs": obs_array})

            # Get action distribution
            dist_inputs = out["action_dist_inputs"]
            dist_class = module.get_train_action_dist_cls()
            dist = dist_class.from_logits(dist_inputs)

            # Choose action (greedy vs stochastic)
            if deterministic:
                if hasattr(dist, "loc"):  
                    # Gaussian distribution (continuous actions)
                    action = dist.loc
                else:
                    # Categorical distribution (discrete actions)
                    action = torch.argmax(dist_inputs, dim=-1)
            else:
                action = dist.sample()

            # Convert to numpy for env
            action = np.array(action.cpu().numpy()).flatten()

            if debug:
                print(f"[Episode {episode}] Agent={agent_id}, Policy={policy_id}, Action={action}")

            action_dict[agent_id] = action

        # Step environment
        obs, rewards, terminated, truncated, infos = env.step(action_dict)

        # Update cumulative rewards
        for agent, r in rewards.items():
            cumulative_rewards[agent] += r

        # Save state for rendering
        states.append(env.get_render_state())

    # Render to video if requested
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

    return cumulative_rewards, {}  # {} placeholder for extra stats
