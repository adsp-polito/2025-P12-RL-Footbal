from xml.parsers.expat import model
from football_tactical_ai.helpers.visuals import render_episode_singleAgent, render_episode_multiAgent
import torch
import numpy as np
import os

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
    show_names=False,       # Show agent IDs above players
    deterministic=True,     # Greedy vs stochastic actions
    debug=True,             # Save per-agent logs
    policy_mapping_fn=None  # Map agent_id → policy_id
):
    """
    Evaluate a trained RLlib PPO model in a multi-agent football environment.

    Runs one full episode with the trained model,
    optionally saving debug logs and rendering to video.
    """

    # Reset environment
    obs, infos = env.reset()

    states = [env.get_render_state()]
    cumulative_rewards = {agent: 0.0 for agent in env.agents}

    # Debug log files
    log_files = {}
    if debug:
        base_dir = f"training/debug_logs/episode_{episode}"
        os.makedirs(base_dir, exist_ok=True)
        for agent in env.agents:
            fname = os.path.join(base_dir, f"{agent}.log")
            log_files[agent] = open(fname, "w", encoding="utf-8")
            log_files[agent].write(f"Debug log for {agent} (Episode {episode})\n")
            log_files[agent].write("=" * 80 + "\n")

    step_count = 0
    terminations, truncations = {}, {}

    # Evaluation loop → stop when ANY agent is terminated or truncated
    while not any(terminations.values()) and not any(truncations.values()):
        step_count += 1
        action_dict = {}

        for agent_id in env.agents:
            # Map agent_id → policy_id
            policy_id = policy_mapping_fn(agent_id) if policy_mapping_fn else agent_id
            module = model.get_module(policy_id)

            # Obs → tensor
            obs_array = torch.tensor(
                np.array(obs[agent_id], dtype=np.float32).reshape(1, -1),
                dtype=torch.float32
            )

            # Forward inference
            with torch.no_grad():
                out = module.forward_inference({"obs": obs_array})

            # Build distribution
            dist_inputs = out.get("action_dist_inputs", out.get("logits"))
            dist_class = module.get_train_action_dist_cls()
            dist = dist_class.from_logits(dist_inputs)

            if deterministic:

                if hasattr(dist, "loc"):  # Gaussian
                    action = torch.tanh(dist.loc)
                else:  # Categorical
                    action = torch.argmax(dist_inputs, dim=-1)
            else:
                action = dist.sample()

            # Convert to numpy
            action = np.array(action.cpu().numpy()).flatten()

            # clip/tanh actions to valid range
            low, high = env.action_space(agent_id).low, env.action_space(agent_id).high
            action = np.clip(action, low, high)

            action_dict[agent_id] = action


        # Step env
        obs, rewards, terminations, truncations, infos = env.step(action_dict)

        # Update rewards + logs
        for agent in env.agents:
            cumulative_rewards[agent] += rewards.get(agent, 0.0)
            if debug and agent in log_files:
                log_files[agent].write(
                    f"[Step {step_count}] "
                    f"Action={action_dict[agent]} | "
                    f"Reward={rewards.get(agent, 0.0):+.3f} | "
                    f"Cum={cumulative_rewards[agent]:+.3f} | "
                    f"Info={infos.get(agent, {})}\n"
                )

        # Save render state
        states.append(env.get_render_state())

    # Close logs
    if debug:
        for f in log_files.values():
            f.close()

    # Render video if requested
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

