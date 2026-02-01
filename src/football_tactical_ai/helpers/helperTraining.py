import os
import matplotlib.pyplot as plt
import numpy as np
import importlib
import json
import random
import torch
from tqdm import trange
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from ray.rllib.algorithms.algorithm import Algorithm
from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.configs import configTrainSingleAgent as CFG_SA
from football_tactical_ai.helpers.helperFunctions import ensure_dirs
from football_tactical_ai.helpers.helperEvaluation import evaluate_and_render

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from football_tactical_ai.env.scenarios.multiAgent.multiAgentEnv import FootballMultiEnv
from football_tactical_ai.configs import configTrainMultiAgent as CFG_MA
from football_tactical_ai.configs import configTrainWhatIF as CFG_WI
from football_tactical_ai.helpers.helperEvaluation import evaluate_and_render_multi



import warnings
import ray
import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ray").setLevel(logging.ERROR)

# Reduce RLlib/Ray log verbosity
logger = logging.getLogger("ray")
logger.setLevel(logging.ERROR)

ray.shutdown()
ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False, num_gpus=1)











def train_SingleAgent(scenario="move"):
    """
    Train a PPO agent for a single-agent scenario (move, shot, view),
    using the configuration in configTrainSingleAgent.
    """

    # Load scenario configuration
    cfg = CFG_SA.SCENARIOS[scenario]

    # Load environment class dynamically
    env_path, class_name = cfg["env_class"].split(":")
    module = importlib.import_module(env_path)
    env_class = getattr(module, class_name)

    # Prepare directories
    ensure_dirs(cfg["paths"]["save_render_dir"])
    ensure_dirs(os.path.dirname(cfg["paths"]["save_model_path"]))
    ensure_dirs(cfg["paths"]["rewards_dir"])

    # Core parameters
    fps = cfg["fps"]
    episodes = cfg["episodes"]
    eval_every = cfg["eval_every"]
    max_steps = cfg["max_steps"]
    total_timesteps = episodes * max_steps

    pitch = Pitch()

    # Create environments
    env = Monitor(env_class(pitch=pitch, max_steps=max_steps, fps=fps))
    eval_env = env_class(pitch=pitch, max_steps=max_steps, fps=fps)

    # SB3 PPO
    model = PPO("MlpPolicy", env, **cfg["ppo"])

    # Storage for rewards
    episode_rewards_log = []

    # HEADER
    print("\n" + "=" * 125)
    print("Starting PPO Single-Agent Training".center(125))
    print("=" * 125)

    print(f"{'Scenario:':25} {scenario}")
    print(f"{'Episodes:':25} {episodes}")
    print(f"{'Evaluation every:':25} {eval_every}")
    print(f"{'Seconds per episode:':25} {cfg['seconds_per_episode']}")
    print(f"{'FPS:':25} {fps}")
    print(f"{'Steps per episode:':25} {max_steps}")
    print(f"{'Total timesteps:':25} {total_timesteps}")

    print("\nPPO Training Parameters:")
    for key, val in cfg["ppo"].items():
        print(f"  {key:25} {val}")
    print("=" * 125 + "\n")

    print("Starting training...\n")

    # TRAINING LOOP
    for ep in trange(1, episodes + 1, desc="Episodes Progress"):

        # One episode worth of timesteps
        model.learn(total_timesteps=max_steps, reset_num_timesteps=False)

        # FAST EVALUATION (no render)
        cumulative_reward_quick = evaluate_and_render(
            model=model,
            env=eval_env,
            pitch=pitch,
            save_path=None,
            episode=ep,
            fps=fps,
            eval_mode="fast",
            **cfg["render"],
        )

        episode_rewards_log.append({
            "episode": ep,
            "reward": cumulative_reward_quick
        })

        # FULL EVALUATION (with render)
        if ep % eval_every == 0 or ep in (1, episodes):

            save_render = os.path.join(cfg["paths"]["save_render_dir"],
                                       f"episode_{ep}.mp4")

            cumulative_reward_full = evaluate_and_render(
                model=model,
                env=eval_env,
                pitch=pitch,
                save_path=save_render,
                episode=ep,
                fps=fps,
                eval_mode="full",
                **cfg["render"],
            )

            print(f"[Episode {ep}] Eval reward = {cumulative_reward_full:.4f}")

    # SAVE REWARD LOG
    rewards_path = os.path.join(cfg["paths"]["rewards_dir"], "rewards.json")
    with open(rewards_path, "w") as f:
        json.dump(episode_rewards_log, f, indent=2)

    print(f"\nSaved reward log → {rewards_path}")

    # SAVE MODEL
    model.save(cfg["paths"]["save_model_path"])
    print(f"Model saved → {cfg['paths']['save_model_path']}")
















def env_creator(config):
    """
    Create a FootballMultiEnv instance for RLlib.
    """
    env_cfg = config.get("env_config", config)
    return FootballMultiEnv(env_cfg)



def make_schedule_fn(schedule: list):
    """
    Convert a schedule [[t0, v0], [t1, v1], ...] into a function f(timestep) → value.

    Args:
        schedule (list): List of [timestep, value] pairs.
    
    Returns:
        function: A function that interpolates the value according to the timestep.
    """
    def fn(timestep: int):
        # Linear interpolation between schedule points
        for i in range(len(schedule) - 1):
            t0, v0 = schedule[i]
            t1, v1 = schedule[i + 1]
            if t0 <= timestep < t1:
                alpha = (timestep - t0) / float(t1 - t0)
                return v0 + alpha * (v1 - v0)
        return schedule[-1][1]  # after the last point → last value
    return fn

def train_MultiAgent(scenario: str = "multiagent", role_based: bool = False):
    """
    Train a PPO multi-agent setup with RLlib (new API stack).

    Args:
        scenario (str): Name of the scenario defined in CFG_MA.SCENARIOS.
        role_based (bool): 
            - If True → one shared policy per role (attacker, defender, goalkeeper).
            - If False → one independent policy per agent (e.g. att_1, att_2, def_1, gk_1).
    """

    # Register env with RLlib
    register_env("football_multi_env", env_creator)

    # Load scenario configuration
    cfg = CFG_MA.SCENARIOS[scenario]
    ensure_dirs(cfg["paths"]["save_render_dir"])
    ensure_dirs(os.path.dirname(cfg["paths"]["save_model_path"]))

    # Build a base environment to extract obs/action spaces
    base_env = FootballMultiEnv(cfg["env_config"])

    # Define policies and mapping
    if role_based:
        # Shared policy per role type (attacker, defender, goalkeeper)
        policies = {}

        if base_env.attacker_ids:
            policies["attacker_policy"] = (
                None,
                base_env.observation_space(base_env.attacker_ids[0]),
                base_env.action_space(base_env.attacker_ids[0]),
                {},
            )
        if base_env.defender_ids:
            policies["defender_policy"] = (
                None,
                base_env.observation_space(base_env.defender_ids[0]),
                base_env.action_space(base_env.defender_ids[0]),
                {},
            )
        if base_env.gk_ids:
            policies["goalkeeper_policy"] = (
                None,
                base_env.observation_space(base_env.gk_ids[0]),
                base_env.action_space(base_env.gk_ids[0]),
                {},
            )

        def policy_mapping_fn(agent_id, *args, **kwargs):
            if agent_id.startswith("att"):
                return "attacker_policy"
            if agent_id.startswith("def"):
                return "defender_policy"
            if agent_id.startswith("gk"):
                return "goalkeeper_policy"
            raise ValueError(f"Unknown agent_id: {agent_id}")

    else:
        # Independent policy for each agent
        policies = {
            agent_id: (
                None,
                base_env.observation_space(agent_id),
                base_env.action_space(agent_id),
                {},
            )
            for agent_id in base_env.agents
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id


    # RLlib PPO configuration
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=cfg["rllib"]["model"]["uses_new_env_api"],
            enable_env_runner_and_connector_v2=cfg["rllib"]["model"]["uses_new_env_api"])
        .environment("football_multi_env", env_config=cfg["env_config"])
        .framework(cfg["rllib"]["framework"])
        .training(
            lr=cfg["rllib"]["lr"],
            #lr_schedule=cfg["rllib"].get("lr_schedule", None),
            gamma=cfg["rllib"]["gamma"],
            lambda_=cfg["rllib"]["lambda"],
            entropy_coeff=cfg["rllib"]["entropy_coeff"],
            #entropy_coeff_schedule=cfg["rllib"].get("entropy_coeff_schedule", None),
            clip_param=cfg["rllib"]["clip_param"],
            vf_clip_param=cfg["rllib"].get("vf_clip_param", None),
            vf_loss_coeff=cfg["rllib"].get("vf_loss_coeff", None),
            grad_clip=cfg["rllib"].get("grad_clip", None),
            train_batch_size=cfg["rllib"]["train_batch_size"],
            minibatch_size=cfg["rllib"]["minibatch_size"],
            num_epochs=cfg["rllib"]["num_epochs"],
        )
        .resources(num_gpus=cfg["rllib"].get("num_gpus", 0))
        .rl_module(model_config=cfg["rllib"]["model"])  
        .env_runners(
            num_env_runners=cfg["rllib"]["num_workers"],
            rollout_fragment_length=cfg["rllib"]["rollout_fragment_length"],
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys()),
        )
    )

    # Build PPO algorithm
    algo = config.build_algo()

    # Force policies on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nForcing all RLlib modules to device: {device} ...")

    try:
        multi_module = algo.env_runner.module
        for pid, submodule in multi_module._rl_modules.items():
            try:
                submodule.model.to(device)
                print(f"  → moved '{pid}' to {device}")
            except Exception as e:
                print(f"  → failed to move '{pid}': {e}")
    except Exception as e:
        print(f"Could not move RLlib modules: {e}")

    # LOGGING HEADER
    print("\n" + "=" * 125)
    print("Starting PPO Multi-Agent Training".center(125))
    print("=" * 125)

    print(f"{'Scenario:':25} {scenario}")
    print(f"{'Episodes:':25} {cfg['episodes']}")
    print(f"{'Time per episode:':25} {cfg['seconds_per_episode']} seconds")
    print(f"{'Evaluation every:':25} {cfg['eval_every']} episodes")
    print(f"{'FPS:':25} {cfg['fps']}")
    print(f"{'Steps per episode:':25} {cfg['env_config']['max_steps']}")
    print(f"{'Total timesteps:':25} {cfg['episodes'] * cfg['env_config']['max_steps']}")

    print("\nRLlib Training Parameters:")
    for key, val in cfg["rllib"].items():
        print(f"  {key:25} {val}")
    print("=" * 125 + "\n")


    # List to store (all) episode rewards
    episode_rewards_log = []

    # TRAINING LOOP
    for ep in trange(1, cfg["episodes"] + 1, desc="Episodes Progress"):
        result = algo.train()

        # Always evaluate INSTANTLY without render to get per-agent rewards
        eval_env_quick = FootballMultiEnv(cfg["env_config"])
        cumulative_reward_quick = evaluate_and_render_multi(
            model=algo,
            env=eval_env_quick,
            pitch=Pitch(),
            episode=ep,
            fps=cfg["fps"],
            policy_mapping_fn=policy_mapping_fn,
            eval_mode="fast",      # <-- FAST MODE
        )

        # save episode rewards
        episode_rewards_log.append({
            "episode": ep,
            "rewards": cumulative_reward_quick
        })

        # Evaluate at first, every eval_every, and last episode
        if ep % cfg["eval_every"] == 0 or ep in (1, cfg["episodes"]):
            eval_env = FootballMultiEnv(cfg["env_config"])
            save_render = os.path.join(cfg["paths"]["save_render_dir"], f"episode_{ep}.mp4")

            cumulative_reward = evaluate_and_render_multi(
                model=algo,
                env=eval_env,
                pitch=Pitch(),
                save_path=save_render,
                episode=ep,
                fps=cfg["fps"],
                policy_mapping_fn=policy_mapping_fn,
                eval_mode="full",     # <-- FULL MODE
                **cfg["render"],
            )

            # Log results
            print(f"\n[Episode {ep}] Evaluation results:")
            for aid, rew in cumulative_reward.items():
                role = eval_env.players[aid].get_role()
                print(f"  {aid:10s} ({role}) -> {rew:.2f}")


    # SAVE REWARD LOG
    save_plot_dir = os.path.abspath(cfg["paths"]["rewards_dir"])

    # Ensure directory exists
    ensure_dirs(save_plot_dir)

    rewards_path = os.path.join(save_plot_dir, "rewards.json")
    with open(rewards_path, "w") as f:
        json.dump(episode_rewards_log, f, indent=2)

    print(f"Saved reward log → {rewards_path}")

    # SAVE MODEL
    checkpoint_dir = algo.save(os.path.abspath(cfg["paths"]["save_model_path"]))
    print(f"Model saved at {checkpoint_dir}")

def copy_policies_between_algos(source_algo: Algorithm, target_algo: Algorithm, policy_ids: list):
    """
    Improved: Copy policies using standard Ray RLlib API.
    This ensures we get the LATEST weights from the Learner (GPU), not stale local weights.
    """
    try:
        weights = source_algo.get_weights(policy_ids)
        target_algo.set_weights(weights)
        print(f"Successfully synced policies {policy_ids}.")
            
    except Exception as e:
        print(f"Error copying policies: {e}")
        import traceback
        traceback.print_exc()



def train_Adversarial(
    scenario: str = "multiagent",
    switch_frequency: int = 500,  # Episodes per cycle before switching sides
    total_cycles: int = 6,  # Number of complete cycles
    use_pretrained_init: bool = True,  # Start from pre-trained models
    attacker_overrides: dict = None,
    defender_overrides: dict = None,
    entropy_decay: dict = None,  # Entropy decay settings for attacker/defender
    opponent_pool: dict = None,  # Opponent pool settings for self-play
):
    """
    Adversarial learning training.
    
    Key improvements:
    1. Independent policies (not role-shared)
    2. Better initialization
    3. Optimized for adversarial learning
    
    Args:
        scenario: Environment scenario name
        switch_frequency: Episodes to train one side before switching
        total_cycles: Number of training cycles
        use_pretrained_init: Whether to pre-warm policies
        attacker_overrides: Hyperparameter overrides for attackers
        defender_overrides: Hyperparameter overrides for defenders
    """
    
    register_env("football_multi_env", env_creator)
    
    cfg = CFG_MA.SCENARIOS[scenario]
    ensure_dirs(cfg["paths"]["save_render_dir"])
    ensure_dirs(os.path.dirname(cfg["paths"]["save_model_path"]))
    
    # Create directories
    base_model_dir = os.path.abspath(os.path.dirname(cfg["paths"]["save_model_path"]))
    attacker_model_dir = os.path.join(base_model_dir, "adversarial_attacker")
    defender_model_dir = os.path.join(base_model_dir, "adversarial_defender")
    ensure_dirs(attacker_model_dir)
    ensure_dirs(defender_model_dir)
    
    # Build base environment
    base_env = FootballMultiEnv(cfg["env_config"])
    
    # Use INDEPENDENT policies (not role-shared)
    policies = {}
    for agent_id in base_env.agents:
        policies[agent_id] = (
            None,
            base_env.observation_space(agent_id),
            base_env.action_space(agent_id),
            {},
        )
    
    # Policy mapping: each agent uses its own policy
    def policy_mapping_fn(agent_id, *args, **kwargs):
        return agent_id  # Direct mapping: agent_id -> policy
    
    # Get agent groups
    attacker_ids = base_env.attacker_ids  # ["att_1", "att_2"]
    defender_ids = base_env.defender_ids  # ["def_1"]
    gk_ids = base_env.gk_ids              # ["gk_1"] if exists
    
    print(f"Attackers: {attacker_ids}")
    print(f"Defenders: {defender_ids}")
    print(f"Goalkeepers: {gk_ids}")
    
    # Adversarial-optimized hyperparameters
    attacker_cfg = cfg["rllib"].copy()
    if attacker_overrides:
        attacker_cfg.update(attacker_overrides)
    else:
        # Default: optimized for adversarial learning
        attacker_cfg.update({
            "lr": 1e-4,
            "entropy_coeff": 0.05,
            "train_batch_size": 1024,
        })
    
    defender_cfg = cfg["rllib"].copy()
    if defender_overrides:
        defender_cfg.update(defender_overrides)
    else:
        defender_cfg.update({
            "lr": 1e-4,
            "entropy_coeff": 0.01,
            "train_batch_size": 1024,
        })
    
    # Setup entropy decay
    def compute_entropy(start, end, progress, decay_type="linear"):
        """Compute entropy coefficient based on training progress (0 to 1)."""
        if decay_type == "exponential":
            # Exponential decay: start * (end/start)^progress
            return start * (end / start) ** progress
        else:  # linear
            return start + (end - start) * progress
    
    # Initialize entropy decay settings
    attacker_entropy_cfg = None
    defender_entropy_cfg = None
    if entropy_decay:
        attacker_entropy_cfg = entropy_decay.get("attacker", None)
        defender_entropy_cfg = entropy_decay.get("defender", None)
    
    print("\n" + "=" * 130)
    print("ADVERSARIAL TRAINING (IMPROVED VERSION)".center(130))
    print("=" * 130)
    print(f"Scenario: {scenario}")
    print(f"Total episodes: {total_cycles * switch_frequency}")
    print(f"Switch frequency: {switch_frequency}")
    print(f"Total cycles: {total_cycles}")
    print(f"Use pre-trained model: {use_pretrained_init}")
    if attacker_entropy_cfg:
        print(f"Attacker entropy decay: {attacker_entropy_cfg['start']} -> {attacker_entropy_cfg['end']} ({attacker_entropy_cfg['decay_type']})")
    if defender_entropy_cfg:
        print(f"Defender entropy decay: {defender_entropy_cfg['start']} -> {defender_entropy_cfg['end']} ({defender_entropy_cfg['decay_type']})")
    
    # Initialize opponent pool settings
    use_opponent_pool = opponent_pool is not None and opponent_pool.get("enabled", True)
    latest_prob = opponent_pool.get("latest_prob", 0.5) if opponent_pool else 0.5  # Probability to use latest model
    min_pool_size = opponent_pool.get("min_pool_size", 3) if opponent_pool else 3  # Min checkpoints before sampling
    
    # Opponent pool: stores paths to saved checkpoints
    attacker_pool = []  # Pool of attacker checkpoints (used as opponents for defender training)
    defender_pool = []  # Pool of defender checkpoints (used as opponents for attacker training)
    
    if use_opponent_pool:
        print(f"Opponent pool enabled: latest_prob={latest_prob}, min_pool_size={min_pool_size}")
    else:
        print("Opponent pool disabled: always using latest opponent")
    
    def load_opponent_from_checkpoint(checkpoint_path, target_algo, policy_ids):
        """Load specific policies from a checkpoint into target algorithm."""
        try:
            temp_algo = Algorithm.from_checkpoint(checkpoint_path)
            temp_module = temp_algo.env_runner.module
            
            for pid in policy_ids:
                if pid in temp_module._rl_modules:
                    target_algo.env_runner.module._rl_modules[pid].load_state_dict(
                        temp_module._rl_modules[pid].state_dict()
                    )
            
            temp_algo.stop()
            del temp_algo
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return True
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return False
    
    def sample_opponent_checkpoint(pool, latest_checkpoint, latest_prob, min_pool_size):
        """Sample a checkpoint from pool or use latest based on probability."""
        # If pool is too small, always use latest
        if len(pool) < min_pool_size:
            return latest_checkpoint, "latest (pool too small)"
        
        # Random sampling
        if random.random() < latest_prob:
            return latest_checkpoint, "latest"
        else:
            # Sample from historical checkpoints (excluding the latest one)
            historical = [p for p in pool if p != latest_checkpoint]
            if historical:
                sampled = random.choice(historical)
                cycle_num = sampled.split("cycle_")[-1] if "cycle_" in sampled else "?"
                return sampled, f"historical (cycle {cycle_num})"
            else:
                return latest_checkpoint, "latest (no historical)"
    
    print("=" * 130 + "\n")
    
    # Create attacker algorithm
    attacker_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True)
        .environment("football_multi_env", env_config=cfg["env_config"])
        .framework("torch")
        .training(
            lr=attacker_cfg["lr"],
            gamma=attacker_cfg["gamma"],
            lambda_=attacker_cfg["lambda"],
            entropy_coeff=attacker_cfg["entropy_coeff"],
            clip_param=attacker_cfg["clip_param"],
            train_batch_size=attacker_cfg["train_batch_size"],
            minibatch_size=attacker_cfg["minibatch_size"],
            num_epochs=attacker_cfg["num_epochs"],
        )
        .resources(num_gpus=1)
        .rl_module(model_config=cfg["rllib"]["model"])
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length="auto",
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=attacker_ids,
        )
    )
    attacker_algo = attacker_config.build_algo()
    
    # Create defender algorithm
    defender_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True)
        .environment("football_multi_env", env_config=cfg["env_config"])
        .framework("torch")
        .training(
            lr=defender_cfg["lr"],
            gamma=defender_cfg["gamma"],
            lambda_=defender_cfg["lambda"],
            entropy_coeff=defender_cfg["entropy_coeff"],
            clip_param=defender_cfg["clip_param"],
            train_batch_size=defender_cfg["train_batch_size"],
            minibatch_size=defender_cfg["minibatch_size"],
            num_epochs=defender_cfg["num_epochs"],
        )
        .resources(num_gpus=1)
        .rl_module(model_config=cfg["rllib"]["model"])
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length="auto",
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=defender_ids + gk_ids,
        )
    )
    defender_algo = defender_config.build_algo()
    
    # Force to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Moving algorithms to device: {device}")
    
    for algo in [attacker_algo, defender_algo]:
        multi_module = algo.env_runner.module
        for agent_id, rl_module in multi_module._rl_modules.items():
            try:
                if hasattr(rl_module, "model"):
                    rl_module.model.to(device)
            except:
                pass
    
    # Initialize opponent models
    print("\nInitializing opponent models...")
    
    if use_pretrained_init:
        # Load from pre-trained standard multiagent model if available
        pretrained_path = os.path.abspath(cfg["paths"]["save_model_path"])
        
        if os.path.exists(pretrained_path):
            print(f"Loading pre-trained model from: {pretrained_path}")
            try:
                pretrained_algo = Algorithm.from_checkpoint(pretrained_path)
                pretrained_module = pretrained_algo.env_runner.module
                
                # Copy all policies from pre-trained model to both algos
                for agent_id in base_env.agents:
                    if agent_id in pretrained_module._rl_modules:
                        pretrained_rl_module = pretrained_module._rl_modules[agent_id]
                        
                        # Copy to attacker algo
                        if agent_id in attacker_algo.env_runner.module._rl_modules:
                            # In RLlib 2.9+, RLModule IS the model (no .model attribute)
                            attacker_algo.env_runner.module._rl_modules[agent_id].load_state_dict(
                                pretrained_rl_module.state_dict()
                            )
                        
                        # Copy to defender algo
                        if agent_id in defender_algo.env_runner.module._rl_modules:
                            # In RLlib 2.9+, RLModule IS the model (no .model attribute)
                            defender_algo.env_runner.module._rl_modules[agent_id].load_state_dict(
                                pretrained_rl_module.state_dict()
                            )
                        
                        print(f"Loaded pre-trained weights for {agent_id}")
                
                pretrained_algo.stop()
                del pretrained_algo
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                print("Pre-trained initialization complete")
                
            except Exception as e:
                print(f"Failed to load pre-trained model: {e}")
                print("Falling back to mutual random initialization")
                copy_policies_between_algos(attacker_algo, defender_algo, attacker_ids)
                copy_policies_between_algos(defender_algo, attacker_algo, defender_ids + gk_ids)
        else:
            print(f"Pre-trained model not found at: {pretrained_path}")
            print("Using mutual random initialization")
            copy_policies_between_algos(attacker_algo, defender_algo, attacker_ids)
            copy_policies_between_algos(defender_algo, attacker_algo, defender_ids + gk_ids)
    else:
        # Use mutual random initialization (both sides start equal)
        print("Using mutual random initialization...")
        copy_policies_between_algos(attacker_algo, defender_algo, attacker_ids)
        copy_policies_between_algos(defender_algo, attacker_algo, defender_ids + gk_ids)
        print("Mutual initialization complete")
    
    # Training loop
    episode_log = []
    current_cycle = 1
    current_side = "attacker"
    
    total_episodes = total_cycles * switch_frequency
    
    for ep in trange(1, total_episodes + 1, desc="Adversarial Training"):
        # Determine which side to train this episode
        cycle_position = (ep - 1) % switch_frequency + 1
        
        # Switch sides at cycle boundaries
        if cycle_position == 1 and ep > 1:
            current_side = "defender" if current_side == "attacker" else "attacker"
            current_cycle = (ep - 1) // switch_frequency + 1
            
            print(f"\n[Cycle {current_cycle}] Switching to train {current_side}...")
            
            # Save checkpoint for the side that just finished training
            if current_side == "attacker":
                # We just finished training defenders -> save defender checkpoint
                defender_checkpoint_path = os.path.join(defender_model_dir, f"cycle_{current_cycle - 1}")
                defender_algo.save(defender_checkpoint_path)
                defender_pool.append(defender_checkpoint_path)  # Add to opponent pool
                print(f"Saved defender checkpoint cycle_{current_cycle - 1} (pool size: {len(defender_pool)})")
                
                # Select opponent from pool or use latest
                if use_opponent_pool:
                    selected_checkpoint, selection_type = sample_opponent_checkpoint(
                        defender_pool, defender_checkpoint_path, latest_prob, min_pool_size
                    )
                    if selection_type != "latest" and selection_type != "latest (pool too small)" and selection_type != "latest (no historical)":
                        # Load historical checkpoint as opponent
                        print(f"Loading opponent: {selection_type}")
                        load_opponent_from_checkpoint(selected_checkpoint, attacker_algo, defender_ids + gk_ids)
                    else:
                        # Use latest
                        print(f"Using opponent: {selection_type}")
                        copy_policies_between_algos(defender_algo, attacker_algo, defender_ids + gk_ids)
                else:
                    # Always use latest
                    copy_policies_between_algos(defender_algo, attacker_algo, defender_ids + gk_ids)
            else:
                # We just finished training attackers -> save attacker checkpoint
                attacker_checkpoint_path = os.path.join(attacker_model_dir, f"cycle_{current_cycle - 1}")
                attacker_algo.save(attacker_checkpoint_path)
                attacker_pool.append(attacker_checkpoint_path)  # Add to opponent pool
                print(f"Saved attacker checkpoint cycle_{current_cycle - 1} (pool size: {len(attacker_pool)})")
                
                # Select opponent from pool or use latest
                if use_opponent_pool:
                    selected_checkpoint, selection_type = sample_opponent_checkpoint(
                        attacker_pool, attacker_checkpoint_path, latest_prob, min_pool_size
                    )
                    if selection_type != "latest" and selection_type != "latest (pool too small)" and selection_type != "latest (no historical)":
                        # Load historical checkpoint as opponent
                        print(f"Loading opponent: {selection_type}")
                        load_opponent_from_checkpoint(selected_checkpoint, defender_algo, attacker_ids)
                    else:
                        # Use latest
                        print(f"Using opponent: {selection_type}")
                        copy_policies_between_algos(attacker_algo, defender_algo, attacker_ids)
                else:
                    # Always use latest
                    copy_policies_between_algos(attacker_algo, defender_algo, attacker_ids)
        
        # Update entropy coefficient based on progress
        progress = ep / total_episodes  # Progress from 0 to 1
        
        if current_side == "attacker" and attacker_entropy_cfg:
            new_entropy = compute_entropy(
                attacker_entropy_cfg["start"],
                attacker_entropy_cfg["end"],
                progress,
                attacker_entropy_cfg.get("decay_type", "linear")
            )
            # Update entropy coefficient in learner config
            try:
                attacker_algo.config.training(entropy_coeff=new_entropy)
                # Also update in learner group if available
                if hasattr(attacker_algo, 'learner_group') and attacker_algo.learner_group:
                    attacker_algo.learner_group.set_state({"entropy_coeff": new_entropy})
            except Exception as e:
                pass  # Some RLlib versions may not support dynamic update
            
            if ep % 100 == 0:  # Log every 100 episodes
                print(f"[Ep {ep}] Attacker entropy: {new_entropy:.6f}")
        
        elif current_side == "defender" and defender_entropy_cfg:
            new_entropy = compute_entropy(
                defender_entropy_cfg["start"],
                defender_entropy_cfg["end"],
                progress,
                defender_entropy_cfg.get("decay_type", "linear")
            )
            try:
                defender_algo.config.training(entropy_coeff=new_entropy)
                if hasattr(defender_algo, 'learner_group') and defender_algo.learner_group:
                    defender_algo.learner_group.set_state({"entropy_coeff": new_entropy})
            except Exception as e:
                pass
            
            if ep % 100 == 0:
                print(f"[Ep {ep}] Defender entropy: {new_entropy:.6f}")
        
        # Train
        if current_side == "attacker":
            result = attacker_algo.train()
            current_algo = attacker_algo
            train_ids = attacker_ids
        else:
            result = defender_algo.train()
            current_algo = defender_algo
            train_ids = defender_ids + gk_ids

        # Evaluation only at key points (not every episode)
        # if ep % (switch_frequency // 2) == 0 or ep == total_episodes:
        if ep % (switch_frequency // 2) == 0 or ep == total_episodes:
            # Set model to evaluation mode
            for agent_id, rl_module in current_algo.env_runner.module._rl_modules.items():
                rl_module.eval()
            
            # eval_env = FootballMultiEnv(cfg["env_config"])
            # Create longer evaluation env (30 seconds instead of 10)
            eval_env_config = cfg["env_config"].copy()
            eval_env_config["max_steps"] = cfg["env_config"]["max_steps"] * 3  # 720 frames = 30 seconds
            eval_env = FootballMultiEnv(eval_env_config)

            save_render = os.path.join(
                cfg["paths"]["save_render_dir"],
                f"adversarial_ep{ep}.mp4"
            )
            
            eval_rewards = evaluate_and_render_multi(
                model=current_algo,
                env=eval_env,
                pitch=Pitch(),
                save_path=save_render,
                episode=ep,
                fps=cfg["fps"],
                policy_mapping_fn=policy_mapping_fn,
                eval_mode="full",
                **cfg["render"],
            )
            
            print(f"\n[Ep {ep}, Cycle {current_cycle}] {current_side.upper()} evaluation:")
            for agent_id, reward in eval_rewards.items():
                role = eval_env.players[agent_id].get_role() if agent_id in eval_env.players else "?"
                print(f"  {agent_id} ({role}): {reward:.2f}")
            
            # Set model back to training mode
            for agent_id, rl_module in current_algo.env_runner.module._rl_modules.items():
                rl_module.train()
            
            episode_log.append({
                "episode": ep,
                "cycle": current_cycle,
                "side": current_side,
                "rewards": eval_rewards
            })
    
    # Save checkpoint for the last training cycle
    if current_side == "attacker":
        attacker_checkpoint_path = os.path.join(attacker_model_dir, f"cycle_{total_cycles}")
        attacker_algo.save(attacker_checkpoint_path)
        print(f"\nSaved final attacker checkpoint cycle_{total_cycles}")
    else:
        defender_checkpoint_path = os.path.join(defender_model_dir, f"cycle_{total_cycles}")
        defender_algo.save(defender_checkpoint_path)
        print(f"\nSaved final defender checkpoint cycle_{total_cycles}")
    
    # Final mutual sync so saved checkpoints contain latest policies
    copy_policies_between_algos(attacker_algo, defender_algo, attacker_ids)
    copy_policies_between_algos(defender_algo, attacker_algo, defender_ids + gk_ids)

    # Save final models
    final_att_path = os.path.join(attacker_model_dir, "final")
    final_def_path = os.path.join(defender_model_dir, "final")
    attacker_algo.save(final_att_path)
    defender_algo.save(final_def_path)
    print(f"\nFinal attacker model: {final_att_path}")
    print(f"Final defender model: {final_def_path}")
    
    # Save logs
    log_path = os.path.join(
        cfg["paths"]["rewards_dir"],
        "adversarial_rewards.json"
    )
    ensure_dirs(os.path.dirname(log_path))
    with open(log_path, "w") as f:
        json.dump(episode_log, f, indent=2)
    print(f"Saved log: {log_path}")