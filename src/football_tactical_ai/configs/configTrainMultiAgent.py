"""
Multi-agent training settings for football tactical AI

This configuration file defines how the multi-agent football environment
is set up and how RLlib should train PPO agents in this scenario
"""

# Import helper to build environment configuration
from .configMultiAgentEnv import get_config  

# Dictionary to store different training scenarios
SCENARIOS: dict[str, dict] = {}

# Base parameters for the "multiagent" training scenario
multiagent_params = {
    # Path to the environment class that RLlib will use
    "env_class": (
        "football_tactical_ai.env.scenarios.multiAgent."
        "multiAgentEnv:FootballMultiEnv"
    ),

    # Episode duration in real-world seconds
    "seconds_per_episode": 10,

    # Frames per second (simulation runs at 24 steps per second)
    "fps": 24,

    # Total number of training episodes
    "episodes": 3000,

    # Frequency of evaluation in episodes
    "eval_every": 300,

    # Rendering configuration for training and evaluation
    "render": {
        "show_grid": False,     
        "show_heatmap": False,  
        "show_rewards": False,  
        "full_pitch": True,     
        "show_fov": False,      
        "show_names": True,     
    },

    # path settings for saving outputs
    "paths": {
        "rewards_dir": "src/football_tactical_ai/training/rewards/multiAgent",              
        "save_model_path": "src/football_tactical_ai/training/models/multiAgentModel",               
        "save_render_dir": "src/football_tactical_ai/training/renders/multiAgent"
    },

    # Environment-specific settings for multi-agent scenario
    # NOTE: increase defenders/GK here to test larger scenarios (e.g. 2v2, 3v3, 3v2+GK)
    "env_settings": {
        "n_attackers": 2,            # Number of attackers (Team A)
        "n_defenders": 1,            # Number of defenders (Team B)
        "include_goalkeeper": False,  # Whether to include a goalkeeper
    },

    # RLlib PPO configuration parameters
    "rllib": {
        "framework": "torch",  # Use PyTorch backend for PPO

        # Learning rate
        "lr": 2e-4,

        # Learning Rate Schedule
        #"lr": [
        #    [0,         2e-4],     
        #   [180_000,   1e-4],     
        #    [360_000,   5e-5],
        #    [480_000, 2e-5],
        #    [720_000, 1e-5],
        #],

        # Core RL Parameters
        "gamma": 0.95,          # Discount factor for future rewards
        "lambda": 0.97,         # GAE parameter for advantage estimation


        # Exploratrion coefficient
        "entropy_coeff": 0.02,

        # Exploration coefficient schedule (can be a fixed float or a schedule)
        #"entropy_coeff": [
        #    [0,        0.1],    
        #    [180_000,  0.075],     
        #    [360_000,  0.05],      
        #    [480_000,  0.025],
        #    [720_000,  0.001],
        #],
        

        "clip_param": 0.2,           # PPO clipping parameter for stable updates
        "vf_clip_param": 20.0,       # Value function clipping to stabilize training
        "vf_loss_coeff": 1.0,        # Weight of value function loss in total loss
        "grad_clip": 1.0,            # Gradient clipping to prevent exploding gradients
                

        # Training Dynamics
        "train_batch_size": 2048,            # Number of timesteps per training batch
        "rollout_fragment_length": "auto",   # Number of steps per rollout fragment
        "minibatch_size": 256,               # Size of minibatches for SGD
        "num_epochs": 3,                     # Number of passes over each batch of data

        # Parallelism
        # Each worker simulates multiple environments in parallel (if in local ==> 3 is ok)
        "num_workers": 0, 
        "num_envs_per_worker": 1,

        # Use GPUs if available (set to 0 if none available)
        "num_gpus": 1,  # Adjust based on your hardware

        # Model Architecture
        "model": {
            "fcnet_hiddens": [256, 128, 64],
            "fcnet_activation": "relu",
            "uses_new_env_api": True,   # True to use the new env API
            "custom_action_dist": "mixed_gauss_bernoulli", # Custom action distribution

        },
    },

}

# Build the full environment configuration
multiagent_params["env_config"] = get_config(
    fps=multiagent_params["fps"],
    seconds=multiagent_params["seconds_per_episode"],
    **multiagent_params["env_settings"]
)

# Register the scenario
SCENARIOS["multiagent"] = multiagent_params