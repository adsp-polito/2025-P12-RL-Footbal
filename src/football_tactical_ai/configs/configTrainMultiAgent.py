"""
Multi-agent training settings.
This configuration file defines how the multi-agent football environment
is set up and how RLlib should train PPO agents in this scenario.
"""

from .configMultiAgentEnv import get_config  # Import helper to build environment configuration

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
    "seconds_per_episode": 15,

    # Frames per second (simulation runs at 24 steps per second)
    "fps": 24,

    # Total number of training episodes
    "episodes": 3000,

    # Frequency of evaluation in episodes
    "eval_every": 300,

    # Rendering configuration (used in evaluation/visualization)
    "render": {
        "show_grid": False,     # If True, draw spatial reward grid
        "show_heatmap": False,  # If True, render reward heatmap
        "show_rewards": False,  # If True, overlay numeric reward values
        "full_pitch": True,     # If True, render full pitch instead of half
        "show_fov": False,      # If True, display players' field of view
        "show_names": True,     # If True, show agent IDs above players
    },

    # Paths for saving models, renders, and plots
    "paths": {
        "save_model_path": "training/models/multiAgentModel",               # Checkpoint directory
        "save_render_dir": "training/renders/multiAgent",                   # Folder for rendered videos
        "plot_path": "training/renders/multiAgent/multiAgentRewards.png",   # Reward curve output
    },

    # Environment-specific settings for multi-agent scenario
    "env_settings": {
        "n_attackers": 3,            # Number of attackers (Team A)
        "n_defenders": 0,            # Number of defenders (Team B)
        "include_goalkeeper": True, # Whether to include a goalkeeper
        # NOTE: increase defenders/GK here to test larger scenarios (e.g. 2v2, 3v3, 3v2+GK)
    },

    # RLlib PPO configuration parameters
    "rllib": {
        "framework": "torch",  # Use PyTorch backend for PPO

        # Learning Rate Schedule
        # The LR starts higher to encourage fast learning in early stages,
        # then decays progressively to stabilize the policy
        "lr": [
            [0,         2e-4],     # Initial exploration phase
            [180_000,   1e-4],     # Gradual decay
            [360_000,   5e-5],
            [480_000, 2e-5],
            [720_000, 1e-5],
        ],

        #"lr": 2e-4,  # Fixed learning rate
        #"lr_schedule" : [
        #    [0,         2e-4],     # Initial exploration phase
        #    [180_000,   1e-4],     # Gradual decay
        #    [360_000,   5e-5],
        #    [480_000, 2e-5],
        #    [720_000, 1e-5],    
        #],


        # Core RL Parameters
        "gamma": 0.96,          # Discount factor for future rewards → shorter horizon
        "lambda": 0.97,         # GAE smoothing factor → balances bias vs. variance

        # Exploration and Stability
        "entropy_coeff": [
            [0,        0.1],    # Encourage exploration early on
            [180_000,  0.075],     # Decay over time to focus on exploitation
            [360_000,  0.05],      
            [480_000,  0.025],
            [720_000,  0.01],
        ],

        #"entropy_coeff": 0.01,  # Fixed entropy coefficient
        #"entropy_coeff_schedule": [
        #    [0,        0.1],    # Encourage exploration early on
        #    [180_000,  0.075],     # Decay over time to focus on exploitation
        #    [360_000,  0.05],      
        #    [480_000,  0.025],
        #    [720_000,  0.01],
        #],
        

        "clip_param": 0.2,           # PPO clipping parameter for stable updates
        "vf_clip_param": 20.0,       # Clipping for value function updates → avoids large jumps
        "vf_loss_coeff": 1.5,        # Weight of value function loss → higher stabilizes training
        "grad_clip": 1.0,            # Clipping for gradients → prevents exploding gradients
                


        # Training Dynamics
        "train_batch_size": 10_800,       # Number of timesteps per training batch
        "rollout_fragment_length": 360,   # Number of steps per rollout fragment
        "minibatch_size": 512,            # Size of minibatches for SGD
        "num_epochs": 6,                  # Number of passes over each batch of data

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
            #"uses_new_env_api": True,  # True if it is in local, False if in remote
            "uses_new_env_api": True, 
        },
    },

}

# Build the full environment configuration (ensures consistency)
multiagent_params["env_config"] = get_config(
    fps=multiagent_params["fps"],
    seconds=multiagent_params["seconds_per_episode"],
    **multiagent_params["env_settings"]
)

# Register the scenario
SCENARIOS["multiagent"] = multiagent_params