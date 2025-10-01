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
    "episodes": 1000,

    # Frequency of evaluation in episodes
    "eval_every": 250,

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
        "include_goalkeeper": False, # Whether to include a goalkeeper
        # NOTE: increase defenders/GK here to test larger scenarios (e.g. 2v2, 3v3, 3v2+GK)
    },

    # RLlib PPO configuration parameters
    "rllib": {
        "framework": "torch",  # Neural network backend

        # Learning rate for optimizer (with schedule for decay)
        "lr": 5e-5,
        "lr_schedule": [
            [0,        5e-5],   
            [500_000,  2e-5],   
            [1_000_000, 1e-5]
        ],

        # Discount factor & GAE
        "gamma": 0.995,
        "lambda": 0.95,

        # Exploration
        #"entropy_coeff": 0.01,
        "entropy_coeff": 0.03,

        # Rollout / Training settings
        #"train_batch_size": 16_000,     # Large enough for stable updates
        "train_batch_size": 8_000,     # Large enough for stable updates
        "rollout_fragment_length": 400, # Steps per worker before sending batch
        "minibatch_size": 256,          # For SGD updates
        "num_epochs": 8,                # Gradient passes per batch

        # Parallelism
        "num_workers": 4,               # Number of rollout workers
        "num_envs_per_worker": 2,       # Each worker runs 2 envs → total 8 envs in parallel

        # Model architecture
        "model": {
            "fcnet_hiddens": [256, 128],  # Hidden layers
            "fcnet_activation": "relu",        # Non-linearity
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