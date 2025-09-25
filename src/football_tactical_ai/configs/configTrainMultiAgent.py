"""
Multi-agent training settings
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
    "seconds_per_episode": 20,

    # Frames per second (simulation runs at 24 steps per second)
    "fps": 24,

    # Total number of training episodes
    "episodes": 5000,

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

    # Number of parallel environments used during training
    "num_envs": 4,

    # Environment-specific settings for multi-agent scenario
    "env_settings": {
        "n_attackers": 3,           # Number of attackers (Team A)
        "n_defenders": 0,           # Number of defenders (Team B)
        "include_goalkeeper": False # Whether to include a goalkeeper
    },

    # RLlib PPO configuration parameters
    "rllib": {
        "framework": "torch",    # Neural network backend (PyTorch)

        # Learning rate for optimizer
        "lr": 5e-4,

        # Learning rate schedule: defines how lr decreases as training progresses
        # Format: [global_timestep, lr_value]
        "lr_schedule": [
            [0,        5e-4],   # At step 0 → 0.0005
            [800_000,  2e-4],   # At ~1/3 of training (800k steps) → 0.0002
            [1_600_000, 1e-4],  # At ~2/3 of training (1.6M steps) → 0.0001
            [2_400_000, 5e-5],  # At final step (2.4M steps) → 0.00005
        ],

        # Discount factor for future rewards (high = long-term focus)
        "gamma": 0.99,

        # GAE (Generalized Advantage Estimation) parameter
        "lambda": 0.95,

        # Entropy coefficient (encourages exploration)
        "entropy_coeff": 0.03,

        # Training batch size (number of collected samples per update)
        "train_batch_size": 5000,

        # Number of steps in each rollout fragment
        "rollout_fragment_length": 300,

        # Size of mini-batches used for gradient updates
        "minibatch_size": 128,

        # Number of epoches over each training batch
        "num_epochs": 6,

        # Number of parallel workers collecting rollouts
        "num_workers": 3,

        # Number of environments per worker
        "num_envs_per_worker": 2,

        # Neural network architecture for policy/value function
        "model": {
            "fcnet_hiddens": [512, 256],  # Two hidden layers with 512 and 256 neurons
            "fcnet_activation": "relu"    # Activation function for hidden layers
        },
    },
}

# Build the full environment configuration by calling helper function
# This ensures consistency in how episodes, fps, and players are initialized
multiagent_params["env_config"] = get_config(
    fps=multiagent_params["fps"],
    seconds=multiagent_params["seconds_per_episode"],
    **multiagent_params["env_settings"]
)

# Register the scenario in the SCENARIOS dictionary
SCENARIOS["multiagent"] = multiagent_params
