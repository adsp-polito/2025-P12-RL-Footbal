"""
Multi-agent training settings.
"""

from .configMultiAgentEnv import get_config

SCENARIOS: dict[str, dict] = {}

# Base parameters
multiagent_params = {
    "env_class": (
        "football_tactical_ai.env.scenarios.multiAgent."
        "multiAgentEnv:FootballMultiEnv"
    ),
    "seconds_per_episode": 20,
    "fps": 24,
    "episodes": 5000,
    "eval_every": 100,
    "render": {
        "show_grid": False,
        "show_heatmap": False,
        "show_rewards": False,
        "full_pitch": True,
        "show_fov": False,
        "show_names": True,   # if True, shows agent IDs above players
    },
    "paths": {
        "save_model_path": "training/models/multiAgentModel",
        "save_render_dir": "training/renders/multiAgent",
        "plot_path": "training/renders/multiAgent/multiAgentRewards.png",
    },
    "num_envs": 4,
    # Custom environment settings for multi-agent scenario
    "env_settings": {
        "n_attackers": 3,
        "n_defenders": 0,
        "include_goalkeeper": False,
    },
        "rllib": {
        "framework": "torch",
        "lr": 5e-4,
        "lr_schedule": [
                [0, 5e-4],
                [800000, 2e-4],
                [1600000, 1e-4],
                [2400000, 5e-5],
            ],
        "gamma": 0.96,
        "lambda": 0.95,
        "entropy_coeff": 0.02,   
        "train_batch_size": 6000,
        "rollout_fragment_length": 300,
        "minibatch_size": 128,
        "num_epochs": 6,
        "num_workers": 3,
        "num_envs_per_worker": 2,
        "model": {"fcnet_hiddens": [256, 256, 128]},
    },

}

# Build the full scenario dict
multiagent_params["env_config"] = get_config(
    fps=multiagent_params["fps"],
    seconds=multiagent_params["seconds_per_episode"],
    **multiagent_params["env_settings"]
)

SCENARIOS["multiagent"] = multiagent_params
