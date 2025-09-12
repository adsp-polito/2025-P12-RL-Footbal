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
    "episodes": 10000,
    "eval_every": 1000,
    "render": {
        "show_grid": False,
        "show_heatmap": False,
        "show_rewards": False,
        "full_pitch": True,
        "show_fov": False,
    },
    "paths": {
        "save_model_path": "training/models/MultiAgentModel",
        "save_render_dir": "training/renders/MultiAgent",
        "plot_path": "training/renders/MultiAgent/MultiAgentRewards.png",
    },
    "num_envs": 4,
    # Custom environment settings (only those you care about)
    "env_settings": {
        "n_attackers": 3,
        "n_defenders": 2,
        "include_goalkeeper": True,
    },
    "rllib": {
        "framework": "torch",
        "lr": 1e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "entropy_coeff": 0.01,
        "train_batch_size": 1600,
        "rollout_fragment_length": "auto",
        "minibatch_size": 256,
        "num_epochs": 5,
        "num_workers": 4,
        "model": {"fcnet_hiddens": [256, 256]},
    },
}

# Build the full scenario dict
multiagent_params["env_config"] = get_config(
    fps=multiagent_params["fps"],
    seconds=multiagent_params["seconds_per_episode"],
    **multiagent_params["env_settings"]
)

SCENARIOS["multiagent"] = multiagent_params
