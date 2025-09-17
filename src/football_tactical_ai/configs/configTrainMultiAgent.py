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
    "episodes": 1000,
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
        "save_model_path": "training/models/MultiAgentModel",
        "save_render_dir": "training/renders/MultiAgent",
        "plot_path": "training/renders/MultiAgent/MultiAgentRewards.png",
    },
    "num_envs": 4,
    # Custom environment settings for multi-agent scenario
    "env_settings": {
        "n_attackers": 3,
        "n_defenders": 2,
        "include_goalkeeper": True,
    },
    "rllib": {
        "framework": "torch",
        "lr": 1e-4,
        "lr_schedule": [[0, 1e-4], [240000, 1e-5]],
        "gamma": 0.99,
        "lambda": 0.95,
        "entropy_coeff": 0.01,
        "train_batch_size": 4000,  
        "rollout_fragment_length": 200,    
        "minibatch_size": 256,
        "num_epochs": 5,
        "num_workers": 1,
        "num_envs_per_worker": 1,
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
