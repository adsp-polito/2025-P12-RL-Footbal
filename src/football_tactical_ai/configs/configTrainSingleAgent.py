"""
Single-agent training settings for the three basic scenarios.

Usage:
    from football_tactical_ai.configs import train_single_agent as CFG
    cfg = CFG.SCENARIOS["move"]           # oppure "shot", "view"
    model = PPO("MlpPolicy", env, **cfg["ppo"])
"""

SCENARIOS: dict[str, dict] = {
    # MOVE scenario configuration
    "move": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "move:OffensiveScenarioMoveSingleAgent"
        ),
        "seconds_per_episode": 10,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 2000,
        "render": {
            "show_grid": False,
            "show_heatmap": False,
            "show_rewards": False,
            "full_pitch": True,
            "show_info": True,
            "show_fov": False
        },
        "paths": {
            "save_model_path": "training/models/singleAgentMoveModel",
            "save_render_dir": "training/renders/singleAgentMove",
            "plot_path": "training/renders/singleAgentMove/SingleAgentMoveRewards.png"
        },
        "ppo": {
            "batch_size": 48,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "device": "cpu",
            "seed": 42,
            "verbose": 0,
        }
    },

    # SHOT scenario configuration
    "shot": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "shot:OffensiveScenarioShotSingleAgent"
        ),
        "seconds_per_episode": 10,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 2000,
        "render": {
            "show_grid": False,
            "show_heatmap": False,
            "show_rewards": False,
            "full_pitch": True,
            "show_info": True,
            "show_fov": False
        },
        "paths": {
            "save_model_path": "training/models/singleAgentShotModel",
            "save_render_dir": "training/renders/singleAgentShot",
            "plot_path": "training/renders/singleAgentShot/SingleAgentShotRewards.png"
        },
        "ppo": {
            "batch_size": 48,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "device": "cpu",
            "seed": 42,
            "verbose": 0,
        }
    },

    # VIEW scenario configuration
    "view": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "view:OffensiveScenarioViewSingleAgent"
        ),
        "seconds_per_episode": 10,
        "fps": 24,
        "episodes": 30000,
        "eval_every": 6000,
        "render": {
            "show_grid": False,
            "show_heatmap": False,
            "show_rewards": False,
            "full_pitch": True,
            "show_info": True,
            "show_fov": True
        },
        "paths": {
            "save_model_path": "training/models/singleAgentViewModel",
            "save_render_dir": "training/renders/singleAgentView",
            "plot_path": "training/renders/singleAgentView/SingleAgentViewRewards.png"
        },
        "ppo": {
            "batch_size": 48,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.02,
            "device": "cpu",
            "seed": 42,
            "verbose": 0,
        }
    },
}

# Automatically compute n_steps and max_steps for each scenario
for cfg in SCENARIOS.values():
    cfg["ppo"]["n_steps"] = cfg["seconds_per_episode"] * cfg["fps"]
    cfg["max_steps"] = cfg["ppo"]["n_steps"]