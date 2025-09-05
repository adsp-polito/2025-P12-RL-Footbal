"""
Multi-agent training settings.

Usage:
    from football_tactical_ai.configs import train_multi_agent as CFG
    cfg = CFG.SCENARIOS["multiagent"]
    model = PPO("MlpPolicy", env, **cfg["ppo"])
"""

from .configMultiAgentEnv import get_config

SCENARIOS: dict[str, dict] = {
    # MULTIAGENT scenario configuration
    "multiagent": {
        "env_class": (
            "football_tactical_ai.env.scenarios.multiAgent."
            "multiAgentEnv:FootballMultiEnv"
        ),
        "seconds_per_episode": 10,
        "fps": 24,
        "episodes": 1000,
        "eval_every": 50,
        "render": {
            "show_grid": False,
            "show_heatmap": False,
            "show_rewards": False,
            "full_pitch": True,
            "show_fov": False
        },
        "paths": {
            "save_model_path": "training/models/MultiAgentModel",
            "save_render_dir": "training/renders/MultiAgent",
            "plot_path": "training/renders/MultiAgent/MultiAgentRewards.png"
        },
        "ppo": {
            "batch_size": 128,          # larger batch size respect to single agent because of multiple agents
            "learning_rate": 5e-4,      # slightly lower for stability
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "device": "cpu",
            "seed": None,
            "verbose": 0,
        },

         "env_config": get_config(
            fps=24,
            seconds=10,
            n_attackers=3,
            n_defenders=2,
            include_goalkeeper=True
        ),

        "num_envs": 4,  # parallel environments
    }
}

# Automatically compute n_steps and max_steps for the scenario
for cfg in SCENARIOS.values():
    cfg["ppo"]["n_steps"] = cfg["seconds_per_episode"] * cfg["fps"]
    cfg["max_steps"] = cfg["ppo"]["n_steps"]
