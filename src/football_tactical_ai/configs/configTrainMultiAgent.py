"""
Multi-agent training settings.

Usage:
    from football_tactical_ai.configs import train_multi_agent as CFG
    cfg = CFG.SCENARIOS["multiagent"]
    model = PPO("MlpPolicy", env, **cfg["ppo"])
"""

from .configMultiAgentEnv import get_config

SCENARIOS: dict[str, dict] = {
    "multiagent": {
        "env_class": (
            "football_tactical_ai.env.scenarios.multiAgent."
            "multiAgentEnv:FootballMultiEnv"
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
            "show_fov": False
        },
        "paths": {
            "save_model_path": "training/models/MultiAgentModel",
            "save_render_dir": "training/renders/MultiAgent",
            "plot_path": "training/renders/MultiAgent/MultiAgentRewards.png"
        },
        "env_config": get_config(
            fps=24,
            seconds=10,
            n_attackers=3,
            n_defenders=2,
            include_goalkeeper=True
        ),
        "num_envs": 4,

        # RLlib config
        "rllib": {
            "framework": "torch",
            "lr": 1e-3,
            "gamma": 0.99,
            "lambda": 0.95,
            "entropy_coeff": 0.01,
            "train_batch_size": 4000,
            "rollout_fragment_length": 200,
            "minibatch_size": 256,
            "num_epochs": 10,
            "num_workers": 4,
            "model": {"fcnet_hiddens": [256, 256]}
        }
    }
}