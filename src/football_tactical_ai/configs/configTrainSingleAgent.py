"""
Single-agent training settings fully aligned with Stable-Baselines3 PPO.
Minimal, clean and stable hyperparameter sets for MOVE, SHOT and VIEW.
"""

SCENARIOS: dict[str, dict] = {

    #  MOVE SCENARIO
    "move": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "move:OffensiveScenarioMoveSingleAgent"
        ),

        # Episode timing
        "seconds_per_episode": 10,
        "fps": 24,
        "episodes": 15000,
        "eval_every": 1500,

        # Rendering settings
        "render": {
            "show_grid": False,
            "show_heatmap": True,
            "show_rewards": False,
            "full_pitch": True,
            "show_fov": False,
            "show_info": True,
        },

        # Paths
        "paths": {
            "rewards_dir": "src/football_tactical_ai/training/rewards/singleAgentMove",
            "save_model_path": "src/football_tactical_ai/training/models/singleAgentMoveModel",
            "save_render_dir": "src/football_tactical_ai/training/renders/singleAgentMove",
        },

        # PPO hyperparameters
        "ppo": {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,

            "ent_coef": 0.01,
            "clip_range": 0.2,

            "batch_size": 240,
            "n_epochs": 4,

            "seed": 42,
            "verbose": 0,
        },
    },





    #  SHOT SCENARIO
    "shot": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "shot:OffensiveScenarioShotSingleAgent"
        ),

        "seconds_per_episode": 10,
        "fps": 24,
        "episodes": 15000,
        "eval_every": 1500,

        "render": {
            "show_grid": False,
            "show_heatmap": False,
            "show_rewards": False,
            "full_pitch": True,
            "show_fov": False,
            "show_info": True,
        },

        "paths": {
            "rewards_dir": "src/football_tactical_ai/training/rewards/singleAgentShot",
            "save_model_path": "src/football_tactical_ai/training/models/singleAgentShotModel",
            "save_render_dir": "src/football_tactical_ai/training/renders/singleAgentShot",
        },

        # PPO hyperparameters
        "ppo": {
            "learning_rate": 7.5e-5,
            "gamma": 0.99,
            "gae_lambda": 0.95,

            "ent_coef": 0.02,    
            "clip_range": 0.2,

            "batch_size": 120,   # 120 is a clean divisor of the rollout buffer (240 steps), 
                                 # ensuring SB3 can form full mini-batches without truncation 
                                 # and avoiding stability warnings

            "n_epochs": 5,

            "seed": 42,
            "verbose": 0,
        },
    },


    #  VIEW SCENARIO
    "view": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "view:OffensiveScenarioViewSingleAgent"
        ),

        "seconds_per_episode": 10,
        "fps": 24,
        "episodes": 15000,
        "eval_every": 1500,

        "render": {
            "show_grid": False,
            "show_heatmap": False,
            "show_rewards": False,
            "full_pitch": True,
            "show_fov": True,   # VIEW needs FOV
            "show_info": True,
        },

        "paths": {
            "rewards_dir": "src/football_tactical_ai/training/rewards/singleAgentView",
            "save_model_path": "src/football_tactical_ai/training/models/singleAgentViewModel",
            "save_render_dir": "src/football_tactical_ai/training/renders/singleAgentView",
        },

        # PPO hyperparameters
        "ppo": {
            "learning_rate": 5e-5,    # low LR = stability for perceptual tasks
            "gamma": 0.99,
            "gae_lambda": 0.95,

            "ent_coef": 0.03,         # strong exploration
            "clip_range": 0.2,

            "batch_size": 60,
            "n_epochs": 6,

            "seed": 42,
            "verbose": 0,
        },
    },
}


#  Compute n_steps and max_steps for each scenario
for cfg in SCENARIOS.values():
    cfg["ppo"]["n_steps"] = cfg["seconds_per_episode"] * cfg["fps"]
    cfg["max_steps"] = cfg["ppo"]["n_steps"]
