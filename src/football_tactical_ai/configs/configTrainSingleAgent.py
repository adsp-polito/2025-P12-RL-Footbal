"""
Single-agent training settings fully aligned with Stable-Baselines3 PPO
Hyperparameter setting for MOVE, SHOT and VIEW scenarios
"""

SCENARIOS: dict[str, dict] = {

    # MOVE SCENARIO
    "move": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "move:OffensiveScenarioMoveSingleAgent"
        ),

        # Episode timing
        "seconds_per_episode": 15,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 1000,

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
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,

            "ent_coef": 0.01,       # Less exploration, dense rewards
            "clip_range": 0.2,

            "batch_size": 360,      # 360 is a clean divisor of the rollout buffer (360 steps),
                                    # ensuring SB3 can form full mini-batches without truncation
                                    # and avoiding stability warnings

            "n_epochs": 4,          # moderate epochs, rapid learning

            "seed": 42,
            "verbose": 0,
        },
    },
 
     # FAST PLAYER MOVE SCENARIO
    "move_fast": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "move:OffensiveScenarioMoveSingleAgent"
        ),

        "env_kwargs": {                               # different from MOVE basic
            "attacker_speed": 0.85
        },

        # Episode timing
        "seconds_per_episode": 15,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 1000,

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
            "rewards_dir": "src/football_tactical_ai/training/rewards/singleAgentMoveFast",
            "save_model_path": "src/football_tactical_ai/training/models/singleAgentMoveFastModel",
            "save_render_dir": "src/football_tactical_ai/training/renders/singleAgentMoveFast",
        },

        # PPO hyperparameters
        "ppo": {
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,

            "ent_coef": 0.01,       
            "clip_range": 0.2,

            "batch_size": 360,      
                                    
                                    

            "n_epochs": 4,          

            "seed": 42,
            "verbose": 0,
        },
    },

     #SLOW PLAYER MOVE SCENARIO
     "move_slow": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "move:OffensiveScenarioMoveSingleAgent"
        ),

        "env_kwargs": {
            "attacker_speed": 0.45
        },

        # Episode timing
        "seconds_per_episode": 15,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 1000,

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
            "rewards_dir": "src/football_tactical_ai/training/rewards/singleAgentMoveSlow",
            "save_model_path": "src/football_tactical_ai/training/models/singleAgentMoveSlowModel",
            "save_render_dir": "src/football_tactical_ai/training/renders/singleAgentMoveSlow",
        },

        # PPO hyperparameters
        "ppo": {
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,

            "ent_coef": 0.01,       
            "clip_range": 0.2,

            "batch_size": 360,      
                                    
                                    

            "n_epochs": 4,          

            "seed": 42,
            "verbose": 0,
        },
    },




    # SHOT SCENARIO
    "shot": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "shot:OffensiveScenarioShotSingleAgent"
        ),

        "seconds_per_episode": 15,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 1000,

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
            "learning_rate": 2e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,

            "ent_coef": 0.03,    
            "clip_range": 0.2,

            "batch_size": 360,   
            "n_epochs": 6,
            "seed": 42,
            "verbose": 0,
        },
    },

    # SHOT WEAK PLAYER SCENARIO
    "shot_weak": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "shot:OffensiveScenarioShotWeakSingleAgent"
        ),

        "seconds_per_episode": 15,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 1000,

        "render": {
            "show_grid": False,
            "show_heatmap": False,
            "show_rewards": False,
            "full_pitch": True,
            "show_fov": False,
            "show_info": True,
        },

        "paths": {
            "rewards_dir": "src/football_tactical_ai/training/rewards/singleAgentShotWeak",
            "save_model_path": "src/football_tactical_ai/training/models/singleAgentShotWeakModel",
            "save_render_dir": "src/football_tactical_ai/training/renders/singleAgentShotWeak",
        },

        # PPO hyperparameters
        "ppo": {
            "learning_rate": 2e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,

            "ent_coef": 0.03,    
            "clip_range": 0.2,

            "batch_size": 360,   
            "n_epochs": 6,
            "seed": 42,
            "verbose": 0,
        },
    },
    

    # SHOT NORMAL PLAYER SCENARIO
    "shot_normal": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "shot:OffensiveScenarioShotNormalSingleAgent"
        ),

        "seconds_per_episode": 15,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 1000,

        "render": {
            "show_grid": False,
            "show_heatmap": False,
            "show_rewards": False,
            "full_pitch": True,
            "show_fov": False,
            "show_info": True,
        },

        "paths": {
            "rewards_dir": "src/football_tactical_ai/training/rewards/singleAgentShotNormal",
            "save_model_path": "src/football_tactical_ai/training/models/singleAgentShotNormalModel",
            "save_render_dir": "src/football_tactical_ai/training/renders/singleAgentShotNormal",
        },

        # PPO hyperparameters
        "ppo": {
            "learning_rate": 2e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,

            "ent_coef": 0.03,    
            "clip_range": 0.2,

            "batch_size": 360,   
            "n_epochs": 6,
            "seed": 42,
            "verbose": 0,
        },
    },


    # SHOT STRONG PLAYER SCENARIO
    "shot_strong": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "shot:OffensiveScenarioShotStrongSingleAgent"
        ),

        "seconds_per_episode": 15,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 1000,

        "render": {
            "show_grid": False,
            "show_heatmap": False,
            "show_rewards": False,
            "full_pitch": True,
            "show_fov": False,
            "show_info": True,
        },

        "paths": {
            "rewards_dir": "src/football_tactical_ai/training/rewards/singleAgentShotStrong",
            "save_model_path": "src/football_tactical_ai/training/models/singleAgentShotStrongModel",
            "save_render_dir": "src/football_tactical_ai/training/renders/singleAgentShotStrong",
        },

        # PPO hyperparameters
        "ppo": {
            "learning_rate": 2e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,

            "ent_coef": 0.03,    
            "clip_range": 0.2,

            "batch_size": 360,   
            "n_epochs": 6,
            "seed": 42,
            "verbose": 0,
        },
    },




    # VIEW SCENARIO
    "view": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "view:OffensiveScenarioViewSingleAgent"
        ),

        "seconds_per_episode": 15,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 1000,

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
            "learning_rate": 1e-4,    # low LR = stability for perceptual tasks
            "gamma": 0.995,
            "gae_lambda": 0.97,

            "ent_coef": 0.07,         # strong exploration
            "clip_range": 0.2,

            "batch_size": 360,     

            "n_epochs": 8,

            "seed": 42,
            "verbose": 0,
        },
    },
}


# Compute n_steps and max_steps for each scenario
for cfg in SCENARIOS.values():
    cfg["ppo"]["n_steps"] = cfg["seconds_per_episode"] * cfg["fps"]
    cfg["max_steps"] = cfg["ppo"]["n_steps"]