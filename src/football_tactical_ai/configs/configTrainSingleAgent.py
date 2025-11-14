"""
Single-agent training settings fully aligned with the multi-agent configuration style,
with optimized hyperparameters for stable learning.
"""

SCENARIOS: dict[str, dict] = {

    # MOVE scenario
    "move": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "move:OffensiveScenarioMoveSingleAgent"
        ),

        # Episode parameters
        "seconds_per_episode": 10,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 500,

        # Rendering
        "render": {
            "show_grid": False,
            "show_heatmap": False,
            "show_rewards": False,
            "full_pitch": True,
            "show_fov": False,
            "show_names": False,
        },

        # Paths
        "paths": {
            "rewards_dir": "training/plots/singleAgentMove",
            "save_model_path": "training/models/singleAgentMoveModel",
            "save_render_dir": "training/renders/singleAgentMove",
            "plot_path": "training/plots/singleAgentMove/SingleAgentMoveRewards.png",
        },

        # PPO hyperparameters
        "ppo": {
            # Learning dynamics
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "lambda": 0.95,

            # Entropy schedule
            "entropy_coeff": [
                [0,       0.015],
                [200000,  0.010],
                [400000,  0.005],
                [600000,  0.003],
            ],

            # PPO stability
            "clip_param": 0.2,
            "vf_clip_param": 20.0,
            "vf_loss_coeff": 1.5,
            "grad_clip": 1.0,

            # Training batches
            "train_batch_size": 2048,
            "rollout_fragment_length": 128,
            "minibatch_size": 256,
            "num_epochs": 3,

            # Parallelism
            "num_workers": 0,
            "num_envs_per_worker": 1,
            "num_gpus": 1,

            # Model architecture
            "model": {
                "fcnet_hiddens": [256, 128, 64],
                "fcnet_activation": "relu",
                "uses_new_env_api": True,
            },

            "seed": 42,
            "device": "cpu",
            "verbose": 0,
        },
    },





    # SHOT scenario
    "shot": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "shot:OffensiveScenarioShotSingleAgent"
        ),

        "seconds_per_episode": 10,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 500,

        "render": {
            "show_grid": False,
            "show_heatmap": False,
            "show_rewards": False,
            "full_pitch": True,
            "show_fov": False,
            "show_names": False,
        },

        "paths": {
            "rewards_dir": "training/plots/singleAgentShot",
            "save_model_path": "training/models/singleAgentShotModel",
            "save_render_dir": "training/renders/singleAgentShot",
            "plot_path": "training/plots/singleAgentShot/SingleAgentShotRewards.png",
        },

        # PPO hyperparameters
        "ppo": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "lambda": 0.95,

            # Entropy schedule
            "entropy_coeff": [
                [0,       0.015],
                [200000,  0.010],
                [400000,  0.005],
                [600000,  0.003],
            ],

            "clip_param": 0.2,
            "vf_clip_param": 20.0,
            "vf_loss_coeff": 1.5,
            "grad_clip": 1.0,

            "train_batch_size": 2048,
            "rollout_fragment_length": 128,
            "minibatch_size": 256,
            "num_epochs": 3,

            "num_workers": 0,
            "num_envs_per_worker": 1,
            "num_gpus": 1,

            "model": {
                "fcnet_hiddens": [256, 128, 64],
                "fcnet_activation": "relu",
                "uses_new_env_api": True,
            },

            "seed": 42,
            "device": "cpu",
            "verbose": 0,
        },
    },



    # VIEW scenario
    "view": {
        "env_class": (
            "football_tactical_ai.env.scenarios.singleAgent."
            "view:OffensiveScenarioViewSingleAgent"
        ),

        "seconds_per_episode": 10,
        "fps": 24,
        "episodes": 10000,
        "eval_every": 500,

        "render": {
            "show_grid": False,
            "show_heatmap": False,
            "show_rewards": False,
            "full_pitch": True,
            "show_fov": True,
            "show_names": False,
        },

        "paths": {
            "rewards_dir": "training/plots/singleAgentView",
            "save_model_path": "training/models/singleAgentViewModel",
            "save_render_dir": "training/renders/singleAgentView",
            "plot_path": "training/plots/singleAgentView/SingleAgentViewRewards.png",
        },

        # PPO hyperparameters
        "ppo": {
            "learning_rate": 1e-4,   # lower LR helps complex perception tasks
            "gamma": 0.99,
            "lambda": 0.95,

            # VIEW needs more exploration (entropy is higher)
            "entropy_coeff": [
                [0,       0.02],
                [200000,  0.015],
                [400000,  0.010],
                [600000,  0.005],
            ],

            "clip_param": 0.2,
            "vf_clip_param": 20.0,
            "vf_loss_coeff": 1.5,
            "grad_clip": 1.0,

            "train_batch_size": 2048,
            "rollout_fragment_length": 128,
            "minibatch_size": 256,
            "num_epochs": 3,

            "num_workers": 0,
            "num_envs_per_worker": 1,
            "num_gpus": 1,

            "model": {
                "fcnet_hiddens": [256, 128, 64],
                "fcnet_activation": "relu",
                "uses_new_env_api": True,
            },

            "seed": 42,
            "device": "cpu",
            "verbose": 0,
        },
    },
}

# Compute max_steps and n_steps
for cfg in SCENARIOS.values():
    cfg["ppo"]["n_steps"] = cfg["seconds_per_episode"] * cfg["fps"]
    cfg["max_steps"] = cfg["ppo"]["n_steps"]
