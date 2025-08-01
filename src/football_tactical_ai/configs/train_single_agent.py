"""
Single-agent training settings for the three basic scenarios.

Usage:
    from football_tactical_ai.configs import train_single_agent as CFG
    cfg = CFG.SCENARIOS["move"]           # oppure "shot", "view"
    model = PPO("MlpPolicy", env, **cfg["ppo"])
"""

# Configuration for training a single agent in football tactical AI scenarios
SECONDS_PER_EPISODE = 10
FPS                 = 24
BATCH_SIZE          = 48
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
ENT_COEF            = 0.01


# Prepare the configuration for the scenarios                                           
SCENARIOS: dict[str, dict] = {
    # 1) MOVE
    "move": {
        "env_class": (
            "football_tactical_ai.env.scenarios.single_agent."
            "offensiveScenarioMoveSingleAgent:OffensiveScenarioMoveSingleAgent"
        ),
        "max_steps": SECONDS_PER_EPISODE * FPS,   # 240
        "ppo": {
            "n_steps": SECONDS_PER_EPISODE * FPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": 1e-3,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "ent_coef": ENT_COEF,
            "device": "cpu",
            "seed": 42,
            "verbose": 0,
        },
        "eval_every": 200,
    },

    # 2) SHOT
    "shot": {
        "env_class": (
            "football_tactical_ai.env.scenarios.single_agent."
            "offensiveScenarioShotSingleAgent:OffensiveScenarioShotSingleAgent"
        ),
        "max_steps": SECONDS_PER_EPISODE * FPS,
        "ppo": {
            "n_steps": SECONDS_PER_EPISODE * FPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": 1e-3,      
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "ent_coef": ENT_COEF,
            "device": "cpu",
            "seed": 42,
            "verbose": 0,
        },
        "eval_every": 1000,
    },

    # 3) VIEW
    "view": {
        "env_class": (
            "football_tactical_ai.env.scenarios.single_agent."
            "offensiveScenarioViewSingleAgent:OffensiveScenarioViewSingleAgent"
        ),
        "max_steps": SECONDS_PER_EPISODE * FPS,
        "ppo": {
            "n_steps": SECONDS_PER_EPISODE * FPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": 1e-3,      
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "ent_coef": ENT_COEF,
            "device": "cpu",
            "seed": None,               # None for random seed
            "verbose": 0,
        },
        "eval_every": 3000,
    },
}

