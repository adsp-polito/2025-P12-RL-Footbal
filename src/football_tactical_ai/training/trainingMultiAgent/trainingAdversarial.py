from football_tactical_ai.helpers.helperTraining import train_Adversarial
import warnings
import ray
import os
import logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["RAY_DEDUP_LOGS"] = "0" 
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)
logging.getLogger("ray.tune").setLevel(logging.ERROR)
logging.getLogger("ray.rllib.utils.deprecation").setLevel(logging.ERROR)
logger = logging.getLogger("gymnasium")
logger.setLevel(logging.ERROR)


ray.shutdown()
ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False, num_gpus=1)

if __name__ == "__main__":
    train_Adversarial(
        scenario="multiagent",
        switch_frequency=500,
        total_cycles=6,
        use_pretrained_init=False,
        attacker_overrides={
            "lr": 1e-4,                  # default 1e-4
            "entropy_coeff": 0.05,       # default 0.05
            "train_batch_size": 1024,
        },
        defender_overrides={
            "lr": 1e-4,                  # default 1e-4
            "entropy_coeff": 0.01,       # default 0.01
            "train_batch_size": 1024,
        },
        
        # Entropy decay settings
        entropy_decay={
            "attacker": {
                "start": 0.05,           # Initial entropy coefficient
                "end": 0.002,            # Final entropy coefficient
                "decay_type": "exponential", # "linear" or "exponential"
            },
            "defender": {
                "start": 0.01,           # Initial entropy coefficient
                "end": 0.0005,            # Final entropy coefficient
                "decay_type": "exponential", # "linear" or "exponential"
            },
        },

        # Opponent pool settings for self-play diversity
        opponent_pool={
            "enabled": True,             # Enable opponent pool sampling
            "latest_prob": 0.5,          # Probability to use latest opponent (vs historical)
            "min_pool_size": 3,          # Min checkpoints before sampling from pool
        },
    )