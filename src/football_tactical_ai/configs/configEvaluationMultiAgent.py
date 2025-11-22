"""
Multi-Agent Evaluation Configuration File

Defines evaluation settings for 2v1 scenarios with two attackers and one defender.
Can be further customized to add more scenarios or modify existing ones.
"""

# COMMON SETTINGS — 10 SECONDS PER EPISODE
COMMON = {
    "seconds": 10,
    "fps": 24,
    "max_steps": 10 * 24       # = 240 steps
}

# MULTI-AGENT 2 VS 1 SCENARIO
MULTI_2V1 = {

    # Path to trained model
    "model_path": "src/football_tactical_ai/training/models/multiAgent_2v1_Model.zip",

    # Output directories
    "save_video_dir": "src/football_tactical_ai/evaluation/results/videos/multiAgent",
    "save_logs_dir":  "src/football_tactical_ai/evaluation/results/logs/multiAgent",

    # Rendering settings
    "render": {
        "show_grid": False,
        "show_heatmap": False,
        "show_rewards": False,
        "full_pitch": True,
        "show_fov": False,
        "show_names": True,
    },

    # TEST CASES (Start positions in METERS)
    # NOTE: attackers_start has TWO positions: A1, A2

    "test_cases": [

        {
            "name": "center",
            "attackers_start": [
                (65, 40),   # A1 central 
                (60, 50),   # A2 more inside on the left
            ],
            "defender_start": (100, 40),
        },

        {
            "name": "left",
            "attackers_start": [
                (60, 30),   # A1 on the left channel
                (70, 40),   # A2 more inside
            ],
            "defender_start": (100, 40),
        },

        {
            "name": "right",
            "attackers_start": [
                (60, 55),   # A1 on the right channel
                (70, 45),   # A2 more inside
            ],
            "defender_start": (100, 40),
        },

        {
            "name": "deepStart",
            "attackers_start": [
                (50, 35),   # A1 very deep
                (55, 45),   # A2 slightly more advanced
            ],
            "defender_start": (100, 40),
        },
    ],
}


# GLOBAL DICT OF SCENARIOS
SCENARIOS_MULTI = {
    "2v1": MULTI_2V1,
}
