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
    # You need to change this to your absolute path
    "model_path": "/Users/manuelemustari/Desktop/Università/Politecnico di Torino/2° year/2° period/Football-Tactical-AI/src/football_tactical_ai/training/models/multiAgentModel",

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
                (60, 30),  
                (60, 50),  
            ],
            "defender_start": (100, 40),
        },

        {
            "name": "left",
            "attackers_start": [
                (75, 15),   
                (70, 25),   
            ],
            "defender_start": (100, 40),
        },

        {
            "name": "right",
            "attackers_start": [
                (75, 65),   
                (70, 55),   
            ],
            "defender_start": (100, 40),
        },

        {
            "name": "deepStart",
            "attackers_start": [
                (45, 30),   
                (45, 50),   
            ],
            "defender_start": (100, 40),
        },
    ],
}

# MULTI-AGENT WHAT IF SCENARIO
WHAT_IF = {

    # Path to trained model
    # You need to change this to your absolute path
    "model_path": "D:/PoliTo/Anno 5/DS project/footballAI-ads/src/football_tactical_ai/training/models/whatIFModel",

    # Output directories
    "save_video_dir": "D:/PoliTo/Anno 5/DS project/footballAI-ads/src/football_tactical_ai/evaluation/results/videos/WhatIF",
    "save_logs_dir":  "D:/PoliTo/Anno 5/DS project/footballAI-ads/src/football_tactical_ai/evaluation/results/logs/WhatIF",

    # Rendering settings
    "render": {
        "show_grid": False,
        "show_heatmap": False,
        "show_rewards": False,
        "full_pitch": True,
        "show_fov": False,
        "show_names": True,
    },
}


# GLOBAL DICT OF SCENARIOS
SCENARIOS_MULTI = {
    "2v1": MULTI_2V1,
    "what_if": WHAT_IF,
}
