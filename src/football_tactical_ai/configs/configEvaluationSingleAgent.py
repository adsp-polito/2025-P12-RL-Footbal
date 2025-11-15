"""
Single-Agent Evaluation Configuration File
------------------------------------------

A single Python module defining all evaluation settings
for MOVE, SHOT, and VIEW scenarios.
"""

# COMMON SETTINGS
COMMON = {
    "seconds": 10,
    "fps": 24,
    # 10 seconds * 24 FPS
    "max_steps": 10 * 24,
}

# MOVE SCENARIO
MOVE = {
    "model_path": "src/football_tactical_ai/training/models/singleAgentMoveModel.zip",

    "save_video_dir": "src/football_tactical_ai/evaluation/results/videos/move",
    "save_logs_dir":  "src/football_tactical_ai/evaluation/results/logs",


    "render": {
        "show_grid": False,
        "show_heatmap": True,
        "show_rewards": False,
        "full_pitch": True,
        "show_fov": False,  
        "show_info": True,
    },

    # Predefined evaluation cases
    "test_cases": [
        {
            "name": "center",
            "attacker_start": (60, 40),
            "defender_start": (100, 40),
        },
        {
            "name": "left",
            "attacker_start": (70, 20),
            "defender_start": (100, 40),
        },
        {
            "name": "right",
            "attacker_start": (70, 60),
            "defender_start": (100, 40),
        },
        {
            "name": "deep_start",
            "attacker_start": (40, 40),
            "defender_start": (100, 40),
        },
        {
            "name": "close_defender",
            "attacker_start": (80, 40),
            "defender_start": (100, 40),
        }
    ],
}


# SHOT SCENARIO
SHOT = {
    "model_path": "src/football_tactical_ai/training/models/singleAgentShotModel.zip",

    "save_video_dir": "src/football_tactical_ai/evaluation/results/videos/shot",
    "save_logs_dir":  "src/football_tactical_ai/evaluation/results/logs",

    "render": {
        "show_grid": False,
        "show_heatmap": False,
        "show_rewards": False,
        "full_pitch": True,
        "show_fov": False,  
        "show_info": True,
    },

    "test_cases": [
        {
            "name": "edge_box",
            "attacker_start": (88, 40),
            "defender_start": (100, 40),
        },
        {
            "name": "left",
            "attacker_start": (80, 25),
            "defender_start": (98, 30),
        },
        {
            "name": "right",
            "attacker_start": (80, 55),
            "defender_start": (98, 50),
        },
        {
            "name": "central_far",
            "attacker_start": (70, 40),
            "defender_start": (100, 40),
        },
    ],
}


# VIEW SCENARIO
VIEW = {
    "model_path": "src/football_tactical_ai/training/models/singleAgentViewModel.zip",

    "save_video_dir": "src/football_tactical_ai/evaluation/results/videos/view",
    "save_logs_dir":  "src/football_tactical_ai/evaluation/results/logs",

    "render": {
        "show_grid": False,
        "show_heatmap": False,
        "show_rewards": False,
        "full_pitch": True,
        "show_fov": True,   # ONLY VIEW SHOWS FIELD OF VIEW
        "show_info": True,
    },

    "test_cases": [
        {
            "name": "center",
            "attacker_start": (60, 40),
            "defender_start": (100, 40),
        },
        {
            "name": "left",
            "attacker_start": (55, 20),
            "defender_start": (95, 30),
        },
        {
            "name": "right",
            "attacker_start": (55, 60),
            "defender_start": (95, 50),
        },
        {
            "name": "diagonal",
            "attacker_start": (50, 30),
            "defender_start": (90, 50),
        },
    ],
}


# GLOBAL DICT
SCENARIOS = {
    "move": MOVE,
    "shot": SHOT,
    "view": VIEW,
}
