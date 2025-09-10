import numpy as np
from football_tactical_ai.env.objects.pitch import Pitch


def is_out_of_play(pitch: Pitch, x: float, y: float) -> bool:
    return x < 0 or x > pitch.width or y < 0 or y > pitch.height


def build_attacker_grid(pitch: Pitch, 
                        role: str = None,
                        min_reward: float = -0.1, 
                        max_reward: float = 0.1,
                        y_center_penalty_scale: float = 0.5) -> np.ndarray:
    """
    Reward grid for attackers (role-specific):
    - Encourages attacking play in opponent's half (X axis)
    - Penalizes deviation from preferred vertical lane (Y axis)
    - Out of play = strong penalty (-5)
    - Range always [min_reward, max_reward]
    - If role is None or unknown → fallback to base logic
    """
    grid = np.zeros((pitch.num_cells_x, pitch.num_cells_y))

    # Hotspot target for each role (x in meters, y in normalized [0,1])
    role_targets = {
        "LW":  (0.75 * pitch.width, 0.15),
        "RW":  (0.75 * pitch.width, 0.85),
        "CF":  (pitch.width - 8, 0.5),
        "LCF": (pitch.width - 10, 0.35),
        "RCF": (pitch.width - 10, 0.65),
        "SS":  (pitch.width - 12, 0.5),  # secondary striker below CF
    }

    focus_sharpness = 2.5  # sensitivity factor

    # Iterate over each cell in the grid
    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):

            # Calculate cell center coordinates
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Out of play penalty
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -5.0
                continue

            # Normalize coordinates
            x_norm = (cell_x - pitch.x_min) / (pitch.x_max - pitch.x_min)
            y_norm = (cell_y - pitch.y_min) / (pitch.y_max - pitch.y_min)

            if role in role_targets:
                # Role-specific hotspot
                target_x, target_y = role_targets[role]

                # Calculate normalized distance to target
                dx = abs(cell_x - target_x) / pitch.width
                dy = abs(y_norm - target_y)
                dist = np.sqrt(dx**2 + dy**2)

                # Exponential decay for sharper focus
                score = np.exp(-focus_sharpness * dist)  # sharper focus
            else:
                # Fallback: base logic for generic ATT
                x_reward = -0.5 + x_norm
                y_penalty = -0.5 * y_center_penalty_scale * abs(y_norm - 0.5) * 2
                score = x_reward + y_penalty
                score = (score + 1) / 2  # normalize back to [0,1]

            grid[i, j] = min_reward + (max_reward - min_reward) * score

    return grid


def build_defender_grid(pitch: Pitch, 
                        role: str = None,
                        min_reward: float = -0.1,
                        max_reward: float = 0.1,
                        amplification: float = 4.5) -> np.ndarray:
    """
    Reward grid for defenders (LCB, RCB, CB):
    - Maximum reward near the penalty box edge
    - Role-specific vertical preference (LCB = top, RCB = bottom, CB = central)
    - Out-of-play cells receive strong penalty (-5)
    - Range always in [min_reward, max_reward]
    - If role is None or unknown → fallback to base logic
    """
    grid = np.zeros((pitch.num_cells_x, pitch.num_cells_y))

    # Hotspot target (slightly in front of goal area, mirrored vertically)
    role_targets = {
        "LCB": (pitch.width - pitch.penalty_depth, 0.65),  
        "RCB": (pitch.width - pitch.penalty_depth, 0.35),  
        "CB":  (pitch.width - pitch.penalty_depth, 0.5),   
    }

    # Sensitivity factor: higher = narrower hotspot
    focus_sharpness = 2.0

    # Iterate over each cell in the grid
    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):

            # Calculate cell center coordinates
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Out of play penalty
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -5.0
                continue

            # Normalize Y coordinate
            y_norm = (cell_y - pitch.y_min) / (pitch.y_max - pitch.y_min)

            if role in role_targets:
                # Role-specific hotspot with sharper decay
                target_x, target_y = role_targets[role]

                # Calculate normalized distance to target
                dx = abs(cell_x - target_x) / pitch.width
                dy = abs(y_norm - target_y)
                dist = np.sqrt(dx**2 + dy**2)

                # Exponential decay for sharper focus
                score = np.exp(-focus_sharpness * dist)  # sharper focus
            else:
                # Fallback: base logic for generic DEF
                center_x = pitch.width - pitch.penalty_depth
                max_dist = max(center_x - pitch.x_min, pitch.x_max - center_x)
                x_dist = abs(cell_x - center_x) / max_dist
                x_reward = (1.0 - x_dist) ** amplification
                y_penalty = -0.5 * abs(y_norm - 0.5) * 2
                score = x_reward + y_penalty
                score = np.clip((score + 1) / 2, 0, 1)

            grid[i, j] = min_reward + (max_reward - min_reward) * score

    return grid


def build_goalkeeper_grid(pitch: Pitch,
                          min_reward: float = -0.1,
                          max_reward: float = 0.1,
                          decay_scale: float = 0.015) -> np.ndarray:
    """
    Reward grid for goalkeepers:
    - Full reward inside the goal area (small box)
    - Outside: reward decays more slowly with distance (so keeper can move a bit)
    - Out-of-play cells receive strong penalty (-5)
    - Range always in [min_reward, max_reward]
    """
    grid = np.full((pitch.num_cells_x, pitch.num_cells_y), fill_value=min_reward)

    # Goal area boundaries
    x_min_area = pitch.width - pitch.goal_area_depth
    x_max_area = pitch.width
    y_min_area = pitch.center_y - pitch.goal_area_height / 2
    y_max_area = pitch.center_y + pitch.goal_area_height / 2

    # Iterate over each cell in the grid
    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):

            # Calculate cell center coordinates
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Out of play penalty
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -5.0
                continue

            # Full reward inside goal area
            if x_min_area <= cell_x <= x_max_area and y_min_area <= cell_y <= y_max_area:
                grid[i, j] = max_reward
            else:
                # Calculate distance to nearest point in goal area
                dx = max(x_min_area - cell_x, 0, cell_x - x_max_area)
                dy = max(y_min_area - cell_y, 0, cell_y - y_max_area)
                dist = np.sqrt(dx**2 + dy**2)

                # Decay with distance
                score = max_reward - decay_scale * dist
                grid[i, j] = np.clip(score, min_reward, max_reward)

    return grid
