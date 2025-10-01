import numpy as np
from football_tactical_ai.env.objects.pitch import Pitch

# some good values were:
# ATTACKERS:
# - min_reward: -0.02
# - max_reward: 0.1
# DEFENDERS:
# - min_reward: -0.02
# - max_reward: 0.08
# GOALKEEPERS:
# - min_reward: -0.02
# - max_reward: 0.08




def is_out_of_play(pitch: Pitch, x: float, y: float) -> bool:
    return x < 0 or x > pitch.width or y < 0 or y > pitch.height


import numpy as np
from football_tactical_ai.env.objects.pitch import Pitch


def build_attacker_grid(pitch: Pitch, 
                        role: str,
                        team: str = "A",
                        min_reward: float = -0.01, 
                        max_reward: float = 0.01,
                        y_center_penalty_scale: float = 0.5,
                        focus_sharpness: float = 2.5) -> np.ndarray:
    """
    Reward grid for attackers (role-specific, team-oriented):
    - Encourages attacking play in opponent's half (X axis)
    - Penalizes deviation from role-preferred vertical lane (Y axis)
    - Orientation depends on attacking direction:
        * Team A attacks RIGHT → targets on the right side
        * Team B attacks LEFT  → mirrored horizontally and vertically
    - Out of play = strong penalty (-5)
    """

    grid = np.zeros((pitch.num_cells_x, pitch.num_cells_y))

    if team == "A":  # attacking to the RIGHT
        role_targets = {
            "LW":  (0.75 * pitch.width, 0.15),
            "RW":  (0.75 * pitch.width, 0.85),
            "CF":  (pitch.width - 8, 0.50),
            "LCF": (pitch.width - 10, 0.35),
            "RCF": (pitch.width - 10, 0.65),
            "SS":  (pitch.width - 12, 0.50),
        }
    elif team == "B":  # attacking to the LEFT (mirrored)
        role_targets = {
            "LW":  (0.25 * pitch.width, 0.85),  # vertical flip
            "RW":  (0.25 * pitch.width, 0.15),  # vertical flip
            "CF":  (8, 0.50),                  # close to left goal
            "LCF": (10, 0.65),                 # mirrored vertically
            "RCF": (10, 0.35),
            "SS":  (12, 0.50),
        }
    else:
        raise ValueError(f"Unknown team: {team}")

    # Iterate over each cell
    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):

            # Cell center coordinates
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Out-of-play check
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -0.1
                continue

            # Normalize coords
            x_norm = (cell_x - pitch.x_min) / (pitch.x_max - pitch.x_min)
            y_norm = (cell_y - pitch.y_min) / (pitch.y_max - pitch.y_min)

            if role in role_targets:
                target_x, target_y = role_targets[role]

                # Distance from role hotspot
                dx = abs(cell_x - target_x) / pitch.width
                dy = abs(y_norm - target_y)
                dist = np.sqrt(dx**2 + dy**2)

                # Exponential decay → sharper hotspot
                score = np.exp(-focus_sharpness * dist)
            else:
                # Generic ATT fallback: advance along X, stay near center in Y
                x_reward = -0.5 + x_norm
                y_penalty = -0.5 * y_center_penalty_scale * abs(y_norm - 0.5) * 2
                score = x_reward + y_penalty
                score = (score + 1) / 2

            # Map into [min_reward, max_reward]
            grid[i, j] = min_reward + (max_reward - min_reward) * score

    return grid


def build_defender_grid(pitch: Pitch, 
                        role: str,
                        team: str = "B",
                        min_reward: float = -0.01,
                        max_reward: float = 0.01,
                        amplification: float = 4.5,
                        focus_sharpness: float = 2.5) -> np.ndarray:
    """
    Reward grid for defenders (LCB, RCB, CB)

    Orientation depends on the defending side:
    - Team A defends the LEFT goal (x ~ penalty_depth)
    - Team B defends the RIGHT goal (x ~ width - penalty_depth)

    Vertical roles (y target):
    - If defending RIGHT goal:
        * LCB ~ 0.65 
        * RCB ~ 0.35 
    - If defending LEFT goal: mirrored vertically
        * LCB ~ 0.35
        * RCB ~ 0.65
    - CB always ~ 0.50 (central)
    """

    grid = np.zeros((pitch.num_cells_x, pitch.num_cells_y))

    # Horizontal hotspot: depends on which side is being defended
    if team == "A":  # defending LEFT goal
        base_x = pitch.penalty_depth
        y_roles = {"LCB": 0.35, "RCB": 0.65, "CB": 0.50}  # mirrored vertically
    elif team == "B":  # defending RIGHT goal
        base_x = pitch.width - pitch.penalty_depth
        y_roles = {"LCB": 0.65, "RCB": 0.35, "CB": 0.50}
    else:
        raise ValueError(f"Unknown team: {team}")

    # Role-specific targets
    role_targets = {
        "LCB": (base_x, y_roles["LCB"]),
        "RCB": (base_x, y_roles["RCB"]),
        "CB":  (base_x, y_roles["CB"]),
    }

    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):

            # Cell center coordinates
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Out-of-play penalty
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -0.2
                continue

            # Normalize Y to [0,1]
            y_norm = (cell_y - pitch.y_min) / (pitch.y_max - pitch.y_min)

            if role in role_targets:
                target_x, target_y = role_targets[role]

                # Normalized distance to target
                dx = abs(cell_x - target_x) / pitch.width
                dy = abs(y_norm - target_y)
                dist = np.sqrt(dx**2 + dy**2)

                # Exponential decay for sharper hotspot
                score = np.exp(-focus_sharpness * dist)

            else:
                # Generic DEF fallback: reward staying near penalty depth
                center_x = base_x
                max_dist = max(center_x - pitch.x_min, pitch.x_max - center_x)
                x_dist = abs(cell_x - center_x) / max_dist
                x_reward = (1.0 - x_dist) ** amplification

                y_penalty = -0.5 * abs(y_norm - 0.5) * 2
                score = x_reward + y_penalty
                score = np.clip((score + 1) / 2, 0, 1)

            # Map into [min_reward, max_reward]
            grid[i, j] = min_reward + (max_reward - min_reward) * score

    return grid

def build_goalkeeper_grid(pitch: Pitch,
                          team: str = "B",
                          min_reward: float = -0.01,
                          max_reward: float = 0.01,
                          decay_scale: float = 0.025) -> np.ndarray:
    """
    Reward grid for goalkeepers (team-aware).

    - Team A defends the LEFT goal → hotspot = left goal area
    - Team B defends the RIGHT goal → hotspot = right goal area
    - Full reward inside own goal area (small box)
    - Outside: reward decays with distance, allowing small movements
    - Out-of-play cells receive strong penalty (-5)
    - Reward range is always in [min_reward, max_reward]
    """

    grid = np.full((pitch.num_cells_x, pitch.num_cells_y), fill_value=min_reward)

    if team == "A":  # defending LEFT goal
        x_min_area = 0.0
        x_max_area = pitch.goal_area_depth
    elif team == "B":  # defending RIGHT goal
        x_min_area = pitch.width - pitch.goal_area_depth
        x_max_area = pitch.width
    else:
        raise ValueError(f"Unknown team: {team}")

    # Vertical goal area limits are the same for both sides
    y_min_area = pitch.center_y - pitch.goal_area_height / 2
    y_max_area = pitch.center_y + pitch.goal_area_height / 2

    # Iterate over each cell in the grid
    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):

            # Cell center
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Out of play penalty
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -0.2
                continue

            # Full reward if inside own goal area
            if x_min_area <= cell_x <= x_max_area and y_min_area <= cell_y <= y_max_area:
                grid[i, j] = max_reward
            else:
                # Distance to nearest point in goal area
                dx = max(x_min_area - cell_x, 0, cell_x - x_max_area)
                dy = max(y_min_area - cell_y, 0, cell_y - y_max_area)
                dist = np.sqrt(dx**2 + dy**2)

                # Reward decays with distance
                score = max_reward - decay_scale * dist
                grid[i, j] = np.clip(score, min_reward, max_reward)

    return grid
