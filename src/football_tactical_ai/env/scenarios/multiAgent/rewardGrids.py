import numpy as np
from football_tactical_ai.env.objects.pitch import Pitch

def is_out_of_play(pitch: Pitch, x: float, y: float) -> bool:
    return x < 0 or x > pitch.width or y < 0 or y > pitch.height

def build_attacker_grid(pitch: Pitch,
                        role: str,
                        team: str = "A",
                        min_reward: float = -0.03,
                        max_reward: float = 0.07,
                        focus_sharpness: float = 2.5
                    ) -> np.ndarray:
    """
    Build the role-dependent spatial reward grid for attacking players 
    (LW, RW, CF, LCF, RCF, SS) or generic attackers (ATT)

    PARAMETERS:
        pitch (Pitch): pitch geometry with grid parameters
        role (str): attacking role
        team (str): 'A' attacks RIGHT, 'B' attacks LEFT
        min_reward (float): lowest possible grid value
        max_reward (float): highest possible grid value
        focus_sharpness (float): controls exponential decay from the role target

    RETURNS:
        np.ndarray: 2D grid (num_cells_x * num_cells_y) with spatial rewards
    """

    grid = np.zeros((pitch.num_cells_x, pitch.num_cells_y))

    # DEFINE TARGET HOTSPOTS FOR EACH ROLE (in field meters)
    if team == "A":  # Attacking to the RIGHT
        role_targets = {
            "LW":  (pitch.width - 8, 0.25),
            "RW":  (pitch.width - 8, 0.75),
            "CF":  (pitch.width - 8, 0.50),
            "LCF": (pitch.width - 10, 0.35),
            "RCF": (pitch.width - 10, 0.65),
            "SS":  (pitch.width - 12, 0.50),
        }

    elif team == "B":  # Attacking to the LEFT
        role_targets = {
            "LW":  (8, 0.75),  # vertically mirrored
            "RW":  (8, 0.25),
            "CF":  (8, 0.50),
            "LCF": (10, 0.65),
            "RCF": (10, 0.35),  
            "SS":  (12, 0.50),
        }

    else:
        raise ValueError(f"Unknown team: {team}")

    # BUILD GRID VALUES
    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):

            # Cell center coordinates (in meters)
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Out-of-play cells: negative value
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -0.05
                continue

            # Normalized coordinates [0, 1]
            x_norm = (cell_x - pitch.x_min) / (pitch.x_max - pitch.x_min)
            y_norm = (cell_y - pitch.y_min) / (pitch.y_max - pitch.y_min)

            # ROLE-SPECIFIC HOTSPOT REWARD
            if role in role_targets:
                target_x, target_y = role_targets[role]

                # Distance to the role-specific target
                dx = abs(cell_x - target_x) / pitch.width
                dy = abs(y_norm - target_y)
                dist = np.sqrt(dx**2 + dy**2)

                # Exponential decay around the target point
                score = np.exp(-focus_sharpness * dist)

            else:
                # GENERIC ATTACKER BEHAVIOR → ELLIPTICAL REWARD

                # Reference point to start (in meters)
                # Attacking right
                if team == "A":   
                    gx = pitch.width - 11.0         # slightly before goal line (penalty spot is the reference)
                    gy = pitch.center_y
                
                # Attacking left
                else:             
                    gx = pitch.x_min + 11.0         # slightly after goal line (penalty spot is the reference)
                    gy = pitch.center_y

                # Ellipse radii (tunable)
                # a = horizontal spread
                # b = vertical tightness (smaller → stronger central bias)
                a = 50.0
                b = 50.0

                # Elliptical distance to goal center
                dx = cell_x - gx
                dy = cell_y - gy

                dist_ell = np.sqrt((dx*dx) / (a*a) + (dy*dy) / (b*b))

                # Elliptical Gaussian score ∈ (0,1]
                score = np.exp(-(dist_ell**2))

            # MAP SCORE INTO [min_reward, max_reward]
            grid[i, j] = min_reward + (max_reward - min_reward) * score

    return grid



def build_defender_grid(pitch: Pitch,
                        role: str,
                        team: str = "B",
                        min_reward: float = -0.03,
                        max_reward: float = 0.07,
                        focus_sharpness: float = 4.5
                    ) -> np.ndarray:
    """
    Build the spatial reward grid for defensive players 
    (LCB, RCB, CB) or generic defenders (DEF)

    PARAMETERS:
        pitch (Pitch): pitch geometry and discretization settings
        role (str): defensive role ('LCB', 'RCB', 'CB')
        team (str): 'A' defends LEFT goal, 'B' defends RIGHT goal
        min_reward (float): minimum possible reward value
        max_reward (float): maximum possible reward value
        focus_sharpness (float): controls how fast the reward decays
                                 from the ideal defensive position

    RETURNS:
        np.ndarray: 2D reward grid of shape (num_cells_x, num_cells_y)
    """

    # Initialize the reward grid
    grid = np.zeros((pitch.num_cells_x, pitch.num_cells_y))

    # Define role-specific target zones for each team
    if team == "A":  # Team A defends LEFT
        base_x = pitch.penalty_depth
        y_targets = {"LCB": 0.35, "RCB": 0.65, "CB": 0.50}
    elif team == "B":  # Team B defends RIGHT
        base_x = pitch.width - pitch.penalty_depth
        y_targets = {"LCB": 0.65, "RCB": 0.35, "CB": 0.50}
    else:
        raise ValueError(f"Unknown team: {team}")

    # Role-based hotspots (in meters)
    role_targets = {
        "LCB": (base_x, y_targets["LCB"]),
        "RCB": (base_x, y_targets["RCB"]),
        "CB":  (base_x, y_targets["CB"]),
    }

    # Iterate through all grid cells
    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):

            # Compute cell center coordinates (in meters)
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Out-of-play → negative reward
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -0.05
                continue

            # Normalize Y coordinate to [0, 1]
            y_norm = (cell_y - pitch.y_min) / (pitch.y_max - pitch.y_min)

            # ROLE-SPECIFIC HOTSPOT LOGIC
            if role in role_targets:
                target_x, target_y = role_targets[role]

                # Compute normalized distances from the role target
                dx = abs(cell_x - target_x) / pitch.width
                dy = abs(y_norm - target_y)
                dist = np.sqrt(dx**2 + dy**2)

                # Exponential decay: closer cells get higher reward
                score = np.exp(-focus_sharpness * dist)

            else:
                # Generic fallback behavior (if role not recognized)
                # Simply reward staying close to the penalty area center
                if team == "A":
                    x_target = pitch.penalty_depth
                else:
                    x_target = pitch.width - pitch.penalty_depth

                dx = abs(cell_x - x_target) / pitch.width
                dy = abs(y_norm - 0.5)
                dist = np.sqrt(dx**2 + dy**2)
                score = np.exp(-focus_sharpness * dist)

            # Map score into the defined reward range
            grid[i, j] = min_reward + (max_reward - min_reward) * score

    return grid



def build_goalkeeper_grid(pitch: Pitch,
                          team: str = "B",
                          min_reward: float = -0.05,
                          max_reward: float = 0.03
                        ) -> np.ndarray:
    """
    Build the spatial reward grid for goalkeepers (team-aware)

    PARAMETERS:
        pitch (Pitch): pitch geometry and discretization settings
        team (str): 'A' defends LEFT goal, 'B' defends RIGHT goal
        min_reward (float): lowest cell value (farthest / wrong positions)
        max_reward (float): highest cell value (ideal goalkeeper zone)

    RETURNS:
        np.ndarray: 2D grid (num_cells_x * num_cells_y) with spatial rewards
    """

    # Fixed decay parameter (controls how fast reward decreases with distance)
    decay_scale = 0.0075

    # Initialize grid with the minimum reward everywhere
    grid = np.full((pitch.num_cells_x, pitch.num_cells_y), fill_value=min_reward)

    # Define goal area bounds depending on the defending side
    if team == "A":  # Team A defends the LEFT goal
        x_min_area = 0.0
        x_max_area = pitch.goal_area_depth
    elif team == "B":  # Team B defends the RIGHT goal
        x_min_area = pitch.width - pitch.goal_area_depth
        x_max_area = pitch.width
    else:
        raise ValueError(f"Unknown team: {team}")

    # Vertical limits (identical for both sides)
    y_min_area = pitch.center_y - pitch.goal_area_height / 2
    y_max_area = pitch.center_y + pitch.goal_area_height / 2

    # Build the reward grid
    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):

            # Compute the center coordinates of the current cell (in meters)
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Out-of-play check (outside the pitch boundaries)
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -0.05  # Penalty for invalid positions
                continue

            # Inside own goal area → maximum reward
            if x_min_area <= cell_x <= x_max_area and y_min_area <= cell_y <= y_max_area:
                grid[i, j] = max_reward
            else:
                # OUTSIDE GOAL AREA: apply smooth reward decay
                # Compute Euclidean distance from the closest edge of the goal area
                dx = max(x_min_area - cell_x, 0, cell_x - x_max_area)
                dy = max(y_min_area - cell_y, 0, cell_y - y_max_area)
                dist = np.sqrt(dx**2 + dy**2)

                # Reward decreases linearly with distance from goal area
                score = max_reward - decay_scale * dist

                # Clamp the result to the allowed reward range
                grid[i, j] = np.clip(score, min_reward, max_reward)

    return grid