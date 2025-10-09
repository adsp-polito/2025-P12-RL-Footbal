import numpy as np
from football_tactical_ai.env.objects.pitch import Pitch

def is_out_of_play(pitch: Pitch, x: float, y: float) -> bool:
    return x < 0 or x > pitch.width or y < 0 or y > pitch.height

def build_attacker_grid(pitch: Pitch,
                        role: str,
                        team: str = "A",
                        min_reward: float = -0.05,
                        max_reward: float = 0.05,
                        focus_sharpness: float = 2.0
                    ) -> np.ndarray:
    """
    Build the spatial reward grid for attacking players (role-dependent)

    PURPOSE:
    - Encourage offensive play in the opponent's half (along X-axis)
    - Provide role-specific hotspots
    - Penalize deviation from vertical alignment (Y-axis balance)
    - Mirror logic for opposite attacking directions

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
            "LW":  (0.75 * pitch.width, 0.15),
            "RW":  (0.75 * pitch.width, 0.85),
            "CF":  (pitch.width - 8, 0.50),
            "LCF": (pitch.width - 10, 0.35),
            "RCF": (pitch.width - 10, 0.65),
            "SS":  (pitch.width - 12, 0.50),
        }

    elif team == "B":  # Attacking to the LEFT
        role_targets = {
            "LW":  (0.25 * pitch.width, 0.85),  # vertically mirrored
            "RW":  (0.25 * pitch.width, 0.15),
            "CF":  (8, 0.50),
            "LCF": (10, 0.65),
            "RCF": (10, 0.35),  
            "SS":  (12, 0.50),
        }

    else:
        raise ValueError(f"Unknown team: {team}")

    # FIXED PARAMETERS FOR Y-AXIS PENALIZATION (center alignment)
    y_center_penalty = 0.3  # fixed intensity for deviation from center

    # BUILD GRID VALUES
    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):

            # Cell center coordinates (in meters)
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Out-of-play cells: strong negative value
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -0.2
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
                # GENERIC ATTACKER BEHAVIOR (fallback)
                # Reward progress along X (towards attacking goal)
                x_reward = x_norm

                # Penalize excessive deviation from central Y area
                y_penalty = y_center_penalty * abs(y_norm - 0.5)

                # Combine and rescale into [0, 1]
                score = np.clip(x_reward - y_penalty, 0, 1)

            # MAP SCORE INTO [min_reward, max_reward]
            grid[i, j] = min_reward + (max_reward - min_reward) * score

    return grid


def build_defender_grid(pitch: Pitch,
                        role: str,
                        team: str = "B",
                        min_reward: float = -0.05,
                        max_reward: float = 0.05,
                        focus_sharpness: float = 2.0
                    ) -> np.ndarray:
    """
    Build the spatial reward grid for defensive players (LCB, RCB, CB)

    PURPOSE:
    - Encourage defenders to maintain compact positioning near the defensive line
    - Reward staying close to the goal and within their vertical defensive channel
    - Mirror logic depending on which side the team is defending
    - Penalize wide or advanced positions (moving too far from goal)

    PARAMETERS:
        pitch (Pitch): pitch geometry and discretization settings
        role (str): defensive role ('LCB', 'RCB', 'CB', or others)
        team (str): 'A' defends LEFT goal, 'B' defends RIGHT goal
        min_reward (float): lowest cell value (farthest / wrong positions)
        max_reward (float): highest cell value (ideal defensive zone)
        focus_sharpness (float): controls decay from the role hotspot

    RETURNS:
        np.ndarray: 2D grid (num_cells_x * num_cells_y) with spatial rewards
    """

    grid = np.zeros((pitch.num_cells_x, pitch.num_cells_y))

    # DEFENSIVE REFERENCE PARAMETERS
    # Y-axis alignment penalty → keeps vertical structure (0=center, 1=stronger)
    y_alignment_penalty = 0.4

    # Defensive depth bonus → how much we encourage staying near own penalty area
    depth_importance = 1.0  # scaling factor for distance to base_x

    # DEFINE TARGET ZONES BASED ON TEAM SIDE AND ROLE
    if team == "A":  # Defending the LEFT goal
        base_x = pitch.penalty_depth
        y_targets = {"LCB": 0.35, "RCB": 0.65, "CB": 0.50}
    elif team == "B":  # Defending the RIGHT goal
        base_x = pitch.width - pitch.penalty_depth
        y_targets = {"LCB": 0.65, "RCB": 0.35, "CB": 0.50}
    else:
        raise ValueError(f"Unknown team: {team}")

    # Role-based hotspot targets (in meters)
    role_targets = {
        "LCB": (base_x, y_targets["LCB"]),
        "RCB": (base_x, y_targets["RCB"]),
        "CB":  (base_x, y_targets["CB"]),
    }

    # BUILD GRID VALUES
    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):

            # Cell center coordinates
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Out-of-play → negative value
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -0.2
                continue

            # Normalize Y coordinate [0, 1]
            y_norm = (cell_y - pitch.y_min) / (pitch.y_max - pitch.y_min)

            # ROLE-SPECIFIC HOTSPOT
            if role in role_targets:
                target_x, target_y = role_targets[role]

                # Normalized distances
                dx = abs(cell_x - target_x) / pitch.width
                dy = abs(y_norm - target_y)
                dist = np.sqrt(dx**2 + dy**2)

                # Exponential decay around the hotspot (local tactical zone)
                score = np.exp(-focus_sharpness * dist)

                # Apply vertical alignment penalty
                y_penalty = y_alignment_penalty * abs(y_norm - target_y)
                score = np.clip(score - y_penalty, 0, 1)

                # Encourage staying closer to own goal (depth reward)
                if team == "A":
                    depth_factor = np.exp(-depth_importance * (cell_x / pitch.width))
                else:
                    depth_factor = np.exp(-depth_importance * ((pitch.width - cell_x) / pitch.width))

                score *= depth_factor

            else:
                # GENERIC DEFENDER FALLBACK (for undefined roles)
                # Reward proximity to penalty box depth
                if team == "A":
                    dist_x = cell_x / pitch.width
                else:
                    dist_x = (pitch.width - cell_x) / pitch.width

                base_reward = np.exp(-depth_importance * dist_x)
                y_penalty = y_alignment_penalty * abs(y_norm - 0.5)
                score = np.clip(base_reward - y_penalty, 0, 1)

            # MAP TO [min_reward, max_reward]
            grid[i, j] = min_reward + (max_reward - min_reward) * score

    return grid





def build_goalkeeper_grid(pitch: Pitch,
                          team: str = "B",
                          min_reward: float = -0.05,
                          max_reward: float = 0.05
                        ) -> np.ndarray:
    """
    Build the spatial reward grid for goalkeepers (team-aware)

    PURPOSE:
    - Encourage the goalkeeper to remain within or near the goal area
    - Assign full reward inside the 6-yard box (own goal area)
    - Gradually decay the reward as distance from the goal area increases
      allowing realistic, small lateral or forward movements
    - Penalize positions that are too far from the goal or outside the field
    - Reward values are clamped within [min_reward, max_reward]

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
                grid[i, j] = -0.2  # Strong penalty for invalid positions
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