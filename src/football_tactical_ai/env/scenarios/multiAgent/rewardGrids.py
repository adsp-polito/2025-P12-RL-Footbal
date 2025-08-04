import numpy as np
from football_tactical_ai.env.objects.pitch import Pitch


def is_out_of_play(pitch: Pitch, x: float, y: float) -> bool:
    return x < 0 or x > pitch.width or y < 0 or y > pitch.height

def build_attacker_grid(pitch: Pitch, 
                        position_reward_scale: float = 1.0, 
                        y_center_penalty_scale: float = 0.5) -> np.ndarray:
    """
    Reward grid for attackers:
    - Encourages attacking play in opponent's half.
    - Penalizes being too far from the center.
    - if out of play, strong penalty.
    """
    # Initialize grid with zeros
    grid = np.zeros((pitch.num_cells_x, pitch.num_cells_y))

    # Iterate over each cell in the grid
    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):

            # Calculate the center of the cell
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Check if the cell is out of play or inside the goal area
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -5.0
                continue

            # Calculate normalized position
            x_norm = (cell_x - pitch.x_min) / (pitch.x_max - pitch.x_min) # [0, 1]
            y_norm = (cell_y - pitch.y_min) / (pitch.y_max - pitch.y_min) # [0, 1]

            # Calculate rewards
            x_reward = -0.5 * position_reward_scale + position_reward_scale * x_norm # range [-0.5, 0.5] if position_reward_scale = 1.0
            y_penalty = -0.5 * y_center_penalty_scale * abs(y_norm - 0.5) * 2 # range [-0.25, 0.0] if y_center_penalty_scale = 0.5

            # Combine rewards and penalties
            grid[i, j] = x_reward + y_penalty  # Total in [-1.0, 0.5], will normalize if needed

    return grid


def build_defender_grid(pitch: Pitch,
                        position_reward_scale: float = 1.0,
                        y_center_penalty_scale: float = 0.5,
                        amplification: float = 3.5) -> np.ndarray:
    """
    Reward grid for defenders:
    - Maximum reward at the edge of the penalty area.
    - Penalizes distance from vertical center.
    - Symmetric decay left and right from peak zone.
    - Out-of-play cells receive strong penalty.
    """
    # Initialize grid
    grid = np.zeros((pitch.num_cells_x, pitch.num_cells_y))

    # Use the actual penalty box edge as peak (e.g., 103.5 if field width = 120)
    center_x = pitch.width - pitch.penalty_depth
    max_dist = max(center_x - pitch.x_min, pitch.x_max - center_x)

    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):
            # Cell center in meters
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Check if the cell is out of play
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -5.0
                continue

            # X reward: linearly decreases away from center_x
            x_dist = abs(cell_x - center_x) / max_dist  # normalize to [0, 1]
            x_reward = (1.0 - x_dist)**amplification * position_reward_scale - 0.5  # range [-0.5, 0.5]


            # Y penalty: away from vertical center
            y_norm = (cell_y - pitch.y_min) / (pitch.y_max - pitch.y_min)
            y_penalty = -0.5 * y_center_penalty_scale * abs(y_norm - 0.5) * 2

            grid[i, j] = x_reward + y_penalty

    return grid


def build_goalkeeper_grid(pitch: Pitch, 
                          position_reward_scale: float = 0.5,
                          decay_scale: float = 0.04) -> np.ndarray:
    """
    Reward grid for goalkeepers:
    - Full reward inside the penalty box (area di rigore).
    - Outside: reward decays smoothly with distance from the penalty area.
    - Out-of-play cells receive strong penalty.
    """
    # Initialize grid with negative values
    grid = np.full((pitch.num_cells_x, pitch.num_cells_y), fill_value=-0.5)

    # Penalty area boundaries
    x_min_area = pitch.width - pitch.goal_area_depth
    x_max_area = pitch.width
    y_min_area = pitch.center_y - pitch.goal_area_height / 2
    y_max_area = pitch.center_y + pitch.goal_area_height / 2

    # Iterate over each cell in the grid
    for i in range(pitch.num_cells_x):
        for j in range(pitch.num_cells_y):
            # Cell center in meters
            cell_x = pitch.x_min + (i + 0.5) * pitch.cell_size
            cell_y = pitch.y_min + (j + 0.5) * pitch.cell_size

            # Out of play → strong penalty
            if is_out_of_play(pitch, cell_x, cell_y):
                grid[i, j] = -5.0
                continue

            # Inside goal area → full reward
            if x_min_area <= cell_x <= x_max_area and y_min_area <= cell_y <= y_max_area:
                grid[i, j] = position_reward_scale
            else:
                # Compute distance from nearest point of the goal area
                dx = max(x_min_area - cell_x, 0, cell_x - x_max_area)
                dy = max(y_min_area - cell_y, 0, cell_y - y_max_area)
                dist = np.sqrt(dx**2 + dy**2)

                # Decay reward linearly with distance
                reward = position_reward_scale - decay_scale * dist
                grid[i, j] = reward

    return np.clip(grid, -0.5, 0.5)
