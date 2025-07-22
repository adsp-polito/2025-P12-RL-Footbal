import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Rectangle

# GRID INFORMATION (USED FOR RL)
# The pitch is overlaid with a 5x5m (or 1x1m) cell grid for RL training, visual debugging, and reward shaping.

# ==> Full pitch grid:
#       - Area: 130m (x) × 90m (y) = full pitch with 5m margin on all sides
#       - If using 5x5m cells:
#           - N_COLS_FULL = 130 / 5 = 26 columns
#           - N_ROWS_FULL = 90 / 5 = 18 rows
#           - Total cells: 26 × 18 = 468

# ==> Half pitch grid:
#       - Area: 70m (x) × 90m (y) = half pitch with 5m margin on all sides
#       - If using 5x5m cells:
#           - N_COLS_HALF = 70 / 5 = 14 columns
#           - N_ROWS_HALF = 90 / 5 = 18 rows
#           - Total cells: 14 × 18 = 252

# Z-ORDER LAYERS EXPLANATION
# Z-order defines the drawing priority (higher values appear on top)
# We use the following convention to control rendering layers:

# Z-order defines the drawing priority (higher values appear on top).
# The following convention is used to control rendering layers:

# zorder = 0 ==> Field background stripes (base layer)

# zorder = 1 ==> Debug grid lines (optional, for RL debugging)

# zorder = 2 ==> Static pitch elements:
#                   - Field boundaries
#                   - Penalty areas and goal areas
#                   - Center line and circle
#                   - Penalty arcs and corner arcs
#                   - Text labels for cell indices (debugging)
#                These are usually drawn without explicitly setting zorder, as matplotlib defaults place them

# zorder = 3 ==> Dynamic elements (always drawn on top):
#              - Attacking player
#              - Defenders
#              - Ball

# Most matplotlib drawing functions (e.g., ax.plot, ax.add_patch) default to zorder=2.
# We make it explicit where needed to ensure consistent rendering layers.

# Global field dimensions
FIELD_WIDTH = 120
FIELD_HEIGHT = 80
HALF_FIELD_X = FIELD_WIDTH // 2
CENTER_Y = FIELD_HEIGHT // 2
STRIPE_WIDTH = 5

# Fields Margins (for a 120x80 pitch)
X_MIN, Y_MIN = -5, -5
X_MAX, Y_MAX = 125, 85

# Fields Proportions (for a 120x80 pitch)
PENALTY_AREA_DEPTH = 18           # 18 meters depth
PENALTY_AREA_HEIGHT = 44          # from y=18 to y=62 (62-18 = 44 meters)
GOAL_AREA_DEPTH = 6               # 6 meters depth
GOAL_AREA_HEIGHT = 20             # from y=30 to y=50 (50-30 = 20 meters) or y = 35 to y = 45 if we consider the additional 5m margin
PENALTY_SPOT_X_RIGHT = 109        # x position for right side
PENALTY_SPOT_X_LEFT = 11          # x position for left side
CENTER_CIRCLE_RADIUS = 10         # 10 meters radius
GOAL_HEIGHT = 7.32                # 7.32 meters (standard)
GOAL_DEPTH = 2.44                 # 2.44 meters depth


# Grid cell size for reward shaping.
# Defines the resolution of the pitch grid.
# Smaller cells (e.g., 1m) allow fine-grained shaping but increase computational cost.
# Larger cells (e.g., 5m) reduce detail but are more efficient.
# The cell size should match the environment's scale and be a divisor of the pitch dimensions (e.g., 120x80).
CELL_SIZE = 1  # meters (e.g., 5x5m, or 1x1m for fine resolution)

def draw_penalty_area(ax, side='right', lc='white'):
    """
    Function to draw the penalty area, goal area, penalty spot and penalty arc on the specified side of the pitch.
    
    Args:
        ax (matplotlib.axes.Axes): The axis to draw on.
        side (str): 'right' or 'left' to specify which side of the pitch.
        lc (str): Line color for the penalty area, goal area, and penalty spot.
    
    Returns:
        None
    """
    direction = 1 if side == 'right' else -1
    x_base = FIELD_WIDTH if side == 'right' else 0

    # Calculate x positions for penalty area and goal area depth
    penalty_x = x_base - direction * PENALTY_AREA_DEPTH
    goal_x = x_base - direction * GOAL_AREA_DEPTH

    # Calculate y positions for penalty area and goal area height
    y1_pen = (FIELD_HEIGHT - PENALTY_AREA_HEIGHT) / 2
    y2_pen = (FIELD_HEIGHT + PENALTY_AREA_HEIGHT) / 2
    y1_goal = (FIELD_HEIGHT - GOAL_AREA_HEIGHT) / 2
    y2_goal = (FIELD_HEIGHT + GOAL_AREA_HEIGHT) / 2

    # Draw penalty area rectangle
    ax.plot([x_base, penalty_x], [y1_pen, y1_pen], color=lc, linewidth=2, zorder=2)
    ax.plot([penalty_x, penalty_x], [y1_pen, y2_pen], color=lc, linewidth=2, zorder=2)
    ax.plot([x_base, penalty_x], [y2_pen, y2_pen], color=lc, linewidth=2, zorder=2)

    # Draw goal area rectangle
    ax.plot([x_base, goal_x], [y1_goal, y1_goal], color=lc, linewidth=2, zorder=2)
    ax.plot([goal_x, goal_x], [y1_goal, y2_goal], color=lc, linewidth=2, zorder=2)
    ax.plot([x_base, goal_x], [y2_goal, y2_goal], color=lc, linewidth=2, zorder=2)

    # Compute penalty spot and arc center
    if side == 'right':
        penalty_spot_x = PENALTY_SPOT_X_RIGHT
        arc_center_x = penalty_spot_x - 1.8  # Slight offset toward the goal line
        arc_angle = 180
    else:
        penalty_spot_x = PENALTY_SPOT_X_LEFT
        arc_center_x = penalty_spot_x + 1.8  # Slight offset toward the goal line
        arc_angle = 0

    # Draw penalty spot
    ax.plot([penalty_spot_x], [CENTER_Y], marker='o', markersize=5, color=lc, zorder=2)

    # Draw penalty arc
    arc = Arc(
        (arc_center_x, CENTER_Y), 
        20, 20, angle=arc_angle, 
        theta1=301, theta2=59, 
        color=lc, linewidth=2, zorder=2
    )
    ax.add_patch(arc)

def draw_goal(ax, side='right', lc='white'):
    """ Function to draw the goal on the specified side of the pitch.
    Args:
        ax (matplotlib.axes.Axes): The axis to draw on.
        side (str): 'right' or 'left' to specify which side of the pitch
        lc (str): Line color for the goal.
    Returns:
        None
    """
    # Goal rectangle
    x_goal = FIELD_WIDTH if side == 'right' else 0
    direction = 1 if side == 'right' else -1
    goal_y1 = CENTER_Y - GOAL_HEIGHT / 2
    ax.add_patch(Rectangle((x_goal, goal_y1), direction * GOAL_DEPTH, GOAL_HEIGHT, linewidth=2, edgecolor=lc, facecolor='none', zorder=2))


# Draw the offensive half of the pitch
def draw_half_pitch(
    ax=None,
    field_color='green',
    stripes=False,
    show_grid=False,
    show_heatmap=False,
    show_rewards=False,
    env=None
):
    """
    Draw the offensive half of a 120x80m football pitch with 5m margin.
    Optionally overlay reward shaping grid with grid lines, heatmap or reward values.

    Parameters:
        - ax (matplotlib.axes.Axes, optional): Axis to draw the pitch on.
        - field_color (str): Pitch background color.
        - stripes (bool): Whether to draw alternating mowing stripes.
        - show_grid (bool): Draws grid lines on the field.
        - show_heatmap (bool): Fills cells with colors based on reward.
        - show_rewards (bool): Writes reward values inside each cell.
        - env (OffensiveScenarioMoveSingleAgent, optional): The environment for computing reward.

    Returns:
        - matplotlib.axes.Axes: The axis with the rendered half-pitch.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))

    x_min, x_max = 55, 125
    y_min, y_max = -5, 85

    # Background color
    if field_color == 'green':
        ax.set_facecolor('#4CAF50')
    else:
        ax.set_facecolor(field_color)

    lc = 'whitesmoke'
    border_color = 'white'

    # Mowing stripes (optional)
    if field_color == 'green' and stripes:
        for i in range(0, FIELD_WIDTH, STRIPE_WIDTH):
            stripe_color = '#43A047' if (i // STRIPE_WIDTH) % 2 == 0 else '#4CAF50'
            ax.add_patch(Rectangle((i, 0), STRIPE_WIDTH, FIELD_HEIGHT, color=stripe_color, zorder=0))

    # Half-pitch boundaries
    ax.plot([HALF_FIELD_X, HALF_FIELD_X], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([FIELD_WIDTH, FIELD_WIDTH], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([HALF_FIELD_X, FIELD_WIDTH], [0, 0], color=border_color, linewidth=3)
    ax.plot([HALF_FIELD_X, FIELD_WIDTH], [FIELD_HEIGHT, FIELD_HEIGHT], color=border_color, linewidth=3)

    # Center arc
    ax.add_patch(Arc((HALF_FIELD_X, CENTER_Y), 20, 20, angle=0, theta1=270, theta2=90, color=lc, linewidth=2))

    # Penalty area and goal
    draw_penalty_area(ax, side='right', lc=lc)
    draw_goal(ax, side='right', lc=lc)

    # Corner arcs (right side only)
    for (x, y), angle in zip([(FIELD_WIDTH, 0), (FIELD_WIDTH, FIELD_HEIGHT)], [90, 180]):
        ax.add_patch(Arc((x, y), 4, 4, angle=angle, theta1=0, theta2=90, color=lc, linewidth=2))

    # Optional reward grid (grid, heatmap, numbers)
    if env is not None:
        num_cells_x = env.num_cells_x
        num_cells_y = env.num_cells_y
        cell_width = CELL_SIZE
        cell_height = CELL_SIZE

        for i in range(num_cells_x):
            for j in range(num_cells_y):
                x0 = X_MIN + i * cell_width
                y0 = Y_MIN + j * cell_height
                x_center = x0 + cell_width / 2
                y_center = y0 + cell_height / 2
                reward = env._get_position_reward(x_center, y_center)

                normalized_reward = (reward + 0.5) / 1.0  # [-0.5, 0.5] → [0, 1]
                color = plt.cm.coolwarm(normalized_reward)


                # Heatmap
                if show_heatmap:
                    rect = Rectangle(
                        (x0, y0),
                        cell_width,
                        cell_height,
                        facecolor=color,
                        edgecolor='black' if show_grid else 'none',
                        linewidth=0.4,
                        alpha=0.6,
                        zorder=1
                    )
                    ax.add_patch(rect)

                # Grid (lines only if heatmap edges not shown)
                if show_grid and not show_heatmap:
                    ax.plot([x0, x0 + cell_width], [y0, y0], color='black', linewidth=0.4, zorder=1)
                    ax.plot([x0, x0 + cell_width], [y0 + cell_height, y0 + cell_height], color='black', linewidth=0.4, zorder=1)
                    ax.plot([x0, x0], [y0, y0 + cell_height], color='black', linewidth=0.4, zorder=1)
                    ax.plot([x0 + cell_width, x0 + cell_width], [y0, y0 + cell_height], color='black', linewidth=0.4, zorder=1)

                # Rewards (numbers)
                if show_rewards:
                    font_size = CELL_SIZE * 0.8 # Dynamic font size for clarity
                    ax.text(
                        x_center,
                        y_center,
                        f"{reward:.2f}",
                        ha='center',
                        va='center',
                        fontsize=font_size,
                        color='black',
                        alpha=0.6,
                        zorder=2
                    )

    # Final plot adjustments
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

def draw_pitch(
    ax=None,
    field_color='green',
    stripes=False,
    show_grid=False,
    show_heatmap=False,
    show_rewards=False,
    env=None
):
    """
    Draw the full 120x80m football pitch with 5m margin and optional RL overlays.

    Parameters:
        - ax (matplotlib.axes.Axes, optional): Axis to draw the pitch on.
        - field_color (str): Background color of the pitch.
        - stripes (bool): Whether to draw alternating mowing stripes.
        - show_grid (bool): Draws only the black grid lines (cells borders).
        - show_heatmap (bool): Fills cells with color based on reward value.
        - show_rewards (bool): Displays reward value as numbers in the cells.
        - env (OffensiveScenarioMoveSingleAgent, optional): Environment for accessing grid reward.

    Returns:
        - ax (matplotlib.axes.Axes): Axis with the completed pitch drawing.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Set field boundaries with margins
    x_min, x_max = X_MIN, X_MAX
    y_min, y_max = Y_MIN, Y_MAX

    # Background color
    if field_color == 'green':
        ax.set_facecolor('#4CAF50')
    else:
        ax.set_facecolor(field_color)

    lc = 'whitesmoke'
    border_color = 'white'

    # Optional mowing stripes
    if field_color == 'green' and stripes:
        for i in range(0, FIELD_WIDTH, STRIPE_WIDTH):
            stripe_color = '#43A047' if (i // STRIPE_WIDTH) % 2 == 0 else '#4CAF50'
            ax.add_patch(Rectangle(
                (i, 0), STRIPE_WIDTH, FIELD_HEIGHT,
                color=stripe_color, zorder=0
            ))

    # Draw outer pitch boundaries
    ax.plot([0, 0], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([FIELD_WIDTH, FIELD_WIDTH], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([0, FIELD_WIDTH], [0, 0], color=border_color, linewidth=3)
    ax.plot([0, FIELD_WIDTH], [FIELD_HEIGHT, FIELD_HEIGHT], color=border_color, linewidth=3)

    # Center line and circle
    ax.plot([HALF_FIELD_X, HALF_FIELD_X], [0, FIELD_HEIGHT], color=lc, linewidth=2, zorder=2)
    ax.add_patch(Circle((HALF_FIELD_X, CENTER_Y), 10, color=lc, fill=False, linewidth=2, zorder=2))

    # Penalty and goal areas (both sides)
    draw_penalty_area(ax, side='left', lc=lc)
    draw_goal(ax, side='left', lc=lc)
    draw_penalty_area(ax, side='right', lc=lc)
    draw_goal(ax, side='right', lc=lc)

    # Corner arcs
    corners = [(0, 0), (FIELD_WIDTH, 0), (0, FIELD_HEIGHT), (FIELD_WIDTH, FIELD_HEIGHT)]
    angles = [0, 90, 270, 180]
    for (x, y), angle in zip(corners, angles):
        arc = Arc((x, y), 4, 4, angle=angle, theta1=0, theta2=90, color=lc, linewidth=2, zorder=2)
        ax.add_patch(arc)

    # Draw grid / heatmap / reward only if env is provided
    if env is not None:
        num_cells_x = env.num_cells_x
        num_cells_y = env.num_cells_y
        cell_width = CELL_SIZE
        cell_height = CELL_SIZE

        for i in range(num_cells_x):
            for j in range(num_cells_y):
                x0 = X_MIN + i * cell_width
                y0 = Y_MIN + j * cell_height
                x_center = x0 + cell_width / 2
                y_center = y0 + cell_height / 2
                reward = env._get_position_reward(x_center, y_center)

                normalized_reward = (reward + 0.5) / 1.0  # [-0.5, 0.5] → [0, 1]
                color = plt.cm.coolwarm(normalized_reward)

                # Heatmap: filled colored cells
                if show_heatmap:
                    rect = Rectangle(
                        (x0, y0),
                        cell_width,
                        cell_height,
                        facecolor=color,
                        edgecolor='black' if show_grid else 'none',
                        linewidth=0.4,
                        alpha=0.6,
                        zorder=1
                    )
                    ax.add_patch(rect)

                # Grid: draw lines if not already done via heatmap edges
                if show_grid and not show_heatmap:
                    ax.plot([x0, x0 + cell_width], [y0, y0], color='black', linewidth=0.4, zorder=1)
                    ax.plot([x0, x0 + cell_width], [y0 + cell_height, y0 + cell_height], color='black', linewidth=0.4, zorder=1)
                    ax.plot([x0, x0], [y0, y0 + cell_height], color='black', linewidth=0.4, zorder=1)
                    ax.plot([x0 + cell_width, x0 + cell_width], [y0, y0 + cell_height], color='black', linewidth=0.4, zorder=1)

                # Rewards: annotate reward value inside the cell
                if show_rewards:
                    font_size = CELL_SIZE * 0.8  # Dynamic font size for clarity
                    ax.text(
                        x_center,
                        y_center,
                        f"{reward:.2f}",
                        ha='center',
                        va='center',
                        fontsize=font_size,
                        color='black',
                        alpha=0.6,
                        zorder=2
                    )

    # Final axis formatting
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax




