import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Rectangle

# === GRID INFORMATION (USED FOR RL) ===
# The pitch is overlaid with a 5x5m cell grid for RL training, visual debugging, and reward shaping.

# → Full pitch grid:
#     - Area: 130m (x) × 90m (y) = full pitch with 5m margin on all sides
#     - N_COLS_FULL = 130 / 5 = 26 columns
#     - N_ROWS = 90 / 5 = 18 rows
#     - Total cells: 26 × 18 = 468

# → Half pitch grid:
#     - Area: 70m (x) × 90m (y) = half pitch with 5m margin on all sides
#     - N_COLS_HALF = 70 / 5 = 14 columns
#     - N_ROWS = 90 / 5 = 18 rows
#     - Total cells: 14 × 18 = 252

# === Z-ORDER LAYERS EXPLANATION ===
# Z-order defines the drawing priority (higher values appear on top)
# We use the following convention to control rendering layers:

# Z-ORDER LAYERS EXPLANATION
# Z-order defines the drawing priority (higher values appear on top).
# The following convention is used to control rendering layers:

# zorder = 0 → Field background stripes and goals (base layer)
# zorder = 1 → Debug grid lines (optional, for RL debugging)
# zorder = 2 → Static pitch elements:
#              - Field boundaries
#              - Penalty areas and goal areas
#              - Center line and circle
#              - Penalty arcs and corner arcs
#              - Text labels for cell indices (debugging)
#              These are usually drawn without explicitly setting zorder, as matplotlib defaults place them between z=1 and z=3.
# zorder = 3 → Dynamic elements (always drawn on top):
#              - Attacking player (blue)
#              - Defenders (red)
#              - Ball (white)

# Most matplotlib drawing functions (e.g., ax.plot, ax.add_patch) default to zorder=2.
# We make it explicit where needed to ensure consistent rendering layers.

# Global field dimensions
FIELD_WIDTH = 120
FIELD_HEIGHT = 80
HALF_FIELD_X = FIELD_WIDTH // 2
CENTER_Y = FIELD_HEIGHT // 2
CELL_SIZE = 5  # meters (e.g., 5x5m, or 1x1m for fine resolution)
STRIPE_WIDTH = 5

# Fields Margins (for a 120x80 pitch)
X_MIN, Y_MIN = -5, -5
X_MAX, Y_MAX = 125, 85

# Fields Proportions (for a 120x80 pitch)
PENALTY_AREA_DEPTH = 18           # 18 meters depth
PENALTY_AREA_HEIGHT = 44          # from y=18 to y=62 (62-18 = 44 meters)
GOAL_AREA_DEPTH = 6               # 6 meters depth
GOAL_AREA_HEIGHT = 20             # from y=30 to y=50 (50-30 = 20 meters)
PENALTY_SPOT_X_RIGHT = 109        # x position for right side
PENALTY_SPOT_X_LEFT = 11          # x position for left side
CENTER_CIRCLE_RADIUS = 10         # 10 meters radius
GOAL_HEIGHT = 7.32                # 7.32 meters (standard)
GOAL_DEPTH = 2.44                 # 2.44 meters depth

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
    ax.add_patch(Rectangle((x_goal, goal_y1), direction * GOAL_DEPTH, GOAL_HEIGHT, linewidth=2, edgecolor=lc, facecolor='none', zorder=0))


# Draw the offensive half of the pitch
def draw_half_pitch(ax=None, field_color='green', show_grid=False, show_cell_ids=False, stripes=False):
    """
    Draw the offensive half of a 120x80 meters football pitch with 5m padding.

    Parameters:
        - ax (matplotlib.axes.Axes, optional): Axis to draw the pitch on.
        - field_color (str): Pitch background color.
        - show_grid (bool): Whether to show debug grid.
        - show_cell_ids (bool): Whether to show grid cell indices.
        - stripes (bool): Whether to draw alternating mowing stripes.

    Returns:
        - matplotlib.axes.Axes: The axis with the half pitch drawn on it.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))

    # Derived fixed margins
    x_min = 55
    x_max = 125
    y_min = -5
    y_max = 85

    # Set background color
    if field_color == 'green':
        ax.set_facecolor('#4CAF50')
    else:
        ax.set_facecolor(field_color)

    lc = 'whitesmoke'
    border_color = 'white'

    # Draw stripes only if green field is selected
    if field_color == 'green' and stripes:
        for i in range(0, FIELD_WIDTH, STRIPE_WIDTH):
            stripe_color = '#43A047' if (i // STRIPE_WIDTH) % 2 == 0 else '#4CAF50'
            ax.add_patch(
                Rectangle(
                    (i, 0),
                    STRIPE_WIDTH,
                    FIELD_HEIGHT,
                    color=stripe_color,
                    zorder=0
                )
            )

    # Outer pitch boundaries
    ax.plot([HALF_FIELD_X, HALF_FIELD_X], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([FIELD_WIDTH, FIELD_WIDTH], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([HALF_FIELD_X, FIELD_WIDTH], [0, 0], color=border_color, linewidth=3)
    ax.plot([HALF_FIELD_X, FIELD_WIDTH], [FIELD_HEIGHT, FIELD_HEIGHT], color=border_color, linewidth=3)

    # Center arc
    ax.add_patch(Arc((HALF_FIELD_X, CENTER_Y), 20, 20, angle=0, theta1=270, theta2=90, color=lc, linewidth=2))

    # Draw penalty area, goal area and goal on the right side
    draw_penalty_area(ax, side='right', lc=lc)
    draw_goal(ax, side='right', lc=lc)

    # Corner arcs
    for (x, y), angle in zip([(FIELD_WIDTH, 0), (FIELD_WIDTH, FIELD_HEIGHT)], [90, 180]):
        ax.add_patch(Arc((x, y), 4, 4, angle=angle, theta1=0, theta2=90, color=lc, linewidth=2))

    # Optional debug grid
    if show_grid:
        for x in range(x_min, x_max + 1, CELL_SIZE):
            ax.plot([x, x], [y_min, y_max], color='black', linestyle='-', alpha=0.2, linewidth=0.5, zorder=1)
        for y in range(y_min, y_max + 1, CELL_SIZE):
            ax.plot([x_min, x_max], [y, y], color='black', linestyle='-', alpha=0.2, linewidth=0.5, zorder=1)

    # Optional cell IDs
    if show_cell_ids:
        idx = 0
        for gx in range(x_min, x_max, CELL_SIZE):
            for gy in range(y_min, y_max, CELL_SIZE):
                cx = gx + CELL_SIZE / 2
                cy = gy + CELL_SIZE / 2
                ax.text(cx, cy, str(idx), fontsize=6, color='black', ha='center', va='center', alpha=0.4, zorder=2)
                idx += 1

    # Axis configuration
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

# Draw the full pitch
def draw_pitch(ax=None, field_color='green', show_grid=False, show_cell_ids=False, stripes=False):
    """
    Draw a 120x80 meters football pitch with 5m outer buffer for debugging.

    Parameters:
        - ax (matplotlib.axes.Axes, optional): Axis to draw on
        - field_color (str): Background color of the pitch
        - show_grid (bool): Whether to draw the debug grid
        - show_cell_ids (bool): Whether to number grid cells
        - stripes (bool): Show alternating mowing stripes

    Returns:
        - matplotlib.axes.Axes: The axis with the full pitch rendered
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Derived limits for full pitch
    x_min = -CELL_SIZE                   # -5
    x_max = FIELD_WIDTH + CELL_SIZE      # 125
    y_min = -CELL_SIZE                   # -5
    y_max = FIELD_HEIGHT + CELL_SIZE     # 85

    # Set background color
    if field_color == 'green':
        ax.set_facecolor('#4CAF50')
    else:
        ax.set_facecolor(field_color)

    lc = 'whitesmoke'
    border_color = 'white'

    # Draw stripes only if green field is selected
    if field_color == 'green' and stripes:
        for i in range(0, FIELD_WIDTH, STRIPE_WIDTH):
            stripe_color = '#43A047' if (i // STRIPE_WIDTH) % 2 == 0 else '#4CAF50'
            ax.add_patch(
                Rectangle(
                    (i, 0),
                    STRIPE_WIDTH,
                    FIELD_HEIGHT,
                    color=stripe_color,
                    zorder=0
                )
            )


    # Field boundary
    ax.plot([0, 0], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([FIELD_WIDTH, FIELD_WIDTH], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([0, FIELD_WIDTH], [0, 0], color=border_color, linewidth=3)
    ax.plot([0, FIELD_WIDTH], [FIELD_HEIGHT, FIELD_HEIGHT], color=border_color, linewidth=3)

    # Center line and circle
    ax.plot([HALF_FIELD_X, HALF_FIELD_X], [0, FIELD_HEIGHT], color=lc, linewidth=2)
    ax.add_patch(Circle((HALF_FIELD_X, CENTER_Y), 10, color=lc, fill=False, linewidth=2))

    # Penalty and goal areas (both sides)
    draw_penalty_area(ax, side='left', lc=lc)
    draw_goal(ax, side='left', lc=lc)
    draw_penalty_area(ax, side='right', lc=lc)
    draw_goal(ax, side='right', lc=lc)

    # Corner arcs
    corners = [(0, 0), (FIELD_WIDTH, 0), (0, FIELD_HEIGHT), (FIELD_WIDTH, FIELD_HEIGHT)]
    angles = [0, 90, 270, 180]
    for (x, y), angle in zip(corners, angles):
        arc = Arc((x, y), 4, 4, angle=angle, theta1=0, theta2=90, color=lc, linewidth=2)
        ax.add_patch(arc)

    # Debug grid (optional)
    if show_grid:
        for x in range(x_min, x_max + 1, CELL_SIZE):
            ax.plot([x, x], [y_min, y_max], color='black', linestyle='-', alpha=0.2, linewidth=0.5, zorder=1)
        for y in range(y_min, y_max + 1, CELL_SIZE):
            ax.plot([x_min, x_max], [y, y], color='black', linestyle='-', alpha=0.2, linewidth=0.5, zorder=1)

    # Cell indices (optional)
    if show_cell_ids:
        idx = 0
        for gx in range(x_min, x_max, CELL_SIZE):
            for gy in range(y_min, y_max, CELL_SIZE):
                cx = gx + CELL_SIZE / 2
                cy = gy + CELL_SIZE / 2
                ax.text(cx, cy, str(idx), fontsize=6, color='black', ha='center', va='center', alpha=0.4, zorder=2)
                idx += 1

    # Axis setup
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax