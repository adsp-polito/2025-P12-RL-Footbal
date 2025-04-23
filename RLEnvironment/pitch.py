import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Rectangle

# === GRID INFO (used for RL) ===
# The pitch is overlaid with a 5x5m cell grid for RL training, visual debugging, and reward shaping.
#
# → Full pitch grid:
#     - Area: 130m (x) × 90m (y) = extended full pitch with 5m margin
#     - N_COLS_FULL = 130 / 5 = 26
#     - N_ROWS = 90 / 5 = 18
#     - → Total cells: 26 × 18 = 468
#
# → Half pitch grid:
#     - Area: 70m (x) × 90m (y) = extended half pitch with 5m margin
#     - N_COLS_HALF = 70 / 5 = 14
#     - N_ROWS = 90 / 5 = 18
#     - → Total cells: 14 × 18 = 252

# === Z-ORDER LAYERS EXPLANATION ===
# Z-order defines the drawing priority (higher values appear on top)
# We use the following convention to control rendering layers:

# zorder = 0 → Field background stripes and goals (base layer)
# zorder = 1 → Debug grid lines (optional for RL debugging)
# zorder = 2 → Static pitch elements:
#              - Field boundaries
#              - Penalty areas and goal areas
#              - Center line and circle
#              - Penalty arcs and corner arcs
#              - Text labels for cell indices (debugging)
#              These are not explicitly defined with zorder unless needed, as matplotlib defaults place them between z=1 and z=3
# zorder = 3 → Dynamic elements (always visible on top):
#              - Attacking player (blue)
#              - Defenders (red)
#              - Ball (white)

# Most matplotlib drawing functions (like ax.plot, ax.add_patch) default to zorder=2
# We make it explicit where needed to ensure consistent rendering layers

# Global field dimensions
FIELD_WIDTH = 120
FIELD_HEIGHT = 80
HALF_FIELD_X = FIELD_WIDTH // 2
CENTER_Y = FIELD_HEIGHT // 2
CELL_SIZE = 5  # meters (e.g., 5x5m, or 1x1m for fine resolution)
STRIPE_WIDTH = 5

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

    # Set pitch background
    if field_color == 'green':
        ax.set_facecolor('#4CAF50')
        lc = 'whitesmoke'
        border_color = 'white'
        if stripes:
            for i in range(HALF_FIELD_X, FIELD_WIDTH, STRIPE_WIDTH):
                stripe_color = '#43A047' if ((i - HALF_FIELD_X) // STRIPE_WIDTH) % 2 == 0 else '#4CAF50'
                ax.add_patch(Rectangle((i, 0), STRIPE_WIDTH, FIELD_HEIGHT, color=stripe_color, zorder=0))
    else:
        lc = 'black'
        border_color = 'white'

    # Outer pitch boundaries
    ax.plot([HALF_FIELD_X, HALF_FIELD_X], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([FIELD_WIDTH, FIELD_WIDTH], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([HALF_FIELD_X, FIELD_WIDTH], [0, 0], color=border_color, linewidth=3)
    ax.plot([HALF_FIELD_X, FIELD_WIDTH], [FIELD_HEIGHT, FIELD_HEIGHT], color=border_color, linewidth=3)

    # Center arc
    ax.add_patch(Arc((HALF_FIELD_X, CENTER_Y), 20, 20, angle=0, theta1=270, theta2=90, color=lc, linewidth=2))

    # Penalty and goal areas
    ax.plot([FIELD_WIDTH, 102], [18, 18], color=lc, linewidth=2)
    ax.plot([102, 102], [18, 62], color=lc, linewidth=2)
    ax.plot([102, FIELD_WIDTH], [62, 62], color=lc, linewidth=2)
    ax.plot([FIELD_WIDTH, 114], [30, 30], color=lc, linewidth=2)
    ax.plot([114, 114], [30, 50], color=lc, linewidth=2)
    ax.plot([114, FIELD_WIDTH], [50, 50], color=lc, linewidth=2)
    ax.plot([109], [CENTER_Y], marker='o', markersize=5, color=lc)
    ax.add_patch(Arc((108.2, CENTER_Y), 20, 20, angle=180, theta1=308, theta2=52, color=lc, linewidth=2))

    # Goal
    goal_y1 = CENTER_Y - 3.66
    ax.add_patch(Rectangle((FIELD_WIDTH, goal_y1), 2.44, 7.32, linewidth=2, edgecolor=lc, facecolor='none', zorder=0))

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
    x_min = -CELL_SIZE                    # -5
    x_max = FIELD_WIDTH + CELL_SIZE      # 125
    y_min = -CELL_SIZE                   # -5
    y_max = FIELD_HEIGHT + CELL_SIZE     # 85

    # Set background color
    if field_color == 'green':
        ax.set_facecolor('#4CAF50')
        lc = 'whitesmoke'
        border_color = 'white'
        if stripes:
            for i in range(0, FIELD_WIDTH, STRIPE_WIDTH):
                stripe_color = '#43A047' if (i // STRIPE_WIDTH) % 2 == 0 else '#4CAF50'
                ax.add_patch(Rectangle((i, 0), STRIPE_WIDTH, FIELD_HEIGHT, color=stripe_color, zorder=0))
    else:
        lc = 'black'
        border_color = 'white'

    # Field boundary
    ax.plot([0, 0], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([FIELD_WIDTH, FIELD_WIDTH], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([0, FIELD_WIDTH], [0, 0], color=border_color, linewidth=3)
    ax.plot([0, FIELD_WIDTH], [FIELD_HEIGHT, FIELD_HEIGHT], color=border_color, linewidth=3)

    # Center line and circle
    ax.plot([HALF_FIELD_X, HALF_FIELD_X], [0, FIELD_HEIGHT], color=lc, linewidth=2)
    ax.add_patch(Circle((HALF_FIELD_X, CENTER_Y), 10, color=lc, fill=False, linewidth=2))

    # Penalty and goal areas (both sides)
    for x in [0, FIELD_WIDTH]:
        penalty_x = 18 if x == 0 else 102
        goal_x = 6 if x == 0 else 114
        penalty_spot = 11 if x == 0 else 109
        arc_center = (12, CENTER_Y) if x == 0 else (108.2, CENTER_Y)
        arc_angle = 0 if x == 0 else 180

        ax.plot([x, penalty_x], [18, 18], color=lc, linewidth=2)
        ax.plot([penalty_x, penalty_x], [18, 62], color=lc, linewidth=2)
        ax.plot([x, penalty_x], [62, 62], color=lc, linewidth=2)

        ax.plot([x, goal_x], [30, 30], color=lc, linewidth=2)
        ax.plot([goal_x, goal_x], [30, 50], color=lc, linewidth=2)
        ax.plot([x, goal_x], [50, 50], color=lc, linewidth=2)

        ax.plot([penalty_spot], [CENTER_Y], marker='o', markersize=5, color=lc)
        ax.add_patch(Arc(arc_center, 20, 20, angle=arc_angle, theta1=308, theta2=52, color=lc, linewidth=2))

    # Goals
    goal_y1 = CENTER_Y - 3.66
    ax.add_patch(Rectangle((0, goal_y1), -2.44, 7.32, linewidth=2, edgecolor=lc, facecolor='none', zorder=0))
    ax.add_patch(Rectangle((FIELD_WIDTH, goal_y1), 2.44, 7.32, linewidth=2, edgecolor=lc, facecolor='none', zorder=0))

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