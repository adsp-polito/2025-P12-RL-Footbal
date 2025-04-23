import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Rectangle


# → Full pitch grid = N_COLS_FULL x N_ROWS = total cells
# → Half pitch grid = N_COLS_HALF x N_ROWS = total cells
# Example (if CELL_SIZE = 5):
# Full pitch grid = 24 x 16 = 384 cells
# Half pitch grid = 12 x 16 = 192 cells


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
STRIPE_WIDTH = 5
CENTER_Y = FIELD_HEIGHT // 2
CELL_SIZE = 5  # meters (e.g., 5x5m, or 1x1m for fine resolution)

# Render a football state on the offensive half-pitch
def render_state(state, ax=None, show_grid=False):
    """
    Render the current football state on the offensive half-pitch.

    Parameters:
        - state (dict): Current game state with:
            - 'player': instance of Player
            - 'ball': instance of Ball
            - 'opponents': list of Player instances
        - ax (matplotlib.axes.Axes, optional): Axis to draw on. If None, one is created.
        - show_grid (bool): Whether to overlay a debugging grid.

    Returns:
        - matplotlib.axes.Axes: Axis with the rendered state
    """

    # Draw the half pitch background
    ax = draw_half_pitch(ax=ax, show_grid=show_grid)

    # Draw attacking player (blue circle)
    if 'player' in state and state['player']:
        x, y = state['player'].get_position()
        ax.plot(x, y, 'o', color='dodgerblue', markersize=12, label='Attacker', zorder=3)

    # Draw ball (white circle)
    if 'ball' in state and state['ball']:
        x, y = state['ball'].get_position()
        ax.plot(x, y, 'o', color='white', markersize=7, label='Ball', zorder=3)

    # Draw defenders (red circles)
    opponents = state.get('opponents', [])
    for opp in opponents:
        x, y = opp.get_position()
        ax.plot(x, y, 'o', color='crimson', markersize=10, label='Defender', zorder=3)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc='lower left')

    return ax

# Draw the offensive half of a football pitch
def draw_half_pitch(ax=None, field_color='green', show_grid=False, show_cell_ids=False):
    """
    Draw the offensive half of a 120x80 meters football pitch (from 60 to 120 meters)

    Parameters:
        - ax (matplotlib.axes.Axes, optional): The axis to draw the pitch on. If None, a new figure and axis are created
        - field_color (str): Background color of the pitch. Defaults to 'green'
        - show_grid (bool): Whether to show grid lines on the pitch. Defaults to False
        - show_cell_ids (bool): Whether to show cell indices for debugging. Defaults to False

    Returns:
        - matplotlib.axes.Axes: The axis with the half pitch drawn on it
    """
    # Create a new figure and axis if none is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))

    # Set pitch background
    if field_color == 'green':
        ax.set_facecolor('#4CAF50')
        lc = 'whitesmoke'
        border_color = 'white'

        # Draw alternating green stripes
        for i in range(HALF_FIELD_X, FIELD_WIDTH, STRIPE_WIDTH):
            stripe_color = '#43A047' if ((i - HALF_FIELD_X) // STRIPE_WIDTH) % 2 == 0 else '#4CAF50'
            ax.add_patch(Rectangle((i, 0), STRIPE_WIDTH, FIELD_HEIGHT, color=stripe_color, zorder=0))
    else:
        lc = 'black'
        border_color = 'white'

    # Outer boundaries
    ax.plot([HALF_FIELD_X, HALF_FIELD_X], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([FIELD_WIDTH, FIELD_WIDTH], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([HALF_FIELD_X, FIELD_WIDTH], [0, 0], color=border_color, linewidth=3)
    ax.plot([HALF_FIELD_X, FIELD_WIDTH], [FIELD_HEIGHT, FIELD_HEIGHT], color=border_color, linewidth=3)

    # Center circle (half)
    center_arc = Arc((HALF_FIELD_X, CENTER_Y), 20, 20, angle=0, theta1=270, theta2=90, color=lc, linewidth=2)
    ax.add_patch(center_arc)

    # Penalty and goal areas
    ax.plot([FIELD_WIDTH, 102], [18, 18], color=lc, linewidth=2)
    ax.plot([102, 102], [18, 62], color=lc, linewidth=2)
    ax.plot([102, FIELD_WIDTH], [62, 62], color=lc, linewidth=2)

    ax.plot([FIELD_WIDTH, 114], [30, 30], color=lc, linewidth=2)
    ax.plot([114, 114], [30, 50], color=lc, linewidth=2)
    ax.plot([114, FIELD_WIDTH], [50, 50], color=lc, linewidth=2)

    ax.plot([109], [CENTER_Y], marker='o', markersize=5, color=lc)
    penalty_arc = Arc((108.2, CENTER_Y), 20, 20, angle=180, theta1=308, theta2=52, color=lc, linewidth=2)
    ax.add_patch(penalty_arc)

    # Goal 
    goal_y1 = CENTER_Y - 3.66
    ax.add_patch(Rectangle((FIELD_WIDTH, goal_y1), 2.44, 7.32, linewidth=2, edgecolor=lc, facecolor='none', zorder=0))

    # Corner arcs
    for (x, y), angle in zip([(FIELD_WIDTH, 0), (FIELD_WIDTH, FIELD_HEIGHT)], [90, 180]):
        ax.add_patch(Arc((x, y), 4, 4, angle=angle, theta1=0, theta2=90, color=lc, linewidth=2))

    # Optional grid
    if show_grid:
        for x in range(HALF_FIELD_X, FIELD_WIDTH + 1, CELL_SIZE):
            ax.plot([x, x], [0, FIELD_HEIGHT], color='black', linestyle='-', alpha=0.2, linewidth=0.5, zorder=1)
        for y in range(0, FIELD_HEIGHT + 1, CELL_SIZE):
            ax.plot([HALF_FIELD_X, FIELD_WIDTH], [y, y], color='black', linestyle='-', alpha=0.2, linewidth=0.5, zorder=1)

    # Optional cell indices for RL debugging
    if show_cell_ids:
        idx = 0
        for gx in range(HALF_FIELD_X, FIELD_WIDTH, CELL_SIZE):
            for gy in range(0, FIELD_HEIGHT, CELL_SIZE):
                cx = gx + CELL_SIZE / 2
                cy = gy + CELL_SIZE / 2
                ax.text(cx, cy, str(idx), 
                        fontsize=6, color='black', 
                        ha='center', va='center', 
                        alpha=0.4, zorder=2)
                idx += 1

    # Axis setup
    ax.set_xlim(HALF_FIELD_X - 5, FIELD_WIDTH + 5)
    ax.set_ylim(FIELD_HEIGHT + 5, -5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

# Draw the full pitch
def draw_pitch(ax=None, field_color='green', show_grid=False, show_cell_ids=False):
    """
    Draw a 120x80 meters football pitch

    Parameters:
        - ax (matplotlib.axes.Axes, optional): The axis to draw the pitch on. If None, a new figure and axis are created
        - field_color (str): Background color of the pitch. Defaults to 'green'
        - show_grid (bool): Whether to show grid lines on the pitch. Defaults to False
        - show_cell_ids(bool): Whether to show cell indices for debugging. Defaults to False

    Returns:
        - matplotlib.axes.Axes: The axis with the football pitch drawn on it
    """
    # Create a new figure and axis if none is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set pitch background
    if field_color == 'green':
        ax.set_facecolor('#4CAF50')
        lc = 'whitesmoke'
        border_color = 'white'
        stripe_width = 10

        # Draw alternating green stripes
        for i in range(0, FIELD_WIDTH, stripe_width):
            stripe_color = '#43A047' if (i // stripe_width) % 2 == 0 else '#4CAF50'
            ax.add_patch(Rectangle((i, 0), stripe_width, FIELD_HEIGHT, color=stripe_color, zorder=0))
    else:
        lc = 'black'
        border_color = 'white'

    # Outer boundaries
    ax.plot([0, 0], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([FIELD_WIDTH, FIELD_WIDTH], [0, FIELD_HEIGHT], color=border_color, linewidth=3)
    ax.plot([0, FIELD_WIDTH], [0, 0], color=border_color, linewidth=3)
    ax.plot([0, FIELD_WIDTH], [FIELD_HEIGHT, FIELD_HEIGHT], color=border_color, linewidth=3)

    # Center line and circle
    ax.plot([HALF_FIELD_X, HALF_FIELD_X], [0, FIELD_HEIGHT], color=lc, linewidth=2)
    ax.add_patch(Circle((HALF_FIELD_X, CENTER_Y), 10, color=lc, fill=False, linewidth=2))

    # Penalty areas and arcs (both sides)
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
        penalty_arc = Arc(arc_center, 20, 20, angle=arc_angle, theta1=308, theta2=52, color=lc, linewidth=2)
        ax.add_patch(penalty_arc)

    # Goals
    goal_y1 = CENTER_Y - 3.66
    left_goal = Rectangle((0, goal_y1), -2.44, 7.32, linewidth=2, edgecolor=lc, facecolor='none', zorder=0)
    right_goal = Rectangle((FIELD_WIDTH, goal_y1), 2.44, 7.32, linewidth=2, edgecolor=lc, facecolor='none', zorder=0)
    ax.add_patch(left_goal)
    ax.add_patch(right_goal)

    # Corners
    corner_positions = [(0, 0), (FIELD_WIDTH, 0), (0, FIELD_HEIGHT), (FIELD_WIDTH, FIELD_HEIGHT)]
    corner_angles = [0, 90, 270, 180]

    for (x, y), angle in zip(corner_positions, corner_angles):
        corner_arc = Arc((x, y), 4, 4, angle=angle, theta1=0, theta2=90, color=lc, linewidth=2)
        ax.add_patch(corner_arc)

    # Optional grid
    if show_grid:
        for x in range(0, FIELD_WIDTH + 1, CELL_SIZE):
            ax.plot([x, x], [0, FIELD_HEIGHT], color='black', linestyle='-', alpha=0.2, linewidth=0.5, zorder=1)
        for y in range(0, FIELD_HEIGHT + 1, CELL_SIZE):
            ax.plot([0, FIELD_WIDTH], [y, y], color='black', linestyle='-', alpha=0.2, linewidth=0.5, zorder=1)

    # Optional cell indices for RL debugging
    if show_cell_ids:
        idx = 0
        for gx in range(0, FIELD_WIDTH, CELL_SIZE):
            for gy in range(0, FIELD_HEIGHT, CELL_SIZE):
                cx = gx + CELL_SIZE / 2
                cy = gy + CELL_SIZE / 2
                ax.text(cx, cy, str(idx), 
                        fontsize=6, color='black', 
                        ha='center', va='center', 
                        alpha=0.4, zorder=2)
                idx += 1

    # Axis config
    ax.set_xlim(-5, FIELD_WIDTH + 5)
    ax.set_ylim(FIELD_HEIGHT + 5, -5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    return ax