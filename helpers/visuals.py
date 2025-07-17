import matplotlib.pyplot as plt
import matplotlib.animation as animation
from env.pitch import draw_half_pitch, FIELD_WIDTH, FIELD_HEIGHT
from matplotlib.patches import Circle

# COORDINATE SYSTEMS AND NORMALIZATION

# The environment operates on a normalized coordinate system [0, 1] × [0, 1],
# while the pitch visualization uses absolute physical units in meters.
#
# The physical field space includes a padded area around the official pitch
# to accommodate edge cases and grid alignment:
#   - Horizontal range (X): [X_MIN, X_MAX] (e.g., [-5, 65] or [55, 125])
#   - Vertical range   (Y): [Y_MIN, Y_MAX] (e.g., [-5, 85])
#
# This design allows:
#   - A learning-friendly normalized space for RL agents.
#   - Accurate and flexible rendering with grid overlays and real dimensions.
#
# All entity positions (players, ball) are stored in normalized units and
# must be denormalized before rendering:
#   - x_absolute = x_normalized * PITCH_WIDTH + X_MIN
#   - y_absolute = y_normalized * PITCH_HEIGHT + Y_MIN
#
# Example:
#   x_normalized = 0.5, PITCH_WIDTH = 70, X_MIN = -5 →
#   x_absolute = 0.5 * 70 + (-5) = 30 meters (center of the official pitch)
#
# This ensures consistent and interpretable placement of entities across both
# the simulation and the rendered visualization.

# Render a single state
def render_state(state, ax=None, show_grid=False, show_cell_ids=False, stripes=False):
    """
    Renders a single football state on the offensive half-pitch.

    Parameters:
        - state (dict): Contains 'player', 'ball', and optionally 'opponents'.
            Example:
                state = {
                    'player': Player instance,
                    'ball': Ball instance,
                    'opponents': list of Player instances (optional)
                }
        - ax (matplotlib.axes.Axes, optional): Axis to draw on. If None, a new one is created.
        - show_grid (bool): Whether to overlay a debug grid.
        - show_cell_ids (bool): Whether to show cell indices for RL debugging.
        - stripes (bool): Whether to draw pitch stripes.

    Returns:
        - matplotlib.axes.Axes: The axis with the rendered state.
    """
    ax = draw_half_pitch(ax=ax, show_grid=show_grid, show_cell_ids=show_cell_ids, stripes=stripes)

   # Draw attacker (blue)
    if 'player' in state and state['player']:
        x, y = state['player'].get_position()
        x *= FIELD_WIDTH
        y *= FIELD_HEIGHT
        ax.add_patch(Circle((x, y), radius=1.0, color='dodgerblue', ec='black', lw=1, zorder=3))

    # Draw ball (white)
    if 'ball' in state and state['ball']:
        x, y = state['ball'].get_position()
        x *= FIELD_WIDTH
        y *= FIELD_HEIGHT
        ax.add_patch(Circle((x, y), radius=0.5, color='white', ec='black', lw=1, zorder=3))

    # Draw defenders (red)
    opponents = state.get('opponents', [])
    if opponents:
        for defender in opponents:
            x, y = defender.get_position()
            x *= FIELD_WIDTH
            y *= FIELD_HEIGHT
            ax.add_patch(Circle((x, y), radius=1.0, color='crimson', ec='black', lw=1, zorder=3))

    return ax

# Render episode as animation
def render_episode(states, save_path=None, fps=24, show_grid=False, show_cell_ids=False):
    """
    Creates and optionally saves an animation from a list of game states.

    Parameters:
        - states (list of dict): Sequence of environment states (one per timestep)
        - save_path (str or None): Path to save the animation (e.g., 'out.mp4' or 'out.gif')
        - fps (int): Frames per second for the animation
        - show_grid (bool): Whether to overlay the debug grid on each frame
        - show_cell_ids (bool): Whether to show grid cell indices

    Returns:
        - matplotlib.animation.FuncAnimation object
    """
    fig, ax = plt.subplots(figsize=(6, 8))
     
    def update(frame_idx):
        ax.clear()
        render_state(states[frame_idx], ax=ax, show_grid=show_grid, show_cell_ids=show_cell_ids)
        ax.set_title(f"Frame {frame_idx+1}/{len(states)}", fontsize=16, fontweight='bold', color='black')

    anim = animation.FuncAnimation(fig, update, frames=len(states), interval=1000/fps, repeat=False)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer='ffmpeg', fps=fps)

    return anim