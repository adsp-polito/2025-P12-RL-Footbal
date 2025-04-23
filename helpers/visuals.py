import matplotlib.pyplot as plt
import matplotlib.animation as animation
from RLEnvironment.pitch import draw_half_pitch, FIELD_WIDTH, FIELD_HEIGHT
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D


# =============================================================================
# Coordinate System & Normalization
# -----------------------------------------------------------------------------
# This visualization module renders the football pitch using real-world
# dimensions in meters (e.g., 120x80). However, entities like players and
# the ball use normalized coordinates in the range [0, 1].
#
# This design choice allows the simulation and RL environment to operate
# on normalized values (which are model-friendly), while keeping the rendering
# faithful to the real field layout.
#
# As a result, when rendering entities (e.g., players, ball), their normalized
# coordinates must be denormalized by multiplying by FIELD_WIDTH and FIELD_HEIGHT.
#
# Example:
#   x_normalized = 0.5 → x_position_on_pitch = 0.5 * 120 = 60
#
# This ensures consistent rendering and accurate placement on the physical pitch.
# =============================================================================

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