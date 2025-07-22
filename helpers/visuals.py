import matplotlib.pyplot as plt
import matplotlib.animation as animation
from env.objects.pitch import draw_half_pitch, draw_pitch, X_MIN, Y_MIN, X_MAX, Y_MAX
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
def render_state(state, ax=None, stripes=False, full_pitch=False,
                 show_grid=False, show_heatmap=False, show_rewards=False, env=None):
    """
    Renders a single football state on the pitch.

    Parameters:
        - state (dict): Contains 'player', 'ball', and optionally 'opponents'.
        - ax (matplotlib.axes.Axes, optional): Axis to draw on. If None, creates a new figure.
        - stripes (bool): Whether to draw pitch stripes.
        - full_pitch (bool): If True, draws the full pitch; otherwise, half-pitch.
        - show_grid (bool): Draws grid lines on the field.
        - show_heatmap (bool): Fills cells with colors based on reward.
        - show_rewards (bool): Writes reward numbers in the center of each cell.
        - env (OffensiveScenarioMoveSingleAgent, optional): The environment used to compute the grid rewards.

    Returns:
        - matplotlib.axes.Axes: The axis with the rendered state.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Draw static pitch with the desired options
    if full_pitch:
        ax = draw_pitch(ax=ax, stripes=stripes,
                        show_grid=show_grid,
                        show_heatmap=show_heatmap,
                        show_rewards=show_rewards,
                        env=env)
    else:
        ax = draw_half_pitch(ax=ax, stripes=stripes,
                             show_grid=show_grid,
                             show_heatmap=show_heatmap,
                             show_rewards=show_rewards,
                             env=env)

    # Draw attacker (red circle)
    if 'player' in state and state['player']:
        x, y = state['player'].get_position()
        x = x * (X_MAX - X_MIN) + X_MIN
        y = y * (Y_MAX - Y_MIN) + Y_MIN
        ax.add_patch(Circle((x, y), radius=1.0, color='crimson', ec='black', lw=1, zorder=5))

    # Draw ball (white circle)
    if 'ball' in state and state['ball']:
        x, y = state['ball'].get_position()
        x = x * (X_MAX - X_MIN) + X_MIN
        y = y * (Y_MAX - Y_MIN) + Y_MIN
        ax.add_patch(Circle((x, y), radius=0.5, color='white', ec='black', lw=1, zorder=5))

    # Draw defenders (roles with colors)
    opponents = state.get('opponents', [])
    if opponents:
        for defender in opponents:
            x, y = defender.get_position()
            x = x * (X_MAX - X_MIN) + X_MIN
            y = y * (Y_MAX - Y_MIN) + Y_MIN
            role = defender.get_role()
            color = 'dodgerblue' if role == "DEF" else 'darkorange' if role == "GK" else 'grey'
            ax.add_patch(Circle((x, y), radius=1.0, color=color, ec='black', lw=1, zorder=5))

    return ax

# Render an episode as an animation
def render_episode(states, save_path=None, fps=24, stripes=False, full_pitch=False,
                   show_grid=False, show_heatmap=False, show_rewards=False, env=None):
    """
    Renders an episode as an animation.
    Parameters:
        - states (list): List of state dictionaries containing 'player', 'ball', and '
            opponents'.
        - save_path (str, optional): Path to save the animation file.
        - fps (int): Frames per second for the animation.
        - stripes (bool): Whether to draw pitch stripes.
        - full_pitch (bool): If True, draws the full pitch; otherwise, draws half-pitch.
        - show_grid (bool): If True, overlays the grid on the pitch.
        - show_heatmap (bool): If True, fills cells with colors.
        - show_rewards (bool): If True, annotates reward values inside the cells.
        - env (OffensiveScenarioMoveSingleAgent, optional): Environment instance for grid drawing.
    Returns:
        - matplotlib.animation.FuncAnimation: The animation object.
    """

    # Create a figure and axis for the animation
    fig, ax = plt.subplots(figsize=(6, 8))

    # Draw static pitch elements
    if full_pitch:
        draw_pitch(ax=ax, stripes=stripes, show_grid=show_grid, show_heatmap=show_heatmap, show_rewards=show_rewards, env=env)
    else:
        draw_half_pitch(ax=ax, stripes=stripes, show_grid=show_grid, show_heatmap=show_heatmap, show_rewards=show_rewards, env=env)

    dynamic_patches = []

    def update(frame_idx):
        # Remove previous dynamic patches
        for patch in dynamic_patches:
            patch.remove()
        dynamic_patches.clear()

        state = states[frame_idx]

        # Attacker
        if 'player' in state and state['player']:
            x, y = state['player'].get_position()
            x = x * (X_MAX - X_MIN) + X_MIN
            y = y * (Y_MAX - Y_MIN) + Y_MIN
            circle = Circle((x, y), radius=1.0, color='crimson', ec='black', lw=1, zorder=5)
            ax.add_patch(circle)
            dynamic_patches.append(circle)

        # Ball
        if 'ball' in state and state['ball']:
            x, y = state['ball'].get_position()
            x = x * (X_MAX - X_MIN) + X_MIN
            y = y * (Y_MAX - Y_MIN) + Y_MIN
            circle = Circle((x, y), radius=0.5, color='white', ec='black', lw=1, zorder=5)
            ax.add_patch(circle)
            dynamic_patches.append(circle)

        # Defenders
        opponents = state.get('opponents', [])
        for defender in opponents:
            x, y = defender.get_position()
            x = x * (X_MAX - X_MIN) + X_MIN
            y = y * (Y_MAX - Y_MIN) + Y_MIN
            role = defender.get_role()
            color = 'dodgerblue' if role == "DEF" else 'darkorange' if role == "GK" else 'grey'
            circle = Circle((x, y), radius=1.0, color=color, ec='black', lw=1, zorder=5)
            ax.add_patch(circle)
            dynamic_patches.append(circle)

        ax.set_title(f"Frame {frame_idx+1}/{len(states)}", fontsize=16, fontweight='bold', color='black')

    anim = animation.FuncAnimation(fig, update, frames=len(states), interval=1000/fps, repeat=False)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer='ffmpeg', fps=fps)

    return anim