import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# COORDINATE SYSTEMS AND NORMALIZATION

# The environment operates on a normalized coordinate system [0, 1] × [0, 1],
# while the pitch visualization uses absolute physical units in meters.
#
# The physical field space includes a padded area around the official pitch
# to accommodate edge cases and grid alignment:
#   - Horizontal range (X): [X_MIN, X_MAX]
#   - Vertical range   (Y): [Y_MIN, Y_MAX]
#
# All entity positions (players, ball) are stored in normalized units and
# must be denormalized before rendering:
#   - x_absolute = x_normalized * (X_MAX - X_MIN) + X_MIN
#   - y_absolute = y_normalized * (Y_MAX - Y_MIN) + Y_MIN

def render_episode(states, pitch, save_path=None, fps=24, stripes=False, full_pitch=False,
                   show_grid=False, show_heatmap=False, show_rewards=False, reward_grid=None):

    """
    Renders an episode as an animation.

    Parameters:
        states (list): List of state dictionaries containing 'player', 'ball', and 'opponents'.
        pitch (Pitch): The pitch object used for drawing.
        save_path (str, optional): Path to save the animation file (gif or mp4).
        fps (int): Frames per second for the animation.
        stripes (bool): Whether to draw pitch stripes.
        full_pitch (bool): If True, draws the full pitch; otherwise, draws half-pitch.
        show_grid (bool): If True, overlays the grid on the pitch.
        show_heatmap (bool): If True, fills cells with colors.
        show_rewards (bool): If True, annotates reward values inside the cells.
        reward_grid (np.ndarray, optional): Precomputed reward grid for the episode.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """

    # Create a figure and axis for the animation
    fig, ax = plt.subplots(figsize=(6, 8))

    # Draw static pitch elements
    if full_pitch:
        pitch.draw_pitch(ax=ax, stripes=stripes, show_grid=show_grid,
                         show_heatmap=show_heatmap, show_rewards=show_rewards, reward_grid=reward_grid)
    else:
        pitch.draw_half_pitch(ax=ax, stripes=stripes, show_grid=show_grid,
                              show_heatmap=show_heatmap, show_rewards=show_rewards, reward_grid=reward_grid)

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
            x = x * (pitch.X_MAX - pitch.X_MIN) + pitch.X_MIN
            y = y * (pitch.Y_MAX - pitch.Y_MIN) + pitch.Y_MIN
            circle = Circle((x, y), radius=1.0, color='crimson', ec='black', lw=1, zorder=5)
            ax.add_patch(circle)
            dynamic_patches.append(circle)

        # Ball
        if 'ball' in state and state['ball']:
            x, y = state['ball'].get_position()
            x = x * (pitch.X_MAX - pitch.X_MIN) + pitch.X_MIN
            y = y * (pitch.Y_MAX - pitch.Y_MIN) + pitch.Y_MIN
            circle = Circle((x, y), radius=0.5, color='white', ec='black', lw=1, zorder=5)
            ax.add_patch(circle)
            dynamic_patches.append(circle)

        # Defenders
        opponents = state.get('opponents', [])
        for defender in opponents:
            x, y = defender.get_position()
            x = x * (pitch.X_MAX - pitch.X_MIN) + pitch.X_MIN
            y = y * (pitch.Y_MAX - pitch.Y_MIN) + pitch.Y_MIN
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