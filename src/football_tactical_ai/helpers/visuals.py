import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Wedge
import numpy as np


# COORDINATE SYSTEMS AND NORMALIZATION
# The environment works on a normalized coordinate system [0, 1] × [0, 1],
# while the pitch visualization uses real physical units (meters).
#
# The pitch includes a margin around the official field to handle edge cases.
# Positions of all entities (players, ball) are normalized and need to be
# converted to absolute coordinates for rendering:
# x_absolute = x_normalized * (X_MAX - X_MIN) + X_MIN
# y_absolute = y_normalized * (Y_MAX - Y_MIN) + Y_MIN

def render_episode_singleAgent(states, pitch, save_path=None, fps=24, stripes=False, full_pitch=False,
                   show_grid=False, show_heatmap=False, show_rewards=False, reward_grid=None,
                   rewards_per_frame=None, show_info=True, show_fov = False):
    """
    Render an animated soccer episode with player, ball, defenders on the pitch.
    Displays info lines above the pitch with frame/time and reward stats.

    Parameters:
        states (list): List of dicts containing 'player', 'ball', 'opponents' objects.
        pitch (Pitch): Pitch object responsible for drawing pitch and grids.
        save_path (str or None): Path to save animation as gif/mp4. No save if None.
        fps (int): Frames per second for animation speed.
        stripes (bool): Whether to draw mowing stripes on pitch.
        full_pitch (bool): True to render full pitch, False for half pitch.
        show_grid (bool): Overlay grid lines if True.
        show_heatmap (bool): Show reward heatmap colors if True.
        show_rewards (bool): Display numeric reward values on grid cells if True.
        reward_grid (np.ndarray or None): Reward grid for heatmap coloring.
        rewards_per_frame (list or np.ndarray or None): Rewards for each frame.
        show_info (bool): Show detailed info lines above pitch if True (only for single Agent).
        show_fov (bool): Show field of view cone for the player if True.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """

    # Create figure and axis, draw pitch (full or half) with optional overlays
    if full_pitch:
        fig, ax = plt.subplots(figsize=(12, 8))
        pitch.draw_pitch(ax=ax, stripes=stripes, show_grid=show_grid,
                         show_heatmap=show_heatmap, show_rewards=show_rewards, reward_grid=reward_grid)
    else:
        fig, ax = plt.subplots(figsize=(6, 8))
        pitch.draw_half_pitch(ax=ax, stripes=stripes, show_grid=show_grid,
                              show_heatmap=show_heatmap, show_rewards=show_rewards, reward_grid=reward_grid)

    # Create patches representing players and ball; add to axis
    player_circle = Circle((0, 0), radius=1.0, color='crimson', ec='black', lw=1, zorder=5)
    ax.add_patch(player_circle)

    ball_circle = Circle((0, 0), radius=0.5, color='white', ec='black', lw=1, zorder=5)
    ax.add_patch(ball_circle)

    # Create a wedge for the field of view if show_fov is True
    fov_wedge = Wedge(center=(0, 0), r=20, theta1=0, theta2=0, color='crimson', alpha=0.15, zorder=3)
    ax.add_patch(fov_wedge)

    # Determine maximum number of defenders in any state, create corresponding circles
    max_opponents = max(len(state.get('opponents', [])) for state in states)
    defender_circles = []
    for _ in range(max_opponents):
        c = Circle((0, 0), radius=1.0, color='dodgerblue', ec='black', lw=1, zorder=5)
        ax.add_patch(c)
        defender_circles.append(c)

    # Dictionary to hold references to info text objects (for updating/removal)
    info_texts = {
        'line1_left': None,
        'line1_center': None,
        'line1_right': None,
        'line2_left': None,
        'line2_center': None,
        'line2_right': None,
        'frame_only': None,
    }

    def update(frame_idx):
        # Get the current frame state (positions and roles)
        state = states[frame_idx]

        # Update attacker position; denormalize coordinates to absolute meters
        if 'player' in state and state['player']:
            x, y = state['player'].get_position()
            x = x * (pitch.x_max - pitch.x_min) + pitch.x_min
            y = y * (pitch.y_max - pitch.y_min) + pitch.y_min
            player_circle.set_center((x, y))
            player_circle.set_visible(True)
        else:
            player_circle.set_visible(False)

        
        # Show Field of View (FOV) cone if enabled and player exists
        if show_fov and 'player' in state and state['player']:
            player = state['player']

            # Convert player's normalized position to meters
            px, py = player.get_position()
            px = px * (pitch.x_max - pitch.x_min) + pitch.x_min
            py = py * (pitch.y_max - pitch.y_min) + pitch.y_min

            # Get last movement direction to orient the FOV
            direction = getattr(player, "last_action_direction", np.array([1.0, 0.0]))  # fallback forward
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 0 else np.array([1.0, 0.0])

            # Compute orientation angle in degrees
            angle_deg = np.degrees(np.arctan2(direction[1], direction[0]))

            # Determine radius of vision cone from vision range factor and max range (in meters)
            view_range = getattr(player, "fov_range", 0.5)
            max_range = getattr(player, "max_fov_range", 90.0)
            view_radius = view_range * max_range
            fov_wedge.set_radius(view_radius)

            # Determine FOV angle (spread) from fov_angle [0,1] and max_fov_angle [0, 180]
            fov_factor = getattr(player, "fov_angle", 0.5)
            max_angle = getattr(player, "max_fov_angle", 180.0)
            fov_angle = fov_factor * max_angle

            # Update wedge parameters
            fov_wedge.set_center((px, py))
            fov_wedge.theta1 = angle_deg - fov_angle / 2
            fov_wedge.theta2 = angle_deg + fov_angle / 2
            fov_wedge.set_visible(True)
        else:
            fov_wedge.set_visible(False)


        # Update ball position similarly
        if 'ball' in state and state['ball']:
            x, y = state['ball'].get_position()
            x = x * (pitch.x_max - pitch.x_min) + pitch.x_min
            y = y * (pitch.y_max - pitch.y_min) + pitch.y_min
            ball_circle.set_center((x, y))
            ball_circle.set_visible(True)
        else:
            ball_circle.set_visible(False)

        # Update defenders: set positions, colors based on role, visibility
        opponents = state.get('opponents', [])
        for i, circle in enumerate(defender_circles):
            if i < len(opponents):
                x, y = opponents[i].get_position()
                x = x * (pitch.x_max - pitch.x_min) + pitch.x_min
                y = y * (pitch.y_max - pitch.y_min) + pitch.y_min
                role = opponents[i].get_role()
                # Color blue for defenders, orange for goalkeepers, gray otherwise
                color = 'dodgerblue' if role == "DEF" else 'darkorange' if role == "GK" else 'grey'
                circle.set_center((x, y))
                circle.set_color(color)
                circle.set_edgecolor('black')  # Ensure black border for visibility
                circle.set_visible(True)
            else:
                circle.set_visible(False)

        # Calculate elapsed time in seconds and total frames count
        elapsed_sec = frame_idx / fps
        total_frames = len(states)-1

        # Get current reward and cumulative reward if available
        if rewards_per_frame is not None:
            reward_val = rewards_per_frame[frame_idx]
            cum_reward = sum(rewards_per_frame[:frame_idx+1])
        else:
            reward_val = None
            cum_reward = None

        # Remove previous info text objects to update fresh text every frame
        for key in info_texts:
            if info_texts[key] is not None:
                info_texts[key].remove()
                info_texts[key] = None

        if show_info:
            center_x = 0.5  # Center horizontally in axis coordinates
            fontsize = 18
            base_y_top = 1.12     # Y position for first info line (frame/time)
            base_y_bottom = 1.06  # Y position for second info line (reward info)

            # Compose first line text parts with separators "|"
            left_text_1 = f"Frame: {frame_idx}/{total_frames}"
            center_text_1 = "|"
            right_text_1 = f"Time: {elapsed_sec:.2f} s"

            # Left-aligned frame info (right-aligned to separator)
            info_texts['line1_left'] = ax.text(
                center_x - 0.01, base_y_top, left_text_1,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='right',
                color='black',
            )
            # Center separator "|"
            info_texts['line1_center'] = ax.text(
                center_x, base_y_top, center_text_1,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='center',
                color='black',
            )
            # Right-aligned time info (left-aligned to separator)
            info_texts['line1_right'] = ax.text(
                center_x + 0.01, base_y_top, right_text_1,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='left',
                color='black',
            )

            # Compose second line texts (reward and cumulative reward)
            reward_str = f"Reward: {reward_val:.3f}" if reward_val is not None else "Reward: N/A"
            cum_str = f"Cumulative: {cum_reward:.3f}" if cum_reward is not None else "Cumulative: N/A"
            center_text_2 = "|"

            # Determine colors: green if >= 0, else red
            reward_color = 'green' if reward_val is not None and reward_val >= 0 else 'red'
            cum_color = 'green' if cum_reward is not None and cum_reward >= 0 else 'red'

            # Left aligned reward text (right aligned to separator)
            info_texts['line2_left'] = ax.text(
                center_x - 0.01, base_y_bottom, reward_str,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='right',
                color=reward_color,
            )
            # Center separator "|"
            info_texts['line2_center'] = ax.text(
                center_x, base_y_bottom, center_text_2,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='center',
                color='black',
            )
            # Right aligned cumulative reward text (left aligned to separator)
            info_texts['line2_right'] = ax.text(
                center_x + 0.01, base_y_bottom, cum_str,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='left',
                color=cum_color,
            )

        else:
            # If show_info=False, show only the frame number, centered and larger font
            center_x = 0.5
            base_y = 1.1
            fontsize = 24
            line_text = f"Frame: {frame_idx}/{total_frames}"
            info_texts['frame_only'] = ax.text(
                center_x, base_y, line_text,
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='center',
                color='black',
            )

        # Return all patches and text objects for blitting optimization
        return [player_circle, ball_circle, fov_wedge] + defender_circles + [t for t in info_texts.values() if t is not None]

    # Create the animation object with the update function and frame count
    anim = animation.FuncAnimation(fig, update, frames=len(states),
                                   interval=1000/fps, blit=True, repeat=False)

    # Save animation if requested
    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer='ffmpeg', fps=fps)

    return anim

def render_episode_multiAgent(states, pitch, save_path=None, fps=24, stripes=False, full_pitch=True,
                               show_grid=False, show_heatmap=False, show_rewards=False, reward_grid=None,
                               show_fov=False):
    """
    Render an animated soccer episode with multiple agents (attackers, defenders, goalkeeper).
    
    Parameters:
        states (list): List of dicts with keys 'players' (dict of agent_id:player), 'ball' (Ball object).
        pitch (Pitch): Pitch object responsible for drawing pitch and grids.
        save_path (str or None): Path to save animation as gif/mp4. No save if None.
        fps (int): Frames per second for animation speed.
        stripes (bool): Whether to draw mowing stripes on pitch.
        full_pitch (bool): True to render full pitch, False for half pitch.
        show_grid (bool): Overlay grid lines if True.
        show_heatmap (bool): Show reward heatmap colors if True.
        show_rewards (bool): Display numeric reward values on grid cells if True.
        reward_grid (np.ndarray or None): Reward grid for heatmap coloring (if applicable).
        show_fov (bool): Show FOV cones per agent if True.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object.
    """

      # Create figure and axis, draw pitch (full or half) with optional overlays
    if full_pitch:
        fig, ax = plt.subplots(figsize=(12, 8))
        pitch.draw_pitch(ax=ax, stripes=stripes, show_grid=show_grid,
                         show_heatmap=show_heatmap, show_rewards=show_rewards, reward_grid=reward_grid)
    else:
        fig, ax = plt.subplots(figsize=(6, 8))
        pitch.draw_half_pitch(ax=ax, stripes=stripes, show_grid=show_grid,
                              show_heatmap=show_heatmap, show_rewards=show_rewards, reward_grid=reward_grid)

    # Create player and FOV objects
    player_patches = {}
    fov_patches = {}
    colors_by_role = {"ATT": "crimson", "DEF": "dodgerblue", "GK": "darkorange"}

    # Infer all possible agent_ids
    all_agent_ids = list(states[0].get("players", {}).keys())
    for agent_id in all_agent_ids:
        player = states[0]["players"][agent_id]
        role = player.get_role()
        color = colors_by_role.get(role, "gray")

        circle = Circle((0, 0), radius=1.0, color=color, ec='black', lw=1, zorder=5)
        wedge = Wedge(center=(0, 0), r=20, theta1=0, theta2=0, color=color, alpha=0.15, zorder=3)

        ax.add_patch(circle)
        ax.add_patch(wedge)

        player_patches[agent_id] = circle
        fov_patches[agent_id] = wedge

    # Create the ball
    ball_circle = Circle((0, 0), radius=0.5, color='white', ec='black', lw=1, zorder=6)
    ax.add_patch(ball_circle)

    # Dictionary to hold references to info text objects (for updating/removal)
    info_texts = {
    'frame_only': ax.text(
        0.5, 1.1, "",
        transform=ax.transAxes,
        fontsize=24,
        fontweight='bold',
        verticalalignment='top',
        horizontalalignment='center',
        color='black',
    )
}

    def update(frame_idx):
        state = states[frame_idx]

        # Update players
        for agent_id, player in state.get("players", {}).items():
            px, py = player.get_position()
            px = px * (pitch.x_max - pitch.x_min) + pitch.x_min
            py = py * (pitch.y_max - pitch.y_min) + pitch.y_min
            player_patches[agent_id].set_center((px, py))
            player_patches[agent_id].set_visible(True)

            if show_fov:
                direction = getattr(player, "last_action_direction", np.array([1.0, 0.0]))
                norm = np.linalg.norm(direction)
                direction = direction / norm if norm > 0 else np.array([1.0, 0.0])
                angle_deg = np.degrees(np.arctan2(direction[1], direction[0]))

                view_range = getattr(player, "fov_range", 0.5)
                max_range = getattr(player, "max_fov_range", 90.0)
                radius = view_range * max_range

                fov_factor = getattr(player, "fov_angle", 0.5)
                max_angle = getattr(player, "max_fov_angle", 180.0)
                fov_angle = fov_factor * max_angle

                fov_patches[agent_id].set_center((px, py))
                fov_patches[agent_id].theta1 = angle_deg - fov_angle / 2
                fov_patches[agent_id].theta2 = angle_deg + fov_angle / 2
                fov_patches[agent_id].set_radius(radius)
                fov_patches[agent_id].set_visible(True)
            else:
                fov_patches[agent_id].set_visible(False)

        # Update ball
        if 'ball' in state and state['ball']:
            bx, by = state['ball'].get_position()
            bx = bx * (pitch.x_max - pitch.x_min) + pitch.x_min
            by = by * (pitch.y_max - pitch.y_min) + pitch.y_min
            ball_circle.set_center((bx, by))
            ball_circle.set_visible(True)
        else:
            ball_circle.set_visible(False)

        # Calculate total frames count
        total_frames = len(states)-1
        
        # Update info text with current frame
        line_text = f"Frame: {frame_idx}/{total_frames}"
        info_texts['frame_only'].set_text(line_text)

        return list(player_patches.values()) + list(fov_patches.values()) + [ball_circle]

    anim = animation.FuncAnimation(fig, update, frames=len(states), interval=1000/fps, blit=True, repeat=False)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer='ffmpeg', fps=fps)

    return anim

