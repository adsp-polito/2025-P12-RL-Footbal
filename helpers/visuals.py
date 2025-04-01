import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np
from helpers.drawPitch import draw_pitch


def render_single_frame(env, title="Tactical Frame"):
    """
    Renders a single frame of the environment

    Args:
        - env (FootballEnv): The current environment instance with player states.
        - title (str): Title displayed above the pitch.
    """
    _, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    # Draw players
    team_colors = [env.teams[0].color] * 11 + [env.teams[1].color] * 11
    for i, player in enumerate(env.players):
        pos = player.get_position() * np.array([120, 80])
        ax.plot(pos[0], pos[1], 'o', markersize=10, color=team_colors[i], zorder=5)

    # Title and display
    ax.set_title(title, fontsize=18, color='black', fontweight = "bold", pad=15)
    plt.show()


def animate_simulation(env, num_frames=48, interval_ms=1000/24):
    """
    Animates a full simulation using FuncAnimation
    Players move frame by frame according to random (or future learned) actions

    The simulation is rendered on a 120x80 meter pitch using matplotlib
    Each frame reflects the new positions of all players and the ball
    The animation timing is controlled by the interval_ms parameter,
    which defines the delay (in milliseconds) between each frame,
    allowing control over playback speed (e.g., for real-time 24 FPS use 1000/24)

    Args:
        - env (FootballEnv): The initialized football environment
        - num_frames (int): Total number of frames to simulate and render
        - interval_ms (float): Milliseconds between frames (controls playback speed, e.g., 1000/24 for 24 FPS)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    # Initial player dots setup
    player_dots = []
    role_labels = []
    team_colors = [env.teams[0].color] * 11 + [env.teams[1].color] * 11
    for i, player in enumerate(env.players):
        pos = player.get_position() * np.array([120, 80])
        dot, = ax.plot(pos[0], pos[1], 'o', markersize=10, color=team_colors[i], markeredgecolor="black", zorder=5)
        # Determine label for each player (prefer abbreviation if available)
        if hasattr(player, 'abbr'):
            label_text = player.abbr
        else:
            label_text = player.role[:2]  # fallback: use first two letters of role

        label = ax.text(
            pos[0], pos[1] - 2,
            label_text,
            ha='center', va='center',
            fontsize=8, color='black', fontweight='bold', zorder=6
        )
        player_dots.append(dot)
        role_labels.append(label)

    # Ball dot (black, above players)
    ball_pos = env.ball.get_position() * np.array([120, 80])
    ball_circle = Circle(
        (ball_pos[0], ball_pos[1]),
        radius=0.65,               
        facecolor='white',
        edgecolor='black',
        linewidth=1.5,
        zorder=5
    )
    ax.add_patch(ball_circle)

    # Title
    # interval_ms is now passed as an argument
    interval_s = interval_ms / 1000
    title = ax.set_title("Football Tactical Simulation — Frame 1 (0.00 s)", fontsize=18, fontweight="bold", color='black', pad=15)

    # This inner function updates the plot at each frame of the animation
    # It is defined here to access local variables like ax, player_dots, title, etc
    def update(frame):
        if frame >= num_frames:
            return player_dots + [ball_circle, title] + role_labels

        # Random actions (to be replaced by agent policy)
        actions = np.random.randint(0, 9, size=22)
        env.step(actions)

        # Update player positions and labels
        for i, player in enumerate(env.players):
            pos = player.get_position() * np.array([120, 80])
            player_dots[i].set_data(pos[0], pos[1])
            role_labels[i].set_position((pos[0], pos[1] - 2))

        # Update ball position
        ball_pos = env.ball.get_position() * np.array([120, 80])
        ball_circle.set_center((ball_pos[0], ball_pos[1]))

        # Calculate elapsed time in seconds
        # Add +1 to match the displayed frame count (starting from 1) and ensure that the final frame (e.g., 1440) 
        # aligns exactly with full seconds (e.g., 60.00 s at 24 FPS)
        elapsed_time = (frame + 1) / 24
        title.set_text(f"Football Tactical Simulation — Frame {frame + 1} ({elapsed_time:.2f} s)")
        return player_dots + [ball_circle, title] + role_labels

    _ = FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, blit=False, repeat=False)
    plt.show()
