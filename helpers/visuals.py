import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np
from helpers.drawPitch import draw_pitch

def highlight_possessor(ax):
    """
    Initializes and returns visual elements for highlighting the current ball possessor.

    Args:
        - ax: Matplotlib axis object where the simulation is drawn

    Returns:
        - highlight_circle (Circle): Yellow ring around the player in possession
        - possessor_ring (Circle): Black outer ring to enhance visibility
    """
    # Yellow highlight circle for the player who owns the ball
    highlight_circle = Circle(
        (-10, -10),
        radius=1.0,
        edgecolor='yellow',
        facecolor='none',
        linewidth=4,
        zorder=4
    )
    ax.add_patch(highlight_circle)

    # Additional black ring for better contrast and visibility
    possessor_ring = Circle(
        (-10, -10),
        radius=1.3,
        edgecolor='black',
        facecolor='none',
        linewidth=2,
        zorder=3
    )
    ax.add_patch(possessor_ring)

    return highlight_circle, possessor_ring

def animate_simulation(env, num_frames=24, interval_ms=1000/24):
    """
    Animates a full football simulation using matplotlib's FuncAnimation
    Each frame updates player and ball positions, visualizing their movements on a football pitch

    Args:
        - env (FootballEnv): The initialized football environment
        - num_frames (int): Total number of frames to simulate and render
        - interval_ms (float): Milliseconds between frames (controls playback speed)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    player_dots = []
    role_labels = []
    team_colors = [env.teams[0].color] * 11 + [env.teams[1].color] * 11

    # Create dots and role labels for each player
    for i, player in enumerate(env.players):
        pos = player.get_position() * np.array([120, 80])
        dot, = ax.plot(pos[0], pos[1], 'o', markersize=10, color=team_colors[i], markeredgecolor="black", zorder=5)
        label_text = getattr(player, 'abbr', player.role[:2])
        label = ax.text(
            pos[0], pos[1] - 2,
            label_text,
            ha='center', va='center',
            fontsize=8, color='black', fontweight='bold', zorder=6
        )
        player_dots.append(dot)
        role_labels.append(label)

    # Initialize the ball as a white circle with a black outline
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

    # Add highlight visuals for the current possessor
    highlight_circle, possessor_ring = highlight_possessor(ax)

    # Title to show simulation time and frame number
    title = ax.set_title("Simulation — Frame 1 (0.00 s)", fontsize=18, fontweight="bold", color='black', pad=15)

    def update(frame):
        """
        Updates the positions of all elements in the animation for the current frame.
        """
        if frame >= num_frames:
            return player_dots + [ball_circle, title] + role_labels

        # Default: all players idle (action 0)
        actions = [0] * 22

        # Identify current ball possessor
        possessor_id = env.ball.owner_id

        if possessor_id is not None:
            # Define receiver: pass to the next player in the list (cyclic)
            receiver_id = (possessor_id - 1) % 22

            # Assign pass action to the possessor
            actions[possessor_id] = ("pass", receiver_id)

            # Optional debug:
            print(f"Frame {frame}: Player {possessor_id} passes to Player {receiver_id}")

        # Apply actions to environment
        env.step(actions)

        # Update player markers and role labels
        for i, player in enumerate(env.players):
            pos = player.get_position() * np.array([120, 80])
            player_dots[i].set_data(pos[0], pos[1])
            role_labels[i].set_position((pos[0], pos[1] - 2))

        # Update ball position
        ball_pos = env.ball.get_position() * np.array([120, 80])
        ball_circle.set_center((ball_pos[0], ball_pos[1]))

        # Highlight the current ball possessor, if any
        if env.ball.owner_id is not None:
            pos = env.players[env.ball.owner_id].get_position() * np.array([120, 80])
            highlight_circle.set_center((pos[0], pos[1]))
            possessor_ring.set_center((pos[0], pos[1]))
        else:
            # Move highlights off-screen
            highlight_circle.set_center((-10, -10))
            possessor_ring.set_center((-10, -10))

        # Update time counter in the title
        elapsed_time = (frame + 1) / 24  # seconds (since you're running at 24 FPS)
        title.set_text(f"Simulation — Frame {frame + 1} ({elapsed_time:.2f} s)")

        return player_dots + [ball_circle, highlight_circle, possessor_ring, title] + role_labels

    _ = FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, blit=False, repeat=False)
    plt.show()
