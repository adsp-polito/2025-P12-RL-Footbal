import matplotlib.pyplot as plt

from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.env.scenarios.multiAgent.rewardGrids  import build_attacker_grid, build_defender_grid, build_goalkeeper_grid

def main():
    # Create the Pitch
    pitch = Pitch()

    # Build the grids
    attacker_grid = build_attacker_grid(pitch)
    defender_grid = build_defender_grid(pitch)
    goalkeeper_grid = build_goalkeeper_grid(pitch)

    # Create subplots
    _, axes = plt.subplots(1, 3, figsize=(12, 8))

    # Attacker grid
    pitch.draw_pitch(
        ax=axes[0],
        show_grid=False,
        show_heatmap=True,
        show_rewards=False,
        reward_grid=attacker_grid
    )
    axes[0].set_title("Attacker Grid")

    # Defender grid
    pitch.draw_pitch(
        ax=axes[1],
        show_grid=False,
        show_heatmap=True,
        show_rewards=False,
        reward_grid=defender_grid
    )
    axes[1].set_title("Defender Grid")

    # Goalkeeper grid
    pitch.draw_pitch(
        ax=axes[2],
        show_grid=False,
        show_heatmap=True,
        show_rewards=False,
        reward_grid=goalkeeper_grid
    )
    axes[2].set_title("Goalkeeper Grid")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()