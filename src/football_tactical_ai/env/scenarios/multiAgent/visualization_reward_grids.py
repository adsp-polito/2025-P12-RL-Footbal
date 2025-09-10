import matplotlib.pyplot as plt
from tqdm import tqdm

from football_tactical_ai.env.objects.pitch import Pitch
from football_tactical_ai.env.scenarios.multiAgent.rewardGrids import (
    build_attacker_grid, build_defender_grid, build_goalkeeper_grid
)


def main():
    # Create the Pitch
    pitch = Pitch()

    # Define roles
    attacker_roles = ["LW", "RW", "CF", "LCF", "RCF", "SS", "ATT"]
    defender_roles = ["LCB", "RCB", "CB", "DEF"]
    goalkeeper_roles = ["GK"]  # uniform handling

    roles = attacker_roles + defender_roles + goalkeeper_roles  # totale = 12

    # Create subplots (3 rows x 4 cols = 12)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    # Iterate with tqdm for progress bar
    for idx, role in enumerate(tqdm(roles, desc="Building grids")):
        if role in attacker_roles:
            grid = build_attacker_grid(pitch, role=role)
        elif role in defender_roles:
            grid = build_defender_grid(pitch, role=role)
        else:  # GK
            grid = build_goalkeeper_grid(pitch)

        pitch.draw_pitch(
            ax=axes[idx],
            show_grid=False,
            show_heatmap=True,
            show_rewards=False,
            reward_grid=grid
        )
        axes[idx].set_title(f"{role} Grid")

    # Hide extra subplot if roles < subplot slots
    for ax in axes[len(roles):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
