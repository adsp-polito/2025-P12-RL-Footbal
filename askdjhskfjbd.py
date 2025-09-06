import pickle
import matplotlib.pyplot as plt
from football_tactical_ai.env.objects.pitch import Pitch

# Carica le heatmap per ruolo (dal pickle creato prima)
with open("statsbomb360/data/role_heatmaps.pkl", "rb") as f:
    heatmaps = pickle.load(f)

def plot_roles(roles_to_plot):
    pitch = Pitch()

    fig, axes = plt.subplots(1, len(roles_to_plot), figsize=(6*len(roles_to_plot), 8))

    if len(roles_to_plot) == 1:
        axes = [axes]  # fix per un solo ruolo

    for ax, role in zip(axes, roles_to_plot):
        if role not in heatmaps:
            print(f"⚠️ Role {role} not found in heatmaps")
            continue

        grid = heatmaps[role]

        # Check and fix shape mismatch
        if grid.shape != (pitch.num_cells_x, pitch.num_cells_y):
            print(f"Adjusting shape for {role}: {grid.shape} -> {(pitch.num_cells_x, pitch.num_cells_y)}")
            if grid.shape == (pitch.num_cells_y, pitch.num_cells_x):
                grid = grid.T  # transpose if just flipped
            else:
                # resize (e.g., crop/pad) if mismatch
                from skimage.transform import resize
                grid = resize(grid, (pitch.num_cells_x, pitch.num_cells_y), mode="reflect", anti_aliasing=True)

        pitch.draw_pitch(
            ax=ax,
            show_grid=False,
            show_heatmap=True,
            show_rewards=False,
            reward_grid=grid
        )
        ax.set_title(role)


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Esempio: plot per tre ruoli
    plot_roles(["Centre Forward", "Left Back", "Goalkeeper"])
