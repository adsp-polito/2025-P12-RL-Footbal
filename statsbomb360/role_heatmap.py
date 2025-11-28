import os
import ast
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsbombpy import sb

import warnings
warnings.filterwarnings("ignore")

# Import pitch settings
from football_tactical_ai.configs.pitchSettings import FIELD_WIDTH, FIELD_HEIGHT, CELL_SIZE


def load_lineups(match_id: int) -> pd.DataFrame:
    """
    Load lineups for a match from StatsBomb and return DataFrame with player_id and role.
    Uses ast.literal_eval to parse the 'positions' field.
    """
    lineups = sb.lineups(match_id=match_id)
    records = []
    for df in lineups.values():
        for _, row in df.iterrows():
            role = "Unknown"
            try:
                # Parse string like "[{'position_id': 2, 'position': 'Right Back'}]"
                if isinstance(row["positions"], str):
                    positions = ast.literal_eval(row["positions"])
                elif isinstance(row["positions"], list):
                    positions = row["positions"]
                else:
                    positions = []

                if positions and isinstance(positions[0], dict):
                    role = positions[0].get("position", "Unknown")
            except Exception:
                raise(f"Error parsing positions for player_id {row['player_id']}: {row['positions']}")

            records.append({
                "player_id": int(row["player_id"]),   # for consistency
                "player_name": row["player_name"],
                "role": role
            })

    return pd.DataFrame(records)


def load_events(match_id: int) -> pd.DataFrame:
    """
    Load events for a match and extract player_id with event location.
    Uses flatten_attrs=True so player_id is numeric and location is a list.
    """
    events = sb.events(match_id=match_id, flatten_attrs=True)

    if "location" not in events.columns or "player_id" not in events.columns:
        return pd.DataFrame()

    # Keep only events with valid location
    mask = events["location"].apply(lambda loc: isinstance(loc, list) and len(loc) == 2)
    df = events.loc[mask, ["player_id", "location"]].copy()



    if df.empty:
        return pd.DataFrame()

    df["x"] = df["location"].apply(lambda l: float(l[0]))
    df["y"] = df["location"].apply(lambda l: float(l[1]))
    df = df.drop(columns="location")

    df["player_id"] = df["player_id"].astype("Int64")  # pandas nullable int


    return df


def build_role_heatmaps(all_events: pd.DataFrame, all_lineups: pd.DataFrame) -> dict:
    """
    Build role-based heatmaps from event locations.
    Merge events with lineups to map player_id -> role.
    """
    num_cells_x = int(FIELD_WIDTH // CELL_SIZE)
    num_cells_y = int(FIELD_HEIGHT // CELL_SIZE)

    # Merge to assign role to each event
    events_with_roles = all_events.merge(
        all_lineups[["player_id", "role"]],
        on="player_id",
        how="left"
    )

    heatmaps = {}
    roles = events_with_roles["role"].dropna().unique()

    # Iterate with progress bar
    for role in tqdm(roles, desc="Building role heatmaps"):
        heatmap = np.zeros((num_cells_x, num_cells_y))

        role_positions = events_with_roles.loc[
            events_with_roles["role"] == role, ["x", "y"]
        ]

        for _, row in role_positions.iterrows():
            i = int(row["x"] // CELL_SIZE)
            j = int(row["y"] // CELL_SIZE)
            if 0 <= i < num_cells_x and 0 <= j < num_cells_y:
                heatmap[i, j] += 1

        if heatmap.sum() > 0:
            heatmap = heatmap / heatmap.max()  # normalize [0, 1]

        print(f"  Role '{role}': {role_positions.shape[0]} events mapped")
        heatmaps[role] = heatmap

    return heatmaps


if __name__ == "__main__":
    OUTPUT_DIR = "statsbomb360/data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    COMPETITION_ID = 11  # Serie A
    SEASON_ID = 27       # 2015/16

    print("Loading matches for Serie A 2015/16...")
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)

    all_events = []
    all_lineups = []

    # Iterate over matches and collect events and lineups
    for match_id in tqdm(matches["match_id"].tolist(), desc="Matches"):
        events_df = load_events(match_id)
        lineups_df = load_lineups(match_id)

        if not events_df.empty and not lineups_df.empty:
            all_events.append(events_df)
            all_lineups.append(lineups_df)

    if not all_events or not all_lineups:
        raise RuntimeError("No event data collected for Serie A 2015/16")

    all_events_df = pd.concat(all_events, ignore_index=True)
    all_lineups_df = pd.concat(all_lineups, ignore_index=True)

    print("\nBuilding role heatmaps for Serie A 2015/16 (using event locations)...")
    heatmaps = build_role_heatmaps(all_events_df, all_lineups_df)

    output_path = os.path.join(OUTPUT_DIR, "role_heatmaps.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(heatmaps, f)

    print(f"\nSaved role heatmaps to {output_path}")
