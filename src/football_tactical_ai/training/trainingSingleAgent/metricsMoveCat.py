import os
import json
import numpy as np
import matplotlib.pyplot as plt


FAST_JSON = r"src/football_tactical_ai/training/rewards/singleAgentMoveFast/rewards.json"
SLOW_JSON = r"src/football_tactical_ai/training/rewards/singleAgentMoveSlow/rewards.json"

OUT_DIR = r"src/football_tactical_ai/training/analysis_fast_vs_slow"
os.makedirs(OUT_DIR, exist_ok=True)


def load_rewards(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    ep = np.array([d["episode"] for d in data], dtype=int)
    r = np.array([d["reward"] for d in data], dtype=float)
    return ep, r


def moving_average(x: np.ndarray, w: int):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def summarize(ep: np.ndarray, r: np.ndarray, tail_frac: float = 0.10):
    n = len(r)
    tail_n = max(1, int(round(n * tail_frac)))
    tail = r[-tail_n:]

    # threshold time-to-reach (on moving avg)
    w = 100
    ma = moving_average(r, w)
    # align episodes for moving avg
    ma_ep = ep[w - 1:] if len(ep) >= w else ep
    threshold = 10.0
    hit = np.where(ma >= threshold)[0]
    ep_to_thresh = int(ma_ep[hit[0]]) if len(hit) else None

    return {
        "N": int(n),
        "tail_frac": tail_frac,
        "tail_N": int(tail_n),

        "mean_tail": float(np.mean(tail)),
        "median_tail": float(np.median(tail)),
        "std_tail": float(np.std(tail)),
        "p25_tail": float(np.percentile(tail, 25)),
        "p75_tail": float(np.percentile(tail, 75)),

        "neg_rate_all": float(np.mean(r < 0.0)),
        "neg_rate_tail": float(np.mean(tail < 0.0)),

        "best_reward": float(np.max(r)),
        "best_episode": int(ep[int(np.argmax(r))]),

        "ep_to_thresh_ma100_ge10": ep_to_thresh,
    }


def print_block(name, s):
    print("\n" + "=" * 70)
    print(f"{name}".center(70))
    print("=" * 70)
    print(f"N episodes: {s['N']}")
    print(f"Final {int(s['tail_frac']*100)}% window: last {s['tail_N']} episodes")
    print("")
    print(f"Mean reward (tail):    {s['mean_tail']:.3f}")
    print(f"Median reward (tail):  {s['median_tail']:.3f}")
    print(f"Std reward (tail):     {s['std_tail']:.3f}")
    print(f"IQR (tail):            [{s['p25_tail']:.3f}, {s['p75_tail']:.3f}]")
    print("")
    print(f"Negative-rate (all):   {100*s['neg_rate_all']:.1f}%")
    print(f"Negative-rate (tail):  {100*s['neg_rate_tail']:.1f}%")
    print("")
    print(f"Best reward:           {s['best_reward']:.3f} (episode {s['best_episode']})")
    print(f"Episode to reach MA100 ≥ 10: {s['ep_to_thresh_ma100_ge10']}")


def main():
    ep_f, r_f = load_rewards(FAST_JSON)
    ep_s, r_s = load_rewards(SLOW_JSON)

    s_f = summarize(ep_f, r_f, tail_frac=0.10)
    s_s = summarize(ep_s, r_s, tail_frac=0.10)

    print_block("FAST", s_f)
    print_block("SLOW", s_s)

    # Save summary JSON (per slide)
    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"FAST": s_f, "SLOW": s_s}, f, indent=2)
    print("\nSaved summary ->", summary_path)


    # 1) moving average
    w = 100
    ma_f = moving_average(r_f, w)
    ma_s = moving_average(r_s, w)


    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ep_f[w-1:], ma_f, linewidth=2.2, label="Fast player")
    ax.plot(ep_s[w-1:], ma_s, linewidth=2.2, label="Slow player")


    fig.suptitle("Training Dynamics: FAST vs SLOW Attacker", fontsize=20, fontweight="bold", y=0.95)
    fig.text(0.5, 0.85, "Moving average of episode reward (window = 100)",
            ha="center", va="center", fontsize=12)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper left", frameon=True)

    # Leave room at the top for title + subtitle
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    fig.savefig(os.path.join(OUT_DIR, "02_reward_ma100.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


    # 2) best-so-far
    plt.figure()
    plt.plot(ep_f, np.maximum.accumulate(r_f))
    plt.plot(ep_s, np.maximum.accumulate(r_s))
    plt.title("Best reward achieved over time")
    plt.xlabel("Episode")
    plt.ylabel("Best reward so far")
    plt.legend(["FAST", "SLOW"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_best_so_far.png"), dpi=200)
    plt.close()

    print("Saved plots ->", OUT_DIR)


if __name__ == "__main__":
    main()