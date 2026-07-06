"""
Plot training metrics and compute speedup analysis.

Reads logs from WorkerLogger (list[dict]) and generates:
    1. Reward curves per worker
    2. Portfolio value curves per worker
    3. Speedup bar chart (sequential vs parallel)
"""

import sys, os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def plot_reward_curves(metrics: list[dict], title: str = "Reward per Episode",
                       save_path: str = None):
    """Plot reward curves separated by worker_id."""
    workers = defaultdict(list)
    for m in metrics:
        workers[m["worker_id"]].append(m)

    # Sort by episode within each worker
    for wid in workers:
        workers[wid].sort(key=lambda x: x["episode"])

    fig, ax = plt.subplots(figsize=(10, 5))
    for wid in sorted(workers.keys()):
        episodes = [m["episode"] for m in workers[wid]]
        rewards = [m["total_reward"] for m in workers[wid]]
        ax.plot(episodes, rewards, label=f"Worker {wid}", alpha=0.7)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_portfolio_curves(metrics: list[dict],
                          title: str = "Portfolio Value per Episode",
                          save_path: str = None):
    """Plot final portfolio value curves separated by worker_id."""
    workers = defaultdict(list)
    for m in metrics:
        workers[m["worker_id"]].append(m)

    for wid in workers:
        workers[wid].sort(key=lambda x: x["episode"])

    fig, ax = plt.subplots(figsize=(10, 5))
    for wid in sorted(workers.keys()):
        episodes = [m["episode"] for m in workers[wid]]
        values = [m["final_value"] for m in workers[wid]]
        ax.plot(episodes, values, label=f"Worker {wid}", alpha=0.7)

    ax.axhline(y=10000, color="gray", linestyle="--", alpha=0.5, label="Initial ($10k)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_speedup(seq_time: float, par_time: float, n_workers: int,
                 save_path: str = None):
    """Bar chart comparing sequential vs parallel training time + speedup."""
    speedup = seq_time / par_time if par_time > 0 else 0
    ideal_speedup = n_workers

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Time comparison ──
    bars = ax1.bar(["Sequential\n(1 worker)", f"Parallel\n({n_workers} workers)"],
                   [seq_time, par_time],
                   color=["#4e79a7", "#e15759"], width=0.5)
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Training Time Comparison")
    for bar, val in zip(bars, [seq_time, par_time]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.1f}s", ha="center", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # ── Speedup ──
    bars2 = ax2.bar(["Measured\nSpeedup", "Ideal\nSpeedup"],
                    [speedup, ideal_speedup],
                    color=["#59a14f", "#bab0ac"], width=0.5)
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title(f"Speedup Analysis ({n_workers} workers)")
    for bar, val in zip(bars2, [speedup, ideal_speedup]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}×", ha="center", fontweight="bold")
    ax2.set_ylim(0, max(speedup, ideal_speedup) * 1.3)
    ax2.grid(True, alpha=0.3, axis="y")

    # Efficiency
    efficiency = (speedup / ideal_speedup) * 100 if ideal_speedup > 0 else 0
    fig.suptitle(f"Parallel Efficiency: {efficiency:.1f}%", fontsize=13,
                 fontweight="bold", y=0.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_convergence_comparison(seq_metrics: list[dict], par_metrics: list[dict],
                                save_path: str = None):
    """
    Compare learning convergence: reward over wall-clock time
    for sequential vs parallel training.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Sequential — single worker
    seq_sorted = sorted(seq_metrics, key=lambda x: x["timestamp"])
    if seq_sorted:
        t0 = seq_sorted[0]["timestamp"]
        seq_times = [m["timestamp"] - t0 for m in seq_sorted]
        seq_rewards = [m["total_reward"] for m in seq_sorted]
        # Rolling average
        window = min(10, len(seq_rewards))
        seq_smooth = np.convolve(seq_rewards, np.ones(window) / window, mode="valid")
        ax.plot(seq_times[:len(seq_smooth)], seq_smooth,
                label="Sequential", linewidth=2, color="#4e79a7")

    # Parallel — aggregate all workers by wall-clock time
    par_sorted = sorted(par_metrics, key=lambda x: x["timestamp"])
    if par_sorted:
        t0 = par_sorted[0]["timestamp"]
        par_times = [m["timestamp"] - t0 for m in par_sorted]
        par_rewards = [m["total_reward"] for m in par_sorted]
        window = min(10, len(par_rewards))
        par_smooth = np.convolve(par_rewards, np.ones(window) / window, mode="valid")
        ax.plot(par_times[:len(par_smooth)], par_smooth,
                label="Parallel", linewidth=2, color="#e15759")

    ax.set_xlabel("Wall-clock Time (seconds)")
    ax.set_ylabel("Reward (rolling avg)")
    ax.set_title("Convergence: Sequential vs Parallel")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def save_metrics(metrics: list[dict], path: str):
    """Save metrics list to JSON."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {path}")


def load_metrics(path: str) -> list[dict]:
    """Load metrics list from JSON."""
    with open(path, "r") as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot A3C training results")
    parser.add_argument("--seq", type=str, help="Path to sequential metrics JSON")
    parser.add_argument("--par", type=str, help="Path to parallel metrics JSON")
    parser.add_argument("--seq-time", type=float, help="Sequential training time (s)")
    parser.add_argument("--par-time", type=float, help="Parallel training time (s)")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.par:
        par_metrics = load_metrics(args.par)
        plot_reward_curves(par_metrics, "Parallel: Reward per Episode",
                           os.path.join(args.outdir, "parallel_rewards.png"))
        plot_portfolio_curves(par_metrics, "Parallel: Portfolio Value",
                              os.path.join(args.outdir, "parallel_portfolio.png"))

    if args.seq:
        seq_metrics = load_metrics(args.seq)
        plot_reward_curves(seq_metrics, "Sequential: Reward per Episode",
                           os.path.join(args.outdir, "sequential_rewards.png"))

    if args.seq and args.par:
        seq_metrics = load_metrics(args.seq)
        par_metrics = load_metrics(args.par)
        plot_convergence_comparison(
            seq_metrics, par_metrics,
            os.path.join(args.outdir, "convergence_comparison.png"))

    if args.seq_time and args.par_time:
        plot_speedup(args.seq_time, args.par_time, args.n_workers,
                     os.path.join(args.outdir, "speedup.png"))
