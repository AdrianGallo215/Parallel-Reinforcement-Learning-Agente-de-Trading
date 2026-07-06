"""
Sequential (single-worker) training baseline.

Runs the same A3C logic but with a single worker and no multiprocessing.
This serves as the baseline to measure parallel speedup.
"""

import sys, os
import time
import queue
import torch
import multiprocessing as mp

# Ensure project root is on the path when running standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.data_loader import download_and_clean_data, get_train_test
from model.actor_critic import ActorCritic
from training.a3c_worker import a3c_worker
from utils.workerLogger import WorkerLogger


def run_sequential(global_model, optimizer, train_data,
                   n_episodes: int, metrics_queue: mp.Queue,
                   lock=None) -> list[dict]:
    """
    Train a single worker sequentially.
    Same interface as run_training() in train_parallel.py so results
    are directly comparable.
    """
    # Run worker 0 directly in the main process (no spawning)
    a3c_worker(
        worker_id=0,
        global_model=global_model,
        optimizer=optimizer,
        train_data=train_data,
        n_episodes=n_episodes,
        metrics_queue=metrics_queue,
        lock=lock,
    )

    # Collect metrics
    all_metrics = []
    while True:
        try:
            m = metrics_queue.get(timeout=1)
            all_metrics.append(m)
        except queue.Empty:
            break

    return all_metrics


# ────────────────────────────────────────────────────────────── #
#                          CLI ENTRY                              #
# ────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    # ── Data ──
    ticker = "AAPL"
    df = download_and_clean_data(ticker, start="2020-01-01", end="2024-01-01")
    train_data, test_data = get_train_test(df, test_ratio=0.2)
    print(f"Train rows: {len(train_data)}, Test rows: {len(test_data)}")

    # ── Model ──
    obs_dim = 20 * 2 + 3  # window=20
    global_model = ActorCritic(obs_dim=obs_dim)
    global_model.share_memory()

    # ── Optimizer ──
    optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-4)

    # ── Shared-memory optimizer state ──
    _, value = global_model(torch.zeros(obs_dim))
    value.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.share_memory_()

    # ── Run ──
    metrics_queue = mp.Queue()
    n_episodes = 50

    print(f"\n{'='*50}")
    print(f"Sequential training: 1 worker × {n_episodes} episodes")
    print(f"{'='*50}")

    t0 = time.time()
    metrics = run_sequential(global_model, optimizer, train_data,
                             n_episodes, metrics_queue)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.2f}s")
    print(f"Episodes logged: {len(metrics)}")
    if metrics:
        avg_reward = sum(m["total_reward"] for m in metrics) / len(metrics)
        print(f"Avg reward: {avg_reward:.4f}")

    # ── Save checkpoint ──
    os.makedirs("checkpoints", exist_ok=True)
    global_model.save_checkpoint("checkpoints/sequential_model.pt")
    print("Checkpoint saved → checkpoints/sequential_model.pt")
