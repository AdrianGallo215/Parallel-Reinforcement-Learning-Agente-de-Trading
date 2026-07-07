import os

import torch.multiprocessing as mp
from env.data_loader import download_and_clean_data, get_train_test
from training.a3c_worker import a3c_worker
from model.actor_critic import ActorCritic
from utils.workerLogger import WorkerLogger
from results.plot_results import save_metrics
import time, random, queue, torch

def run_training(global_model, optimizer, train_data, n_workers: int, n_episodes_per_worker: int, metrics_queue: mp.Queue, use_lock: bool = False) -> list[dict]: 
    """
    Lanza n_workers procesos en paralelo.
    Devuelve la lista de todas las métricas recolectadas.
    """
    lock = mp.Lock() if use_lock else None

    barrier = mp.Barrier(n_workers + 1)

    process = []
    all_metrics = []

    try:
        for worker_id in range(n_workers):
            p = mp.Process(target=a3c_worker, args=(worker_id, global_model, optimizer, train_data, n_episodes_per_worker, metrics_queue, lock), kwargs={"barrier": barrier},)
            p.start()
            process.append(p)
        
        # El proceso principal espera a que TODOS los workers
        # terminen su setup (imports, modelo local, entorno).
        # Solo entonces empieza a medir tiempo de cómputo real.
        barrier.wait()
        t0 = time.time()

        while any(p.is_alive() for p in process):
            try:
                metrics = metrics_queue.get(timeout=0.5)
                all_metrics.append(metrics)
            except queue.Empty:
                continue
        
        for p in process:
            p.join()

        compute_elapsed = time.time() - t0
    finally:
        for p in process:
            if p.is_alive():
                p.terminate()
                p.join()

    while True:
        try:
            metrics = metrics_queue.get(timeout=1)
            all_metrics.append(metrics)
        except queue.Empty:
            break

    return all_metrics, compute_elapsed


if __name__ == "__main__":

    mp.set_start_method('spawn')

    ticker = "AAPL"
    df = download_and_clean_data(ticker, start="2020-01-01", end="2024-01-01")
    train_data, test_data = get_train_test(df, test_ratio=0.2)
    print(f"Train rows: {len(train_data)}, Test rows: {len(test_data)}")


    obs_dim = 20 * 2 + 3
    global_model = ActorCritic(obs_dim=obs_dim)
    global_model.share_memory()
    optimizer = torch.optim.Adam(params=global_model.parameters(), lr=1e-4)

    probs, value = global_model(torch.zeros(obs_dim))

    loss = value.sum()
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    metrics_queue = mp.Queue(maxsize=0)

    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            for k,v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.share_memory_()

    n_workers = 12

    t0 = time.time()
    all_metrics, compute_elapsed = run_training(global_model, optimizer, train_data, n_workers, 5000, metrics_queue, use_lock=False)
    save_metrics(all_metrics, "results/metrics_full_train.json")
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.2f}s (cómputo puro: {compute_elapsed:.2f}s)")
    print(f"Total episodes logged: {len(all_metrics)}")
    if all_metrics:
        avg_reward = sum(m["total_reward"] for m in all_metrics) / len(all_metrics)
        print(f"Avg reward: {avg_reward:.4f}")

    # ── Save checkpoint ──
    os.makedirs("checkpoints", exist_ok=True)
    global_model.save_checkpoint("checkpoints/parallel_model.pt")
    print("Checkpoint saved → checkpoints/parallel_model.pt")


