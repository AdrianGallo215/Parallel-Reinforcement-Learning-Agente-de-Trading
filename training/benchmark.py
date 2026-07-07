"""
benchmark.py — Mide el speedup del entrenamiento paralelo A3C variando n_workers.

Para cada n en [1, 2, 4, 8]:
    1. Crea un modelo y optimizador NUEVOS (mismo punto de partida siempre)
    2. Corre run_training() con ese n_workers
    3. Mide el tiempo total
    4. Guarda tiempo + métricas

Al final:
    - Grafica tiempo vs n_workers
    - Grafica speedup empírico vs speedup ideal vs curva de Amdahl ajustada
    - Imprime la fracción serial (s) estimada por curve_fit
"""

import sys, os
import json
import time
import queue
import threading
import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.data_loader import download_and_clean_data, get_train_test
from model.actor_critic import ActorCritic
from training.train_parallel import run_training


# ────────────────────────────────────────────────────────────── #
#                        CONFIGURACIÓN                            #
# ────────────────────────────────────────────────────────────── #
TICKER = "AAPL"
START = "2020-01-01"
END = "2024-01-01"
TOTAL_EPISODES = 12000
WORKER_COUNTS = [1, 2, 4, 8, 12]
USE_LOCK = False   # A3C sin lock (Hogwild) — el diferenciador del proyecto


def fresh_model_and_optimizer(obs_dim: int):
    """Crea un modelo y optimizador nuevos, con el estado de Adam
    ya inicializado y en shared memory. Mismo punto de partida
    para cada corrida del benchmark."""
    model = ActorCritic(obs_dim=obs_dim)
    model.share_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    _, value = model(torch.zeros(obs_dim))
    value.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    for group in optimizer.param_groups:
        for p in group["params"]:
            for k, v in optimizer.state[p].items():
                if isinstance(v, torch.Tensor):
                    optimizer.state[p][k] = v.share_memory_()

    return model, optimizer


def monitor_progress(metrics_queue: mp.Queue, bars: list, collected: list,
                     stop_event: threading.Event):
    """
    Corre en un HILO del proceso principal (no toca la sección paralela):
    los workers solo hacen put() en metrics_queue, y aquí consumimos cada
    métrica para avanzar la barra tqdm del worker correspondiente.
    """
    while True:
        try:
            m = metrics_queue.get(timeout=0.5)
            collected.append(m)
            bars[m["worker_id"]].update(1)
        except queue.Empty:
            if stop_event.is_set():
                break


def amdahl_speedup(n, s):
    """Speedup(N) = 1 / (s + (1-s)/N)"""
    return 1.0 / (s + (1.0 - s) / n)


def plot_amdahl_curve(results: list[dict], save_path: str = None):
    """
    Grafica tiempo vs n_workers y speedup empírico vs ideal vs
    curva de Amdahl ajustada por curve_fit.
    """
    n_workers = np.array([r["n_workers"] for r in results])
    times = np.array([r["elapsed"] for r in results])
    seq_time = times[n_workers == 1][0]
    speedups = seq_time / times

    # Ajuste de Amdahl con scipy
    popt, _ = curve_fit(amdahl_speedup, n_workers, speedups, p0=[0.1], bounds=(0, 1))
    s_fit = popt[0]

    n_smooth = np.linspace(1, max(n_workers), 200)
    ideal = n_smooth
    fitted = amdahl_speedup(n_smooth, s_fit)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Tiempo vs n_workers ──
    ax1.plot(n_workers, times, "o-", color="#e15759", linewidth=2, markersize=8)
    ax1.set_xlabel("Número de workers")
    ax1.set_ylabel("Tiempo (segundos)")
    ax1.set_title("Tiempo de entrenamiento vs N workers")
    ax1.grid(True, alpha=0.3)

    # ── Speedup: empírico vs ideal vs Amdahl ajustado ──
    ax2.plot(n_workers, speedups, "o", color="#59a14f", markersize=10,
             label="Speedup empírico", zorder=3)
    ax2.plot(n_smooth, ideal, "--", color="#bab0ac", label="Speedup ideal (lineal)")
    ax2.plot(n_smooth, fitted, "-", color="#4e79a7", linewidth=2,
             label=f"Amdahl ajustado (s={s_fit:.3f})")
    ax2.set_xlabel("Número de workers")
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("Speedup: empírico vs ideal vs Amdahl")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Guardado: {save_path}")
    plt.show()

    print(f"\nFracción serial estimada (s) por curve_fit: {s_fit:.4f}")
    return s_fit


if __name__ == "__main__":
    mp.set_start_method("spawn")

    # ── Datos (una sola descarga para todo el benchmark) ──
    df = download_and_clean_data(TICKER, start=START, end=END)
    train_data, _ = get_train_test(df, test_ratio=0.2)
    print(f"Train rows: {len(train_data)}")

    obs_dim = 20 * 2 + 3  # 43

    # ── Correr cada n_workers desde un punto de partida limpio ──
    results = []
    for n_workers in WORKER_COUNTS:
        episodes_per_worker = TOTAL_EPISODES // n_workers
        print(f"\n{'='*50}")
        print(f"Benchmark: {n_workers} workers × {episodes_per_worker} episodios")
        print(f"{'='*50}")

        global_model, optimizer = fresh_model_and_optimizer(obs_dim)
        metrics_queue = mp.Queue()

        # ── Barras de progreso: una por worker, actualizadas desde un
        #    hilo del proceso principal que lee metrics_queue ──
        bars = [
            tqdm(total=episodes_per_worker, desc=f"Worker {i}",
                 position=i, leave=True, dynamic_ncols=True)
            for i in range(n_workers)
        ]
        collected = []
        stop_event = threading.Event()
        monitor = threading.Thread(
            target=monitor_progress,
            args=(metrics_queue, bars, collected, stop_event),
            daemon=True,
        )
        monitor.start()

        t0_total = time.time()
        leftover, compute_elapsed = run_training(
            global_model, optimizer, train_data,
            n_workers=n_workers,
            n_episodes_per_worker=episodes_per_worker,
            metrics_queue=metrics_queue,
            use_lock=USE_LOCK,
        )
        total_elapsed = time.time() - t0_total

        stop_event.set()
        monitor.join()

        for m in leftover:
            bars[m["worker_id"]].update(1)
        for b in bars:
            b.close()
        all_metrics = collected + leftover

        print(f"Tiempo total (con arranque de procesos): {total_elapsed:.2f}s")
        print(f"Tiempo de cómputo puro (post-barrera): {compute_elapsed:.2f}s")
        print(f"Episodios logueados: {len(all_metrics)}")

        results.append({
            "n_workers": n_workers,
            "elapsed": compute_elapsed,        # usa este para la curva de Amdahl
            "elapsed_total": total_elapsed,     # dato honesto adicional para el informe
            "n_episodes_logged": len(all_metrics),
        })
        os.makedirs("results", exist_ok=True)
        with open("results/benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Resultados parciales guardados en results/benchmark_results.json")

    # ── Guardar resultados crudos ──

    print(f"\nResultados guardados en results/benchmark_results.json")

    # ── Graficar ──
    plot_amdahl_curve(results, save_path="results/amdahl_speedup.png")