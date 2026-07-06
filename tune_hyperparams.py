"""
tune_hyperparams.py — Script temporal para comparar configuraciones de hiperparámetros del A3C.

USO: Solo modifica el diccionario CONFIGS más abajo con los valores que quieras
probar. Cada configuración se entrena desde cero (modelo y optimizador nuevos),
se guarda en results/tune_<nombre>.json, y al final se grafican todas juntas
para comparar visualmente.

No es parte del entregable final — es solo para decidir qué hiperparámetros usar.
"""

import sys, os
import queue
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.data_loader import download_and_clean_data, get_train_test
from model.actor_critic import ActorCritic
from training.a3c_worker import a3c_worker
from results.plot_results import save_metrics, load_metrics, plot_reward_curves


# ────────────────────────────────────────────────────────────── #
#   EDITA SOLO ESTO — agrega o cambia configuraciones aquí       #
# ────────────────────────────────────────────────────────────── #
CONFIGS = {
    "default":   dict(gamma=0.99, beta_entropy=0.01, t_max=20, value_coeff=0.5),
    "variant_1": dict(gamma=0.95, beta_entropy=0.02, t_max=20, value_coeff=0.5),
    "variant_2": dict(gamma=0.99, beta_entropy=0.01, t_max=10, value_coeff=0.5),
    # agrega más configuraciones aquí siguiendo el mismo formato
}

N_EPISODES = 150   # episodios por configuración (sube si quieres más señal)
TICKER = "AAPL"
START = "2020-01-01"
END = "2024-01-01"
# ────────────────────────────────────────────────────────────── #


def fresh_model_and_optimizer(obs_dim: int):
    """Crea un modelo y optimizador nuevos, con el estado de Adam
    ya inicializado y en shared memory."""
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


def run_one_config(name: str, params: dict, train_data, obs_dim: int):
    """Entrena una configuración de principio a fin y guarda sus métricas."""
    print(f"\n{'='*50}")
    print(f"Corriendo configuración: {name}  →  {params}")
    print(f"{'='*50}")

    model, optimizer = fresh_model_and_optimizer(obs_dim)
    metrics_queue = mp.Queue()

    a3c_worker(
        worker_id=0,
        global_model=model,
        optimizer=optimizer,
        train_data=train_data,
        n_episodes=N_EPISODES,
        metrics_queue=metrics_queue,
        lock=None,
        **params,
    )

    metrics = []
    while True:
        try:
            metrics.append(metrics_queue.get(timeout=1))
        except queue.Empty:
            break

    os.makedirs("results", exist_ok=True)
    out_path = f"results/tune_{name}.json"
    save_metrics(metrics, out_path)
    print(f"[{name}] {len(metrics)} episodios guardados en {out_path}")

    return metrics


if __name__ == "__main__":
    # ── Datos (una sola descarga, compartida por todas las configs) ──
    df = download_and_clean_data(TICKER, start=START, end=END)
    train_data, _ = get_train_test(df, test_ratio=0.2)
    print(f"Train rows: {len(train_data)}")

    obs_dim = 20 * 2 + 3  # 43

    # ── Correr cada configuración ──
    all_results = {}
    for name, params in CONFIGS.items():
        all_results[name] = run_one_config(name, params, train_data, obs_dim)

    # ── Comparar visualmente ──
    print("\nGenerando gráficas comparativas...")
    os.makedirs("results", exist_ok=True)
    for name in CONFIGS:
        metrics = load_metrics(f"results/tune_{name}.json")
        plot_reward_curves(
            metrics,
            title=f"Reward por Episodio — {name}",
            save_path=f"results/tune_{name}_rewards.png",
        )

    print("\nListo. Compara los .png en results/ para elegir la mejor configuración.")