import torch.multiprocessing as mp
from utils.workerLogger import WorkerLogger
import time, random, queue, torch

def worker_stub(worker_id, global_model, optimizer, train_data, n_episodes, metrics_queue):
    """
    Stub para probar el orquestador sin depender de a3c_worker.
    Simula N episodios enviando métricas ficticias.
    ESTE STUB DEBE IR EN EL MISMO ARCHIVO, DE LO CONTRARIO PODRÍA NO FUNCIONAR EL ORQUESTADOR PORQUE EL PROCESO NO ENCONTRARÍA LA FUNCIÓN.
    """
    logger = WorkerLogger(worker_id, metrics_queue)
    for episode in range(n_episodes):
        time.sleep(0.1)
        logger.log_episode(episode, reward=random.uniform(-1, 3), final_portfolio_value=10000*(1+ random.uniform(-0.05, 0.1)), steps=random.randint(50, 200))  

def run_training(global_model, optimizer, train_data, n_workers: int, n_episodes_per_worker: int, metrics_queue: mp.Queue) -> list[dict]: 
    """
    Lanza n_workers procesos en paralelo.
    Devuelve la lista de todas las métricas recolectadas.
    """
    process = []
    try:
        for worker_id in range(n_workers):
            p = mp.Process(target=worker_stub, args=(worker_id, global_model, optimizer, train_data, n_episodes_per_worker, metrics_queue))
            p.start()
            process.append(p)
        for p in process:
            p.join()
    finally:
        for p in process:
            if p.is_alive():
                p.terminate()
                p.join()

    all_metrics = []

    while True:
        try:
            metrics = metrics_queue.get(timeout=1)
            all_metrics.append(metrics)
        except queue.Empty:
            break

    return all_metrics


if __name__ == "__main__":

    mp.set_start_method('spawn')
    global_model = None
    metrics_queue = mp.Queue(maxsize=0)
    #optimizer = torch.optim.Adam(params=global_model.parameters(), lr = 1e-4)
    optimizer = None

    # for group in optimizer.param_groups:
    #     for p in group['params']:
    #         state = optimizer.state[p]
    #         for k,v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.share_memory_()

    n_workers = 4

    all_metrics = run_training(global_model, optimizer, None, n_workers, 10, metrics_queue)
    print(all_metrics)


