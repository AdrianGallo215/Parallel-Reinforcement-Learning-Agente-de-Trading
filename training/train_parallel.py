import torch.multiprocessing as mp

def launch_workers(global_model, n_workers):

    process = []
    try:    
        for worker_id in range(n_workers):
            p = mp.Process(target=worker_fn, args=(worker_id, global_model))
            p.start()
            process.append(p)
        for p in process:
            p.join()
    finally:    
        for p in process:
            if p.is_alive():
                p.terminate()
                p.join()

def worker_fn(worker_id, global_model):
    print(f"Worker {worker_id} corriendo")

if __name__ == "__main__":

    mp.set_start_method('spawn')

    global_model = None
    n_workers = 4
    launch_workers(global_model, n_workers)
