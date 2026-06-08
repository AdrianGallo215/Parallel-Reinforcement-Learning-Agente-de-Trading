import multiprocessing as mp
import time
import queue

class WorkerLogger:
    def __init__(self, worker_id: int, queue: mp.Queue):
        self.worker_id = worker_id
        self.queue = queue

    def log_episode(self, episode: int, reward: float, final_portfolio_value:float, steps: int) -> None:
        """El diccionario debe incluir al menos:
        {
            'worker_id': int,        # para separar curvas por worker en el plot
            'episode': int,          # número de episodio de ese worker
            'total_reward': float,   # suma de rewards del episodio
            'final_value': float,    # valor del portafolio al terminar
            'steps': int,            # duración del episodio
            'timestamp': float,      # time.time() — para calcular speedup real
        }"""
        log_dict = {
            'worker_id': self.worker_id,
            'episode': episode,
            'total_reward': reward,
            'final_value': final_portfolio_value,
            'steps': steps,
            'timestamp': time.time()
        }

        try:
            self.queue.put(log_dict, timeout=5)
        except queue.Full:
            pass
    @staticmethod
    def get_all_records(q: mp.Queue) -> list[dict]:
        records = []
        while True:
            try:
                record = q.get(timeout=1)
                records.append(record)
            except queue.Empty:
                break
        return records