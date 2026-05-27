import gymnasium as gym
from gymnasium import spaces
import numpy as np

ENVINFO = {
    "initial_cash": 10000,
    "initial_shares": 0,
    "window_size": 20,
    "position": {
        0: "No Position, igual a no tener acciones",
        1: "En el mercado, shares > 0",
    },
    "current_step": 0,
    "dimension": "w*2 + 3",
    "actions": {
        0: "Hold",
        1: "Buy",
        2: "Sell"
    }
}

class TradingEnv(gym.Env):
    def __init__(self, data, window=20):
        super().__init__()
        self.data = data
        self.window = window
        self.prev_value = 10000
        self.cash = 10000
        self.shares = 0
        self.current_step = 0
        self.position = 0
        self.obs_dim = window*2 + 3
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.info = ENVINFO
        pass

    def step(self, action):

        self.current_step += 1
        reward = 0
        truncated = False
        info = {}
        terminated = self.current_step >= len(self.data) - 1
        
        return self.__get_obs(), reward, terminated, truncated, info
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = 10000
        self.shares = 0
        self.position = 0
        self.current_step = self.window

        return self.__get_obs(), {}
    
    def __get_obs(self):
        close_window = self.data['close_norm'].iloc[self.current_step - self.window:self.current_step].values
        volume_window = self.data['volume_norm'].iloc[self.current_step - self.window:self.current_step].values
        rsi = self.data['RSI'].iloc[self.current_step]
        macd_signal = self.data['macd_signal'].iloc[self.current_step]
        obs = np.concatenate([close_window, volume_window, [rsi, macd_signal, self.position]])

        return obs.astype(np.float32)



