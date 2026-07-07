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
    def __init__(self, data, window=20, initial_cash=10000.0, trade_penalty=0.001):
        super().__init__()

        self.data = data
        self._close      = data['Close'].to_numpy(dtype=np.float64)
        self._close_norm = data['close_norm'].to_numpy(dtype=np.float32)
        self._vol_norm   = data['volume_norm'].to_numpy(dtype=np.float32)
        self._rsi        = data['RSI'].to_numpy(dtype=np.float32)
        self._macd_sig   = data['macd_signal'].to_numpy(dtype=np.float32)
        self.window = window
        self.initial_cash = initial_cash
        self.trade_penalty = trade_penalty

        self.min_step = window - 1
        self.max_step = len(self.data) - 2

        assert self.max_step > self.min_step, (
            f"Dataset demasiado corto: {len(self.data)} filas con window={window}. "
            f"Mínimo requerido: {window + 2} filas."
        )

        self.obs_dim = window*2 + 3
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        
        self.cash = initial_cash
        self.shares = 0.0
        self.position = 0
        self.prev_value = initial_cash
        self.current_step = self.min_step
        
        self.info = ENVINFO

    def step(self, action):
        price_now = self._close[self.current_step]
        
        trade_executed = self._execute_action(action, price_now)
        
        self.current_step += 1
        
        reward = self._get_reward(trade_executed)
        
        self.prev_value = self.cash + self.shares * self._close[self.current_step]          
        obs = self._get_obs()
        
        truncated = False
        info = {
            "portfolio_value": self.prev_value,
            "position": self.position,
            "step": self.current_step,
            "trade_executed": trade_executed,
        }
        terminated = self.current_step >= self.max_step
        
        return obs, float(reward), terminated, truncated, info
        
    def _get_reward(self, trade_executed):
        asset_price = self._close[self.current_step]
        v_t = self.cash + self.shares * asset_price

        if self.prev_value == 0:
            r_t = 0.0
        else:
            r_t = (v_t - self.prev_value) / self.prev_value
    
        if trade_executed:
            reward = r_t - self.trade_penalty
        else: 
            reward = r_t

        return float(reward)

    def _execute_action(self, action, asset_price): 
        if action == 1: #BUY
            if self.position == 1 or self.shares > 0:
                return False
            self.shares = self.cash / asset_price
            self.cash = 0.0
            self.position = 1
            return True
        elif action == 2: #SELL
            if self.position == 0 or self.shares == 0:
                return False
            self.cash = self.shares * asset_price
            self.shares = 0.0
            self.position = 0
            return True
        elif action == 0: #HOLD
            return False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.cash = self.initial_cash
        self.shares = 0.0
        self.position = 0
        self.prev_value = self.initial_cash
        
        self.current_step = np.random.randint(self.min_step, self.max_step - 1 )
        

        return self._get_obs(), {}
    
    def _get_obs(self):
        i = self.current_step
        w = self.window
        close_window = self._close_norm[i - w + 1:i + 1]
        volume_window = self._vol_norm[i - w + 1:i + 1]
        rsi = self._rsi[i:i + 1]
        macd_signal = self._macd_sig[i:i + 1]
        position = np.array([self.position], dtype=np.float32)

        obs = np.concatenate([close_window, volume_window, rsi, macd_signal, position])
        return obs.astype(np.float32)


    def render(self):
        pass


