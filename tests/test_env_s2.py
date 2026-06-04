import pandas as pd
import numpy as np
from env.trading_env import TradingEnv

df = pd.DataFrame({
    'close_norm': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    'volume_norm': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'RSI': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'macd_signal': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
})

env = TradingEnv(data = df, window = 3)

def test_increasing_prices():
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1) #BUY
    print("reward:", reward)
    assert reward > 0, "Reward should be positive when buying in an increasing price scenario"

def test_sell_without_position():
    env.reset()
    obs, reward, terminated, truncated, info = env.step(2) #SELL
    print("reward:", reward)
    print("info:", info)
    print("cash", env.cash)
    assert reward <= 0, "Reward should be non-positive when selling without a position"

def test_until_termination():
    env.reset()
    total_reward = 0.0
    obs, reward, terminated, truncated, info = env.step(1) #BUY
    while terminated == False:
        obs, reward, terminated, truncated, info = env.step(0) #HOLD
        total_reward += reward
        print(f"Step: {info['step']}, Reward: {reward:.4f}, Portfolio Value: {info['portfolio_value']:.2f}")
        if terminated:
            print("Episode terminated.")
            break
    print("Total reward:", total_reward)
    assert terminated == True, "Episode should terminate after max steps"

def test_obs_dtype():
    obs, _ = env.reset()
    assert obs.dtype == np.float32, "Observation dtype should be float32"

if __name__ == "__main__":
    print("Testing increasing prices scenario:")
    test_increasing_prices()
    print("\nTesting selling without position:")
    test_sell_without_position()
    print("\nTesting until termination:")
    test_until_termination()
    print("\nChecking observation dtype:")
    test_obs_dtype()