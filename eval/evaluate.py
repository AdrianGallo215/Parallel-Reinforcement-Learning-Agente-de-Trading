"""
Evaluate a trained A3C model on held-out test data.

Metrics reported:
    - Total return  (final portfolio value / initial cash - 1)
    - Total reward  (sum of per-step rewards)
    - Number of trades
    - Sharpe ratio   (annualized, using daily returns)
"""

import sys, os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.data_loader import download_and_clean_data, get_train_test
from env.trading_env import TradingEnv
from model.actor_critic import ActorCritic


def evaluate(model: ActorCritic, test_data, window: int = 20,
             initial_cash: float = 10000.0, verbose: bool = True):
    """
    Run one full episode on test_data using a greedy policy (argmax).

    Returns a dict with evaluation metrics.
    """
    env = TradingEnv(data=test_data, window=window, initial_cash=initial_cash)
    obs, _ = env.reset()
    # Force starting at the beginning of the test set for reproducibility
    env.current_step = env.min_step
    env.cash = initial_cash
    env.shares = 0.0
    env.position = 0
    env.prev_value = initial_cash
    obs = env._get_obs()

    model.eval()

    total_reward = 0.0
    n_trades = 0
    daily_returns = []
    actions_taken = []
    prev_portfolio = initial_cash

    done = False
    step = 0

    with torch.no_grad():
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            probs, value = model(obs_t)

            # Greedy action (argmax)
            action = probs.argmax(dim=-1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            actions_taken.append(action)

            if action in (1, 2):  # Buy or Sell
                n_trades += 1

            # Track daily portfolio returns
            portfolio = info["portfolio_value"]
            if prev_portfolio > 0:
                daily_returns.append((portfolio - prev_portfolio) / prev_portfolio)
            prev_portfolio = portfolio
            step += 1

    # ── Compute metrics ──
    final_value = info["portfolio_value"]
    total_return = (final_value / initial_cash - 1) * 100  # percentage

    daily_returns = np.array(daily_returns)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    results = {
        "final_value": final_value,
        "total_return_pct": total_return,
        "total_reward": total_reward,
        "n_trades": n_trades,
        "n_steps": step,
        "sharpe_ratio": sharpe,
        "actions": actions_taken,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"  EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"  Initial cash:    ${initial_cash:,.2f}")
        print(f"  Final value:     ${final_value:,.2f}")
        print(f"  Total return:    {total_return:+.2f}%")
        print(f"  Total reward:    {total_reward:.4f}")
        print(f"  Steps:           {step}")
        print(f"  Trades:          {n_trades}")
        print(f"  Sharpe ratio:    {sharpe:.4f}")
        print(f"{'='*50}")

        # Action distribution
        from collections import Counter
        dist = Counter(actions_taken)
        total = len(actions_taken)
        print(f"  Actions: Hold={dist[0]/total*100:.1f}%  "
              f"Buy={dist[1]/total*100:.1f}%  "
              f"Sell={dist[2]/total*100:.1f}%")

    return results


# ────────────────────────────────────────────────────────────── #
#                          CLI                                    #
# ────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained A3C model")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/parallel_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")
    args = parser.parse_args()

    # Data
    df = download_and_clean_data(args.ticker, start=args.start, end=args.end)
    _, test_data = get_train_test(df, test_ratio=0.2)
    print(f"Test data: {len(test_data)} rows")

    # Load model
    model = ActorCritic.load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    results = evaluate(model, test_data)
