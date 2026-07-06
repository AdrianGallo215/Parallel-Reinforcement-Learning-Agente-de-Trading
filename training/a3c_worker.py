"""
A3C Worker — runs in its own process.

Each worker:
1.  Creates a LOCAL copy of the Actor-Critic model.
2.  Syncs local weights ← global model.
3.  Interacts with TradingEnv for T_MAX steps (n-step return).
4.  Computes advantage  A = R - V(s).
5.  Computes actor loss (policy gradient) + critic loss (MSE) + entropy bonus.
6.  Back-props on the local model, then pushes gradients to the global model
    via the shared optimizer (Hogwild-style, no lock).
7.  Logs episode results with WorkerLogger.

Reference: Mnih et al., "Asynchronous Methods for Deep RL" (2016)
           + paper arXiv:2405.19982 (A3C for Forex trading)
"""

import torch
import numpy as np

from env.trading_env import TradingEnv
from model.actor_critic import ActorCritic
from utils.workerLogger import WorkerLogger


# ─────────────────────────────────────────────────────────────── #
#                   HYPER-PARAMETERS (defaults)                   #
# ─────────────────────────────────────────────────────────────── #
T_MAX = 20          # n-step return horizon
GAMMA = 0.99        # discount factor
BETA_ENTROPY = 0.01 # entropy bonus coefficient
VALUE_COEFF = 0.5   # critic loss coefficient


# ─────────────────────────────────────────────────────────────── #
#                     N-STEP RETURN HELPER                        #
# ─────────────────────────────────────────────────────────────── #
def compute_n_step_returns(rewards: list, values: list,
                           done: bool, gamma: float = GAMMA):
    """
    Compute discounted n-step returns R_t for each step in the buffer.

    If the episode is NOT done, bootstrap from the last value:
        R = V(s_T)                       (bootstrap)
    If the episode IS done:
        R = 0

    Then walk backwards:
        R_t = r_t + gamma * R_{t+1}
    """
    R = 0.0 if done else values[-1].item()
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


# ─────────────────────────────────────────────────────────────── #
#                        MAIN WORKER                              #
# ─────────────────────────────────────────────────────────────── #
def a3c_worker(worker_id: int, global_model: ActorCritic,
               optimizer, train_data, n_episodes: int,
               metrics_queue, lock=None,
               t_max: int = T_MAX, gamma: float = GAMMA,
               beta_entropy: float = BETA_ENTROPY,
               value_coeff: float = VALUE_COEFF):
    """
    A3C worker process.

    Parameters
    ----------
    worker_id       : unique integer identifying this worker
    global_model    : shared-memory ActorCritic (all workers read/write)
    optimizer       : shared optimizer (its state is also in shared memory)
    train_data      : pandas DataFrame with market data
    n_episodes      : how many episodes this worker will run
    metrics_queue   : mp.Queue for logging
    lock            : optional threading.Lock / mp.Lock for synchronised updates
    t_max           : n-step horizon
    gamma           : discount factor
    beta_entropy    : entropy bonus weight
    value_coeff     : critic loss weight
    """

    torch.set_num_threads(1) #Solo 1 thread para operaciones tensoriales

    # ── 1.  Local model (same architecture, separate weights) ──
    obs_dim = global_model.shared[0].in_features
    n_actions = global_model.actor.out_features
    hidden = global_model.shared[0].out_features

    local_model = ActorCritic(obs_dim=obs_dim, n_actions=n_actions,
                              hidden=hidden)

    # ── 2.  Environment ──
    env = TradingEnv(data=train_data, window=20)

    # ── 3.  Logger ──
    logger = WorkerLogger(worker_id, metrics_queue)

    # ── 4.  Training loop ──
    for episode in range(n_episodes):
        # Sync local ← global
        local_model.load_state_dict(global_model.state_dict())

        obs, _ = env.reset()
        obs_t = torch.FloatTensor(obs)

        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done:
            # ── Collect n-step trajectory ──
            log_probs = []
            values = []
            rewards = []
            entropies = []

            for _ in range(t_max):
                probs, value = local_model(obs_t.unsqueeze(0))
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                obs_next, reward, terminated, truncated, info = env.step(action.item())

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                entropies.append(entropy)

                episode_reward += reward
                episode_steps += 1

                done = terminated or truncated
                obs_t = torch.FloatTensor(obs_next)

                if done:
                    break

            # ── Compute n-step returns & advantages ──
            # If not done, bootstrap with V(s_T)
            if not done:
                with torch.no_grad():
                    _, v_bootstrap = local_model(obs_t.unsqueeze(0))
                R = v_bootstrap.item()
            else:
                R = 0.0

            returns = []
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)

            returns_t = torch.FloatTensor(returns)
            log_probs_t = torch.stack(log_probs).squeeze()
            values_t = torch.stack(values).squeeze()
            entropies_t = torch.stack(entropies).squeeze()

            # Advantage A = R_t - V(s_t)
            advantages = returns_t - values_t.detach()

            # ── Losses ──
            actor_loss = -(log_probs_t * advantages).mean()
            critic_loss = value_coeff * (returns_t - values_t).pow(2).mean()
            entropy_loss = -beta_entropy * entropies_t.mean()

            total_loss = actor_loss + critic_loss + entropy_loss

            # ── Backprop on LOCAL model ──
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping (stabilizes A3C)
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=40.0)

            # ── Push gradients to GLOBAL model ──
            if lock is not None:
                lock.acquire()

            for local_param, global_param in zip(local_model.parameters(),
                                                  global_model.parameters()):
                if global_param.grad is None:
                    global_param._grad = local_param.grad.clone()
                else:
                    global_param._grad = local_param.grad.clone()

            optimizer.step()

            if lock is not None:
                lock.release()

            # Re-sync local ← global after each update
            local_model.load_state_dict(global_model.state_dict())

        # ── Episode done → log metrics ──
        final_value = info.get("portfolio_value", 0.0)
        logger.log_episode(episode, episode_reward, final_value, episode_steps)
