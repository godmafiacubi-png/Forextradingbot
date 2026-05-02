"""
RL Trading Agent — DQN with Experience Replay
Fixed: torch.load weights_only=True
"""

import os
import logging
import random
import numpy as np
from collections import deque, defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — RL agent will use random actions")


# ============================================================
# DQN Network
# ============================================================
if TORCH_AVAILABLE:
    class DQN(nn.Module):
        def __init__(self, state_size, action_size, hidden=128):
            super(DQN, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, action_size),
            )

        def forward(self, x):
            return self.net(x)


# ============================================================
# Experience Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states), np.array(actions), np.array(rewards),
            np.array(next_states), np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# RL Trading Agent
# ============================================================
class RLTradingAgent:
    """
    DQN-based RL agent for trade decision adjustment.

    Actions:
        0 = HOLD (skip trade)
        1 = BUY
        2 = SELL

    State:  [price_norm, atr_pct, rsi_norm, adx_norm, macd_norm,
             stoch_norm, htf_trend, ml_prob, confidence, has_position, pnl_pct]
    """

    STATE_SIZE = 11
    ACTION_SIZE = 3

    def __init__(self, model_dir='./models/rl', lr=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.998,
                 batch_size=64, target_update=100, buffer_size=10000):

        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.train_step = 0

        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')

        if TORCH_AVAILABLE:
            self.policy_net = DQN(self.STATE_SIZE, self.ACTION_SIZE).to(self.device)
            self.target_net = DQN(self.STATE_SIZE, self.ACTION_SIZE).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.criterion = nn.SmoothL1Loss()
        else:
            self.policy_net = None
            self.target_net = None

        self.replay_buffer = ReplayBuffer(buffer_size)

        # Tracking
        self.trade_history = defaultdict(list)
        self.open_states = {}
        self.open_actions = {}
        self.total_trades = 0
        self.total_wins = 0
        self.total_reward = 0.0
        self.hold_rewards = 0.0
        self.action_counts = defaultdict(int)

        # Load existing model
        self._load()

    def build_state(self, market_data, signal_data, has_position, pnl_pct):
        """Build state vector from market data"""
        try:
            price_norm = market_data.get('price_vs_sma200', 0)
            atr_pct = market_data.get('atr_pct', 0) * 100
            rsi_norm = (market_data.get('rsi', 50) - 50) / 50
            adx_norm = market_data.get('adx', 0) / 50
            macd_norm = np.clip(market_data.get('macd_hist', 0) * 10000, -5, 5)
            stoch_norm = (market_data.get('stoch_k', 50) - 50) / 50

            htf_map = {'BULL': 1, 'BEAR': -1, 'MIXED': 0}
            htf = htf_map.get(market_data.get('htf_trend', 'MIXED'), 0)

            ml_prob = signal_data.get('ml_prob', 0.5)
            confidence = signal_data.get('confidence', 0)

            state = np.array([
                price_norm, atr_pct, rsi_norm, adx_norm, macd_norm,
                stoch_norm, htf, ml_prob, confidence,
                1.0 if has_position else 0.0,
                np.clip(pnl_pct * 100, -10, 10),
            ], dtype=np.float32)

            state = np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
            return state

        except Exception as e:
            logger.debug(f"State build error: {e}")
            return np.zeros(self.STATE_SIZE, dtype=np.float32)

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if not TORCH_AVAILABLE or self.policy_net is None:
            return random.randint(0, self.ACTION_SIZE - 1)

        if random.random() < self.epsilon:
            return random.randint(0, self.ACTION_SIZE - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def get_rl_adjustment(self, state, base_signal, base_confidence):
        """
        Get RL-adjusted signal.

        Returns: (adjusted_signal, adjusted_confidence, rl_action, source)
        """
        rl_action = self.select_action(state)
        self.action_counts[rl_action] += 1

        # RL action mapping
        if rl_action == 0:
            # HOLD — reduce confidence
            adj_signal = base_signal
            adj_conf = base_confidence * 0.5
            source = 'rl_hold'
        elif rl_action == 1:
            # BUY
            if base_signal == 1:
                adj_signal = 1
                adj_conf = min(base_confidence * 1.1, 1.0)
                source = 'rl_confirm_buy'
            elif base_signal == -1:
                # RL disagrees — reduce confidence
                adj_signal = -1
                adj_conf = base_confidence * 0.7
                source = 'rl_disagree_buy'
            else:
                adj_signal = 0
                adj_conf = base_confidence
                source = 'rl_no_signal'
        elif rl_action == 2:
            # SELL
            if base_signal == -1:
                adj_signal = -1
                adj_conf = min(base_confidence * 1.1, 1.0)
                source = 'rl_confirm_sell'
            elif base_signal == 1:
                adj_signal = 1
                adj_conf = base_confidence * 0.7
                source = 'rl_disagree_sell'
            else:
                adj_signal = 0
                adj_conf = base_confidence
                source = 'rl_no_signal'
        else:
            adj_signal = base_signal
            adj_conf = base_confidence
            source = 'rl_passthrough'

        return adj_signal, adj_conf, rl_action, source

    def record_trade_open(self, symbol, state, action):
        """Record state when trade opens"""
        self.open_states[symbol] = state.copy()
        self.open_actions[symbol] = action
        self.total_trades += 1

    def record_trade_close(self, symbol, next_state, pnl, pnl_pct):
        """Record trade result and push to replay buffer"""
        if symbol not in self.open_states:
            return

        state = self.open_states.pop(symbol)
        action = self.open_actions.pop(symbol, 0)

        # Reward shaping
        if pnl > 0:
            reward = 1.0 + min(pnl_pct * 10, 5.0)
            self.total_wins += 1
        else:
            reward = -1.0 + max(pnl_pct * 10, -5.0)

        self.total_reward += reward

        # Push to replay buffer
        self.replay_buffer.push(state, action, reward, next_state, False)

        # Train if enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Track
        self.trade_history[symbol].append({
            'pnl': pnl, 'pnl_pct': pnl_pct, 'reward': reward,
            'action': action, 'epsilon': self.epsilon,
            'time': datetime.now().isoformat(),
        })

    def record_hold_reward(self, symbol, state, had_signal):
        """Small reward/penalty for holding"""
        if had_signal:
            reward = -0.1  # Missed opportunity
        else:
            reward = 0.05  # Good to wait

        self.hold_rewards += reward
        next_state = state.copy()
        self.replay_buffer.push(state, 0, reward, next_state, False)

    def _train_step(self):
        """Single training step on batch from replay buffer"""
        if not TORCH_AVAILABLE or self.policy_net is None:
            return
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_step += 1

        # Update target network
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_stats(self):
        """Get RL agent statistics"""
        wr = self.total_wins / self.total_trades if self.total_trades > 0 else 0
        return {
            'epsilon': round(self.epsilon, 4),
            'rl_trades': self.total_trades,
            'rl_wins': self.total_wins,
            'rl_win_rate': round(wr, 4),
            'total_reward': round(self.total_reward, 2),
            'hold_rewards': round(self.hold_rewards, 2),
            'buffer_size': len(self.replay_buffer),
            'train_steps': self.train_step,
            'action_counts': dict(self.action_counts),
            'device': str(self.device),
        }

    def _save(self):
        """Save model and state"""
        try:
            if TORCH_AVAILABLE and self.policy_net is not None:
                policy_path = os.path.join(self.model_dir, 'rl_policy.pth')
                target_path = os.path.join(self.model_dir, 'rl_target.pth')
                torch.save(self.policy_net.state_dict(), policy_path)
                torch.save(self.target_net.state_dict(), target_path)

            # Save metadata
            import json
            meta = {
                'epsilon': self.epsilon,
                'train_step': self.train_step,
                'total_trades': self.total_trades,
                'total_wins': self.total_wins,
                'total_reward': self.total_reward,
                'hold_rewards': self.hold_rewards,
                'action_counts': dict(self.action_counts),
                'saved_at': datetime.now().isoformat(),
            }
            meta_path = os.path.join(self.model_dir, 'rl_meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            logger.debug(f"RL saved (eps={self.epsilon:.4f}, steps={self.train_step})")

        except Exception as e:
            logger.error(f"RL save error: {e}")

    def _load(self):
        """Load model and state"""
        try:
            if TORCH_AVAILABLE and self.policy_net is not None:
                policy_path = os.path.join(self.model_dir, 'rl_policy.pth')
                target_path = os.path.join(self.model_dir, 'rl_target.pth')

                if os.path.exists(policy_path) and os.path.exists(target_path):
                    # ✅ FIX: เพิ่ม weights_only=True แก้ FutureWarning
                    self.policy_net.load_state_dict(
                        torch.load(policy_path, map_location=self.device, weights_only=True)
                    )
                    self.target_net.load_state_dict(
                        torch.load(target_path, map_location=self.device, weights_only=True)
                    )
                    logger.info(f"[RL] Loaded policy + target from {self.model_dir}")

            # Load metadata
            import json
            meta_path = os.path.join(self.model_dir, 'rl_meta.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.epsilon = meta.get('epsilon', self.epsilon)
                self.train_step = meta.get('train_step', 0)
                self.total_trades = meta.get('total_trades', 0)
                self.total_wins = meta.get('total_wins', 0)
                self.total_reward = meta.get('total_reward', 0.0)
                self.hold_rewards = meta.get('hold_rewards', 0.0)
                ac = meta.get('action_counts', {})
                for k, v in ac.items():
                    self.action_counts[int(k)] = v
                logger.info(f"[RL] Meta: eps={self.epsilon:.4f} trades={self.total_trades} WR={self.total_wins}/{self.total_trades}")

        except Exception as e:
            logger.warning(f"RL load error (starting fresh): {e}")