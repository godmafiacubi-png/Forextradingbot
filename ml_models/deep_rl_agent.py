"""
Deep RL Trading Agent v2.1
- Dueling DQN + Noisy Nets
- Prioritized Experience Replay (PER)
- Shaped Reward (Sharpe + DD penalty)
- Market Regime Detection
- Multi-symbol Knowledge Sharing
- N-step Returns
- RL Power Scaling (ยังไม่ train → ไม่มีอำนาจ nerf signal)
"""

import os
import math
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
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — Deep RL will use fallback")


# ============================================================
# 1. NOISY LINEAR LAYER
# ============================================================
if TORCH_AVAILABLE:
    class NoisyLinear(nn.Module):
        def __init__(self, in_features, out_features, sigma_init=0.5):
            super(NoisyLinear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features

            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
            self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))

            self.sigma_init = sigma_init
            self.reset_parameters()
            self.reset_noise()

        def reset_parameters(self):
            bound = 1 / math.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-bound, bound)
            self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-bound, bound)
            self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

        def _scale_noise(self, size):
            x = torch.randn(size, device=self.weight_mu.device)
            return x.sign().mul_(x.abs().sqrt_())

        def reset_noise(self):
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)

        def forward(self, x):
            if self.training:
                return F.linear(x,
                    self.weight_mu + self.weight_sigma * self.weight_epsilon,
                    self.bias_mu + self.bias_sigma * self.bias_epsilon)
            else:
                return F.linear(x, self.weight_mu, self.bias_mu)


# ============================================================
# 2. DUELING DQN + NOISY NET
# ============================================================
if TORCH_AVAILABLE:
    class DuelingDQN(nn.Module):
        def __init__(self, state_size, action_size, hidden=256):
            super(DuelingDQN, self).__init__()

            self.feature = nn.Sequential(
                nn.Linear(state_size, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden // 2),
            )

            self.value_stream = nn.Sequential(
                NoisyLinear(hidden // 2, hidden // 4),
                nn.ReLU(),
                NoisyLinear(hidden // 4, 1),
            )

            self.advantage_stream = nn.Sequential(
                NoisyLinear(hidden // 2, hidden // 4),
                nn.ReLU(),
                NoisyLinear(hidden // 4, action_size),
            )

        def forward(self, x):
            features = self.feature(x)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q

        def reset_noise(self):
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


# ============================================================
# 3. PRIORITIZED EXPERIENCE REPLAY (PER)
# ============================================================
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            if data is None or (isinstance(data, (int, float)) and data == 0):
                continue
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        if len(batch) == 0:
            return None, None, None

        probs = np.array(priorities) / (self.tree.total() + 1e-10)
        weights = (self.tree.n_entries * probs + 1e-10) ** (-beta)
        weights /= weights.max()

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        return (states, actions, rewards, next_states, dones), indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# ============================================================
# 4. MARKET REGIME DETECTOR
# ============================================================
class MarketRegimeDetector:
    REGIMES = ['TRENDING', 'RANGING', 'VOLATILE', 'QUIET']

    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=50))
        self.current_regime = {}

    def detect(self, symbol, adx, atr_pct, bb_width, vol_ratio, rsi):
        scores = {'TRENDING': 0, 'RANGING': 0, 'VOLATILE': 0, 'QUIET': 0}

        if adx > 30: scores['TRENDING'] += 3
        elif adx > 25: scores['TRENDING'] += 2
        elif adx > 20: scores['TRENDING'] += 1
        if rsi > 65 or rsi < 35: scores['TRENDING'] += 1

        if adx < 20: scores['RANGING'] += 2
        if adx < 15: scores['RANGING'] += 1
        if 40 < rsi < 60: scores['RANGING'] += 1

        if atr_pct > 0.02: scores['VOLATILE'] += 2
        if bb_width > 0.03: scores['VOLATILE'] += 2
        if vol_ratio > 1.5: scores['VOLATILE'] += 1

        if atr_pct < 0.005: scores['QUIET'] += 2
        if bb_width < 0.01: scores['QUIET'] += 2
        if vol_ratio < 0.5: scores['QUIET'] += 1

        regime = max(scores, key=scores.get)
        confidence = scores[regime] / (sum(scores.values()) + 1e-10)

        self.current_regime[symbol] = {
            'regime': regime, 'confidence': confidence, 'scores': scores,
        }
        self.history[symbol].append(regime)
        return regime, confidence

    def get_regime_encoding(self, symbol):
        regime = self.current_regime.get(symbol, {}).get('regime', 'QUIET')
        encoding = [0, 0, 0, 0]
        idx = self.REGIMES.index(regime) if regime in self.REGIMES else 3
        encoding[idx] = 1
        return encoding

    def get_stats(self):
        return {sym: data['regime'] for sym, data in self.current_regime.items()}


# ============================================================
# 5. SHAPED REWARD CALCULATOR
# ============================================================
class ShapedRewardCalculator:
    def __init__(self, risk_free_rate=0.0001):
        self.risk_free = risk_free_rate
        self.returns_history = deque(maxlen=100)
        self.peak_equity = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0

    def calculate_trade_reward(self, pnl, pnl_pct, equity, hold_bars, regime, rr_ratio=1.5):
        reward = 0.0

        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            # Prop firm: reward only if R:R was good
            if rr_ratio >= 2.0:
                reward += 2.0 + min(pnl_pct * 20, 5.0)   # excellent RR
            elif rr_ratio >= 1.5:
                reward += 1.0 + min(pnl_pct * 20, 5.0)   # acceptable RR
            else:
                reward += 0.3   # profit but bad RR = not a good trade
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            reward += -1.0 + max(pnl_pct * 20, -5.0)

        self.returns_history.append(pnl_pct)
        if len(self.returns_history) >= 10:
            returns = np.array(self.returns_history)
            sharpe = (returns.mean() - self.risk_free) / (returns.std() + 1e-10)
            reward += np.clip(sharpe * 0.5, -2, 2)

        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity
            if dd > 0.05: reward -= dd * 10
            elif dd > 0.02: reward -= dd * 5

        if hold_bars > 24: reward -= (hold_bars - 24) * 0.02   # penalize holding too long (opportunity cost)
        if hold_bars > 48: reward -= (hold_bars - 48) * 0.05   # cumulative extra penalty for very long holds
        if self.consecutive_wins >= 3: reward += 0.3 * min(self.consecutive_wins - 2, 5)
        if self.consecutive_losses >= 2: reward -= 0.5 * min(self.consecutive_losses - 1, 5)

        if regime == 'TRENDING' and pnl > 0: reward += 0.5
        elif regime == 'RANGING' and pnl > 0: reward += 0.3
        elif regime == 'VOLATILE' and pnl < 0: reward -= 0.3
        elif regime == 'QUIET': reward -= 0.2   # penalize trades in QUIET market

        return np.clip(reward, -10, 10)

    def calculate_hold_reward(self, had_signal, regime):
        if regime == 'QUIET': return 0.1
        elif regime == 'VOLATILE': return 0.15
        elif regime == 'TRENDING' and had_signal: return -0.2
        elif regime == 'RANGING': return 0.05
        return 0.0

    def get_stats(self):
        wr = 0
        if self.returns_history:
            wins = sum(1 for r in self.returns_history if r > 0)
            wr = wins / len(self.returns_history)
        return {
            'recent_returns': len(self.returns_history),
            'win_rate': round(wr, 3),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'peak_equity': round(self.peak_equity, 2),
        }


# ============================================================
# 6. N-STEP RETURN CALCULATOR
# ============================================================
class NStepBuffer:
    def __init__(self, n_step=3, gamma=0.99):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def get(self):
        if len(self.buffer) < self.n_step:
            return None
        reward = 0
        for i in range(self.n_step):
            reward += self.buffer[i][2] * (self.gamma ** i)
        state = self.buffer[0][0]
        action = self.buffer[0][1]
        next_state = self.buffer[-1][3]
        done = self.buffer[-1][4]
        return state, action, reward, next_state, done

    def reset(self):
        self.buffer.clear()


# ============================================================
# 7. DEEP RL TRADING AGENT
# ============================================================
class DeepRLTradingAgent:
    """
    Deep RL Agent v2.1
    State: 25 features
    Actions: 0=HOLD, 1=BUY, 2=SELL
    RL Power Scaling: ยังไม่ train → ไม่มีอำนาจ nerf signal
    """

    STATE_SIZE = 25
    ACTION_SIZE = 3

    def __init__(self, model_dir='./models/deep_rl',
                 lr=0.0003, gamma=0.99, batch_size=64,
                 target_update=200, buffer_size=50000,
                 n_step=3, hidden=256):

        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.train_step = 0
        self.total_frames = 0

        self.device = torch.device('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')

        if TORCH_AVAILABLE:
            self.policy_net = DuelingDQN(self.STATE_SIZE, self.ACTION_SIZE, hidden).to(self.device)
            self.target_net = DuelingDQN(self.STATE_SIZE, self.ACTION_SIZE, hidden).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1e-8)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.95)
        else:
            self.policy_net = None
            self.target_net = None

        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_size, alpha=0.6, beta_start=0.4, beta_frames=100000)

        self.n_step_buffers = defaultdict(lambda: NStepBuffer(n_step=n_step, gamma=gamma))
        self.regime_detector = MarketRegimeDetector()
        self.reward_calculator = ShapedRewardCalculator()

        self.open_states = {}
        self.open_actions = {}
        self.open_regimes = {}
        self.open_bars = defaultdict(int)
        self.total_trades = 0
        self.total_wins = 0
        self.total_reward = 0.0
        self.action_counts = defaultdict(int)

        self.losses = deque(maxlen=100)
        self.q_values = deque(maxlen=100)
        self.td_errors = deque(maxlen=100)

        self.symbol_performance = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'total_pnl': 0, 'rewards': []
        })

        self._load()
        logger.info(f"[DeepRL] Dueling DQN + PER + N-step | State: {self.STATE_SIZE} | Device: {self.device}")

    def build_state(self, market_data, signal_data, has_position, pnl_pct, symbol=''):
        """Build state vector — 25 features"""
        try:
            f1 = market_data.get('price_vs_sma200', 0)
            f2 = market_data.get('atr_pct', 0) * 100
            f3 = (market_data.get('rsi', 50) - 50) / 50
            f4 = market_data.get('adx', 0) / 50
            f5 = np.clip(market_data.get('macd_hist', 0) * 10000, -5, 5)
            f6 = (market_data.get('stoch_k', 50) - 50) / 50
            f7 = market_data.get('bb_pct', 0.5) - 0.5
            f8 = np.clip(market_data.get('vol_ratio', 1.0) - 1.0, -2, 5)

            htf_map = {'BULL': 1, 'BEAR': -1, 'MIXED': 0}
            f9 = htf_map.get(market_data.get('htf_trend', 'MIXED'), 0)
            f10 = market_data.get('structure', 0)
            f11 = 1.0 if market_data.get('vol_spike', 0) else 0.0

            f12 = signal_data.get('ml_prob', 0.5)
            f13 = signal_data.get('confidence', 0)

            self.regime_detector.detect(
                symbol,
                market_data.get('adx', 0),
                market_data.get('atr_pct', 0),
                market_data.get('bb_width', 0.02),
                market_data.get('vol_ratio', 1.0),
                market_data.get('rsi', 50)
            )
            regime_enc = self.regime_detector.get_regime_encoding(symbol)
            f14 = regime_enc[0]
            f15 = regime_enc[1]
            f16 = regime_enc[2]
            f17 = regime_enc[3]

            f18 = 1.0 if has_position else 0.0
            f19 = np.clip(pnl_pct * 100, -10, 10)
            f20 = min(self.open_bars.get(symbol, 0) / 50, 1.0)

            equity = market_data.get('equity', 0)
            peak = self.reward_calculator.peak_equity
            if peak > 0 and equity > 0:
                f21 = np.clip((peak - equity) / peak, 0, 1)
            else:
                f21 = 0.0

            sym_perf = self.symbol_performance.get(symbol, {})
            sym_trades = max(sym_perf.get('trades', 0), 1)
            f22 = sym_perf.get('wins', 0) / sym_trades
            rewards_list = sym_perf.get('rewards', [0])
            f23 = np.clip(np.mean(rewards_list[-10:]) if rewards_list else 0, -5, 5)

            session = market_data.get('session', 'OFF')
            f24 = 1.0 if 'LONDON' in session or 'OVERLAP' in session else 0.0
            f25 = 1.0 if 'ASIAN' in session else 0.0

            state = np.array([
                f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11,
                f12, f13,
                f14, f15, f16, f17,
                f18, f19, f20,
                f21,
                f22, f23,
                f24, f25,
            ], dtype=np.float32)

            assert len(state) == self.STATE_SIZE, f"State size {len(state)} != {self.STATE_SIZE}"
            state = np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
            return state

        except Exception as e:
            logger.debug(f"Deep state build error: {e}")
            return np.zeros(self.STATE_SIZE, dtype=np.float32)

    def select_action(self, state):
        if not TORCH_AVAILABLE or self.policy_net is None:
            return random.randint(0, self.ACTION_SIZE - 1)

        self.total_frames += 1

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.policy_net.eval()
            q_values = self.policy_net(state_t)
            self.policy_net.train()
            self.q_values.append(q_values.mean().item())
            return q_values.argmax(dim=1).item()

    def _get_rl_power(self):
        """
        คำนวณอำนาจ RL ตาม training progress
        ยังไม่ train → 0% | train มากขึ้น → อำนาจมากขึ้น
        """
        if self.train_step < 100:
            power = 0.0
        elif self.train_step < 500:
            power = 0.15
        elif self.train_step < 2000:
            power = 0.30
        elif self.train_step < 5000:
            power = 0.50
        elif self.train_step < 10000:
            power = 0.70
        else:
            power = 0.85

        # WR bonus/penalty
        if self.total_trades >= 20:
            wr = self.total_wins / self.total_trades
            if wr >= 0.55:
                power = min(power + 0.15, 0.95)
            elif wr < 0.40:
                power = max(power - 0.15, 0.0)

        return power

    def get_rl_adjustment(self, state, base_signal, base_confidence, symbol=''):
        rl_action = self.select_action(state)
        self.action_counts[rl_action] += 1

        if symbol in self.open_states:
            self.open_bars[symbol] += 1

        rl_power = self._get_rl_power()

        # RL เป็น CONFIRMER เท่านั้น — ห้ามเปลี่ยน direction
        adj_signal = base_signal  # signal ไม่เปลี่ยน

        if rl_action == 0:  # RL says HOLD
            penalty = rl_power * 0.20   # reduce confidence max 20%
            adj_conf = base_confidence * (1.0 - penalty)
            source = 'rl_hold_penalty'
        elif (rl_action == 1 and base_signal == 1) or (rl_action == 2 and base_signal == -1):
            # RL agrees with base signal
            boost = 1.0 + (rl_power * 0.15)   # boost max 15%
            adj_conf = min(base_confidence * boost, 1.0)
            source = 'rl_confirm'
        elif (rl_action == 1 and base_signal == -1) or (rl_action == 2 and base_signal == 1):
            # RL disagrees — penalty but don't flip
            penalty = rl_power * 0.35   # reduce confidence max 35%
            adj_conf = base_confidence * (1.0 - penalty)
            source = 'rl_disagree_penalty'
        else:
            # base_signal == 0 or other
            adj_conf = base_confidence
            source = 'rl_pass'

        # Log RL influence
        if base_confidence > 0 and abs(adj_conf - base_confidence) > 0.01:
            change_pct = (adj_conf - base_confidence) / base_confidence * 100
            logger.debug(f"[DeepRL] {symbol} power={rl_power:.0%} steps={self.train_step} "
                        f"conf: {base_confidence:.1%} -> {adj_conf:.1%} ({change_pct:+.1f}%) src={source}")

        return adj_signal, adj_conf, rl_action, source

    def record_trade_open(self, symbol, state, action):
        self.open_states[symbol] = state.copy()
        self.open_actions[symbol] = action
        self.open_bars[symbol] = 0
        self.open_regimes[symbol] = self.regime_detector.current_regime.get(
            symbol, {}).get('regime', 'QUIET')
        self.total_trades += 1

    def record_trade_close(self, symbol, next_state, pnl, pnl_pct, equity=0, rr_ratio=1.5):
        if symbol not in self.open_states:
            return

        state = self.open_states.pop(symbol)
        action = self.open_actions.pop(symbol, 0)
        hold_bars = self.open_bars.pop(symbol, 0)
        regime = self.open_regimes.pop(symbol, 'QUIET')

        if pnl > 0:
            self.total_wins += 1

        reward = self.reward_calculator.calculate_trade_reward(
            pnl, pnl_pct, equity, hold_bars, regime, rr_ratio=rr_ratio)
        self.total_reward += reward

        n_buf = self.n_step_buffers[symbol]
        n_buf.push(state, action, reward, next_state, False)
        n_step_transition = n_buf.get()
        if n_step_transition:
            self.replay_buffer.push(*n_step_transition)

        self.replay_buffer.push(state, action, reward, next_state, False)

        if len(self.replay_buffer) >= self.batch_size:
            loss = self._train_step()
            if loss is not None:
                self.losses.append(loss)

        sp = self.symbol_performance[symbol]
        sp['trades'] += 1
        if pnl > 0: sp['wins'] += 1
        sp['total_pnl'] += pnl
        sp['rewards'].append(reward)
        if len(sp['rewards']) > 50:
            sp['rewards'] = sp['rewards'][-50:]

        if TORCH_AVAILABLE and self.policy_net:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

        logger.debug(f"[DeepRL] {symbol} closed: pnl=${pnl:.2f} reward={reward:.2f} regime={regime}")

    def record_hold_reward(self, symbol, state, had_signal):
        regime = self.regime_detector.current_regime.get(symbol, {}).get('regime', 'QUIET')
        reward = self.reward_calculator.calculate_hold_reward(had_signal, regime)
        next_state = state.copy()
        self.replay_buffer.push(state, 0, reward, next_state, False)

    def _train_step(self):
        if not TORCH_AVAILABLE or self.policy_net is None:
            return None
        if len(self.replay_buffer) < self.batch_size:
            return None

        result = self.replay_buffer.sample(self.batch_size)
        if result[0] is None:
            return None

        (states, actions, rewards, next_states, dones), indices, weights = result

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        td_errors = (current_q - target_q).detach().cpu().numpy().flatten()
        self.td_errors.extend(td_errors.tolist())

        loss = (weights_t * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()

        self.replay_buffer.update_priorities(indices, td_errors)
        self.train_step += 1

        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug(f"[DeepRL] Target net updated (step {self.train_step})")

        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        return loss.item()

    def get_stats(self):
        wr = self.total_wins / max(self.total_trades, 1)
        avg_loss = np.mean(self.losses) if self.losses else 0
        avg_q = np.mean(self.q_values) if self.q_values else 0
        avg_td = np.mean(np.abs(self.td_errors)) if self.td_errors else 0
        rl_power = self._get_rl_power()

        sym_stats = {}
        for sym, sp in self.symbol_performance.items():
            sym_wr = sp['wins'] / max(sp['trades'], 1)
            sym_stats[sym] = {
                'trades': sp['trades'],
                'win_rate': round(sym_wr, 3),
                'total_pnl': round(sp['total_pnl'], 2),
            }

        return {
            'type': 'DeepRL_v2.1',
            'architecture': 'Dueling_DQN_NoisyNet',
            'device': str(self.device),
            'total_trades': self.total_trades,
            'total_wins': self.total_wins,
            'win_rate': round(wr, 4),
            'total_reward': round(self.total_reward, 2),
            'avg_loss': round(avg_loss, 6),
            'avg_q_value': round(avg_q, 4),
            'avg_td_error': round(avg_td, 4),
            'train_steps': self.train_step,
            'total_frames': self.total_frames,
            'buffer_size': len(self.replay_buffer),
            'rl_power': round(rl_power, 2),
            'action_counts': dict(self.action_counts),
            'lr': self.optimizer.param_groups[0]['lr'] if TORCH_AVAILABLE and self.policy_net else 0,
            'regimes': self.regime_detector.get_stats(),
            'reward_stats': self.reward_calculator.get_stats(),
            'symbol_performance': sym_stats,
        }

    def _save(self):
        try:
            if TORCH_AVAILABLE and self.policy_net is not None:
                torch.save(self.policy_net.state_dict(),
                          os.path.join(self.model_dir, 'deep_rl_policy.pth'))
                torch.save(self.target_net.state_dict(),
                          os.path.join(self.model_dir, 'deep_rl_target.pth'))
                torch.save(self.optimizer.state_dict(),
                          os.path.join(self.model_dir, 'deep_rl_optimizer.pth'))

            import json
            meta = {
                'train_step': self.train_step,
                'total_frames': self.total_frames,
                'total_trades': self.total_trades,
                'total_wins': self.total_wins,
                'total_reward': self.total_reward,
                'action_counts': dict(self.action_counts),
                'symbol_performance': {
                    sym: {k: v for k, v in sp.items() if k != 'rewards'}
                    for sym, sp in self.symbol_performance.items()
                },
                'saved_at': datetime.now().isoformat(),
            }
            with open(os.path.join(self.model_dir, 'deep_rl_meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)
            logger.debug(f"[DeepRL] Saved (steps={self.train_step})")

        except Exception as e:
            logger.error(f"[DeepRL] Save error: {e}")

    def _load(self):
        try:
            if TORCH_AVAILABLE and self.policy_net is not None:
                policy_path = os.path.join(self.model_dir, 'deep_rl_policy.pth')
                target_path = os.path.join(self.model_dir, 'deep_rl_target.pth')
                optim_path = os.path.join(self.model_dir, 'deep_rl_optimizer.pth')

                if os.path.exists(policy_path) and os.path.exists(target_path):
                    self.policy_net.load_state_dict(
                        torch.load(policy_path, map_location=self.device, weights_only=True))
                    self.target_net.load_state_dict(
                        torch.load(target_path, map_location=self.device, weights_only=True))
                    if os.path.exists(optim_path):
                        self.optimizer.load_state_dict(
                            torch.load(optim_path, map_location=self.device, weights_only=True))
                    logger.info(f"[DeepRL] Loaded from {self.model_dir}")

            import json
            meta_path = os.path.join(self.model_dir, 'deep_rl_meta.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                self.train_step = meta.get('train_step', 0)
                self.total_frames = meta.get('total_frames', 0)
                self.total_trades = meta.get('total_trades', 0)
                self.total_wins = meta.get('total_wins', 0)
                self.total_reward = meta.get('total_reward', 0)
                for k, v in meta.get('action_counts', {}).items():
                    self.action_counts[int(k)] = v
                for sym, sp in meta.get('symbol_performance', {}).items():
                    self.symbol_performance[sym].update(sp)
                logger.info(f"[DeepRL] Meta: trades={self.total_trades} steps={self.train_step}")

        except Exception as e:
            logger.warning(f"[DeepRL] Load error (starting fresh): {e}")