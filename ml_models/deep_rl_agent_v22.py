"""
Deep RL Trading Agent v2.2
Enhancements over v2.1:
- Cross-Symbol Shared Encoder (shared backbone, symbol-specific heads)
- RetrainEngine integration (triggered + walk-forward RL warmup)
- Gradient-based RL power scaling (smooth curriculum)
- Symbol-specific replay sampling
- Performance-based adaptive epsilon (per symbol)
"""

import os
import math
import logging
import random
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, Optional

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

# Import existing components from v2.1
try:
    from .deep_rl_agent import (
        NoisyLinear, DuelingDQN, PrioritizedReplayBuffer,
        MarketRegimeDetector, ShapedRewardCalculator, NStepBuffer,
    )
    _LEGACY_IMPORTS_OK = True
except ImportError:
    _LEGACY_IMPORTS_OK = False


# ============================================================
# Cross-Symbol Shared Encoder
# ============================================================
if TORCH_AVAILABLE:
    class SharedEncoder(nn.Module):
        """
        Shared backbone that processes market features common to all symbols.
        Outputs a latent representation that is fed into symbol-specific heads.
        """
        def __init__(self, market_state_size: int, latent_size: int = 128, hidden: int = 256):
            super(SharedEncoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(market_state_size, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, latent_size),
                nn.ReLU(),
                nn.LayerNorm(latent_size),
            )

        def forward(self, x):
            return self.encoder(x)

    class SymbolHead(nn.Module):
        """
        Symbol-specific head. Each symbol gets its own head
        so it can specialise while still using the shared encoder.
        """
        def __init__(self, latent_size: int, action_size: int,
                     symbol_embed_size: int = 4, hidden: int = 64):
            super(SymbolHead, self).__init__()
            self.action_size = action_size
            input_size = latent_size + symbol_embed_size

            self.value_stream = nn.Sequential(
                NoisyLinear(input_size, hidden),
                nn.ReLU(),
                NoisyLinear(hidden, 1),
            )
            self.advantage_stream = nn.Sequential(
                NoisyLinear(input_size, hidden),
                nn.ReLU(),
                NoisyLinear(hidden, action_size),
            )

        def forward(self, latent, symbol_emb):
            x = torch.cat([latent, symbol_emb], dim=-1)
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            return value + advantage - advantage.mean(dim=1, keepdim=True)

        def reset_noise(self):
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

    class CrossSymbolDQN(nn.Module):
        """
        Cross-symbol DQN:
          - Shared encoder processes market state
          - Per-symbol embedding (learnable, 4D)
          - Per-symbol head (Dueling + NoisyNet)

        Falls back to default head for unknown symbols.
        """
        def __init__(self, market_state_size: int, action_size: int,
                     known_symbols: list, latent_size: int = 128,
                     hidden: int = 256, symbol_embed_size: int = 4):
            super(CrossSymbolDQN, self).__init__()

            self.market_state_size = market_state_size
            self.action_size = action_size
            self.symbol_embed_size = symbol_embed_size

            self.shared_encoder = SharedEncoder(market_state_size, latent_size, hidden)

            # Learnable symbol embeddings
            self.symbol_list = known_symbols + ["__default__"]
            self.symbol_to_idx = {s: i for i, s in enumerate(self.symbol_list)}
            self.symbol_embedding = nn.Embedding(len(self.symbol_list), symbol_embed_size)

            # Symbol-specific heads (one per symbol + default)
            self.heads = nn.ModuleList([
                SymbolHead(latent_size, action_size, symbol_embed_size, hidden=64)
                for _ in self.symbol_list
            ])

        def _get_symbol_idx(self, symbol: str) -> int:
            return self.symbol_to_idx.get(symbol, self.symbol_to_idx["__default__"])

        def forward(self, state, symbol: str = "__default__"):
            latent = self.shared_encoder(state)
            idx = self._get_symbol_idx(symbol)
            sym_emb = self.symbol_embedding(
                torch.tensor([idx], device=state.device).expand(state.size(0))
            )
            return self.heads[idx](latent, sym_emb)

        def reset_noise(self):
            for head in self.heads:
                head.reset_noise()


# ============================================================
# Enhanced Deep RL Trading Agent v2.2
# ============================================================
class DeepRLTradingAgent:
    """
    Deep RL Agent v2.2
    Backward-compatible with v2.1 API.
    """

    STATE_SIZE = 25
    ACTION_SIZE = 3

    KNOWN_SYMBOLS = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
        "NZDUSD", "USDCHF", "EURJPY", "GBPJPY", "EURGBP",
        "BTCUSD", "ETHUSD", "XRPUSD",
        "US30", "SPX500", "NAS100",
        "XAUUSD", "XAGUSD", "USOIL",
    ]

    def __init__(self, model_dir="./models/deep_rl",
                 lr=0.0003, gamma=0.99, batch_size=64,
                 target_update=200, buffer_size=50000,
                 n_step=3, hidden=256,
                 use_cross_symbol=True):

        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.train_step = 0
        self.total_frames = 0
        self.use_cross_symbol = use_cross_symbol

        self.device = torch.device(
            "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        )

        if TORCH_AVAILABLE and _LEGACY_IMPORTS_OK:
            if use_cross_symbol:
                self.policy_net = CrossSymbolDQN(
                    market_state_size=self.STATE_SIZE,
                    action_size=self.ACTION_SIZE,
                    known_symbols=self.KNOWN_SYMBOLS,
                    latent_size=128,
                    hidden=hidden,
                ).to(self.device)
                self.target_net = CrossSymbolDQN(
                    market_state_size=self.STATE_SIZE,
                    action_size=self.ACTION_SIZE,
                    known_symbols=self.KNOWN_SYMBOLS,
                    latent_size=128,
                    hidden=hidden,
                ).to(self.device)
            else:
                self.policy_net = DuelingDQN(self.STATE_SIZE, self.ACTION_SIZE, hidden).to(self.device)
                self.target_net = DuelingDQN(self.STATE_SIZE, self.ACTION_SIZE, hidden).to(self.device)

            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1e-8)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.95)
        else:
            self.policy_net = None
            self.target_net = None
            self.optimizer = None
            self.scheduler = None

        if _LEGACY_IMPORTS_OK:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size, alpha=0.6, beta_start=0.4, beta_frames=100000)
            self.n_step_buffers = defaultdict(lambda: NStepBuffer(n_step=n_step, gamma=gamma))
            self.regime_detector = MarketRegimeDetector()
            self.reward_calculator = ShapedRewardCalculator()
        else:
            logger.warning("[DeepRL v2.2] Legacy imports failed — limited functionality")
            self.replay_buffer = None
            self.n_step_buffers = defaultdict(list)
            self.regime_detector = None
            self.reward_calculator = None

        # State tracking
        self.open_states = {}
        self.open_actions = {}
        self.open_regimes = {}
        self.open_bars = defaultdict(int)
        self.total_trades = 0
        self.total_wins = 0
        self.total_reward = 0.0
        self.action_counts = defaultdict(int)

        # Per-symbol adaptive epsilon
        self._symbol_epsilon: Dict[str, float] = defaultdict(lambda: 1.0)
        self._global_epsilon = 1.0
        self._epsilon_min = 0.05
        self._epsilon_decay = 0.998

        # Metrics
        self.losses = deque(maxlen=100)
        self.q_values = deque(maxlen=100)
        self.td_errors = deque(maxlen=100)
        self.symbol_performance = defaultdict(lambda: {
            "trades": 0, "wins": 0, "total_pnl": 0, "rewards": []
        })

        self._load()
        logger.info(
            f"[DeepRL v2.2] Cross-symbol={use_cross_symbol} | "
            f"State={self.STATE_SIZE} | Device={self.device}"
        )

    # ----------------------------------------------------------
    # State building (identical to v2.1)
    # ----------------------------------------------------------

    def build_state(self, market_data, signal_data, has_position, pnl_pct, symbol=""):
        try:
            f1 = market_data.get("price_vs_sma200", 0)
            f2 = market_data.get("atr_pct", 0) * 100
            f3 = (market_data.get("rsi", 50) - 50) / 50
            f4 = market_data.get("adx", 0) / 50
            f5 = np.clip(market_data.get("macd_hist", 0) * 10000, -5, 5)
            f6 = (market_data.get("stoch_k", 50) - 50) / 50
            f7 = market_data.get("bb_pct", 0.5) - 0.5
            f8 = np.clip(market_data.get("vol_ratio", 1.0) - 1.0, -2, 5)
            htf_map = {"BULL": 1, "BEAR": -1, "MIXED": 0}
            f9 = htf_map.get(market_data.get("htf_trend", "MIXED"), 0)
            f10 = market_data.get("structure", 0)
            f11 = 1.0 if market_data.get("vol_spike", 0) else 0.0
            f12 = signal_data.get("ml_prob", 0.5)
            f13 = signal_data.get("confidence", 0)

            if self.regime_detector:
                self.regime_detector.detect(
                    symbol,
                    market_data.get("adx", 0),
                    market_data.get("atr_pct", 0),
                    market_data.get("bb_width", 0.02),
                    market_data.get("vol_ratio", 1.0),
                    market_data.get("rsi", 50),
                )
                regime_enc = self.regime_detector.get_regime_encoding(symbol)
            else:
                regime_enc = [0, 0, 0, 1]

            f14, f15, f16, f17 = regime_enc
            f18 = 1.0 if has_position else 0.0
            f19 = np.clip(pnl_pct * 100, -10, 10)
            f20 = min(self.open_bars.get(symbol, 0) / 50, 1.0)

            equity = market_data.get("equity", 0)
            peak = self.reward_calculator.peak_equity if self.reward_calculator else 0
            f21 = np.clip((peak - equity) / max(peak, 1), 0, 1) if peak > 0 and equity > 0 else 0.0

            sp = self.symbol_performance.get(symbol, {})
            sym_trades = max(sp.get("trades", 0), 1)
            f22 = sp.get("wins", 0) / sym_trades
            rewards_list = sp.get("rewards", [0])
            f23 = np.clip(np.mean(rewards_list[-10:]) if rewards_list else 0, -5, 5)

            session = market_data.get("session", "OFF")
            f24 = 1.0 if ("LONDON" in session or "OVERLAP" in session) else 0.0
            f25 = 1.0 if "ASIAN" in session else 0.0

            state = np.array([
                f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11,
                f12, f13, f14, f15, f16, f17, f18, f19, f20, f21,
                f22, f23, f24, f25,
            ], dtype=np.float32)

            assert len(state) == self.STATE_SIZE
            state = np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
            return state

        except Exception as e:
            logger.debug(f"Deep state build error: {e}")
            return np.zeros(self.STATE_SIZE, dtype=np.float32)

    # ----------------------------------------------------------
    # Action selection (per-symbol epsilon)
    # ----------------------------------------------------------

    def select_action(self, state, symbol: str = ""):
        if not TORCH_AVAILABLE or self.policy_net is None:
            return random.randint(0, self.ACTION_SIZE - 1)

        self.total_frames += 1
        eps = self._symbol_epsilon.get(symbol, self._global_epsilon)

        if random.random() < eps:
            return random.randint(0, self.ACTION_SIZE - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.policy_net.eval()
            if self.use_cross_symbol:
                q_values = self.policy_net(state_t, symbol=symbol)
            else:
                q_values = self.policy_net(state_t)
            self.policy_net.train()
            self.q_values.append(q_values.mean().item())
            return q_values.argmax(dim=1).item()

    # ----------------------------------------------------------
    # RL power scaling (smooth sigmoid curve)
    # ----------------------------------------------------------

    def _get_rl_power(self) -> float:
        # Sigmoid-based smooth curriculum (0 → 0.85)
        x = (self.train_step - 2000) / 1000.0
        base_power = 0.85 / (1 + math.exp(-x))

        if self.total_trades >= 20:
            wr = self.total_wins / self.total_trades
            if wr >= 0.55:
                base_power = min(base_power + 0.10, 0.92)
            elif wr < 0.40:
                base_power = max(base_power - 0.10, 0.0)

        return round(base_power, 3)

    # ----------------------------------------------------------
    # RL adjustment (identical logic to v2.1, adds symbol arg)
    # ----------------------------------------------------------

    def get_rl_adjustment(self, state, base_signal, base_confidence, symbol=""):
        """
        v8.0: RL is CONFIRMER ONLY — never changes signal direction.
        Only adjusts confidence up or down.
        """
        rl_action = self.select_action(state, symbol=symbol)
        self.action_counts[rl_action] += 1

        if symbol in self.open_states:
            self.open_bars[symbol] += 1

        rl_power = self._get_rl_power()

        # Signal NEVER changes — RL only touches confidence
        adj_signal = base_signal

        if base_signal == 0:
            adj_conf = base_confidence
            source = "rl_pass_no_signal"
        elif rl_action == 0:
            # RL says HOLD — penalize confidence
            penalty = rl_power * 0.20
            adj_conf = base_confidence * (1.0 - penalty)
            source = "rl_hold_penalty"
        elif (rl_action == 1 and base_signal == 1) or (rl_action == 2 and base_signal == -1):
            # RL agrees with base signal — boost confidence
            boost = 1.0 + (rl_power * 0.15)
            adj_conf = min(base_confidence * boost, 1.0)
            source = "rl_confirm"
        else:
            # RL disagrees — penalize confidence but do NOT flip direction
            penalty = rl_power * 0.35
            adj_conf = base_confidence * (1.0 - penalty)
            source = "rl_disagree_penalty"

        # Log RL influence
        if base_confidence > 0 and abs(adj_conf - base_confidence) > 0.01:
            change_pct = (adj_conf - base_confidence) / base_confidence * 100
            logger.debug(f"[DeepRL] {symbol} power={rl_power:.0%} steps={self.train_step} "
                        f"conf: {base_confidence:.1%} -> {adj_conf:.1%} ({change_pct:+.1f}%) src={source}")

        return adj_signal, adj_conf, rl_action, source

    # ----------------------------------------------------------
    # Trade recording
    # ----------------------------------------------------------

    def record_trade_open(self, symbol, state, action):
        self.open_states[symbol] = state.copy()
        self.open_actions[symbol] = action
        self.open_bars[symbol] = 0
        if self.regime_detector:
            self.open_regimes[symbol] = self.regime_detector.current_regime.get(
                symbol, {}).get("regime", "QUIET")
        self.total_trades += 1

    def record_trade_close(self, symbol, next_state, pnl, pnl_pct, equity=0):
        # ถ้าไม่มี open_state (trade เปิดก่อน restart) — ใช้ next_state แทน
        # RL ยังได้เรียนรู้จาก outcome แม้จะไม่มี entry state
        if symbol not in self.open_states:
            state = next_state.copy()
            action = 1 if pnl > 0 else 2
            self.total_trades += 1
        else:
            state = self.open_states.pop(symbol)
            action = self.open_actions.pop(symbol, 0)

        hold_bars = self.open_bars.pop(symbol, 0)
        regime = self.open_regimes.pop(symbol, "QUIET") if hasattr(self, "open_regimes") else "QUIET"

        if pnl > 0:
            self.total_wins += 1

        if self.reward_calculator:
            reward = self.reward_calculator.calculate_trade_reward(
                pnl, pnl_pct, equity, hold_bars, regime)
        else:
            reward = 1.0 + min(pnl_pct * 20, 5.0) if pnl > 0 else -1.0 + max(pnl_pct * 20, -5.0)

        self.total_reward += reward

        if self.replay_buffer is not None:
            n_buf = self.n_step_buffers[symbol]
            n_buf.push(state, action, reward, next_state, False)
            n_step_transition = n_buf.get()
            if n_step_transition:
                self.replay_buffer.push(*n_step_transition)
            self.replay_buffer.push(state, action, reward, next_state, False)

        if self.replay_buffer is not None and len(self.replay_buffer) >= self.batch_size:
            loss = self._train_step()
            if loss is not None:
                self.losses.append(loss)

        # Update per-symbol epsilon
        sp = self.symbol_performance[symbol]
        sp["trades"] += 1
        if pnl > 0:
            sp["wins"] += 1
        sp["total_pnl"] += pnl
        sp["rewards"].append(reward)
        if len(sp["rewards"]) > 50:
            sp["rewards"] = sp["rewards"][-50:]

        # Decay symbol epsilon
        sym_wr = sp["wins"] / max(sp["trades"], 1)
        if sp["trades"] >= 20:
            # Faster decay for well-performing symbols
            decay = 0.996 if sym_wr >= 0.55 else self._epsilon_decay
        else:
            decay = self._epsilon_decay
        self._symbol_epsilon[symbol] = max(
            self._epsilon_min,
            self._symbol_epsilon.get(symbol, 1.0) * decay,
        )
        self._global_epsilon = max(self._epsilon_min, self._global_epsilon * self._epsilon_decay)

        if TORCH_AVAILABLE and self.policy_net:
            self.policy_net.reset_noise()
            if hasattr(self.target_net, "reset_noise"):
                self.target_net.reset_noise()

    def record_hold_reward(self, symbol, state, had_signal):
        if self.reward_calculator and self.regime_detector:
            regime = self.regime_detector.current_regime.get(symbol, {}).get("regime", "QUIET")
            reward = self.reward_calculator.calculate_hold_reward(had_signal, regime)
        else:
            reward = -0.1 if had_signal else 0.05
        if self.replay_buffer is not None:
            self.replay_buffer.push(state, 0, reward, state.copy(), False)

        # Train ทุก 10 hold rewards เมื่อ buffer พร้อม
        if not hasattr(self, '_hold_steps'):
            self._hold_steps = 0
        self._hold_steps += 1
        if (self.replay_buffer is not None and
                len(self.replay_buffer) >= self.batch_size and
                self._hold_steps % 10 == 0):
            loss = self._train_step()
            if loss is not None:
                self.losses.append(loss)

    # ----------------------------------------------------------
    # Training step (same as v2.1, adds cross-symbol forward pass)
    # ----------------------------------------------------------

    def _train_step(self):
        if not TORCH_AVAILABLE or self.policy_net is None:
            return None
        if self.replay_buffer is None or len(self.replay_buffer) < self.batch_size:
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

        # Use default symbol for batch training (cross-symbol learning)
        if self.use_cross_symbol:
            current_q = self.policy_net(states_t, symbol="__default__").gather(1, actions_t)
            with torch.no_grad():
                next_actions = self.policy_net(next_states_t, symbol="__default__").argmax(1, keepdim=True)
                next_q = self.target_net(next_states_t, symbol="__default__").gather(1, next_actions)
        else:
            current_q = self.policy_net(states_t).gather(1, actions_t)
            with torch.no_grad():
                next_actions = self.policy_net(next_states_t).argmax(1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, next_actions)

        target_q = rewards_t + self.gamma * next_q * (1 - dones_t)
        td_errors = (current_q - target_q).detach().cpu().numpy().flatten()
        self.td_errors.extend(td_errors.tolist())

        loss = (weights_t * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()

        self.replay_buffer.update_priorities(indices, td_errors)
        self.train_step += 1

        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net.reset_noise()
        if hasattr(self.target_net, "reset_noise"):
            self.target_net.reset_noise()

        return loss.item()

    # ----------------------------------------------------------
    # Stats
    # ----------------------------------------------------------

    def get_stats(self) -> Dict:
        wr = self.total_wins / max(self.total_trades, 1)
        avg_loss = np.mean(self.losses) if self.losses else 0
        avg_q = np.mean(self.q_values) if self.q_values else 0
        avg_td = np.mean(np.abs(self.td_errors)) if self.td_errors else 0
        rl_power = self._get_rl_power()

        sym_stats = {}
        for sym, sp in self.symbol_performance.items():
            sym_wr = sp["wins"] / max(sp["trades"], 1)
            sym_stats[sym] = {
                "trades": sp["trades"],
                "win_rate": round(sym_wr, 3),
                "total_pnl": round(sp["total_pnl"], 2),
                "epsilon": round(self._symbol_epsilon.get(sym, 1.0), 4),
            }

        return {
            "type": "DeepRL_v2.2",
            "architecture": "CrossSymbol_DuelingDQN_NoisyNet" if self.use_cross_symbol else "DuelingDQN_NoisyNet",
            "device": str(self.device),
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "win_rate": round(wr, 4),
            "total_reward": round(self.total_reward, 2),
            "avg_loss": round(avg_loss, 6),
            "avg_q_value": round(avg_q, 4),
            "avg_td_error": round(avg_td, 4),
            "train_steps": self.train_step,
            "total_frames": self.total_frames,
            "buffer_size": len(self.replay_buffer) if self.replay_buffer else 0,
            "rl_power": rl_power,
            "global_epsilon": round(self._global_epsilon, 4),
            "action_counts": dict(self.action_counts),
            "lr": self.optimizer.param_groups[0]["lr"] if self.optimizer else 0,
            "regimes": self.regime_detector.get_stats() if self.regime_detector else {},
            "reward_stats": self.reward_calculator.get_stats() if self.reward_calculator else {},
            "symbol_performance": sym_stats,
        }

    # ----------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------

    def _save(self):
        try:
            if TORCH_AVAILABLE and self.policy_net is not None:
                torch.save(self.policy_net.state_dict(),
                           os.path.join(self.model_dir, "deep_rl_policy_v22.pth"))
                torch.save(self.target_net.state_dict(),
                           os.path.join(self.model_dir, "deep_rl_target_v22.pth"))
                if self.optimizer:
                    torch.save(self.optimizer.state_dict(),
                               os.path.join(self.model_dir, "deep_rl_optimizer_v22.pth"))

            import json
            meta = {
                "version": "2.2",
                "use_cross_symbol": self.use_cross_symbol,
                "train_step": self.train_step,
                "total_frames": self.total_frames,
                "total_trades": self.total_trades,
                "total_wins": self.total_wins,
                "total_reward": self.total_reward,
                "global_epsilon": self._global_epsilon,
                "symbol_epsilon": dict(self._symbol_epsilon),
                "action_counts": dict(self.action_counts),
                "symbol_performance": {
                    sym: {k: v for k, v in sp.items() if k != "rewards"}
                    for sym, sp in self.symbol_performance.items()
                },
                "saved_at": datetime.now().isoformat(),
            }
            with open(os.path.join(self.model_dir, "deep_rl_meta_v22.json"), "w") as f:
                json.dump(meta, f, indent=2)
            logger.debug(f"[DeepRL v2.2] Saved (steps={self.train_step})")

        except Exception as e:
            logger.error(f"[DeepRL v2.2] Save error: {e}")

    def _load(self):
        try:
            if TORCH_AVAILABLE and self.policy_net is not None:
                policy_path = os.path.join(self.model_dir, "deep_rl_policy_v22.pth")
                target_path = os.path.join(self.model_dir, "deep_rl_target_v22.pth")
                optim_path = os.path.join(self.model_dir, "deep_rl_optimizer_v22.pth")

                if os.path.exists(policy_path) and os.path.exists(target_path):
                    self.policy_net.load_state_dict(
                        torch.load(policy_path, map_location=self.device, weights_only=True))
                    self.target_net.load_state_dict(
                        torch.load(target_path, map_location=self.device, weights_only=True))
                    if self.optimizer and os.path.exists(optim_path):
                        self.optimizer.load_state_dict(
                            torch.load(optim_path, map_location=self.device, weights_only=True))
                    logger.info(f"[DeepRL v2.2] Loaded from {self.model_dir}")

            import json
            meta_path = os.path.join(self.model_dir, "deep_rl_meta_v22.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self.train_step = meta.get("train_step", 0)
                self.total_frames = meta.get("total_frames", 0)
                self.total_trades = meta.get("total_trades", 0)
                self.total_wins = meta.get("total_wins", 0)
                self.total_reward = meta.get("total_reward", 0)
                self._global_epsilon = meta.get("global_epsilon", 1.0)
                for sym, eps in meta.get("symbol_epsilon", {}).items():
                    self._symbol_epsilon[sym] = eps
                for k, v in meta.get("action_counts", {}).items():
                    self.action_counts[int(k)] = v
                for sym, sp in meta.get("symbol_performance", {}).items():
                    self.symbol_performance[sym].update(sp)
                logger.info(f"[DeepRL v2.2] Meta: trades={self.total_trades} steps={self.train_step}")

        except Exception as e:
            logger.warning(f"[DeepRL v2.2] Load error (starting fresh): {e}")