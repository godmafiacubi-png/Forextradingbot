"""
ML Hub v1.0 — Central integration point
Wires together:
  - EnsembleModel v2.0 (Regime-Aware + Cross-Symbol)
  - DeepRLTradingAgent v2.2 (Cross-Symbol DQN)
  - TemporalEncoderWrapper (LSTM + Attention + Transformer)
  - MetaLearner (when to trade)
  - RetrainEngine (walk-forward + triggered)

Public API (drop-in, replaces scattered calls):
  hub.on_bar(symbol, market_data, signal_data)
  hub.get_signal(symbol, base_signal, base_confidence, market_data, signal_data)
  hub.on_trade_open(symbol, ...)
  hub.on_trade_close(symbol, pnl, pnl_pct, equity)
  hub.get_stats()
  hub.save() / hub.load()
"""

import os
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

from .ensemble import EnsembleModel
from .temporal_encoder import TemporalEncoderWrapper
from .meta_learner import MetaLearner

try:
    from .deep_rl_agent_v22 import DeepRLTradingAgent
    _RL_V22 = True
except ImportError:
    try:
        from .deep_rl_agent import DeepRLTradingAgent
        _RL_V22 = False
        logger.warning("[MLHub] Using DeepRL v2.1 (v2.2 not found)")
    except ImportError:
        DeepRLTradingAgent = None
        _RL_V22 = False


# ============================================================
# Signal result dataclass
# ============================================================
@dataclass
class SignalResult:
    signal: int               # -1, 0, 1
    confidence: float         # 0.0 – 1.0
    ml_prob: float            # raw ensemble probability
    rl_action: int            # 0=hold, 1=buy, 2=sell
    rl_power: float           # current RL authority
    meta_activity: float      # meta-learner activity score
    meta_trade: bool          # meta-learner says trade?
    meta_conf_scale: float    # confidence multiplier from meta
    source: str               # e.g. "deep_rl_confirm_buy"
    regime: str               # current market regime
    temporal_enriched: bool   # was temporal encoder used?
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ============================================================
# ML Hub
# ============================================================
class MLHub:
    """
    Single entry-point for the entire ML stack.

    Typical usage in your trading bot:

        hub = MLHub(model_dir="./models")
        hub.load()

        # Every bar:
        hub.on_bar(symbol, market_data, signal_data)

        # When you have a base signal:
        result = hub.get_signal(symbol, base_signal=1, base_confidence=0.65,
                                market_data=md, signal_data=sd)
        if result.meta_trade and result.confidence > 0.60:
            # enter trade
            hub.on_trade_open(symbol, result.signal)

        # When trade closes:
        hub.on_trade_close(symbol, pnl=120.0, pnl_pct=0.012, equity=10120.0)

        # Periodic save:
        hub.save()
    """

    def __init__(
        self,
        model_dir: str = "./models",
        use_temporal_encoder: bool = True,
        use_meta_learner: bool = True,
        use_rl: bool = True,
        use_regime_models: bool = True,
        seq_len: int = 32,
        temporal_output_size: int = 128,
        rl_state_size: int = 25,
        retrain_every_n_trades: int = 100,
        meta_activity_threshold: float = 0.40,
        # Walk-forward retrain tuning
        # bot loop=60s, 5 symbols → ~5 bars/min → 300 bars/hr
        # min_bars=200  : รอแค่ ~40 นาทีก็ retrain ได้
        # train_window=500: ใช้ข้อมูลย้อนหลัง ~100 นาที
        # min_new_bars=50 : trigger ทุก ~10 นาทีที่มีข้อมูลใหม่
        train_window: int = 500,
        min_bars: int = 200,
        min_new_bars_for_wf: int = 50,
    ):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.use_temporal = use_temporal_encoder
        self.use_meta = use_meta_learner
        self.use_rl = use_rl

        # ---- Ensemble (v2.0 with regime-aware + cross-symbol) ----
        self.ensemble = EnsembleModel(
            use_regime_models=use_regime_models,
            use_retrain_engine=True,
            retrain_every_n_trades=retrain_every_n_trades,
            train_window=train_window,
            min_bars=min_bars,
            min_new_bars_for_wf=min_new_bars_for_wf,
        )

        # ---- Temporal encoder ----
        self.temporal: Optional[TemporalEncoderWrapper] = None
        if use_temporal_encoder:
            self.temporal = TemporalEncoderWrapper(
                raw_feature_size=rl_state_size,
                output_size=temporal_output_size,
                seq_len=seq_len,
            )

        # ---- Meta-learner ----
        self.meta: Optional[MetaLearner] = None
        if use_meta_learner:
            self.meta = MetaLearner(
                activity_threshold=meta_activity_threshold,
                min_train_samples=200,
                train_every=50,
            )

        # ---- RL agent ----
        self.rl: Optional[Any] = None
        if use_rl and DeepRLTradingAgent is not None:
            rl_dir = os.path.join(model_dir, "deep_rl")
            kwargs = dict(
                model_dir=rl_dir,
                batch_size=64,
                buffer_size=50000,
            )
            if _RL_V22:
                kwargs["use_cross_symbol"] = True
            self.rl = DeepRLTradingAgent(**kwargs)

        # ---- Open trade state ----
        self._open_states: Dict[str, np.ndarray] = {}
        self._open_actions: Dict[str, int] = {}
        self._open_meta_ctx: Dict[str, np.ndarray] = {}

        # ---- RL state cache (populated by get_signal, consumed by callers) ----
        self._last_rl_state: Dict[str, np.ndarray] = {}

        logger.info(
            f"[MLHub] Init: temporal={use_temporal_encoder} "
            f"meta={use_meta_learner} rl={use_rl} "
            f"regime_models={use_regime_models}"
        )

    # ----------------------------------------------------------
    # Bar feed (call every bar, before get_signal)
    # ----------------------------------------------------------

    def on_bar(
        self,
        symbol: str,
        market_data: Dict,
        signal_data: Dict,
    ):
        """
        Feed new bar data. Must be called before get_signal on the same bar.
        """
        # Build raw RL state for temporal buffer
        if self.rl is not None:
            raw_state = self.rl.build_state(
                market_data, signal_data,
                has_position=(symbol in self._open_states),
                pnl_pct=0.0,
                symbol=symbol,
            )
        else:
            raw_state = np.zeros(25, dtype=np.float32)

        # Push to temporal buffer
        if self.temporal:
            self.temporal.push(symbol, raw_state)

        # Push bar to ensemble retrain engine
        # ต้องมี 'c' (close price) เพื่อให้ retrain engine สร้าง target y ได้
        # sym_data จาก main.py ใช้ key 'price' → map เป็น 'c'
        price = (
            market_data.get('c') or
            market_data.get('close') or
            market_data.get('price') or
            market_data.get('bid') or
            0.0
        )
        bar_dict = {
            **market_data,
            **signal_data,
            "symbol": symbol,
            "c": price,   # บังคับให้มีเสมอ — retrain engine ต้องการสร้าง y
        }
        self.ensemble.on_bar(bar_dict)

        # Update symbol stats in ensemble
        self.ensemble.update_symbol_stats(
            symbol,
            atr_pct=market_data.get("atr_pct", 0.0),
            vol_ratio=market_data.get("vol_ratio", 1.0),
        )

        # Update meta-learner ATR history
        if self.meta:
            self.meta.record_atr(symbol, market_data.get("atr_pct", 0.0))

    # ----------------------------------------------------------
    # Signal generation
    # ----------------------------------------------------------

    def get_signal(
        self,
        symbol: str,
        base_signal: int,
        base_confidence: float,
        market_data: Dict,
        signal_data: Dict,
        has_position: bool = False,
        pnl_pct: float = 0.0,
    ) -> SignalResult:
        """
        Main entry point. Returns SignalResult with fully adjusted signal + confidence.
        """
        # ---- Regime context ----
        regime = "GLOBAL"
        regime_confidence = 1.0
        if self.rl and hasattr(self.rl, "regime_detector") and self.rl.regime_detector:
            regime_info = self.rl.regime_detector.current_regime.get(symbol, {})
            regime = regime_info.get("regime", "GLOBAL")
            regime_confidence = regime_info.get("confidence", 1.0)

        # Set context on ensemble
        self.ensemble.set_context(symbol, regime, regime_confidence)

        # ---- Build RL state ----
        if self.rl:
            raw_state = self.rl.build_state(
                market_data, signal_data, has_position, pnl_pct, symbol
            )
        else:
            raw_state = np.zeros(25, dtype=np.float32)

        # Cache for callers that need the state after get_signal() returns
        self._last_rl_state[symbol] = raw_state

        # ---- Temporal encoding ----
        temporal_enriched = False
        if self.temporal:
            enriched = self.temporal.encode_or_raw(symbol, raw_state)
            temporal_enriched = (enriched is not raw_state and
                                  self.temporal.seq_buffer.is_ready(symbol, min_steps=4))
        else:
            enriched = raw_state

        # ---- Meta-learner ----
        meta_activity = 0.6
        meta_trade = True
        meta_conf_scale = 1.0
        meta_source = "meta_disabled"

        if self.meta:
            session = market_data.get("session", "OFF")
            meta_ctx = self.meta.context.build(
                symbol=symbol,
                regime=regime,
                regime_confidence=regime_confidence,
                atr_pct=market_data.get("atr_pct", 0.0),
                adx=market_data.get("adx", 0.0),
                session=session,
                ml_prob=signal_data.get("ml_prob", 0.5),
                base_confidence=base_confidence,
                timestamp=datetime.now(),
            )
            meta_result = self.meta.evaluate(meta_ctx, base_confidence)
            meta_activity = meta_result.activity_score
            meta_trade = meta_result.trade
            meta_conf_scale = meta_result.confidence_scale
            meta_source = meta_result.source

            # Store meta context for recording outcome later
            self._open_meta_ctx[symbol] = meta_ctx

        # ---- RL adjustment ----
        rl_action = 1
        rl_source = "rl_disabled"
        rl_power = 0.0

        if self.rl:
            adj_signal, adj_conf, rl_action, rl_source = self.rl.get_rl_adjustment(
                raw_state,   # ใช้ raw_state (25 dims) เสมอ — DQN expect STATE_SIZE=25
                base_signal,
                base_confidence * meta_conf_scale,
                symbol=symbol if _RL_V22 else "",
            )
            rl_power = self.rl._get_rl_power()
        else:
            adj_signal = base_signal
            adj_conf = base_confidence * meta_conf_scale

        # ---- Final confidence clamp ----
        final_signal = adj_signal
        final_confidence = float(np.clip(adj_conf, 0.0, 1.0))

        # If meta says stay out, soft-suppress confidence
        if not meta_trade:
            final_confidence *= 0.3

        return SignalResult(
            signal=final_signal,
            confidence=final_confidence,
            ml_prob=float(signal_data.get("ml_prob", 0.5)),
            rl_action=rl_action,
            rl_power=rl_power,
            meta_activity=meta_activity,
            meta_trade=meta_trade,
            meta_conf_scale=meta_conf_scale,
            source=rl_source,
            regime=regime,
            temporal_enriched=temporal_enriched,
        )

    # ----------------------------------------------------------
    # RL state accessor
    # ----------------------------------------------------------

    def get_last_rl_state(self, symbol: str) -> Optional[np.ndarray]:
        """Return the RL state that was built during the most recent
        get_signal() call for *symbol*.  Returns None if get_signal()
        has not been called yet for that symbol.
        """
        return self._last_rl_state.get(symbol)

    # ----------------------------------------------------------
    # Trade lifecycle
    # ----------------------------------------------------------

    def on_trade_open(self, symbol: str, action: int, market_data: Dict,
                      signal_data: Dict, pnl_pct: float = 0.0):
        """Record trade open"""
        if self.rl:
            raw_state = self.rl.build_state(
                market_data, signal_data, True, pnl_pct, symbol
            )
            self._open_states[symbol] = raw_state
            self._open_actions[symbol] = action
            self.rl.record_trade_open(symbol, raw_state, action)

    def on_trade_close(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        equity: float = 0.0,
        market_data: Optional[Dict] = None,
        signal_data: Optional[Dict] = None,
    ):
        """Record trade close — updates all components"""
        market_data = market_data or {}
        signal_data = signal_data or {}

        # RL
        if self.rl:
            next_state = self.rl.build_state(
                market_data, signal_data, False, 0.0, symbol
            )
            buf_before = len(self.rl.replay_buffer) if self.rl.replay_buffer else 0
            self.rl.record_trade_close(symbol, next_state, pnl, pnl_pct, equity)
            buf_after = len(self.rl.replay_buffer) if self.rl.replay_buffer else 0
            logger.info(
                f"[MLHub] on_trade_close {symbol} pnl={pnl:.2f} "
                f"buf: {buf_before}→{buf_after} "
                f"open_states_had={symbol in self.rl.open_states or buf_after > buf_before}"
            )

        # Ensemble retrain engine
        self.ensemble.on_trade_closed(
            pnl=pnl,
            pnl_pct=pnl_pct,
            symbol=symbol,
            atr_pct=market_data.get("atr_pct", 0.0),
            vol_ratio=market_data.get("vol_ratio", 1.0),
        )

        # Meta-learner outcome
        if self.meta:
            self.meta.record_trade(symbol, pnl, pnl_pct)
            ctx = self._open_meta_ctx.pop(symbol, None)
            if ctx is not None:
                self.meta.record_outcome(ctx, pnl > 0)

        # Clean up
        self._open_states.pop(symbol, None)
        self._open_actions.pop(symbol, None)

        logger.debug(
            f"[MLHub] Trade closed: {symbol} pnl=${pnl:.2f} ({pnl_pct:.2%})"
        )

    def on_hold(self, symbol: str, had_signal: bool, market_data: Dict,
                signal_data: Dict):
        """Call when no trade was entered (hold reward for RL)"""
        if self.rl:
            raw_state = (
                self._last_rl_state[symbol] if symbol in self._last_rl_state
                else self.rl.build_state(market_data, signal_data, False, 0.0, symbol)
            )
            self.rl.record_hold_reward(symbol, raw_state, had_signal)

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def train(self, df, symbol: str = "") -> bool:
        """Initial training of ensemble + regime models"""
        self.ensemble.set_context(symbol, "GLOBAL", 1.0)
        return self.ensemble.train(df, symbol=symbol) if hasattr(self.ensemble.train, "__code__") \
               and "symbol" in self.ensemble.train.__code__.co_varnames \
               else self.ensemble.train(df)

    # ----------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------

    def save(self):
        """Save all components"""
        self.ensemble.save(os.path.join(self.model_dir, "ensemble"))
        if self.temporal:
            self.temporal.save(os.path.join(self.model_dir, "temporal_encoder.pth"))
        if self.meta:
            self.meta.save(os.path.join(self.model_dir, "meta_learner.pkl"))
        if self.rl and hasattr(self.rl, "_save"):
            self.rl._save()
        logger.info(f"[MLHub] All components saved to {self.model_dir}")

    def load(self):
        """Load all components"""
        self.ensemble.load(os.path.join(self.model_dir, "ensemble"))
        if self.temporal:
            self.temporal.load(os.path.join(self.model_dir, "temporal_encoder.pth"))
        if self.meta:
            self.meta.load(os.path.join(self.model_dir, "meta_learner.pkl"))
        logger.info(f"[MLHub] All components loaded from {self.model_dir}")

    # ----------------------------------------------------------
    # Stats
    # ----------------------------------------------------------

    def get_stats(self) -> Dict:
        stats: Dict = {"timestamp": datetime.now().isoformat()}
        stats["ensemble"] = self.ensemble.get_stats() if hasattr(self.ensemble, "get_stats") else {}
        if self.temporal:
            stats["temporal_encoder"] = self.temporal.get_stats()
        if self.meta:
            stats["meta_learner"] = self.meta.get_stats()
        if self.rl and hasattr(self.rl, "get_stats"):
            stats["rl_agent"] = self.rl.get_stats()
        return stats