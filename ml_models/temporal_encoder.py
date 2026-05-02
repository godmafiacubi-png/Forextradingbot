"""
Temporal State Encoder v1.0
- LSTM encoder for recent candle sequences
- Multi-Head Self-Attention over time steps
- Lightweight Transformer block (2 layers)
- Outputs enriched state vector for RL agent + ML models
- Plugs in between raw features and DQN / Ensemble
"""

import logging
import numpy as np
from collections import deque
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — TemporalEncoder disabled")


# ============================================================
# 1. Positional Encoding
# ============================================================
if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for Transformer"""
        def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x):
            # x: (batch, seq, d_model)
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


# ============================================================
# 2. LSTM Encoder
# ============================================================
if TORCH_AVAILABLE:
    class LSTMEncoder(nn.Module):
        """
        Bidirectional LSTM over the sequence of market states.
        Returns the last hidden state as a fixed-size vector.
        """
        def __init__(self, input_size: int, hidden_size: int = 64,
                     num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            # Bidirectional: hidden = 2 * hidden_size
            self.proj = nn.Linear(hidden_size * 2, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)

        def forward(self, x):
            # x: (batch, seq, input_size)
            out, (h, _) = self.lstm(x)
            # Concatenate last forward + backward hidden
            h_fwd = h[-2]   # last layer, forward
            h_bwd = h[-1]   # last layer, backward
            h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (batch, 2*hidden)
            return self.norm(F.relu(self.proj(h_cat)))  # (batch, hidden)


# ============================================================
# 3. Temporal Self-Attention
# ============================================================
if TORCH_AVAILABLE:
    class TemporalAttention(nn.Module):
        """
        Multi-head self-attention over time steps.
        Learns which candles in the window matter most.
        Returns:
          - context vector (attended sum) of shape (batch, d_model)
          - attention weights (batch, n_heads, seq, seq) for interpretability
        """
        def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads,
                dropout=dropout, batch_first=True,
            )
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            # x: (batch, seq, d_model)
            attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
            # Aggregate over time dim → (batch, d_model)
            context = attn_out.mean(dim=1)
            return self.norm(context), attn_weights


# ============================================================
# 4. Mini Transformer Block
# ============================================================
if TORCH_AVAILABLE:
    class TransformerEncoderBlock(nn.Module):
        """
        Single Transformer encoder block:
        LayerNorm → Multi-Head Attention → residual
        LayerNorm → FFN → residual
        """
        def __init__(self, d_model: int, n_heads: int = 4,
                     ffn_dim: int = 256, dropout: float = 0.1):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads,
                dropout=dropout, batch_first=True,
            )
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            # Self-attention with residual
            x2 = self.norm1(x)
            attn_out, _ = self.attn(x2, x2, x2)
            x = x + attn_out
            # FFN with residual
            x = x + self.ffn(self.norm2(x))
            return x


# ============================================================
# 5. Full Temporal State Encoder
# ============================================================
if TORCH_AVAILABLE:
    class TemporalStateEncoder(nn.Module):
        """
        Full temporal encoder pipeline:

          raw_features (seq, F)
            ↓  Linear projection → d_model
            ↓  Positional Encoding
            ↓  Transformer blocks (n_layers=2)
            ↓  LSTM encoder
            ↓  Temporal Attention
            ↓  Fusion layer
            → enriched_state (output_size,)

        Output is a drop-in replacement for the flat state vector
        fed into the DQN / Ensemble models.
        """
        def __init__(
            self,
            raw_feature_size: int,
            d_model: int = 64,
            lstm_hidden: int = 64,
            n_transformer_layers: int = 2,
            n_heads: int = 4,
            output_size: int = 128,
            dropout: float = 0.1,
            max_seq_len: int = 32,
        ):
            super().__init__()
            self.d_model = d_model
            self.output_size = output_size

            # Project raw features → d_model
            self.input_proj = nn.Sequential(
                nn.Linear(raw_feature_size, d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model),
            )
            self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

            # Transformer layers
            self.transformer_layers = nn.ModuleList([
                TransformerEncoderBlock(d_model, n_heads=n_heads, ffn_dim=d_model * 4, dropout=dropout)
                for _ in range(n_transformer_layers)
            ])

            # LSTM on top of Transformer output
            self.lstm = LSTMEncoder(d_model, hidden_size=lstm_hidden, num_layers=2, dropout=dropout)

            # Attention over Transformer output
            self.temporal_attn = TemporalAttention(d_model, n_heads=n_heads, dropout=dropout)

            # Fusion: LSTM output + Attention context + last-step feature
            fusion_input = lstm_hidden + d_model + d_model
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input, output_size),
                nn.ReLU(),
                nn.LayerNorm(output_size),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            """
            x: (batch, seq, raw_feature_size)
            Returns: (batch, output_size)
            """
            # Project + positional encoding
            x = self.input_proj(x)           # (B, S, d_model)
            x = self.pos_enc(x)              # (B, S, d_model)

            # Transformer
            for layer in self.transformer_layers:
                x = layer(x)                 # (B, S, d_model)

            # LSTM
            lstm_out = self.lstm(x)          # (B, lstm_hidden)

            # Attention
            attn_ctx, attn_weights = self.temporal_attn(x)  # (B, d_model)

            # Last time step
            last_step = x[:, -1, :]          # (B, d_model)

            # Fuse all three signals
            fused = torch.cat([lstm_out, attn_ctx, last_step], dim=-1)
            return self.fusion(fused), attn_weights  # (B, output_size), weights


# ============================================================
# 6. Sequence Buffer (feeds raw candles to encoder)
# ============================================================
class SequenceBuffer:
    """
    Rolling buffer of the last `seq_len` feature vectors per symbol.
    Thread-safe via list copy.
    """
    def __init__(self, seq_len: int = 32, feature_size: int = 25):
        self.seq_len = seq_len
        self.feature_size = feature_size
        self._buffers: Dict[str, deque] = {}

    def _get_buf(self, symbol: str) -> deque:
        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=self.seq_len)
        return self._buffers[symbol]

    def push(self, symbol: str, features: np.ndarray):
        """Add one timestep of features"""
        f = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        self._get_buf(symbol).append(f.astype(np.float32))

    def get_sequence(self, symbol: str) -> Optional[np.ndarray]:
        """
        Returns (seq_len, feature_size) array, zero-padded at the start
        if fewer than seq_len steps have been seen.
        """
        buf = self._get_buf(symbol)
        data = list(buf)
        if len(data) == 0:
            return np.zeros((self.seq_len, self.feature_size), dtype=np.float32)
        # Pad with zeros at the front
        if len(data) < self.seq_len:
            pad = [np.zeros(self.feature_size, dtype=np.float32)] * (self.seq_len - len(data))
            data = pad + data
        return np.stack(data, axis=0)  # (seq_len, feature_size)

    def is_ready(self, symbol: str, min_steps: int = 8) -> bool:
        return len(self._get_buf(symbol)) >= min_steps

    def get_stats(self) -> Dict:
        return {sym: len(buf) for sym, buf in self._buffers.items()}


# ============================================================
# 7. TemporalEncoderWrapper (stateful, numpy interface)
# ============================================================
class TemporalEncoderWrapper:
    """
    High-level wrapper used by trading agent.

    Usage:
        encoder = TemporalEncoderWrapper(raw_feature_size=25, output_size=128)

        # Each bar:
        encoder.push(symbol, raw_state_vector)

        # Before DQN select_action:
        enriched = encoder.encode(symbol)  # → np.ndarray (output_size,)

        # Or fallback to raw if not enough data:
        enriched = encoder.encode_or_raw(symbol, raw_state_vector)
    """

    def __init__(
        self,
        raw_feature_size: int = 25,
        d_model: int = 64,
        lstm_hidden: int = 64,
        output_size: int = 128,
        seq_len: int = 32,
        n_transformer_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        self.raw_feature_size = raw_feature_size
        self.output_size = output_size
        self.seq_len = seq_len

        self.device = torch.device(
            device if device else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        )

        self.seq_buffer = SequenceBuffer(seq_len=seq_len, feature_size=raw_feature_size)
        self._enabled = TORCH_AVAILABLE

        if TORCH_AVAILABLE:
            self.model = TemporalStateEncoder(
                raw_feature_size=raw_feature_size,
                d_model=d_model,
                lstm_hidden=lstm_hidden,
                n_transformer_layers=n_transformer_layers,
                n_heads=n_heads,
                output_size=output_size,
                dropout=dropout,
                max_seq_len=seq_len,
            ).to(self.device)
            self.model.eval()

            # Optimizer for supervised pre-training / fine-tuning
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            self._trained = False
            logger.info(
                f"[TemporalEncoder] LSTM+Attention+Transformer | "
                f"feature={raw_feature_size} seq={seq_len} out={output_size} device={self.device}"
            )
            if model_path and __import__("os").path.exists(model_path):
                self.load(model_path)
        else:
            self.model = None
            self._trained = False

    # ----------------------------------------------------------
    # Buffer management
    # ----------------------------------------------------------

    def push(self, symbol: str, raw_state: np.ndarray):
        """Add a timestep. Call once per bar."""
        self.seq_buffer.push(symbol, raw_state)

    # ----------------------------------------------------------
    # Encoding
    # ----------------------------------------------------------

    def encode(self, symbol: str) -> Optional[np.ndarray]:
        """
        Returns enriched state (output_size,) or None if not ready.
        """
        if not self._enabled or self.model is None:
            return None
        if not self.seq_buffer.is_ready(symbol, min_steps=4):
            return None

        seq = self.seq_buffer.get_sequence(symbol)  # (seq_len, F)
        return self._forward(seq)

    def encode_or_raw(self, symbol: str, raw_state: np.ndarray) -> np.ndarray:
        """
        Returns enriched state if ready, else raw state.
        Always returns a 1-D numpy array.
        """
        enriched = self.encode(symbol)
        if enriched is not None:
            return enriched
        # Pad raw state to output_size with zeros
        padded = np.zeros(self.output_size, dtype=np.float32)
        n = min(len(raw_state), self.output_size)
        padded[:n] = raw_state[:n]
        return padded

    def _forward(self, seq: np.ndarray) -> np.ndarray:
        """Run encoder on (seq_len, F) numpy array → (output_size,) numpy"""
        try:
            t = torch.FloatTensor(seq).unsqueeze(0).to(self.device)  # (1, S, F)
            with torch.no_grad():
                out, _ = self.model(t)
            return out.squeeze(0).cpu().numpy()
        except Exception as e:
            logger.debug(f"[TemporalEncoder] Forward error: {e}")
            return np.zeros(self.output_size, dtype=np.float32)

    def get_attention_weights(self, symbol: str) -> Optional[np.ndarray]:
        """
        Returns attention weights (n_heads, seq, seq) for the symbol.
        Useful for visualising which candles the model attends to.
        """
        if not self._enabled or self.model is None:
            return None
        if not self.seq_buffer.is_ready(symbol):
            return None
        seq = self.seq_buffer.get_sequence(symbol)
        try:
            t = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, weights = self.model(t)
            return weights.squeeze(0).cpu().numpy()  # (n_heads, seq, seq)
        except Exception as e:
            logger.debug(f"[TemporalEncoder] Attention error: {e}")
            return None

    # ----------------------------------------------------------
    # Supervised fine-tuning (optional — pre-train on historical data)
    # ----------------------------------------------------------

    def pretrain_step(self, sequences: np.ndarray, targets: np.ndarray) -> float:
        """
        One gradient step for supervised pre-training.
        sequences: (batch, seq_len, F)
        targets:   (batch,) binary 0/1 (next-bar direction)
        Returns: loss value
        """
        if not TORCH_AVAILABLE or self.model is None:
            return 0.0
        self.model.train()
        seqs_t = torch.FloatTensor(sequences).to(self.device)
        tgts_t = torch.FloatTensor(targets).to(self.device)

        encoded, _ = self.model(seqs_t)  # (batch, output_size)
        # Simple linear head for pre-training
        if not hasattr(self, "_pt_head"):
            self._pt_head = nn.Linear(self.output_size, 1).to(self.device)
            self._pt_optimizer = torch.optim.Adam(
                list(self.model.parameters()) + list(self._pt_head.parameters()), lr=1e-4
            )
        logits = self._pt_head(encoded).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, tgts_t)
        self._pt_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self._pt_optimizer.step()
        self.model.eval()
        self._trained = True
        return loss.item()

    # ----------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------

    def save(self, path: str):
        if not TORCH_AVAILABLE or self.model is None:
            return
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"[TemporalEncoder] Saved to {path}")
        except Exception as e:
            logger.error(f"[TemporalEncoder] Save error: {e}")

    def load(self, path: str):
        if not TORCH_AVAILABLE or self.model is None:
            return
        try:
            self.model.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            self._trained = True
            logger.info(f"[TemporalEncoder] Loaded from {path}")
        except Exception as e:
            logger.warning(f"[TemporalEncoder] Load error: {e}")

    def get_stats(self) -> Dict:
        n_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        return {
            "enabled": self._enabled,
            "trained": self._trained,
            "output_size": self.output_size,
            "seq_len": self.seq_len,
            "n_params": n_params,
            "device": str(self.device),
            "symbol_buffers": self.seq_buffer.get_stats(),
        }
