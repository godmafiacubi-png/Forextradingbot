"""
AGGRESSIVE MODE — Target 15-20%/month | DD < 10%
Risk ปานกลาง-สูง | Compounding ON | Confidence Scaling ON
Optimised: BE เร็ว + TP กว้าง + filter เข้มขึ้น
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# BOT MODE
# ============================================================
BOT_MODE = 'AGGRESSIVE'
TRADING_MODE = 'AGGRESSIVE'

# ============================================================
# MT5 CONNECTION
# ============================================================
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '0'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
MT5_SERVER = os.getenv('MT5_SERVER', '')
MT5_PATH = os.getenv('MT5_PATH', '')

# ============================================================
# DASHBOARD
# ============================================================
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '5004'))

# ============================================================
# TELEGRAM
# ============================================================
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# ============================================================
# SYMBOLS & TIMEFRAMES
# ============================================================
SYMBOLS = {
    'FOREX': ['EURUSDm', 'GBPUSDm', 'USDJPYm'],
    'CRYPTO': ['BTCUSDm'],
    'GOLD': ['XAUUSDm'],
}

SYMBOL_POINTS = {
    'EURUSDm': 0.00001,     # digits=5 point=1e-05
    'GBPUSDm': 0.00001,     # digits=5 point=1e-05
    'USDJPYm': 0.0001,       # digits=3 point=0.001
    'BTCUSDm': 0.01,        # digits=2 point=0.01
    'XAUUSDm': 0.001,       # digits=3 point=0.001
}

SYMBOL_CURRENCIES = {
    'EURUSDm': ['EUR', 'USD'],
    'GBPUSDm': ['GBP', 'USD'],
    'USDJPYm': ['USD', 'JPY'],
    'BTCUSDm': ['BTC', 'USD'],
    'XAUUSDm': ['XAU', 'USD'],
}

CORRELATION_GROUPS = {
    'EUR_GBP': ['EURUSDm', 'GBPUSDm'],
    'USD_MAJORS': ['EURUSDm', 'GBPUSDm', 'USDJPYm'],  # ทุกคู่ที่ correlate กับ USD
}

# จำกัดให้เปิด BUY USD pairs พร้อมกันได้แค่ 1 ตัว (EUR+GBP+JPY ทิศเดียวกัน = risky)
MAX_SAME_DIRECTION_CORRELATED = 2

SYMBOL_BEST_SESSIONS = {
    'EURUSDm': ['ASIAN', 'LONDON', 'NY', 'OVERLAP'],
    'GBPUSDm': ['ASIAN', 'LONDON', 'NY', 'OVERLAP'],
    'USDJPYm': ['ASIAN', 'LONDON', 'NY', 'OVERLAP'],
    'BTCUSDm': ['ASIAN', 'LONDON', 'NY', 'OVERLAP'],
    'XAUUSDm': ['ASIAN', 'LONDON', 'NY', 'OVERLAP'],
}

import MetaTrader5 as mt5

TIMEFRAMES = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
}

# ============================================================
# TIMEFRAME
# ============================================================
LOOKBACK_PERIOD = 200
PRIMARY_TIMEFRAME = 'H1'
HIGHER_TIMEFRAME = 'H4'
LOWER_TIMEFRAME = 'M30'
UPDATE_INTERVAL = 60

# ============================================================
# ML THRESHOLDS
# ============================================================
ML_THRESHOLD_BUY = 0.53
ML_THRESHOLD_SELL = 0.47
MIN_CONFIDENCE = 0.55
SIGNAL_COOLDOWN = 2

# ============================================================
# POSITION SIZING
# ============================================================
POSITION_SIZING_METHOD = 'ATR'
ACCOUNT_RISK_PERCENT = 0.7
MAX_DAILY_RISK_PERCENT = 5.0
MAX_OPEN_TRADES = 5
MAX_TRADES_PER_SYMBOL = 1

# ============================================================
# SL/TP
# ============================================================
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0
TRAILING_STOP_ATR = 1.0
BREAKEVEN_ATR = 0.6
MAX_DRAWDOWN_PERCENT = 10.0

# ============================================================
# FILTERS
# ============================================================
MIN_ADX = 24
MIN_ICT_SCORE = 2
PULLBACK_RSI_BUY_MAX = 60
PULLBACK_RSI_SELL_MIN = 40

REQUIRE_PULLBACK = True
REQUIRE_HTF_ALIGNMENT = False
REQUIRE_ICT_CONFLUENCE = True

# ============================================================
# QUALITY FILTER
# ============================================================
MIN_QUALITY_SCORE = 55    # เพิ่มจาก 50 → 55 กรองสัญญาณคุณภาพต่ำออก (A/B grade only)

# ============================================================
# QUIET MARKET KILL SWITCH
# ============================================================
QUIET_MARKET_ADX_THRESHOLD = 25  # block trades when regime=QUIET and ADX below this

# ============================================================
# SESSION FILTER — True = เปิดกรอง / False = ปิดกรอง (เทรดได้ทุกเวลา)
# ============================================================
SESSION_FILTER_ENABLED = False

# ============================================================
# M30 MULTI-TF
# ============================================================
USE_M30 = True
M30_CONFIRMATION = 'signal'
M30_CONF_BOOST = 0.12
M30_CONF_PENALTY = 0.10

# ============================================================
# DAILY LIMITS
# ============================================================
DAILY_LOSS_LIMIT_PCT = 5.0
DAILY_PROFIT_TARGET_PCT = 10.0
CONSECUTIVE_LOSS_COOLDOWN = 2
COOLDOWN_MINUTES = 60

# ============================================================
# PARTIAL CLOSE
# ============================================================
PARTIAL_CLOSE_ENABLED = True
PARTIAL_CLOSE_1_ATR = 0.5
PARTIAL_CLOSE_1_PCT = 0.35
PARTIAL_CLOSE_2_ATR = 1.0
PARTIAL_CLOSE_2_PCT = 0.35

# ============================================================
# RECOVERY
# ============================================================
RECOVERY_DRAWDOWN_TRIGGER = 6.0
RECOVERY_RISK_PERCENT = 0.5
RECOVERY_MAX_TRADES = 2

# ============================================================
# CORRELATION
# ============================================================
CORRELATION_FILTER_ENABLED = True
MAX_SAME_DIRECTION_CORRELATED = 2
MAX_CURRENCY_EXPOSURE = 5

# ============================================================
# SPREAD
# ============================================================
MAX_SPREAD_MULTIPLIER = 3.0
SPREAD_AVG_PERIOD = 50

# ============================================================
# COMPOUNDING & CONFIDENCE SCALING — เปิด!
# ============================================================
COMPOUNDING_ENABLED = True
CONFIDENCE_SCALING_ENABLED = True
CONFIDENCE_SCALING_MIN = 0.5
CONFIDENCE_SCALING_MAX = 1.5

# ============================================================
# AUTO-RESTART
# ============================================================
AUTO_RESTART_ENABLED = True
MAX_RESTART_ATTEMPTS = 5
RESTART_DELAY_SECONDS = 30
HEALTH_CHECK_INTERVAL = 60

# ============================================================
# TRAINING
# ============================================================
RETRAIN_INTERVAL = 200
TRAIN_BARS_MULTIPLIER = 6
MIN_TRAINING_SAMPLES = 100

# ============================================================
# PER-SYMBOL SETTINGS
# ============================================================
SYMBOL_SETTINGS = {
    'EURUSDm': {
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 3.0,
        'risk_pct': 0.7,
        'min_confidence': 0.50,
        'min_adx': 24,
        'min_ict_score': 1,
        'ml_buy_threshold': 0.54,
        'ml_sell_threshold': 0.46,
        'pullback_rsi_buy_max': 60,
        'pullback_rsi_sell_min': 40,
        'session_filter': False,
        'use_m30': True,
        'm30_confirmation': 'both',
        'm30_conf_boost': 0.12,
        'm30_conf_penalty': 0.10,
        'require_htf': False,
        'require_pullback': True,
        'max_per_symbol': 1,
        'max_lot': 2.0,
    },
    # RISK REDUCED: underperforming / insufficient sample
    'GBPUSDm': {
        'sl_atr_mult': 2.0,
        'tp_atr_mult': 3.0,
        'risk_pct': 0.3,
        'min_confidence': 0.52,
        'min_adx': 24,
        'min_ict_score': 1,
        'ml_buy_threshold': 0.54,
        'ml_sell_threshold': 0.46,
        'pullback_rsi_buy_max': 60,
        'pullback_rsi_sell_min': 40,
        'session_filter': False,
        'use_m30': True,
        'm30_confirmation': 'both',
        'm30_conf_boost': 0.12,
        'm30_conf_penalty': 0.10,
        'require_htf': False,
        'require_pullback': True,
        'max_per_symbol': 1,
        'max_lot': 0.5,
    },
    'USDJPYm': {
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 3.0,
        'risk_pct': 0.7,
        'min_confidence': 0.50,
        'min_adx': 24,
        'min_ict_score': 1,
        'ml_buy_threshold': 0.54,
        'ml_sell_threshold': 0.46,
        'pullback_rsi_buy_max': 60,
        'pullback_rsi_sell_min': 40,
        'session_filter': False,
        'use_m30': True,
        'm30_confirmation': 'both',
        'm30_conf_boost': 0.12,
        'm30_conf_penalty': 0.10,
        'require_htf': False,
        'require_pullback': True,
        'max_per_symbol': 1,
        'max_lot': 2.0,
    },
    # RISK REDUCED: underperforming / insufficient sample
    'BTCUSDm': {
        'sl_atr_mult': 1.8,
        'tp_atr_mult': 3.0,
        'risk_pct': 0.3,
        'min_confidence': 0.50,
        'min_adx': 25,
        'min_ict_score': 2,
        'ml_buy_threshold': 0.54,
        'ml_sell_threshold': 0.46,
        'pullback_rsi_buy_max': 60,
        'pullback_rsi_sell_min': 40,
        'session_filter': False,
        'use_m30': True,
        'm30_confirmation': 'both',
        'm30_conf_boost': 0.12,
        'm30_conf_penalty': 0.10,
        'require_htf': False,
        'require_pullback': True,
        'max_per_symbol': 1,
        'max_lot': 0.2,
    },
    'XAUUSDm': {
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 3.0,
        'risk_pct': 0.8,
        'min_confidence': 0.50,
        'min_adx': 24,
        'min_ict_score': 2,
        'ml_buy_threshold': 0.53,
        'ml_sell_threshold': 0.47,
        'pullback_rsi_buy_max': 60,
        'pullback_rsi_sell_min': 40,
        'session_filter': False,
        'use_m30': True,
        'm30_confirmation': 'both',
        'm30_conf_boost': 0.12,
        'm30_conf_penalty': 0.10,
        'require_htf': False,
        'require_pullback': True,
        'max_per_symbol': 1,
        'max_lot': 2.0,
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_symbol_setting(symbol, key, default=None):
    return SYMBOL_SETTINGS.get(symbol, {}).get(key, default)


def get_symbol_config(symbol):
    defaults = {
        'sl_atr_mult': ATR_SL_MULTIPLIER,
        'tp_atr_mult': ATR_TP_MULTIPLIER,
        'risk_pct': ACCOUNT_RISK_PERCENT,
        'min_confidence': MIN_CONFIDENCE,
        'min_adx': MIN_ADX,
        'min_ict_score': MIN_ICT_SCORE,
        'ml_buy_threshold': ML_THRESHOLD_BUY,
        'ml_sell_threshold': ML_THRESHOLD_SELL,
        'pullback_rsi_buy_max': PULLBACK_RSI_BUY_MAX,
        'pullback_rsi_sell_min': PULLBACK_RSI_SELL_MIN,
        'require_htf': REQUIRE_HTF_ALIGNMENT,
        'require_pullback': REQUIRE_PULLBACK,
        'session_filter': SESSION_FILTER_ENABLED,
        'use_m30': USE_M30,
        'm30_confirmation': M30_CONFIRMATION,
        'm30_conf_boost': M30_CONF_BOOST,
        'm30_conf_penalty': M30_CONF_PENALTY,
        'max_per_symbol': MAX_TRADES_PER_SYMBOL,
        'max_lot': 5.0,
    }
    sym_cfg = SYMBOL_SETTINGS.get(symbol, {})
    return {**defaults, **sym_cfg}