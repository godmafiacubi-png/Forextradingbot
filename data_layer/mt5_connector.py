import logging
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MT5Connector:
    """Small MetaTrader 5 adapter used by data, risk, and execution modules."""

    def __init__(self, login, password, server, path):
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        self.connect()

    def connect(self):
        """Connect to MT5 and fail closed when explicit credentials are rejected."""
        try:
            initialized = False
            if self.path:
                initialized = mt5.initialize(path=self.path)
                if not initialized:
                    logger.warning("MT5 initialize with configured path failed, trying auto-discovery")
            if not initialized and not mt5.initialize():
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False

            if self.login > 0 and self.password:
                if not mt5.login(self.login, self.password, self.server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    self.connected = False
                    return False

            self.connected = True
            logger.info("✅ MT5 connected successfully")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return False

    def get_ohlcv(self, symbol, timeframe, bars=100):
        """Fetch OHLCV data."""
        try:
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Symbol {symbol} not available")
                return None

            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if isinstance(rates, np.ndarray) and len(rates) > 0:
                return self._rates_to_df(rates, symbol)

            logger.debug(f"Fallback: Trying copy_rates_range for {symbol}")
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
            if isinstance(rates, np.ndarray) and len(rates) > 0:
                return self._rates_to_df(rates[-bars:], symbol)

            logger.warning(f"⚠️ No data for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def _rates_to_df(rates, symbol):
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'open': 'o',
            'high': 'h',
            'low': 'l',
            'close': 'c',
            'tick_volume': 'v',
        })
        logger.debug(f"✅ Got {len(df)} candles for {symbol}")
        return df[['time', 'o', 'h', 'l', 'c', 'v']].copy()

    def get_account_info(self):
        """Get account info with both MT5 and dashboard-compatible margin keys."""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning("Account info not available - using safe defaults")
                return {
                    'balance': 10000,
                    'equity': 10000,
                    'margin': 0,
                    'margin_free': 10000,
                    'free_margin': 10000,
                    'margin_level': 0,
                    'profit': 0,
                }
            margin_free = account_info.margin_free
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': margin_free,
                'free_margin': margin_free,
                'margin_level': account_info.margin_level,
                'profit': account_info.profit,
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    def get_symbol_info(self, symbol):
        """Get normalized symbol info needed by sizing and execution modules."""
        try:
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Symbol {symbol} not available")
                return None

            info = mt5.symbol_info(symbol)
            if info is None:
                logger.warning(f"Symbol info not available for {symbol}")
                return None

            point = info.point or self._fallback_point(symbol)
            bid = info.bid
            ask = info.ask
            return {
                'point': point,
                'digits': info.digits,
                'bid': bid,
                'ask': ask,
                'spread': (ask - bid) / point if point > 0 else 0,
                'trade_tick_value': getattr(info, 'trade_tick_value', 0),
                'trade_tick_size': getattr(info, 'trade_tick_size', point),
                'trade_contract_size': getattr(info, 'trade_contract_size', 0),
                'volume_min': getattr(info, 'volume_min', 0.01),
                'volume_max': getattr(info, 'volume_max', 100.0),
                'volume_step': getattr(info, 'volume_step', 0.01),
                'trade_stops_level': getattr(info, 'trade_stops_level', 0),
            }
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    @staticmethod
    def _fallback_point(symbol):
        symbol_upper = symbol.upper()
        if 'JPY' in symbol_upper or 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
            return 0.001
        if 'BTC' in symbol_upper:
            return 0.01
        return 0.00001

    def disconnect(self):
        """Disconnect from MT5."""
        try:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
