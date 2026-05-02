import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MT5Connector:
    def __init__(self, login, password, server, path):
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        self.connect()
    
    def connect(self):
        """Connect to MT5"""
        try:
            if not mt5.initialize(path=self.path):
                logger.warning(f"MT5 initialize with path failed, trying without path")
                if not mt5.initialize():
                    logger.error("MT5 initialize failed")
                    return False
            
            # Try login (optional)
            if self.login > 0 and self.password:
                if not mt5.login(self.login, self.password, self.server):
                    logger.warning(f"MT5 login failed: {mt5.last_error()}")
            
            self.connected = True
            logger.info("✅ MT5 connected successfully")
            return True
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def get_ohlcv(self, symbol, timeframe, bars=100):
        """Fetch OHLCV data"""
        try:
            # Ensure symbol exists
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Symbol {symbol} not available")
            
            # Method 1: copy_rates_from_pos
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            
            if isinstance(rates, np.ndarray) and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Rename columns
                df = df.rename(columns={
                    'open': 'o',
                    'high': 'h',
                    'low': 'l',
                    'close': 'c',
                    'tick_volume': 'v'
                })
                
                logger.debug(f"✅ Got {len(df)} candles for {symbol}")
                return df[['time', 'o', 'h', 'l', 'c', 'v']].copy()
            
            # Method 2: copy_rates_range
            logger.debug(f"Fallback: Trying copy_rates_range for {symbol}")
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            
            rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
            
            if isinstance(rates, np.ndarray) and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df.tail(bars)
                
                df = df.rename(columns={
                    'open': 'o',
                    'high': 'h',
                    'low': 'l',
                    'close': 'c',
                    'tick_volume': 'v'
                })
                
                logger.debug(f"✅ Got {len(df)} candles for {symbol} (from range)")
                return df[['time', 'o', 'h', 'l', 'c', 'v']].copy()
            
            logger.warning(f"⚠️ No data for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_account_info(self):
        """Get account info"""
        try:
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning("Account info not available - using defaults")
                return {
                    'balance': 10000,
                    'equity': 10000,
                    'margin': 0,
                    'margin_free': 10000,
                    'margin_level': 0,
                    'profit': 0
                }
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'profit': account_info.profit
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_symbol_info(self, symbol):
        """Get symbol info"""
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.warning(f"Symbol info not available for {symbol}")
                return {
                    'point': 0.0001,
                    'bid': 1.0,
                    'ask': 1.0,
                    'spread': 2.0
                }
            return {
                'point': info.point,
                'bid': info.bid,
                'ask': info.ask,
                'spread': (info.ask - info.bid) / info.point if info.point > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from MT5"""
        try:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")