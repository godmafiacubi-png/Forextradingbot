#!/usr/bin/env python
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

import warnings
warnings.warn(
    "run_live.py is a legacy entry point and lacks RL, Meta-Learner, "
    "Temporal Encoder, and Regime-Aware features. "
    "Use 'python main.py' for the full ML stack.",
    DeprecationWarning,
    stacklevel=1,
)

# Import after logging setup
from config.settings import *
from data_layer.mt5_connector import MT5Connector
from feature_engineering.ict_features import ICTFeatures
from feature_engineering.ml_features import MLFeatures
from ml_models.ensemble import EnsembleModel
from ml_models.model_manager import ModelManager
from strategy.signal_generator import SignalGenerator
from risk_management.position_sizer import PositionSizer
from execution.order_manager import OrderManager
from monitoring.simple_dashboard import SimpleMonitor
from monitoring.performance_tracker import PerformanceTracker
from monitoring.telegram_alerts import TelegramAlerts
import time
from datetime import datetime
import MetaTrader5 as mt5

class LiveTradingBot:
    def __init__(self):
        logger.warning(
            "⚠️  run_live.py is LEGACY — missing RL, Meta-Learner, Temporal & Regime features. "
            "Run 'python main.py' for the full ML stack."
        )
        logger.info("="*80)
        logger.info("🚀 TRADING BOT INITIALIZATION")
        logger.info("="*80)
        
        try:
            # Connect to MT5
            logger.info("Connecting to MT5...")
            self.mt5 = MT5Connector(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH)
            
            if not self.mt5.connected:
                logger.error("❌ MT5 connection failed")
                raise Exception("MT5 not connected")
            
            # Get account info
            account_info = self.mt5.get_account_info()
            logger.info(f"Account Balance: ${account_info['balance']:.2f}")
            logger.info(f"Account Equity: ${account_info['equity']:.2f}")
            
            # Initialize ML Model
            logger.info("Initializing ML Model...")
            self.ml_model = EnsembleModel()
            self.model_manager = ModelManager('./models')
            
            # Try to load existing model
            if self.model_manager.load_model(self.ml_model, 'trading_model'):
                logger.info("✅ Loaded existing model")
            else:
                logger.warning("⚠️ No existing model, will train new one")
            
            # Initialize strategy components
            self.signal_gen = SignalGenerator(self.ml_model)
            self.position_sizer = PositionSizer(method=POSITION_SIZING_METHOD)
            self.order_manager = OrderManager(self.mt5)
            
            # Initialize monitoring
            self.monitor = SimpleMonitor()
            self.tracker = PerformanceTracker()
            self.telegram = TelegramAlerts(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
            
            if self.telegram.enabled:
                logger.info("✅ Telegram alerts enabled")
            else:
                logger.warning("⚠️ Telegram not configured")
            
            self.active_trades = {}
            self.iteration = 0
            
            logger.info("="*80)
            logger.info("✅ BOT READY!")
            logger.info("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def fetch_and_process_data(self, symbol, timeframe):
        """Fetch and process data"""
        try:
            tf_value = TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_H1)
            df = self.mt5.get_ohlcv(symbol, tf_value, bars=LOOKBACK_PERIOD * 2)
            
            if df is None or len(df) == 0:
                return None
            
            # Add ICT features
            ict = ICTFeatures(df)
            df = ict.get_ict_features()
            
            # Add ML features
            ml = MLFeatures(df)
            df = ml.get_ml_features()
            
            return df
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None
    
    def run_strategy(self, symbol, timeframe):
        """Run strategy on symbol"""
        try:
            # Fetch data
            df = self.fetch_and_process_data(symbol, timeframe)
            if df is None or len(df) == 0:
                return
            
            # Generate signals
            df = self.signal_gen.generate_signals(df)
            
            latest_signal = df.iloc[-1]['signal']
            confidence = df.iloc[-1]['confidence']
            price = df.iloc[-1]['c']
            
            # Log signal
            self.monitor.log_signal(symbol, latest_signal, confidence)
            
            # Check confidence threshold
            if confidence < 0.5:
                return
            
            # Print signal
            signal_name = "🟢 BUY" if latest_signal == 1 else "🔴 SELL" if latest_signal == -1 else "⚪ HOLD"
            logger.info(f"{symbol:12} {timeframe:4} {signal_name:15} Conf: {confidence:.2%} Price: {price:.5f}")
            
            # Send Telegram alert
            if latest_signal != 0:
                self.telegram.send_signal_alert(symbol, latest_signal, confidence, price)
        
        except Exception as e:
            logger.error(f"Strategy error for {symbol}: {e}")
            self.telegram.send_error_alert(f"Strategy error {symbol}: {str(e)}")
    
    def live_trading(self):
        """Main live trading loop"""
        logger.info("📊 Starting live trading...")
        logger.info(f"Symbols: {SYMBOLS}")
        logger.info(f"Update interval: 5 minutes\n")
        
        all_symbols = (SYMBOLS['FOREX'] + 
                      SYMBOLS.get('CRYPTO', []) + 
                      SYMBOLS.get('GOLD', []))
        
        timeframe = 'H1'
        
        try:
            while True:
                self.iteration += 1
                
                # Print header
                logger.info("\n" + "="*80)
                logger.info(f"⏰ Iteration {self.iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("="*80)
                
                # Run strategy for each symbol
                for symbol in all_symbols:
                    self.run_strategy(symbol, timeframe)
                
                # Print monitoring status every 10 iterations
                if self.iteration % 10 == 0:
                    self.monitor.print_status()
                    self.tracker.print_stats()
                
                # Save model every 50 iterations
                if self.iteration % 50 == 0:
                    logger.info("💾 Saving model...")
                    self.model_manager.save_model(self.ml_model, 'trading_model')
                
                # Wait for next iteration
                logger.info("💤 Waiting 5 minutes until next update...")
                time.sleep(300)
        
        except KeyboardInterrupt:
            logger.info("\n⛔ Bot stopped by user")
            self._shutdown()
        
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.telegram.send_error_alert(f"Fatal error: {str(e)}")
            self._shutdown()
    
    def _shutdown(self):
        """Shutdown bot"""
        logger.info("🛑 Shutting down...")
        
        # Save model
        logger.info("Saving final model...")
        self.model_manager.save_model(self.ml_model, 'trading_model')
        
        # Print stats
        self.tracker.print_stats()
        
        # Disconnect MT5
        self.mt5.disconnect()
        
        logger.info("✅ Bot shutdown complete")

if __name__ == "__main__":
    try:
        bot = LiveTradingBot()
        bot.live_trading()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)