import logging
import asyncio

logger = logging.getLogger(__name__)

try:
    from telegram import Bot
    from telegram.error import TelegramError
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    logger.warning("python-telegram-bot not installed")


class TelegramAlerts:
    """Send alerts via Telegram — handles async event loop properly"""

    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.bot = None
        self.enabled = False

        if HAS_TELEGRAM and token and chat_id:
            try:
                self.bot = Bot(token=token)
                self.enabled = True
            except Exception as e:
                logger.error(f"Telegram bot init error: {e}")

    def _run_async(self, coro):
        """Run async coroutine safely — handles event loop reuse"""
        try:
            # Try to get existing running loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result(timeout=10)
            elif loop.is_closed():
                # If loop is closed, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # Fallback: create new event loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Telegram async error: {e}")
                return False

    async def _send_async(self, message):
        """Send message async"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            logger.info("Telegram message sent")
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    def send_message(self, message):
        """Send message to Telegram (sync wrapper)"""
        if not self.enabled:
            return False

        try:
            return self._run_async(self._send_async(message))
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

    def send_signal_alert(self, symbol, signal, confidence, price):
        """Send signal alert"""
        if not self.enabled:
            return

        signal_name = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
        emoji = "🟢" if signal == 1 else "🔴" if signal == -1 else "⚪"

        message = (
            f"<b>{emoji} Signal Alert</b>\n\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Signal:</b> {signal_name}\n"
            f"<b>Confidence:</b> {confidence:.2%}\n"
            f"<b>Price:</b> {price:.5f}"
        )

        self.send_message(message)

    def send_trade_alert(self, symbol, side, entry_price, size, sl, tp):
        """Send trade execution alert"""
        if not self.enabled:
            return

        emoji = "📈" if side == 'BUY' else "📉"

        message = (
            f"<b>{emoji} Trade Executed</b>\n\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Side:</b> {side}\n"
            f"<b>Entry:</b> {entry_price:.5f}\n"
            f"<b>Size:</b> {size} lots\n"
            f"<b>SL:</b> {sl:.5f}\n"
            f"<b>TP:</b> {tp:.5f}"
        )

        self.send_message(message)

    def send_error_alert(self, error_message):
        """Send error alert"""
        if not self.enabled:
            return

        message = (
            f"<b>⚠️ Bot Error</b>\n\n"
            f"<code>{error_message}</code>"
        )

        self.send_message(message)