import asyncio
import logging
import os
from datetime import datetime
from typing import Optional
from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Handles sending trading signals to Telegram group in real-time."""
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None
    ):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token (defaults to TELEGRAM_BOT_TOKEN env var)
            chat_id: Telegram chat/group ID (defaults to TELEGRAM_CHAT_ID env var)
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)
        self.last_sent_signals = {}
        
        if self.enabled:
            self.bot = Bot(token=self.bot_token)
            logger.info(f"Telegram notifier initialized for chat {self.chat_id}")
        else:
            logger.warning(
                "Telegram notifier disabled: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars"
            )
    
    def _format_timestamp(self, timestamp_value) -> str:
        if timestamp_value is None:
            return "N/A"

        try:
            timestamp = int(timestamp_value)
            if timestamp > 1e12:
                timestamp = timestamp / 1000.0
            return datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return str(timestamp_value)

    def _confidence_label(self, confidence: Optional[float]) -> str:
        if confidence is None:
            return "N/A"
        if confidence > 10.0:
            return "High"
        if confidence > 3.0:
            return "Medium"
        return "Low"

    def _format_signal_message(self, signal_data: dict) -> str:
        """
        Format signal data into a readable Telegram message.
        
        Args:
            signal_data: Signal result from pipeline
            
        Returns:
            Formatted message string
        """
        symbol = signal_data.get("symbol", "UNKNOWN")
        signal = signal_data.get("signal", "UNKNOWN")
        prediction = signal_data.get("prediction", 0)
        prediction_pct = signal_data.get("prediction_pct")
        regime = signal_data.get("regime", 0)
        current_price = signal_data.get("current_price")
        volatility_pct = signal_data.get("volatility_pct")
        ma_fast = signal_data.get("ma_fast")
        ma_slow = signal_data.get("ma_slow")
        regime_mean_pct = signal_data.get("regime_mean_pct")
        trend_bias = signal_data.get("trend_bias", "N/A")
        confidence_pct = signal_data.get("confidence_pct")
        timestamp = signal_data.get("timestamp")
        directional_accuracy = signal_data.get("directional_accuracy")
        cv_directional_accuracy = signal_data.get("cv_directional_accuracy")

        signal_emoji = "🚀" if signal == "BUY" else "🔴" if signal == "SELL" else "⏸️"
        regime_label = "Uptrend" if regime == 1 else "Downtrend" if regime == 0 else f"State {regime}"
        confidence_label = self._confidence_label(confidence_pct or 0)
        timestamp_text = self._format_timestamp(timestamp)
        prediction_text = (
            f"{prediction_pct:+.2f}%" if prediction_pct is not None else f"{prediction:+.6f}"
        )
        confidence_text = (
            f"{confidence_pct:.2f}%" if confidence_pct is not None else "N/A"
        )
        regime_momentum_text = (
            f"{regime_mean_pct:+.2f}%" if regime_mean_pct is not None else "N/A"
        )
        price_text = f"${current_price:,.2f}" if current_price is not None else "N/A"
        volatility_text = (
            f"{volatility_pct:.2f}%" if volatility_pct is not None else "N/A"
        )
        ma_fast_text = f"{ma_fast:,.2f}" if ma_fast is not None else "N/A"
        ma_slow_text = f"{ma_slow:,.2f}" if ma_slow is not None else "N/A"
        accuracy_text = (
            f"{directional_accuracy:.2f}%" if directional_accuracy is not None else "N/A"
        )
        cv_accuracy_text = (
            f"{cv_directional_accuracy:.2f}%" if cv_directional_accuracy is not None else None
        )

        message = (
            f"{signal_emoji} <b>FlowAlpha Signals — {symbol}</b>\n\n"
            f"<b>Action:</b> {signal}\n"
            f"<b>Estimated Move:</b> {prediction_text}\n"
            f"<b>Signal Confidence:</b> {confidence_label} ({confidence_text})\n"
            f"<b>Market Regime:</b> {regime_label}\n"
            f"<b>Regime Strength:</b> {regime_momentum_text}\n"
            f"<b>Trend Bias:</b> {trend_bias}\n"
            f"<b>Volatility:</b> {volatility_text}\n"
            f"<b>MA 5 / 20:</b> {ma_fast_text} / {ma_slow_text}\n"
            f"<b>Model Accuracy:</b> {accuracy_text}\n"
        )
        if cv_accuracy_text is not None:
            message += f"<b>CV Accuracy:</b> {cv_accuracy_text}\n"
        message += (
            f"<b>Price:</b> {price_text}\n"
            f"<b>Timestamp:</b> {timestamp_text}\n\n"
            f"💡 <i>FlowAlpha Signals</i>"
        )
        return message

    def _should_send(self, signal_data: dict) -> bool:
        symbol = signal_data.get("symbol")
        signal = signal_data.get("signal")

        if not symbol or signal is None:
            return True

        last = self.last_sent_signals.get(symbol)
        if last is None:
            return True

        if last.get("signal") == signal:
            logger.info(f"Duplicate signal for {symbol} suppressed: {signal}")
            return False

        return True

    def _record_sent(self, signal_data: dict) -> None:
        symbol = signal_data.get("symbol")
        if not symbol:
            return
        self.last_sent_signals[symbol] = {
            "signal": signal_data.get("signal")
        }

    async def send_signal_async(self, signal_data: dict) -> bool:
        """
        Send signal to Telegram group asynchronously.
        
        Args:
            signal_data: Signal result from pipeline
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        if not self._should_send(signal_data):
            return False

        try:
            message = self._format_signal_message(signal_data)
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="HTML"
            )
            self._record_sent(signal_data)
            logger.info(f"Signal sent to Telegram for {signal_data.get('symbol')}")
            return True
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False
    
    def send_signal(self, signal_data: dict) -> bool:
        """
        Send signal to Telegram group synchronously.
        
        Args:
            signal_data: Signal result from pipeline
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Create a new event loop if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # If loop is already running, schedule as task
                asyncio.create_task(self.send_signal_async(signal_data))
                return True
            else:
                # Run the async function in the current loop
                return loop.run_until_complete(self.send_signal_async(signal_data))
        except Exception as e:
            logger.error(f"Error in send_signal: {e}")
            return False
    
    async def send_alert(self, title: str, message: str) -> bool:
        """
        Send a general alert to Telegram group.
        
        Args:
            title: Alert title
            message: Alert message
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            formatted_message = f"<b>{title}</b>\n\n{message}"
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_message,
                parse_mode="HTML"
            )
            logger.info(f"Alert sent to Telegram: {title}")
            return True
        except TelegramError as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram alert: {e}")
            return False


# Global notifier instance
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create the global Telegram notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


def send_trading_signal(signal_data: dict) -> bool:
    """
    Convenience function to send a trading signal to Telegram.
    
    Args:
        signal_data: Signal result from pipeline
        
    Returns:
        True if sent successfully, False otherwise
    """
    return get_notifier().send_signal(signal_data)
