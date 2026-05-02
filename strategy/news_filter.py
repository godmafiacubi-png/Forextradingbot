import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    logger.warning("feedparser not installed — using manual news schedule")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ============================================================
# STATIC HIGH-IMPACT NEWS SCHEDULE (weekly recurring)
# ============================================================
RECURRING_HIGH_IMPACT = [
    # (day_of_week, hour_utc, minute, currency, event_name)
    # Monday
    # Tuesday
    # Wednesday
    (2, 18, 0, 'USD', 'FOMC Minutes / Rate Decision'),
    # Thursday
    (3, 12, 30, 'USD', 'Unemployment Claims'),
    (3, 12, 30, 'USD', 'GDP'),
    # Friday
    (4, 12, 30, 'USD', 'NFP / Non-Farm Payrolls'),
    (4, 12, 30, 'USD', 'Unemployment Rate'),
]

# Monthly events (approximate — real dates vary)
MONTHLY_HIGH_IMPACT = {
    'CPI': {'currency': 'USD', 'typical_day_range': (10, 15), 'hour': 12, 'minute': 30},
    'PPI': {'currency': 'USD', 'typical_day_range': (11, 16), 'hour': 12, 'minute': 30},
    'Retail Sales': {'currency': 'USD', 'typical_day_range': (14, 18), 'hour': 12, 'minute': 30},
    'ECB Rate': {'currency': 'EUR', 'typical_day_range': (10, 15), 'hour': 12, 'minute': 45},
    'BOE Rate': {'currency': 'GBP', 'typical_day_range': (1, 7), 'hour': 11, 'minute': 0},
    'BOJ Rate': {'currency': 'JPY', 'typical_day_range': (18, 22), 'hour': 3, 'minute': 0},
}

# Symbol to currency mapping
SYMBOL_CURRENCIES = {
    'EURUSDm': ['EUR', 'USD'],
    'GBPUSDm': ['GBP', 'USD'],
    'USDJPYm': ['USD', 'JPY'],
    'BTCUSDm': ['USD'],
    'XAUUSDm': ['USD'],
}


class NewsEvent:
    """Single news event"""

    def __init__(self, event_time, currency, title, impact='HIGH'):
        self.event_time = event_time  # datetime (UTC)
        self.currency = currency
        self.title = title
        self.impact = impact  # HIGH, MEDIUM, LOW

    def __repr__(self):
        return f"[{self.impact}] {self.event_time.strftime('%Y-%m-%d %H:%M')} {self.currency} {self.title}"

    def to_dict(self):
        return {
            'time': self.event_time.strftime('%Y-%m-%d %H:%M'),
            'currency': self.currency,
            'title': self.title,
            'impact': self.impact,
        }


class NewsFilter:
    """
    News filter — blocks/adjusts trading around high-impact events.

    Blackout zones:
    - HIGH impact: 30 min before → 30 min after
    - MEDIUM impact: 15 min before → 15 min after

    Position protection:
    - Tighten SL before news
    - Lock profit if possible
    - Force close if deeply underwater
    """

    def __init__(self, blackout_before_min=30, blackout_after_min=30, cache_dir='./cache'):
        self.blackout_before_high = timedelta(minutes=blackout_before_min)
        self.blackout_after_high = timedelta(minutes=blackout_after_min)
        self.blackout_before_med = timedelta(minutes=15)
        self.blackout_after_med = timedelta(minutes=15)

        self.events = []  # list of NewsEvent
        self.last_fetch = None
        self.fetch_interval = 3600  # refetch every 1 hour

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, 'news_events.json')

        # Load cached events
        self._load_cache()

        # Generate recurring schedule
        self._generate_recurring_events()

        # Try fetching from online
        self._fetch_events_background()

        logger.info(f"[NEWS] Initialized with {len(self.events)} events")

    def _generate_recurring_events(self):
        """Generate this week's recurring events"""
        now = datetime.now(timezone.utc)
        start_of_week = now - timedelta(days=now.weekday())
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)

        for dow, hour, minute, currency, name in RECURRING_HIGH_IMPACT:
            event_time = start_of_week + timedelta(days=dow, hours=hour, minutes=minute)
            # Add for this week and next week
            for week_offset in [0, 7]:
                t = event_time + timedelta(days=week_offset)
                if t > now - timedelta(hours=2):  # don't add past events
                    self._add_event_unique(NewsEvent(t, currency, name, 'HIGH'))

    def _add_event_unique(self, event):
        """Add event if not duplicate"""
        for e in self.events:
            if (abs((e.event_time - event.event_time).total_seconds()) < 60
                    and e.currency == event.currency):
                return
        self.events.append(event)

    def _fetch_events_background(self):
        """Fetch news from Forex Factory RSS in background"""
        def fetch():
            try:
                self._fetch_forex_factory()
            except Exception as e:
                logger.debug(f"[NEWS] Fetch error: {e}")
            try:
                self._fetch_forexapi()
            except Exception as e:
                logger.debug(f"[NEWS] ForexAPI error: {e}")
            self._save_cache()

        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()

    def _fetch_forex_factory(self):
        """Parse Forex Factory RSS feed"""
        if not HAS_FEEDPARSER:
            return

        try:
            feed = feedparser.parse('https://www.forexfactory.com/rss')
            if not feed.entries:
                return

            for entry in feed.entries[:50]:
                title = entry.get('title', '')
                # Try to extract currency and impact from title
                currency = 'USD'
                impact = 'MEDIUM'

                for cur in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD']:
                    if cur in title.upper():
                        currency = cur
                        break

                high_keywords = ['NFP', 'CPI', 'FOMC', 'rate decision', 'GDP', 'employment',
                                 'non-farm', 'inflation', 'retail sales', 'PMI']
                for kw in high_keywords:
                    if kw.lower() in title.lower():
                        impact = 'HIGH'
                        break

                # Parse time
                published = entry.get('published_parsed', None)
                if published:
                    event_time = datetime(*published[:6], tzinfo=timezone.utc)
                    self._add_event_unique(NewsEvent(event_time, currency, title, impact))

            logger.info(f"[NEWS] Fetched {len(feed.entries)} from Forex Factory")
        except Exception as e:
            logger.debug(f"[NEWS] FF RSS error: {e}")

    def _fetch_forexapi(self):
        """Fetch from free ForexAPI/Nager"""
        if not HAS_REQUESTS:
            return

        try:
            # Try nagerapi for economic calendar
            url = 'https://nager.at/api/v3/NextPublicHolidays/US'
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                holidays = resp.json()
                for h in holidays[:5]:
                    event_time = datetime.strptime(h['date'], '%Y-%m-%d').replace(
                        hour=12, minute=0, tzinfo=timezone.utc
                    )
                    self._add_event_unique(NewsEvent(
                        event_time, 'USD', f"US Holiday: {h['name']}", 'MEDIUM'
                    ))
        except Exception:
            pass

    def _save_cache(self):
        try:
            data = [e.to_dict() for e in self.events]
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_cache(self):
        try:
            if not os.path.exists(self.cache_file):
                return
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            for item in data:
                event_time = datetime.strptime(item['time'], '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
                if event_time > datetime.now(timezone.utc) - timedelta(hours=2):
                    self._add_event_unique(NewsEvent(
                        event_time, item['currency'], item['title'], item.get('impact', 'HIGH')
                    ))
        except Exception:
            pass

    def refresh_if_needed(self):
        """Periodically refresh news"""
        now = time.time()
        if self.last_fetch is None or (now - self.last_fetch) > self.fetch_interval:
            self.last_fetch = now
            self._fetch_events_background()
            self._generate_recurring_events()
            # Clean old events
            cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
            self.events = [e for e in self.events if e.event_time > cutoff]

    def get_upcoming_events(self, symbol=None, hours_ahead=24):
        """Get upcoming events (optionally filtered by symbol's currencies)"""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        upcoming = []
        for e in self.events:
            if now - timedelta(hours=1) <= e.event_time <= cutoff:
                if symbol:
                    currencies = SYMBOL_CURRENCIES.get(symbol, [])
                    if e.currency not in currencies:
                        continue
                upcoming.append(e)

        upcoming.sort(key=lambda x: x.event_time)
        return upcoming

    def is_blackout(self, symbol=None):
        """
        Check if we're in a news blackout zone.
        Returns: (is_blackout, reason, minutes_to_event, event)
        """
        now = datetime.now(timezone.utc)

        relevant_events = self.get_upcoming_events(symbol, hours_ahead=2)

        for event in relevant_events:
            time_diff = (event.event_time - now).total_seconds() / 60  # minutes

            if event.impact == 'HIGH':
                before = self.blackout_before_high.total_seconds() / 60
                after = self.blackout_after_high.total_seconds() / 60
            else:
                before = self.blackout_before_med.total_seconds() / 60
                after = self.blackout_after_med.total_seconds() / 60

            # Before event
            if 0 < time_diff <= before:
                return True, f"NEWS in {time_diff:.0f}min: {event.title}", time_diff, event

            # During/after event
            if -after <= time_diff <= 0:
                return True, f"NEWS active ({-time_diff:.0f}min ago): {event.title}", time_diff, event

        return False, "", 0, None

    def get_position_action(self, symbol, current_profit, atr_value, entry_price):
        """
        Decide what to do with existing position when news is coming.

        Returns: (action, details)
        Actions:
            'NORMAL'     — no news nearby, trade normally
            'TIGHTEN_SL' — tighten SL by 50%
            'BREAKEVEN'  — move SL to breakeven
            'LOCK_PROFIT' — move SL to lock current profit
            'FORCE_CLOSE' — close position immediately
        """
        blackout, reason, minutes_to, event = self.is_blackout(symbol)

        if not blackout:
            return 'NORMAL', ''

        profit_in_atr = current_profit / (atr_value + 1e-10) if atr_value > 0 else 0

        # === DURING NEWS (already happening) ===
        if minutes_to <= 0:
            if profit_in_atr < -0.5:
                return 'FORCE_CLOSE', f"News active + losing > 0.5 ATR → close. {reason}"
            elif profit_in_atr > 0.5:
                return 'LOCK_PROFIT', f"News active + profitable → lock profit. {reason}"
            else:
                return 'TIGHTEN_SL', f"News active → tighten SL. {reason}"

        # === BEFORE NEWS (approaching) ===
        if minutes_to <= 5:
            # Very close to news
            if profit_in_atr < -0.3:
                return 'FORCE_CLOSE', f"News in {minutes_to:.0f}min + losing → close. {reason}"
            elif profit_in_atr > 1.0:
                return 'LOCK_PROFIT', f"News in {minutes_to:.0f}min + good profit → lock. {reason}"
            else:
                return 'BREAKEVEN', f"News in {minutes_to:.0f}min → breakeven. {reason}"

        elif minutes_to <= 15:
            if profit_in_atr > 1.0:
                return 'LOCK_PROFIT', f"News in {minutes_to:.0f}min + profitable → lock. {reason}"
            elif profit_in_atr > 0:
                return 'BREAKEVEN', f"News in {minutes_to:.0f}min → breakeven. {reason}"
            else:
                return 'TIGHTEN_SL', f"News in {minutes_to:.0f}min → tighten. {reason}"

        else:
            # 15-30 min before
            return 'TIGHTEN_SL', f"News approaching ({minutes_to:.0f}min). {reason}"

    def get_trade_permission(self, symbol):
        """
        Check if new trades are allowed.
        Returns: (allowed, reason, confidence_multiplier)
        """
        blackout, reason, minutes_to, event = self.is_blackout(symbol)

        if not blackout:
            return True, '', 1.0

        if event and event.impact == 'HIGH':
            return False, f"BLOCKED: {reason}", 0.0
        else:
            # Medium impact — allow with reduced confidence
            return True, f"CAUTION: {reason}", 0.5

    def get_dashboard_data(self):
        """Get news data for web dashboard"""
        now = datetime.now(timezone.utc)
        upcoming = self.get_upcoming_events(hours_ahead=48)

        events_data = []
        for e in upcoming[:20]:
            time_diff = (e.event_time - now).total_seconds() / 60
            status = "ACTIVE" if time_diff <= 0 else ("SOON" if time_diff <= 30 else "UPCOMING")

            events_data.append({
                'time': e.event_time.strftime('%Y-%m-%d %H:%M'),
                'currency': e.currency,
                'title': e.title,
                'impact': e.impact,
                'minutes_to': round(time_diff),
                'status': status,
            })

        # Blackout status per symbol
        blackout_status = {}
        for symbol in SYMBOL_CURRENCIES.keys():
            is_bo, reason, mins, evt = self.is_blackout(symbol)
            blackout_status[symbol] = {
                'blackout': is_bo,
                'reason': reason,
                'minutes_to': round(mins) if mins else 0,
            }

        return {
            'events': events_data,
            'blackout_status': blackout_status,
        }