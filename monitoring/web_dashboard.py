"""
Default web dashboard compatibility module.

The main bot imports this module whenever ``BOT_MODE`` is not ``AGGRESSIVE``.
The project currently ships one full dashboard implementation
(``web_dashboard_aggressive``), so this module re-exports the same public API
with a neutral default mode to keep non-aggressive configurations importable.
"""

from monitoring.web_dashboard_aggressive import (  # noqa: F401
    add_log,
    dashboard_state,
    start_dashboard,
    update_dashboard,
)

# Make the fallback dashboard identify itself as the default/non-aggressive UI
# until a dedicated conservative dashboard is added.
dashboard_state['mode'] = 'DEFAULT'
