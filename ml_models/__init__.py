"""ML model package initialisation.

Patch the legacy DeepRL reward calculator at package import time so both
``deep_rl_agent`` and ``deep_rl_agent_v22`` use the same sign-consistent reward
logic without changing their public APIs.
"""

try:
    from . import deep_rl_agent as _deep_rl_agent
    from .reward_calculator import SignConsistentShapedRewardCalculator

    _deep_rl_agent.ShapedRewardCalculator = SignConsistentShapedRewardCalculator
except Exception:
    # Import-time patching should never prevent the bot from starting.  If the
    # legacy module cannot be imported here, its own fallback paths still apply.
    pass
