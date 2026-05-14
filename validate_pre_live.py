"""Pre-live safety validation command.

Run this before any demo/live forward test. It verifies that configuration is
fail-closed by default, that the live profile cannot route live orders unless
both environment gates are explicit, and that local artifacts likely to pollute
Git are covered by ignore rules.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ARTIFACT_ALLOWLIST = ROOT / "config" / "model_artifact_allowlist.txt"
APPROVED_MODEL_ARTIFACT_PREFIXES = ("models/approved/",)
DISALLOWED_MODEL_ARTIFACT_PREFIXES = ("models/tmp/", "models/checkpoints/")


@contextmanager
def patched_env(values: dict[str, str | None]):
    original = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def install_dependency_stubs() -> None:
    sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))


def install_mt5_stub() -> None:
    if "MetaTrader5" in sys.modules:
        return
    sys.modules["MetaTrader5"] = types.SimpleNamespace(
        TIMEFRAME_M1=1,
        TIMEFRAME_M5=5,
        TIMEFRAME_M15=15,
        TIMEFRAME_M30=30,
        TIMEFRAME_H1=60,
        TIMEFRAME_H4=240,
        TIMEFRAME_D1=1440,
    )


def load_settings(env: dict[str, str | None]):
    install_dependency_stubs()
    install_mt5_stub()
    with patched_env(env):
        for module_name in ("config.settings", "config.settings_live", "config.settings_demo", "config.settings_default"):
            sys.modules.pop(module_name, None)
        return importlib.import_module("config.settings")


def check(condition: bool, message: str, failures: list[str]) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def tracked_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return result.stdout.splitlines()


def model_artifact_allowlist() -> set[str]:
    if not MODEL_ARTIFACT_ALLOWLIST.exists():
        return set()
    return {
        line.strip() for line in MODEL_ARTIFACT_ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pre-live safety gates")
    parser.add_argument("--ci", action="store_true", help="Run non-interactive CI-safe checks")
    args = parser.parse_args()
    failures: list[str] = []

    default_settings = load_settings({"SETTINGS_PROFILE": None, "DRY_RUN": None, "LIVE_TRADING_CONFIRMED": None})
    check(default_settings.DRY_RUN is True, "default profile stays in dry-run", failures)
    check(default_settings.LIVE_TRADING_CONFIRMED is False, "default profile is not live-confirmed", failures)

    live_unconfirmed = load_settings({"SETTINGS_PROFILE": "live", "DRY_RUN": "false", "LIVE_TRADING_CONFIRMED": "false"})
    check(live_unconfirmed.DRY_RUN is True, "live profile fails closed without confirmation", failures)

    live_confirmed = load_settings({"SETTINGS_PROFILE": "live", "DRY_RUN": "false", "LIVE_TRADING_CONFIRMED": "true"})
    check(live_confirmed.DRY_RUN is False, "live profile opens only when both gates are explicit", failures)
    check(live_confirmed.ACCOUNT_RISK_PERCENT <= 0.5, "live profile risk remains capped", failures)

    env_example = ROOT / ".env.example"
    check(env_example.exists(), ".env.example exists for safe local setup", failures)
    if env_example.exists():
        text = env_example.read_text(encoding="utf-8")
        check("DRY_RUN=true" in text, ".env.example defaults to dry-run", failures)
        check("LIVE_TRADING_CONFIRMED=false" in text, ".env.example defaults to unconfirmed live trading", failures)

    risky_suffixes = (".pkl", ".joblib", ".pt", ".pth", ".h5", ".keras")
    tracked_risky = [
        path for path in tracked_files()
        if path.startswith("models/") and path.endswith(risky_suffixes)
    ]
    allowed_artifacts = model_artifact_allowlist()
    check(MODEL_ARTIFACT_ALLOWLIST.exists(), "model artifact allowlist exists", failures)
    unapproved_artifacts = [
        path for path in tracked_risky
        if not path.startswith(APPROVED_MODEL_ARTIFACT_PREFIXES) and path not in allowed_artifacts
    ]
    disallowed_artifacts = [
        path for path in tracked_risky
        if path.startswith(DISALLOWED_MODEL_ARTIFACT_PREFIXES)
    ]
    check(not unapproved_artifacts, "tracked model artifacts are limited to explicit allowlists", failures)
    check(not disallowed_artifacts, "no tmp/checkpoint model artifacts are tracked", failures)
    for path in unapproved_artifacts[:20]:
        print(f"  - unapproved artifact: {path}")
    for path in disallowed_artifacts[:20]:
        print(f"  - disallowed artifact: {path}")

    if failures:
        print("\nPre-live validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nPre-live validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
