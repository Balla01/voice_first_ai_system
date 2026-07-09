"""
logging_config.py — shared logging setup.

Call setup_logging() once, at app startup (main.py does this before
anything else). After that, every module just uses
logging.getLogger(<name>).debug/info/warning/error(...) as normal — the
split between "shows on terminal" and "only goes to the log file" is
handled entirely by HANDLER levels, not by which logger you use.

Convention for this codebase:
  logger.info(...)   -> meant to be visible on the terminal. Right now
                        that's conversation lines ([agent]/[customer] ...)
                        and high-level trigger-gate decisions worth
                        narrating live during a demo.
  logger.debug(...)  -> internal step-by-step detail (cooldown state, tier
                        scores, why a gate did or didn't fire, etc.) — only
                        ever goes to the log file, never the terminal.

Currently only layer4/*.py emits debug-level detail. Layer 1/2/3 will get
the same treatment later — this setup already supports it with zero changes
needed here, since it's just based on log LEVEL, not which layer is logging.
"""

import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_FILE = LOG_DIR / "app.log"


def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # let everything through; handlers below decide what's shown/stored
    root.handlers.clear()          # remove any handler a prior logging.basicConfig() call may have added

    # Terminal: clean, message-only, INFO and above — this is what makes the
    # console show "just the conversation" instead of internal gate detail.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console_handler)

    # File: everything, with full context — this is where you go to see
    # exactly why a gate fired, didn't fire, or what a tier scored.
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    root.addHandler(file_handler)

    # Quiet noisy third-party libraries so they don't clutter either output
    # (e.g. httpx logs every LLM API call at INFO by default).
    for noisy_logger in ("httpx", "httpcore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logging.getLogger("insureassist").debug(f"Logging initialized: console=INFO only, file=DEBUG at {LOG_FILE}")