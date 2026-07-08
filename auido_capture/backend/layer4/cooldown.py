"""
layer4/cooldown.py — minimum 3 seconds between fired triggers, per session.

Exists so the gate doesn't fire on every sentence while a customer is still
adding context in quick succession (see design doc Section 3.5). Pure logic,
no I/O — the caller supplies "now" so tests don't depend on real wall-clock
time.
"""

import logging

logger = logging.getLogger("insureassist.layer4")


class CooldownTracker:
    COOLDOWN_S = 3.0

    def __init__(self):
        self._last_trigger_time: float | None = None

    def is_in_cooldown(self, now: float) -> bool:
        if self._last_trigger_time is None:
            logger.debug("Cooldown: no prior trigger yet, not in cooldown")
            return False
        elapsed = now - self._last_trigger_time
        in_cooldown = elapsed < self.COOLDOWN_S
        if in_cooldown:
            logger.debug(f"Cooldown: ACTIVE, {self.COOLDOWN_S - elapsed:.2f}s remaining")
        else:
            logger.debug(f"Cooldown: cleared ({elapsed:.2f}s since last trigger)")
        return in_cooldown

    def record_trigger(self, now: float) -> None:
        logger.debug(f"Cooldown: recording new trigger time at {now:.3f}")
        self._last_trigger_time = now