"""
TEJAS Phase 8 — Confirmation Gate (Fix 2)
==========================================
A stateful confirmation gate that requires K consecutive threshold
exceedances before firing a CONFIRMED alert.  Suppresses single-step
spike-driven false alarms while preserving lead time for real pre-flare
events.

State machine
-------------
    IDLE      -> PENDING (on first exceedance)
    PENDING   -> PENDING (streak++)
              -> CONFIRMED (streak reaches K, alert_fired=True, enter COOLDOWN)
              -> IDLE     (streak broken by non-exceedance)
    COOLDOWN  -> COOLDOWN (tick down cooldown_s)
              -> IDLE     (cooldown reaches 0)

No-future-data guarantee: the gate consumes only past probability values
and an evolving internal counter.  There is no look-ahead of any kind.
"""

from dataclasses import dataclass, field

from tejas.config import (
    ALERT_CADENCE_S,
    ALERT_CONFIRMATION_K,
    ALERT_COOLDOWN_S,
)


IDLE = "idle"
PENDING = "pending"
CONFIRMED = "confirmed"
COOLDOWN = "cooldown"


@dataclass
class ConfirmationGate:
    """Stateful per-horizon confirmation gate.

    Attributes:
        horizon_min: The forecasting horizon this gate watches.
        decision_threshold: Probability threshold from the model
            (typically ``Forecaster.decision_thresholds()[horizon_min]``).
        k_consecutive: Number of consecutive exceedances required to
            confirm.  Default 3 (30 seconds at 10 s cadence).
        cooldown_s: Post-alert suppression window in seconds.  Default 300.
        cadence_s: Prediction cadence in seconds (used to tick the
            cooldown counter).  Default 10.
    """

    horizon_min: int
    decision_threshold: float
    k_consecutive: int = ALERT_CONFIRMATION_K
    cooldown_s: int = ALERT_COOLDOWN_S
    cadence_s: int = ALERT_CADENCE_S

    _state: str = field(default=IDLE, init=False, repr=False)
    _streak: int = field(default=0, init=False, repr=False)
    _cooldown_remaining: int = field(default=0, init=False, repr=False)

    def step(self, prob, timestamp=None):
        """Process one prediction step.

        Args:
            prob: Calibrated probability at the current step.
            timestamp: Optional pd.Timestamp for the current step
                (echoed back in the result).

        Returns:
            Dict with keys ``state``, ``alert_fired``, ``streak``,
            ``horizon_min``, ``timestamp``.
        """
        above = float(prob) >= float(self.decision_threshold)

        # Tick down cooldown first
        if self._state == COOLDOWN:
            self._cooldown_remaining -= self.cadence_s
            if self._cooldown_remaining <= 0:
                self._state = IDLE
                self._cooldown_remaining = 0

        alert_fired = False

        if self._state in (IDLE, PENDING):
            if above:
                self._streak += 1
                if self._streak >= self.k_consecutive:
                    self._state = CONFIRMED
                    alert_fired = True
                    self._streak = 0
                    self._cooldown_remaining = self.cooldown_s
                else:
                    self._state = PENDING
            else:
                self._streak = 0
                self._state = IDLE

        # In CONFIRMED / COOLDOWN, exceedances are ignored until cooldown ends.
        # Move from CONFIRMED to COOLDOWN so subsequent calls tick down.
        if self._state == CONFIRMED and not alert_fired:
            self._state = COOLDOWN

        # Alert was just fired this step → next step will transition
        # CONFIRMED → COOLDOWN.  Do not toggle now so the caller sees
        # state="confirmed" this step.
        if alert_fired:
            # Immediately enter COOLDOWN for subsequent steps.
            self._state = CONFIRMED

        return {
            "state": self._state,
            "alert_fired": alert_fired,
            "streak": self._streak,
            "horizon_min": self.horizon_min,
            "timestamp": timestamp,
        }

    def reset(self):
        """Reset gate state.  Use when jumping to a new simulation segment."""
        self._state = IDLE
        self._streak = 0
        self._cooldown_remaining = 0


class MultiHorizonGate:
    """Bundle four ConfirmationGates (one per horizon)."""

    def __init__(self, thresholds,
                 k_consecutive=None, cooldown_s=None, cadence_s=None):
        """
        Args:
            thresholds: Dict {horizon_min: decision_threshold}
                (from Forecaster.decision_thresholds()).
            k_consecutive: Override for K.  Falls back to config.
            cooldown_s:    Override for cooldown_s.  Falls back to config.
            cadence_s:     Override for cadence_s.  Falls back to config.
        """
        k = k_consecutive if k_consecutive is not None else ALERT_CONFIRMATION_K
        cd = cooldown_s if cooldown_s is not None else ALERT_COOLDOWN_S
        cad = cadence_s if cadence_s is not None else ALERT_CADENCE_S
        self.gates = {
            int(h): ConfirmationGate(
                horizon_min=int(h),
                decision_threshold=float(t),
                k_consecutive=k,
                cooldown_s=cd,
                cadence_s=cad,
            )
            for h, t in thresholds.items()
        }

    def step(self, probs, timestamp=None):
        """Process one prediction step across all horizons.

        Args:
            probs: Dict {"prob_5m": ..., "prob_15m": ..., ...}
                (from Forecaster.predict_row()).
            timestamp: Optional current timestamp.

        Returns:
            List of result dicts (one per horizon).
        """
        results = []
        for h, gate in self.gates.items():
            p = probs.get(f"prob_{h}m", 0.0)
            results.append(gate.step(p, timestamp=timestamp))
        return results

    def reset_all(self):
        for gate in self.gates.values():
            gate.reset()


def simulate_gate_on_frame(prob_frame, thresholds, horizon_min,
                            k_consecutive=None, cooldown_s=None,
                            cadence_s=None):
    """Retrospective simulation of a ConfirmationGate over a probability
    time series.

    Used for offline benchmarking: given a series of calibrated
    probabilities from a Forecaster, replay them row-by-row through the
    gate and return the mask of rows where a CONFIRMED alert fired.

    Args:
        prob_frame: DataFrame with a column ``prob_{h}m`` for
            ``h == horizon_min``.
        thresholds: Dict {horizon_min: decision_threshold}.
        horizon_min: Which horizon to simulate.
        k_consecutive: Override for K (else config).
        cooldown_s:    Override for cooldown_s (else config).
        cadence_s:     Override for cadence_s (else config).

    Returns:
        Numpy boolean array of length len(prob_frame): True where a
        CONFIRMED alert fired at that step.
    """
    import numpy as np

    col = f"prob_{horizon_min}m"
    probs = prob_frame[col].to_numpy(dtype=float)
    gate = ConfirmationGate(
        horizon_min=int(horizon_min),
        decision_threshold=float(thresholds[horizon_min]),
        k_consecutive=(k_consecutive if k_consecutive is not None
                        else ALERT_CONFIRMATION_K),
        cooldown_s=(cooldown_s if cooldown_s is not None
                     else ALERT_COOLDOWN_S),
        cadence_s=(cadence_s if cadence_s is not None
                    else ALERT_CADENCE_S),
    )
    fired = np.zeros(len(probs), dtype=bool)
    for i, p in enumerate(probs):
        r = gate.step(p)
        fired[i] = r["alert_fired"]
    return fired
