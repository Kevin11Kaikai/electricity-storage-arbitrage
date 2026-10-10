"""Synthetic check of the exact storage scheduler.

The price path is constructed in this file. It is not the competition series,
not a historical replay, and not evidence for the recorded 7.72% score
difference. That result is in report/shared/data/summary.json.

The printed quantity is the raw score sum(price * coded power). It is not
currency and not a trading return.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase_b.contracts import (  # noqa: E402
    BLOCK_STEPS,
    POWER_MAGNITUDE,
    assert_valid_day_power,
)
from src.phase_b.evaluation import oracle_day, score_day  # noqa: E402
from src.phase_b.dispatch import optimize_day  # noqa: E402


def synthetic_realized_prices() -> np.ndarray:
    """One made-up day: cheap hours 2–4, expensive hours 16–18."""
    prices = np.full(96, 40.0)
    prices[8:16] = 10.0
    prices[64:72] = 80.0
    return prices


def shifted_forecast(realized: np.ndarray) -> np.ndarray:
    """A forecast that moves the high-price block four hours earlier."""
    forecast = realized.copy()
    forecast[64:72] = 40.0
    forecast[48:56] = 80.0
    return forecast


def task_unit_storage(power: np.ndarray) -> np.ndarray:
    """Cumulative stored task units. Charge is coded as negative power."""
    stored = np.cumsum(-power)
    return np.concatenate([[0.0], stored])


def main() -> int:
    print("SYNTHETIC DEMO")
    print("This curve is invented for the scheduler check.")
    print("It does not reproduce the historical 59-day scores.")
    print()

    realized = synthetic_realized_prices()
    forecast = shifted_forecast(realized)
    chosen = optimize_day(forecast)
    perfect = oracle_day(realized)
    assert_valid_day_power(chosen.power)
    assert_valid_day_power(perfect.power)

    stored = task_unit_storage(chosen.power)
    peak = float(stored.max())
    end = float(stored[-1])
    if chosen.idle:
        raise SystemExit("synthetic forecast unexpectedly selected idle")
    if not (0.0 <= stored.min() and peak == BLOCK_STEPS * POWER_MAGNITUDE and end == 0.0):
        raise SystemExit(
            f"task-unit storage path failed: min={stored.min()}, peak={peak}, end={end}"
        )
    if chosen.td is None or chosen.tc is None or chosen.td < chosen.tc + BLOCK_STEPS:
        raise SystemExit("charge-before-discharge check failed")

    forecast_score = score_day(realized, chosen.power)
    oracle_score = score_day(realized, perfect.power)
    print(f"forecast-chosen window: charge slot {chosen.tc}, discharge slot {chosen.td}")
    print(f"oracle window:          charge slot {perfect.tc}, discharge slot {perfect.td}")
    print(f"raw score of the forecast schedule on the realized curve: {forecast_score:.1f}")
    print(f"raw score of the perfect-information schedule:            {oracle_score:.1f}")
    print(f"task-unit storage peak / end: {peak:.0f} / {end:.0f}")
    print()
    print("Constraints checked: 96 slots, eight charge then eight discharge,")
    print("coded power ±1000, charge before discharge, storage in {0…8000},")
    print("and a zero terminal task-unit balance.")
    print("Efficiency, degradation, and cycling costs are not in this model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
