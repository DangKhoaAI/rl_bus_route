"""Causal historical demand forecasts. Fit on train arrival logs only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bus_rl.domain import SimConfig


@dataclass(frozen=True)
class Forecast:
    expected: np.ndarray
    version: str = "historical-v1"


def history_from_tape(
    tape: np.ndarray, time_s: int, tick_s: int = 30, interval_s: int = 120
) -> dict[str, np.ndarray]:
    """Past-only 5×2-minute arrival history, scaled like the observation tensor."""
    end_tick = max(0, time_s // tick_s)
    ticks_per_bin = interval_s // tick_s
    history = np.zeros((4, 2, 8, 5), np.float32)
    route_count, directions, stops = tape.shape[1], tape.shape[2], tape.shape[3]
    for lag in range(5):
        start = max(0, end_tick - (lag + 1) * ticks_per_bin)
        stop = max(0, end_tick - lag * ticks_per_bin)
        if stop > start:
            chunk = tape[start:stop].sum(axis=0).sum(axis=-1)
            history[:route_count, :directions, :stops, lag] = chunk / 40.0
    return {"arrival_history": history}


class HistoricalForecaster:
    """Time-bin mean of next-15-minute arrivals with a recent-10-minute correction."""

    def __init__(self, bin_s: int = 900, tick_s: int = 30):
        self.bin_s = bin_s
        self.tick_s = tick_s
        self.next15: np.ndarray | None = None
        self.past10: np.ndarray | None = None

    def fit(self, train_logs: list[np.ndarray], config: SimConfig | None = None) -> None:
        if not train_logs:
            raise ValueError("forecaster requires at least one train arrival log")
        config = config or SimConfig()
        ticks = config.horizon_s // self.tick_s
        n_bins = max(1, config.horizon_s // self.bin_s)
        ticks_per_bin = self.bin_s // self.tick_s
        past_ticks = 600 // self.tick_s
        shape = (n_bins, config.route_count, 2, config.stops_per_route)
        next15 = np.zeros(shape, np.float64)
        past10 = np.zeros(shape, np.float64)
        for tape in train_logs:
            if "seed" in getattr(tape, "keys", lambda: ())():
                raise ValueError("train logs must be arrival tapes, not scenario objects")
            arrivals = np.asarray(tape)
            if arrivals.shape[0] < ticks:
                raise ValueError("arrival log is shorter than the horizon")
            for bin_id in range(n_bins):
                start = bin_id * ticks_per_bin
                stop = min(start + ticks_per_bin, ticks)
                next15[bin_id] += arrivals[start:stop].sum(axis=0).sum(axis=-1)
                past_start = max(0, start - past_ticks)
                past10[bin_id] += arrivals[past_start:start].sum(axis=0).sum(axis=-1)
        n_days = len(train_logs)
        self.next15 = next15 / n_days
        self.past10 = past10 / n_days

    def predict(self, history: dict, time_s: int) -> Forecast:
        if self.next15 is None or self.past10 is None:
            raise RuntimeError("forecaster must be fit before predict")
        if "arrival_tape" in history or "seed" in history:
            raise ValueError("forecast history must not include future tapes or scenario seeds")
        recent = np.asarray(history["arrival_history"], dtype=np.float64).sum(axis=-1) * 40.0
        bin_id = min(max(time_s, 0) // self.bin_s, self.next15.shape[0] - 1)
        routes, directions, stops = self.next15.shape[1:]
        base = self.next15[bin_id]
        past = self.past10[bin_id]
        correction = np.clip((recent[:routes, :directions, :stops] + 1.0) / (past + 1.0), 0.5, 2.0)
        expected = np.zeros((4, 2, 8), np.float32)
        expected[:routes, :directions, :stops] = (base * correction / 40.0).astype(np.float32)
        if not np.isfinite(expected).all():
            raise ValueError("forecast produced NaN")
        return Forecast(expected)

    def evaluate_logs(self, tapes: list[np.ndarray], config: SimConfig | None = None) -> dict:
        config = config or SimConfig()
        errors = []
        biases = []
        ticks_per_bin = self.bin_s // self.tick_s
        for tape in tapes:
            for time_s in range(0, config.horizon_s, self.bin_s):
                history = history_from_tape(tape, time_s, config.tick_s, config.control_interval_s)
                predicted = self.predict(history, time_s).expected
                start = time_s // config.tick_s
                stop = min(start + ticks_per_bin, tape.shape[0])
                actual = np.zeros((4, 2, 8), np.float64)
                chunk = tape[start:stop].sum(axis=0).sum(axis=-1) / 40.0
                r, d, s = chunk.shape
                actual[:r, :d, :s] = chunk
                errors.append(np.abs(predicted - actual).mean())
                biases.append((predicted - actual).mean())
        return {
            "mae": float(np.mean(errors)),
            "bias": float(np.mean(biases)),
            "n": float(len(errors)),
        }

    def save(self, path) -> None:
        if self.next15 is None or self.past10 is None:
            raise RuntimeError("cannot save an unfitted forecaster")
        np.savez(
            path,
            next15=self.next15,
            past10=self.past10,
            bin_s=self.bin_s,
            tick_s=self.tick_s,
        )

    @classmethod
    def load(cls, path) -> HistoricalForecaster:
        with np.load(path, allow_pickle=False) as data:
            forecaster = cls(bin_s=int(data["bin_s"]), tick_s=int(data["tick_s"]))
            forecaster.next15 = data["next15"]
            forecaster.past10 = data["past10"]
        return forecaster
