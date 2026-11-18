#!/usr/bin/env python3
"""
IMP  .  energy ledger   (power -> energy -> cost / carbon over a reporting period)

The attribution engine produces instantaneous WATTS per entity (workload / VM /
pod / business unit). This turns a stream of those readings into ENERGY (Wh /
kWh) integrated over time, then into COST ($) and CARBON (kgCO2e) over a
reporting period. Persist to survive restarts / cover a day or month.

Usage:
    led = EnergyLedger(tariff_per_kwh=0.12, carbon_g_per_kwh=400)
    # on every sample (e.g. once a second/minute):
    led.record({"Research": 1800.0, "Finance": 160.0})   # watts per entity
    ...
    report = led.report()          # kWh, $, kgCO2e per entity + totals
    led.save("ledger.json")        # persist across restarts

Integration is trapezoidal between consecutive samples (power held/averaged over
the interval). Entities that appear or disappear are handled (a new entity's
power applies forward; a departed entity's last power applies to its final
interval). Timestamps may be injected for deterministic testing.
"""

from __future__ import annotations

import json
import time
from typing import Optional


class EnergyLedger:
    def __init__(self, tariff_per_kwh: Optional[float] = None,
                 carbon_g_per_kwh: Optional[float] = None):
        self.wh: dict[str, float] = {}        # entity -> accumulated watt-hours
        self.last_power: dict[str, float] = {}  # entity -> last watts seen
        self.last_ts: Optional[float] = None
        self.period_start: Optional[float] = None
        self.samples: int = 0
        self.tariff = tariff_per_kwh
        self.carbon = carbon_g_per_kwh

    # ------------------------------------------------------------- ingest
    def record(self, power_by_entity: dict, timestamp: Optional[float] = None):
        """Add one sample of {entity: watts}. Integrates energy since the last
        sample. `timestamp` (epoch seconds) may be injected; else time.time()."""
        ts = timestamp if timestamp is not None else time.time()
        if self.period_start is None:
            self.period_start = ts

        if self.last_ts is not None:
            dt_h = (ts - self.last_ts) / 3600.0
            if dt_h > 0:
                for e in set(power_by_entity) | set(self.last_power):
                    p0 = self.last_power.get(e)
                    p1 = power_by_entity.get(e)
                    if p0 is not None and p1 is not None:
                        avg = (p0 + p1) / 2.0          # trapezoid
                    elif p1 is not None:
                        avg = p1                        # newly appeared
                    else:
                        avg = p0                        # departed
                    self.wh[e] = self.wh.get(e, 0.0) + avg * dt_h

        self.last_power = dict(power_by_entity)
        self.last_ts = ts
        self.samples += 1

    # ------------------------------------------------------------- report
    def report(self, tariff_per_kwh: Optional[float] = None,
               carbon_g_per_kwh: Optional[float] = None) -> dict:
        tariff = tariff_per_kwh if tariff_per_kwh is not None else self.tariff
        carbon = carbon_g_per_kwh if carbon_g_per_kwh is not None else self.carbon
        dur_h = ((self.last_ts - self.period_start) / 3600.0
                 if self.last_ts is not None and self.period_start is not None
                 else 0.0)

        entities = []
        for e, wh in sorted(self.wh.items(), key=lambda kv: kv[1], reverse=True):
            kwh = wh / 1000.0
            row = {"entity": e, "wh": round(wh, 3), "kwh": round(kwh, 5),
                   "avg_w": round(wh / dur_h, 1) if dur_h > 0 else 0.0}
            if tariff is not None:
                row["cost"] = round(kwh * tariff, 5)
            if carbon is not None:
                row["kg_co2e"] = round(kwh * carbon / 1000.0, 5)
            entities.append(row)

        total_wh = sum(self.wh.values())
        out = {
            "period_start": self.period_start,
            "period_end": self.last_ts,
            "duration_hours": round(dur_h, 5),
            "samples": self.samples,
            "total_wh": round(total_wh, 3),
            "total_kwh": round(total_wh / 1000.0, 5),
            "tariff_per_kwh": tariff,
            "carbon_g_per_kwh": carbon,
            "entities": entities,
        }
        if tariff is not None:
            out["total_cost"] = round(total_wh / 1000.0 * tariff, 5)
        if carbon is not None:
            out["total_kg_co2e"] = round(total_wh / 1000.0 * carbon / 1000.0, 5)
        return out

    # ------------------------------------------------------------- lifecycle
    def reset(self):
        """Start a fresh reporting period (e.g. daily/monthly rollover)."""
        self.wh.clear()
        self.last_power.clear()
        self.last_ts = None
        self.period_start = None
        self.samples = 0

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"wh": self.wh, "last_power": self.last_power,
                       "last_ts": self.last_ts, "period_start": self.period_start,
                       "samples": self.samples, "tariff": self.tariff,
                       "carbon": self.carbon}, f)

    @classmethod
    def load(cls, path: str) -> "EnergyLedger":
        with open(path) as f:
            d = json.load(f)
        led = cls(tariff_per_kwh=d.get("tariff"), carbon_g_per_kwh=d.get("carbon"))
        led.wh = {k: float(v) for k, v in d.get("wh", {}).items()}
        led.last_power = {k: float(v) for k, v in d.get("last_power", {}).items()}
        led.last_ts = d.get("last_ts")
        led.period_start = d.get("period_start")
        led.samples = d.get("samples", 0)
        return led
