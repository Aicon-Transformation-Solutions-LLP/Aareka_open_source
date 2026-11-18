#!/usr/bin/env python3
"""
Aareka  .  GPU waste engine   (processor)

Turns the raw per-GPU readings a collector sends (utilisation, memory, power,
owners) into:

    1. a STATE for each GPU at each reading
         empty            no process on it, utilisation below the threshold
         loaded_waiting   a process holds it, utilisation below the threshold
         light_use        utilisation above the threshold, power close to idle
         working          utilisation above the threshold, power well above idle
         unknown          the readings needed are not available
    2. each GPU's IDLE POWER LEVEL, learned from its own history
    3. IDLE SPELLS - back-to-back readings in a waste state for one GPU + owner
    4. per-day GPU-hours and energy, per owner, split at UTC midnight

Nothing here is stored as the only record: every output is derived from the raw
readings, so it can be rebuilt when a threshold changes (replay the readings in
time order through a fresh GpuTracker).

Rules this module keeps:
    * a missing reading stays missing (None) - it is never treated as zero
    * time with no readings is never counted as idle
    * modeled power is never used to learn an idle level or to classify

Pure logic: no database, no network, standard library only.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from datetime import datetime, timezone

# ------------------------------------------------------------------ states
EMPTY = "empty"
LOADED_WAITING = "loaded_waiting"
LIGHT_USE = "light_use"
WORKING = "working"
UNKNOWN = "unknown"

ALL_STATES = (EMPTY, LOADED_WAITING, LIGHT_USE, WORKING, UNKNOWN)
STATE_LABELS = {
    EMPTY: "Empty",
    LOADED_WAITING: "Loaded, waiting",
    LIGHT_USE: "Light use",
    WORKING: "Working",
    UNKNOWN: "Unknown",
}

# ------------------------------------------------------------------ thresholds
# Defaults a new account gets. An account's own values (from its settings)
# replace these one by one - see resolve_thresholds().
DEFAULT_THRESHOLDS = {
    "util_threshold_pct": 5.0,        # below this a GPU counts as not working
    "idle_power_margin_pct": 15.0,    # power within this % of idle = close to idle
    "waste_states": [EMPTY, LOADED_WAITING],
}
# States a user may choose to count as waste. "unknown" can never be waste.
WASTE_STATE_CHOICES = (EMPTY, LOADED_WAITING, LIGHT_USE)

# How the idle level and spells are measured (the method, not user thresholds).
IDLE_RUN_READINGS = 5                 # readings in a row that make a draw "sustained"
IDLE_LOOKBACK_S = 30 * 86400          # only the last 30 days can set the idle level
MIN_GAP_LIMIT_S = 300.0               # a gap longer than this ends a run / spell ...
GAP_INTERVAL_FACTOR = 3.0             # ... or 3x the GPU's usual interval, if longer
GAP_HISTORY = 21                      # recent gaps kept to find the usual interval

NO_OWNER = ""                         # owner key for a GPU with no workload on it

# Categories a stretch of time is filed under for the daily totals.
CAT_IDLE, CAT_WORKING, CAT_OTHER = "idle", "working", "other"


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def resolve_thresholds(user: "dict | None") -> dict:
    """Defaults, overridden by whichever values the account has set. An invalid
    or missing value falls back to its default. `user_set` lists the keys the
    account's own value is in use for."""
    user = user or {}
    out = {k: (list(v) if isinstance(v, list) else v)
           for k, v in DEFAULT_THRESHOLDS.items()}
    user_set = []

    v = user.get("util_threshold_pct")
    if _is_num(v) and 0 < v <= 100:
        out["util_threshold_pct"] = float(v)
        user_set.append("util_threshold_pct")

    v = user.get("idle_power_margin_pct")
    if _is_num(v) and 0 <= v <= 1000:
        out["idle_power_margin_pct"] = float(v)
        user_set.append("idle_power_margin_pct")

    v = user.get("waste_states")
    if isinstance(v, (list, tuple)) and all(s in WASTE_STATE_CHOICES for s in v):
        out["waste_states"] = [s for s in WASTE_STATE_CHOICES if s in v]
        user_set.append("waste_states")

    out["user_set"] = user_set
    return out


# ------------------------------------------------------------------ settings
# Every setting is optional. `clean_settings` keeps only the values an account
# has actually set; anything missing falls back to the shipped default (for
# thresholds) or to "no money figure" (for rates).
DEFAULT_TICKET_IDLE_HOURS = 4.0       # idle this long before a ticket is raised
THRESHOLD_KEYS = ("util_threshold_pct", "idle_power_margin_pct", "waste_states")
RECALC_MESSAGE = ("Past waste readings will be recalculated using the new threshold. "
                  "Figures already shown on the dashboard will change.")


def _unset(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def clean_settings(raw: "dict | None") -> tuple:
    """Validate what an account wants to save -> (clean, errors).
    `clean` holds only the values that are set. None, "" or a missing key means
    "not set". `errors` is a list of plain sentences; when it is not empty,
    nothing should be saved."""
    raw = raw if isinstance(raw, dict) else {}
    clean, errors = {}, []

    def as_number(v):
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                return None
        return float(v) if _is_num(v) else None

    def number(key, label, lo, hi, above_lo=True):
        if _unset(raw.get(key)):
            return
        v = as_number(raw.get(key))
        if v is None or (v <= lo if above_lo else v < lo) or v > hi:
            errors.append(f"{label} must be {'more than' if above_lo else 'at least'} "
                          f"{lo:g} and at most {hi:g}.")
            return
        clean[key] = v

    number("electricity_rate_per_kwh", "Electricity rate per kWh", 0, 1e6)
    number("pue", "PUE", 1, 5, above_lo=False)
    number("carbon_g_per_kwh", "Carbon factor (g CO2e per kWh)", 0, 2000)
    number("util_threshold_pct", "The utilisation threshold", 0, 100)
    number("idle_power_margin_pct", "The idle power margin", 0, 1000, above_lo=False)
    number("ticket_idle_hours", "Hours idle before a ticket", 0, 24 * 90)

    cur = raw.get("currency")
    if not _unset(cur):
        if not isinstance(cur, str) or len(cur.strip()) > 8:
            errors.append("Currency must be at most 8 characters.")
        else:
            clean["currency"] = cur.strip()

    ws = raw.get("waste_states")
    if ws is not None:
        if not isinstance(ws, (list, tuple)) or any(st not in WASTE_STATE_CHOICES for st in ws):
            errors.append("States counted as waste must be chosen from: "
                          + ", ".join(STATE_LABELS[st] for st in WASTE_STATE_CHOICES) + ".")
        else:
            clean["waste_states"] = [st for st in WASTE_STATE_CHOICES if st in ws]

    rates = raw.get("gpu_hour_rates")
    if rates is not None:
        if not isinstance(rates, dict) or len(rates) > 200:
            errors.append("GPU-hour rates must be a list of at most 200 GPU models.")
        else:
            out = {}
            for model, v in rates.items():
                if _unset(v):
                    continue
                n = as_number(v)
                if (not isinstance(model, str) or not model.strip() or len(model) > 120
                        or n is None or n <= 0 or n > 1e6):
                    errors.append(f"The GPU-hour rate for {str(model)[:60]} must be more "
                                  f"than 0 and at most 1e+06.")
                    continue
                out[model.strip()] = n
            if out:
                clean["gpu_hour_rates"] = out
    return clean, errors


def resolve_rates(settings: "dict | None") -> dict:
    """The rates an account has set. A rate that is not set is None (or absent
    from gpu_hour_rates) - it is never replaced by a made-up figure."""
    clean, _ = clean_settings(settings)
    return {"electricity_rate_per_kwh": clean.get("electricity_rate_per_kwh"),
            "currency": clean.get("currency"),
            "pue": clean.get("pue"),
            "carbon_g_per_kwh": clean.get("carbon_g_per_kwh"),
            "gpu_hour_rates": clean.get("gpu_hour_rates", {})}


def resolve_ticket_idle_hours(settings: "dict | None") -> float:
    clean, _ = clean_settings(settings)
    return clean.get("ticket_idle_hours", DEFAULT_TICKET_IDLE_HOURS)


# ------------------------------------------------------------------ cost
# Waste cost has two parts, always kept apart:
#     GPU-hour part  = idle GPU-hours x the GPU-hour rate for that GPU's model
#     energy part    = idle energy (kWh) x electricity rate x PUE
# A part whose rate is not set is None: no money figure, not zero.
def gpu_hour_rate(model, rates: dict):
    """The GPU-hour rate set for this GPU model, or None."""
    return rates["gpu_hour_rates"].get(model) if model else None


def energy_cost(energy_wh, rates: dict):
    """Idle energy x electricity rate x PUE. None without an electricity rate.
    When no PUE is set none is applied: the figure is then IT energy only."""
    rate = rates.get("electricity_rate_per_kwh")
    if rate is None or energy_wh is None:
        return None
    return energy_wh / 1000.0 * rate * (rates.get("pue") or 1.0)


# ------------------------------------------------------------------ classify
def classify(util_pct, watts, power_measured: bool, has_owner: bool,
             idle_w, th: dict) -> tuple:
    """One reading -> (state, note). `note` says when the power test could not
    be applied or disagreed; None otherwise.

    The power test needs measured watts AND a learned idle level. Without both,
    the GPU is classified on utilisation and ownership alone."""
    if util_pct is None:
        return UNKNOWN, "utilisation not available"

    if watts is None:
        why_no_power = "power reading not available"
    elif not power_measured:
        why_no_power = "power is modeled, not measured"
    elif idle_w is None:
        why_no_power = "idle power level still learning"
    else:
        why_no_power = None
    near_idle = (why_no_power is None and
                 watts <= idle_w * (1.0 + th["idle_power_margin_pct"] / 100.0))

    if util_pct < th["util_threshold_pct"]:                 # not working
        if has_owner:
            return LOADED_WAITING, None
        if why_no_power:
            return EMPTY, why_no_power
        return EMPTY, (None if near_idle else "power above idle level")

    if why_no_power:                                        # in use, power test off
        return WORKING, why_no_power
    return (LIGHT_USE if near_idle else WORKING), None


# ------------------------------------------------------------------ helpers
def normalise_owners(owners: "dict | None") -> dict:
    """{name: {"share", "department"}} with shares scaled to add up to 1.
    Shares that are missing or all zero become an equal split."""
    owners = owners or {}
    if not owners:
        return {}
    shares = {n: (float(i.get("share")) if _is_num(i.get("share")) and i.get("share") > 0
                  else 0.0) for n, i in owners.items()}
    total = sum(shares.values())
    out = {}
    for n, i in owners.items():
        share = (shares[n] / total) if total > 0 else 1.0 / len(owners)
        out[n] = {"share": share, "department": i.get("department")}
    return out


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def split_by_day(t0: float, t1: float) -> list:
    """[(utc_day, seconds)] for the stretch t0..t1, cut at each UTC midnight."""
    parts = []
    cur = t0
    while cur < t1:
        nxt = (math.floor(cur / 86400.0) + 1) * 86400.0
        end = min(nxt, t1)
        parts.append((_day(cur), end - cur))
        cur = end
    return parts


# ------------------------------------------------------------------ tracker
class GpuTracker:
    """Everything remembered about ONE GPU between readings. Feed readings in
    time order with step(); save to_state() and pass it back in to carry on
    later. Replaying the same readings always gives the same result."""

    def __init__(self, state: "dict | None" = None):
        s = state or {}
        self.prev = s.get("prev")                       # the previous reading
        self.run = list(s.get("run", []))               # watts of the current low-use run
        self.cands = deque(tuple(c) for c in s.get("cands", []))   # [(ts, watts)]
        self.gaps = list(s.get("gaps", []))
        self.spells = dict(s.get("spells", {}))         # owner -> open spell
        self.last_owner = s.get("last_owner")
        self.last = s.get("last")                       # latest classification

    def to_state(self) -> dict:
        return {"prev": self.prev, "run": self.run, "cands": [list(c) for c in self.cands],
                "gaps": self.gaps, "spells": self.spells, "last_owner": self.last_owner,
                "last": self.last}

    # ---- gaps
    def gap_limit(self) -> float:
        """Longest gap between two readings that still counts as back to back."""
        usual = statistics.median(self.gaps) if self.gaps else 0.0
        return max(MIN_GAP_LIMIT_S, GAP_INTERVAL_FACTOR * usual)

    # ---- idle level
    def _learn_idle(self, ts, watts, measured, low_use, back_to_back) -> None:
        """Lowest sustained draw: every run of IDLE_RUN_READINGS back-to-back
        measured readings with utilisation below the threshold gives a
        candidate (the highest reading in the run). The idle level is the
        lowest candidate of the last IDLE_LOOKBACK_S."""
        if not (measured and low_use):
            self.run = []
            return
        if not back_to_back:
            self.run = []
        self.run = (self.run + [watts])[-IDLE_RUN_READINGS:]
        if len(self.run) == IDLE_RUN_READINGS:
            cand = max(self.run)
            while self.cands and self.cands[-1][1] >= cand:   # keep only useful ones
                self.cands.pop()
            self.cands.append((ts, cand))

    def idle_level(self, ts: float):
        """The learned idle power in watts, or None while still learning."""
        while self.cands and self.cands[0][0] < ts - IDLE_LOOKBACK_S:
            self.cands.popleft()
        return self.cands[0][1] if self.cands else None

    # ---- one reading
    def step(self, r: dict, th: dict) -> "dict | None":
        """Apply one reading. Returns None if it is not newer than the last one
        (out of order - it stays in the raw store for a rebuild). Otherwise:
            {"classification": {...},   the GPU's state at this reading
             "segments": [...],         time since the previous reading, per day/owner
             "spells": [...]}           idle spells started, extended or ended
        `r` keys: ts, watts, method, util_pct, mem_used_mb, mem_total_mb,
                  gpu_model, mode, owners {name: {share, department}}."""
        ts = float(r["ts"])
        prev = self.prev
        back_to_back = False
        gap = None
        if prev is not None:
            gap = ts - prev["ts"]
            if gap <= 0:
                return None
            back_to_back = gap <= self.gap_limit()
            self.gaps = (self.gaps + [gap])[-GAP_HISTORY:]

        watts = r.get("watts") if _is_num(r.get("watts")) else None
        util = r.get("util_pct") if _is_num(r.get("util_pct")) else None
        measured = watts is not None and r.get("method") == "measured"
        owners = normalise_owners(r.get("owners"))
        shared = len(owners) > 1 or r.get("mode") in ("vgpu", "mig")
        low_use = util is not None and util < th["util_threshold_pct"]

        self._learn_idle(ts, watts, measured, low_use, back_to_back)
        idle_w = self.idle_level(ts)
        state, note = classify(util, watts, measured, bool(owners), idle_w, th)
        waste = state in th["waste_states"]
        holders = owners or {NO_OWNER: {"share": 1.0, "department": None}}

        segments = []
        energy_wh = None
        modeled = False
        if back_to_back:
            if watts is not None and prev.get("watts") is not None:
                energy_wh = (watts + prev["watts"]) / 2.0 * gap / 3600.0
                modeled = not (measured and prev.get("measured"))
            prev_waste = prev["state"] in th["waste_states"]
            for owner, info in prev["holders"].items():
                if prev_waste and waste and owner in holders:
                    cat = CAT_IDLE
                elif prev["state"] == WORKING and state == WORKING:
                    cat = CAT_WORKING
                else:
                    cat = CAT_OTHER
                for day, secs in split_by_day(prev["ts"], ts):
                    segments.append({
                        "day": day, "owner": owner, "department": info.get("department"),
                        "category": cat, "state": prev["state"],
                        "gpu_hours": secs / 3600.0 * info["share"],
                        "energy_wh": (energy_wh * secs / gap * info["share"]
                                      if energy_wh is not None else None),
                        "missing_s": 0.0 if energy_wh is not None else secs * info["share"],
                        "estimated": bool(prev.get("shared")), "modeled": modeled})

        # ---- idle spells: back-to-back waste readings for the same GPU + owner
        events, open_now = [], {}
        if waste:
            for owner, info in holders.items():
                sp = self.spells.get(owner) if back_to_back else None
                if sp is not None:
                    share = prev["holders"].get(owner, info)["share"]
                    sp["end_ts"] = ts
                    sp["duration_s"] = ts - sp["start_ts"]
                    sp["gpu_hours"] += gap / 3600.0 * share
                    if energy_wh is not None:
                        sp["energy_wh"] += energy_wh * share
                        sp["modeled"] = sp["modeled"] or modeled
                    else:
                        sp["missing_s"] += gap
                    sp["readings"] += 1
                else:
                    sp = {"owner": owner, "start_ts": ts, "end_ts": ts, "duration_s": 0.0,
                          "gpu_hours": 0.0, "energy_wh": 0.0, "missing_s": 0.0,
                          "readings": 1, "modeled": False, "estimated": False}
                sp["state"] = state
                sp["department"] = info.get("department")
                sp["estimated"] = sp["estimated"] or shared
                sp["open"] = True
                open_now[owner] = sp
                events.append(sp)
        for owner, sp in self.spells.items():
            if open_now.get(owner) is not sp:              # ended at its last reading
                sp["open"] = False
                events.append(sp)
        self.spells = open_now

        if owners:
            self.last_owner = {"ts": ts, "owners": {n: i.get("department")
                                                    for n, i in owners.items()}}
        self.prev = {"ts": ts, "watts": watts, "measured": measured, "state": state,
                     "holders": holders, "shared": shared}
        self.last = {
            "ts": ts, "state": state, "label": STATE_LABELS[state], "note": note,
            "waste": waste, "idle_w": idle_w,
            "idle_status": "learned" if idle_w is not None else "learning",
            "util_pct": util, "watts": watts,
            "power_method": r.get("method") if watts is not None else None,
            "mem_used_mb": r.get("mem_used_mb"), "mem_total_mb": r.get("mem_total_mb"),
            "gpu_model": r.get("gpu_model"), "mode": r.get("mode"),
            "owners": {n: {"share": round(i["share"], 4), "department": i.get("department")}
                       for n, i in owners.items()},
            "shared": shared,
            "last_owner": None if owners else self.last_owner,
        }
        # copies, so results handed out do not change when a spell is extended later
        return {"classification": dict(self.last), "segments": segments,
                "spells": [dict(sp) for sp in events]}
