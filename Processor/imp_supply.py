#!/usr/bin/env python3
"""
IMP  .  supply-side constraint detection      (the trigger)

Reads the SUPPLY side - power coming IN and the limits on it - and answers one
question: is there a constraint right now, and how many watts must come off?

This is what turns IMPS from an observer into a decision system. Nothing here
actuates; it only detects and quantifies a constraint so the reasoning engine
has something to reason about.

Constraint sources (each a signal the site already has, read-only):
    grid        utility feed up/down, brownout                (BMS / ATS / meter)
    pdu_cap     rack/PDU draw vs its rated cap                (smart PDU / DCIM)
    cooling     cooling headroom / chiller fault              (BMS)
    budget      contracted power / demand-charge ceiling      (contract + meter)
    grid_signal utility curtailment / demand-response event   (utility API)

A constraint is BINDING when the site must reduce load. The output is a
ShedRequirement: how many watts to shed, why, and how urgent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Urgency(str, Enum):
    NONE = "none"
    ADVISORY = "advisory"      # not binding yet; heads-up (approaching a limit)
    ELEVATED = "elevated"      # binding, act soon (minutes)
    CRITICAL = "critical"      # binding now (seconds) - grid down, over hard cap


@dataclass
class SupplyState:
    """A snapshot of the supply side. Fed from BMS/PDU/meter/contract, or, on a
    bench, declared. All powers in watts."""
    current_load_w: float                 # what the site is drawing now
    grid_up: bool = True                  # utility feed healthy
    pdu_cap_w: Optional[float] = None     # rated cap of the rack/PDU/feed
    cooling_headroom_w: Optional[float] = None   # spare cooling capacity (W of IT)
    power_budget_w: Optional[float] = None       # contracted / demand-charge ceiling
    on_backup: bool = False               # running on battery/DG (finite runway)
    backup_runway_w: Optional[float] = None  # load the backup can sustain
    dr_event_target_w: Optional[float] = None  # utility asked to drop TO this


@dataclass
class ShedRequirement:
    binding: bool
    shed_w: float                         # watts that must come off (0 if none)
    urgency: Urgency
    reasons: list                         # human-readable, per triggered constraint

    @property
    def headline(self) -> str:
        if not self.binding:
            return "supply OK - no shed required"
        return (f"shed {self.shed_w:.0f} W [{self.urgency.value}] - "
                + "; ".join(self.reasons))


def detect_constraint(s: SupplyState,
                      advisory_margin: float = 0.90) -> ShedRequirement:
    """
    Evaluate every supply constraint against the current load. Returns the
    LARGEST shed any single constraint demands (you must satisfy the tightest),
    at the HIGHEST urgency triggered.

    advisory_margin: fraction of a cap at which to raise an advisory before it
    actually binds (default 90%).
    """
    load = s.current_load_w
    demands = []   # (shed_w, urgency, reason)

    # 1. grid down + on finite backup -> shed to what backup sustains
    if not s.grid_up and s.on_backup and s.backup_runway_w is not None:
        if load > s.backup_runway_w:
            demands.append((load - s.backup_runway_w, Urgency.CRITICAL,
                            f"grid down; load {load:.0f}W exceeds backup runway "
                            f"{s.backup_runway_w:.0f}W"))
    elif not s.grid_up and not s.on_backup:
        # grid down, no backup engaged -> everything non-critical must go now
        demands.append((load, Urgency.CRITICAL, "grid down, no backup power"))

    # 2. PDU / feed hard cap
    if s.pdu_cap_w is not None:
        if load > s.pdu_cap_w:
            demands.append((load - s.pdu_cap_w, Urgency.CRITICAL,
                            f"over PDU cap: {load:.0f}W > {s.pdu_cap_w:.0f}W"))
        elif load > advisory_margin * s.pdu_cap_w:
            demands.append((0.0, Urgency.ADVISORY,
                            f"approaching PDU cap ({100*load/s.pdu_cap_w:.0f}%)"))

    # 3. cooling headroom (IT load cooling can remove)
    if s.cooling_headroom_w is not None and s.cooling_headroom_w < 0:
        demands.append((-s.cooling_headroom_w, Urgency.ELEVATED,
                        f"cooling deficit: {-s.cooling_headroom_w:.0f}W over capacity"))

    # 4. contracted power / demand-charge ceiling
    if s.power_budget_w is not None:
        if load > s.power_budget_w:
            demands.append((load - s.power_budget_w, Urgency.ELEVATED,
                            f"over power budget: {load:.0f}W > {s.power_budget_w:.0f}W"))
        elif load > advisory_margin * s.power_budget_w:
            demands.append((0.0, Urgency.ADVISORY,
                            f"approaching power budget ({100*load/s.power_budget_w:.0f}%)"))

    # 5. utility demand-response / curtailment event
    if s.dr_event_target_w is not None and load > s.dr_event_target_w:
        demands.append((load - s.dr_event_target_w, Urgency.ELEVATED,
                        f"utility DR event: drop to {s.dr_event_target_w:.0f}W"))

    if not demands:
        return ShedRequirement(False, 0.0, Urgency.NONE, [])

    # must satisfy the tightest constraint; carry the highest urgency + all reasons
    shed_w = max(d[0] for d in demands)
    order = [Urgency.NONE, Urgency.ADVISORY, Urgency.ELEVATED, Urgency.CRITICAL]
    urgency = max((d[1] for d in demands), key=order.index)
    reasons = [d[2] for d in demands]
    binding = shed_w > 0
    return ShedRequirement(binding, shed_w, urgency, reasons)


# --------------------------------------------------------------- demo
if __name__ == "__main__":
    scenarios = {
        "healthy": SupplyState(current_load_w=3600, grid_up=True,
                               pdu_cap_w=5000, power_budget_w=4500),
        "approaching cap": SupplyState(current_load_w=4700, pdu_cap_w=5000),
        "over PDU cap": SupplyState(current_load_w=5400, pdu_cap_w=5000),
        "grid down on backup": SupplyState(current_load_w=3600, grid_up=False,
                                           on_backup=True, backup_runway_w=2000),
        "DR event": SupplyState(current_load_w=3600, dr_event_target_w=2500),
        "cooling deficit": SupplyState(current_load_w=3600,
                                       cooling_headroom_w=-400),
    }
    print("\n  SUPPLY-SIDE CONSTRAINT DETECTION\n")
    for name, st in scenarios.items():
        req = detect_constraint(st)
        print(f"  {name:<22} -> {req.headline}")
    print()
