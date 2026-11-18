#!/usr/bin/env python3
"""
IMP  .  reasoning engine                  (the brain - recommendation only)

Given (a) a ShedRequirement from the supply side and (b) the attributed
workloads with their business priority, produce a RANKED shed recommendation:
which workloads to shed first, in what order, until the deficit is closed -
with rationale and blast radius for each.

Hard rule: this NEVER executes. It emits a recommendation with an explicit
approval step. Turning an approved recommendation into an action is a separate
layer, and even then IMPS hands the action to the orchestrator (K8s/vCenter);
it never actuates infrastructure directly. (FR-1 / recommendation-only.)

Ranking logic:
    shed the LOWEST business priority first, and among equal priority, the
    workloads that free the most watts per unit of disruption. Protected tiers
    (T0) are never recommended for shedding, even if that leaves the deficit
    unmet - in which case the shortfall is surfaced, not silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# priority tiers (lower number = more critical = shed later / never)
TIER_PROTECTED = 0     # clinical/patient, regulated, core txn - NEVER shed
TIER_CRITICAL = 1      # customer-facing production
TIER_STANDARD = 2      # normal services
TIER_FLEXIBLE = 3      # batch, training, dev - shed first


@dataclass
class Workload:
    wid: str                    # id (pod/VM/app)
    watts: float                # attributed power
    tier: int                   # priority tier (0..3)
    bu: str = "unmapped"
    kind: str = "workload"      # for the action verb: pod->evict, vm->throttle...
    confidence: float = 1.0     # attribution confidence (from the splitter)


@dataclass
class ShedAction:
    wid: str
    watts: float
    tier: int
    bu: str
    action: str                 # recommended verb (evict / throttle / pause)
    rationale: str

@dataclass
class Recommendation:
    required_w: float
    recommended_w: float        # watts the recommended set actually frees
    actions: list               # ordered ShedActions
    shortfall_w: float          # unmet deficit after protected tiers excluded
    urgency: str
    approved: bool = False      # flips only via explicit human approval

    @property
    def meets_requirement(self) -> bool:
        return self.shortfall_w <= 0

    def approve(self):
        """The human-in-the-loop step. Nothing executes even after this - a
        separate action layer would hand approved actions to the orchestrator."""
        self.approved = True
        return self


def _verb(kind: str) -> str:
    return {"pod": "evict", "vm": "throttle/suspend",
            "process": "stop"}.get(kind, "shed")


def recommend_shed(required_w: float, workloads: list[Workload],
                   urgency: str = "elevated") -> Recommendation:
    """
    Rank workloads and select a shed set that closes required_w with least
    business impact. Protected (T0) workloads are never selected.
    """
    if required_w <= 0:
        return Recommendation(0.0, 0.0, [], 0.0, urgency)

    # candidates = everything except protected; shed flexible tiers first
    # (highest tier number). Within the working tier, close the deficit with the
    # LEAST over-shed: if one workload covers it alone, pick the smallest such;
    # otherwise accumulate smallest-first until covered.
    candidates = [w for w in workloads if w.tier > TIER_PROTECTED]

    actions, freed = [], 0.0
    # process tier by tier, flexible (3) -> critical (1)
    for tier in sorted({w.tier for w in candidates}, reverse=True):
        if freed >= required_w:
            break
        tier_ws = sorted([w for w in candidates if w.tier == tier],
                         key=lambda w: w.watts)   # smallest first
        remaining = required_w - freed
        # if a single workload in this tier covers the remaining deficit, take
        # the smallest one that does (minimises over-shed and disruption).
        covering = [w for w in tier_ws if w.watts >= remaining]
        if covering:
            tier_ws = [covering[0]]               # smallest single that covers
        for w in tier_ws:
            if freed >= required_w:
                break
            tier_name = {1: "critical", 2: "standard", 3: "flexible"}.get(w.tier, "?")
            actions.append(ShedAction(
                wid=w.wid, watts=w.watts, tier=w.tier, bu=w.bu,
                action=_verb(w.kind),
                rationale=(f"tier={tier_name}, {w.watts:.0f}W, BU={w.bu}, "
                           f"attribution conf {w.confidence*100:.0f}%")))
            freed += w.watts

    shortfall = max(required_w - freed, 0.0)
    return Recommendation(required_w, freed, actions, shortfall, urgency)


# --------------------------------------------------------------- render
def render(rec: Recommendation) -> str:
    L = ["\n  SHED RECOMMENDATION  (recommendation only - nothing executed)\n"]
    L.append(f"  required : {rec.required_w:.0f} W   urgency: {rec.urgency}")
    L.append(f"  {'ORDER':<6}{'WORKLOAD':<16}{'WATTS':>8}{'ACTION':>16}   RATIONALE")
    L.append("  " + "-" * 74)
    for i, a in enumerate(rec.actions, 1):
        L.append(f"  {i:<6}{a.wid:<16}{a.watts:>8.0f}{a.action:>16}   {a.rationale}")
    L.append("  " + "-" * 74)
    L.append(f"  recommended shed: {rec.recommended_w:.0f} W "
             f"({len(rec.actions)} workloads)")
    if rec.meets_requirement:
        L.append(f"  -> closes the {rec.required_w:.0f} W deficit. Awaiting approval.")
    else:
        L.append(f"  !! SHORTFALL {rec.shortfall_w:.0f} W - protected workloads "
                 f"excluded. Human must decide whether to touch a protected tier.")
    L.append(f"  status: {'APPROVED' if rec.approved else 'PENDING HUMAN APPROVAL'}"
             f"  (IMPS does not execute; on approval the orchestrator acts)")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------- demo
if __name__ == "__main__":
    from imp_supply import SupplyState, detect_constraint

    # supply: over PDU cap -> must shed
    supply = SupplyState(current_load_w=5400, pdu_cap_w=5000)
    req = detect_constraint(supply)
    print("\n  SUPPLY:", req.headline)

    # workloads with attributed power + business priority (from the roll-up)
    workloads = [
        Workload("clinical-db", 300, TIER_PROTECTED, "Health", "vm", 0.95),
        Workload("payments-api", 500, TIER_CRITICAL, "Finance", "pod", 0.9),
        Workload("web-store", 400, TIER_STANDARD, "Retail", "pod", 0.85),
        Workload("llm-train", 1700, TIER_FLEXIBLE, "Research", "pod", 0.88),
        Workload("batch-eval", 250, TIER_FLEXIBLE, "Research", "pod", 0.7),
        Workload("render-farm", 900, TIER_FLEXIBLE, "Studio", "vm", 0.92),
    ]

    rec = recommend_shed(req.shed_w, workloads, req.urgency.value)
    print(render(rec))

    # the human-in-the-loop step
    print("  [human reviews and approves]")
    rec.approve()
    print(render(rec))
