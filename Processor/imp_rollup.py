#!/usr/bin/env python3
"""
IMP  .  business-unit roll-up             (final attribution layer)

The splitter gives watts per VM / pod. This aggregates those leaf workloads UP:

    workload (VM/pod)  ->  application  ->  service  ->  business unit

Two facts join here, from different places:
  * workload -> application : mostly already known (K8s 'app' label, VM tag).
  * application -> BU       : has NO technical source - it is an ORG fact.
    IMPS reads it from a customer-provided mapping (or a direct 'business-unit'
    label if the workload carries one). It NEVER infers a BU.

Resolution order per workload (the "both" strategy):
    1. a direct business-unit label on the workload, if present   (authoritative)
    2. else app -> (service, BU) from the customer mapping file
    3. else "unmapped"  - surfaced honestly, not hidden

This is aggregation, not splitting: watt-weighted sums, and confidence rolls up
watt-weighted from the leaf attributions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------- inputs
@dataclass
class Leaf:
    """A workload with attributed power (flattened from the splitter tree)."""
    cid: str
    watts: float
    confidence: float
    app: Optional[str] = None          # from label/tag if known
    bu_label: Optional[str] = None     # a direct business-unit label, if present


def collect_leaves(attributions, app_by_cid=None, bu_label_by_cid=None) -> list[Leaf]:
    """Flatten the splitter's Attribution tree to its LEAVES (no children).
    Leaves are the real workloads; their power sums to the parent total."""
    app_by_cid = app_by_cid or {}
    bu_label_by_cid = bu_label_by_cid or {}
    out = []

    def walk(items):
        for a in items:
            if a.children:
                walk(a.children)
            else:
                out.append(Leaf(a.cid, a.total_w, a.confidence,
                                app=app_by_cid.get(a.cid),
                                bu_label=bu_label_by_cid.get(a.cid)))
    walk(attributions)
    return out


# --------------------------------------------------------------- mapping
def load_bu_mapping(source) -> dict:
    """
    app -> {"service": ..., "bu": ...}.
    Accepts a dict, or a path to a JSON file of the same shape.
    """
    if isinstance(source, dict):
        return source
    if isinstance(source, str):
        with open(source) as f:
            return json.load(f)
    return {}


def resolve(leaf: Leaf, bu_mapping: dict) -> tuple[str, str, str]:
    """Return (app, service, bu) for a leaf, per the both-strategy order."""
    app = leaf.app or _app_from_cid(leaf.cid)
    # 1. direct BU label wins
    if leaf.bu_label:
        entry = bu_mapping.get(app, {})
        return app, entry.get("service", "unmapped"), leaf.bu_label
    # 2. app -> mapping file
    entry = bu_mapping.get(app)
    if entry:
        return app, entry.get("service", "unmapped"), entry.get("bu", "unmapped")
    # 3. unmapped
    return app, "unmapped", "unmapped"


def _app_from_cid(cid: str) -> str:
    """Best-effort app name when no label: strip a 'namespace/' prefix."""
    return cid.split("/", 1)[1] if "/" in cid else cid


# --------------------------------------------------------------- roll-up
@dataclass
class Bucket:
    label: str
    watts: float = 0.0
    conf_weighted: float = 0.0         # sum(conf * watts), for watt-weighted conf
    workloads: int = 0
    children: dict = field(default_factory=dict)

    @property
    def confidence(self):
        return (self.conf_weighted / self.watts) if self.watts else 0.0


def rollup_to_bu(leaves: list[Leaf], bu_mapping: dict) -> Bucket:
    """Aggregate leaves up to BU -> service -> app. Confidence is watt-weighted."""
    root = Bucket("ALL")
    for lf in leaves:
        app, service, bu = resolve(lf, bu_mapping)
        path = [(bu, root)]
        bu_b = root.children.setdefault(bu, Bucket(bu))
        svc_b = bu_b.children.setdefault(service, Bucket(service))
        app_b = svc_b.children.setdefault(app, Bucket(app))
        for b in (root, bu_b, svc_b, app_b):
            b.watts += lf.watts
            b.conf_weighted += lf.confidence * lf.watts
            b.workloads += 1
    return root


# --------------------------------------------------------------- render
def render(root: Bucket) -> str:
    L = ["\n  POWER BY BUSINESS UNIT\n"]
    L.append(f"  {'ENTITY':<26}{'WATTS':>9}{'kW':>8}{'CONF':>7}{'WORKLOADS':>11}")
    L.append("  " + "-" * 61)

    def walk(bucket, depth):
        for name, b in sorted(bucket.children.items(),
                              key=lambda kv: kv[1].watts, reverse=True):
            ind = "  " * depth
            flag = "  <- UNMAPPED" if name == "unmapped" and depth == 0 else ""
            L.append(f"  {ind + name:<26}{b.watts:>9.0f}{b.watts/1000:>8.2f}"
                     f"{b.confidence*100:>6.0f}%{b.workloads:>11}{flag}")
            walk(b, depth + 1)
    walk(root, 0)

    L.append("  " + "-" * 61)
    L.append(f"  {'TOTAL':<26}{root.watts:>9.0f}{root.watts/1000:>8.2f}"
             f"{root.confidence*100:>6.0f}%{root.workloads:>11}")
    unmapped = root.children.get("unmapped")
    if unmapped:
        pct = 100 * unmapped.watts / root.watts if root.watts else 0
        L.append(f"\n  !! {pct:.0f}% of power is UNMAPPED to a business unit - "
                 f"provide an app->BU mapping to attribute it.")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------- demo
if __name__ == "__main__":
    # Reuse the splitter to produce a real attribution tree, then roll it up.
    from imp_power_split import Gpu, Child, split_power

    gpus = [Gpu(f"gpu{i}", 700, "passthrough",
                owner=("VM-A" if i < 2 else "VM-B")) for i in range(4)]
    vms = [
        Child("VM-A", 0.5, "util", gpu_ids=["gpu0", "gpu1"], children=[
            Child("ml/llm-train", 0.75, "util", gpu_ids=["gpu0", "gpu1"]),
            Child("ml/eval", 0.25, "util")]),
        Child("VM-B", 0.3, "util", gpu_ids=["gpu2", "gpu3"]),
        Child("VM-C", 0.2, "util"),
    ]
    attrs, report = split_power(700 * 4 + 800, gpus, vms)

    # app tags for each leaf workload (from labels in a real read)
    app_by_cid = {"ml/llm-train": "llm-train", "ml/eval": "eval",
                  "VM-B": "rendering", "VM-C": "analytics"}
    # one workload carries a direct BU label (authoritative); others use the file
    bu_label_by_cid = {"VM-C": "Finance"}

    # customer-provided app -> service/BU mapping (the org fact)
    bu_mapping = {
        "llm-train": {"service": "model-training", "bu": "Research"},
        "eval":      {"service": "model-training", "bu": "Research"},
        "rendering": {"service": "media", "bu": "Studio"},
        # "analytics" intentionally left out -> VM-C falls back to its BU label
    }

    leaves = collect_leaves(attrs, app_by_cid, bu_label_by_cid)
    root = rollup_to_bu(leaves, bu_mapping)
    print(render(root))
