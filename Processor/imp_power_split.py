#!/usr/bin/env python3
"""
IMP  .  power attribution splitter        (virtualization layer)

Distributes a node's MEASURED power down a topology tree to virtual servers,
workloads, and finally apps/services. Works at any depth:

    bare metal:     server -> app
    VMware:         host -> VM -> app
    Kubernetes:     node -> pod -> app
    K8s on VMware:  host -> VM -> pod -> app

One rule, applied recursively:
    child_watts = parent_watts x (child's share of the resource)

Two resource classes are split differently:
  * GPU power
      - passthrough: whole physical GPU -> its single owner (no split)   ACCURATE
      - shared (MIG / vGPU): split the physical GPU's watts across
        instances by utilization / slice size                            ESTIMATED
  * CPU / RAM / node overhead
      - always split by CPU share (allocation or utilization)            ESTIMATED

Discipline (FR-13):
  * children MUST reconcile to the parent total; any gap is surfaced, not hidden
  * every node carries a confidence that DROPS with each proportional split
  * GPU-passthrough stays near-exact; CPU-split and shared-GPU carry error
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# Memory-power model. Allocated capacity is NOT the same as memory bandwidth
# used, so a flat W/GB overcharges idle, memory-heavy workloads. Path A models
# per-workload memory power as a baseline (refresh/static) term plus an activity
# term:
#     watts = GB * (IDLE + (ACTIVE - IDLE) * active_fraction)
# ACTIVE ~0.4 and IDLE ~0.1 W/GB are DDR4/5 rule-of-thumb figures
# (Micron/Crucial); all adjustable. When no per-workload activity signal is
# available (e.g. vSphere mem.active, OS working-set), DEFAULT_ACTIVE_FRAC is
# assumed and the result is flagged lower-confidence.
MEM_W_PER_GB = 0.4                 # ~fully active
MEM_W_PER_GB_IDLE = 0.1            # refresh / static floor
MEM_DEFAULT_ACTIVE_FRAC = 0.5      # assumed activity when no signal is provided


# --------------------------------------------------------------- inputs
@dataclass
class Gpu:
    """A physical GPU and how it is shared."""
    gpu_id: str
    watts: float                       # measured (NVML/DCGM)
    mode: str = "passthrough"          # passthrough | mig | vgpu
    # owner(s): for passthrough, one child id; for shared, {child_id: share}
    owner: Optional[str] = None                 # passthrough
    shares: dict = field(default_factory=dict)  # shared: child_id -> util share


@dataclass
class Child:
    """A virtual entity under a parent: VM, pod, or app."""
    cid: str
    cpu_share: float                   # 0..1 of the parent's CPU/overhead
    cpu_share_method: str = "alloc"    # alloc | util  (util is more accurate)
    mem_mb: float = 0.0                # allocated memory (MiB); 0 -> CPU-only split
    mem_active_frac: Optional[float] = None  # 0..1 active memory; None -> assumed
    gpu_ids: list = field(default_factory=list)   # physical GPUs it owns (passthrough)
    children: list = field(default_factory=list)  # deeper level (nesting)


@dataclass
class Attribution:
    cid: str
    gpu_w: float
    shared_w: float                    # CPU + overhead share (non-GPU, non-memory)
    attribution: str                   # "exact" | "estimated"  (of the SPLIT, not
                                       #  the sensor; sensor quality is upstream)
    confidence: float                  # 0..1  attribution confidence
    children: list = field(default_factory=list)
    child_report: Optional[dict] = None  # nested reconciliation, surfaced
    mem_w: float = 0.0                 # memory (DRAM) share

    @property
    def total_w(self) -> float:
        return self.gpu_w + self.shared_w + self.mem_w


def _finite_nonneg(x) -> bool:
    try:
        return math.isfinite(x) and x >= 0
    except TypeError:
        return False


# --------------------------------------------------------------- splitter
def split_power(parent_total_w: float,
                gpus: list[Gpu],
                children: list[Child],
                parent_confidence: float = 1.0,
                dram_w: Optional[float] = None
                ) -> tuple[list[Attribution], dict]:
    """
    Split parent_total_w across children. Returns (attributions, report).
    report carries reconciliation + the unattributed remainder.

    dram_w: measured DRAM power for this level, if available (e.g. RAPL 'dram'
    or a vSphere RAMWatt reading). When given, the memory pool is that value,
    split by memory share (path B, measured); otherwise memory is modeled at
    MEM_W_PER_GB per GB (path A), and the remainder is the CPU pool.
    """
    # 0. validate inputs - reject non-finite / negative telemetry rather than
    #    producing nonsensical allocations downstream.
    problems = []
    if not _finite_nonneg(parent_total_w):
        problems.append(f"parent_total_w={parent_total_w} not finite/non-negative")
    for g in gpus:
        if not _finite_nonneg(g.watts):
            problems.append(f"gpu {g.gpu_id} watts={g.watts} invalid")
    for c in children:
        if not _finite_nonneg(c.cpu_share):
            problems.append(f"child {c.cid} cpu_share={c.cpu_share} invalid")
    if problems:
        return [], {"parent_total_w": parent_total_w, "attributed_w": 0.0,
                    "remainder_w": parent_total_w, "reconciled": False,
                    "error": "; ".join(problems)}

    # 1. GPU power flows first (whole where passthrough, split where shared)
    gpu_total = sum(g.watts for g in gpus)

    # guard: measured GPU power must not exceed the parent total. If it does the
    # two readings are incompatible (clock skew / bad reading) - surface it,
    # don't silently allocate more than the box drew.
    if gpu_total > parent_total_w + max(1.0, 0.02 * parent_total_w):
        return [], {"parent_total_w": parent_total_w, "attributed_w": 0.0,
                    "remainder_w": parent_total_w, "reconciled": False,
                    "error": (f"gpu_total {gpu_total:.0f}W exceeds parent "
                              f"{parent_total_w:.0f}W - incompatible telemetry")}

    gpu_to_child: dict[str, float] = {c.cid: 0.0 for c in children}
    gpu_conf_penalty: dict[str, float] = {c.cid: 1.0 for c in children}

    for g in gpus:
        if g.mode == "passthrough":
            if g.owner in gpu_to_child:
                gpu_to_child[g.owner] += g.watts       # whole GPU, exact
        else:  # mig / vgpu -> split by util share, ESTIMATED
            tot = sum(g.shares.values()) or 1e-9
            for cid, sh in g.shares.items():
                if cid in gpu_to_child:
                    gpu_to_child[cid] += g.watts * (sh / tot)
                    gpu_conf_penalty[cid] = 0.75       # shared-GPU estimate

    # 2. non-GPU power (CPU + RAM + fans + PSU loss) = parent total - GPU total
    non_gpu_w = max(parent_total_w - gpu_total, 0.0)

    # 3. carve the non-GPU envelope into a MEMORY pool and a CPU pool.
    #    Memory power is baseline (refresh) + activity, NOT flat-by-capacity, so
    #    an idle memory-heavy workload is not overcharged. Each child carries an
    #    activity-aware weight; the pool is split by that weight.
    #    B: a measured DRAM reading (dram_w) sizes the pool. A: the model does.
    def _mem_weight(c):
        gb = (c.mem_mb or 0.0) / 1024.0
        a = c.mem_active_frac if c.mem_active_frac is not None \
            else MEM_DEFAULT_ACTIVE_FRAC
        a = min(max(a, 0.0), 1.0)
        return gb * (MEM_W_PER_GB_IDLE + (MEM_W_PER_GB - MEM_W_PER_GB_IDLE) * a)

    mem_weights = {c.cid: _mem_weight(c) for c in children}
    total_mem_mb = sum((c.mem_mb or 0.0) for c in children)
    mem_aware = total_mem_mb > 0
    mem_method = None
    if mem_aware:
        if dram_w is not None and dram_w > 0:
            mem_pool = min(dram_w, non_gpu_w)
            mem_method = "measured"
        else:
            mem_pool = min(sum(mem_weights.values()), non_gpu_w)
            # "modeled" if any child had a real activity signal, else assumed
            mem_method = ("modeled" if any(c.mem_active_frac is not None
                                           for c in children)
                          else "modeled(assumed-activity)")
    else:
        mem_pool = 0.0
    cpu_pool = max(non_gpu_w - mem_pool, 0.0)

    share_sum = sum(c.cpu_share for c in children) or 1e-9
    mem_weight_sum = sum(mem_weights.values()) or 1e-9

    # map for flowing owned GPUs into nested levels
    gpu_map = {g.gpu_id: g for g in gpus}

    attributions = []
    for c in children:
        cpu_w = cpu_pool * (c.cpu_share / share_sum)
        mem_w = mem_pool * (mem_weights[c.cid] / mem_weight_sum) if mem_aware else 0.0
        shared_w = cpu_w                                # CPU/overhead portion
        gpu_w = gpu_to_child[c.cid]
        non_gpu_share = cpu_w + mem_w                   # proportional (estimated) part
        denom = gpu_w + non_gpu_share
        est_fraction = (non_gpu_share / denom) if denom else 0.0

        # confidence: exact for a pure passthrough-GPU child; drops with the
        # proportional (CPU+memory) portion, and only then is allocation-vs-util
        # a penalty (a child with no shared portion has no split to be unsure of).
        conf = 1.0 - 0.30 * est_fraction
        if non_gpu_share > 0 and c.cpu_share_method == "alloc":
            conf *= 0.9                                 # allocation < utilization
        conf *= gpu_conf_penalty[c.cid]                # shared-GPU penalty
        conf *= parent_confidence                      # error compounds down tree
        # "exact" ONLY when every part came from an exact source: a whole
        # passthrough GPU (no shared-GPU estimate), NO proportional CPU/memory
        # split, and an exact parent. Any estimated portion -> "estimated",
        # regardless of the aggregate confidence score (a high number does not
        # make a partly-estimated attribution exact).
        fully_exact = (non_gpu_share <= 1e-9
                       and gpu_conf_penalty[c.cid] >= 1.0
                       and parent_confidence >= 1.0)
        attribution = "exact" if fully_exact else "estimated"

        # 4. recurse - flow this child's OWNED passthrough GPUs down, re-owning
        #    each to the grandchild that lists it. Unclaimed GPU power surfaces
        #    as the nested remainder (honest), not silently dropped.
        nested, child_report = [], None
        if c.children:
            nested_gpus = []
            for g in gpus:
                if g.mode == "passthrough" and g.owner == c.cid:
                    gc = next((x for x in c.children if g.gpu_id in x.gpu_ids), None)
                    if gc:
                        nested_gpus.append(Gpu(g.gpu_id, g.watts, "passthrough",
                                               owner=gc.cid))
                    else:
                        nested_gpus.append(g)   # no pod claims it -> nested remainder
            nested, child_report = split_power(denom, nested_gpus, c.children,
                                               parent_confidence=conf)

        attributions.append(Attribution(c.cid, gpu_w, shared_w, attribution,
                                         conf, nested, child_report, mem_w=mem_w))

    # 5. reconcile (this level)
    attributed = sum(a.total_w for a in attributions)
    remainder = parent_total_w - attributed
    this_reconciled = abs(remainder) < max(1.0, 0.01 * parent_total_w)

    # nested reconciliation is surfaced, not swallowed: a child that failed to
    # reconcile makes the whole report not-fully-reconciled.
    nested_ok = all(
        (a.child_report is None) or a.child_report.get("all_reconciled", True)
        for a in attributions)

    report = {
        "parent_total_w": parent_total_w,
        "attributed_w": attributed,
        "remainder_w": remainder,
        "reconciled": this_reconciled,             # this level only
        "all_reconciled": this_reconciled and nested_ok,  # this level + descendants
        "mem_method": mem_method,                  # measured (DRAM) | modeled | None
    }
    return attributions, report


# --------------------------------------------------------------- render
def render(attrs: list[Attribution], report: dict, indent=2) -> str:
    pad = " " * indent
    L = [f"\n{pad}POWER ATTRIBUTION  (virtual servers)\n"]
    if report.get("error"):
        L.append(f"{pad}!! REJECTED: {report['error']}\n")
        return "\n".join(L)
    L.append(f"{pad}{'ENTITY':<12}{'GPU W':>8}{'CPU W':>8}{'MEM W':>8}"
             f"{'TOTAL W':>10}{'ATTRIB':>11}{'CONF':>7}")
    L.append(pad + "-" * 66)

    def walk(items, depth):
        for a in sorted(items, key=lambda x: x.total_w, reverse=True):
            name = ("  " * depth) + a.cid
            L.append(f"{pad}{name:<12}{a.gpu_w:>8.1f}{a.shared_w:>8.1f}"
                     f"{a.mem_w:>8.1f}{a.total_w:>10.1f}{a.attribution:>11}"
                     f"{a.confidence*100:>6.0f}%")
            walk(a.children, depth + 1)
    walk(attrs, 0)

    L.append(pad + "-" * 66)
    L.append(f"{pad}parent total : {report['parent_total_w']:.1f} W")
    L.append(f"{pad}attributed   : {report['attributed_w']:.1f} W")
    r = report["remainder_w"]
    tag = "reconciles" if report["reconciled"] else "!! GAP - check shares"
    L.append(f"{pad}remainder    : {r:.1f} W  ({tag})")
    if not report.get("all_reconciled", report["reconciled"]):
        L.append(f"{pad}!! a nested level did NOT reconcile - check the tree")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------- demo
if __name__ == "__main__":
    # Worked example: one host, 1150 W (BMC), 2 x 410W GPUs, 3 VMs.
    gpus = [
        Gpu("gpu0", 410, mode="passthrough", owner="VM-A"),
        Gpu("gpu1", 410, mode="passthrough", owner="VM-B"),
    ]
    vms = [
        Child("VM-A", cpu_share=0.50, cpu_share_method="util", gpu_ids=["gpu0"]),
        Child("VM-B", cpu_share=0.30, cpu_share_method="util", gpu_ids=["gpu1"]),
        Child("VM-C", cpu_share=0.20, cpu_share_method="util"),
    ]
    attrs, report = split_power(1150.0, gpus, vms)
    print(render(attrs, report))

    print("  --- shared GPU (MIG/vGPU) example ---")
    gpus2 = [Gpu("gpu0", 500, mode="mig",
                 shares={"pod-a": 0.5, "pod-b": 0.3, "pod-c": 0.2})]
    pods = [Child("pod-a", 0.4, "util"), Child("pod-b", 0.35, "util"),
            Child("pod-c", 0.25, "util")]
    a2, r2 = split_power(650.0, gpus2, pods)
    print(render(a2, r2))
