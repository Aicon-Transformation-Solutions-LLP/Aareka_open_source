#!/usr/bin/env python3
"""
IMP  .  GPU -> VM power attribution   (virtualization-aware)

Maps a physical GPU's MEASURED power to the VM(s) that use it, honoring how the
card is virtualized:

  * passthrough (DirectPath I/O): the whole card belongs to one VM.
      -> that VM gets 100% of the card's watts.  EXACT.
      (reuses imp_power_split.split_power's passthrough mode, already exact)

  * vGPU (shared / sliced): several VMs share one physical card.
      A card can't sleep while any tenant is busy, so its power is split as:
        - IDLE BASELINE  -> by profile allocation (they all keep it awake)
        - DYNAMIC (above idle) -> by measured per-vGPU utilization
      ESTIMATED. Falls back to allocation-only split when all tenants are idle.

GPU wattage is always MEASURED (nvidia-smi / DCGM). Only the assignment - which
VM owns which card / vGPU profile - is provided (a topology fact read from
vCenter in production; never inferred). vCenter reports the mode per VM, so the
caller routes each card to the right branch.

Read placement (why this is virtualization-aware):
  * vGPU     -> the NVIDIA vGPU host driver runs on ESXi, so nvidia-smi is read
                HOST-SIDE (this module's default read).
  * passthrough -> the host is blind to the card; nvidia-smi is read INSIDE the
                guest VM. Same code, run in the guest; the card's watts map 1:1
                to that VM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import imp_power_split as psplit
import gpu_workloads as gw

# Enterprise-GPU idle floor (watts) used to size the shared-card baseline that
# is split by allocation. Article-grounded default (~25-50 W idle); a per-card
# value can be passed in. This is a model input, not a measurement - adjustable.
DEFAULT_IDLE_FLOOR_W = 30.0

# vGPU (time-sliced) attribution is a proportional estimate, not a per-slice
# measurement. MIG is spatially partitioned (dedicated SMs/memory per instance),
# so per-instance apportionment is more defensible -> higher confidence than
# time-sliced vGPU.
VGPU_CONFIDENCE = 0.75
MIG_CONFIDENCE = 0.85


@dataclass
class GpuAssignment:
    """How one physical GPU is handed to VMs (from vCenter in production)."""
    gpu_id: str                      # matches nvidia-smi index
    mode: str                        # "passthrough" | "vgpu" | "mig"
    owner: Optional[str] = None      # passthrough: the single owning VM
    # vgpu/mig: {vm_name: {"alloc": slice_share, "util": 0..100}}
    vgpu_vms: dict = field(default_factory=dict)


@dataclass
class _VmGpu:
    vm: str
    watts: float = 0.0
    method: str = "measured"         # "measured" (exact) | "estimated"
    confidence: float = 1.0
    gpus: list = field(default_factory=list)


def _vgpu_split(card_watts: float, vms: dict, idle_floor_w: float) -> dict:
    """Split one shared card's watts across its vGPU VMs.
    baseline (idle floor) by allocation, dynamic (above idle) by utilization."""
    total_alloc = sum(v.get("alloc", 0.0) for v in vms.values()) or 1e-9
    total_util = sum(v.get("util", 0.0) for v in vms.values())
    baseline_pool = min(idle_floor_w, card_watts)
    dynamic_pool = max(card_watts - baseline_pool, 0.0)

    out = {}
    for vm, v in vms.items():
        base = baseline_pool * (v.get("alloc", 0.0) / total_alloc)
        if total_util > 0:
            dyn = dynamic_pool * (v.get("util", 0.0) / total_util)
        else:                        # every tenant idle -> baseline behaviour
            dyn = dynamic_pool * (v.get("alloc", 0.0) / total_alloc)
        out[vm] = base + dyn
    return out


def attribute_gpu_to_vms(assignments: list[GpuAssignment],
                         idle_floor_w: float = DEFAULT_IDLE_FLOOR_W,
                         gpus: Optional[list] = None) -> dict:
    """
    Attribute REAL per-GPU power to VMs per the assignment. `gpus` may be
    injected (list of gpu_workloads.GpuInfo) for testing; otherwise read live.
    """
    if gpus is None:
        if not gw.nvidia_smi_available():
            return {"available": False, "reason": "nvidia-smi not found",
                    "gpus": [], "vms": [], "total_gpu_w": 0.0,
                    "idle_floor_w": idle_floor_w}
        gpus = gw.read_gpus()
    if not gpus:
        return {"available": False, "reason": "no NVIDIA GPU detected",
                "gpus": [], "vms": [], "total_gpu_w": 0.0,
                "idle_floor_w": idle_floor_w}

    by_index = {g.index: g for g in gpus}
    vms: dict[str, _VmGpu] = {}
    gpu_rows = []
    warnings: list[str] = []
    total_w = 0.0

    for a in assignments:
        g = by_index.get(a.gpu_id)
        if g is None:
            warnings.append(f"gpu {a.gpu_id}: assigned but not in the GPU read")
            continue

        # Determine measured card watts, honoring the read-placement rule.
        if a.mode in ("vgpu", "mig"):
            # vGPU/MIG power is readable ONLY host-side. A None here means the
            # read came from a guest, where NVIDIA masks physical power (N/A).
            # Do NOT model it from the guest's isolated util - flag it instead.
            if g.watts is None:
                warnings.append(
                    f"gpu {g.index}: {a.mode} power unreadable here (N/A) - read "
                    f"host-side (nvidia-smi on the ESXi host), not in the guest")
                gpu_rows.append({"gpu_id": g.index, "name": g.name, "watts": None,
                                 "util": g.util, "mode": a.mode, "measured": False,
                                 "unreadable": True, "vms": []})
                continue
            watts, measured = g.watts, True
        else:  # passthrough - guest sees the whole card, power included
            if g.watts is not None:
                watts, measured = g.watts, True
            elif g.limit_w is not None and g.util is not None:
                watts, measured = g.limit_w * (g.util / 100.0), False   # hidden draw
            else:
                warnings.append(
                    f"gpu {g.index}: passthrough power unreadable - read "
                    f"nvidia-smi INSIDE the owning guest VM (host is blind to it)")
                gpu_rows.append({"gpu_id": g.index, "name": g.name, "watts": None,
                                 "util": g.util, "mode": a.mode, "measured": False,
                                 "unreadable": True, "vms": []})
                continue
        total_w += watts

        row = {"gpu_id": g.index, "name": g.name, "watts": round(watts, 1),
               "util": g.util, "mode": a.mode, "measured": measured, "vms": []}

        if a.mode == "passthrough" and a.owner:
            # reuse the product splitter: passthrough is exact
            gpu_obj = psplit.Gpu(g.index, watts, "passthrough", owner=a.owner)
            child = psplit.Child(a.owner, cpu_share=1.0, cpu_share_method="util")
            attrs, _ = psplit.split_power(watts, [gpu_obj], [child])
            w = attrs[0].gpu_w if attrs else watts
            vm = vms.setdefault(a.owner, _VmGpu(a.owner))
            vm.watts += w
            vm.gpus.append(g.index)
            # passthrough exact only if the card power itself is measured
            if not measured:
                vm.method, vm.confidence = "estimated", min(vm.confidence, 0.75)
            row["vms"].append({"vm": a.owner, "watts": round(w, 1),
                               "share": "whole card"})

        elif a.mode in ("vgpu", "mig") and a.vgpu_vms:
            # shared card: baseline by allocation + dynamic by per-tenant
            # activity. MIG is spatial (per-instance) -> more defensible than
            # time-sliced vGPU, so higher confidence.
            conf = MIG_CONFIDENCE if a.mode == "mig" else VGPU_CONFIDENCE
            split = _vgpu_split(watts, a.vgpu_vms, idle_floor_w)
            for vm_name, w in split.items():
                vm = vms.setdefault(vm_name, _VmGpu(vm_name))
                vm.watts += w
                vm.gpus.append(g.index)
                vm.method = "estimated"
                vm.confidence = min(vm.confidence, conf)
                row["vms"].append({"vm": vm_name, "watts": round(w, 1),
                                   "share": a.mode})
        gpu_rows.append(row)

    vm_list = sorted(vms.values(), key=lambda v: v.watts, reverse=True)
    return {
        "available": True,
        "reason": "",
        "idle_floor_w": idle_floor_w,
        "total_gpu_w": round(total_w, 1),
        "warnings": warnings,
        "gpus": gpu_rows,
        "vms": [{"vm": v.vm, "gpu_w": round(v.watts, 1), "method": v.method,
                 "confidence": round(v.confidence, 3), "gpus": v.gpus}
                for v in vm_list],
    }
