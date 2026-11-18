#!/usr/bin/env python3
"""
IMP  .  GPU -> VM collector   (places the reads by virtualization mode)

Turns live GPU telemetry + a per-GPU mode map into the GpuAssignment list that
gpu_vm_power.attribute_gpu_to_vms consumes - reading each card from the RIGHT
place, because power is only visible in one of them:

  passthrough -> the card belongs to the guest; the HOST is blind to it. Read
                 nvidia-smi INSIDE the guest (power + util fully visible) and
                 give the whole card to that VM.
  vGPU        -> the GUEST is blind to power (NVIDIA masks it -> N/A). Read
                 HOST-SIDE: nvidia-smi for card power, and `nvidia-smi vgpu -q`
                 for each slice's VM + utilization; then split.

In production the mode map + passthrough owners come from vCenter. The
`nvidia-smi vgpu -q` parser is written to the documented format and validated
with a mock (there is no vGPU host to test against here); the passthrough path
is live.

  gpu_mode_map = {
    "0": {"mode": "passthrough", "owner": "VM-A"},
    "1": {"mode": "vgpu"},                 # slices/util come from nvidia-smi vgpu
  }
"""

from __future__ import annotations

import re
from typing import Optional

from imps_beta import gpu_workloads as gw
from imps_beta import gpu_vm_power as vmp


# --------------------------------------------------------------- vGPU host read
def vgpu_supported() -> bool:
    """True only on a host running the NVIDIA vGPU manager (nvidia-smi vgpu ok)."""
    return gw._run(["nvidia-smi", "vgpu"]) is not None


def parse_vgpu_q(text: str, bus_to_index: Optional[dict] = None) -> list[dict]:
    """
    Parse `nvidia-smi vgpu -q` into [{gpu, vm, util, fb_mb}], one per active
    vGPU. `gpu` is the physical GPU index if bus_to_index maps its PCI bus,
    else the raw bus string. Best-effort, mock-validated.
    """
    bus_to_index = bus_to_index or {}
    recs: list[dict] = []
    gpu_bus = None
    cur: Optional[dict] = None
    in_util = False
    in_fb = False

    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("GPU ") and ":" in s:          # physical GPU header
            gpu_bus = s.split(None, 1)[1].strip()
            continue
        if s.startswith("vGPU ID"):                    # new vGPU block
            if cur:
                recs.append(cur)
            cur = {"gpu": bus_to_index.get(gpu_bus, gpu_bus),
                   "vm": None, "util": 0.0, "fb_mb": 0.0}
            in_util = in_fb = False
            continue
        if cur is None:
            continue
        if s.startswith("VM Name"):
            cur["vm"] = s.split(":", 1)[1].strip()
        elif s.startswith("FB Memory Usage"):
            in_fb, in_util = True, False
        elif s.startswith("Utilization"):
            in_util, in_fb = True, False
        elif in_fb and s.startswith("Total"):
            m = re.search(r"(\d+)", s)
            if m:
                cur["fb_mb"] = float(m.group(1))
            in_fb = False
        elif in_util and s.startswith("Gpu"):
            m = re.search(r"(\d+)", s)
            if m:
                cur["util"] = float(m.group(1))
            in_util = False
    if cur:
        recs.append(cur)
    return [r for r in recs if r.get("vm")]


def _bus_to_index() -> dict:
    """Map PCI bus id -> GPU index via nvidia-smi (for vGPU host-side matching)."""
    out = gw._run(["nvidia-smi", "--query-gpu=index,pci.bus_id",
                   "--format=csv,noheader"])
    m = {}
    if out:
        for line in out.strip().splitlines():
            f = [c.strip() for c in line.split(",")]
            if len(f) >= 2:
                m[f[1]] = f[0]
    return m


def read_vgpu_instances(text: Optional[str] = None) -> list[dict]:
    """Read active vGPU instances host-side. Empty on a non-vGPU host."""
    if text is None:
        if not vgpu_supported():
            return []
        text = gw._run(["nvidia-smi", "vgpu", "-q"])
        if not text:
            return []
        return parse_vgpu_q(text, _bus_to_index())
    return parse_vgpu_q(text)


# --------------------------------------------------------------- assemble
def build_assignments(gpu_mode_map: dict, gpus: list,
                      vgpu_instances: list[dict]) -> list[vmp.GpuAssignment]:
    """Combine the mode map + live GPUs + vGPU instances into GpuAssignments."""
    vgpu_by_gpu: dict[str, list[dict]] = {}
    for inst in vgpu_instances:
        vgpu_by_gpu.setdefault(str(inst["gpu"]), []).append(inst)

    assignments = []
    for g in gpus:
        spec = gpu_mode_map.get(g.index)
        if not spec:
            continue
        mode = spec.get("mode")
        if mode == "passthrough":
            assignments.append(vmp.GpuAssignment(
                g.index, "passthrough", owner=spec.get("owner")))
        elif mode in ("vgpu", "mig"):
            insts = vgpu_by_gpu.get(g.index, [])
            total_fb = sum(i.get("fb_mb", 0.0) for i in insts) or 0.0
            vms = {}
            for i in insts:
                # slice allocation ~ VRAM/instance share; equal if FB unknown
                alloc = (i["fb_mb"] / total_fb) if total_fb > 0 else (1.0 / len(insts))
                vms[i["vm"]] = {"alloc": alloc, "util": i.get("util", 0.0)}
            assignments.append(vmp.GpuAssignment(g.index, mode, vgpu_vms=vms))
    return assignments


def collect(gpu_mode_map: dict,
            idle_floor_w: float = vmp.DEFAULT_IDLE_FLOOR_W,
            gpus: Optional[list] = None,
            vgpu_text: Optional[str] = None,
            vgpu_instances: Optional[list] = None) -> dict:
    """
    End to end: read GPUs (or use injected `gpus`), obtain vGPU slice data
    (pre-built `vgpu_instances`, else parsed from `vgpu_text`, else read
    host-side), build assignments, attribute.
    """
    if gpus is None:
        if not gw.nvidia_smi_available():
            return {"available": False, "reason": "nvidia-smi not found",
                    "gpus": [], "vms": [], "total_gpu_w": 0.0, "warnings": []}
        gpus = gw.read_gpus()

    needs_vgpu = any(v.get("mode") == "vgpu" for v in gpu_mode_map.values())
    if vgpu_instances is not None:
        vgpu = vgpu_instances
    elif needs_vgpu:
        vgpu = read_vgpu_instances(vgpu_text)
    else:
        vgpu = []

    assignments = build_assignments(gpu_mode_map, gpus, vgpu)
    return vmp.attribute_gpu_to_vms(assignments, idle_floor_w, gpus=gpus)
