#!/usr/bin/env python3
"""
IMPS beta  .  REAL per-process GPU workload attribution (single node)

This is the part that makes a prospect believe the product: it takes the
machine's REAL per-GPU power (nvidia-smi, measured) and splits it across the
REAL processes running on each GPU - reusing the product's own splitter
(imp_power_split.split_power), so the engine you are selling runs on live
telemetry, not a demo fixture.

Honesty ladder (mirrors the product's philosophy):
  * a GPU with exactly ONE compute process  -> passthrough, attribution EXACT
  * a GPU shared by 2+ processes             -> split by GPU-memory share,
                                                attribution ESTIMATED (memory is
                                                a proxy for power, not power)
  * no NVIDIA GPU / nvidia-smi missing        -> empty result, flagged; the
                                                dashboard says so plainly.

Nothing here fabricates a workload. If nvidia-smi reports no compute apps,
the result is empty and honest.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

# Reuse the product's real splitter and GPU model.
import imp_power_split as split


# --------------------------------------------------------------- raw reads
def _run(cmd: list[str]) -> Optional[str]:
    """Run a read-only command; return stdout or None on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _num(x: str) -> Optional[float]:
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


@dataclass
class GpuInfo:
    index: str
    uuid: str
    watts: Optional[float]        # measured power.draw, or None if hidden
    limit_w: Optional[float]
    util: Optional[float]         # 0..100
    name: str


@dataclass
class GpuProcess:
    pid: str
    name: str
    gpu_uuid: str
    used_mem_mib: float


def read_gpus() -> list[GpuInfo]:
    """Per-GPU identity + measured power via nvidia-smi. Empty if none."""
    out = _run(["nvidia-smi",
                "--query-gpu=index,uuid,power.draw,power.limit,utilization.gpu,name",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus = []
    for line in out.strip().splitlines():
        f = [c.strip() for c in line.split(",")]
        if len(f) < 6:
            continue
        gpus.append(GpuInfo(index=f[0], uuid=f[1], watts=_num(f[2]),
                            limit_w=_num(f[3]), util=_num(f[4]), name=f[5]))
    return gpus


def read_gpu_processes() -> list[GpuProcess]:
    """Real compute processes per GPU via nvidia-smi. Empty if none/unsupported."""
    out = _run(["nvidia-smi",
                "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits"])
    if not out:
        return []
    procs = []
    for line in out.strip().splitlines():
        f = [c.strip() for c in line.split(",")]
        if len(f) < 4:
            continue
        mem = _num(f[3]) or 0.0
        procs.append(GpuProcess(pid=f[0], name=f[1], gpu_uuid=f[2],
                                used_mem_mib=mem))
    return procs


# --------------------------------------------------------------- attribution
@dataclass
class WorkloadPower:
    pid: str
    name: str
    watts: float
    method: str            # "measured" | "estimated"
    confidence: float      # 0..1
    gpus: list             # gpu indexes this process touches
    idle: bool = False     # true for an idle GPU with no compute process


def attribute_gpu_power() -> dict:
    """
    Attribute REAL per-GPU power across REAL processes, per GPU, using the
    product splitter. Merge a process's slices across GPUs into one workload.

    Returns a dict the API/dashboard render directly - always honest about
    what was and was not measured.
    """
    if not nvidia_smi_available():
        return {"available": False, "reason": "nvidia-smi not found on this host",
                "gpus": [], "workloads": [], "gpu_total_w": 0.0}

    gpus = read_gpus()
    if not gpus:
        return {"available": False, "reason": "no NVIDIA GPU detected",
                "gpus": [], "workloads": [], "gpu_total_w": 0.0}

    procs = read_gpu_processes()
    procs_by_uuid: dict[str, list[GpuProcess]] = {}
    for p in procs:
        procs_by_uuid.setdefault(p.gpu_uuid, []).append(p)

    # merge per-process attribution across GPUs
    merged: dict[str, WorkloadPower] = {}
    gpu_total_w = 0.0
    gpu_rows = []

    for g in gpus:
        # measured power for this GPU; if the board hides power.draw, model it
        if g.watts is not None:
            watts = g.watts
            gpu_measured = True
        elif g.limit_w is not None and g.util is not None:
            watts = g.limit_w * (g.util / 100.0)
            gpu_measured = False
        else:
            watts = 0.0
            gpu_measured = False
        gpu_total_w += watts
        gpu_procs = procs_by_uuid.get(g.uuid, [])
        gpu_rows.append({
            "index": g.index, "name": g.name, "watts": round(watts, 1),
            "measured": gpu_measured, "util": g.util,
            "processes": len(gpu_procs),
        })

        if not gpu_procs or watts <= 0:
            if watts > 0:
                # Idle GPU: powered but running no compute process. Its watts are
                # real, wasted power - emit an idle workload so totals reconcile
                # to gpu_total and idle-vs-active stays honest (never dropped).
                merged[f"idle:gpu{g.index}"] = WorkloadPower(
                    pid=f"idle:gpu{g.index}", name="(idle GPU)", watts=watts,
                    method=("measured" if gpu_measured else "estimated"),
                    confidence=1.0 if gpu_measured else 0.75,
                    gpus=[g.index], idle=True)
            continue

        if len(gpu_procs) == 1:
            # sole tenant -> whole-GPU passthrough, EXACT split
            p = gpu_procs[0]
            gpu_obj = split.Gpu(g.index, watts, mode="passthrough", owner=p.pid)
            children = [split.Child(p.pid, cpu_share=1.0, cpu_share_method="util")]
        else:
            # shared GPU -> split by memory share, ESTIMATED (reuses MIG path)
            total_mem = sum(pp.used_mem_mib for pp in gpu_procs) or 1e-9
            shares = {pp.pid: (pp.used_mem_mib / total_mem) for pp in gpu_procs}
            gpu_obj = split.Gpu(g.index, watts, mode="mig", shares=shares)
            children = [split.Child(pp.pid, cpu_share=shares[pp.pid],
                                    cpu_share_method="util") for pp in gpu_procs]

        # parent_total == gpu watts so ALL of it is GPU power (no CPU tail here)
        attrs, _ = split.split_power(watts, [gpu_obj], children)
        name_by_pid = {pp.pid: pp.name for pp in gpu_procs}
        for a in attrs:
            w = merged.get(a.cid)
            if w is None:
                merged[a.cid] = WorkloadPower(
                    pid=a.cid, name=name_by_pid.get(a.cid, a.cid),
                    watts=a.gpu_w, method=("measured" if gpu_measured and
                                           a.attribution == "exact" else "estimated"),
                    confidence=a.confidence, gpus=[g.index])
            else:
                w.watts += a.gpu_w
                w.gpus.append(g.index)
                # weakest link wins for method + confidence
                if a.attribution != "exact" or not gpu_measured:
                    w.method = "estimated"
                w.confidence = min(w.confidence, a.confidence)

    workloads = sorted(merged.values(), key=lambda w: w.watts, reverse=True)
    attributed_w = sum(w.watts for w in merged.values())
    unattributed_w = round(max(gpu_total_w - attributed_w, 0.0), 1)
    reconciled = abs(attributed_w - gpu_total_w) < max(1.0, 0.01 * gpu_total_w)
    if unattributed_w > 1.0:
        # a residual the buckets did not capture (rare) - surfaced, not hidden
        workloads.append(WorkloadPower(
            pid="unattributed", name="(unattributed GPU power)",
            watts=unattributed_w, method="measured", confidence=0.0,
            gpus=[], idle=True))
    return {
        "available": True,
        "reason": "",
        "gpu_total_w": round(gpu_total_w, 1),
        "attributed_w": round(attributed_w, 1),
        "unattributed_w": unattributed_w,
        "reconciled": reconciled,
        "active_w": round(sum(w.watts for w in merged.values() if not w.idle), 1),
        "idle_w": round(sum(w.watts for w in merged.values() if w.idle), 1),
        "gpus": gpu_rows,
        "workloads": [{
            "pid": w.pid, "name": w.name, "watts": round(w.watts, 1),
            "method": w.method, "confidence": round(w.confidence, 3),
            "gpus": w.gpus, "idle": w.idle,
        } for w in workloads],
    }


# --------------------------------------------------------------- cli
if __name__ == "__main__":
    import json
    print(json.dumps(attribute_gpu_power(), indent=2))
