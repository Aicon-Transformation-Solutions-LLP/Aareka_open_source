#!/usr/bin/env python3
"""
IMP  .  VMware host power -> per-VM   (the assembled chain)

Ties the VMware pieces into per-VM power, reconciled to the MEASURED host total:

  1. GPU -> VM   via gpu_vm_power (passthrough exact; vGPU baseline+dynamic).
  2. non-GPU     = host_power - total GPU watts, split across VMs by CPU + memory
                   (split_power, memory-aware: measured DRAM if given, else
                   MEM_W_PER_GB per GB).
  3. combine     -> per VM: gpu_w + cpu_w + mem_w, summing back to the host total.

Tiers fall out for free: a VM with no GPU is pure vCPU (gpu_w = 0); a GPU VM is
vCPU+GPU. Every watt is measured (host power counter + nvidia-smi); only the
allocation/topology (from vCenter) is provided.

Note: `gpu_ids` from vCenter are assumed aligned to nvidia-smi indices; in
production they are matched by PCI bus. host_power must be the measured vSphere
Power counter for the non-GPU split to reconcile.
"""

from __future__ import annotations

from typing import Optional

import imp_power_split as psplit
from imps_beta import gpu_workloads as gw
from imps_beta import gpu_vm_collector as col
from imps_beta import gpu_vm_power as vmp
from imps_beta import vcenter_connector as vc


def host_per_vm_power(host: "vc.HostInfo",
                      gpus: Optional[list] = None,
                      cpu_w: Optional[float] = None,
                      vgpu_text: Optional[str] = None,
                      vgpu_instances: Optional[list] = None,
                      dram_w: Optional[float] = None,
                      idle_floor_w: float = vmp.DEFAULT_IDLE_FLOOR_W) -> dict:
    warnings: list[str] = []

    # host-side GPU read sees ALL cards' power (the whole-box total includes them)
    if gpus is None:
        gpus = gw.read_gpus() if gw.nvidia_smi_available() else []
    raw_gpu_total = sum((g.watts or 0.0) for g in gpus)

    # 1. GPU -> VM
    mode_map = vc.VCenterConnector.gpu_mode_map(host)
    gpu_by_vm: dict[str, float] = {}
    if mode_map and gpus:
        gres = col.collect(mode_map, idle_floor_w=idle_floor_w, gpus=gpus,
                           vgpu_text=vgpu_text, vgpu_instances=vgpu_instances)
        warnings += gres.get("warnings", [])
        for v in gres.get("vms", []):
            gpu_by_vm[v["vm"]] = v["gpu_w"]
    attributed_gpu = sum(gpu_by_vm.values())
    unattributed_gpu = round(max(raw_gpu_total - attributed_gpu, 0.0), 1)
    if unattributed_gpu > 1.0:
        warnings.append(f"{unattributed_gpu} W of GPU power not attributed to a "
                        f"VM (unassigned or unreadable card)")

    # 1b. Validate the vCenter host-power counter against measured GPU.
    #     power.power.average can silently omit GPU power (PCIe expansion
    #     chassis / external enclosures / whitebox software-estimate). If the
    #     counter reads below the measured GPU draw it is NOT a whole-box
    #     reading - distrust it and use measured components (GPU + CPU) instead.
    counter = host.power_w
    component_total = raw_gpu_total + (cpu_w or 0.0)
    tol = max(1.0, 0.02 * raw_gpu_total)
    if counter is None:
        host_total = component_total if (raw_gpu_total > 0 or cpu_w) else None
        power_source = "components" if host_total is not None else "none"
        if host_total is None:
            warnings.append("no host power available (no vCenter counter, no "
                            "component readings) - cannot split non-GPU power")
    elif counter + tol < raw_gpu_total:
        host_total = max(component_total, raw_gpu_total)
        power_source = "components(vcenter-counter-distrusted)"
        warnings.append(
            f"vCenter host power {counter:.0f} W is below measured GPU "
            f"{raw_gpu_total:.0f} W - the counter omits GPU power "
            f"(PCIe/whitebox); using measured components as host truth")
    else:
        host_total = counter
        power_source = "vcenter"

    # 2. non-GPU split (CPU + memory), memory-aware
    cpu_by_vm: dict[str, float] = {}
    mem_by_vm: dict[str, float] = {}
    mem_method = None
    non_gpu_w: Optional[float] = None
    if host_total is None:
        warnings.append("host power unavailable - CPU/memory split skipped; "
                        "showing GPU attribution only")
    else:
        non_gpu_w = max(host_total - raw_gpu_total, 0.0)
        children = [psplit.Child(vm.name, cpu_share=float(vm.vcpu),
                                 cpu_share_method="alloc", mem_mb=vm.mem_mb)
                    for vm in host.vms]
        attrs, rep = psplit.split_power(non_gpu_w, [], children, dram_w=dram_w)
        mem_method = rep.get("mem_method")
        for a in attrs:
            cpu_by_vm[a.cid] = a.shared_w
            mem_by_vm[a.cid] = a.mem_w

    # 3. combine -> per VM
    per_vm = []
    for vm in host.vms:
        g = round(gpu_by_vm.get(vm.name, 0.0), 1)
        c = round(cpu_by_vm.get(vm.name, 0.0), 1)
        m = round(mem_by_vm.get(vm.name, 0.0), 1)
        per_vm.append({
            "vm": vm.name,
            "tier": "vcpu+gpu" if (g > 0 or vm.gpu_mode) else "vcpu",
            "gpu_mode": vm.gpu_mode,
            "gpu_w": g, "cpu_w": c, "mem_w": m,
            "total_w": round(g + c + m, 1),
        })
    per_vm.sort(key=lambda x: x["total_w"], reverse=True)

    accounted = sum(x["total_w"] for x in per_vm) + unattributed_gpu
    reconciled = (host_total is not None and
                  abs(accounted - host_total) < max(1.0, 0.01 * host_total))

    return {
        "host": host.name,
        "host_power_w": None if host_total is None else round(host_total, 1),
        "host_power_source": power_source,       # vcenter | components | distrusted
        "vcenter_counter_w": counter,            # raw counter, for transparency
        "host_power_measured": power_source in ("vcenter", "components"),
        "gpu_total_w": round(raw_gpu_total, 1),
        "unattributed_gpu_w": unattributed_gpu,
        "non_gpu_w": None if non_gpu_w is None else round(non_gpu_w, 1),
        "mem_method": mem_method,
        "reconciled": reconciled,
        "warnings": warnings,
        "vms": per_vm,
    }


# --------------------------------------------------------------- render
def render(result: dict) -> str:
    L = [f"\n  VMWARE HOST POWER -> PER VM   ({result['host']})\n"]
    hp = result["host_power_w"]
    L.append(f"  host power : {hp} W  (source: {result.get('host_power_source', '?')}"
             f", vCenter counter: {result.get('vcenter_counter_w')})")
    L.append(f"  gpu total  : {result['gpu_total_w']} W    "
             f"non-GPU: {result['non_gpu_w']} W    "
             f"memory: {result['mem_method']}")
    L.append("  " + "-" * 68)
    L.append(f"  {'VM':<16}{'TIER':<11}{'GPU W':>8}{'CPU W':>8}{'MEM W':>8}"
             f"{'TOTAL W':>10}")
    L.append("  " + "-" * 68)
    for v in result["vms"]:
        L.append(f"  {v['vm']:<16}{v['tier']:<11}{v['gpu_w']:>8.1f}"
                 f"{v['cpu_w']:>8.1f}{v['mem_w']:>8.1f}{v['total_w']:>10.1f}")
    L.append("  " + "-" * 68)
    total = sum(v["total_w"] for v in result["vms"])
    L.append(f"  {'attributed':<16}{'':<11}{'':>8}{'':>8}{'':>8}{total:>10.1f}"
             f"   ({'reconciles' if result['reconciled'] else 'GAP'})")
    for w in result["warnings"]:
        L.append(f"  !! {w}")
    L.append("")
    return "\n".join(L)
