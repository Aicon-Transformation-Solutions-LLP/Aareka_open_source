#!/usr/bin/env python3
"""
Aareka  .  Token efficiency engine   (processor)

Turns the collector's raw inference measurements (GPU/CPU joules + token deltas
per send window) into energy per token:

    1. allocate   - each GPU's joules go to the inference endpoints bound to it,
                    split by output tokens when several endpoints share a GPU;
                    within an endpoint, split across models by output tokens.
                    CPU joules are split across all endpoints by output tokens.
    2. summarise  - per model (and per department): joules per output token,
                    output tokens per joule (= tokens/s per watt), busy-only
                    joules per token, throughput, average watts, and cost per
                    million output tokens at the org's tariff.

Every figure is a LOWER BOUND: the collector measures GPU (and CPU where it
can), not memory, fans, power-supply losses or cooling.
"""

from __future__ import annotations


def allocate_window(inf: dict) -> list[dict]:
    """One collector `inference` block -> one row per (endpoint, model) with the
    energy allocated to it. Endpoints that produced no tokens get no energy; the
    window's unallocated GPU energy is reported on a row with model=None."""
    gpu_j = {g: float(j) for g, j in (inf.get("gpu_energy_j") or {}).items()}
    cpu_j = inf.get("cpu_energy_j")
    eps = inf.get("endpoints") or []

    out_by_ep = [sum(max(0, m.get("output_tokens", 0)) for m in ep.get("models") or [])
                 for ep in eps]
    # output tokens per GPU, from every endpoint bound to it
    gpu_tokens = {g: 0.0 for g in gpu_j}
    for ep, toks in zip(eps, out_by_ep):
        for g in ep.get("gpu_ids") or []:
            if g in gpu_tokens:
                gpu_tokens[g] += toks
    total_out = sum(out_by_ep)

    rows, allocated = [], 0.0
    for ep, toks in zip(eps, out_by_ep):
        ep_gpu_j = 0.0
        for g in ep.get("gpu_ids") or []:
            if g in gpu_j and gpu_tokens[g] > 0:
                ep_gpu_j += gpu_j[g] * toks / gpu_tokens[g]
        ep_cpu_j = (float(cpu_j) * toks / total_out) if (cpu_j is not None and total_out) else None
        busy_s = float(ep.get("busy_s") or 0.0)
        busy_j = float(ep.get("busy_gpu_energy_j") or 0.0)
        models = ep.get("models") or []
        for m in models:
            mt = max(0, m.get("output_tokens", 0))
            share = (mt / toks) if toks else 0.0
            rows.append({
                "endpoint": ep.get("url"), "server": ep.get("server"),
                "model": m.get("model"), "department": m.get("department"),
                "prompt_tokens": max(0, m.get("prompt_tokens", 0)),
                "output_tokens": mt,
                "gpu_energy_j": ep_gpu_j * share,
                "cpu_energy_j": (ep_cpu_j * share) if ep_cpu_j is not None else None,
                "busy_s": busy_s * share,
                "busy_gpu_energy_j": busy_j * share,
            })
        allocated += ep_gpu_j
    idle = sum(gpu_j.values()) - allocated
    if idle > 1e-6:
        rows.append({"endpoint": None, "server": None, "model": None, "department": None,
                     "prompt_tokens": 0, "output_tokens": 0, "gpu_energy_j": idle,
                     "cpu_energy_j": None, "busy_s": 0.0, "busy_gpu_energy_j": 0.0})
    return rows


def _ratios(out_tok, energy_j, busy_j, busy_s, window_s, tariff):
    r = {
        "output_tokens": int(out_tok),
        "energy_j": round(energy_j, 1),
        "j_per_output_token": (energy_j / out_tok) if out_tok else None,
        "output_tokens_per_joule": (out_tok / energy_j) if energy_j > 0 and out_tok else None,
        "busy_j_per_output_token": (busy_j / out_tok) if out_tok and busy_j > 0 else None,
        "output_tokens_per_s": (out_tok / busy_s) if busy_s > 0 else None,
        "avg_busy_w": (busy_j / busy_s) if busy_s > 0 else None,
        "busy_s": round(busy_s, 1),
        "window_s": round(window_s, 1),
    }
    if tariff and r["j_per_output_token"] is not None:
        # J/token * 1e6 tokens / 3.6e6 J per kWh * price per kWh
        r["cost_per_million_output_tokens"] = r["j_per_output_token"] * 1e6 / 3.6e6 * tariff
    return r


def summarise(rows: list[dict], tariff: float | None = None) -> dict:
    """Stored allocation rows -> totals, per model and per department."""
    def agg(key):
        groups = {}
        for r in rows:
            if r.get("model") is None:
                continue
            k = key(r)
            g = groups.setdefault(k, {"out": 0, "prompt": 0, "j": 0.0, "bj": 0.0, "bs": 0.0,
                                      "ws": 0.0, "host": set(), "server": set()})
            g["out"] += r["output_tokens"]
            g["prompt"] += r["prompt_tokens"]
            g["j"] += r["gpu_energy_j"] + (r.get("cpu_energy_j") or 0.0)
            g["bj"] += r["busy_gpu_energy_j"]
            g["bs"] += r["busy_s"]
            g["ws"] += r.get("window_s") or 0.0
            if r.get("host"):
                g["host"].add(r["host"])
            if r.get("server"):
                g["server"].add(r["server"])
        return groups

    by_model = []
    for (model, host), g in agg(lambda r: (r["model"], r.get("host"))).items():
        row = {"model": model, "host": host, "server": ", ".join(sorted(g["server"])),
               "prompt_tokens": g["prompt"]}
        row.update(_ratios(g["out"], g["j"], g["bj"], g["bs"], g["ws"], tariff))
        by_model.append(row)
    by_model.sort(key=lambda r: r["output_tokens"], reverse=True)

    by_department = []
    for dept, g in agg(lambda r: r.get("department") or "unmapped").items():
        row = {"department": dept, "prompt_tokens": g["prompt"]}
        row.update(_ratios(g["out"], g["j"], g["bj"], g["bs"], g["ws"], tariff))
        by_department.append(row)
    by_department.sort(key=lambda r: r["output_tokens"], reverse=True)

    out = sum(r["output_tokens"] for r in rows if r.get("model") is not None)
    used = sum(r["gpu_energy_j"] + (r.get("cpu_energy_j") or 0.0)
               for r in rows if r.get("model") is not None)
    unalloc = sum(r["gpu_energy_j"] for r in rows if r.get("model") is None)
    bj = sum(r["busy_gpu_energy_j"] for r in rows if r.get("model") is not None)
    bs = sum(r["busy_s"] for r in rows if r.get("model") is not None)
    totals = {"prompt_tokens": sum(r["prompt_tokens"] for r in rows)}
    totals.update(_ratios(out, used, bj, bs, 0.0, tariff))
    totals["unallocated_gpu_energy_j"] = round(unalloc, 1)
    totals["cpu_measured"] = any(r.get("cpu_energy_j") is not None for r in rows)
    totals.pop("window_s")
    return {
        "totals": totals,
        "by_model": by_model,
        "by_department": by_department,
        "lower_bound": True,
        "tariff_per_kwh": tariff,
    }
