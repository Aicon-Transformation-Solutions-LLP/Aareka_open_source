#!/usr/bin/env python3
"""
IMPS beta  .  API + dashboard server

A thin FastAPI layer over the product's REAL logic. It does not reimplement
anything - it calls read_server_power() and split_power() and serializes the
result to JSON. The dashboard (served at /) polls these endpoints.

Endpoints
  GET  /                     the live dashboard (HTML)
  GET  /health               liveness probe
  GET  /api/v1/power         REAL power for THIS machine (node total + sources)
  GET  /api/v1/workloads     REAL per-process GPU attribution for THIS machine
  POST /api/v1/split         reasoning API: split a node's power across workloads
  POST /api/v1/rollup        reasoning API: roll workloads up to business units

The two GET endpoints read live hardware. The two POST endpoints are the pure
"send us your data, get attribution back" API - they never touch this machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Product logic (real).
import imp_server_power_source as sps
import imp_power_split as psplit
import imp_rollup as roll
from imps_beta import __version__
from imps_beta import gpu_workloads

app = FastAPI(
    title="IMPS beta",
    version=__version__,
    description="Self-hosted power attribution - reads only this machine; "
                "nothing leaves the host.",
)

_DASHBOARD = Path(__file__).parent / "dashboard.html"


# =============================================================== dashboard
@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return _DASHBOARD.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


# =============================================================== live reads
@app.get("/api/v1/power")
def power() -> JSONResponse:
    """REAL power for this machine, with honest measured/estimated split."""
    sp = sps.read_server_power()
    lo, hi = sp.est_total_range
    payload = {
        "gpu_tier": sp.gpu_tier,
        "cpu_tier": sp.cpu_tier,
        "measured_w": round(sp.measured_w, 1),
        "modeled_w": round(sp.modeled_w, 1),
        "total_w": round(sp.total_w, 1),
        "measured_fraction": round(sp.measured_fraction, 4),
        "has_node_total": sp.has_node_total,
        "low_confidence": sp.low_confidence,
        "est_total_low_w": round(lo, 1),
        "est_total_high_w": round(hi, 1),
        "warning": sp.warning,
        "note": sp.note,
        "sources": [{
            "entity_id": r.entity_id,
            "scope": r.scope,
            "watts": round(r.watts, 1),
            "method": r.method.value,
            "error_pct": r.error_pct,
            "tier": r.tier,
        } for r in sorted(sp.readings, key=lambda x: x.watts, reverse=True)],
    }
    return JSONResponse(payload)


@app.get("/api/v1/workloads")
def workloads() -> JSONResponse:
    """REAL per-process GPU attribution for this machine (empty if no GPU)."""
    return JSONResponse(gpu_workloads.attribute_gpu_power())


# =============================================================== reasoning API
# Pure "send data, get attribution" endpoints. They mirror the product
# dataclasses so an external caller can POST JSON and get structured results.
class GpuIn(BaseModel):
    gpu_id: str
    watts: float
    mode: str = "passthrough"
    owner: Optional[str] = None
    shares: dict[str, float] = Field(default_factory=dict)


class ChildIn(BaseModel):
    cid: str
    cpu_share: float
    cpu_share_method: str = "alloc"
    gpu_ids: list[str] = Field(default_factory=list)
    children: list["ChildIn"] = Field(default_factory=list)


class SplitIn(BaseModel):
    parent_total_w: float
    gpus: list[GpuIn] = Field(default_factory=list)
    children: list[ChildIn] = Field(default_factory=list)


ChildIn.model_rebuild()


def _to_child(c: ChildIn) -> psplit.Child:
    return psplit.Child(c.cid, c.cpu_share, c.cpu_share_method,
                        gpu_ids=list(c.gpu_ids),
                        children=[_to_child(x) for x in c.children])


def _attr_to_dict(a: psplit.Attribution) -> dict:
    return {
        "cid": a.cid,
        "gpu_w": round(a.gpu_w, 2),
        "shared_w": round(a.shared_w, 2),
        "total_w": round(a.total_w, 2),
        "attribution": a.attribution,
        "confidence": round(a.confidence, 3),
        "children": [_attr_to_dict(c) for c in a.children],
    }


@app.post("/api/v1/split")
def split_endpoint(body: SplitIn) -> JSONResponse:
    gpus = [psplit.Gpu(g.gpu_id, g.watts, g.mode, owner=g.owner,
                       shares=dict(g.shares)) for g in body.gpus]
    children = [_to_child(c) for c in body.children]
    attrs, report = psplit.split_power(body.parent_total_w, gpus, children)
    return JSONResponse({
        "attributions": [_attr_to_dict(a) for a in attrs],
        "report": report,
    })


class LeafIn(BaseModel):
    cid: str
    watts: float
    confidence: float = 1.0
    app: Optional[str] = None
    bu_label: Optional[str] = None


class RollupIn(BaseModel):
    leaves: list[LeafIn]
    bu_mapping: dict[str, dict[str, str]] = Field(default_factory=dict)


def _bucket_to_dict(b: roll.Bucket) -> dict:
    return {
        "name": b.label,
        "watts": round(b.watts, 1),
        "kw": round(b.watts / 1000, 3),
        "confidence": round(b.confidence, 3),
        "workloads": b.workloads,
        "children": [_bucket_to_dict(c) for c in
                     sorted(b.children.values(), key=lambda x: x.watts,
                            reverse=True)],
    }


@app.post("/api/v1/rollup")
def rollup_endpoint(body: RollupIn) -> JSONResponse:
    leaves = [roll.Leaf(l.cid, l.watts, l.confidence, app=l.app,
                        bu_label=l.bu_label) for l in body.leaves]
    root = roll.rollup_to_bu(leaves, body.bu_mapping)
    unmapped = root.children.get("unmapped")
    return JSONResponse({
        "total_watts": round(root.watts, 1),
        "confidence": round(root.confidence, 3),
        "workloads": root.workloads,
        "unmapped_watts": round(unmapped.watts, 1) if unmapped else 0.0,
        "tree": _bucket_to_dict(root),
    })
