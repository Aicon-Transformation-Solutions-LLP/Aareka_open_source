#!/usr/bin/env python3
"""
Aareka  .  Ingest API   (processor)

The "mailbox" from the reference architecture. Customer collectors POST their
power readings AND the ownership map (topology + tags) here. This service:

    1. checks the caller's org key            (is this a customer we know?)
    2. stamps the org id server-side          (from the key - never the body)
    3. validates the batch                     (finite watts, known methods, ...)
    4. ATTRIBUTES it with the real engine      (split_power -> rollup_to_bu)
    5. stores raw readings + attributed result (org id on every row)
    6. replies with a receipt + the attribution

Every read is filtered by org id, so no tenant can ever see another's data.

The batch carries three things (the data contract):
    (A) power     - node_total_w, gpus[] (watts + mode + owner/shares), cpu_readings[]
    (B) topology  - workloads[] with cpu_share, mem_mb, gpu_ids   (who runs where)
    (C) tags      - department on each workload                    (who owns it)

Endpoints
    GET  /health          liveness probe (no auth)
    POST /v1/signup       email + password (+ optional key) -> new org & key, or join; token
    POST /v1/login        email + password -> signed token (humans view dashboard)
    POST /v1/ingest       receive a batch -> attribute -> store   (auth: org key)
    GET  /v1/readings     recent raw component readings, org-scoped (key or token)
    GET  /v1/attribution  recent per-workload attributed power, org-scoped (key or token)
    GET  /v1/rollup       power by department (latest snapshot per host) (key or token)
    GET  /v1/tokens       energy per token by model / department, last N hours (key or token)
    GET  /v1/waste        GPU states, idle spells, idle GPU-hours / energy / cost, last N days (key or token)
    GET  /v1/settings     the account's rates and waste thresholds, with defaults (key or token)
    PUT  /v1/settings     save them; a threshold change needs confirm_recalculation (token)
    GET  /v1/snapshot     the latest power per workload for each server (key or token)

Two ways to run it
    * Single site (the default). One organisation runs its own processor. There
      are no accounts: the dashboard opens straight to the data, and everything
      collectors send belongs to this site. Collectors still need the site's
      collector key, which is printed when the processor starts (or set it with
      AAREKA_COLLECTOR_KEY). Anyone who can reach the processor's address can
      see the dashboard, so keep it on an internal network or behind your own
      proxy / single sign-on.
    * Accounts (AAREKA_ACCOUNTS=1). Many organisations share one processor, each
      with its own sign-up, login and collector key - for a hosted service.
    GET  /v1/org          the caller's org + its collector key (key or token)

Security
    * The org-key -> org-id mapping is a SECRET, from the environment
      (AAREKA_API_KEYS as JSON) or a keys file. Never hard-coded.
    * org id is derived from the key ON THE SERVER, never from the body.

Storage
    Local SQLite by default (zero setup). For production set AAREKA_DB_PATH, or
    swap the Storage class for Postgres/RDS - the schema and the "always filter
    by org_id" rule are identical.

Run
    pip install fastapi "uvicorn[standard]"
    export AAREKA_API_KEYS='{"key-for-customer-a":"org_a"}'
    uvicorn ingest_api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

# The real engine (already in this folder) - the brains, unchanged.
import imp_power_split as psplit
import imp_rollup as roll
import gpu_vm_power as vmp          # virtualization-aware GPU attribution (vGPU/MIG)
import gpu_workloads as gw          # GpuInfo type used by gpu_vm_power
import imp_token_efficiency as tok  # energy per token for inference windows
import gpu_waste as waste           # GPU states, idle spells, idle GPU-hours

log = logging.getLogger("aareka.ingest")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

ALLOWED_METHODS = {"measured", "modeled"}
ALLOWED_MODES = {"passthrough", "vgpu", "mig"}


# ------------------------------------------------------------ config / secrets
# Single site unless AAREKA_ACCOUNTS turns accounts on (see the module docstring).
ACCOUNTS = os.environ.get("AAREKA_ACCOUNTS", "").strip().lower() in ("1", "true", "yes", "on")
SITE_ORG = "site"


def load_keys() -> dict:
    """Return {api_key: org_id}. From AAREKA_API_KEYS (JSON) or a keys file."""
    raw = os.environ.get("AAREKA_API_KEYS")
    if raw:
        try:
            return dict(json.loads(raw))
        except (ValueError, TypeError):
            log.error("AAREKA_API_KEYS is not valid JSON - no keys loaded")
            return {}
    path = os.environ.get("AAREKA_KEYS_FILE", "keys.json")
    if Path(path).exists():
        try:
            return dict(json.loads(Path(path).read_text()))
        except (ValueError, TypeError, OSError):
            log.error("could not read keys file %s", path)
    if ACCOUNTS:
        log.info("no static API keys configured; accounts create their own keys at sign-up")
    return {}


KEYS = load_keys()
DB_PATH = os.environ.get("AAREKA_DB_PATH", "aareka_ingest.db")


# ------------------------------------------------------------ user login (auth)
# Two credentials live side by side, both resolving to an org_id:
#   * collector key (X-Aareka-Key)   -> machines SEND data
#   * user token    (Bearer <token>) -> humans VIEW the dashboard
# Passwords are salted + PBKDF2-hashed (never stored in the clear). The login
# token is a small HMAC-signed blob {org, email, exp}; it is NOT stored server-
# side (stateless) - the signature is the proof. Uses only the stdlib.
PBKDF2_ROUNDS = 200_000
TOKEN_TTL = int(os.environ.get("AAREKA_TOKEN_TTL", str(8 * 3600)))   # 8h default
# Secret used to sign login tokens. Set AAREKA_TOKEN_SECRET so tokens survive a
# restart; otherwise a random one is used (all sessions re-login after restart).
TOKEN_SECRET = os.environ.get("AAREKA_TOKEN_SECRET") or secrets.token_hex(32)
if ACCOUNTS and not os.environ.get("AAREKA_TOKEN_SECRET"):
    log.warning("AAREKA_TOKEN_SECRET not set - using a random secret; logins will "
                "not survive a restart. Set it in the environment for stability.")


def new_salt() -> str:
    return base64.b64encode(os.urandom(16)).decode()


def hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(),
                             base64.b64decode(salt), PBKDF2_ROUNDS)
    return base64.b64encode(dk).decode()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _sign(body: str) -> str:
    return _b64(hmac.new(TOKEN_SECRET.encode(), body.encode(), hashlib.sha256).digest())


def mint_token(org_id: str, email: str, ttl: int = TOKEN_TTL) -> str:
    payload = {"org": org_id, "email": email, "exp": int(time.time()) + ttl}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    return f"{body}.{_sign(body)}"


def verify_token(token: str) -> Optional[dict]:
    """Return the token payload if the signature is valid and not expired, else None."""
    try:
        body, sig = token.split(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(body)):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except (ValueError, TypeError):
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload


# ------------------------------------------------------------ storage (SQLite)
# Production note: swap for a Postgres/RDS-backed class. Same schema, same
# "always filter by org_id" rule.
class Storage:
    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS readings(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id TEXT NOT NULL, host TEXT NOT NULL,
                captured_at TEXT, received_at TEXT NOT NULL,
                entity_id TEXT, scope TEXT, watts REAL,
                method TEXT, tier TEXT, error_pct REAL)""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS attributions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id TEXT NOT NULL, host TEXT NOT NULL,
                captured_at TEXT, received_at TEXT NOT NULL,
                workload TEXT, department TEXT, watts REAL,
                attribution TEXT, confidence REAL)""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users(
                email TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL, salt TEXT NOT NULL,
                org_id TEXT NOT NULL, created_at TEXT)""")
        # Self-serve orgs: a signup with no key mints a NEW org + collector key here.
        # The collector auth checks this table (plus the static env/keys.json keys).
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS org_keys(
                api_key TEXT PRIMARY KEY,
                org_id TEXT NOT NULL, org_name TEXT, created_at TEXT)""")
        # Inference: one row per (window, endpoint, model) with the energy the
        # token engine allocated to it. The raw window is kept as JSON too.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS inference(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id TEXT NOT NULL, host TEXT NOT NULL,
                window_start TEXT, window_end TEXT, window_s REAL,
                received_at TEXT NOT NULL,
                endpoint TEXT, server TEXT, model TEXT, department TEXT,
                prompt_tokens INTEGER, output_tokens INTEGER,
                gpu_energy_j REAL, cpu_energy_j REAL,
                busy_s REAL, busy_gpu_energy_j REAL)""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS inference_raw(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id TEXT NOT NULL, host TEXT NOT NULL,
                received_at TEXT NOT NULL, body TEXT)""")
        # GPU waste. gpu_readings is the RAW record (utilisation, memory, power,
        # owners per GPU per batch). The three tables after it are DERIVED from
        # it by gpu_waste and can be deleted and rebuilt at any time
        # (rebuild_waste) - e.g. when an account changes a threshold.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS gpu_readings(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id TEXT NOT NULL, host TEXT NOT NULL, gpu_id TEXT NOT NULL,
                ts REAL NOT NULL, captured_at TEXT, received_at TEXT NOT NULL,
                watts REAL, method TEXT, mode TEXT,
                util_pct REAL, mem_used_mb REAL, mem_total_mb REAL, gpu_model TEXT,
                owners TEXT, collector_version TEXT)""")
        # Per-account settings: only the values the account has changed. No row
        # (or a missing key) means the shipped default applies.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS org_settings(
                org_id TEXT PRIMARY KEY, body TEXT NOT NULL, updated_at TEXT)""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS gpu_waste_state(
                org_id TEXT NOT NULL, host TEXT NOT NULL, gpu_id TEXT NOT NULL,
                last_ts REAL, state TEXT NOT NULL,
                PRIMARY KEY(org_id, host, gpu_id))""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS gpu_spells(
                org_id TEXT NOT NULL, host TEXT NOT NULL, gpu_id TEXT NOT NULL,
                owner TEXT NOT NULL, start_ts REAL NOT NULL, end_ts REAL,
                duration_s REAL, gpu_hours REAL, energy_wh REAL, missing_s REAL,
                state TEXT, department TEXT, estimated INTEGER, modeled INTEGER,
                open INTEGER, readings INTEGER,
                PRIMARY KEY(org_id, host, gpu_id, owner, start_ts))""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS gpu_waste_daily(
                org_id TEXT NOT NULL, day TEXT NOT NULL, host TEXT NOT NULL,
                gpu_id TEXT NOT NULL, owner TEXT NOT NULL, department TEXT NOT NULL,
                category TEXT NOT NULL, state TEXT NOT NULL,
                estimated INTEGER NOT NULL, modeled INTEGER NOT NULL,
                gpu_hours REAL NOT NULL, energy_wh REAL NOT NULL, missing_s REAL NOT NULL,
                PRIMARY KEY(org_id, day, host, gpu_id, owner, department, category,
                            state, estimated, modeled))""")
        # Which kind of infrastructure each server was collected from (collector 0.7.0+).
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS host_sources(
                org_id TEXT NOT NULL, host TEXT NOT NULL, source TEXT, last_seen TEXT,
                PRIMARY KEY(org_id, host))""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_gr_gpu ON gpu_readings(org_id, host, gpu_id, ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_i_org ON inference(org_id, received_at)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ok_org ON org_keys(org_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_r_org ON readings(org_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_a_org ON attributions(org_id)")
        self.conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        """This thread's own connection to the database file. The server answers
        several requests at once (the dashboard asks for four things together),
        and one SQLite connection must not be used by two threads at the same
        time. Writes from different threads wait for each other (timeout)."""
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30)
            self._local.conn = c
        return c

    def create_user(self, email, password_hash, salt, org_id) -> None:
        # INSERT OR REPLACE: re-running make_user for an email updates that account.
        self.conn.execute(
            "INSERT OR REPLACE INTO users(email,password_hash,salt,org_id,created_at) "
            "VALUES(?,?,?,?,?)", (email.strip().lower(), password_hash, salt, org_id,
                                  datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def get_user(self, email) -> Optional[dict]:
        cur = self.conn.execute(
            "SELECT email,password_hash,salt,org_id FROM users WHERE email=?",
            (email.strip().lower(),))
        r = cur.fetchone()
        if not r:
            return None
        return {"email": r[0], "password_hash": r[1], "salt": r[2], "org_id": r[3]}

    def create_org_key(self, api_key, org_id, org_name) -> None:
        self.conn.execute(
            "INSERT INTO org_keys(api_key,org_id,org_name,created_at) VALUES(?,?,?,?)",
            (api_key, org_id, org_name, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def org_for_key(self, api_key) -> Optional[str]:
        cur = self.conn.execute("SELECT org_id FROM org_keys WHERE api_key=?", (api_key,))
        r = cur.fetchone()
        return r[0] if r else None

    def key_for_org(self, org_id) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT api_key FROM org_keys WHERE org_id=? ORDER BY created_at LIMIT 1", (org_id,))
        r = cur.fetchone()
        return r[0] if r else None

    def org_id_exists(self, org_id) -> bool:
        cur = self.conn.execute("SELECT 1 FROM org_keys WHERE org_id=? LIMIT 1", (org_id,))
        return cur.fetchone() is not None

    def insert_readings(self, org_id, host, captured_at, rows) -> int:
        now = datetime.now(timezone.utc).isoformat()
        data = [(org_id, host, captured_at, now, e, s, w, m, t, err)
                for (e, s, w, m, t, err) in rows]
        self.conn.executemany(
            "INSERT INTO readings(org_id,host,captured_at,received_at,entity_id,"
            "scope,watts,method,tier,error_pct) VALUES(?,?,?,?,?,?,?,?,?,?)", data)
        self.conn.commit()
        return len(data)

    def insert_attributions(self, org_id, host, captured_at, workloads) -> int:
        now = datetime.now(timezone.utc).isoformat()
        data = [(org_id, host, captured_at, now, w["workload"], w["department"],
                 w["total_w"], w["attribution"], w["confidence"]) for w in workloads]
        self.conn.executemany(
            "INSERT INTO attributions(org_id,host,captured_at,received_at,workload,"
            "department,watts,attribution,confidence) VALUES(?,?,?,?,?,?,?,?,?)", data)
        self.conn.commit()
        return len(data)

    def recent_readings(self, org_id, limit=50) -> list[dict]:
        cur = self.conn.execute(
            "SELECT host,captured_at,received_at,entity_id,scope,watts,method,tier "
            "FROM readings WHERE org_id=? ORDER BY id DESC LIMIT ?", (org_id, limit))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def recent_attributions(self, org_id, limit=50) -> list[dict]:
        cur = self.conn.execute(
            "SELECT host,captured_at,workload,department,watts,attribution,confidence "
            "FROM attributions WHERE org_id=? ORDER BY id DESC LIMIT ?", (org_id, limit))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def insert_inference(self, org_id, host, inf: dict, rows: list[dict]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO inference_raw(org_id,host,received_at,body) VALUES(?,?,?,?)",
            (org_id, host, now, json.dumps(inf)))
        data = [(org_id, host, inf.get("window_start"), inf.get("window_end"),
                 inf.get("window_s"), now, r["endpoint"], r["server"], r["model"],
                 r["department"], r["prompt_tokens"], r["output_tokens"],
                 r["gpu_energy_j"], r["cpu_energy_j"], r["busy_s"], r["busy_gpu_energy_j"])
                for r in rows]
        self.conn.executemany(
            "INSERT INTO inference(org_id,host,window_start,window_end,window_s,received_at,"
            "endpoint,server,model,department,prompt_tokens,output_tokens,gpu_energy_j,"
            "cpu_energy_j,busy_s,busy_gpu_energy_j) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            data)
        self.conn.commit()
        return len(data)

    def inference_since(self, org_id, since_iso) -> list[dict]:
        cur = self.conn.execute(
            "SELECT host,window_start,window_end,window_s,received_at,endpoint,server,model,"
            "department,prompt_tokens,output_tokens,gpu_energy_j,cpu_energy_j,busy_s,"
            "busy_gpu_energy_j FROM inference WHERE org_id=? AND received_at>=? "
            "ORDER BY id", (org_id, since_iso))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def set_host_source(self, org_id, host, source) -> None:
        """Record the server's infrastructure. A batch that does not say (an older
        collector) keeps what an earlier batch reported."""
        self.conn.execute(
            "INSERT INTO host_sources(org_id,host,source,last_seen) VALUES(?,?,?,?) "
            "ON CONFLICT(org_id,host) DO UPDATE SET source=COALESCE(excluded.source, source), "
            "last_seen=excluded.last_seen",
            (org_id, host, source, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def host_sources(self, org_id) -> dict:
        cur = self.conn.execute("SELECT host, source FROM host_sources WHERE org_id=?", (org_id,))
        return {h: s for h, s in cur.fetchall()}

    def latest_by_host(self, org_id) -> list[dict]:
        """Every workload row from the most recent batch of each host."""
        cur = self.conn.execute("""
            SELECT a.host, a.captured_at, a.received_at, a.workload, a.department,
                   a.watts, a.attribution, a.confidence
            FROM attributions a
            JOIN (SELECT host, MAX(received_at) mr FROM attributions
                  WHERE org_id=? GROUP BY host) latest
              ON a.host=latest.host AND a.received_at=latest.mr
            WHERE a.org_id=? ORDER BY a.host, a.watts DESC""", (org_id, org_id))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def rollup_by_dept(self, org_id) -> list[dict]:
        # Power by department over the LATEST snapshot per host (max received_at),
        # so we don't double-count across time. Always scoped to one org_id.
        cur = self.conn.execute("""
            SELECT a.department AS department, ROUND(SUM(a.watts),1) AS watts,
                   ROUND(COALESCE(SUM(a.watts*a.confidence)/NULLIF(SUM(a.watts),0),0),3)
                       AS confidence
            FROM attributions a
            JOIN (SELECT host, MAX(received_at) mr FROM attributions
                  WHERE org_id=? GROUP BY host) latest
              ON a.host=latest.host AND a.received_at=latest.mr
            WHERE a.org_id=?
            GROUP BY a.department ORDER BY watts DESC""", (org_id, org_id))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    # ---- GPU waste -------------------------------------------------------
    def get_settings(self, org_id) -> dict:
        """The values this account has set. {} when it has set none."""
        cur = self.conn.execute("SELECT body FROM org_settings WHERE org_id=?", (org_id,))
        r = cur.fetchone()
        if not r:
            return {}
        try:
            body = json.loads(r[0])
        except (ValueError, TypeError):
            return {}
        return body if isinstance(body, dict) else {}

    def save_settings(self, org_id, clean: dict) -> None:
        """Replace the account's settings with `clean` (only values that are set)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO org_settings(org_id,body,updated_at) VALUES(?,?,?)",
            (org_id, json.dumps(clean, sort_keys=True), datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def gpu_models(self, org_id) -> dict:
        """{(host, gpu_id): the model name it most recently reported}. GPUs whose
        collector never sent a model are absent."""
        cur = self.conn.execute(
            "SELECT g.host, g.gpu_id, g.gpu_model FROM gpu_readings g "
            "JOIN (SELECT host, gpu_id, MAX(id) mid FROM gpu_readings "
            "      WHERE org_id=? AND gpu_model IS NOT NULL GROUP BY host, gpu_id) m "
            "  ON g.id = m.mid", (org_id,))
        return {(h, gid): model for h, gid, model in cur.fetchall()}

    def insert_gpu_readings(self, org_id, host, captured_at, collector_version,
                            readings: list[dict]) -> None:
        """Store the raw per-GPU readings exactly as received (None stays None)."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.executemany(
            "INSERT INTO gpu_readings(org_id,host,gpu_id,ts,captured_at,received_at,watts,"
            "method,mode,util_pct,mem_used_mb,mem_total_mb,gpu_model,owners,collector_version) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(org_id, host, r["gpu_id"], r["ts"], captured_at, now, r["watts"], r["method"],
              r["mode"], r["util_pct"], r["mem_used_mb"], r["mem_total_mb"], r["gpu_model"],
              json.dumps(r["owners"]), collector_version) for r in readings])
        self.conn.commit()

    def _write_waste(self, org_id, host, gpu_id, out: dict) -> None:
        for s in out["segments"]:
            self.conn.execute(
                "INSERT INTO gpu_waste_daily(org_id,day,host,gpu_id,owner,department,category,"
                "state,estimated,modeled,gpu_hours,energy_wh,missing_s) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(org_id,day,host,gpu_id,owner,department,category,state,estimated,"
                "modeled) DO UPDATE SET gpu_hours=gpu_hours+excluded.gpu_hours, "
                "energy_wh=energy_wh+excluded.energy_wh, missing_s=missing_s+excluded.missing_s",
                (org_id, s["day"], host, gpu_id, s["owner"], s["department"] or "",
                 s["category"], s["state"], int(s["estimated"]), int(s["modeled"]),
                 s["gpu_hours"], s["energy_wh"] or 0.0, s["missing_s"]))
        for sp in out["spells"]:
            self.conn.execute(
                "INSERT OR REPLACE INTO gpu_spells(org_id,host,gpu_id,owner,start_ts,end_ts,"
                "duration_s,gpu_hours,energy_wh,missing_s,state,department,estimated,modeled,"
                "open,readings) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (org_id, host, gpu_id, sp["owner"], sp["start_ts"], sp["end_ts"],
                 sp["duration_s"], sp["gpu_hours"], sp["energy_wh"], sp["missing_s"],
                 sp["state"], sp["department"], int(sp["estimated"]), int(sp["modeled"]),
                 int(sp["open"]), sp["readings"]))

    def _save_waste_state(self, org_id, host, gpu_id, tracker) -> None:
        last = tracker.last or {}
        self.conn.execute(
            "INSERT OR REPLACE INTO gpu_waste_state(org_id,host,gpu_id,last_ts,state) "
            "VALUES(?,?,?,?,?)",
            (org_id, host, gpu_id, last.get("ts"), json.dumps(tracker.to_state())))

    def apply_gpu_reading(self, org_id, host, reading: dict, th: dict) -> Optional[dict]:
        """Classify one new reading and update that GPU's spells and daily totals.
        Returns the classification, or None if the reading is not newer than the
        last one processed (it is still in gpu_readings for a rebuild)."""
        gpu_id = reading["gpu_id"]
        cur = self.conn.execute(
            "SELECT state FROM gpu_waste_state WHERE org_id=? AND host=? AND gpu_id=?",
            (org_id, host, gpu_id))
        r = cur.fetchone()
        tracker = waste.GpuTracker(json.loads(r[0]) if r else None)
        out = tracker.step(reading, th)
        if out is None:
            return None
        self._write_waste(org_id, host, gpu_id, out)
        self._save_waste_state(org_id, host, gpu_id, tracker)
        self.conn.commit()
        return out["classification"]

    def rebuild_waste(self, org_id, th: dict) -> int:
        """Delete everything derived for this org and replay its raw GPU readings
        in time order with the given thresholds. Returns readings replayed."""
        for table in ("gpu_waste_state", "gpu_spells", "gpu_waste_daily"):
            self.conn.execute(f"DELETE FROM {table} WHERE org_id=?", (org_id,))
        cur = self.conn.execute(
            "SELECT host,gpu_id,ts,watts,method,mode,util_pct,mem_used_mb,mem_total_mb,"
            "gpu_model,owners FROM gpu_readings WHERE org_id=? ORDER BY host,gpu_id,ts,id",
            (org_id,))
        cols = [c[0] for c in cur.description]
        key, tracker, n = None, None, 0
        for row in cur.fetchall():
            reading = dict(zip(cols, row))
            reading["owners"] = json.loads(reading["owners"] or "{}")
            k = (reading["host"], reading["gpu_id"])
            if k != key:
                if tracker is not None:
                    self._save_waste_state(org_id, key[0], key[1], tracker)
                key, tracker = k, waste.GpuTracker()
            out = tracker.step(reading, th)
            if out is not None:
                self._write_waste(org_id, k[0], k[1], out)
                n += 1
        if tracker is not None:
            self._save_waste_state(org_id, key[0], key[1], tracker)
        self.conn.commit()
        return n

    def waste_gpu_states(self, org_id) -> list[dict]:
        cur = self.conn.execute(
            "SELECT host,gpu_id,state FROM gpu_waste_state WHERE org_id=? ORDER BY host,gpu_id",
            (org_id,))
        return [{"host": h, "gpu_id": g, "tracker": json.loads(s)} for h, g, s in cur.fetchall()]

    def waste_spells(self, org_id, since_ts, limit=500) -> list[dict]:
        cur = self.conn.execute(
            "SELECT host,gpu_id,owner,start_ts,end_ts,duration_s,gpu_hours,energy_wh,missing_s,"
            "state,department,estimated,modeled,open,readings FROM gpu_spells "
            "WHERE org_id=? AND end_ts>=? ORDER BY start_ts DESC LIMIT ?",
            (org_id, since_ts, limit))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def waste_daily(self, org_id, since_day) -> list[dict]:
        cur = self.conn.execute(
            "SELECT day,host,gpu_id,owner,department,category,state,estimated,modeled,"
            "gpu_hours,energy_wh,missing_s FROM gpu_waste_daily WHERE org_id=? AND day>=? "
            "ORDER BY day", (org_id, since_day))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


STORE = Storage(DB_PATH)
# Classifying a new batch and rebuilding after a threshold change both rewrite
# the derived waste tables; this keeps the two from interleaving.
WASTE_LOCK = threading.Lock()


def _site_collector_key() -> Optional[str]:
    """Single-site mode: the one collector key for this site - from
    AAREKA_COLLECTOR_KEY, else the one made on an earlier start, else a new one."""
    if ACCOUNTS:
        return None
    env_key = (os.environ.get("AAREKA_COLLECTOR_KEY") or "").strip()
    if env_key:
        if STORE.org_for_key(env_key) is None:
            STORE.create_org_key(env_key, SITE_ORG, "This site")
        return env_key
    key = STORE.key_for_org(SITE_ORG)
    if not key:
        key = "ak_" + secrets.token_hex(16)
        STORE.create_org_key(key, SITE_ORG, "This site")
    return key


SITE_KEY = _site_collector_key()
if SITE_KEY:
    # Printed so the person who starts the processor can paste it into each
    # collector. It is never sent to the dashboard, which has no login here.
    print("\n  Aareka processor - single site (no accounts)\n"
          f"  Collector key: {SITE_KEY}\n"
          "  Set AAREKA_ORG_KEY to this on every collector.\n", flush=True)


def org_for_api_key(api_key: str) -> Optional[str]:
    """Resolve a collector key to its org_id, checking the static keys (env /
    keys.json) first, then self-serve keys minted at signup (org_keys table)."""
    if api_key in KEYS:
        return KEYS[api_key]
    return STORE.org_for_key(api_key)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:24] or "org"


def new_org_id(org_name: str) -> str:
    """A readable, unique org_id: slug of the name + a short random suffix."""
    base = _slug(org_name)
    while True:
        cand = f"{base}-{secrets.token_hex(3)}"
        if not STORE.org_id_exists(cand) and cand not in set(KEYS.values()):
            return cand


# ------------------------------------------------------------ request models
class Reading(BaseModel):
    entity_id: str
    scope: str
    watts: float
    method: str
    tier: Optional[str] = None
    error_pct: Optional[float] = None

    @field_validator("watts")
    @classmethod
    def _finite_nonneg(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0:
            raise ValueError("watts must be a finite, non-negative number")
        return v

    @field_validator("method")
    @classmethod
    def _known_method(cls, v: str) -> str:
        if v not in ALLOWED_METHODS:
            raise ValueError(f"method must be one of {sorted(ALLOWED_METHODS)}")
        return v


class GpuIn(BaseModel):
    gpu_id: str
    watts: float
    method: str = "measured"
    mode: str = "passthrough"
    owner: Optional[str] = None                # passthrough: the owning workload id
    shares: dict[str, float] = Field(default_factory=dict)  # vgpu/mig: workload -> share (simple)
    # vgpu/mig (rich): workload -> {"alloc": slice share, "util": 0..100}. Used by
    # gpu_vm_power's baseline(alloc)+dynamic(util) split. Preferred over `shares`.
    vgpu_vms: dict[str, dict] = Field(default_factory=dict)
    # Card-level detail for GPU waste (collector 0.6.0+). All optional: an older
    # collector leaves them out, and that GPU's state is then "unknown".
    util_pct: Optional[float] = None
    mem_used_mb: Optional[float] = None
    mem_total_mb: Optional[float] = None
    gpu_model: Optional[str] = None

    @field_validator("watts")
    @classmethod
    def _finite(cls, v):
        if not math.isfinite(v) or v < 0:
            raise ValueError("watts must be finite, non-negative")
        return v

    @field_validator("util_pct", "mem_used_mb", "mem_total_mb", mode="before")
    @classmethod
    def _reading_or_none(cls, v):
        # an unusable value means "not available": the batch is not rejected for
        # it, and it is never turned into zero
        if v is None or isinstance(v, bool):
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) and f >= 0 else None

    @field_validator("gpu_model", mode="before")
    @classmethod
    def _model_or_none(cls, v):
        return v.strip()[:120] if isinstance(v, str) and v.strip() else None

    @field_validator("mode")
    @classmethod
    def _mode(cls, v):
        if v not in ALLOWED_MODES:
            raise ValueError(f"mode must be one of {sorted(ALLOWED_MODES)}")
        return v


class WorkloadIn(BaseModel):
    id: str
    kind: Optional[str] = None
    cpu_share: float = 0.0
    mem_mb: float = 0.0
    gpu_ids: list[str] = Field(default_factory=list)
    department: Optional[str] = None           # (C) the ownership tag; None -> unmapped


class ModelTokensIn(BaseModel):
    model: str
    prompt_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    department: Optional[str] = None


class EndpointIn(BaseModel):
    url: Optional[str] = None
    server: Optional[str] = None
    gpu_ids: list[str] = Field(default_factory=list)
    busy_s: float = Field(default=0.0, ge=0)
    busy_gpu_energy_j: float = Field(default=0.0, ge=0)
    counter_reset: bool = False
    error: Optional[str] = None
    models: list[ModelTokensIn] = Field(default_factory=list)


class InferenceIn(BaseModel):
    """(D) inference - raw joules + token deltas for one collector send window."""
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    window_s: float = Field(gt=0)
    sample_interval_ms: Optional[int] = None
    gpu_energy_j: dict[str, float] = Field(default_factory=dict)
    cpu_energy_j: Optional[float] = None
    not_measured: list[str] = Field(default_factory=list)
    endpoints: list[EndpointIn] = Field(default_factory=list)

    @field_validator("gpu_energy_j")
    @classmethod
    def _finite_j(cls, v):
        for g, j in v.items():
            if not math.isfinite(j) or j < 0:
                raise ValueError(f"gpu_energy_j[{g}] must be finite, non-negative")
        return v


class IngestBatch(BaseModel):
    host: str
    captured_at: Optional[str] = None
    collector_version: Optional[str] = None
    node_total_w: Optional[float] = None
    node_total_method: str = "measured"
    gpus: list[GpuIn] = Field(default_factory=list)
    cpu_readings: list[Reading] = Field(default_factory=list)
    workloads: list[WorkloadIn] = Field(default_factory=list)
    readings: list[Reading] = Field(default_factory=list)   # legacy flat form
    inference: Optional[InferenceIn] = None
    # Collector 0.7.0+: the mode it collected in. Older collectors leave it out.
    source: Optional[str] = None

    @field_validator("source", mode="before")
    @classmethod
    def _known_source(cls, v):
        # an unknown value is treated as not reported, never guessed
        return v if v in ("local", "vcenter", "kubernetes", "fleet") else None


class LoginIn(BaseModel):
    email: str
    password: str


class SignupIn(BaseModel):
    email: str
    password: str
    # No org_key -> CREATE a new org (self-serve); a key -> JOIN that existing org.
    org_key: Optional[str] = None
    org_name: Optional[str] = None      # label for a newly created org (optional)


# ------------------------------------------------------------ attribution
def attribute_batch(batch: IngestBatch) -> Optional[dict]:
    """Attribute one batch with the virtualization-aware engine:
      * GPU power -> gpu_vm_power.attribute_gpu_to_vms
          - passthrough: whole card -> one workload (exact)
          - vGPU/MIG: idle baseline by alloc + dynamic by UTILISATION (estimated)
      * non-GPU power -> split across workloads by CPU/memory share
    then roll up to department. None if there's no topology to attribute."""
    if not batch.workloads:
        return None

    # --- GPU attribution via the engine's vGPU/MIG-aware attributor ---
    gpuinfos = [gw.GpuInfo(g.gpu_id, g.gpu_id, g.watts, None, None, "") for g in batch.gpus]
    assignments = []
    for g in batch.gpus:
        if g.mode == "passthrough":
            assignments.append(vmp.GpuAssignment(g.gpu_id, "passthrough", owner=g.owner))
        else:  # vgpu / mig
            if g.vgpu_vms:
                vms = {vm: {"alloc": float(d.get("alloc", 0.0)),
                            "util": float(d.get("util", 0.0))}
                       for vm, d in g.vgpu_vms.items()}
            else:  # only simple shares -> treat as allocation, no utilisation signal
                vms = {vm: {"alloc": sh, "util": 0.0} for vm, sh in g.shares.items()}
            assignments.append(vmp.GpuAssignment(g.gpu_id, g.mode, vgpu_vms=vms))
    gres = vmp.attribute_gpu_to_vms(assignments, gpus=gpuinfos)
    gpu_by_vm = {v["vm"]: v for v in gres.get("vms", [])}
    total_gpu_w = gres.get("total_gpu_w", 0.0)

    # --- non-GPU (CPU + memory) split across workloads ---
    comp = sum(g.watts for g in batch.gpus) + sum(r.watts for r in batch.cpu_readings)
    parent_total = batch.node_total_w if batch.node_total_w else comp
    non_gpu = max(parent_total - total_gpu_w, 0.0)
    children = [psplit.Child(w.id, cpu_share=w.cpu_share, cpu_share_method="util",
                             mem_mb=w.mem_mb) for w in batch.workloads]
    cattrs, _ = psplit.split_power(non_gpu, [], children)
    cpu_by = {a.cid: (a.shared_w + a.mem_w) for a in cattrs}

    # --- combine per workload, then roll up to department ---
    dept_by_id = {w.id: w.department for w in batch.workloads}
    leaves, per_workload = [], []
    for w in batch.workloads:
        gv = gpu_by_vm.get(w.id) or {}
        gpu_w = gv.get("gpu_w", 0.0)
        cpu_w = cpu_by.get(w.id, 0.0)
        total_w = gpu_w + cpu_w
        conf = gv.get("confidence", 1.0)
        method = gv.get("method", "measured")
        exact = (method == "measured") and (cpu_w <= 1e-9)   # any CPU split -> estimated
        dept = dept_by_id.get(w.id)
        leaves.append(roll.Leaf(w.id, total_w, conf, bu_label=dept))
        per_workload.append({
            "workload": w.id, "department": dept or "unmapped",
            "gpu_w": round(gpu_w, 1), "cpu_w": round(cpu_w, 1),
            "total_w": round(total_w, 1),
            "attribution": "exact" if exact else "estimated",
            "confidence": round(conf, 3)})

    root = roll.rollup_to_bu(leaves, {})
    by_department = [{"department": n, "watts": round(b.watts, 1),
                      "confidence": round(b.confidence, 3)}
                     for n, b in sorted(root.children.items(),
                                        key=lambda kv: kv[1].watts, reverse=True)]
    attributed = sum(lf.watts for lf in leaves)
    return {
        "parent_total_w": round(parent_total, 1),
        "attributed_w": round(attributed, 1),
        "remainder_w": round(parent_total - attributed, 1),
        "reconciled": attributed <= parent_total + max(1.0, 0.02 * parent_total),
        "workloads": per_workload,
        "by_department": by_department,
    }


# ------------------------------------------------------------ GPU waste
def _epoch(iso: Optional[str]) -> Optional[float]:
    """ISO timestamp -> seconds since the epoch (UTC if no zone given), or None."""
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _iso(ts: Optional[float]) -> Optional[str]:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts is not None else None


def gpu_readings_from_batch(batch: IngestBatch) -> list[dict]:
    """One raw reading per GPU in the batch, with who is on the card and their
    department. Fields the collector did not send stay None."""
    dept = {w.id: w.department for w in batch.workloads}
    ts = _epoch(batch.captured_at) or time.time()
    out = []
    for g in batch.gpus:
        if g.vgpu_vms:
            owners = {vm: {"share": d.get("alloc"), "department": dept.get(vm)}
                      for vm, d in g.vgpu_vms.items()}
        elif g.shares:
            owners = {n: {"share": sh, "department": dept.get(n)} for n, sh in g.shares.items()}
        elif g.owner:
            owners = {g.owner: {"share": 1.0, "department": dept.get(g.owner)}}
        else:
            owners = {}
        out.append({"gpu_id": g.gpu_id, "ts": ts, "watts": g.watts, "method": g.method,
                    "mode": g.mode, "util_pct": g.util_pct, "mem_used_mb": g.mem_used_mb,
                    "mem_total_mb": g.mem_total_mb, "gpu_model": g.gpu_model,
                    "owners": owners})
    return out


def classify_batch(org_id: str, batch: IngestBatch) -> Optional[list[dict]]:
    """Store the batch's raw GPU readings, then classify each GPU with the
    account's thresholds (defaults where it has set none). None if the batch
    carries no per-GPU records."""
    readings = gpu_readings_from_batch(batch)
    if not readings:
        return None
    states = []
    with WASTE_LOCK:
        STORE.insert_gpu_readings(org_id, batch.host, batch.captured_at,
                                  batch.collector_version, readings)
        th = waste.resolve_thresholds(STORE.get_settings(org_id))
        for r in readings:
            c = STORE.apply_gpu_reading(org_id, batch.host, r, th)
            if c is not None:
                states.append({"gpu_id": r["gpu_id"], "state": c["state"],
                               "label": c["label"], "note": c["note"],
                               "idle_status": c["idle_status"], "idle_w": c["idle_w"]})
    return states


def build_waste_report(org_id: str, days: float) -> dict:
    """GPU states now, idle spells, and idle GPU-hours / energy per workload and
    per department per day - all derived from the stored raw readings.

    The range is whole UTC days: `days` = today plus the days before it, so the
    spells and the daily totals cover exactly the same period."""
    now = time.time()
    days = max(1, math.ceil(days))
    first = datetime.fromtimestamp(now, timezone.utc).date() - timedelta(days=days - 1)
    since = datetime(first.year, first.month, first.day, tzinfo=timezone.utc).timestamp()
    settings = STORE.get_settings(org_id)
    th = waste.resolve_thresholds(settings)
    rates = waste.resolve_rates(settings)
    models = STORE.gpu_models(org_id)

    def rate_for(host, gpu_id):
        return waste.gpu_hour_rate(models.get((host, gpu_id)), rates)

    def money(x):
        return None if x is None else round(x, 4)

    def cost(h_rated_cost, h_rated, h_unrated, energy_wh, energy_missing_s):
        """The two parts of waste cost, kept apart. A part is None when the rate
        it needs is not set; GPU-hours with no rate for their model are counted
        in gpu_hours_without_rate rather than priced."""
        return {"gpu_hours_part": money(h_rated_cost) if h_rated > 0 else None,
                "gpu_hours_priced": round(h_rated, 5),
                "gpu_hours_without_rate": round(h_unrated, 5),
                "energy_part": money(waste.energy_cost(energy_wh, rates)),
                "energy_missing_s": round(energy_missing_s, 1)}

    gpus, reporting = [], {}
    for row in STORE.waste_gpu_states(org_id):
        tracker = waste.GpuTracker(row["tracker"])
        last = tracker.last or {}
        live = last.get("ts") is not None and now - last["ts"] <= tracker.gap_limit()
        reporting[(row["host"], row["gpu_id"])] = live
        lo = last.get("last_owner")
        gpus.append({
            "host": row["host"], "gpu_id": row["gpu_id"],
            "state": last.get("state"), "label": last.get("label"), "note": last.get("note"),
            "waste": last.get("waste"),
            "idle_status": last.get("idle_status"), "idle_w": last.get("idle_w"),
            "util_pct": last.get("util_pct"), "watts": last.get("watts"),
            "power_method": last.get("power_method"),
            "mem_used_mb": last.get("mem_used_mb"), "mem_total_mb": last.get("mem_total_mb"),
            "gpu_model": last.get("gpu_model"), "mode": last.get("mode"),
            "owners": last.get("owners") or {}, "shared": last.get("shared"),
            "last_owner": ({"seen_at": _iso(lo["ts"]), "owners": lo["owners"]} if lo else None),
            "last_seen": _iso(last.get("ts")), "reporting": live})

    def energy_label(estimated, modeled):
        return "estimated" if estimated else ("modeled" if modeled else "measured")

    spells = []
    for s in STORE.waste_spells(org_id, since):
        has_energy = s["duration_s"] == 0 or s["missing_s"] < s["duration_s"]
        rate = rate_for(s["host"], s["gpu_id"])
        spells.append({
            "host": s["host"], "gpu_id": s["gpu_id"],
            "workload": s["owner"] or None,                 # None = no workload on the GPU
            "department": s["department"] or "unmapped",
            "state": s["state"], "label": waste.STATE_LABELS.get(s["state"], s["state"]),
            "start": _iso(s["start_ts"]), "last_reading": _iso(s["end_ts"]),
            "duration_s": round(s["duration_s"], 1),
            "idle_gpu_hours": round(s["gpu_hours"], 5),
            "energy_wh": round(s["energy_wh"], 3) if has_energy else None,
            "energy_missing_s": round(s["missing_s"], 1),
            "hours_label": "estimated" if s["estimated"] else "measured",
            "energy_label": energy_label(s["estimated"], s["modeled"]) if has_energy else None,
            "ongoing": bool(s["open"]) and reporting.get((s["host"], s["gpu_id"]), False),
            "readings": s["readings"],
            "gpu_model": models.get((s["host"], s["gpu_id"])),
            "cost": cost(s["gpu_hours"] * rate if rate else 0.0,
                         s["gpu_hours"] if rate else 0.0, 0.0 if rate else s["gpu_hours"],
                         s["energy_wh"] if has_energy else None, s["missing_s"])})

    from_day = first.isoformat()
    allocated = working = idle = idle_wh = idle_missing = 0.0
    idle_cost = idle_priced = idle_unpriced = 0.0
    by_workload, by_dept = {}, {}
    for r in STORE.waste_daily(org_id, from_day):
        allocated += r["gpu_hours"]
        if r["category"] == waste.CAT_WORKING:
            working += r["gpu_hours"]
        if r["category"] != waste.CAT_IDLE:
            continue
        idle += r["gpu_hours"]
        idle_wh += r["energy_wh"]
        idle_missing += r["missing_s"]
        rate = rate_for(r["host"], r["gpu_id"])
        if rate:
            idle_cost += r["gpu_hours"] * rate
            idle_priced += r["gpu_hours"]
        else:
            idle_unpriced += r["gpu_hours"]
        dept = r["department"] or "unmapped"
        for key, groups in (((r["day"], r["host"], r["owner"], dept), by_workload),
                            ((r["day"], dept), by_dept)):
            g = groups.setdefault(key, {"h": 0.0, "wh": 0.0, "miss": 0.0, "est": False,
                                        "mod": False, "hc": 0.0, "hp": 0.0, "hu": 0.0})
            if rate:
                g["hc"] += r["gpu_hours"] * rate
                g["hp"] += r["gpu_hours"]
            else:
                g["hu"] += r["gpu_hours"]
            g["h"] += r["gpu_hours"]
            g["wh"] += r["energy_wh"]
            g["miss"] += r["missing_s"]
            g["est"] = g["est"] or bool(r["estimated"])
            g["mod"] = g["mod"] or bool(r["modeled"])

    def figures(g):
        return {"idle_gpu_hours": round(g["h"], 5), "idle_energy_wh": round(g["wh"], 3),
                "energy_missing_s": round(g["miss"], 1),
                "hours_label": "estimated" if g["est"] else "measured",
                "energy_label": energy_label(g["est"], g["mod"]),
                "cost": cost(g["hc"], g["hp"], g["hu"], g["wh"], g["miss"])}

    by_workload_day = [dict({"day": d, "host": h, "workload": o or None, "department": dp},
                            **figures(g)) for (d, h, o, dp), g in by_workload.items()]
    by_workload_day.sort(key=lambda x: (x["day"], x["idle_gpu_hours"]), reverse=True)
    by_department_day = [dict({"day": d, "department": dp}, **figures(g))
                         for (d, dp), g in by_dept.items()]
    by_department_day.sort(key=lambda x: (x["day"], x["idle_gpu_hours"]), reverse=True)

    return {
        "org": org_id, "days": days, "from_day": from_day, "day_boundary": "UTC",
        "thresholds": {k: {"value": th[k], "default": waste.DEFAULT_THRESHOLDS[k],
                           "user_set": k in th["user_set"]}
                       for k in waste.DEFAULT_THRESHOLDS},
        "method": {"idle_run_readings": waste.IDLE_RUN_READINGS,
                   "idle_lookback_days": waste.IDLE_LOOKBACK_S // 86400,
                   "min_gap_s": waste.MIN_GAP_LIMIT_S,
                   "gap_interval_factor": waste.GAP_INTERVAL_FACTOR},
        "state_labels": waste.STATE_LABELS,
        "totals": {"allocated_gpu_hours": round(allocated, 5),
                   "working_gpu_hours": round(working, 5),
                   "idle_gpu_hours": round(idle, 5),
                   "idle_share": round(idle / allocated, 4) if allocated > 0 else None,
                   "idle_energy_wh": round(idle_wh, 3),
                   "idle_energy_missing_s": round(idle_missing, 1),
                   "cost": cost(idle_cost, idle_priced, idle_unpriced, idle_wh, idle_missing)},
        "rates": rates_in_use(rates, models),
        "gpus": gpus,
        "spells": spells,
        "by_workload_day": by_workload_day,
        "by_department_day": by_department_day,
    }


def rates_in_use(rates: dict, models: dict) -> dict:
    """The account's rates as shown next to money figures, plus the GPU models
    this account has reported and the GPU-hour rate set for each (None = not set)."""
    seen = sorted({m for m in models.values() if m})
    return {"label": "calculated from your rates",
            "currency": rates["currency"],
            "electricity_rate_per_kwh": rates["electricity_rate_per_kwh"],
            "pue": rates["pue"],
            "carbon_g_per_kwh": rates["carbon_g_per_kwh"],
            "gpu_hour_rates": {m: rates["gpu_hour_rates"].get(m) for m in seen},
            "gpu_hour_rates_other_models": {m: r for m, r in rates["gpu_hour_rates"].items()
                                            if m not in seen}}


# ------------------------------------------------------------ settings
class SettingsIn(BaseModel):
    """The whole settings record the account wants. A value left out (or null)
    is not set: a threshold then uses its default, a rate gives no money figure."""
    settings: dict = Field(default_factory=dict)
    # A threshold change rewrites past figures, so it is refused until the
    # person has seen the warning and confirmed.
    confirm_recalculation: bool = False


def settings_view(org_id: str) -> dict:
    current = STORE.get_settings(org_id)
    models = STORE.gpu_models(org_id)
    return {
        "org": org_id,
        "values": waste.clean_settings(current)[0],
        "defaults": {"util_threshold_pct": waste.DEFAULT_THRESHOLDS["util_threshold_pct"],
                     "idle_power_margin_pct": waste.DEFAULT_THRESHOLDS["idle_power_margin_pct"],
                     "waste_states": list(waste.DEFAULT_THRESHOLDS["waste_states"]),
                     "ticket_idle_hours": waste.DEFAULT_TICKET_IDLE_HOURS},
        "waste_state_choices": [{"id": st, "label": waste.STATE_LABELS[st]}
                                for st in waste.WASTE_STATE_CHOICES],
        "gpu_models_seen": sorted({m for m in models.values() if m}),
        "recalculation_message": waste.RECALC_MESSAGE,
    }


def save_settings(org_id: str, body: SettingsIn) -> dict:
    clean, errors = waste.clean_settings(body.settings)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    with WASTE_LOCK:
        old_th = waste.resolve_thresholds(STORE.get_settings(org_id))
        new_th = waste.resolve_thresholds(clean)
        changed = [k for k in waste.THRESHOLD_KEYS if old_th[k] != new_th[k]]
        if changed and not body.confirm_recalculation:
            raise HTTPException(status_code=409, detail={
                "requires_confirmation": True, "changed": changed,
                "message": waste.RECALC_MESSAGE})
        STORE.save_settings(org_id, clean)
        replayed = STORE.rebuild_waste(org_id, new_th) if changed else 0
    log.info("settings saved org=%s thresholds_changed=%s readings_recalculated=%d",
             org_id, changed, replayed)
    return {"status": "ok", "recalculated": bool(changed), "changed_thresholds": changed,
            "readings_recalculated": replayed, **settings_view(org_id)}


# ------------------------------------------------------------ auth dependency
def require_org(x_aareka_key: Optional[str] = Header(default=None,
                                                     alias="X-Aareka-Key")) -> str:
    """Collector-key only. Used by /v1/ingest (machines send data). In
    single-site mode every valid key sends to the site."""
    org = org_for_api_key(x_aareka_key) if x_aareka_key else None
    if not org:
        log.warning("rejected: missing or invalid API key")
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return org if ACCOUNTS else SITE_ORG


def require_org_any(x_aareka_key: Optional[str] = Header(default=None,
                                                        alias="X-Aareka-Key"),
                    authorization: Optional[str] = Header(default=None)) -> str:
    """Accept EITHER a collector key (machines) OR a Bearer login token (humans).
    Both resolve to an org_id. Used by the read endpoints the dashboard calls.
    In single-site mode there are no accounts: reads need no credentials."""
    if not ACCOUNTS:
        return SITE_ORG
    if x_aareka_key:
        org = org_for_api_key(x_aareka_key)
        if org:
            return org
    if authorization and authorization.startswith("Bearer "):
        payload = verify_token(authorization[7:].strip())
        if payload and payload.get("org"):
            return payload["org"]
    raise HTTPException(status_code=401, detail="invalid or missing credentials")


def require_org_user(authorization: Optional[str] = Header(default=None)) -> str:
    """A signed-in person only (Bearer login token). Changing settings rewrites
    past figures, so a collector key - which machines hold - cannot do it.
    In single-site mode there are no logins, so the dashboard can save them."""
    if not ACCOUNTS:
        return SITE_ORG
    if authorization and authorization.startswith("Bearer "):
        payload = verify_token(authorization[7:].strip())
        if payload and payload.get("org"):
            return payload["org"]
    raise HTTPException(status_code=401, detail="sign in to change settings")


# ------------------------------------------------------------ app
app = FastAPI(
    title="Aareka Ingest API",
    version="0.4.0",
    description="Receives readings + ownership, attributes with the engine, stores "
                "and serves - all scoped per org.",
)


_HERE = Path(__file__).parent
_DASHBOARD = _HERE / "dashboard.html"                 # single site: no login
_ACCOUNTS_DASHBOARD = _HERE / "tenant_dashboard.html"  # accounts: login + sign-up


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """The dashboard. Single site: opens straight to this site's data.
    Accounts mode: the sign-in dashboard, if it is installed next to this file."""
    page = _ACCOUNTS_DASHBOARD if ACCOUNTS else _DASHBOARD
    if not page.exists():
        return HTMLResponse(f"<p>Dashboard file {page.name} not found next to "
                            f"ingest_api.py.</p>", status_code=404)
    return page.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "accounts" if ACCOUNTS else "single-site",
            "orgs_configured": len(KEYS)}


def _require_accounts() -> None:
    if not ACCOUNTS:
        raise HTTPException(status_code=404, detail=(
            "accounts are off: this processor runs as a single site. "
            "Set AAREKA_ACCOUNTS=1 to use sign-up and login."))


@app.post("/v1/login")
def login(body: LoginIn) -> dict:
    """Human login: email + password -> a short-lived signed token carrying org_id.
    Same 'invalid' reply whether the email is unknown or the password is wrong."""
    _require_accounts()
    user = STORE.get_user(body.email)
    ok = False
    if user:
        calc = hash_password(body.password, user["salt"])
        ok = hmac.compare_digest(calc, user["password_hash"])
    if not ok:
        log.warning("login failed for %s", (body.email or "").strip().lower())
        raise HTTPException(status_code=401, detail="invalid email or password")
    token = mint_token(user["org_id"], user["email"])
    log.info("login ok email=%s org=%s", user["email"], user["org_id"])
    return {"token": token, "org": user["org_id"], "expires_in": TOKEN_TTL}


@app.post("/v1/signup")
def signup(body: SignupIn) -> dict:
    """Self-serve account creation.
      * No org_key  -> mint a BRAND-NEW org + its own collector key (returned so the
                       person can point their collector at it). They own a fresh,
                       isolated tenant - no key needed from us.
      * With org_key -> JOIN that existing org (a colleague joining a team). The key
                       maps to an org_id SERVER-SIDE, so they can only join the org
                       the key belongs to.
    Auto-logs in on success."""
    _require_accounts()
    email = (body.email or "").strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="please enter a valid email address")
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="password must be at least 8 characters")
    if STORE.get_user(email):
        raise HTTPException(status_code=409,
                            detail="an account with this email already exists - sign in instead")

    collector_key = None
    if body.org_key and body.org_key.strip():          # JOIN an existing org
        org_id = org_for_api_key(body.org_key.strip())
        if not org_id:
            raise HTTPException(status_code=401, detail="organisation key not recognised")
    else:                                              # CREATE a new org
        org_name = (body.org_name or "").strip() or email.split("@", 1)[1]
        org_id = new_org_id(org_name)
        collector_key = "ak_" + secrets.token_hex(16)
        STORE.create_org_key(collector_key, org_id, org_name)
        log.info("new org created org=%s name=%s", org_id, org_name)

    salt = new_salt()
    STORE.create_user(email, hash_password(body.password, salt), salt, org_id)
    log.info("signup ok email=%s org=%s created_org=%s", email, org_id, bool(collector_key))
    return {"token": mint_token(org_id, email), "org": org_id, "expires_in": TOKEN_TTL,
            "collector_key": collector_key, "created_org": bool(collector_key)}


@app.post("/v1/ingest")
def ingest(batch: IngestBatch, org_id: str = Depends(require_org)) -> dict:
    # 1. collect every component reading into the readings table
    comps = []
    for r in batch.readings:
        comps.append((r.entity_id, r.scope, r.watts, r.method, r.tier, r.error_pct))
    for r in batch.cpu_readings:
        comps.append((r.entity_id, r.scope, r.watts, r.method, r.tier, r.error_pct))
    for g in batch.gpus:
        comps.append((g.gpu_id, "gpu", g.watts, g.method, g.mode, None))
    if batch.node_total_w is not None:
        comps.append(("node", "node", batch.node_total_w, batch.node_total_method,
                      "total", None))
    if not comps:
        raise HTTPException(status_code=400,
                            detail="batch has no readings (gpus/cpu_readings/readings)")
    stored = STORE.insert_readings(org_id, batch.host, batch.captured_at, comps)
    STORE.set_host_source(org_id, batch.host, batch.source)

    # 2. attribute (if the batch carries topology + tags)
    attribution = attribute_batch(batch)
    attributed = 0
    if attribution and "workloads" in attribution:
        attributed = STORE.insert_attributions(org_id, batch.host, batch.captured_at,
                                               attribution["workloads"])

    # 3. inference: allocate the window's joules to models, store, summarise
    tokens = None
    if batch.inference is not None:
        inf = batch.inference.model_dump()
        rows = tok.allocate_window(inf)
        STORE.insert_inference(org_id, batch.host, inf, rows)
        for r in rows:
            r["window_s"] = inf["window_s"]
        tokens = tok.summarise(rows)

    # 4. GPU waste: store the raw per-GPU readings and classify each GPU. The
    #    power data above is already stored, so a fault here must not fail the
    #    ingest - it is logged and the batch is still accepted.
    gpu_states = None
    try:
        gpu_states = classify_batch(org_id, batch)
    except Exception:
        log.exception("gpu waste step failed org=%s host=%s", org_id, batch.host)

    log.info("ingest org=%s host=%s readings=%d attributed=%d inference=%s",
             org_id, batch.host, stored, attributed, tokens is not None)
    return {"status": "ok", "org": org_id, "host": batch.host,
            "stored_readings": stored, "attributed_workloads": attributed,
            "attribution": attribution, "tokens": tokens, "gpu_states": gpu_states}


@app.get("/v1/readings")
def readings(limit: int = 50, org_id: str = Depends(require_org_any)) -> dict:
    limit = max(1, min(limit, 500))
    rows = STORE.recent_readings(org_id, limit)
    return {"org": org_id, "count": len(rows), "readings": rows}


@app.get("/v1/attribution")
def attribution(limit: int = 50, org_id: str = Depends(require_org_any)) -> dict:
    limit = max(1, min(limit, 500))
    rows = STORE.recent_attributions(org_id, limit)
    return {"org": org_id, "count": len(rows), "workloads": rows}


@app.get("/v1/snapshot")
def snapshot(org_id: str = Depends(require_org_any)) -> dict:
    """The most recent batch from each server: power per workload, with its
    department and whether the split is exact or estimated."""
    return {"org": org_id, "rows": STORE.latest_by_host(org_id),
            "hosts": {h: {"source": s} for h, s in STORE.host_sources(org_id).items()}}


@app.get("/v1/rollup")
def rollup(org_id: str = Depends(require_org_any)) -> dict:
    rows = STORE.rollup_by_dept(org_id)
    return {"org": org_id, "by_department": rows}


@app.get("/v1/tokens")
def tokens(hours: float = 24.0, tariff: Optional[float] = None,
           org_id: str = Depends(require_org_any)) -> dict:
    """Energy per token over the last `hours`, per model and per department.
    `tariff` (price per kWh) adds cost per million output tokens."""
    hours = max(0.1, min(hours, 24 * 90))
    since = datetime.fromtimestamp(time.time() - hours * 3600, timezone.utc).isoformat()
    rows = STORE.inference_since(org_id, since)
    out = tok.summarise(rows, tariff if tariff and tariff > 0 else None)
    out.update({"org": org_id, "hours": hours,
                "windows": len({(r["host"], r["window_end"]) for r in rows})})
    return out


@app.get("/v1/waste")
def waste_report(days: float = 7.0, org_id: str = Depends(require_org_any)) -> dict:
    """GPU waste over the last `days` UTC days (today counts as day 1): each
    GPU's state now, idle spells, and idle GPU-hours and energy per workload and
    per department per day, with the thresholds in use."""
    return build_waste_report(org_id, max(1.0, min(days, 3650.0)))


@app.get("/v1/settings")
def get_settings(org_id: str = Depends(require_org_any)) -> dict:
    """The account's rates and waste thresholds (only what it has set), the
    defaults, and the GPU models it has reported."""
    return settings_view(org_id)


@app.put("/v1/settings")
def put_settings(body: SettingsIn, org_id: str = Depends(require_org_user)) -> dict:
    """Replace the account's settings. Values left out are not set. If a waste
    threshold changes, past GPU states, idle spells and waste totals are
    recalculated from the stored readings - only with confirm_recalculation,
    otherwise 409 with the message to show. Recalculation sends no tickets."""
    return save_settings(org_id, body)


@app.get("/v1/org")
def org_info(org_id: str = Depends(require_org_any)) -> dict:
    """The org + its collector key. In single-site mode the key is never
    returned: the dashboard has no login, so anyone could read it."""
    if not ACCOUNTS:
        return {"org": org_id, "mode": "single-site", "collector_key": None}
    return {"org": org_id, "mode": "accounts", "collector_key": STORE.key_for_org(org_id)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    log.info("starting Aareka processor on :%d (%s)", port,
             "accounts" if ACCOUNTS else "single site")
    uvicorn.run(app, host="0.0.0.0", port=port)
