"""
IMPS beta - self-hosted, single-node dashboard.

Runs inside the customer's own environment and shows REAL power + workload
attribution for the one machine it runs on. Everything above single-server
scale (fleet / rack / datacenter) is intentionally NOT shown here - it is a
full-deployment capability, and the beta never fabricates it.

The two hard rules of this package:
  1. No simulated data is ever presented as real. If a value cannot be
     measured, it is shown as estimated (with a range) or omitted.
  2. Nothing leaves the machine. There is no outbound network call.
"""

__version__ = "0.1.0-beta"
