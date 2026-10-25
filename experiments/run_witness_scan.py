from __future__ import annotations
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import glob, json, os, re, sys

# Harvests feasibility witnesses from the solution logs in data/witness_logs/.
#
# A request is FEASIBLE if some run served it and that run's checker reported
# no STRUCTURAL violations. Coverage violations ("request X is served by NO
# vehicle") are not structural - they report other requests being missed, not
# an invalid route - so such a run still witnesses the requests it did serve.
#
# This requires no trust in any solver: the independent checker validated the
# route.

ROOT = _REPO_ROOT
LOG_DIR = os.path.join(ROOT, "data/witness_logs")
SERVED_RE = re.compile(r"(veh_\d+):\s*served=\(([^)]*)\)")
SERVED_RE2 = re.compile(r"(veh_\d+):\s*served=\[([^\]]*)\]")
STRUCT_RE = re.compile(r"structural[_ ]violations[^:]*:\s*(\d+)", re.I)
STRUCT_RE2 = re.compile(r"STRUCTURAL violations[^:]*:\s*(\d+)")


def structural_count(text):
    """Return the structural-violation count a log reports, or None if the log
    never states one. Also derive it from raw VIOLATION lines when possible."""
    best = None
    for rx in (STRUCT_RE2, STRUCT_RE):
        for m in rx.finditer(text):
            best = int(m.group(1))          # keep the LAST reported count
    if best is not None:
        return best
    viol = re.findall(r"VIOLATION: (.+)", text)
    if viol:
        non_cov = [v for v in viol if "served by NO vehicle" not in v]
        return len(non_cov)
    return None


witness = {}
scanned, usable = [], []
for path in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
    name = os.path.basename(path)
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        continue
    scanned.append(name)
    sc = structural_count(text)
    if sc is None or sc != 0:
        continue                              # only zero-structural runs witness
    usable.append((name, sc))
    for rx in (SERVED_RE, SERVED_RE2):
        for m in rx.finditer(text):
            veh, body = m.group(1), m.group(2).strip()
            if not body:
                continue
            for tok in body.split(","):
                tok = tok.strip()
                if not tok.isdigit():
                    continue
                rid = int(tok)
                if rid not in witness:
                    witness[rid] = {"request_id": rid, "log_file": name,
                                    "vehicle": veh, "served_set": body,
                                    "structural_violations": sc}

if not scanned:
    print(f"no .log files found in {LOG_DIR}; run the solver experiments first so "
          f"their logs can be scanned for witnesses")

print(f"scanned {len(scanned)} logs; {len(usable)} had a checker report with ZERO "
      f"structural violations and are therefore usable as witnesses")
for n, sc in usable:
    print(f"   witness-capable: {n} (structural_violations={sc})")
print(f"\nrequests with a checker-validated serving witness: {len(witness)}")
print(f"  ids: {sorted(witness)}")

json.dump({str(k): v for k, v in sorted(witness.items())},
          open(os.path.join(ROOT, "results/coverage/witness_feasible.json"), "w"), indent=2)
print(f"\nwrote results/coverage/witness_feasible.json")

# report requests 2, 3, 11 and 24, which the single-vehicle solo scan cannot serve
for rid in (2, 3, 11, 24):
    w = witness.get(rid)
    if w:
        print(f"  FALSIFIES solo oracle: request {rid} served in {w['log_file']} "
              f"by {w['vehicle']} (served={w['served_set']}, struct_viol={w['structural_violations']})")
    else:
        print(f"  request {rid}: no witness found in any zero-structural log")
