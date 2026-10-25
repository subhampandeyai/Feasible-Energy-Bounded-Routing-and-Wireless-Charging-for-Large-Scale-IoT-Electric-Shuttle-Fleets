"""R=30 baseline comparison under an external hard wall-clock cap (Windows).

Runs the proposed method and the six baselines on the R=30 instance, one method at
a time, each in its own process inside its own Windows Job Object. The cap is
external: when it is reached the driver terminates the whole Job Object, so no
method can run past it. Method arguments are fixed below and are not tuned.

Usage (from the repository root):
    python -B experiments/run_baseline_r30.py                       # all seven, 1800 s cap
    python -B experiments/run_baseline_r30.py --cap 3600 --methods greedy,monolithic,admm,ensemble --tag completion_3600s

Prerequisite: results/coverage/classification_R30.json (run_classify_requests.py --requests 30).
Outputs (never overwritten): results/baseline/baseline_<tag>.log and .csv

Scoring per method:
  COMPLETED          method returned; served = requests served exactly once (independent checker)
  INTERNAL_TIMEOUT   method returned after its own time budget fired; served as above
  EXTERNAL_KILL_<cap> killed by the external cap before returning: served/violations left blank
  CRASHED            exception; served/violations left blank
A run is marked VALID only if the machine stayed awake and on mains power for the whole
run (checked against the Windows Kernel-Power event log), INVALID otherwise, and
UNVERIFIED if the event log cannot be read.
"""
from __future__ import annotations

import argparse, csv, ctypes, datetime, hashlib, json, os, re, subprocess, sys, threading, time, traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO, "data")
CLASSIFICATION = os.path.join(REPO, "results", "coverage", "classification_R30.json")
OUT_DIR = os.path.join(REPO, "results", "baseline")
R = 30
B = 1800.0                      # methods' own wall budget
METHODS = ["cluster", "greedy", "cg", "monolithic", "ggflow", "admm", "ensemble"]

CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200
ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002


# --------------------------------------------------------------------------- child
def run_child(method: str) -> int:
    """Run ONE method with fixed arguments, then the independent checker."""
    import logging
    sys.path.insert(0, REPO)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        stream=sys.stdout, force=True)

    budget_msgs = []

    class _BudgetCatcher(logging.Handler):          # records soft-budget stops only
        def emit(self, rec):
            m = rec.getMessage()
            if "wall budget" in m and "exceeded" in m:
                budget_msgs.append(m)
    logging.getLogger().addHandler(_BudgetCatcher())

    from src.model.readers.dataset_reader import load_dataset_instance
    from src.solve.solver_interface import HighsSolver
    from src.verify.checker import check_solution, VehicleRoute

    inst = load_dataset_instance(DATASET, R)
    out = {"method": method, "internal_status": "", "internal_timeout": False, "extra": ""}

    t0 = time.time()
    if method == "cluster":
        from src.solve.cluster_route import solve_cluster_route, to_vehicle_routes as cl_routes
        cls = json.load(open(CLASSIFICATION))
        ids = [r.id for r in inst.requests]
        infeasible = set(cls["infeasible_ids"])
        pre = ([i for i in ids if i not in infeasible], sorted(infeasible), [])
        t0 = time.time()
        res = solve_cluster_route(inst, engine="insertion", n_candidates=14,
                                  reinsert_passes=3, precomputed_solo=pre)
        wall = time.time() - t0
        routes = cl_routes(res.per_vehicle)
        out["extra"] = f"proven_infeasible={len(res.proven_infeasible)} offered={len(pre[0])}"
    elif method == "greedy":
        from src.solve.greedy_regret import solve_greedy_regret, to_vehicle_routes as g_routes
        gr = solve_greedy_regret(inst, time_budget_per_insertion_s=15.0,
                                 label_budget_per_insertion=200000, wall_budget_s=B)
        wall = time.time() - t0
        routes = g_routes(gr.per_vehicle_selected)
        out["extra"] = f"rounds={gr.rounds} unserved={len(gr.unserved)}"
    elif method == "cg":
        from src.solve.column_generation import solve_column_generation
        solver = HighsSolver(time_limit_s=30.0, mip_rel_gap=0.0)
        cg = solve_column_generation(inst, solver=solver, wall_budget_s=B)
        wall = time.time() - t0
        routes = {}
        for vid, col in (cg.per_vehicle_selected or {}).items():
            if col is not None:
                routes[vid] = VehicleRoute(arcs=list(col.arcs), charge=dict(col.charge),
                                           served=list(col.served))
        out["internal_status"] = f"stop={cg.stop_reason}"
        out["internal_timeout"] = cg.stop_reason == "wall_budget_exceeded"
        out["extra"] = f"rounds={cg.rounds} stop={cg.stop_reason}"
    elif method == "monolithic":
        from src.solve.monolithic_milp import solve_monolithic
        solver = HighsSolver(time_limit_s=B, mip_rel_gap=0.0)
        mres = solve_monolithic(inst, solver=solver)
        wall = time.time() - t0
        routes = mres.routes or {}
        sr = mres.solve_result
        st = str(getattr(sr, "status", "?"))
        out["internal_status"] = f"highs_status={st} is_optimal={getattr(sr, 'is_optimal', '?')}"
        out["internal_timeout"] = "time limit" in st.lower()
        out["extra"] = out["internal_status"]
    elif method == "ggflow":
        from src.solve.lagrangian import solve_lagrangian
        from src.solve.lagrangian_repair import repair_from_subproblems
        solver = HighsSolver(time_limit_s=30.0, mip_rel_gap=0.0)
        # Each iteration's per-vehicle subproblem solutions are turned into a
        # checker-clean partial plan (duplicates dropped, unusable paths
        # discarded, discrete charging repaired); the best one is reported.
        # best_dual bounds the relaxed model, in which every vehicle drives
        # start->end; the repaired plan lets idle vehicles stay home, so
        # best_dual is NOT a lower bound on best_primal and no gap is reported.
        lag = solve_lagrangian(inst, solver=solver, max_iterations=3,
                               repair_fn=repair_from_subproblems)
        routes = lag.best_primal_solution or {}
        wall = time.time() - t0
        out["extra"] = f"best_dual={lag.best_dual} best_primal={lag.best_primal}"
    elif method == "admm":
        from src.solve.admm import solve_admm, to_vehicle_routes as a_routes
        ad = solve_admm(inst, max_iterations=20, rho=5.0, label_budget=200000,
                        top_k=8, parallel=False, measure_parallel_timing=False)
        wall = time.time() - t0
        routes = a_routes(ad.vehicle_results)
        out["extra"] = f"iterations={ad.iterations_run} converged={ad.converged}"
    elif method == "ensemble":
        from src.solve.ensemble import solve_ensemble, to_vehicle_routes as e_routes
        solver = HighsSolver(time_limit_s=30.0, mip_rel_gap=0.0)
        en = solve_ensemble(inst, solver=solver, max_rounds=100, rho=5.0,
                            heuristic_label_budget=200000, top_k=20, parallel=False,
                            wall_budget_s=B)
        wall = time.time() - t0
        routes = e_routes(en.per_vehicle_selected)
        out["internal_status"] = f"stop={en.stop_reason}"
        out["internal_timeout"] = en.stop_reason == "wall_budget_exceeded"
        out["extra"] = f"rounds={en.rounds} stop={en.stop_reason}"
    else:
        raise SystemExit(f"unknown method {method}")

    if method == "greedy" and budget_msgs:
        out["internal_status"] = "soft wall budget exceeded"
        out["internal_timeout"] = True

    tc = time.time()
    rep = check_solution(inst, routes)
    served_by = {r.id: 0 for r in inst.requests}
    for rt in routes.values():
        for u in rt.served:
            served_by[u] = served_by.get(u, 0) + 1
    exactly1 = [u for u in served_by if served_by[u] == 1]
    struct = [v for v in rep.violations if "served by NO vehicle" not in v]
    for v in struct[:10]:
        print(f"  STRUCT: {v}", flush=True)
    out.update({"served": len(exactly1), "total": len(inst.requests), "struct": len(struct),
                "method_wall_s": round(wall, 2), "checker_wall_s": round(time.time() - tc, 2),
                "unserved": sorted(u for u in served_by if served_by[u] != 1),
                "budget_messages": budget_msgs})
    print("RESULT_JSON: " + json.dumps(out), flush=True)
    return 0


def run_selftest_child() -> int:
    gc = subprocess.Popen([sys.executable, "-B", os.path.abspath(__file__), "--selftest-grandchild"])
    print(f"SELFTEST child pid={os.getpid()} started grandchild pid={gc.pid}", flush=True)
    time.sleep(600)
    return 0


# --------------------------------------------------------------------------- parent
class Log:
    def __init__(self, path):
        self.f = open(path, "x", encoding="utf-8", newline="\n")   # never overwrite
        self.lock = threading.Lock()

    def w(self, msg):
        line = f"{datetime.datetime.now().isoformat(timespec='milliseconds')} {msg}"
        with self.lock:
            self.f.write(line + "\n"); self.f.flush()
        try:
            print(line, flush=True)
        except Exception:
            pass


def alive(pid):
    """True only if the process is still running (exit code STILL_ACTIVE)."""
    import win32api, win32process
    try:
        h = win32api.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    except Exception:
        return False
    try:
        return win32process.GetExitCodeProcess(h) == 259
    except Exception:
        return False
    finally:
        win32api.CloseHandle(h)


def py_procs(psutil):
    out = {}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            n = (p.info["name"] or "").lower()
            if (n.startswith("python") or "highs" in n) and alive(p.info["pid"]):
                out[p.info["pid"]] = (n, " ".join(p.info["cmdline"] or [])[:200])
        except Exception:
            pass
    return out


def make_job(win32job):
    job = win32job.CreateJobObject(None, "")
    info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    info["BasicLimitInformation"]["LimitFlags"] = win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
    return job


def job_active(win32job, job):
    return win32job.QueryInformationJobObject(job, win32job.JobObjectBasicAccountingInformation)["ActiveProcesses"]


def launch_in_job(cmd, log, tag, psutil, win32job, win32api, win32con):
    job = make_job(win32job)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         creationflags=CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP)
    h = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, p.pid)
    win32job.AssignProcessToJobObject(job, h)
    lines = []

    def reader():
        for raw in iter(p.stdout.readline, b""):
            s = raw.decode("utf-8", "replace").rstrip("\r\n")
            lines.append(s)
            log.w(f"[{tag}] {s}")
    th = threading.Thread(target=reader, daemon=True)
    psutil.Process(p.pid).resume()
    t0 = time.monotonic()
    th.start()
    return job, h, p, th, lines, t0


def survivor_check(log, psutil, win32job, job, child_pid, seen_desc, before, driver_pid):
    """True if nothing survived: job empty, no descendant alive, no new python/highs process."""
    ok = True
    deadline = time.monotonic() + 15
    while job_active(win32job, job) and time.monotonic() < deadline:
        time.sleep(0.5)
    act = job_active(win32job, job)
    log.w(f"SURVIVOR CHECK: job ActiveProcesses={act}")
    ok &= act == 0
    for pid in sorted({child_pid} | seen_desc):
        if alive(pid):
            log.w(f"SURVIVOR CHECK: FAIL - pid {pid} (child or descendant) still alive")
            ok = False
    new = {pid: v for pid, v in py_procs(psutil).items() if pid not in before and pid != driver_pid}
    for pid, (n, cmd) in new.items():
        log.w(f"SURVIVOR CHECK: FAIL - new {n} pid={pid} cmd={cmd}")
        ok = False
    log.w(f"SURVIVOR CHECK: {'PASS' if ok else 'FAIL'} (tracked descendants={sorted(seen_desc)}, new python/highs={len(new)})")
    return ok


def contention_check(log, psutil, driver_pid):
    cpu = psutil.cpu_percent(interval=1.0)
    procs = py_procs(psutil)
    others = {pid: v for pid, v in procs.items() if pid != driver_pid}
    log.w(f"CONTENTION CHECK: system CPU={cpu:.1f}% ; other python/highs processes={len(others)}")
    clash = False
    for pid, (n, cmd) in others.items():
        log.w(f"CONTENTION CHECK:   pid={pid} {n} cmd={cmd}")
        if REPO.lower().replace("\\", "/") in cmd.lower().replace("\\", "/"):
            clash = True
    return clash, procs


def md5_12(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:12]


def on_ac(psutil):
    b = psutil.sensors_battery()
    return None if b is None else bool(b.power_plugged)


def wait_for_ac(log, psutil):
    waited = 0
    while on_ac(psutil) is False:
        if waited % 60 == 0:
            log.w(f"POWER: on battery -- waiting for mains power before launching (waited {waited}s)")
        time.sleep(5); waited += 5
    log.w(f"POWER: on mains power (power_plugged={on_ac(psutil)})")


def _wevt(query, count, newest_first=False):
    args = ["wevtutil", "qe", "System", f"/q:{query}", "/f:xml", f"/c:{count}"]
    if newest_first:
        args.append("/rd:true")
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=60)
    except Exception as e:
        return None, f"wevtutil failed: {e}"
    if out.returncode != 0:
        return None, f"wevtutil exit {out.returncode}: {out.stderr.strip()[:200]}"
    return out.stdout, None


def power_events(t_start_utc, t_end_utc):
    """Kernel-Power 105 (power source change), 506/507 (enter/exit Modern Standby),
    42 (sleep), 1/107 (resume) inside [start, end]."""
    fmt = "%Y-%m-%dT%H:%M:%S.000Z"
    q = ("*[System[Provider[@Name='Microsoft-Windows-Kernel-Power'] and "
         "(EventID=105 or EventID=506 or EventID=507 or EventID=42 or EventID=107 or EventID=1) and "
         f"TimeCreated[@SystemTime>='{t_start_utc.strftime(fmt)}' and @SystemTime<='{t_end_utc.strftime(fmt)}']]]")
    xml, err = _wevt(q, 500)
    if err:
        return [], err
    return [f"{m.group(2)[11:19]}Z id={m.group(1)}"
            for m in re.finditer(r"<EventID>(\d+)</EventID>.*?<TimeCreated SystemTime='([^']+)'", xml, re.S)], None


def in_standby_at(t_utc):
    """Inside Modern Standby at t? Decided by the last 506 (enter) / 507 (exit) event before t."""
    q = ("*[System[Provider[@Name='Microsoft-Windows-Kernel-Power'] and (EventID=506 or EventID=507) and "
         f"TimeCreated[@SystemTime<'{t_utc.strftime('%Y-%m-%dT%H:%M:%S.000Z')}']]]")
    xml, err = _wevt(q, 1, newest_first=True)
    if err:
        return False, "", err
    m = re.search(r"<EventID>(\d+)</EventID>.*?<TimeCreated SystemTime='([^']+)'", xml, re.S)
    if not m:
        return False, "no prior 506/507 event", None
    return m.group(1) == "506", f"last prior event id={m.group(1)} at {m.group(2)[11:19]}Z", None


def main(cap: float, methods: list, tag: str) -> int:
    if os.name != "nt":
        print("BLOCKED: the external hard cap uses Windows Job Objects and the Windows event log; "
              "this driver runs on Windows only.", flush=True)
        return 10
    import importlib.metadata as md
    import psutil, win32api, win32con, win32job

    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method: {m} (choose from {METHODS})")
    if "cluster" in methods and not os.path.exists(CLASSIFICATION):
        raise SystemExit(f"missing {CLASSIFICATION}: run experiments/run_classify_requests.py --requests 30 first")
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, f"baseline_{tag}.log")
    csv_path = os.path.join(OUT_DIR, f"baseline_{tag}.csv")

    sets = ctypes.windll.kernel32.SetThreadExecutionState
    sets.argtypes, sets.restype = [ctypes.c_uint], ctypes.c_uint
    keep_awake_ok = sets(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED) != 0
    log = Log(log_path)
    me = os.getpid()
    log.w(f"=== R=30 BASELINE COMPARISON: external hard cap {cap:.0f} s per method ===")
    log.w(f"methods={','.join(methods)} ; sequential ; each method once ; method wall budget B={B}")
    log.w(f"python={sys.version.split()[0]} highspy={md.version('highspy')} psutil={psutil.__version__} logical_cpus={psutil.cpu_count()}")
    log.w(f"instance={os.path.relpath(os.path.join(DATASET, 'requests_30.csv'), REPO)} md5={md5_12(os.path.join(DATASET, 'requests_30.csv'))}")
    if os.path.exists(CLASSIFICATION):
        log.w(f"proposed-method candidate input={os.path.relpath(CLASSIFICATION, REPO)} md5={md5_12(CLASSIFICATION)}")
    log.w(f"keep-awake: SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED) accepted={keep_awake_ok}")
    log.w("validity: VALID only if (a) not in Modern Standby at the start of the run, (b) no Kernel-Power "
          "105/506/507/42/107/1 event inside the run window, (c) on mains power at every 5 s poll; else INVALID; "
          "UNVERIFIED if the event log cannot be read")

    # self-test: job kill + survivor verification on a child that starts a grandchild
    log.w("--- SELF-TEST: job kill + survivor verification ---")
    before = py_procs(psutil)
    job, h, p, th, lines, t0 = launch_in_job(
        [sys.executable, "-B", os.path.abspath(__file__), "--selftest-child"], log, "selftest",
        psutil, win32job, win32api, win32con)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and job_active(win32job, job) < 2:
        time.sleep(0.25)
    try:
        seen = {c.pid for c in psutil.Process(p.pid).children(recursive=True)}
    except psutil.NoSuchProcess:
        seen = set()
    n_active = job_active(win32job, job)
    log.w(f"SELF-TEST: job ActiveProcesses before kill={n_active} ; descendants={sorted(seen)}")
    win32job.TerminateJobObject(job, 1)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    th.join(timeout=5)
    passed = survivor_check(log, psutil, win32job, job, p.pid, seen, before, me) and n_active >= 2 and len(seen) >= 1
    win32api.CloseHandle(h); win32api.CloseHandle(job)
    log.w(f"SELF-TEST RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        log.w("ABORT: self-test failed -- no method was run")
        return 2

    fields = ["method", "outcome", "limit_fired", "internal_status", "served", "total",
              "structural_violations", "elapsed_wall_s", "method_wall_s", "cap_reached",
              "exit_code", "survivor_check", "start_time", "end_time", "unserved_ids", "extra",
              "validity", "validity_evidence"]
    with open(csv_path, "x", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for method in methods:
        log.w(f"=== METHOD {method}: start ===")
        wait_for_ac(log, psutil)
        clash, before = contention_check(log, psutil, me)
        if clash:
            log.w("ABORT: another process from this repository is running -- not launching")
            return 3
        cmd = [sys.executable, "-B", os.path.abspath(__file__), "--child", method]
        start = datetime.datetime.now().isoformat(timespec="seconds")
        start_utc = datetime.datetime.utcnow() - datetime.timedelta(seconds=1)
        job, h, p, th, lines, t0 = launch_in_job(cmd, log, method, psutil, win32job, win32api, win32con)
        seen, killed, battery_samples = set(), False, 0
        while True:
            try:
                p.wait(timeout=min(5.0, max(0.0, cap - (time.monotonic() - t0))))
                break
            except subprocess.TimeoutExpired:
                if on_ac(psutil) is False:
                    battery_samples += 1
                    if battery_samples == 1:
                        log.w(f"POWER: {method} -- machine is ON BATTERY during the run (run will be INVALID)")
                try:
                    seen |= {c.pid for c in psutil.Process(p.pid).children(recursive=True)}
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() - t0 >= cap:
                    win32job.TerminateJobObject(job, 1)
                    killed = True
                    log.w(f"EXTERNAL CAP: {cap}s reached -> TerminateJobObject on {method}")
                    try:
                        p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        log.w("EXTERNAL CAP: child did not exit within 30 s after job termination")
                    break
        elapsed = time.monotonic() - t0
        th.join(timeout=10)
        end = datetime.datetime.now().isoformat(timespec="seconds")
        ok = survivor_check(log, psutil, win32job, job, p.pid, seen, before, me)
        win32api.CloseHandle(h); win32api.CloseHandle(job)

        end_utc = datetime.datetime.utcnow() + datetime.timedelta(seconds=1)
        events, ev_err = power_events(start_utc, end_utc)
        sb_start, sb_desc, sb_err = in_standby_at(start_utc)
        if ev_err or sb_err:
            validity, evidence = "UNVERIFIED", f"power event log unreadable: {ev_err or sb_err}"
        elif events or battery_samples or sb_start:
            validity = "INVALID"
            evidence = (f"in standby at start: {sb_start} ({sb_desc}) ; Kernel-Power events in window: "
                        f"{events or 'none'} ; battery samples: {battery_samples}")
        else:
            validity = "VALID"
            evidence = (f"awake at start ({sb_desc}) ; no Kernel-Power event in "
                        f"{start_utc.strftime('%H:%M:%S')}Z-{end_utc.strftime('%H:%M:%S')}Z ; on mains power at every 5 s poll")
        log.w(f"VALIDITY {method}: {validity} -- {evidence}")

        res = None
        for s in lines:
            if s.startswith("RESULT_JSON: "):
                res = json.loads(s[len("RESULT_JSON: "):])
        row = {k: "" for k in fields}
        row.update(method=method, elapsed_wall_s=f"{elapsed:.2f}", exit_code=p.returncode,
                   cap_reached="yes" if killed else "no", survivor_check="PASS" if ok else "FAIL",
                   start_time=start, end_time=end, validity=validity, validity_evidence=evidence)
        if killed:
            row.update(outcome=f"cap bound, killed at {cap:.0f} s, no verified result",
                       limit_fired=f"EXTERNAL_KILL_{cap:.0f}")
        elif res is not None and p.returncode == 0:
            row.update(outcome="INTERNAL_TIMEOUT" if res["internal_timeout"] else "COMPLETED",
                       limit_fired="INTERNAL_TIMEOUT" if res["internal_timeout"] else "none",
                       internal_status=res["internal_status"], served=f"{res['served']}/{res['total']}",
                       total=res["total"], structural_violations=res["struct"],
                       method_wall_s=res["method_wall_s"],
                       unserved_ids=";".join(map(str, res["unserved"])), extra=res["extra"])
        else:
            row.update(outcome="CRASHED", limit_fired="none")
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        log.w(f"METHOD_SUMMARY {method}: outcome={row['outcome']} limit_fired={row['limit_fired']} "
              f"served={row['served'] or '(blank)'} "
              f"struct={row['structural_violations'] if row['structural_violations'] != '' else '(blank)'} "
              f"elapsed={row['elapsed_wall_s']}s method_wall={row['method_wall_s'] or '(n/a)'}s "
              f"exit={row['exit_code']} cap_reached={row['cap_reached']} survivor_check={row['survivor_check']} "
              f"internal_status={row['internal_status'] or '-'} extra={row['extra'] or '-'} validity={validity}")
        if not ok:
            log.w("ABORT: a process survived -- stopping before the next method")
            return 4

    log.w(f"=== DONE: {len(methods)} method(s): {','.join(methods)} ===")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        try:
            sys.exit(run_child(sys.argv[2]))
        except Exception:
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            sys.exit(1)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--selftest-child":
        sys.exit(run_selftest_child())
    elif len(sys.argv) >= 2 and sys.argv[1] == "--selftest-grandchild":
        time.sleep(600)
        sys.exit(0)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cap", type=float, default=1800.0, help="external hard cap per method, seconds")
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    sys.exit(main(a.cap, [m for m in a.methods.split(",") if m], a.tag or f"r30_cap{a.cap:.0f}s"))
