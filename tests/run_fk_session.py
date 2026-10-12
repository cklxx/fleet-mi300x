#!/usr/bin/env python3
"""One precommitted F -> K -> conditional FK session, with durable raw logs.

Historical $2.99/hour (README.md:49), estimates, not a live price:
environment 2m + current prerequisite 5m + F 15m + K 10m + FK 10m +
final validation 3m + export/delete 5m = HARD 50m ($2.49).
Cold prepared-input setup allowance 15m: HARD 65m ($3.24), including setup.
--rental-start is REQUIRED: elapsed provisioning/setup consumes that clock.
No provisioner or cold bootstrap is invoked. Inputs must already be prepared.
All work stages have caps; child deadlines are expected time + stated margin.
The remote worker stops before the 5m export/delete reserve. Nothing is timed
with --repeat medians or stamped binaries. Every verdict uses external pairs.

Before provisioning: commit this script and its adjacent manifest; run
  .venv/bin/python -u tests/run_fk_session.py --dry-run
  .venv/bin/python -u tests/run_fk_session.py --self-test
After preparing the committed checkout and inputs on one MI300X:
  .venv/bin/python -u tests/run_fk_session.py --host "$GPU_SSH" \
    --remote-root /home/ubuntu/fleet-mi300x --rental-start "$RENTAL_START_UTC" \
    --delete-command "$PROVIDER_DELETE_ARGV" --cold
The deletion command is an operator-supplied shell-quoted argv, NOT shell code.
It must delete exactly this rented machine, not merely power it off.
No hardware claims are made by --dry-run or --self-test.
"""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_suffix(".manifest.json")
sys.path.insert(0, str(ROOT / "tests"))
from run_local_assembly_gpu import checked_decode  # noqa: E402
from taskgraph import PACK_FIELDS, PACK_FORMAT, QABS_SLICES, TaskKind  # noqa: E402


class Stop(RuntimeError):
    """Infrastructure, correctness or hard-clock failure: abort the session."""


class Reject(RuntimeError):
    """Measured negative result: record it, then try the next independent arm."""


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path, deadline=float("inf")):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if time.monotonic() >= deadline:
                raise Stop("hard clock reached during checksum verification")
            h.update(chunk)
    return h.hexdigest()


def plan_config():
    cfg = json.loads(MANIFEST.read_text())
    if sum(cfg["budget_minutes"].values()) != cfg["warm_hard_minutes"]:
        raise Stop("manifest budget does not sum to hard cap")
    if [a["name"] for a in cfg["arms"]] != ["F", "K", "FK"] or cfg["pairs"] < 3:
        raise Stop("manifest arm order/pair count invalid")
    return cfg


class Runner:
    """Stream merged stdout/stderr byte-for-byte to a log and the terminal."""

    def __init__(self, out, cfg, seconds, dry=False):
        self.out, self.cfg, self.dry = out, cfg, dry
        self.deadline = time.monotonic() + seconds
        self.stage_deadline = self.deadline
        self.steps = []
        if not dry:
            out.mkdir(parents=True, exist_ok=False)
            (out / "logs").mkdir()
            (out / "artifacts").mkdir()

    def say(self, message):
        line = f"[{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}] {message}\n"
        if not self.dry:
            with (self.out / "session.log").open("a") as f:
                f.write(line)
        try:
            print(line, end="", flush=True)
        except BrokenPipeError:
            pass  # The remote bundle remains the authoritative log.

    def stage(self, name):
        seconds = self.cfg["budget_minutes"][name] * 60
        self.stage_deadline = min(self.deadline, time.monotonic() + seconds)
        self.say(f"STAGE {name}; allocation={seconds}s; hard remaining={self.deadline-time.monotonic():.1f}s")

    def probe(self, name, pid):
        self.say(f"PROBE {name}: child pid={pid}; capturing process state (not filtering output)")
        commands = [["ps", "-p", str(pid), "-o", "pid,ppid,etime,state,command"]]
        if shutil.which("rocm-smi"):
            commands.append(["rocm-smi"])
        for i, command in enumerate(commands):
            log = self.out / "logs" / f"{name}.probe-{i}.log"
            with log.open("ab") as f:
                f.write(("$ " + shlex.join(command) + "\n").encode())
                try:
                    p = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                    try:
                        p.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait()
                except OSError as exc:
                    f.write(str(exc).encode())
            self.say(f"probe saved: {log}")

    def run(self, command, name, kind="decode", *, cwd=ROOT, seconds=None):
        command = list(map(str, command))
        expected, cap = self.cfg["steps_seconds"][kind]
        if seconds is not None:
            cap = seconds
        self.say(f"STEP {name}; expected={expected}s; deadline<={cap}s; $ {shlex.join(command)}")
        row = dict(name=name, command=command, expected_seconds=expected, cap_seconds=cap)
        self.steps.append(row)
        if self.dry:
            return ""
        now = time.monotonic()
        end = min(self.deadline, self.stage_deadline, now + cap)
        if end <= now:
            raise Stop(f"hard/stage clock expired before {name}")
        log = self.out / "logs" / f"{name}.log"
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        # No diagnostic environment override may silently change the baseline.
        for key in tuple(env):
            if key.startswith("FLEET_"):
                del env[key]
        # C/C++ stdio is line buffered, Python unbuffered. No grep/tail filters.
        actual = (["stdbuf", "-oL", "-eL"] if shutil.which("stdbuf") else []) + command
        output = bytearray()
        failure = None
        with log.open("wb") as sink:
            proc = subprocess.Popen(actual, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            last_output, heartbeat, probed, killed_at = now, now, False, None
            try:
                while selector.get_map() or proc.poll() is None:
                    current = time.monotonic()
                    if current >= end and failure is None:
                        failure = f"{name}: deadline reached; elapsed={current-now:.1f}s; pid={proc.pid}"
                        self.say(failure)
                        self.probe(name, proc.pid)
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        killed_at = time.monotonic()
                    if killed_at is not None and current-killed_at >= 1:
                        self.say(f"{name}: bounded post-kill drain ended; closing remaining pipes")
                        break
                    if current - heartbeat >= self.cfg["heartbeat_seconds"]:
                        self.say(f"RUNNING {name}: elapsed={current-now:.1f}s, silent={current-last_output:.1f}s, pid={proc.pid}")
                        heartbeat = current
                    if (not probed and current-now >= expected and
                            current-last_output >= self.cfg["silence_probe_margin_seconds"]):
                        self.probe(name, proc.pid)
                        probed = True
                    events = selector.select(0.2) if selector.get_map() else []
                    if not selector.get_map() and proc.poll() is None:
                        time.sleep(0.05)
                    for key, _ in events:
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        last_output = time.monotonic()
                        sink.write(chunk)
                        sink.flush()
                        output.extend(chunk)
                        try:
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
                        except BrokenPipeError:
                            pass
                try:
                    rc = proc.wait(timeout=1)
                except subprocess.TimeoutExpired as exc:
                    raise Stop(f"{name}: PID {proc.pid} did not reap after kill; machine cleanup required") from exc
            except BaseException:
                self.probe(name, proc.pid)
                raise
            finally:
                selector.close()
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        self.say(f"UNREAPED {name}: pid={proc.pid}; terminating wait at deadline")
                proc.stdout.close()
                row.update(returncode=proc.returncode, elapsed_seconds=time.monotonic()-now,
                           log=str(log.relative_to(self.out)))
                save(self.out / "steps.json", self.steps)
        self.say(f"END {name}: exit={rc}, elapsed={row['elapsed_seconds']:.2f}s")
        if failure or rc:
            raise Stop(failure or f"{name}: exit {rc}; full log {log}")
        return output.decode(errors="replace")


def verdict(pairs, cfg):
    if len(pairs) < cfg["pairs"] or any(not math.isfinite(x) for pair in pairs for x in pair):
        raise Stop("incomplete/nonfinite external pairs")
    # Decimal input precision avoids rejecting an exact 0.030 ms boundary
    # because binary float subtraction produces 0.0299999999999998.
    deltas = [Decimal(str(b))-Decimal(str(a)) for b, a in pairs]
    median, spread = statistics.median(deltas), max(deltas)-min(deltas)
    rule = cfg["verdict"]
    keep = (all(x > 0 for x in deltas) and median >= Decimal(str(rule["minimum_median_delta_ms"]))
            and median > rule["median_strictly_exceeds_spread_times"]*spread)
    return dict(pairs_ms=pairs, deltas_ms=list(map(float, deltas)), median_delta_ms=float(median),
                spread_ms=float(spread), keep=keep)


def strict_decode(data, output, mode):
    checked_decode(data, output)
    if data.get("repeat") != 1 or data.get("mode") != mode:
        raise Stop("wrong decode mode/repeat")
    if any(not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0 for x in data["ms"]):
        raise Stop("invalid token latency")
    # JSON's layers_ok is the full-precision launcher comparison; printed
    # values are rounded, so never substitute rounded numbers for that gate.
    rows = re.findall(r"layer\s+(\d+)\s+\w+\s*:\s*max rel\s+(\S+)\s+cosine\s+(\S+)\s+\[(\w+)\]", output)
    if len(rows) != 27 or {int(r[0]) for r in rows} != set(range(27)):
        raise Stop("missing/duplicate section 6 layer results")
    if any(r[3] != "ok" or not all(math.isfinite(float(x)) for x in r[1:3]) for r in rows):
        raise Stop("failed/nonfinite section 6 layer results")
    if "gate: max rel <= 2e-2, cosine >= 0.999" not in output:
        raise Stop("missing section 6 threshold declaration")


def trace_rows(graph, trace):
    tasks = [dict(zip(PACK_FIELDS, v)) for v in struct.iter_unpack(PACK_FORMAT, graph.read_bytes())]
    stamps = list(struct.iter_unpack("<4Q", trace.read_bytes()))
    if len(tasks) != len(stamps) or not tasks:
        raise Stop("trace/graph length mismatch")
    rows = list(zip(tasks, stamps))
    if any(t["index"] != i or t["xcd"] != s[3] & 255 or not s[0]
           or not s[0] <= s[1] <= s[2] for i, (t, s) in enumerate(rows)):
        raise Stop("missing/nonmonotonic trace stamps")
    return rows


def trace_metrics(graph, trace, feature):
    """Use original GLOBAL event anchor, never the added local publication.

    Consumer phase maxima include all XCDs, queueing, acquisition, assembler,
    local publication, reader waits and staging. K separately measures kv_post
    and body-ready (q_c wait/gather included). q_c task signal is an observed
    proxy for publication; stamp overhead and exact hardware overlap remain U.
    """
    rows = trace_rows(graph, trace)
    producers = {}
    publishers = {}
    for task, stamp in rows:
        producers.setdefault(task["signal_event"], []).append((task, stamp))
        if task["kind"] == TaskKind.Q_ABSORB:
            publishers.setdefault((task["layer"], task["head"]), []).append((task, stamp))
    selected = []
    for t, s in rows:
        if feature == "F":
            wanted = (t["kind"] == TaskKind.NORM_ROUTER or t["kind"] == TaskKind.LM_HEAD
                      or t["kind"] == TaskKind.QKV_FUSED and t["layer"] >= 2)
        else:
            wanted = t["kind"] == TaskKind.ATTENTION
        if not wanted:
            continue
        source = producers[t["wait_event"]]
        local_done = None
        if source[0][0]["kind"] == TaskKind.REDUCE:
            local_done = max(ss[2] for st, ss in source if st["xcd"] == t["xcd"])
            source = producers[source[0][0]["wait_event"]]
        anchor = max(ss[2] for _, ss in source)
        stage = (s[3] >> 8) & 0xffffff
        body = (s[3] >> 32) & 0xffffff
        if feature == "F" and t["kind"] == TaskKind.LM_HEAD:
            stage = body
        if not stage or not body or s[1]+body > s[2] or stage > body:
            raise Stop("missing/invalid prologue stamps")
        ready = s[1]+stage
        if ready < anchor:
            raise Stop("reader predates global producer completion")
        record = dict(kind=int(t["kind"]), layer=t["layer"], xcd=s[3] & 255,
                      task=t["index"], worker=t["worker"], head=t["head"], chunk=t["kv_chunk"],
                      assembly_done_us=None if local_done is None else (local_done-anchor)*.01,
                      reader_ready_us=(s[1]-anchor)*.01, stage_us=(ready-anchor)*.01,
                      body_us=(s[1]+body-anchor)*.01)
        if feature == "K":
            q = publishers.get((t["layer"], t["head"]), [])
            if len(q) != QABS_SLICES or any(qt["signal_event"] != t["local_event"] for qt, _ in q):
                raise Stop("current q_c publication graph missing matching slices/head")
            # Union of publisher activity intervals intersected with kv_post.
            intervals = sorted((max(s[1], qs[1]), min(ready, qs[2])) for _, qs in q)
            overlap, end = 0, 0
            for left, right in intervals:
                if right > max(left, end):
                    overlap += right-max(left, end)
                end = max(end, right)
            qdone = max(qs[2] for _, qs in q)
            record.update(overlap_us=overlap*.01,
                          qc_done_us=(qdone-anchor)*.01,
                          kv_post_start_us=(s[1]-anchor)*.01,
                          late_after_qc_us=max(0, s[1]-qdone)*.01)
        selected.append(record)
    phases, xcds = {}, {}
    for row in selected:
        key = f"{row['kind']}:{row['layer']}"
        xkey = f"{key}:{row['xcd']}"
        for target, k in ((phases, key), (xcds, xkey)):
            item = target.setdefault(k, dict(stage_us=0., body_us=0., overlap_us=0.))
            for metric in ("stage_us", "body_us"):
                item[metric] = max(item[metric], row[metric])
            item["overlap_us"] += row.get("overlap_us", 0.)
    expected = 53 if feature == "F" else 27
    if len(phases) != expected or len(xcds) != expected*8:
        raise Stop(f"incomplete {feature} trace phases/XCDs: {len(phases)}/{len(xcds)}")
    return dict(stage_sum_us=sum(v["stage_us"] for v in phases.values()),
                body_sum_us=sum(v["body_us"] for v in phases.values()),
                phases=phases, per_xcd=xcds, readers=selected)


def screen_metrics(candidate, control, feature):
    reasons = []
    if candidate["stage_sum_us"] >= control["stage_sum_us"]:
        reasons.append("complete reader-ready/kv_post interval not shorter")
    if feature == "K":
        if candidate["body_sum_us"] >= control["body_sum_us"]:
            reasons.append("gather/publication/q_c wait not repaid at attention body-ready")
        def key(row):
            return (row["layer"], row["head"], row["chunk"])
        readers = {key(row): row for row in candidate["readers"]}
        if set(readers) != {key(row) for row in control["readers"]}:
            raise Stop("K candidate/control reader identities differ")
        for before in control["readers"]:
            if before["overlap_us"] > 0 and readers[key(before)]["overlap_us"] <= 0:
                reasons.append(f"q_c/kv_post overlap lost at layer/head/chunk {key(before)}")
    return reasons


class Session:
    def __init__(self, runner):
        self.r, self.cfg, self.out = runner, runner.cfg, runner.out
        self.art = self.out / "artifacts"
        self.graphs, self.binaries, self.arms = {}, {}, {}
        self.captures = []

    def graph(self, name, flags):
        path = self.art / f"graph_{name}.bin"
        self.r.run([sys.executable, "src/host/taskgraph.py", "--kv-chunks", "16",
                    *flags, "--emit", path, "--report"], "graph_"+name, "graph")
        self.graphs[name] = path

    def compile(self, name, defines):
        path = self.art / ("decode_"+name)
        self.r.run(["hipcc", "--offload-arch=gfx942", "-O3", "-std=c++17",
                    "-Rpass-analysis=kernel-resource-usage", "-DFLEET_NT_WEIGHTS=1",
                    *["-D"+d+"=1" for d in defines], "-Isrc",
                    "src/host/fleet_launch.hip", "src/kernels/fleet_kernel.hip", "-o", path],
                   "compile_"+name, "compile")
        self.binaries[name] = path

    def decode(self, binary, graph, name, *, proof=False, teacher=False, trace=False, dump=False):
        path = self.art / (name+".json")
        command = [self.binaries[binary], "--graph", self.graphs[graph], "--tokens", "32",
                   "--context", "1024", "--tokens-per-launch", "32", "--repeat", "1", "--json", path]
        if teacher:
            command.append("--teacher-force")
        if trace:
            command += ["--trace", self.art / (name+".trace")]
            self.captures.append((name, graph))
        if dump:
            command += ["--dump-layers", self.art / (name+".layers")]
        output = self.r.run(command, name)
        if self.r.dry:
            return dict(median_ms=3.5, produced=list(range(32)))
        if binary != "B":
            expected = f"local assembly: fold={int('F' in graph)} kva={int('K' in graph)} payload-verification={int(proof)}"
            if expected not in output:
                raise Stop(f"{name}: wrong matching feature/proof banner, expected {expected}")
        data = json.loads(path.read_text())
        strict_decode(data, output, "teacher" if teacher else "free")
        return data

    def correctness(self, binary, graph, name):
        self.decode(binary, graph, name+"_teacher", teacher=True)
        results = [self.decode(binary, graph, name+f"_det{i}", dump=True) for i in range(3)]
        if not self.r.dry:
            dumps = [(self.art / (name+f"_det{i}.layers")).read_bytes() for i in range(3)]
            if len(dumps[0]) != 27*2048*4 or any(x != dumps[0] for x in dumps[1:]):
                raise Stop(name+": nondeterministic/incomplete first-step layer dumps")
            if any(x["produced"] != results[0]["produced"] for x in results[1:]):
                raise Stop(name+": nondeterministic token IDs")
        self.r.say(f"GATE {name}: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens")

    def arm(self, spec):
        name = spec["name"]
        self.r.stage(name)
        self.arms[name] = dict(status="incomplete")
        self.graph(name, spec["graph_flags"])
        self.compile(name, spec["defines"])
        self.compile(name+"_verify", spec["defines"]+["FLEET_VERIFY_LOCAL_ASSEMBLY"])
        # Loaded consumer visibility probes precede full timing runs.
        self.decode(name+"_verify", name, name+"_payload_free", proof=True)
        self.decode(name+"_verify", name, name+"_payload_teacher", proof=True, teacher=True)
        controls = {"B": ("B", "B"), "C": (name, "F" if name == "FK" else "B")}
        configurations = {**controls, "A": (name, name)}
        if name == "FK":
            configurations.update(Off=(name, "B"), K_only=(name, "K"))
        for label, (binary, graph) in configurations.items():
            self.decode(binary, graph, name+"_trace_"+label, trace=True)
        screens = {}
        for feature in name:
            metrics = {}
            for label, (_, graph) in configurations.items():
                if not self.r.dry:
                    metrics[label] = trace_metrics(self.graphs[graph], self.art / (name+"_trace_"+label+".trace"), feature)
            references = (["B", "Off", "K_only" if feature == "F" else "C"]
                          if name == "FK" else ["B", "C"])
            if not self.r.dry:
                failures = {ref: screen_metrics(metrics["A"], metrics[ref], feature) for ref in references}
                screens[feature] = dict(metrics=metrics, rejection_reasons=failures)
            self.r.say(f"SCREEN {name}/{feature}: full interval vs {references}; "
                       + ("including 2304 B read + 2304 B written/XCD/layer and q_c overlap" if feature == "K" else "assembly/publication/wait/staging charged"))
        if not self.r.dry:
            save(self.art / (name+"_screens.json"), screens)
            if any(reasons for v in screens.values() for reasons in v["rejection_reasons"].values()):
                raise Reject(f"{name}: reader-ready/copy/overlap screen rejected; see {name}_screens.json")
        # Warm and prove every configuration before any performance pairing.
        for label, (binary, graph) in configurations.items():
            self.correctness(binary, graph, name+"_gate_"+label)
        outcomes = {}
        for label, (binary, graph) in controls.items():
            pairs = []
            for i in range(self.cfg["pairs"]):
                order = (label, "A") if i % 2 == 0 else ("A", label)
                data = {}
                for item in order:
                    bn, gn = (binary, graph) if item == label else (name, name)
                    data[item] = self.decode(bn, gn, f"{name}_pair_{label}_{i}_{item}")
                pairs.append((data[label]["median_ms"], data["A"]["median_ms"]))
            if not self.r.dry:
                outcomes[label] = verdict(pairs, self.cfg)
        if not self.r.dry:
            save(self.art / (name+"_pairs.json"), outcomes)
            self.arms[name]["pairs"] = outcomes
            if not all(x["keep"] for x in outcomes.values()):
                raise Reject(f"{name}: fixed external-pair timing rule failed")
        self.arms[name]["status"] = "planned" if self.r.dry else "kept"
        self.r.say(f"ARM {name}: {self.arms[name]['status']}")

    def execute(self, expected_commit):
        r = self.r
        r.stage("environment")
        verify_checkout(r, expected_commit)
        r.run(["hipcc", "--version"], "environment_hipcc", "environment")
        gpu_info = r.run(["rocminfo"], "environment_gpu", "environment")
        r.run(["df", "-h", ".", str(self.out)], "environment_disk", "environment")
        if not r.dry:
            if "gfx942" not in gpu_info:
                raise Stop("rocminfo did not identify gfx942")
            if not shutil.which("stdbuf"):
                raise Stop("GNU stdbuf required for line-buffered native output")
            for name in self.cfg["inputs"]:
                if not (ROOT / name).is_file() or (ROOT / name).stat().st_size == 0:
                    raise Stop(f"missing prepared input {name}")
            if (ROOT / "build/golden_hidden.bin").stat().st_size != 27*2048*4:
                raise Stop("golden_hidden is not fp32 [27][2048]")
            if len((ROOT / "build/golden_tokens.txt").read_text().split()) < 33:
                raise Stop("need initial token plus 32 HF golden tokens")
            save(self.out / "inputs.json", {n: dict(bytes=(ROOT/n).stat().st_size,
                 mtime_ns=(ROOT/n).stat().st_mtime_ns) for n in self.cfg["inputs"]})
        r.stage("prerequisite")
        r.run([sys.executable, str(Path(__file__)), "--native-check-stream", "--out", self.art / "native_check"],
              "native_check", "native_check")
        self.graph("B", [])
        self.compile("B", [])
        r.run(["hipcc", "--offload-arch=gfx942", "-O3", "-std=c++17", "bench/microbench.hip",
               "-o", self.art / "microbench"], "compile_microbench", "compile")
        r.run([self.art / "microbench", "--json", self.art / "microbench.json"], "microbench", "microbench")
        if not r.dry:
            m = json.loads((self.art / "microbench.json").read_text())
            if "gfx942" not in m["arch"] or m["cus"] != 304:
                raise Stop("environment is not the required gfx942/304-CU machine")
            if any(m[k] != 0 for k in ("payload_stale_words", "payload_atomic_stale_words")) or not all(
                    m[k] is True for k in ("payload_cross_xcd", "payload_atomic_cross_xcd")):
                raise Stop("cross-XCD payload microbench gate failed")
        # Mandatory current-build/current-graph prerequisite, before any arm.
        self.decode("B", "B", "prerequisite_trace", trace=True)
        self.decode("B", "B", "prerequisite_unstamped")
        r.say("PREREQUISITE complete: current v0.18 graph/build trace + unstamped baseline; no gain credit")
        self.correctness("B", "B", "prerequisite_gate")
        for spec in self.cfg["arms"]:
            name = spec["name"]
            if spec.get("requires_kept") and not all(
                    self.arms.get(x, {}).get("status") in ("kept", "planned") for x in spec["requires_kept"]):
                self.arms[name] = dict(status="skipped", reason="F and K did not both survive")
                r.say("SKIP FK: F and K did not both survive")
                continue
            try:
                self.arm(spec)
            except Reject as exc:
                self.arms[name].update(status="rejected", reason=str(exc))
                r.say(f"REJECT {exc}")
            if not r.dry:
                save(self.out / "arms.json", self.arms)
        r.stage("final_validation")
        self.correctness("B", "B", "final_B")
        for name, result in self.arms.items():
            if result["status"] in ("kept", "planned"):
                self.correctness(name, name, "final_"+name)
        # Nice-to-have text captures LAST; reuse traces, no extra GPU time.
        for name, graph in self.captures:
            if not r.dry and min(r.deadline, r.stage_deadline)-time.monotonic() < 10:
                r.say("SKIP remaining optional timelines: reserve report/export time")
                break
            r.run([sys.executable, "scripts/trace_timeline.py", self.art / (name+".trace"),
                   self.graphs[graph], "--layer", "5"], "timeline_"+name, "capture")


def verify_checkout(runner, expected=None):
    head = runner.run(["git", "rev-parse", "HEAD"], "checkout_head", "environment").strip()
    dirty = runner.run(["git", "status", "--porcelain", "--untracked-files=no"], "checkout_status", "environment")
    # ls-files alone is insufficient: exact working bytes must match HEAD.
    for path in (Path(__file__), MANIFEST):
        runner.run(["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
                   "tracked_"+path.suffix.lstrip("."), "environment")
    if not runner.dry and (dirty or expected and head != expected):
        raise Stop("requires clean tracked checkout at the same committed HEAD")
    return head


def seal_bundle(out, result):
    save(out / "result.json", result)
    (out / "report.md").write_text(
        "# F/K session result\n\n" + f"Status: **{result['status']}**.\n\n"
        + "```json\n" + json.dumps(result, indent=2) + "\n```\n\n"
        + "Raw stdout/stderr: logs/. Commands/deadlines: steps.json. "
        "Graphs/binaries/traces/proofs/layer dumps/pairs: artifacts/.\n\n"
        "[U] Single-token traces are attribution screens, not timing verdicts; "
        "signal stamps approximate event publication, not hardware cache residency. "
        "Task-op/router/logit-margin section 6 subgates have no new per-op dump here. "
        "Kept means the documented full-decode gates and paired rule passed; "
        "it does not change shipped defaults.\n")
    inventory = {str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=digest(p))
                 for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256.json"}
    save(out / "SHA256.json", inventory)


def verify_bundle(out, deadline=float("inf")):
    inventory = json.loads((out / "SHA256.json").read_text())
    for required in ("result.json", "report.md", "session.log", "steps.json", "manifest.json"):
        if required not in inventory:
            raise Stop("export missing required file " + required)
    actual = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "SHA256.json"}
    if actual != set(inventory):
        raise Stop("export file inventory mismatch")
    for name, entry in inventory.items():
        path = out / name
        if path.is_symlink() or path.resolve().is_relative_to(out.resolve()) is False:
            raise Stop("unsafe exported path")
        if path.stat().st_size != entry["bytes"] or digest(path, deadline) != entry["sha256"]:
            raise Stop("export checksum failed: " + name)
    return len(inventory)


def worker(args, cfg):
    r = Runner(args.out, cfg, args.worker_seconds, args.dry_run)
    session = Session(r)
    result = dict(status="incomplete", arms=session.arms, commit=args.expected_commit,
                  session_id=args.session_id)
    previous_signals = {}
    if not r.dry:
        shutil.copy2(MANIFEST, r.out / "manifest.json")
        save(r.out / "worker.json", dict(pid=os.getpid(), session_id=args.session_id))
    try:
        if not r.dry:
            def expired(signum, frame):
                raise Stop(f"worker signal {signum}: clock/stop requested; sealing logs")
            for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                previous_signals[sig] = signal.getsignal(sig)
                signal.signal(sig, expired)
            signal.setitimer(signal.ITIMER_REAL, args.worker_seconds)
        session.execute(args.expected_commit)
        result["status"] = "dry-run" if r.dry else "completed"
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="aborted", error=f"{type(exc).__name__}: {exc}")
        # No keep verdict may survive incomplete final validation/session abort.
        for arm in session.arms.values():
            if arm["status"] == "kept":
                arm["status"] = "incomplete"
        r.say(result["error"])
    finally:
        if not r.dry:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)
            r.say("WORKER DONE: sealing complete raw bundle before deletion")
            if not (r.out / "steps.json").exists():
                save(r.out / "steps.json", r.steps)
            try:
                seal_bundle(r.out, result)
            finally:
                for sig, handler in previous_signals.items():
                    signal.signal(sig, handler)
    if r.dry:
        r.say("DRY RUN COMPLETE: all conditional arms shown; zero subprocess/GPU/SSH/delete calls; no files written")
    return int(result["status"] == "aborted")


def stop_worker(args):
    """Bounded remote interruption recovery; only the session's own PID."""
    if (args.out / "SHA256.json").is_file():
        print("worker already sealed", flush=True)
        return 0
    info = json.loads((args.out / "worker.json").read_text())
    if info["session_id"] != args.session_id:
        raise Stop("stop request session mismatch")
    cmdline = Path(f"/proc/{info['pid']}/cmdline").read_bytes().split(b"\0")
    if args.session_id.encode() not in cmdline or b"--worker" not in cmdline:
        raise Stop("worker PID identity mismatch; refusing to signal it")
    os.kill(info["pid"], signal.SIGTERM)
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        if (args.out / "SHA256.json").is_file():
            print("worker stopped and sealed", flush=True)
            return 0
        time.sleep(.2)
    raise Stop("worker has not sealed after stop; preserve machine for recovery")


def native_check_stream(out):
    """Reuse the repository byte-comparison gate with streaming compile logs.

    Its original run() captures compiler output until completion. Replace only
    that transport so an interrupted compiler's entire output is retained in
    both commands.log and the enclosing session log. Gate/build logic is intact.
    The outer Runner owns process-group deadlines for this whole subtree.
    """
    import check_default_build as check

    def stream(command, cwd, log, stdin=None):
        if stdin is not None:
            raise Stop("native byte-comparison unexpectedly requested compiler stdin")
        with log.open("ab") as sink:
            header = (f"cwd={cwd}\n$ {shlex.join(command)}\n").encode()
            sink.write(header)
            sys.stdout.buffer.write(header)
            sink.flush()
            sys.stdout.buffer.flush()
            p = subprocess.Popen(["stdbuf", "-oL", "-eL", *command], cwd=cwd,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while True:
                chunk = os.read(p.stdout.fileno(), 65536)
                if not chunk:
                    break
                sink.write(chunk)
                sink.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            rc = p.wait()
            p.stdout.close()
            sink.write(f"\nexit={rc}\n".encode())
            if rc:
                raise Stop(f"native comparison compiler exited {rc}")

    check.run = stream
    sys.argv = ["check_default_build.py", "--native", "--out", str(out)]
    return check.main()


def ssh_command(host, root, args):
    # One safely quoted remote shell command; no interpolation of argv as code.
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
            "cd " + shlex.quote(str(root)) + " && " + shlex.join(args)]


def controller(args, cfg):
    start = dt.datetime.fromisoformat(args.rental_start.replace("Z", "+00:00"))
    if start.tzinfo is None:
        raise Stop("--rental-start needs an explicit UTC offset")
    elapsed = (dt.datetime.now(dt.timezone.utc)-start).total_seconds()
    if elapsed < 0:
        raise Stop("rental start cannot be in the future")
    cap = cfg["cold_hard_minutes"] if args.cold else cfg["warm_hard_minutes"]
    remaining = cap*60-elapsed
    reserve = cfg["budget_minutes"]["export_delete"]*60
    local = args.out.resolve()
    if not local.is_relative_to(ROOT / "results"):
        raise Stop("controller --out must be inside this repository's results/")
    r = Runner(local, cfg, max(0, remaining))
    session_id = uuid.uuid4().hex
    remote = Path("/tmp") / ("fleet-fk-"+session_id)
    bundle = local / "remote"
    result = dict(status="incomplete", remote_directory=str(remote), host=args.host,
                  session_id=session_id)
    remote_started = False
    try:
        if remaining <= reserve+10:
            raise Stop("rental clock already consumed work allowance; no new GPU work started")
        head = verify_checkout(r)
        # Use the same immutable revision; runner itself verifies remote tracked state.
        remote_started = True
        window = min(45*60, r.deadline-time.monotonic()-reserve)
        seconds = window-30  # reserve seal/hash/probe grace before starting export
        if seconds <= 0:
            raise Stop("no worker allowance remains after finalization reserve")
        command = ssh_command(args.host, args.remote_root,
            [".venv/bin/python", "-u", "tests/run_fk_session.py", "--worker", "--out", str(remote),
             "--worker-seconds", str(seconds), "--expected-commit", head, "--session-id", session_id])
        try:
            r.run(command, "remote_session", "transport", seconds=window)
        except (Stop, KeyboardInterrupt) as exc:
            result["worker_error"] = str(exc)
            r.stage_deadline = r.deadline
            r.run(ssh_command(args.host, args.remote_root,
                  [".venv/bin/python", "-u", "tests/run_fk_session.py", "--stop-worker",
                   "--out", str(remote), "--session-id", session_id]),
                  "stop_and_seal", "transport", seconds=35)
        # Both success and failure take the SAME export/verify/delete path.
        r.stage_deadline = r.deadline
        r.say("EXPORT: copying whole remote bundle before any deletion")
        r.run(["scp", "-r", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               f"{args.host}:{remote}", str(bundle)], "export", "transport",
              seconds=max(1, r.deadline-time.monotonic()-50))
        count = verify_bundle(bundle, r.deadline-50)
        remote_result = json.loads((bundle / "result.json").read_text())
        if remote_result.get("session_id") != session_id or remote_result.get("commit") != head:
            raise Stop("export does not belong to this session/revision")
        result.update(export_verified_files=count, worker=remote_result)
        save(local / "export_verified.json", dict(files=count, source=str(remote), verified=True))
        r.say(f"EXPORT VERIFIED: {count} files in repository tree; deleting machine now")
        r.run(shlex.split(args.delete_command), "delete_machine", "delete")
        result.update(status="deleted", delete_exit=0)
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="needs_operator_recovery", error=f"{type(exc).__name__}: {exc}")
        r.say(result["error"])
        if remote_started and "export_verified_files" not in result:
            r.say("DELETION BLOCKED: logs not verified locally. Recover full bundle, verify, then delete immediately.")
            fresh = local / ("recovered-"+uuid.uuid4().hex)
            result["recovery_commands"] = [
                ["scp", "-r", f"{args.host}:{remote}", str(fresh)],
                [sys.executable, "-u", str(Path(__file__)), "--verify-export", "--out", str(fresh),
                 "--session-id", session_id, "--expected-commit", head], shlex.split(args.delete_command)]
            for command in result["recovery_commands"]:
                r.say("RECOVERY (in order; stop if any fails) $ " + shlex.join(command))
        else:
            r.say("Machine deletion has not been confirmed; operator must check provider state immediately.")
    finally:
        save(local / "controller_result.json", result)
        (local / "report.md").write_text("# Session lifecycle\n\n```json\n"+json.dumps(result, indent=2)+"\n```\n")
    return int(result["status"] != "deleted" or result.get("worker", {}).get("status") != "completed")


def self_test(cfg):
    """Fault-inject orchestration locally. No hipcc, ROCm, SSH or provider call."""
    good = verdict([(3.60, 3.56), (3.61, 3.57), (3.59, 3.55)], cfg)
    assert good["keep"]
    assert verdict([(3.60, 3.57)]*3, cfg)["keep"]
    assert not verdict([(3.60, 3.58)]*3, cfg)["keep"]
    assert not verdict([(3.60, 3.50), (3.60, 3.59), (3.60, 3.56)], cfg)["keep"]
    assert not verdict([(3.60, 3.56), (3.60, 3.56), (3.60, 3.61)], cfg)["keep"]
    reader = dict(layer=0, head=0, chunk=0, overlap_us=1)
    before = dict(stage_sum_us=100, body_sum_us=120, readers=[reader])
    after = dict(stage_sum_us=90, body_sum_us=110, readers=[reader])
    assert not screen_metrics(after, before, "K")
    assert screen_metrics(before, before, "F")
    assert screen_metrics({**after, "body_sum_us": 120}, before, "K")
    assert screen_metrics({**after, "readers": [{**reader, "overlap_us": 0}]}, before, "K")
    print("PASS: paired threshold/noise/sign and F/K cost/overlap rejection cases", flush=True)
    with tempfile.TemporaryDirectory(prefix="fleet-fk-selftest-") as temp:
        # Real shipped/combined graph layouts, synthetic clocks only. These
        # exercise counts, event resolution and 2 q_c slices per head without
        # claiming real cache behavior or hardware timing.
        from taskgraph import build, emit, load_cfg
        metrics = {}
        for local in (False, True):
            name = "FK" if local else "B"
            graph = Path(temp)/(name+".bin")
            trace = Path(temp)/(name+".trace")
            g = build(load_cfg(ROOT/"reference/dsv2lite_config.json", 16,
                               local_fold=local, local_kva=local))
            emit(g, graph)
            tasks = [dict(zip(PACK_FIELDS, v)) for v in struct.iter_unpack(PACK_FORMAT, graph.read_bytes())]
            sources, stamps = {}, {}
            for i, task in enumerate(tasks):
                sources.setdefault(task["signal_event"], []).append(i)

            def clock_for(index):
                if index in stamps:
                    return stamps[index]
                task = tasks[index]
                wait = task["wait_event"]
                anchor = max((clock_for(i)[2] for i in sources.get(wait, []) if wait >= 0), default=1000)
                ready = anchor+10
                stage = 80 if task["flags"] & (4096 | 8192) else 200
                body = stage+50
                if task["kind"] == TaskKind.REDUCE:
                    stage, body = 0, 20
                elif task["kind"] == TaskKind.Q_ABSORB:
                    stage, body = 0, 400
                elif task["kind"] == TaskKind.ATTENTION:
                    body = 350 if local else 500
                packed = task["xcd"] | (stage << 8) | (body << 32)
                stamps[index] = (ready-1, ready, ready+body+10, packed)
                return stamps[index]

            trace.write_bytes(b"".join(struct.pack("<4Q", *clock_for(i)) for i in range(len(tasks))))
            metrics[name] = {f: trace_metrics(graph, trace, f) for f in ("F", "K")}
        for feature in ("F", "K"):
            assert not screen_metrics(metrics["FK"][feature], metrics["B"][feature], feature)
            assert screen_metrics(metrics["B"][feature], metrics["FK"][feature], feature)
        damaged = bytearray(trace.read_bytes())
        damaged[24] ^= 1  # xcd stamp disagrees with descriptor
        trace.write_bytes(damaged)
        try:
            trace_rows(graph, trace)
            raise AssertionError("mismatched graph/trace accepted")
        except Stop:
            pass
        print("PASS: real B/FK layouts + synthetic clocks; 53 F phases/424 XCDs; 27 K phases/216 XCDs; graph/trace corruption", flush=True)
        output = "tokens: 32/32 match HF greedy\nconsecutive layers within the gate: 27 of 27\n"
        output += "gate: max rel <= 2e-2, cosine >= 0.999\n"
        output += "\n".join(f"layer {i} moe: max rel 1e-3 cosine 0.9999 [ok]" for i in range(27))
        data = dict(tokens=32, context=1024, tokens_per_launch=32, matched=32, compared=32,
                    layers=27, layers_within_gate=27, produced=list(range(32)), ms=[3.595]*32,
                    median_ms=3.595, repeat=1, mode="free")
        strict_decode(data, output, "free")
        for changed, text in (({**data, "layers_within_gate": 26}, output),
                              ({**data, "repeat": 3}, output),
                              (data, output.replace("layer 26", "layer 25")),
                              (data, output.replace("1e-3", "nan")),
                              (data, output.replace("[ok]", "[FAIL]"))):
            try:
                strict_decode(changed, text, "free")
                raise AssertionError("bad decode gate accepted")
            except RuntimeError:
                pass
        print("PASS: exact counts, finite/distinct section 6 rows, repeat=1; malformed gates rejected", flush=True)
        r = Runner(Path(temp)/"bundle", cfg, 30)
        shutil.copy2(MANIFEST, r.out/"manifest.json")
        text = r.run([sys.executable, "-u", "-c", "import sys; print('stdout full'); print('stderr full', file=sys.stderr)"], "stream", "capture")
        assert "stdout full" in text and "stderr full" in text
        for name, code, cap in (("failure", "raise SystemExit(7)", 5),
                                ("timeout", "import time; print('stuck state', flush=True); time.sleep(30)", .2),
                                ("closed_pipe", "import os,time; os.close(1); os.close(2); time.sleep(30)", .2)):
            try:
                r.run([sys.executable, "-u", "-c", code], name, "capture", seconds=cap)
                raise AssertionError("failure/timeout accepted")
            except Stop:
                pass
        r.say("self-test complete, seal before export")
        seal_bundle(r.out, dict(status="self-test", hardware=False))
        export = Path(temp)/"export"
        shutil.copytree(r.out, export)
        assert verify_bundle(export) > 0
        (export/"result.json").write_text("corrupt")
        try:
            verify_bundle(export)
            raise AssertionError("corrupt export accepted")
        except Stop:
            pass
        # Fault-inject transport/provider boundaries through the real controller.
        # Never launch SSH or a delete command in self-test.
        from unittest.mock import patch
        def clock_abort(session, expected):
            session.r.say("SELF-TEST worker Python analysis clock")
            time.sleep(5)
        worker_args = argparse.Namespace(out=Path(temp)/"worker-clock", worker_seconds=.15,
            dry_run=False, expected_commit="test-commit", session_id="test-clock")
        with patch.object(Session, "execute", clock_abort):
            assert worker(worker_args, cfg) == 1
        assert json.loads((worker_args.out/"result.json").read_text())["status"] == "aborted"
        assert verify_bundle(worker_args.out) > 0
        print("PASS: hard worker clock during Python analysis still seals report and full bundle", flush=True)
        for fault in ("none", "corrupt_export", "worker_failure"):
            test_root = Path(temp).resolve()/fault
            test_root.mkdir()
            observed = []
            current = {}

            def fake_run(self, command, name, kind="decode", **kwargs):
                observed.append(name)
                self.steps.append(dict(name=name, command=list(map(str, command))))
                save(self.out/"steps.json", self.steps)
                self.say("SELF-TEST mocked boundary " + name)
                if name == "remote_session":
                    words = shlex.split(command[-1])
                    current["session_id"] = words[words.index("--session-id")+1]
                    if fault == "worker_failure":
                        raise Stop("synthetic worker failure")
                elif name == "export":
                    target = Path(command[-1])
                    shutil.copytree(r.out, target)
                    seal_bundle(target, dict(status="aborted" if fault == "worker_failure" else "completed",
                                             session_id=current["session_id"], commit="test-commit"))
                    if fault == "corrupt_export":
                        (target/"result.json").write_text("bad transfer")
                elif name == "delete_machine":
                    assert (self.out/"export_verified.json").is_file()
                return ""

            args = argparse.Namespace(rental_start=dt.datetime.now(dt.timezone.utc).isoformat(),
                cold=False, out=test_root/"results"/"session", host="test-host", remote_root=Path("/prepared"),
                delete_command="test-delete exact-machine")
            with patch.object(sys.modules[__name__], "ROOT", test_root), \
                    patch.object(sys.modules[__name__], "verify_checkout", return_value="test-commit"), \
                    patch.object(Runner, "run", fake_run):
                rc = controller(args, cfg)
            assert rc == (0 if fault == "none" else 1)
            assert ("delete_machine" in observed) == (fault != "corrupt_export")
            assert (args.out/"report.md").is_file()
            if fault == "worker_failure":
                assert observed.index("stop_and_seal") < observed.index("export")
        print("PASS: controller success/failure cleanup; stop-before-export; export/verify-before-delete; corrupt export blocks delete", flush=True)
    print("PASS: live stdout/stderr, nonzero exit, deadline kill/probe, report on stop, export hashes/corruption", flush=True)
    print("SELF-TEST PASS: all non-GPU checks completed; no hardware or performance claim", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--stop-worker", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--verify-export", action="store_true", help="verify recovered bundle before manual deletion")
    mode.add_argument("--native-check-stream", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--host", help="SSH alias/user@host of already prepared MI300X")
    ap.add_argument("--remote-root", type=Path, default=Path("/home/ubuntu/fleet-mi300x"))
    ap.add_argument("--rental-start", help="actual provisioning/billing start, ISO-8601 with timezone")
    ap.add_argument("--delete-command", help="quoted provider CLI argv for deleting this exact machine")
    ap.add_argument("--cold", action="store_true", help="65m total cap including <=15m prepared-input setup")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--worker-seconds", type=float, default=45*60, help=argparse.SUPPRESS)
    ap.add_argument("--expected-commit", help=argparse.SUPPRESS)
    ap.add_argument("--session-id", help=argparse.SUPPRESS)
    args = ap.parse_args()
    cfg = plan_config()
    if args.self_test:
        return self_test(cfg)
    if args.native_check_stream:
        if args.out is None:
            ap.error("native-check-stream requires out")
        return native_check_stream(args.out)
    if args.verify_export:
        if args.out is None or not args.session_id or not args.expected_commit:
            ap.error("verify-export requires out, session-id and expected-commit")
        count = verify_bundle(args.out)
        result = json.loads((args.out / "result.json").read_text())
        if result.get("session_id") != args.session_id or result.get("commit") != args.expected_commit:
            raise Stop("recovered bundle session/revision mismatch")
        print(f"EXPORT VERIFIED: {count} files; session {args.session_id}; now delete exact machine", flush=True)
        return 0
    if args.stop_worker:
        if not args.session_id or args.out != Path("/tmp")/("fleet-fk-"+args.session_id):
            ap.error("stop-worker requires exact session directory/id")
        return stop_worker(args)
    if args.dry_run:
        args.out = Path("/tmp/fleet-fk-DRY-RUN")
        print("PLAN: 2+5+15+10+10+3+5=50 warm minutes/$2.49; cold hard 65m/$3.24; historical $2.99/h")
        print("PLAN: local clean committed HEAD -> SSH same revision -> remote worker <=45m -> export whole bundle -> SHA256 verify -> delete exact machine")
        rc = worker(args, cfg)
        print("PLAN CLEANUP (also on failure): scp full directory into results/<session>/remote; verify every SHA256; execute operator delete argv; write controller report")
        print("PLAN FAIL-CLOSED: failed export/hash => NO DELETE, print recovery command; hard clock ends new work and writes report")
        return rc
    if args.worker:
        if args.out is None or not args.expected_commit or not args.session_id or not 0 < args.worker_seconds <= 45*60:
            ap.error("worker requires out, expected commit and 0 < seconds <= 2700")
        return worker(args, cfg)
    if not all((args.host, args.rental_start, args.delete_command)):
        ap.error("controller requires --host, --rental-start and --delete-command")
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", args.host) or args.host.startswith("-"):
        ap.error("invalid SSH host")
    if not args.remote_root.is_absolute() or not shlex.split(args.delete_command):
        ap.error("absolute remote-root and nonempty delete argv required")
    args.out = args.out or ROOT / "results" / ("fk-"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    return controller(args, cfg)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
