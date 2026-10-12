# F/K session preparation — 2026-10-12

Prepared and committed the executable session script and adjacent manifest in **8213362a83bc5a329039c7e24a62b5dfca467ede**. The payload retry passed without changing its code; the repository suite passed **22/22**. This is a preparation result: no GPU session, provisioning, provider deletion, or new performance measurement was performed. The executed commands and their outputs are reproduced below. Implementation: `tests/run_fk_session.py:1`; manifest: `tests/run_fk_session.manifest.json:1`.

The recorded shipped result remains **3.593 / 3.595 / 3.596 ms/token**, approximately 278 tok/s, with 32/32 tokens and 27/27 layers. It is historical context, not a control for this experiment. The runner obtains a current build/current graph trace and a fresh unstamped baseline before F. The retained old phase table predates q_c publication. (`docs/STATUS.md:24`, `docs/STATUS.md:80`; `tests/run_fk_session.py:515`.)

**Operator command**

Run the following from the controller's repository after preparing the same committed checkout and inputs on the MI300X. Set `GPU_SSH` to its SSH alias, `REMOTE_REPO` to the absolute remote repository path, `RENTAL_START_UTC` to the actual provisioning/billing start such as an ISO-8601 UTC timestamp, and `PROVIDER_DELETE_ARGV` to the provider CLI command that deletes this exact machine. The latter is parsed as shell-quoted argv, so it must not contain shell pipelines or an unexpanded variable. These machine-specific values are intentionally supplied by the operator; the session does not choose or provision a machine. (`tests/run_fk_session.py:16`, `tests/run_fk_session.py:699`, `tests/run_fk_session.py:705`, `tests/run_fk_session.py:760`, `tests/run_fk_session.py:948`.)

```bash
cd /Users/ckl/code/fleet-mi300x
: "${GPU_SSH:?set the exact rented machine SSH alias}"
: "${REMOTE_REPO:?set the absolute remote repository path}"
: "${RENTAL_START_UTC:?set the actual billing start with timezone}"
: "${PROVIDER_DELETE_ARGV:?set the provider deletion argv for this exact machine}"
.venv/bin/python -u tests/run_fk_session.py \
  --host "$GPU_SSH" \
  --remote-root "$REMOTE_REPO" \
  --rental-start "$RENTAL_START_UTC" \
  --delete-command "$PROVIDER_DELETE_ARGV" \
  --cold
```

Omit `--cold` for the 50-minute warm cap; with it, the cap is 65 minutes **including already elapsed setup**, never 65 new minutes after setup. Both modes bound remote work to at most 45 minutes including a 30-second sealing allowance, then reserve five minutes for export/deletion. Keep the controller running until it reports the verified export and deletion outcome. (`tests/run_fk_session.py:705`, `tests/run_fk_session.py:731`.)

Prerequisites: prepared Linux MI300X/gfx942 environment, `hipcc`, `rocminfo`, GNU `stdbuf`, SSH/SCP, and a working remote `.venv/bin/python`; use the repository's Python 3.12 setup. The checkout must include the historical baseline commit `85b087e223d43c1e677db19f9cdd74ca73b78532` used by the native byte-comparison gate. Both tracked checkouts must be clean and at the same HEAD. Required inputs are `build/weights.bin`, its `.manifest`, `fleet_cache.bin`, `golden_tokens.txt` (at least 33 IDs), and `golden_hidden.bin` (221,184 bytes). The session neither downloads the model nor invokes the legacy bootstrap, whose measurement flow filters logs and includes unrelated arms. (`scripts/hotaisle_bootstrap.sh:20`, `scripts/hotaisle_bootstrap.sh:36`; `tests/check_default_build.py:34`, `tests/check_default_build.py:156`; `tests/run_fk_session.py:479`, `tests/run_fk_session.py:548`; `tests/run_fk_session.manifest.json:60`.)

**Fixed order and budgets**

The allocations below are [E], at the repository's **historical $2.99/hour**; they are not a current quote or measured execution times. Header and machine-readable caps agree. (`README.md:49`, `README.md:91`; `tests/run_fk_session.py:4`; `tests/run_fk_session.manifest.json:4`.)

- Environment: **2 minutes / $0.10**.
- Shared prerequisite: **5 minutes / $0.25**. Stream the existing native v0.18/default/explicit-off byte-comparison gate, build the current shipped NT executable/default 16-chunk graph, run the repository microbench, collect the current trace and unstamped baseline, and check teacher/free determinism. This prerequisite earns no gain credit. (`tests/run_fk_session.py:479`, `tests/run_fk_session.py:659`.)
- **F: 15 minutes / $0.75**, then **K: 10 minutes / $0.50**. Each builds its feature-capable executable and its separate payload-verification executable, probes actual consumers in free/teacher modes, screens traces, checks correctness/determinism, then runs external timing pairs. A measured rejection of F still permits K; infrastructure or correctness failures stop the session. (`tests/run_fk_session.py:423`, `tests/run_fk_session.py:520`.)
- **FK: 10 minutes / $0.50**, only if both F and K survive. It has its own combined binaries and graph. (`tests/run_fk_session.manifest.json:50`; `tests/run_fk_session.py:520`.)
- Final validation: **3 minutes / $0.15** for the shipped control and each kept arm. Optional layer-5 text timelines use already recorded traces and run last, only while time remains. (`tests/run_fk_session.py:534`.)
- Export and deletion: **5 minutes / $0.25**. Total **50 warm minutes / $2.49**. Cold prepared-input setup allowance adds **15 minutes / $0.75**, for **65 minutes / $3.24** including setup; totals are rounded after calculation. (`tests/run_fk_session.manifest.json:5`; `tests/run_fk_session.py:705`.)

Each command logs its expected duration and deadline. Examples are compile 30/210 seconds, decode 10/190 seconds, and native comparison 90/270 seconds; the active stage and hard clocks can shorten these. Progress is emitted every 30 seconds. Silence triggers bounded process/GPU-state probes, and timeout handling prints the stuck PID, probes it, kills its process group, and bounds pipe draining even after stdout closes. A worker alarm also covers Python analysis. Termination/interruption routes through report sealing; the controller requests a session-specific stop and waits for sealing before exporting an interrupted worker. (`tests/run_fk_session.manifest.json:19`; `tests/run_fk_session.py:86`, `tests/run_fk_session.py:594`, `tests/run_fk_session.py:638`, `tests/run_fk_session.py:738`.)

**Controls, correctness, and the fixed verdict**

B is a fresh shipped NT build with the current default 16-chunk, q_c-published graph. F compares against B and F-capable/default-graph control; K compares against B and K-capable/default-graph control. FK compares against B and **combined-binary/F-only-graph control**, preserving the incremental K comparison. Combined-binary/all-off and K-only configurations additionally support attribution and correctness checks. All feature launches verify the fold/kva/proof banner. The all-off executable is never used to diagnose a graph carrying new feature flags. (`tests/run_fk_session.py:381`, `tests/run_fk_session.py:390`, `tests/run_fk_session.py:433`; `results/first-two-2026-10-12.md:75`.)

For **each required control**, run three independent pairs in control/candidate, candidate/control, control/candidate order. Every process uses `--repeat 1`; no paired verdict process uses `--trace` or a proof executable. Let delta be control `median_ms` minus candidate `median_ms`, in **device ms/token**; the launcher's median is across token positions, not repeated processes. Keep only if **every delta > 0**, **median(delta) >= 0.030 ms**, and **median(delta) > 2 × spread**, where **spread = max(delta) - min(delta)**. Decimal arithmetic preserves the exact 0.030-ms boundary. Both control comparisons must pass. (`tests/run_fk_session.manifest.json:31`; `tests/run_fk_session.py:229`, `tests/run_fk_session.py:461`; `src/host/fleet_launch.hip:850`.)

Every decode must satisfy `matched = compared = tokens = 32`, context 1,024, one 32-token launch, and `layers_within_gate = layers = 27`. The checker also requires all 27 distinct, finite, passing layer rows and the section 6 threshold declaration: **max relative error <= 0.02; cosine >= 0.999**. It trusts the launcher's full-precision gate count, rather than recomputing a boundary decision from rounded stdout. Teacher-forced execution plus three free-running processes must produce identical token IDs and bitwise identical first-step layer dumps. A token-only launcher exit zero is insufficient. On a session abort, provisional kept statuses become incomplete. (`docs/design.md:180`, `docs/design.md:184`; `src/host/fleet_launch.hip:818`, `src/host/fleet_launch.hip:883`; `tests/run_fk_session.py:243`, `tests/run_fk_session.py:412`, `tests/run_fk_session.py:617`.)

**Reject conditions actually measured by the script**

F's [M, traced attribution] interval starts at the last original global producer's signal stamp and ends at the last consumer's fold-staging stamp; final LM_HEAD ends at norm-ready because it lacks a separate staging stamp. Resolving through REDUCE reaches the original event, so assembly, publication, acquisition/wait, queue delay, and staging are charged. The screen records all **53 phase maxima / 424 per-XCD intervals** and requires the sum of the 53 maxima to be strictly shorter than each required control. Concurrent XCD intervals are not added together as serial latency. An equal or longer complete interval rejects F before timing. (`tests/run_fk_session.py:272`, `tests/run_fk_session.py:350`; `src/kernels/fleet_kernel.hip:1029`.)

K explicitly charges the logical **2,304 B read + 2,304 B written per XCD/layer**, or **36,864 B/layer and 995,328 B/token**. This is logical traffic, not a measured HBM count. Its trace records original QKV completion, gather/publication completion, attention-ready/kv_post start, kv_post end, q_c publication completion, and attention body-ready. Both 27-phase tail sums—through kv_post and through body-ready after the retained q_c wait/gather—must shrink. It also matches each layer/head/chunk reader and rejects any formerly positive q_c-publisher/kv_post overlap that disappears. That overlap is a conservative concurrency proxy from existing stamps, not an interior compute measurement. (`results/first-two-2026-10-12.md:11`; `tests/run_fk_session.py:272`, `tests/run_fk_session.py:350`; `src/kernels/fleet_kernel.hip:765`, `src/kernels/fleet_kernel.hip:793`.)

FK repeats both screens against B and matching all-off traces, isolates F against the combined-binary/K-only graph, and isolates K against the combined-binary/F-only graph. The timed comparison against F-only must also clear the fixed bar. No additivity is assumed. (`tests/run_fk_session.py:433`; `tests/run_fk_session.manifest.json:58`.)

**Logs and deletion**

Each invocation creates a new local session directory and a nonce-specific remote directory. Commands, flags, graphs, native binaries, compiler/resource output, proof logs, JSON decode results, layer dumps, raw traces, screen details, pair deltas, and derived text timelines are retained. stdout and stderr are merged and streamed without grep/tail filtering. The existing native comparison's compiler transport is wrapped to stream and retain its output even if interrupted; its build and byte-comparison logic is reused. (`tests/run_fk_session.py:86`, `tests/run_fk_session.py:135`, `tests/run_fk_session.py:659`, `tests/run_fk_session.py:719`.)

The resulting layout is defined by `tests/run_fk_session.py:89`, `tests/run_fk_session.py:375`, `tests/run_fk_session.py:560`, and `tests/run_fk_session.py:749`:

```text
results/fk-<UTC timestamp>/
  session.log
  logs/                         # controller checkout, SSH, export, delete, probes
  steps.json
  export_verified.json          # written only after complete hash verification
  controller_result.json
  report.md                     # lifecycle outcome, including recovery on failure
  remote/                       # complete copied /tmp/fleet-fk-<nonce> bundle
    manifest.json
    worker.json                 # worker PID and session identity
    inputs.json                 # input sizes/mtime metadata
    session.log
    logs/                       # full individual command output and stuck-state probes
    steps.json
    arms.json                   # completed arm states; result.json remains authoritative
    result.json
    report.md
    SHA256.json                 # names, lengths and hashes of every bundle file
    artifacts/
      native_check/             # v0.18/default/explicit-off artifacts and full compiler logs
      decode_*; microbench
      graph_*.bin; graph_*.bin.events.txt
      *.json; *.trace; *.layers
      F_screens.json; K_screens.json; FK_screens.json
      F_pairs.json; K_pairs.json; FK_pairs.json
```

Before machine deletion, in this order: **stop/finish the worker; seal its result/report and SHA256 inventory; copy the entire bundle into this repository; verify its complete file set, lengths, hashes, session nonce, and commit; write the local export verification record; invoke the exact-machine deletion command immediately**. The same path runs after a failed or timed-out experiment. An unsuccessful export blocks deletion and emits fresh-destination recovery, verification, and deletion commands in order; the script exits nonzero and retains its report. It does not silently destroy the only copy of measurements. (`tests/run_fk_session.py:560`, `tests/run_fk_session.py:594`, `tests/run_fk_session.py:738`.)

**[U]**

- **GPU gains, native ROCm comparison, real loaded payload visibility, and hardware gates remain unmeasured in this preparation.** Local trace tests use real graph layouts with synthetic clocks; transport/deletion tests use mocks. This distinction is printed by `--self-test` below and encoded at `tests/run_fk_session.py:782`.
- **Exact interior publication/cache costs and complete section 6 task-op/router/logit-margin diagnostics remain unavailable.** Existing traces sample token 1, and the launcher exposes first-step layer outputs and token agreement, not new per-task output dumps. The new runner enforces available full-decode gates and records that scope; it does not claim every section 6 diagnostic has been newly instrumented. (`src/host/fleet_launch.hip:753`; `docs/design.md:178`; `tests/run_fk_session.py:560`.)
- **The newer broad observability contract is not fully implemented here.** New arrive/fetch stamps, per-task NDJSON with measured-byte fields, and ISA-order artifacts would require additional instrumentation/work beyond this brief's allowed edits. This runner preserves existing raw traces, resource/compiler logs, exact commands, and bundle hashes; large model inputs have size/mtime metadata rather than content hashes. The missing records receive no claimed timing credit. (`docs/OBSERVABILITY.md:29`, `docs/OBSERVABILITY.md:43`, `docs/OBSERVABILITY.md:64`; `tests/run_fk_session.py:498`; `/Users/ckl/.hermes/cache/scratch/fleet-session-brief.md:59`.)
- **Provider-specific deletion semantics and network availability are not verified locally.** The caller must supply an authenticated command that really deletes the exact VM; exit zero is the wrapper's success signal. Loss of the remote host, failed export, failed deletion, disk failure, or an invocation after its budget is exhausted can require operator recovery and can extend actual billing. The hard clock stops controlled work; it cannot guarantee both log preservation and timely deletion during an external outage. (`tests/run_fk_session.py:705`, `tests/run_fk_session.py:762`; `README.md:52`.)
- **Allocation estimates and current price remain unverified.** Historical $2.99/hour and approximately 15-minute bootstrap are the supplied planning basis; no new rental was made. (`README.md:49`, `README.md:91`; local-only commands below.)
- **The brief's repository AGENTS.md section 4 was not present as a local file at inspection.** The brief itself supplies the binding one-script/live-output/deadline/order/fetch-before-destroy requirements, which are followed here. (`/Users/ckl/.hermes/cache/scratch/fleet-session-brief.md:18`; initial inspection command `for p in /AGENTS.md /Users/AGENTS.md /Users/ckl/AGENTS.md /Users/ckl/code/AGENTS.md AGENTS.md; do if test -f "$p"; then printf '\\n%s\\n' "$p"; cat "$p"; fi; done` printed only `/Users/ckl/AGENTS.md`, containing the Cloudflare rules.)

**Executed verification**

All commands below ran from `/Users/ckl/code/fleet-mi300x`. The own-suite and standalone payload commands returned 0; the payload source was unchanged. The suite itself abbreviates its payload row; the separate payload retry supplies the full result.

```text
$ .venv/bin/python -u tests/run_all.py

  PASS  test_absorbed_equivalence.py     4/4 passed   (0.1s)
  PASS  test_descriptor_layout.py        6/6 passed   (0.0s)
  PASS  test_expert_addressing.py        11/11 passed   (0.1s)
  PASS  test_kernel_interface.py         19/19 passed   (0.0s)
  PASS  test_local_assembly_gpu_gate.py  OK   (0.1s)
  PASS  test_local_assembly_graph.py     53/53 passed   (27.7s)
  PASS  test_local_assembly_payload.py   PASS: production F folds (3 boundaries x 8 XCDs), K all 8 sh   (0.9s)
  PASS  test_queue_simulation.py         35/35 passed   (18.4s)
  PASS  test_reference_vs_hf.py          8/8 boundaries passed   (3.4s)
  PASS  test_row_partition.py            17/17 passed   (0.1s)
  PASS  test_validator.py                18/18 passed   (15.1s)
  PASS  graph: default (16 KV chunks)    validation: PASS
  PASS  graph: 8 KV chunks               validation: PASS
  PASS  graph: 1 KV chunk                validation: PASS
  PASS  graph: kv_a replicated           validation: PASS
  PASS  graph: top-k published           validation: PASS
  PASS  graph: q_c per task              validation: PASS
  PASS  graph: idle-worker prefetch      validation: PASS
  PASS  graph: o_proj row split          validation: PASS
  PASS  graph: split workers + tiling    validation: PASS
  PASS  HIP parse                        6/6 translation units clean, 0 errors
  PASS  HIP parse (nt weights)           6/6 translation units clean, 0 errors

22/22 gates passed
exit=0

$ .venv/bin/python -u tests/test_local_assembly_payload.py
PASS: production F folds (3 boundaries x 8 XCDs), K all 8 shards x 8 XCDs; bounds, source lifetime, and 160 payload corruptions detected
exit=0
```

Full final self-test output follows. Its injected failures and printed mock deletion steps are expected test cases, not a real provider operation.

```text
$ .venv/bin/python -u tests/run_fk_session.py --self-test
PASS: paired threshold/noise/sign and F/K cost/overlap rejection cases
wrote 54946 descriptors to /var/folders/tb/g6324rs52jz3skf78xwgk8gh0000gn/T/fleet-fk-selftest-j6v6qufg/B.bin (3434 KB) and B.bin.events.txt
wrote 55586 descriptors to /var/folders/tb/g6324rs52jz3skf78xwgk8gh0000gn/T/fleet-fk-selftest-j6v6qufg/FK.bin (3474 KB) and FK.bin.events.txt
PASS: real B/FK layouts + synthetic clocks; 53 F phases/424 XCDs; 27 K phases/216 XCDs; graph/trace corruption
PASS: exact counts, finite/distinct section 6 rows, repeat=1; malformed gates rejected
[2026-10-12T13:07:05+00:00] STEP stream; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python -u -c 'import sys; print('"'"'stdout full'"'"'); print('"'"'stderr full'"'"', file=sys.stderr)'
stdout full
stderr full
[2026-10-12T13:07:05+00:00] END stream: exit=0, elapsed=0.02s
[2026-10-12T13:07:05+00:00] STEP failure; expected=5s; deadline<=5s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python -u -c 'raise SystemExit(7)'
[2026-10-12T13:07:05+00:00] END failure: exit=7, elapsed=0.02s
[2026-10-12T13:07:05+00:00] STEP timeout; expected=5s; deadline<=0.2s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python -u -c 'import time; print('"'"'stuck state'"'"', flush=True); time.sleep(30)'
stuck state
[2026-10-12T13:07:05+00:00] timeout: deadline reached; elapsed=0.2s; pid=45773
[2026-10-12T13:07:05+00:00] PROBE timeout: child pid=45773; capturing process state (not filtering output)
[2026-10-12T13:07:05+00:00] probe saved: /var/folders/tb/g6324rs52jz3skf78xwgk8gh0000gn/T/fleet-fk-selftest-j6v6qufg/bundle/logs/timeout.probe-0.log
[2026-10-12T13:07:05+00:00] END timeout: exit=-9, elapsed=0.25s
[2026-10-12T13:07:05+00:00] STEP closed_pipe; expected=5s; deadline<=0.2s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python -u -c 'import os,time; os.close(1); os.close(2); time.sleep(30)'
[2026-10-12T13:07:06+00:00] closed_pipe: deadline reached; elapsed=0.2s; pid=45775
[2026-10-12T13:07:06+00:00] PROBE closed_pipe: child pid=45775; capturing process state (not filtering output)
[2026-10-12T13:07:06+00:00] probe saved: /var/folders/tb/g6324rs52jz3skf78xwgk8gh0000gn/T/fleet-fk-selftest-j6v6qufg/bundle/logs/closed_pipe.probe-0.log
[2026-10-12T13:07:06+00:00] END closed_pipe: exit=-9, elapsed=0.41s
[2026-10-12T13:07:06+00:00] self-test complete, seal before export
[2026-10-12T13:07:06+00:00] SELF-TEST worker Python analysis clock
[2026-10-12T13:07:06+00:00] Stop: worker signal 14: clock/stop requested; sealing logs
[2026-10-12T13:07:06+00:00] WORKER DONE: sealing complete raw bundle before deletion
PASS: hard worker clock during Python analysis still seals report and full bundle
[2026-10-12T13:07:06+00:00] SELF-TEST mocked boundary remote_session
[2026-10-12T13:07:06+00:00] EXPORT: copying whole remote bundle before any deletion
[2026-10-12T13:07:06+00:00] SELF-TEST mocked boundary export
[2026-10-12T13:07:06+00:00] EXPORT VERIFIED: 11 files in repository tree; deleting machine now
[2026-10-12T13:07:06+00:00] SELF-TEST mocked boundary delete_machine
[2026-10-12T13:07:06+00:00] SELF-TEST mocked boundary remote_session
[2026-10-12T13:07:06+00:00] EXPORT: copying whole remote bundle before any deletion
[2026-10-12T13:07:06+00:00] SELF-TEST mocked boundary export
[2026-10-12T13:07:06+00:00] Stop: export checksum failed: result.json
[2026-10-12T13:07:06+00:00] DELETION BLOCKED: logs not verified locally. Recover full bundle, verify, then delete immediately.
[2026-10-12T13:07:06+00:00] RECOVERY (in order; stop if any fails) $ scp -r test-host:/tmp/fleet-fk-91c04507324a4f79bd978de4675a7eda /private/var/folders/tb/g6324rs52jz3skf78xwgk8gh0000gn/T/fleet-fk-selftest-j6v6qufg/corrupt_export/results/session/recovered-1b82e99239f547198d43174d97608553
[2026-10-12T13:07:06+00:00] RECOVERY (in order; stop if any fails) $ /Users/ckl/code/fleet-mi300x/.venv/bin/python -u /Users/ckl/code/fleet-mi300x/tests/run_fk_session.py --verify-export --out /private/var/folders/tb/g6324rs52jz3skf78xwgk8gh0000gn/T/fleet-fk-selftest-j6v6qufg/corrupt_export/results/session/recovered-1b82e99239f547198d43174d97608553 --session-id 91c04507324a4f79bd978de4675a7eda --expected-commit test-commit
[2026-10-12T13:07:06+00:00] RECOVERY (in order; stop if any fails) $ test-delete exact-machine
[2026-10-12T13:07:06+00:00] SELF-TEST mocked boundary remote_session
[2026-10-12T13:07:06+00:00] SELF-TEST mocked boundary stop_and_seal
[2026-10-12T13:07:06+00:00] EXPORT: copying whole remote bundle before any deletion
[2026-10-12T13:07:06+00:00] SELF-TEST mocked boundary export
[2026-10-12T13:07:06+00:00] EXPORT VERIFIED: 11 files in repository tree; deleting machine now
[2026-10-12T13:07:06+00:00] SELF-TEST mocked boundary delete_machine
PASS: controller success/failure cleanup; stop-before-export; export/verify-before-delete; corrupt export blocks delete
PASS: live stdout/stderr, nonzero exit, deadline kill/probe, report on stop, export hashes/corruption
SELF-TEST PASS: all non-GPU checks completed; no hardware or performance claim
exit=0
```

Full final dry-run output follows; it expands all conditional arms to make the entire command plan reviewable without subprocess/GPU/SSH/deletion calls.

```text
$ .venv/bin/python -u tests/run_fk_session.py --dry-run
PLAN: 2+5+15+10+10+3+5=50 warm minutes/$2.49; cold hard 65m/$3.24; historical $2.99/h
PLAN: local clean committed HEAD -> SSH same revision -> remote worker <=45m -> export whole bundle -> SHA256 verify -> delete exact machine
[2026-10-12T13:07:02+00:00] STAGE environment; allocation=120s; hard remaining=2700.0s
[2026-10-12T13:07:02+00:00] STEP checkout_head; expected=30s; deadline<=90s; $ git rev-parse HEAD
[2026-10-12T13:07:02+00:00] STEP checkout_status; expected=30s; deadline<=90s; $ git status --porcelain --untracked-files=no
[2026-10-12T13:07:02+00:00] STEP tracked_py; expected=30s; deadline<=90s; $ git ls-files --error-unmatch tests/run_fk_session.py
[2026-10-12T13:07:02+00:00] STEP tracked_json; expected=30s; deadline<=90s; $ git ls-files --error-unmatch tests/run_fk_session.manifest.json
[2026-10-12T13:07:02+00:00] STEP environment_hipcc; expected=30s; deadline<=90s; $ hipcc --version
[2026-10-12T13:07:02+00:00] STEP environment_gpu; expected=30s; deadline<=90s; $ rocminfo
[2026-10-12T13:07:02+00:00] STEP environment_disk; expected=30s; deadline<=90s; $ df -h . /tmp/fleet-fk-DRY-RUN
[2026-10-12T13:07:02+00:00] STAGE prerequisite; allocation=300s; hard remaining=2700.0s
[2026-10-12T13:07:02+00:00] STEP native_check; expected=90s; deadline<=270s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python /Users/ckl/code/fleet-mi300x/tests/run_fk_session.py --native-check-stream --out /tmp/fleet-fk-DRY-RUN/artifacts/native_check
[2026-10-12T13:07:02+00:00] STEP graph_B; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python src/host/taskgraph.py --kv-chunks 16 --emit /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --report
[2026-10-12T13:07:02+00:00] STEP compile_B; expected=30s; deadline<=210s; $ hipcc --offload-arch=gfx942 -O3 -std=c++17 -Rpass-analysis=kernel-resource-usage -DFLEET_NT_WEIGHTS=1 -Isrc src/host/fleet_launch.hip src/kernels/fleet_kernel.hip -o /tmp/fleet-fk-DRY-RUN/artifacts/decode_B
[2026-10-12T13:07:02+00:00] STEP compile_microbench; expected=30s; deadline<=210s; $ hipcc --offload-arch=gfx942 -O3 -std=c++17 bench/microbench.hip -o /tmp/fleet-fk-DRY-RUN/artifacts/microbench
[2026-10-12T13:07:02+00:00] STEP microbench; expected=30s; deadline<=210s; $ /tmp/fleet-fk-DRY-RUN/artifacts/microbench --json /tmp/fleet-fk-DRY-RUN/artifacts/microbench.json
[2026-10-12T13:07:02+00:00] STEP prerequisite_trace; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/prerequisite_trace.json --trace /tmp/fleet-fk-DRY-RUN/artifacts/prerequisite_trace.trace
[2026-10-12T13:07:02+00:00] STEP prerequisite_unstamped; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/prerequisite_unstamped.json
[2026-10-12T13:07:02+00:00] PREREQUISITE complete: current v0.18 graph/build trace + unstamped baseline; no gain credit
[2026-10-12T13:07:02+00:00] STEP prerequisite_gate_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/prerequisite_gate_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP prerequisite_gate_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/prerequisite_gate_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/prerequisite_gate_det0.layers
[2026-10-12T13:07:02+00:00] STEP prerequisite_gate_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/prerequisite_gate_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/prerequisite_gate_det1.layers
[2026-10-12T13:07:02+00:00] STEP prerequisite_gate_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/prerequisite_gate_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/prerequisite_gate_det2.layers
[2026-10-12T13:07:02+00:00] GATE prerequisite_gate: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STAGE F; allocation=900s; hard remaining=2700.0s
[2026-10-12T13:07:02+00:00] STEP graph_F; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python src/host/taskgraph.py --kv-chunks 16 --local-fold --emit /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --report
[2026-10-12T13:07:02+00:00] STEP compile_F; expected=30s; deadline<=210s; $ hipcc --offload-arch=gfx942 -O3 -std=c++17 -Rpass-analysis=kernel-resource-usage -DFLEET_NT_WEIGHTS=1 -DFLEET_LOCAL_FOLD=1 -Isrc src/host/fleet_launch.hip src/kernels/fleet_kernel.hip -o /tmp/fleet-fk-DRY-RUN/artifacts/decode_F
[2026-10-12T13:07:02+00:00] STEP compile_F_verify; expected=30s; deadline<=210s; $ hipcc --offload-arch=gfx942 -O3 -std=c++17 -Rpass-analysis=kernel-resource-usage -DFLEET_NT_WEIGHTS=1 -DFLEET_LOCAL_FOLD=1 -DFLEET_VERIFY_LOCAL_ASSEMBLY=1 -Isrc src/host/fleet_launch.hip src/kernels/fleet_kernel.hip -o /tmp/fleet-fk-DRY-RUN/artifacts/decode_F_verify
[2026-10-12T13:07:02+00:00] STEP F_payload_free; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F_verify --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_payload_free.json
[2026-10-12T13:07:02+00:00] STEP F_payload_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F_verify --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_payload_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP F_trace_B; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_trace_B.json --trace /tmp/fleet-fk-DRY-RUN/artifacts/F_trace_B.trace
[2026-10-12T13:07:02+00:00] STEP F_trace_C; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_trace_C.json --trace /tmp/fleet-fk-DRY-RUN/artifacts/F_trace_C.trace
[2026-10-12T13:07:02+00:00] STEP F_trace_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_trace_A.json --trace /tmp/fleet-fk-DRY-RUN/artifacts/F_trace_A.trace
[2026-10-12T13:07:02+00:00] SCREEN F/F: full interval vs ['B', 'C']; assembly/publication/wait/staging charged
[2026-10-12T13:07:02+00:00] STEP F_gate_B_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_B_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP F_gate_B_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_B_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_B_det0.layers
[2026-10-12T13:07:02+00:00] STEP F_gate_B_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_B_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_B_det1.layers
[2026-10-12T13:07:02+00:00] STEP F_gate_B_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_B_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_B_det2.layers
[2026-10-12T13:07:02+00:00] GATE F_gate_B: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP F_gate_C_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_C_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP F_gate_C_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_C_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_C_det0.layers
[2026-10-12T13:07:02+00:00] STEP F_gate_C_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_C_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_C_det1.layers
[2026-10-12T13:07:02+00:00] STEP F_gate_C_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_C_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_C_det2.layers
[2026-10-12T13:07:02+00:00] GATE F_gate_C: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP F_gate_A_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_A_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP F_gate_A_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_A_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_A_det0.layers
[2026-10-12T13:07:02+00:00] STEP F_gate_A_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_A_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_A_det1.layers
[2026-10-12T13:07:02+00:00] STEP F_gate_A_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_A_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/F_gate_A_det2.layers
[2026-10-12T13:07:02+00:00] GATE F_gate_A: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP F_pair_B_0_B; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_pair_B_0_B.json
[2026-10-12T13:07:02+00:00] STEP F_pair_B_0_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_pair_B_0_A.json
[2026-10-12T13:07:02+00:00] STEP F_pair_B_1_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_pair_B_1_A.json
[2026-10-12T13:07:02+00:00] STEP F_pair_B_1_B; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_pair_B_1_B.json
[2026-10-12T13:07:02+00:00] STEP F_pair_B_2_B; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_pair_B_2_B.json
[2026-10-12T13:07:02+00:00] STEP F_pair_B_2_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_pair_B_2_A.json
[2026-10-12T13:07:02+00:00] STEP F_pair_C_0_C; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_pair_C_0_C.json
[2026-10-12T13:07:02+00:00] STEP F_pair_C_0_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_pair_C_0_A.json
[2026-10-12T13:07:02+00:00] STEP F_pair_C_1_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_pair_C_1_A.json
[2026-10-12T13:07:02+00:00] STEP F_pair_C_1_C; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_pair_C_1_C.json
[2026-10-12T13:07:02+00:00] STEP F_pair_C_2_C; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_pair_C_2_C.json
[2026-10-12T13:07:02+00:00] STEP F_pair_C_2_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/F_pair_C_2_A.json
[2026-10-12T13:07:02+00:00] ARM F: planned
[2026-10-12T13:07:02+00:00] STAGE K; allocation=600s; hard remaining=2700.0s
[2026-10-12T13:07:02+00:00] STEP graph_K; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python src/host/taskgraph.py --kv-chunks 16 --local-kva --emit /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --report
[2026-10-12T13:07:02+00:00] STEP compile_K; expected=30s; deadline<=210s; $ hipcc --offload-arch=gfx942 -O3 -std=c++17 -Rpass-analysis=kernel-resource-usage -DFLEET_NT_WEIGHTS=1 -DFLEET_LOCAL_KVA=1 -Isrc src/host/fleet_launch.hip src/kernels/fleet_kernel.hip -o /tmp/fleet-fk-DRY-RUN/artifacts/decode_K
[2026-10-12T13:07:02+00:00] STEP compile_K_verify; expected=30s; deadline<=210s; $ hipcc --offload-arch=gfx942 -O3 -std=c++17 -Rpass-analysis=kernel-resource-usage -DFLEET_NT_WEIGHTS=1 -DFLEET_LOCAL_KVA=1 -DFLEET_VERIFY_LOCAL_ASSEMBLY=1 -Isrc src/host/fleet_launch.hip src/kernels/fleet_kernel.hip -o /tmp/fleet-fk-DRY-RUN/artifacts/decode_K_verify
[2026-10-12T13:07:02+00:00] STEP K_payload_free; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K_verify --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_payload_free.json
[2026-10-12T13:07:02+00:00] STEP K_payload_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K_verify --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_payload_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP K_trace_B; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_trace_B.json --trace /tmp/fleet-fk-DRY-RUN/artifacts/K_trace_B.trace
[2026-10-12T13:07:02+00:00] STEP K_trace_C; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_trace_C.json --trace /tmp/fleet-fk-DRY-RUN/artifacts/K_trace_C.trace
[2026-10-12T13:07:02+00:00] STEP K_trace_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_trace_A.json --trace /tmp/fleet-fk-DRY-RUN/artifacts/K_trace_A.trace
[2026-10-12T13:07:02+00:00] SCREEN K/K: full interval vs ['B', 'C']; including 2304 B read + 2304 B written/XCD/layer and q_c overlap
[2026-10-12T13:07:02+00:00] STEP K_gate_B_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_B_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP K_gate_B_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_B_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_B_det0.layers
[2026-10-12T13:07:02+00:00] STEP K_gate_B_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_B_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_B_det1.layers
[2026-10-12T13:07:02+00:00] STEP K_gate_B_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_B_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_B_det2.layers
[2026-10-12T13:07:02+00:00] GATE K_gate_B: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP K_gate_C_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_C_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP K_gate_C_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_C_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_C_det0.layers
[2026-10-12T13:07:02+00:00] STEP K_gate_C_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_C_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_C_det1.layers
[2026-10-12T13:07:02+00:00] STEP K_gate_C_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_C_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_C_det2.layers
[2026-10-12T13:07:02+00:00] GATE K_gate_C: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP K_gate_A_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_A_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP K_gate_A_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_A_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_A_det0.layers
[2026-10-12T13:07:02+00:00] STEP K_gate_A_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_A_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_A_det1.layers
[2026-10-12T13:07:02+00:00] STEP K_gate_A_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_A_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/K_gate_A_det2.layers
[2026-10-12T13:07:02+00:00] GATE K_gate_A: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP K_pair_B_0_B; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_pair_B_0_B.json
[2026-10-12T13:07:02+00:00] STEP K_pair_B_0_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_pair_B_0_A.json
[2026-10-12T13:07:02+00:00] STEP K_pair_B_1_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_pair_B_1_A.json
[2026-10-12T13:07:02+00:00] STEP K_pair_B_1_B; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_pair_B_1_B.json
[2026-10-12T13:07:02+00:00] STEP K_pair_B_2_B; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_pair_B_2_B.json
[2026-10-12T13:07:02+00:00] STEP K_pair_B_2_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_pair_B_2_A.json
[2026-10-12T13:07:02+00:00] STEP K_pair_C_0_C; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_pair_C_0_C.json
[2026-10-12T13:07:02+00:00] STEP K_pair_C_0_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_pair_C_0_A.json
[2026-10-12T13:07:02+00:00] STEP K_pair_C_1_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_pair_C_1_A.json
[2026-10-12T13:07:02+00:00] STEP K_pair_C_1_C; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_pair_C_1_C.json
[2026-10-12T13:07:02+00:00] STEP K_pair_C_2_C; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_pair_C_2_C.json
[2026-10-12T13:07:02+00:00] STEP K_pair_C_2_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/K_pair_C_2_A.json
[2026-10-12T13:07:02+00:00] ARM K: planned
[2026-10-12T13:07:02+00:00] STAGE FK; allocation=600s; hard remaining=2700.0s
[2026-10-12T13:07:02+00:00] STEP graph_FK; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python src/host/taskgraph.py --kv-chunks 16 --local-fold --local-kva --emit /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --report
[2026-10-12T13:07:02+00:00] STEP compile_FK; expected=30s; deadline<=210s; $ hipcc --offload-arch=gfx942 -O3 -std=c++17 -Rpass-analysis=kernel-resource-usage -DFLEET_NT_WEIGHTS=1 -DFLEET_LOCAL_FOLD=1 -DFLEET_LOCAL_KVA=1 -Isrc src/host/fleet_launch.hip src/kernels/fleet_kernel.hip -o /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK
[2026-10-12T13:07:02+00:00] STEP compile_FK_verify; expected=30s; deadline<=210s; $ hipcc --offload-arch=gfx942 -O3 -std=c++17 -Rpass-analysis=kernel-resource-usage -DFLEET_NT_WEIGHTS=1 -DFLEET_LOCAL_FOLD=1 -DFLEET_LOCAL_KVA=1 -DFLEET_VERIFY_LOCAL_ASSEMBLY=1 -Isrc src/host/fleet_launch.hip src/kernels/fleet_kernel.hip -o /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK_verify
[2026-10-12T13:07:02+00:00] STEP FK_payload_free; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK_verify --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_payload_free.json
[2026-10-12T13:07:02+00:00] STEP FK_payload_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK_verify --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_payload_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP FK_trace_B; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_B.json --trace /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_B.trace
[2026-10-12T13:07:02+00:00] STEP FK_trace_C; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_C.json --trace /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_C.trace
[2026-10-12T13:07:02+00:00] STEP FK_trace_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_A.json --trace /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_A.trace
[2026-10-12T13:07:02+00:00] STEP FK_trace_Off; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_Off.json --trace /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_Off.trace
[2026-10-12T13:07:02+00:00] STEP FK_trace_K_only; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_K_only.json --trace /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_K_only.trace
[2026-10-12T13:07:02+00:00] SCREEN FK/F: full interval vs ['B', 'Off', 'K_only']; assembly/publication/wait/staging charged
[2026-10-12T13:07:02+00:00] SCREEN FK/K: full interval vs ['B', 'Off', 'C']; including 2304 B read + 2304 B written/XCD/layer and q_c overlap
[2026-10-12T13:07:02+00:00] STEP FK_gate_B_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_B_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP FK_gate_B_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_B_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_B_det0.layers
[2026-10-12T13:07:02+00:00] STEP FK_gate_B_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_B_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_B_det1.layers
[2026-10-12T13:07:02+00:00] STEP FK_gate_B_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_B_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_B_det2.layers
[2026-10-12T13:07:02+00:00] GATE FK_gate_B: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP FK_gate_C_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_C_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP FK_gate_C_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_C_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_C_det0.layers
[2026-10-12T13:07:02+00:00] STEP FK_gate_C_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_C_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_C_det1.layers
[2026-10-12T13:07:02+00:00] STEP FK_gate_C_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_C_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_C_det2.layers
[2026-10-12T13:07:02+00:00] GATE FK_gate_C: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP FK_gate_A_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_A_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP FK_gate_A_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_A_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_A_det0.layers
[2026-10-12T13:07:02+00:00] STEP FK_gate_A_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_A_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_A_det1.layers
[2026-10-12T13:07:02+00:00] STEP FK_gate_A_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_A_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_A_det2.layers
[2026-10-12T13:07:02+00:00] GATE FK_gate_A: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP FK_gate_Off_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_Off_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP FK_gate_Off_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_Off_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_Off_det0.layers
[2026-10-12T13:07:02+00:00] STEP FK_gate_Off_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_Off_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_Off_det1.layers
[2026-10-12T13:07:02+00:00] STEP FK_gate_Off_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_Off_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_Off_det2.layers
[2026-10-12T13:07:02+00:00] GATE FK_gate_Off: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP FK_gate_K_only_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_K_only_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP FK_gate_K_only_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_K_only_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_K_only_det0.layers
[2026-10-12T13:07:02+00:00] STEP FK_gate_K_only_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_K_only_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_K_only_det1.layers
[2026-10-12T13:07:02+00:00] STEP FK_gate_K_only_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_K_only_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/FK_gate_K_only_det2.layers
[2026-10-12T13:07:02+00:00] GATE FK_gate_K_only: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP FK_pair_B_0_B; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_pair_B_0_B.json
[2026-10-12T13:07:02+00:00] STEP FK_pair_B_0_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_pair_B_0_A.json
[2026-10-12T13:07:02+00:00] STEP FK_pair_B_1_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_pair_B_1_A.json
[2026-10-12T13:07:02+00:00] STEP FK_pair_B_1_B; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_pair_B_1_B.json
[2026-10-12T13:07:02+00:00] STEP FK_pair_B_2_B; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_pair_B_2_B.json
[2026-10-12T13:07:02+00:00] STEP FK_pair_B_2_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_pair_B_2_A.json
[2026-10-12T13:07:02+00:00] STEP FK_pair_C_0_C; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_pair_C_0_C.json
[2026-10-12T13:07:02+00:00] STEP FK_pair_C_0_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_pair_C_0_A.json
[2026-10-12T13:07:02+00:00] STEP FK_pair_C_1_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_pair_C_1_A.json
[2026-10-12T13:07:02+00:00] STEP FK_pair_C_1_C; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_pair_C_1_C.json
[2026-10-12T13:07:02+00:00] STEP FK_pair_C_2_C; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_pair_C_2_C.json
[2026-10-12T13:07:02+00:00] STEP FK_pair_C_2_A; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/FK_pair_C_2_A.json
[2026-10-12T13:07:02+00:00] ARM FK: planned
[2026-10-12T13:07:02+00:00] STAGE final_validation; allocation=180s; hard remaining=2700.0s
[2026-10-12T13:07:02+00:00] STEP final_B_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_B_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP final_B_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_B_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/final_B_det0.layers
[2026-10-12T13:07:02+00:00] STEP final_B_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_B_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/final_B_det1.layers
[2026-10-12T13:07:02+00:00] STEP final_B_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_B --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_B_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/final_B_det2.layers
[2026-10-12T13:07:02+00:00] GATE final_B: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP final_F_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_F_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP final_F_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_F_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/final_F_det0.layers
[2026-10-12T13:07:02+00:00] STEP final_F_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_F_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/final_F_det1.layers
[2026-10-12T13:07:02+00:00] STEP final_F_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_F --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_F_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/final_F_det2.layers
[2026-10-12T13:07:02+00:00] GATE final_F: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP final_K_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_K_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP final_K_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_K_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/final_K_det0.layers
[2026-10-12T13:07:02+00:00] STEP final_K_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_K_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/final_K_det1.layers
[2026-10-12T13:07:02+00:00] STEP final_K_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_K --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_K_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/final_K_det2.layers
[2026-10-12T13:07:02+00:00] GATE final_K: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP final_FK_teacher; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_FK_teacher.json --teacher-force
[2026-10-12T13:07:02+00:00] STEP final_FK_det0; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_FK_det0.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/final_FK_det0.layers
[2026-10-12T13:07:02+00:00] STEP final_FK_det1; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_FK_det1.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/final_FK_det1.layers
[2026-10-12T13:07:02+00:00] STEP final_FK_det2; expected=10s; deadline<=190s; $ /tmp/fleet-fk-DRY-RUN/artifacts/decode_FK --graph /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --tokens 32 --context 1024 --tokens-per-launch 32 --repeat 1 --json /tmp/fleet-fk-DRY-RUN/artifacts/final_FK_det2.json --dump-layers /tmp/fleet-fk-DRY-RUN/artifacts/final_FK_det2.layers
[2026-10-12T13:07:02+00:00] GATE final_FK: teacher + 3 free processes; 32/32, 27/27, section 6, bitwise dumps/tokens
[2026-10-12T13:07:02+00:00] STEP timeline_prerequisite_trace; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python scripts/trace_timeline.py /tmp/fleet-fk-DRY-RUN/artifacts/prerequisite_trace.trace /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --layer 5
[2026-10-12T13:07:02+00:00] STEP timeline_F_trace_B; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python scripts/trace_timeline.py /tmp/fleet-fk-DRY-RUN/artifacts/F_trace_B.trace /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --layer 5
[2026-10-12T13:07:02+00:00] STEP timeline_F_trace_C; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python scripts/trace_timeline.py /tmp/fleet-fk-DRY-RUN/artifacts/F_trace_C.trace /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --layer 5
[2026-10-12T13:07:02+00:00] STEP timeline_F_trace_A; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python scripts/trace_timeline.py /tmp/fleet-fk-DRY-RUN/artifacts/F_trace_A.trace /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --layer 5
[2026-10-12T13:07:02+00:00] STEP timeline_K_trace_B; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python scripts/trace_timeline.py /tmp/fleet-fk-DRY-RUN/artifacts/K_trace_B.trace /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --layer 5
[2026-10-12T13:07:02+00:00] STEP timeline_K_trace_C; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python scripts/trace_timeline.py /tmp/fleet-fk-DRY-RUN/artifacts/K_trace_C.trace /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --layer 5
[2026-10-12T13:07:02+00:00] STEP timeline_K_trace_A; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python scripts/trace_timeline.py /tmp/fleet-fk-DRY-RUN/artifacts/K_trace_A.trace /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --layer 5
[2026-10-12T13:07:02+00:00] STEP timeline_FK_trace_B; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python scripts/trace_timeline.py /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_B.trace /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --layer 5
[2026-10-12T13:07:02+00:00] STEP timeline_FK_trace_C; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python scripts/trace_timeline.py /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_C.trace /tmp/fleet-fk-DRY-RUN/artifacts/graph_F.bin --layer 5
[2026-10-12T13:07:02+00:00] STEP timeline_FK_trace_A; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python scripts/trace_timeline.py /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_A.trace /tmp/fleet-fk-DRY-RUN/artifacts/graph_FK.bin --layer 5
[2026-10-12T13:07:02+00:00] STEP timeline_FK_trace_Off; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python scripts/trace_timeline.py /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_Off.trace /tmp/fleet-fk-DRY-RUN/artifacts/graph_B.bin --layer 5
[2026-10-12T13:07:02+00:00] STEP timeline_FK_trace_K_only; expected=5s; deadline<=30s; $ /Users/ckl/code/fleet-mi300x/.venv/bin/python scripts/trace_timeline.py /tmp/fleet-fk-DRY-RUN/artifacts/FK_trace_K_only.trace /tmp/fleet-fk-DRY-RUN/artifacts/graph_K.bin --layer 5
[2026-10-12T13:07:02+00:00] DRY RUN COMPLETE: all conditional arms shown; zero subprocess/GPU/SSH/delete calls; no files written
PLAN CLEANUP (also on failure): scp full directory into results/<session>/remote; verify every SHA256; execute operator delete argv; write controller report
PLAN FAIL-CLOSED: failed export/hash => NO DELETE, print recovery command; hard clock ends new work and writes report
exit=0
```

Commit and scope verification:

```text
$ git show --stat --oneline --summary 8213362
8213362 Prepare bounded F/K GPU session with paired gates and verified log export
 tests/run_fk_session.manifest.json |   63 +++
 tests/run_fk_session.py            | 1014 ++++++++++++++++++++++++++++++++++++
 2 files changed, 1077 insertions(+)
 create mode 100644 tests/run_fk_session.manifest.json
 create mode 100755 tests/run_fk_session.py

$ git diff --exit-code 7b410f3 -- src docs/STATUS.md tests/run_local_assembly_gpu.py
(no output)
exit=0

$ git diff --cached --check
(no output)
exit=0
```
