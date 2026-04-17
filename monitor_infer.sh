#!/usr/bin/env bash
# monitor_infer.sh — Live monitor for Oracle Guided inference runs.
# Usage: ./monitor_infer.sh <infer_logs_dir> [refresh_seconds] [active_threshold] [num_log_lines]
#
# num_log_lines:
#   0  = compact table mode (instance, step, stage, decision, age)
#   1+ = detailed mode showing last N timestamped log lines per instance

LOGDIR="${1:?Usage: $0 <infer_logs_dir> [refresh_sec] [threshold] [num_lines (0=table)]}"
REFRESH="${2:-5}"
THRESH="${3:-600}"
NLINES="${4:-1}"

[ ! -d "$LOGDIR" ] && echo "Error: $LOGDIR not found" >&2 && exit 1

while true; do
  clear
  python3 - "$LOGDIR" "$THRESH" "$NLINES" << 'PYEOF'
import sys, os, re, time, subprocess
from pathlib import Path
from datetime import datetime

logdir = sys.argv[1]
thresh = int(sys.argv[2])
nlines = int(sys.argv[3])
now = time.time()
now_fmt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

TS_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
STEP_RE = re.compile(r'Step (\d+)')
STAGE_RE = re.compile(r'stage=(\w+)')
DEC_RE = re.compile(r'decision=(\w+)')

total = active = idle = done_c = err_c = prep_c = 0
stages = {'exploration':0,'reproduction':0,'analysis_planning':0,'implementation_verification':0,'finish':0}
rows = []
err_rows = []

for f in sorted(Path(logdir).glob('instance_*.log')):
    total += 1
    inst = f.stem.replace('instance_','')

    # Read last 200 lines — enough for timestamp + step/stage/decision
    try:
        with open(f, 'rb') as fh:
            fh.seek(0, 2)
            sz = fh.tell()
            read_bytes = min(sz, 80000)  # ~200 lines × 400 chars
            fh.seek(max(0, sz - read_bytes))
            tail = fh.read().decode('utf-8', errors='replace')
    except Exception:
        idle += 1
        continue

    tail_lines = tail.splitlines()

    # Find last timestamped line (scan backwards)
    ll = ts_str = None
    for line in reversed(tail_lines):
        m = TS_RE.match(line)
        if m:
            ll = line
            ts_str = m.group(1)
            break

    if not ts_str:
        idle += 1
        continue

    try:
        ep = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S').timestamp()
    except Exception:
        idle += 1
        continue
    age = int(now - ep)

    # Terminal states — check last 10 timestamped lines, not just the very last one
    # (FINISHED/STOPPED may be followed by cleanup lines like "Cleaning docker")
    is_done = is_err = False
    checked = 0
    for line in reversed(tail_lines):
        if not TS_RE.match(line):
            continue
        if 'FINISHED' in line or 'STOPPED' in line:
            is_done = True
            break
        if 'STATUS$ERROR' in line or ('AgentState' in line and 'ERROR' in line):
            is_err = True
            break
        checked += 1
        if checked >= 10:
            break
    if is_done:
        done_c += 1
        continue
    if is_err:
        err_c += 1
        # Extract error type from the error line
        err_type = 'unknown'
        for line in reversed(tail_lines):
            if not TS_RE.match(line):
                continue
            if 'STATUS$ERROR' in line:
                m = re.search(r'STATUS\$ERROR_(\w+)', line)
                err_type = m.group(1) if m else 'STATUS_ERROR'
                break
            if 'AgentState' in line and 'ERROR' in line:
                err_type = 'AGENT_ERROR'
                break
            if 'Retryable controller error' in line:
                m = re.search(r'STATUS\$ERROR_(\w+)', line)
                err_type = m.group(1) if m else 'CONTROLLER_RETRY'
                break
        err_rows.append((inst, err_type))
        continue

    if age < thresh:
        active += 1
        # Extract step/stage/decision from tail (scan backwards for most recent)
        step = stage = dec = ''
        prep_status = ''
        for line in reversed(tail_lines):
            if not step:
                sm = STEP_RE.search(line)
                if sm: step = f'Step {sm.group(1)}'
            if not stage:
                stm = STAGE_RE.search(line)
                if stm: stage = stm.group(1)
            if not dec:
                dm = DEC_RE.search(line)
                if dm: dec = dm.group(1)
            # Detect env-preparation signals (for instances not yet at Step 1)
            if not prep_status and not step:
                if 'Building image' in line or 'not found locally' in line:
                    prep_status = 'pulling_image'
                elif 'Successfully pulled' in line:
                    prep_status = 'image_ready'
                elif 'Waiting for env-prepare slot' in line:
                    prep_status = 'waiting_slot'
                elif 'Starting runtime' in line or 'Container started' in line:
                    prep_status = 'starting_runtime'
                elif 'Runtime is ready' in line:
                    prep_status = 'runtime_ready'
                elif 'BEGIN Runtime Initialization' in line or 'END Runtime Initialization' in line:
                    prep_status = 'initializing'
                elif 'Loaded v6 facts' in line:
                    prep_status = 'loading_facts'
            if step and stage and dec:
                break

        # If no step found, use prep_status as the stage
        if not step and prep_status:
            stage = f'env:{prep_status}'
            prep_c += 1
        elif stage in stages:
            stages[stage] += 1

        if age < 60:
            af = f'{age}s'
        else:
            af = f'{age//60}m{age%60}s'
        # Collect last N timestamped lines for display
        last3 = []
        for line in reversed(tail_lines):
            if TS_RE.match(line):
                # Trim to readable length: strip timestamp prefix, truncate
                short = line[22:].strip()  # skip "YYYY-MM-DD HH:MM:SS,mmm - "
                if len(short) > 120:
                    short = short[:117] + '...'
                last3.append(short)
                if len(last3) >= nlines:
                    break
        last3.reverse()
        rows.append((inst, step, stage, dec, af, age, last3))
    else:
        idle += 1

# Docker container count
try:
    r = subprocess.run(['docker','ps','--filter','name=openhands-runtime-','-q'],
                       capture_output=True, text=True, timeout=5)
    ctr = len(r.stdout.strip().splitlines()) if r.stdout.strip() else 0
except Exception:
    ctr = '?'

# Print
print('╔══════════════════════════════════════════════════════════════════════╗')
print(f'║  ORACLE GUIDED MONITOR                        {now_fmt}  ║')
print('╠══════════════════════════════════════════════════════════════════════╣')
print(f'║  Total:{total:<5d}  Active:{active:<5d}  Done:{done_c:<5d}  Error:{err_c:<5d}  Idle:{idle:<5d}     ║')
print(f'║  Containers: {str(ctr):<57s}║')
print('╠══════════════════════════════════════════════════════════════════════╣')
s = stages
print(f'║  explore={s["exploration"]:<3d} repro={s["reproduction"]:<3d} analysis={s["analysis_planning"]:<3d} impl={s["implementation_verification"]:<3d} finish={s["finish"]:<3d} prep={prep_c:<3d}       ║')
print('╚══════════════════════════════════════════════════════════════════════╝')

if rows:
    print()
    if nlines == 0:
        # Compact table mode
        print(f'  {"INSTANCE":<42s} {"STEP":<8s} {"STAGE":<25s} {"DECISION":<10s} {"LAST"}')
        print(f'  {"--------":<42s} {"----":<8s} {"-----":<25s} {"--------":<10s} {"----"}')
        for inst, step, stage, dec, af, _, last3 in sorted(rows, key=lambda r: r[0]):
            print(f'  {inst:<42s} {step:<8s} {stage:<25s} {dec:<10s} {af}')
    else:
        # Detailed log-lines mode — instance + first log on same line, extras indented below
        for inst, step, stage, dec, af, _, last3 in sorted(rows, key=lambda r: r[0]):
            tag = f'{inst:<42s} ({af:>6s})'
            if last3:
                print(f'  {tag}  {last3[0]}')
                for line in last3[1:]:
                    print(f'  {"":42s} {"":8s}  {line}')
            else:
                print(f'  {tag}')
else:
    print()
    print('  (no active instances)')

if err_rows:
    print()
    print(f'  ERRORED INSTANCES ({len(err_rows)})')
    print(f'  {"INSTANCE":<42s} {"ERROR TYPE"}')
    print(f'  {"--------":<42s} {"----------"}')
    for inst, etype in sorted(err_rows, key=lambda r: r[0]):
        print(f'  {inst:<42s} {etype}')

# if rows:
#     print()
#     print(f'  LOG PATHS ({len(rows)} active):')
#     for inst, *_ in sorted(rows, key=lambda r: r[0]):
#         print(f'  {logdir}/instance_{inst}.log')
print()
print(f'  Refresh: $REFRESH s | Ctrl+C to stop')
PYEOF
  sleep "$REFRESH"
done
