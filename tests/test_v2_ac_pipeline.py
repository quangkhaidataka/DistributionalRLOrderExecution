"""
tests/test_v2_ac_pipeline.py
----------------------------
Design-v2 B2.3 — pipeline logic test for the AC study CLI (experiments/run_v2_ac.py).

Standalone script (no pytest). Runs the real CLI as SUBPROCESSES (true staged-job
isolation) with a tiny --smoke config; NO full training. Run from the repo root:

    python3 tests/test_v2_ac_pipeline.py

Checks:
    A. Smoke through the CLI (≤200 eps/agent) → results/_v2_ac/_smoke/... produces
       comparison_table.txt (with cap-frac), all_results.json, is_arrays.pkl,
       alpha_ladder.{txt,tex}.
    B. tables_v2 generators run on the smoke outputs (format check).
    C. Resume-split equivalence: --episodes 0:200 + 200:400 == a single 0:400 run
       (byte-identical checkpoint at ep200 and ep400).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))

import evaluation.tables_v2 as tables_v2

CLI = str(PROJECT_ROOT / 'experiments' / 'run_v2_ac.py')
SMOKE_ROOT = PROJECT_ROOT / 'results' / '_v2_ac' / '_smoke'

_FAILS = []
def check(name, ok, detail=''):
    tag = 'PASS' if ok else 'FAIL'
    print(f'  [{tag}] {name}' + (f'  — {detail}' if detail and not ok else ''))
    if not ok:
        _FAILS.append(name)

def section(t):
    print(f"\n{'─'*70}\n  {t}\n{'─'*70}")

def run_cli(*args, tag=''):
    """Run the CLI as a subprocess from the repo root; return (rc, output)."""
    cmd = [sys.executable, CLI, *args]
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    ok = r.returncode == 0
    if not ok:
        print(f'    !! CLI failed ({tag}) rc={r.returncode}\n'
              f'    args: {" ".join(args)}\n'
              f'    stderr tail:\n' + '\n'.join(r.stdout.splitlines()[-8:]
                                               + r.stderr.splitlines()[-12:]))
    return r.returncode, (r.stdout + r.stderr)


def ckpt_equal(path_a, path_b, keys=('online_net', 'target_net')):
    """True iff two checkpoints have byte-identical weights (+ step)."""
    a = torch.load(str(path_a), map_location='cpu')
    b = torch.load(str(path_b), map_location='cpu')
    for k in keys:
        sa, sb = a[k], b[k]
        if sa.keys() != sb.keys():
            return False
        if not all(torch.equal(sa[t], sb[t]) for t in sa):
            return False
    return a.get('step') == b.get('step')


# ════════════════════════════════════════════════════════════════════════════
# A. Smoke through the CLI
# ════════════════════════════════════════════════════════════════════════════
section('A — smoke pipeline through the CLI (--smoke, 200 eps/agent)')

smoke_out = SMOKE_ROOT / 'pipeline_test'
if smoke_out.exists():
    shutil.rmtree(smoke_out)

ok_all = True
for ag in ('DQN', 'DDQN', 'IQN-neutral'):
    rc, _ = run_cli('--only-agent', ag, '--smoke', '--seed', '42',
                    '--out-dir', str(smoke_out), tag=f'train {ag}')
    ok_all = ok_all and rc == 0
check('train DQN/DDQN/IQN-neutral (smoke) via CLI', ok_all)

rc, _ = run_cli('--assemble-eval', '--smoke', '--seed', '42',
                '--out-dir', str(smoke_out), tag='assemble-eval')
check('assemble-eval (smoke) via CLI', rc == 0)

logs = smoke_out / 'logs'
comp = logs / 'comparison_table.txt'
allr = logs / 'all_results.json'
arrs = logs / 'is_arrays.pkl'
lad_txt = logs / 'alpha_ladder.txt'
lad_tex = logs / 'alpha_ladder.tex'
for f in (comp, allr, arrs, lad_txt, lad_tex):
    check(f'output exists: {f.name}', f.exists())

results = json.loads(allr.read_text()) if allr.exists() else []
names = [r.get('agent_name') for r in results]
check('all 11 agents evaluated (TWAP..IQN-CVaR_0.95)', len(results) == 11,
      f'{len(names)}: {names}')
check('rule-based + MaxSpeed present',
      {'TWAP', 'AC', 'MaxSpeed'} <= set(names), str(names))
check('cap_frac present in every result row',
      all('cap_frac' in r for r in results))
if results:
    ms = next((r for r in results if r['agent_name'] == 'MaxSpeed'), {})
    tw = next((r for r in results if r['agent_name'] == 'TWAP'), {})
    check('MaxSpeed cap_frac ≈ 1.0 (pins the cap)', ms.get('cap_frac', 0) > 0.99,
          f"cap_frac={ms.get('cap_frac')}")
    check('TWAP cap_frac ≈ 0.0 (never at cap)', tw.get('cap_frac', 1) < 0.01,
          f"cap_frac={tw.get('cap_frac')}")

comp_txt = comp.read_text() if comp.exists() else ''
check('comparison_table.txt has cap-frac column', 'cap-frac' in comp_txt)


# ════════════════════════════════════════════════════════════════════════════
# B. Table generators on smoke outputs (format check)
# ════════════════════════════════════════════════════════════════════════════
section('B — tables_v2 generators on smoke outputs (format check)')

if results:
    tbl = tables_v2.format_comparison_table(results, order=names,
                                            title='smoke re-render')
    check('format_comparison_table renders on smoke all_results',
          'cap-frac' in tbl and all(n in tbl for n in ('TWAP', 'MaxSpeed')))
    # seed-aggregate over the single smoke seed dir (format check only)
    agg_out = SMOKE_ROOT / '_aggregate'
    if agg_out.exists():
        shutil.rmtree(agg_out)
    try:
        # write_seed_aggregate takes per-seed DIRS (each with logs/all_results.json)
        txt_p, tex_p = tables_v2.write_seed_aggregate([smoke_out], 'ac_smoke', agg_out)
        check('write_seed_aggregate wrote .txt + .tex',
              txt_p.exists() and tex_p.exists()
              and txt_p.stat().st_size > 0 and tex_p.stat().st_size > 0)
    except Exception as e:
        check('write_seed_aggregate wrote .txt + .tex', False, repr(e))
else:
    check('format_comparison_table renders on smoke all_results', False, 'no results')


# ════════════════════════════════════════════════════════════════════════════
# C. Resume-split equivalence  (0:200 + 200:400 == 0:400)
# ════════════════════════════════════════════════════════════════════════════
section('C — resume-split equivalence (IQN-neutral, byte-identical checkpoint)')

single = SMOKE_ROOT / 'resume_single'
split  = SMOKE_ROOT / 'resume_split'
for d in (single, split):
    if d.exists():
        shutil.rmtree(d)

COMMON = ['--only-agent', 'IQN-neutral', '--smoke', '--seed', '42',
          '--checkpoint-freq', '200']

# single 0:400
rc_s, _ = run_cli(*COMMON, '--episodes', '0:400', '--total-episodes', '400',
                  '--out-dir', str(single), tag='single 0:400')
check('single run 0:400', rc_s == 0)

# split 0:200 then 200:400 (two separate processes, true staged resume)
rc_a, _ = run_cli(*COMMON, '--episodes', '0:200', '--total-episodes', '400',
                  '--out-dir', str(split), tag='split 0:200')
rc_b, _ = run_cli(*COMMON, '--episodes', '200:400', '--total-episodes', '400',
                  '--out-dir', str(split), tag='split 200:400')
check('split run 0:200 + 200:400', rc_a == 0 and rc_b == 0)

c_single_200 = single / 'checkpoints' / 'IQN-neutral_ep200.pt'
c_split_200  = split  / 'checkpoints' / 'IQN-neutral_ep200.pt'
c_single_400 = single / 'checkpoints' / 'IQN-neutral_ep400.pt'
c_split_400  = split  / 'checkpoints' / 'IQN-neutral_ep400.pt'

check('all four checkpoints exist',
      all(p.exists() for p in (c_single_200, c_split_200, c_single_400, c_split_400)))
if c_single_200.exists() and c_split_200.exists():
    check('ep200 checkpoint identical (single vs split job A)',
          ckpt_equal(c_single_200, c_split_200))
if c_single_400.exists() and c_split_400.exists():
    check('ep400 checkpoint identical (single vs resumed split)',
          ckpt_equal(c_single_400, c_split_400))
# resume state file must be cleaned up once the final segment completes
check('resume state removed after final segment',
      not (split / 'logs' / 'IQN-neutral_resume.pt').exists())


# ── summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
if _FAILS:
    print(f'  test_v2_ac_pipeline: {len(_FAILS)} FAILED → {_FAILS}')
    sys.exit(1)
print('  test_v2_ac_pipeline: ALL CHECKS PASSED')
sys.exit(0)
