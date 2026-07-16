"""
tests/test_aggregate.py
-----------------------
Unit test for experiments/aggregate_seeds.py (R1). Pure — no training, no I/O.
Run with: python3 tests/test_aggregate.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments'))

from aggregate_seeds import aggregate_metrics, to_latex, to_plain_text  # noqa: E402

results = []


def check(name, cond, detail=''):
    results.append(bool(cond))
    print(f"  {'✓' if cond else '✗'}  {name}" + (f"  [{detail}]" if detail else ''))


def _agent(name, mean, std, c90, c95, mx):
    return {'agent_name': name, 'mean_IS_bps': mean, 'std_IS_bps': std,
            'CVaR_0.90_bps': c90, 'CVaR_0.95_bps': c95, 'max_IS_bps': mx,
            'GL_ratio': 0.0}


seedA = {
    'IQN-CVaR_0.95': _agent('IQN-CVaR_0.95', 2.0, 0.2, 2.4, 2.5, 3.0),
    'DQN':           _agent('DQN',           2.2, 0.3, 2.7, 2.8, 3.6),
}
seedB = {
    'IQN-CVaR_0.95': _agent('IQN-CVaR_0.95', 2.2, 0.3, 2.6, 2.7, 3.2),
    'DQN':           _agent('DQN',           2.6, 0.5, 2.9, 3.0, 4.0),
}
per_seed = [seedA, seedB]

agents, rows = aggregate_metrics(per_seed)

# mean/std across seeds: IQN mean_IS_bps values [2.0, 2.2] -> mean 2.1, std 0.1
m, s = rows['IQN-CVaR_0.95']['mean_IS_bps']
check("mean over seeds correct", abs(m - 2.1) < 1e-9, f"mean={m}")
check("std over seeds correct (population)", abs(s - 0.1) < 1e-9, f"std={s}")

m2, _ = rows['DQN']['CVaR_0.95_bps']   # [2.8, 3.0] -> 2.9
check("DQN CVaR95 mean correct", abs(m2 - 2.9) < 1e-9, f"mean={m2}")
check("agent order preserved", agents == ['IQN-CVaR_0.95', 'DQN'], str(agents))

tex = to_latex(agents, rows, len(per_seed))
check("latex has booktabs rules",
      all(r in tex for r in [r'\toprule', r'\midrule', r'\bottomrule']))
body_rows = [ln for ln in tex.splitlines() if ln.strip().endswith(r'\\')]
check("latex has header + one row per agent",
      len(body_rows) == len(agents) + 1, f"{len(body_rows)} rows")
check("latex escapes underscores", 'IQN-CVaR\\_0.95' in tex)

txt = to_plain_text(agents, rows, len(per_seed))
check("plain text renders agents+metrics",
      'Mean IS' in txt and 'IQN-CVaR_0.95' in txt)

failed = results.count(False)
print(f"\n  {sum(results)}/{len(results)} checks passed")
sys.exit(0 if failed == 0 else 1)
