"""
experiments/selection_lib.py  (Design-v2 B1)
--------------------------------------------
Reusable, importable CORE of the CRN checkpoint-selection rule that
experiments/run_selection_appendix.py implements inline.

Given a learned agent, the list of its saved checkpoint files, an eval env and a
fixed validation seed, it evaluates EVERY checkpoint on the SAME n_val-episode
Common-Random-Numbers (CRN) validation set: before each checkpoint's rollout the
env AND the global torch/np RNG are reseeded to the identical val_seed, so every
checkpoint sees the identical sequence of price paths and identical IQN tau draws.
This is exactly what run_selection_appendix._rollout does; the logic is extracted
here so other B1 code can reuse it without importing the CLI script.

Selection rules (mirror run_selection_appendix.py exactly):
    PRIMARY   = checkpoint with min val-CVaR_alpha_bps
    SECONDARY = checkpoint with min val-mean_IS_bps
Plus a plateau report: the checkpoints within `plateau_tol` (default 10%) of the
best PRIMARY metric, and the +/-1 checkpoint neighbours of the PRIMARY pick.

Pure library: no argparse, no CLI, no side effects at import time beyond the
sys.path bootstrap every experiments/*.py performs so `import run_simulation`
(top-level module name) resolves regardless of how this module was imported.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Bootstrap sys.path exactly like the other experiments/*.py scripts so that
# `import run_simulation` (a top-level module, not experiments.run_simulation)
# resolves whether this file is imported as `experiments.selection_lib` or run
# from inside experiments/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# run_simulation owns the canonical _seed_global_rng helper (torch + numpy). We
# reuse it so CRN reseeding here is byte-identical to the training/eval scripts.
import run_simulation as RS
from evaluation.metrics import cvar_alpha


# ---------------------------------------------------------------------------
# Checkpoint-path helpers
# ---------------------------------------------------------------------------

def ep_of(path) -> int:
    """Parse the episode number out of a '..._ep<N>.pt' checkpoint filename."""
    return int(Path(path).name.split('_ep')[-1].split('.pt')[0])


def ckpt_paths_for(ckpt_dir, name) -> List[str]:
    """All '{name}_ep*.pt' checkpoint paths under `ckpt_dir`, sorted by episode."""
    paths = glob.glob(str(Path(ckpt_dir) / f'{name}_ep*.pt'))
    return sorted(paths, key=ep_of)


# ---------------------------------------------------------------------------
# CRN validation rollout  (extracted from run_selection_appendix._rollout)
# ---------------------------------------------------------------------------

def crn_rollout(agent, env, n, seed, record_first: bool = False):
    """Run `n` Common-Random-Numbers validation episodes for one loaded agent.

    Reseeding BOTH the env and the global torch/np RNG to the same `seed` BEFORE
    the loop is what makes successive calls compare checkpoints on identical
    price paths and identical IQN tau draws (CRN). Returns
    ``(is_bps_array, first_actions_array)`` with IS reported in basis points.

    This mirrors run_selection_appendix._rollout so selections computed via this
    lib match the appendix's numbers exactly.
    """
    env.seed(seed)
    RS._seed_global_rng(seed)          # CRN: identical episodes + tau draws every call
    is_bps: List[float] = []
    firsts: List[int] = []
    for _ in range(n):
        s = env.reset()
        a = agent.select_action(s, eval_mode=True)
        if record_first:
            firsts.append(int(a))
        done, info = False, {}
        while not done:
            s, _, done, info = env.step(a)
            if not done:
                a = agent.select_action(s, eval_mode=True)
        is_bps.append(float(info['implementation_shortfall']) * 1e4)
    return np.asarray(is_bps), np.asarray(firsts)


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------

def select_checkpoints(
    agent,
    checkpoint_paths,
    eval_env,
    val_seed: int,
    n_val: int = 1200,
    alpha: float = 0.95,
    plateau_tol: float = 0.10,
) -> Dict:
    """Evaluate every checkpoint on a fixed CRN validation set and pick the best.

    Every checkpoint is scored on the SAME `n_val`-episode validation set: the
    env and global RNG are reseeded to `val_seed` before each checkpoint's
    rollout (see :func:`crn_rollout`), so the comparison is apples-to-apples.

    Args:
        agent:            a learned agent exposing ``.load(path)`` and
                          ``.select_action(state, eval_mode=True)``. It is
                          mutated in place (each checkpoint is loaded into it).
        checkpoint_paths: iterable of '..._ep<N>.pt' checkpoint file paths.
        eval_env:         validation env (reseeded per checkpoint via CRN).
        val_seed:         fixed CRN validation seed (env + global RNG).
        n_val:            episodes in the CRN validation set (default 1200).
        alpha:            CVaR level for the PRIMARY rule (default 0.95).
        plateau_tol:      fractional tolerance for the plateau report (0.10=10%).

    Returns a dict:
        'per_ckpt'          : [ {ep, path, mean_IS_bps, <cvar_key>} ] sorted by ep
        'best_cvar'         : path of the min val-CVaR checkpoint (PRIMARY)
        'best_mean'         : path of the min val-mean-IS checkpoint (SECONDARY)
        'best_cvar_ep'      : episode of the PRIMARY pick
        'best_mean_ep'      : episode of the SECONDARY pick
        'plateau'           : per_ckpt entries within `plateau_tol` of best PRIMARY
        'neighbours'        : per_ckpt entries at +/-1 checkpoint around PRIMARY
        'plateau_threshold' : the CVaR value that bounds the plateau
        'alpha', 'cvar_key', 'n_val', 'val_seed', 'plateau_tol'
    """
    cvar_key = f'CVaR_{alpha:.2f}_bps'
    paths = sorted((str(p) for p in checkpoint_paths), key=ep_of)
    if not paths:
        raise ValueError('select_checkpoints: no checkpoint_paths given')

    per_ckpt: List[Dict] = []
    for p in paths:
        agent.load(p)
        arr, _ = crn_rollout(agent, eval_env, n_val, val_seed)
        per_ckpt.append({
            'ep': ep_of(p),
            'path': p,
            'mean_IS_bps': float(arr.mean()),
            cvar_key: float(cvar_alpha(arr, alpha)),
        })

    best_cvar = min(per_ckpt, key=lambda v: v[cvar_key])   # PRIMARY  (min val-CVaR)
    best_mean = min(per_ckpt, key=lambda v: v['mean_IS_bps'])  # SECONDARY (min val-mean)

    # Plateau: every checkpoint whose PRIMARY metric is within plateau_tol of the
    # best. IS/CVaR are positive costs, so "within 10%" == <= best * (1 + tol).
    threshold = best_cvar[cvar_key] * (1.0 + plateau_tol)
    plateau = [v for v in per_ckpt if v[cvar_key] <= threshold]

    # Neighbours: the +/-1 checkpoints (in episode order) around the PRIMARY pick.
    idx = next(i for i, v in enumerate(per_ckpt) if v['path'] == best_cvar['path'])
    neighbours = [per_ckpt[j] for j in (idx - 1, idx + 1) if 0 <= j < len(per_ckpt)]

    return {
        'per_ckpt': per_ckpt,
        'best_cvar': best_cvar['path'],
        'best_mean': best_mean['path'],
        'best_cvar_ep': best_cvar['ep'],
        'best_mean_ep': best_mean['ep'],
        'plateau': plateau,
        'neighbours': neighbours,
        'plateau_threshold': threshold,
        'alpha': alpha,
        'cvar_key': cvar_key,
        'n_val': n_val,
        'val_seed': val_seed,
        'plateau_tol': plateau_tol,
    }
