"""
experiments/exp_utils.py
------------------------
Small shared helpers for the experiment runners:
  - dump_config_json : write a run's full config (dataclass or dict) as JSON
                       beside its outputs (satisfies the "all runs dump config"
                       constraint in PLAN.md).
  - refuse_if_nonempty : results-dir guard (PLAN.md R-2) — never write into an
                         existing non-empty results directory; archive first.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path


def _to_jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def dump_config_json(config, path) -> Path:
    """Write `config` (a dataclass instance or a plain dict) to `path` as JSON.

    Creates parent directories as needed. Returns the written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_to_jsonable(config), f, indent=2, default=str)
    return path


def refuse_if_nonempty(path) -> Path:
    """Raise FileExistsError if `path` is an existing non-empty directory.

    Guards against overwriting prior results (PLAN.md R-2). Archive the old
    directory (see RUNBOOK.md) before re-running into the same location.
    """
    path = Path(path)
    if path.exists() and path.is_dir() and any(path.iterdir()):
        raise FileExistsError(
            f'Refusing to write into non-empty results dir: {path}\n'
            f'  Archive it first (mv it into results/_archive_<date>/) — see '
            f'PLAN.md R-2 / RUNBOOK.md.')
    return path


def prepare_v2_output_dir(path, config=None, *, force_resume: bool = False,
                          subdirs=('checkpoints', 'logs'),
                          config_name: str = 'config.json') -> Path:
    """Set up a Design-v2 output dir under a single guarded entry point.

    Design-v2 runners write ONLY under ``results/_v2_*/`` and must REFUSE to
    write into a non-empty dir unless ``--force-resume`` is passed (PLAN_V2.md
    standing rules). This helper enforces that, creates the standard subdirs,
    and dumps the run config JSON (reusing ``dump_config_json``).

    Args:
        path         : the run output directory (created if absent).
        config       : dataclass/dict run config to persist (skipped if None).
        force_resume : if True, allow writing into an existing non-empty dir
                       (the resumable staged-run path); otherwise guard it.
        subdirs      : subdirectories to create under ``path``.
        config_name  : filename for the dumped config (under ``path``).

    Returns:
        The prepared Path.
    """
    path = Path(path)
    if not force_resume:
        refuse_if_nonempty(path)
    path.mkdir(parents=True, exist_ok=True)
    for sd in subdirs:
        (path / sd).mkdir(parents=True, exist_ok=True)
    if config is not None:
        dump_config_json(config, path / config_name)
    return path
