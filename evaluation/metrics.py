"""
evaluation/metrics.py
---------------------
Financial risk metrics for evaluating execution agents.

All metrics operate on arrays of episode-level Implementation
Shortfall (IS) values collected across multiple evaluation episodes.

Metrics computed (for paper Table 3 and Figure comparisons):
    - Mean IS, Std IS          : basic performance summary
    - VaR_α                    : value-at-risk at level α
    - CVaR_α                   : conditional value-at-risk (expected shortfall)
    - Gain-Loss (GL) ratio     : upside vs downside asymmetry
    - Max IS                   : worst-case single episode
    - Sortino-style ratio      : mean / downside deviation
    - Mean-CVaR frontier       : Pareto points across α levels

Design choices:
    - Pure numpy (no torch dependency) — metrics are post-hoc,
      not part of the training graph
    - All functions are stateless: take arrays, return scalars/dicts
    - EpisodeTracker class accumulates per-step info dictionaries
      from env.step() and produces the IS array for metric computation
    - MetricsSuite wraps everything into one call for the paper table

Usage:
    tracker = EpisodeTracker()
    for episode in range(n_eval):
        state = env.reset()
        tracker.begin_episode()
        done = False
        while not done:
            action = agent.select_action(state, eval_mode=True)
            state, reward, done, info = env.step(action)
            tracker.step(reward, info)
        tracker.end_episode()
    results = tracker.compute_metrics()
    # results is a dict ready to be inserted into a paper table row

Tensor shape conventions:
    is_array : (n_episodes,)   array of IS values per episode
    rewards  : (n_steps,)      per-step rewards within one episode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ============================================================================
# Core metric functions (stateless, pure numpy)
# ============================================================================

def mean_is(is_array: np.ndarray) -> float:
    """Mean implementation shortfall across episodes."""
    return float(np.mean(is_array))


def std_is(is_array: np.ndarray) -> float:
    """Standard deviation of IS (measures execution risk)."""
    return float(np.std(is_array, ddof=1)) if len(is_array) > 1 else 0.0


def var_alpha(is_array: np.ndarray, alpha: float = 0.95) -> float:
    """
    Value-at-Risk at level α.

    VaR_α = the α-quantile of the IS distribution.
    Interpretation: with probability α, IS will not exceed this value.

    Convention: IS > 0 means cost. Higher VaR = worse tail.

    Args:
        is_array: (n_episodes,) IS values
        alpha:    confidence level in (0, 1)

    Returns:
        VaR value (scalar)
    """
    return float(np.percentile(is_array, 100 * alpha))


def cvar_alpha(is_array: np.ndarray, alpha: float = 0.95) -> float:
    """
    Conditional Value-at-Risk (Expected Shortfall) at level α.

    CVaR_α = E[IS | IS ≥ VaR_α]
           = mean of IS values in the worst (1-α) fraction of episodes.

    This is THE key metric for the paper: IQN-CVaR is designed to
    minimise exactly this quantity.

    For α = 0.95: CVaR is the average IS in the worst 5% of episodes.
    For α = 0.90: CVaR is the average IS in the worst 10% of episodes.

    Mathematical note:
        The IQN-CVaR agent optimises CVaR via:
            argmax_a E_{τ~U[0,α]}[z_θ(τ, s, a)]
        This is equivalent to minimising CVaR_α of the return distribution.
        The metric here computes the realised CVaR from evaluation data.

    Args:
        is_array: (n_episodes,) IS values
        alpha:    confidence level in (0, 1), e.g. 0.95

    Returns:
        CVaR value (scalar). Lower = better tail-risk management.
    """
    threshold = np.percentile(is_array, 100 * alpha)
    tail = is_array[is_array >= threshold]
    if len(tail) == 0:
        return float(np.max(is_array))
    return float(np.mean(tail))


def max_is(is_array: np.ndarray) -> float:
    """Worst-case IS across all episodes."""
    return float(np.max(is_array))


def gain_loss_ratio(is_array: np.ndarray, benchmark: float = 0.0) -> float:
    """
    Gain-Loss ratio: mean gain / mean loss relative to benchmark.

    GL = E[max(benchmark - IS, 0)] / E[max(IS - benchmark, 0)]

    Interpretation:
        GL > 1 : agent beats benchmark more often than it loses
        GL = 1 : symmetric around benchmark
        GL < 1 : losses outweigh gains

    For IS, benchmark = 0 means comparison against arrival price.
    A lower IS is better, so "gain" = episodes where IS < benchmark.

    Args:
        is_array:  (n_episodes,) IS values
        benchmark: reference IS (default 0.0 = arrival price)

    Returns:
        GL ratio (scalar). Returns inf if no losses, 0.0 if no gains.
    """
    gains  = np.maximum(benchmark - is_array, 0.0)   # IS < benchmark = gain
    losses = np.maximum(is_array - benchmark, 0.0)    # IS > benchmark = loss

    mean_gain = float(np.mean(gains))
    mean_loss = float(np.mean(losses))

    if mean_loss < 1e-12:
        return float('inf') if mean_gain > 1e-12 else 1.0
    return mean_gain / mean_loss


def sortino_ratio(is_array: np.ndarray, target: float = 0.0) -> float:
    """
    Sortino-style ratio adapted for IS minimisation.

    Sortino = -(mean_IS - target) / downside_deviation

    We negate because lower IS is better. Higher Sortino = better.
    Downside deviation only penalises IS above the target.

    Args:
        is_array: (n_episodes,) IS values
        target:   target IS (default 0.0)

    Returns:
        Sortino ratio (scalar). Higher = better.
    """
    excess = is_array - target
    downside = excess[excess > 0]

    if len(downside) < 2:
        return float('inf') if np.mean(is_array) <= target else 0.0

    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd < 1e-12:
        return float('inf')

    return float(-(np.mean(is_array) - target) / dd)


def mean_cvar_frontier(
    is_array: np.ndarray,
    alphas: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    """
    Mean-CVaR efficient frontier data points.

    For each α level, computes (mean_IS, CVaR_α). Plotting these
    across agents produces the mean-CVaR frontier figure in the paper.

    Args:
        is_array: (n_episodes,) IS values
        alphas:   list of α levels to evaluate

    Returns:
        dict with keys 'alphas', 'means', 'cvars'
    """
    if alphas is None:
        alphas = [0.80, 0.85, 0.90, 0.95, 0.99]

    m = mean_is(is_array)
    cvars = [cvar_alpha(is_array, a) for a in alphas]

    return {
        'alphas': alphas,
        'means' : [m] * len(alphas),
        'cvars' : cvars,
    }


def is_in_bps(is_array: np.ndarray) -> np.ndarray:
    """Convert IS from fractional to basis points (× 10,000)."""
    return is_array * 10_000


# ============================================================================
# Episode tracker — accumulates env.step() outputs
# ============================================================================

@dataclass
class StepRecord:
    """One step within an episode."""
    reward     : float
    x_t        : float   # shares executed
    p_exec     : float   # execution price
    q_remaining: float   # inventory after trade


class EpisodeTracker:
    """
    Collects per-step and per-episode data during evaluation.

    Designed to work with the BaseExecutionEnv.step() info dict.
    Accumulates everything needed for metrics + visualisation.

    Usage:
        tracker = EpisodeTracker()
        for ep in range(n_eval):
            state = env.reset()
            tracker.begin_episode()
            done = False
            while not done:
                action = agent.select_action(state, eval_mode=True)
                state, reward, done, info = env.step(action)
                tracker.step(reward, info)
            tracker.end_episode()

        results = tracker.compute_metrics()
    """

    def __init__(self):
        # Per-episode aggregates
        self.is_values       : List[float]             = []
        self.episode_rewards : List[float]             = []
        self.episode_lengths : List[int]               = []

        # Per-step detail (for the current episode)
        self._current_steps  : List[StepRecord]        = []
        self._current_reward : float                   = 0.0

        # Full per-step history (all episodes, for trajectory plots)
        self.all_trajectories: List[List[StepRecord]]  = []

    def begin_episode(self) -> None:
        """Call at start of each evaluation episode."""
        self._current_steps  = []
        self._current_reward = 0.0

    def step(self, reward: float, info: Dict) -> None:
        """
        Record one environment step.

        Args:
            reward: scalar reward from env.step()
            info:   info dict from env.step()
        """
        self._current_reward += reward
        self._current_steps.append(StepRecord(
            reward      = reward,
            x_t         = info.get('x_t', 0.0),
            p_exec      = info.get('p_exec', 0.0),
            q_remaining = info.get('q_remaining', 0.0),
        ))

    def end_episode(self, info: Optional[Dict] = None) -> None:
        """
        Finalise episode and store IS value.

        The IS is taken from the last step's info dict (which
        BaseExecutionEnv populates at done=True), or from the
        info dict passed here.
        """
        # Get IS from the info dict of the terminal step
        is_val = None
        if info is not None:
            is_val = info.get('implementation_shortfall')

        # Fallback: check last step's info if passed via step()
        if is_val is None and self._current_steps:
            # IS = -sum(rewards) approximately
            is_val = -self._current_reward

        self.is_values.append(float(is_val) if is_val is not None else 0.0)
        self.episode_rewards.append(self._current_reward)
        self.episode_lengths.append(len(self._current_steps))
        self.all_trajectories.append(self._current_steps.copy())

    @property
    def n_episodes(self) -> int:
        return len(self.is_values)

    @property
    def is_array(self) -> np.ndarray:
        """IS values as numpy array, shape (n_episodes,)."""
        return np.array(self.is_values, dtype=np.float64)

    def compute_metrics(
        self,
        alphas: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Compute all metrics at once. Returns a flat dict suitable
        for a paper table row or logging.

        Default α levels: 0.90, 0.95 (standard in risk management).

        Returns:
            dict with keys like:
                'mean_IS', 'std_IS', 'mean_IS_bps', ...
                'VaR_0.90', 'CVaR_0.90', 'VaR_0.95', 'CVaR_0.95',
                'max_IS', 'GL_ratio', 'sortino', 'n_episodes'
        """
        if alphas is None:
            alphas = [0.90, 0.95]

        arr = self.is_array
        bps = is_in_bps(arr)

        results = {
            'n_episodes'    : self.n_episodes,
            'mean_IS'       : mean_is(arr),
            'std_IS'        : std_is(arr),
            'mean_IS_bps'   : mean_is(bps),
            'std_IS_bps'    : std_is(bps),
            'max_IS'        : max_is(arr),
            'max_IS_bps'    : max_is(bps),
            'GL_ratio'      : gain_loss_ratio(arr),
            'sortino'       : sortino_ratio(arr),
            'mean_reward'   : float(np.mean(self.episode_rewards)),
        }

        for a in alphas:
            suffix = f'_{a:.2f}'
            results[f'VaR{suffix}']      = var_alpha(arr, a)
            results[f'CVaR{suffix}']     = cvar_alpha(arr, a)
            results[f'VaR{suffix}_bps']  = var_alpha(bps, a)
            results[f'CVaR{suffix}_bps'] = cvar_alpha(bps, a)

        return results

    def reset(self) -> None:
        """Clear all stored data for a fresh evaluation run."""
        self.is_values.clear()
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.all_trajectories.clear()


# ============================================================================
# MetricsSuite — run full evaluation for one agent
# ============================================================================

class MetricsSuite:
    """
    High-level evaluation driver.

    Runs n_eval episodes with a given agent and environment,
    then returns all metrics as a dict.

    Usage:
        suite = MetricsSuite(n_eval=500, alphas=[0.90, 0.95])
        results = suite.evaluate(agent, env)
        print(results)

    For paper reproducibility:
        suite = MetricsSuite(n_eval=1000, seed=42)
        table_row = suite.evaluate(agent, env)
    """

    def __init__(
        self,
        n_eval : int = 500,
        alphas : Optional[List[float]] = None,
        seed   : Optional[int] = None,
    ):
        self.n_eval = n_eval
        self.alphas = alphas or [0.90, 0.95]
        self.seed   = seed

    def evaluate(self, agent, env) -> Dict[str, float]:
        """
        Run n_eval episodes and compute all metrics.

        Args:
            agent: any agent implementing select_action(state, eval_mode)
                   and optionally reset() (for rule-based agents)
            env:   any env implementing reset() / step()

        Returns:
            dict of all computed metrics
        """
        if self.seed is not None:
            env.seed(self.seed)

        tracker = EpisodeTracker()

        for ep in range(self.n_eval):
            state = env.reset()

            # Reset rule-based agents (TWAP, AC) if they have reset()
            if hasattr(agent, 'reset') and callable(agent.reset):
                agent.reset()

            tracker.begin_episode()
            done = False
            info = {}

            while not done:
                action = agent.select_action(state, eval_mode=True)
                state, reward, done, info = env.step(action)
                tracker.step(reward, info)

            tracker.end_episode(info)

        results = tracker.compute_metrics(alphas=self.alphas)
        results['agent_name'] = getattr(agent, 'name', str(type(agent).__name__))

        return results

    def evaluate_multiple(
        self,
        agents: List,
        env,
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple agents on the same environment.
        Returns list of result dicts (one per agent).

        Each agent is evaluated with the SAME seed for fair comparison.
        """
        all_results = []
        for agent in agents:
            results = self.evaluate(agent, env)
            all_results.append(results)
        return all_results


# ============================================================================
# Utility: format results for printing / paper table
# ============================================================================

def format_table_row(results: Dict[str, float], bps: bool = True) -> str:
    """
    Format a metrics dict as a single-line summary for logging.

    Args:
        results: dict from MetricsSuite.evaluate()
        bps:     if True, report IS in basis points

    Returns:
        Formatted string like:
        "IQN-CVaR_0.95 | IS=3.21±1.45 bps | CVaR₉₅=7.82 bps | GL=1.34"
    """
    name = results.get('agent_name', '???')
    if bps:
        m  = results.get('mean_IS_bps', 0.0)
        s  = results.get('std_IS_bps', 0.0)
        c  = results.get('CVaR_0.95_bps', 0.0)
        mx = results.get('max_IS_bps', 0.0)
    else:
        m  = results.get('mean_IS', 0.0)
        s  = results.get('std_IS', 0.0)
        c  = results.get('CVaR_0.95', 0.0)
        mx = results.get('max_IS', 0.0)

    gl = results.get('GL_ratio', 0.0)
    unit = ' bps' if bps else ''

    return (f"{name:<20s} | IS={m:>7.2f}±{s:<6.2f}{unit} | "
            f"CVaR₉₅={c:>7.2f}{unit} | "
            f"Max={mx:>7.2f}{unit} | GL={gl:.2f}")


def format_comparison_table(
    all_results: List[Dict[str, float]],
    bps: bool = True,
) -> str:
    """
    Format multiple agents' results as an aligned comparison table.

    Produces output ready for paper Table 3:

    Agent                | Mean IS   | Std IS   | CVaR₉₀  | CVaR₉₅  | Max IS  | GL
    ---------------------|-----------|----------|---------|---------|---------|-----
    TWAP                 |   5.23    |   3.41   |  11.82  |  14.21  |  22.34  | 0.89
    ...
    """
    unit = ' bps' if bps else ''
    sfx  = '_bps' if bps else ''

    # Header
    header = (f"{'Agent':<20s} | {'Mean IS':>9s} | {'Std IS':>8s} | "
              f"{'CVaR₉₀':>8s} | {'CVaR₉₅':>8s} | "
              f"{'Max IS':>8s} | {'GL':>5s}")
    sep    = '-' * len(header)

    rows = [header, sep]
    for r in all_results:
        name  = r.get('agent_name', '???')
        m     = r.get(f'mean_IS{sfx}', 0.0)
        s     = r.get(f'std_IS{sfx}', 0.0)
        c90   = r.get(f'CVaR_0.90{sfx}', 0.0)
        c95   = r.get(f'CVaR_0.95{sfx}', 0.0)
        mx    = r.get(f'max_IS{sfx}', 0.0)
        gl    = r.get('GL_ratio', 0.0)

        row = (f"{name:<20s} | {m:>9.4f} | {s:>8.4f} | "
               f"{c90:>8.4f} | {c95:>8.4f} | "
               f"{mx:>8.4f} | {gl:>5.4f}")
        rows.append(row)

    return '\n'.join(rows)