"""
evaluation/visualizer.py
------------------------
All plotting functions for the paper.

Figure catalogue (maps to paper sections):
    1. IS distribution comparison   — histogram/KDE of IS across agents
    2. Mean-CVaR frontier           — Pareto frontier (mean vs tail risk)
    3. Quantile distribution        — IQN's learned Z(s,a) for a given state
    4. Training curves              — loss and reward over episodes
    5. Inventory trajectory         — how each agent liquidates over time
    6. Action distribution          — heatmap of agent actions vs state
    7. Regime analysis              — IS conditioned on volatility regime
    8. Comparison heatmap           — metric matrix across agents × metrics

Design choices:
    - matplotlib only (no seaborn/plotly) for maximum compatibility
      and deterministic PDF output for LaTeX inclusion
    - Every function returns the Figure object — caller decides
      whether to show(), savefig(), or embed in a report
    - Consistent style: 'seaborn-v0_8-whitegrid' with paper-quality
      font sizes and DPI
    - Color palette: qualitative Set2 for agents, sequential Blues
      for single-agent quantile plots

Usage:
    from evaluation.visualizer import Visualizer

    viz = Visualizer(save_dir='results/figures')
    viz.plot_is_distributions(all_results)
    viz.plot_training_curves(training_log)
    viz.plot_quantile_distribution(agent, state, env)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from evaluation.metrics import (
    EpisodeTracker, cvar_alpha, var_alpha, mean_is, is_in_bps,
)


# ============================================================================
# Style constants
# ============================================================================

# Agent color palette — consistent across ALL figures
AGENT_COLORS = {
    'TWAP'           : '#66c2a5',   # teal
    'AC'             : '#fc8d62',   # orange (matches any AC(λ=...) prefix)
    'DQN'            : '#8da0cb',   # blue-grey
    'DDQN'           : '#e78ac3',   # pink
    'QR-DQN'         : '#a6d854',   # lime (matches QR-DQN(N=...) prefix)
    'IQN-neutral'    : '#ffd92f',   # gold
    'IQN-CVaR_0.90'  : '#e5c494',   # tan
    'IQN-CVaR_0.95'  : '#b3b3b3',   # grey
}

# Fallback color cycle for unknown agents
_FALLBACK_COLORS = plt.cm.Set2(np.linspace(0, 1, 8))

FONT_SIZE_TITLE  = 14
FONT_SIZE_LABEL  = 12
FONT_SIZE_TICK   = 10
FONT_SIZE_LEGEND = 10
DPI              = 150
FIGSIZE_SINGLE   = (8, 5)
FIGSIZE_WIDE     = (12, 5)
FIGSIZE_TALL     = (8, 8)


def _get_color(agent_name: str, idx: int = 0) -> str:
    """Look up agent color, with prefix matching and fallback."""
    if agent_name in AGENT_COLORS:
        return AGENT_COLORS[agent_name]
    # Prefix match: 'AC(λ=1e-06)' → 'AC', 'QR-DQN(N=51)' → 'QR-DQN'
    for prefix, color in AGENT_COLORS.items():
        if agent_name.startswith(prefix):
            return color
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


def _apply_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent styling to an axes object."""
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3, linestyle='--')


# ============================================================================
# Visualizer class
# ============================================================================

class Visualizer:
    """
    Central plotting class for all paper figures.

    All plot methods return matplotlib Figure objects.
    If save_dir is set, figures are also saved as PDF + PNG.

    Args:
        save_dir: directory for saving figures (created if needed)
        bps:      if True, report IS in basis points throughout
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        bps: bool = True,
    ):
        self.bps = bps
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig: Figure, name: str) -> None:
        """Save figure as PDF (for LaTeX) and PNG (for quick viewing)."""
        if self.save_dir is None:
            return
        for ext in ['pdf', 'png']:
            path = self.save_dir / f'{name}.{ext}'
            fig.savefig(path, dpi=DPI, bbox_inches='tight')

    def _is_label(self) -> str:
        return 'Implementation Shortfall (bps)' if self.bps else 'Implementation Shortfall'

    def _convert(self, arr: np.ndarray) -> np.ndarray:
        return is_in_bps(arr) if self.bps else arr

    # ------------------------------------------------------------------
    # 1. IS distribution comparison (histogram + KDE overlay)
    # ------------------------------------------------------------------

    def plot_is_distributions(
        self,
        all_results: List[Dict],
        trackers: Optional[Dict[str, EpisodeTracker]] = None,
        bins: int = 50,
    ) -> Figure:
        """
        Overlaid histograms of IS distribution for each agent.

        This is the main visual argument for distributional RL:
        IQN-CVaR should have a thinner right tail than DDQN.

        Args:
            all_results: list of dicts from MetricsSuite.evaluate()
            trackers:    dict mapping agent_name → EpisodeTracker
                         (provides raw IS arrays for histograms)
            bins:        number of histogram bins

        Returns:
            Figure with overlaid semi-transparent histograms
        """
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        if trackers is not None:
            for idx, (name, tracker) in enumerate(trackers.items()):
                arr = self._convert(tracker.is_array)
                color = _get_color(name, idx)
                ax.hist(
                    arr, bins=bins, alpha=0.4, label=name,
                    color=color, edgecolor='white', linewidth=0.5,
                    density=True,
                )
                # Vertical line at CVaR_0.95
                c95 = cvar_alpha(arr, 0.95)
                ax.axvline(c95, color=color, linestyle='--', linewidth=1.5,
                           alpha=0.8)

        _apply_style(ax, 'IS Distribution by Agent',
                     self._is_label(), 'Density')
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')
        fig.tight_layout()
        self._save(fig, 'is_distributions')
        return fig

    # ------------------------------------------------------------------
    # 2. Mean-CVaR efficient frontier
    # ------------------------------------------------------------------

    def plot_mean_cvar_frontier(
        self,
        all_results: List[Dict],
        alpha: float = 0.95,
    ) -> Figure:
        """
        Scatter plot: Mean IS (x) vs CVaR_α IS (y) for each agent.

        The Pareto frontier shows the mean–tail tradeoff.
        IQN-CVaR should dominate the lower-left corner
        (low mean AND low CVaR).

        Args:
            all_results: list of dicts from MetricsSuite.evaluate()
            alpha:       CVaR level to plot

        Returns:
            Figure with labelled scatter points
        """
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        sfx = '_bps' if self.bps else ''
        cvar_key = f'CVaR_{alpha:.2f}{sfx}'
        mean_key = f'mean_IS{sfx}'

        for idx, r in enumerate(all_results):
            name  = r.get('agent_name', f'Agent_{idx}')
            m     = r.get(mean_key, 0.0)
            c     = r.get(cvar_key, 0.0)
            color = _get_color(name, idx)

            ax.scatter(m, c, s=120, color=color, edgecolors='black',
                       linewidths=1.0, zorder=3)
            ax.annotate(
                name, (m, c), fontsize=FONT_SIZE_TICK,
                textcoords='offset points', xytext=(8, 6),
                ha='left',
            )

        # Draw arrow indicating "better" direction
        ax.annotate(
            'Better', xy=(0.15, 0.15), xycoords='axes fraction',
            fontsize=FONT_SIZE_TICK, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            xytext=(0.35, 0.35),
        )

        _apply_style(ax,
                     f'Mean–CVaR{int(alpha*100)} Efficient Frontier',
                     f'Mean {self._is_label()}',
                     f'CVaR{int(alpha*100)} {self._is_label()}')
        fig.tight_layout()
        self._save(fig, 'mean_cvar_frontier')
        return fig

    # ------------------------------------------------------------------
    # 3. Learned quantile distribution Z(s,a)
    # ------------------------------------------------------------------

    def plot_quantile_distribution(
        self,
        agent,
        state: np.ndarray,
        n_tau: int = 200,
        action_names: Optional[List[str]] = None,
    ) -> Figure:
        """
        Visualise the IQN's learned return distribution Z(s,a)
        for a given state, across all actions.

        This figure demonstrates the core value proposition of
        distributional RL: the agent knows the FULL shape of
        the return distribution, not just its mean.

        How it works:
            1. Sample many τ values uniformly in [0, 1]
            2. Pass (state, τ) through the IQN network
            3. Plot the inverse CDF (quantile function) for each action

        Args:
            agent:        IQNAgent with online_net
            state:        (state_dim,) numpy array
            n_tau:        number of τ points (higher = smoother curve)
            action_names: labels for each action

        Returns:
            Figure with quantile function curves per action
        """
        import torch

        if action_names is None:
            # action_names = ['Wait (0%)', 'Sell 10%', 'Sell 20%',
            #                 'Sell 30%','Sell 40%', 'Sell 50%', 'Sell 60%', 'Sell 70%',
            #                 'Sell 80%','Sell 90%', 'Sell 100%']
            action_names = ['Wait (0%)', 'Sell 20%', 'Sell 40%', 'Sell 60%', 'Sell 80%','Sell 100%']

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        state_t = torch.FloatTensor(state).unsqueeze(0)
        if hasattr(agent, 'device'):
            state_t = state_t.to(agent.device)

        # Dense τ grid for smooth quantile function plot
        tau_vals = np.linspace(0.01, 0.99, n_tau)

        # Get quantile values for each τ
        net = agent.online_net if hasattr(agent, 'online_net') else None
        if net is None:
            ax.text(0.5, 0.5, 'Agent has no IQN network',
                    transform=ax.transAxes, ha='center', fontsize=14)
            return fig

        net.eval()
        with torch.no_grad():
            quantiles, _ = net(state_t, n_tau=n_tau,
                               tau_low=0.01, tau_high=0.99)
            # quantiles: (1, n_tau, A)
            quantiles = quantiles.squeeze(0).cpu().numpy()  # (n_tau, A)

        # Sort quantiles along τ dimension for each action
        # (IQN samples τ randomly, so output isn't ordered)
        n_actions = quantiles.shape[1]
        for a in range(n_actions):
            q_sorted = np.sort(quantiles[:, a])
            label = action_names[a] if a < len(action_names) else f'Action {a}'
            ax.plot(tau_vals, q_sorted, linewidth=2.0, label=label, alpha=0.8)

        # Shade CVaR region (left tail: τ ∈ [0, α])
        alpha = getattr(agent.cfg, 'cvar_alpha', 0.95)
        if alpha < 1.0:
            ax.axvspan(0.0, alpha, alpha=0.08, color='red',
                       label=f'CVaR region (α={alpha})')

        _apply_style(ax, 'Learned Return Distribution Z(s, a)',
                     'Quantile level τ', 'Quantile value z(τ, s, a)')
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper left')
        fig.tight_layout()
        self._save(fig, 'quantile_distribution')
        return fig

    # ------------------------------------------------------------------
    # 4. Training curves
    # ------------------------------------------------------------------

    def plot_training_curves(
        self,
        training_log: Dict[str, List[float]],
        window: int = 50,
    ) -> Figure:
        """
        Training loss and episode reward over time.

        Args:
            training_log: dict with keys 'losses' and/or 'episode_rewards'
                          and optionally 'episode_is'
            window:       smoothing window for rolling average

        Returns:
            Figure with 1-3 subplots (loss, reward, IS)
        """
        n_plots = sum(1 for k in ['losses', 'episode_rewards', 'episode_is']
                      if k in training_log and len(training_log[k]) > 0)
        n_plots = max(n_plots, 1)

        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots),
                                 squeeze=False)
        axes = axes.flatten()
        plot_idx = 0

        if 'losses' in training_log and len(training_log['losses']) > 0:
            ax = axes[plot_idx]
            losses = np.array(training_log['losses'])
            ax.plot(losses, alpha=0.2, color='#1f77b4', linewidth=0.5)
            if len(losses) >= window:
                smoothed = np.convolve(losses, np.ones(window) / window,
                                       mode='valid')
                ax.plot(np.arange(window - 1, len(losses)), smoothed,
                        color='#1f77b4', linewidth=2.0, label=f'MA({window})')
                ax.legend(fontsize=FONT_SIZE_LEGEND)
            _apply_style(ax, 'Training Loss', 'Update step', 'Quantile Huber Loss')
            plot_idx += 1

        if 'episode_rewards' in training_log and len(training_log['episode_rewards']) > 0:
            ax = axes[plot_idx]
            rewards = np.array(training_log['episode_rewards'])
            ax.plot(rewards, alpha=0.2, color='#2ca02c', linewidth=0.5)
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window) / window,
                                       mode='valid')
                ax.plot(np.arange(window - 1, len(rewards)), smoothed,
                        color='#2ca02c', linewidth=2.0, label=f'MA({window})')
                ax.legend(fontsize=FONT_SIZE_LEGEND)
            _apply_style(ax, 'Episode Reward', 'Episode', 'Cumulative Reward')
            plot_idx += 1

        if 'episode_is' in training_log and len(training_log['episode_is']) > 0:
            ax = axes[plot_idx]
            is_vals = self._convert(np.array(training_log['episode_is']))
            ax.plot(is_vals, alpha=0.2, color='#d62728', linewidth=0.5)
            if len(is_vals) >= window:
                smoothed = np.convolve(is_vals, np.ones(window) / window,
                                       mode='valid')
                ax.plot(np.arange(window - 1, len(is_vals)), smoothed,
                        color='#d62728', linewidth=2.0, label=f'MA({window})')
                ax.legend(fontsize=FONT_SIZE_LEGEND)
            _apply_style(ax, 'Episode IS', 'Episode', self._is_label())
            plot_idx += 1

        fig.tight_layout()
        self._save(fig, 'training_curves')
        return fig

    # ------------------------------------------------------------------
    # 5. Inventory trajectory comparison
    # ------------------------------------------------------------------

    def plot_inventory_trajectories(
        self,
        trackers: Dict[str, EpisodeTracker],
        q0: float,
        n_periods: int,
        n_sample: int = 5,
    ) -> Figure:
        """
        Plot inventory paths q_t over time for multiple agents.

        Shows how different agents pace their liquidation.
        TWAP → straight line; AC → front-loaded; RL → adaptive.

        Args:
            trackers:  dict mapping agent_name → EpisodeTracker
            q0:        initial inventory (for denormalisation)
            n_periods: number of decision periods N
            n_sample:  number of sample trajectories per agent

        Returns:
            Figure with inventory paths overlaid
        """
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        time_grid = np.arange(n_periods + 1)

        for idx, (name, tracker) in enumerate(trackers.items()):
            color = _get_color(name, idx)
            trajectories = tracker.all_trajectories

            # Plot sample trajectories (thin lines)
            n_plot = min(n_sample, len(trajectories))
            for i in range(n_plot):
                traj = trajectories[i]
                q_path = [q0]
                for step in traj:
                    q_path.append(step.q_remaining)
                # Pad to full length if episode ended early
                while len(q_path) <= n_periods:
                    q_path.append(q_path[-1])
                q_arr = np.array(q_path[:n_periods + 1])
                label = name if i == 0 else None
                ax.plot(time_grid, q_arr, color=color, alpha=0.4,
                        linewidth=1.0, label=label)

            # Plot mean trajectory (thick line)
            if len(trajectories) >= 3:
                all_q = []
                for traj in trajectories:
                    q_path = [q0]
                    for step in traj:
                        q_path.append(step.q_remaining)
                    while len(q_path) <= n_periods:
                        q_path.append(q_path[-1])
                    all_q.append(q_path[:n_periods + 1])
                mean_q = np.mean(all_q, axis=0)
                ax.plot(time_grid, mean_q, color=color, linewidth=3.0,
                        linestyle='--', alpha=0.9)

        _apply_style(ax, 'Inventory Trajectory by Agent',
                     'Decision Period t', 'Remaining Inventory (shares)')
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')
        ax.set_xlim(0, n_periods)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        self._save(fig, 'inventory_trajectories')
        return fig

    # ------------------------------------------------------------------
    # 6. Action distribution heatmap
    # ------------------------------------------------------------------

    def plot_action_heatmap(
        self,
        tracker: EpisodeTracker,
        agent_name: str,
        q0: float,
        n_periods: int,
    ) -> Figure:
        """
        Heatmap of action frequency vs (time_period, inventory_level).

        Shows the agent's learned policy structure.
        IQN-CVaR should show more aggressive selling at high
        inventory + high time pressure compared to IQN-neutral.

        Args:
            tracker:    EpisodeTracker with trajectory data
            agent_name: name for the title
            q0:         initial inventory
            n_periods:  N decision periods

        Returns:
            Figure with heatmap
        """
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

        # Bin inventory into 5 levels, time into N periods
        n_inv_bins = 5
        inv_edges  = np.linspace(0, 1, n_inv_bins + 1)
        counts     = np.zeros((n_inv_bins, n_periods), dtype=np.float64)

        for traj in tracker.all_trajectories:
            for step_idx, step in enumerate(traj):
                if step_idx >= n_periods:
                    break
                q_frac = step.q_remaining / (q0 + 1e-8)
                q_frac = np.clip(q_frac, 0.0, 1.0)
                inv_bin = min(int(q_frac * n_inv_bins), n_inv_bins - 1)
                # Use x_t as a proxy for action intensity
                sell_frac = step.x_t / (step.q_remaining + step.x_t + 1e-8)
                counts[inv_bin, step_idx] += sell_frac

        # Normalise rows (per inventory level)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        counts = counts / row_sums

        im = ax.imshow(counts, aspect='auto', cmap='YlOrRd',
                       origin='lower', interpolation='nearest')
        ax.set_xticks(np.arange(0, n_periods, max(1, n_periods // 10)))
        inv_labels = [f'{inv_edges[i]:.0%}-{inv_edges[i+1]:.0%}'
                      for i in range(n_inv_bins)]
        ax.set_yticks(range(n_inv_bins))
        ax.set_yticklabels(inv_labels, fontsize=FONT_SIZE_TICK)
        fig.colorbar(im, ax=ax, label='Avg Sell Fraction')

        _apply_style(ax, f'Policy Heatmap: {agent_name}',
                     'Decision Period t', 'Remaining Inventory Fraction')
        fig.tight_layout()
        self._save(fig, f'action_heatmap_{agent_name}')
        return fig

    # ------------------------------------------------------------------
    # 7. Regime-conditioned IS comparison
    # ------------------------------------------------------------------

    def plot_regime_comparison(
        self,
        is_by_regime: Dict[str, Dict[str, np.ndarray]],
    ) -> Figure:
        """
        Box plots of IS conditioned on volatility regime.

        Demonstrates that IQN-CVaR's advantage is concentrated
        in the stress regime (where fat tails matter most).

        Args:
            is_by_regime: nested dict:
                {agent_name: {'normal': is_array, 'stress': is_array}}

        Returns:
            Figure with grouped box plots
        """
        agents  = list(is_by_regime.keys())
        regimes = ['normal', 'stress']

        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)

        for r_idx, regime in enumerate(regimes):
            ax = axes[r_idx]
            data = []
            labels = []
            colors = []
            for a_idx, agent in enumerate(agents):
                arr = is_by_regime[agent].get(regime, np.array([]))
                if len(arr) > 0:
                    data.append(self._convert(arr))
                    labels.append(agent)
                    colors.append(_get_color(agent, a_idx))

            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True,
                                showfliers=True, flierprops=dict(markersize=3))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            ax.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE_TICK - 1)
            _apply_style(ax, f'{regime.capitalize()} Regime',
                         '', self._is_label() if r_idx == 0 else '')

        fig.suptitle('IS by Volatility Regime', fontsize=FONT_SIZE_TITLE + 2,
                     fontweight='bold', y=1.02)
        fig.tight_layout()
        self._save(fig, 'regime_comparison')
        return fig

    # ------------------------------------------------------------------
    # 8. Comparison heatmap (agents × metrics)
    # ------------------------------------------------------------------

    def plot_comparison_heatmap(
        self,
        all_results: List[Dict],
        metrics: Optional[List[str]] = None,
    ) -> Figure:
        """
        Colour-coded matrix: rows = agents, columns = metrics.

        Green = best, Red = worst for each metric column.
        Gives a quick visual overview for paper Table 3.

        Args:
            all_results: list of dicts from MetricsSuite.evaluate()
            metrics:     which metrics to include as columns

        Returns:
            Figure with annotated heatmap
        """
        sfx = '_bps' if self.bps else ''

        if metrics is None:
            metrics = [f'mean_IS{sfx}', f'std_IS{sfx}',
                       f'CVaR_0.90{sfx}', f'CVaR_0.95{sfx}',
                       f'max_IS{sfx}', 'GL_ratio']

        agents = [r.get('agent_name', f'Agent_{i}')
                  for i, r in enumerate(all_results)]
        n_agents  = len(agents)
        n_metrics = len(metrics)

        matrix = np.zeros((n_agents, n_metrics))
        for i, r in enumerate(all_results):
            for j, m in enumerate(metrics):
                matrix[i, j] = r.get(m, 0.0)

        fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5),
                                        max(4, n_agents * 0.8)))

        # For IS-like metrics, lower is better → reverse colormap
        # For GL_ratio, higher is better → normal colormap
        # Use a diverging normalisation per column
        from matplotlib.colors import Normalize

        # Normalise each column independently
        norm_matrix = np.zeros_like(matrix)
        for j in range(n_metrics):
            col = matrix[:, j]
            col_min, col_max = col.min(), col.max()
            if col_max - col_min > 1e-12:
                norm_matrix[:, j] = (col - col_min) / (col_max - col_min)
            else:
                norm_matrix[:, j] = 0.5

        # For GL_ratio, higher is better → invert
        for j, m in enumerate(metrics):
            if 'GL' in m or 'sortino' in m:
                norm_matrix[:, j] = 1.0 - norm_matrix[:, j]

        im = ax.imshow(norm_matrix, cmap='RdYlGn_r', aspect='auto',
                       vmin=0, vmax=1)

        # Annotate cells with actual values
        for i in range(n_agents):
            for j in range(n_metrics):
                val = matrix[i, j]
                fmt = f'{val:.2f}' if abs(val) < 100 else f'{val:.1f}'
                ax.text(j, i, fmt, ha='center', va='center',
                        fontsize=FONT_SIZE_TICK,
                        color='black' if 0.3 < norm_matrix[i, j] < 0.7 else 'white')

        # Clean metric names for display
        display_names = []
        for m in metrics:
            name = m.replace('_bps', '').replace('_', ' ')
            display_names.append(name)

        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels(display_names, fontsize=FONT_SIZE_TICK,
                           rotation=30, ha='right')
        ax.set_yticks(range(n_agents))
        ax.set_yticklabels(agents, fontsize=FONT_SIZE_TICK)

        _apply_style(ax, 'Agent Comparison (green = better)',
                     '', '')
        fig.colorbar(im, ax=ax, shrink=0.6,
                     label='Relative rank (0=best, 1=worst)')
        fig.tight_layout()
        self._save(fig, 'comparison_heatmap')
        return fig

    # ------------------------------------------------------------------
    # 9. CVaR sensitivity plot
    # ------------------------------------------------------------------

    def plot_cvar_sensitivity(
        self,
        alpha_results: Dict[float, Dict[str, float]],
    ) -> Figure:
        """
        Line plot: Mean IS and CVaR as a function of α.

        Shows how varying the CVaR confidence level α trades off
        mean performance against tail protection.

        Args:
            alpha_results: dict mapping α → metrics dict
                           e.g. {0.80: {...}, 0.85: {...}, ...}

        Returns:
            Figure with dual-axis plot (mean IS + CVaR)
        """
        fig, ax1 = plt.subplots(figsize=FIGSIZE_SINGLE)

        alphas    = sorted(alpha_results.keys())
        sfx       = '_bps' if self.bps else ''
        means     = [alpha_results[a].get(f'mean_IS{sfx}', 0) for a in alphas]
        cvars     = []
        for a in alphas:
            cvar_key = f'CVaR_{a:.2f}{sfx}'
            # If that exact key doesn't exist, compute from 0.95
            cvars.append(alpha_results[a].get(cvar_key,
                         alpha_results[a].get(f'CVaR_0.95{sfx}', 0)))

        color1 = '#1f77b4'
        color2 = '#d62728'

        ax1.plot(alphas, means, 'o-', color=color1, linewidth=2.0,
                 markersize=8, label='Mean IS')
        ax1.set_xlabel('CVaR confidence level α', fontsize=FONT_SIZE_LABEL)
        ax1.set_ylabel(f'Mean {self._is_label()}', fontsize=FONT_SIZE_LABEL,
                       color=color1)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=FONT_SIZE_TICK)

        ax2 = ax1.twinx()
        ax2.plot(alphas, cvars, 's--', color=color2, linewidth=2.0,
                 markersize=8, label='CVaR')
        ax2.set_ylabel(f'CVaR {self._is_label()}', fontsize=FONT_SIZE_LABEL,
                       color=color2)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=FONT_SIZE_TICK)

        ax1.set_title('CVaR Sensitivity Analysis',
                      fontsize=FONT_SIZE_TITLE, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   fontsize=FONT_SIZE_LEGEND, loc='upper left')

        fig.tight_layout()
        self._save(fig, 'cvar_sensitivity')
        return fig

    # ------------------------------------------------------------------
    # 10. Per-step reward distribution (within-episode analysis)
    # ------------------------------------------------------------------

    def plot_reward_per_step(
        self,
        trackers: Dict[str, EpisodeTracker],
        n_periods: int,
    ) -> Figure:
        """
        Mean reward at each decision step, with ±1 std band.

        Reveals how agents pace their shortfall over the horizon.
        IQN-CVaR may incur more IS early (conservative selling)
        but less in the tail end (protected against adverse moves).

        Args:
            trackers:  dict mapping agent_name → EpisodeTracker
            n_periods: N decision periods

        Returns:
            Figure with per-step reward profiles
        """
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        steps = np.arange(n_periods)

        for idx, (name, tracker) in enumerate(trackers.items()):
            color = _get_color(name, idx)

            # Collect per-step rewards across episodes
            step_rewards = [[] for _ in range(n_periods)]
            for traj in tracker.all_trajectories:
                for t, step in enumerate(traj):
                    if t < n_periods:
                        step_rewards[t].append(step.reward)

            means = np.array([np.mean(sr) if sr else 0.0
                              for sr in step_rewards])
            stds  = np.array([np.std(sr) if len(sr) > 1 else 0.0
                              for sr in step_rewards])

            ax.plot(steps, means, linewidth=2.0, color=color,
                    label=name, alpha=0.9)
            ax.fill_between(steps, means - stds, means + stds,
                            color=color, alpha=0.15)

        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        _apply_style(ax, 'Per-Step Reward Profile',
                     'Decision Period t', 'Reward (normalised IS contribution)')
        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
        fig.tight_layout()
        self._save(fig, 'reward_per_step')
        return fig