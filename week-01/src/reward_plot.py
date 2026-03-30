"""
week-01/src/reward_plot.py

Dedicated reward tracking and visualisation for the LinearCorridor RL loop.
Imports TrainingStats + the environment/agent from rl_loop.py, runs a full
training session, then produces a focused 6-panel reward analysis figure.

Panels
──────
  1. Raw rewards + rolling mean/std band      — did the agent improve?
  2. Rolling success rate (win %)             — how often does it reach the goal?
  3. Reward distribution histogram            — shape of the reward signal
  4. Episode-length scatter (coloured by win) — are wins short and losses long?
  5. Cumulative reward over time              — total value accumulated
  6. TD-error decay                           — is learning converging?

Usage
─────
  python week-01/src/reward_plot.py           # train + plot
  python week-01/src/reward_plot.py --episodes 1000 --cells 15

Commit: feat(w01): reward tracking plot, wrap up week 1
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# Re-use everything from custom_rl_loop-2.py — no duplication
from custom_rl_loop_2 import (
    LinearCorridor,
    EpsilonGreedyAgent,
    TrainingStats,
    train,
    smooth,
)


# ─────────────────────────────────────────────────────────────────────────────
# Palette  (consistent with rl_loop.py dark theme)
# ─────────────────────────────────────────────────────────────────────────────

BG_OUTER  = "#0f0f14"
BG_PANEL  = "#1a1a26"
BG_PANEL2 = "#16161f"
COL_MUTED = "#555568"
COL_TEXT  = "#c8c8dc"
COL_GRID  = "#2a2a3a"

COL_RAW   = "#555568"   # raw reward trace
COL_MEAN  = "#7C6AF7"   # rolling mean  (purple)
COL_BAND  = "#7C6AF7"   # std band fill
COL_WIN   = "#5ED4A0"   # success / positive
COL_LOSS  = "#F76A6A"   # failure / negative
COL_CUMUL = "#F7C26A"   # cumulative reward (amber)
COL_TD    = "#6AB8F7"   # TD error (blue)
COL_EPS   = "#F76A6A"   # epsilon (red)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: rolling statistics
# ─────────────────────────────────────────────────────────────────────────────

def rolling(x: np.ndarray, window: int):
    """Return (centres, means, stds) with valid convolution length."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window:
        window = max(1, n)

    means = np.convolve(x, np.ones(window) / window, mode="valid")
    # rolling std via E[x²] - E[x]²
    means_sq = np.convolve(x ** 2, np.ones(window) / window, mode="valid")
    stds = np.sqrt(np.maximum(0.0, means_sq - means ** 2))

    # Centre index of each valid window
    offset = window // 2
    centres = np.arange(offset, offset + len(means))
    return centres, means, stds


def rolling_rate(x: np.ndarray, window: int):
    """Rolling mean of a 0/1 array — e.g. success flags."""
    return rolling(x, window)


# ─────────────────────────────────────────────────────────────────────────────
# Main plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_rewards(
    stats: TrainingStats,
    env: LinearCorridor,
    agent: EpsilonGreedyAgent,
    window: int = 40,
    save_path: str = "reward_tracking.png",
) -> None:
    """
    Six-panel reward analysis figure.

    Parameters
    ----------
    stats     : TrainingStats collected during train()
    env       : the LinearCorridor instance (used for metadata)
    agent     : trained EpsilonGreedyAgent (used for metadata)
    window    : rolling-average window size in episodes
    save_path : where to save the PNG
    """

    rewards   = np.array(stats.rewards,   dtype=float)
    lengths   = np.array(stats.lengths,   dtype=float)
    td_errors = np.array(stats.td_errors, dtype=float)
    epsilons  = np.array(stats.epsilons,  dtype=float)
    successes = np.array(stats.successes, dtype=float)
    eps_x     = np.arange(1, len(rewards) + 1)          # episode numbers
    n         = len(rewards)

    # ── Figure skeleton ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10), facecolor=BG_OUTER)

    title = (
        f"Reward tracking — LinearCorridor  "
        f"(n_cells={env.n_cells}, α={agent.alpha}, γ={agent.gamma}, "
        f"ε {agent.epsilon_start}→{agent.epsilon_min})"
    )
    fig.suptitle(title, color=COL_TEXT, fontsize=11, fontweight="bold", y=0.995)

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.50, wspace=0.35,
        left=0.07, right=0.97, top=0.95, bottom=0.08,
    )

    axes = [
        fig.add_subplot(gs[0, :2]),  # [0] wide: reward trace
        fig.add_subplot(gs[0, 2]),   # [1] success rate
        fig.add_subplot(gs[1, 0]),   # [2] reward histogram
        fig.add_subplot(gs[1, 1]),   # [3] length scatter (win/loss colour)
        fig.add_subplot(gs[1, 2]),   # [4] cumulative reward  (twin-axis for ε)
    ]

    # shared style
    for ax in axes:
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=COL_TEXT, labelsize=8)
        ax.xaxis.label.set_color(COL_TEXT)
        ax.yaxis.label.set_color(COL_TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(COL_MUTED)
        ax.grid(color=COL_GRID, linewidth=0.4, linestyle="--", alpha=0.6)

    def title_style(ax, txt):
        ax.set_title(txt, color=COL_TEXT, fontsize=9, pad=6)

    # ── Panel 0 : Raw rewards + rolling mean ± std ───────────────────────────
    ax = axes[0]
    rc, rm, rs = rolling(rewards, window)

    ax.plot(eps_x, rewards, color=COL_RAW, lw=0.6, alpha=0.35, label="raw reward")
    ax.fill_between(rc, rm - rs, rm + rs,
                    color=COL_BAND, alpha=0.15, label=f"±1 std  (w={window})")
    ax.plot(rc, rm, color=COL_MEAN, lw=2.0, label=f"rolling mean (w={window})")

    # annotate optimal reward
    optimal_r = env.goal_reward - (env.n_cells - 1) * abs(env.step_penalty)
    ax.axhline(optimal_r, color=COL_WIN, lw=0.9, ls=":", alpha=0.8,
               label=f"optimal  ({optimal_r:.1f})")
    ax.axhline(0, color=COL_MUTED, lw=0.5, ls="--", alpha=0.5)

    # shade exploration phase
    decay_end = min(agent.epsilon_decay_episodes, n)
    ax.axvspan(1, decay_end, color=COL_EPS, alpha=0.05,
               label=f"exploration phase (ep 1–{decay_end})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    title_style(ax, "Episode reward  (raw + rolling mean ± std)")
    ax.legend(fontsize=7.5, facecolor=BG_PANEL2, labelcolor=COL_TEXT,
              framealpha=0.7, loc="lower right")

    # annotate final mean in top-right corner
    final_mean = rm[-1] if len(rm) else rewards[-1]
    ax.annotate(
        f"final avg  {final_mean:+.2f}",
        xy=(0.98, 0.92), xycoords="axes fraction",
        ha="right", va="top",
        color=COL_MEAN, fontsize=8, fontweight="bold",
    )

    # ── Panel 1 : Rolling success rate ───────────────────────────────────────
    ax = axes[1]
    sc, sm_rate, _ = rolling_rate(successes, window)

    ax.fill_between(sc, 0, sm_rate * 100, color=COL_WIN, alpha=0.20)
    ax.plot(sc, sm_rate * 100, color=COL_WIN, lw=2.0)
    ax.axhline(100, color=COL_WIN, lw=0.7, ls=":", alpha=0.5)
    ax.set_ylim(-5, 108)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win rate")
    title_style(ax, f"Rolling success rate  (w={window})")

    # annotate current win rate
    final_rate = sm_rate[-1] * 100 if len(sm_rate) else successes[-1] * 100
    ax.annotate(
        f"{final_rate:.1f}%",
        xy=(0.97, 0.92), xycoords="axes fraction",
        ha="right", color=COL_WIN, fontsize=10, fontweight="bold",
    )

    # ── Panel 2 : Reward histogram ────────────────────────────────────────────
    ax = axes[2]

    bins = min(40, max(10, n // 15))
    neg_mask = rewards < 0
    pos_mask = rewards >= 0

    if neg_mask.any():
        ax.hist(rewards[neg_mask], bins=bins // 2,
                color=COL_LOSS, alpha=0.75, edgecolor=BG_PANEL, lw=0.4,
                label="failed episodes")
    if pos_mask.any():
        ax.hist(rewards[pos_mask], bins=bins,
                color=COL_WIN, alpha=0.75, edgecolor=BG_PANEL, lw=0.4,
                label="successful episodes")

    ax.axvline(float(np.mean(rewards)), color=COL_MEAN, lw=1.5, ls="--",
               label=f"mean  {np.mean(rewards):.2f}")
    ax.axvline(float(np.median(rewards)), color=COL_CUMUL, lw=1.2, ls=":",
               label=f"median  {np.median(rewards):.2f}")

    ax.set_xlabel("Total reward")
    ax.set_ylabel("Count")
    title_style(ax, "Reward distribution")
    ax.legend(fontsize=7, facecolor=BG_PANEL2, labelcolor=COL_TEXT,
              framealpha=0.7)

    # ── Panel 3 : Episode-length scatter coloured by outcome ─────────────────
    ax = axes[3]

    win_idx  = np.where(successes == 1)[0]
    loss_idx = np.where(successes == 0)[0]

    if len(loss_idx):
        ax.scatter(loss_idx + 1, lengths[loss_idx],
                   c=COL_LOSS, s=4, alpha=0.35, linewidths=0, label="failed")
    if len(win_idx):
        ax.scatter(win_idx + 1, lengths[win_idx],
                   c=COL_WIN, s=4, alpha=0.35, linewidths=0, label="success")

    # rolling mean of lengths
    lc, lm, _ = rolling(lengths, window)
    ax.plot(lc, lm, color=COL_MEAN, lw=1.8, label=f"rolling mean (w={window})")

    optimal_steps = env.n_cells - 1
    ax.axhline(optimal_steps, color=COL_TEXT, lw=0.8, ls="--", alpha=0.5,
               label=f"optimal  ({optimal_steps} steps)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    title_style(ax, "Episode length  (green = goal reached)")
    ax.legend(fontsize=7, facecolor=BG_PANEL2, labelcolor=COL_TEXT,
              framealpha=0.7)

    # ── Panel 4 : Cumulative reward + ε overlay ───────────────────────────────
    ax = axes[4]
    ax_eps = ax.twinx()   # second y-axis for epsilon
    ax_eps.set_facecolor(BG_PANEL)
    ax_eps.tick_params(colors=COL_EPS, labelsize=7)
    ax_eps.spines["right"].set_edgecolor(COL_EPS)
    ax_eps.spines["left"].set_edgecolor(COL_MUTED)
    for s in ["top", "bottom"]:
        ax_eps.spines[s].set_edgecolor(COL_MUTED)

    cumulative = np.cumsum(rewards)
    ax.plot(eps_x, cumulative, color=COL_CUMUL, lw=1.8, label="cumulative reward")
    ax.fill_between(eps_x, 0, cumulative,
                    where=(cumulative >= 0), color=COL_CUMUL, alpha=0.12)
    ax.fill_between(eps_x, 0, cumulative,
                    where=(cumulative < 0),  color=COL_LOSS,  alpha=0.12)
    ax.axhline(0, color=COL_MUTED, lw=0.5, ls="--")

    ax_eps.plot(eps_x, epsilons, color=COL_EPS, lw=1.0,
                alpha=0.55, ls="--", label="ε")
    ax_eps.set_ylim(-0.05, 1.15)
    ax_eps.set_ylabel("ε", color=COL_EPS, fontsize=8)
    ax_eps.yaxis.label.set_color(COL_EPS)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    title_style(ax, "Cumulative reward  (dashed = ε schedule)")

    # combined legend
    handles = [
        Line2D([0], [0], color=COL_CUMUL, lw=1.8, label="cumulative reward"),
        Line2D([0], [0], color=COL_EPS,   lw=1.0, ls="--", label="ε (right axis)"),
    ]
    ax.legend(handles=handles, fontsize=7, facecolor=BG_PANEL2,
              labelcolor=COL_TEXT, framealpha=0.7)

    # annotate total accumulated reward
    ax.annotate(
        f"total  {cumulative[-1]:+,.0f}",
        xy=(0.05, 0.92), xycoords="axes fraction",
        ha="left", color=COL_CUMUL, fontsize=8, fontweight="bold",
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Plot saved → {save_path}")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train LinearCorridor agent and produce reward tracking plot."
    )
    p.add_argument("--episodes", type=int, default=500,
                   help="Number of training episodes  (default: 500)")
    p.add_argument("--cells",    type=int, default=10,
                   help="Corridor length              (default: 10)")
    p.add_argument("--alpha",    type=float, default=0.1,
                   help="Learning rate α              (default: 0.1)")
    p.add_argument("--gamma",    type=float, default=0.99,
                   help="Discount factor γ            (default: 0.99)")
    p.add_argument("--slip",     type=float, default=0.0,
                   help="Slip probability (stochastic, default: 0.0)")
    p.add_argument("--window",   type=int, default=40,
                   help="Rolling-average window size  (default: 40)")
    p.add_argument("--seed",     type=int, default=42,
                   help="Global random seed           (default: 42)")
    p.add_argument("--out",      type=str,
                   default="reward_tracking.png",
                   help="Output PNG path")
    return p.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)

    print("=" * 62)
    print("  Week 01 — Reward Tracking Plot")
    print(f"  episodes={args.episodes}  cells={args.cells}  "
          f"α={args.alpha}  γ={args.gamma}  slip={args.slip}")
    print("=" * 62)

    # ── Build env + agent ────────────────────────────────────────────────────
    env = LinearCorridor(
        n_cells=args.cells,
        max_steps=args.cells * 25,
        slip_prob=args.slip,
    )

    agent = EpsilonGreedyAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay_episodes=int(args.episodes * 0.70),
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print("\nTraining …")
    stats = train(env, agent, n_episodes=args.episodes, verbose_every=100)

    # ── Summary stats ────────────────────────────────────────────────────────
    rewards = np.array(stats.rewards)
    wins    = np.array(stats.successes)

    print(f"\n{'─'*40}")
    print(f"  Total episodes   : {len(rewards)}")
    print(f"  Overall win rate : {wins.mean()*100:.1f}%")
    print(f"  Mean reward      : {rewards.mean():+.3f}")
    print(f"  Std  reward      : {rewards.std():.3f}")
    print(f"  Min / Max        : {rewards.min():+.2f} / {rewards.max():+.2f}")
    print(f"  Cumulative       : {rewards.sum():+,.1f}")

    # late-training window (last 20%)
    late = rewards[int(len(rewards) * 0.8):]
    print(f"\n  Last-20% avg     : {late.mean():+.3f}  "
          f"(win rate {wins[int(len(wins)*0.8):].mean()*100:.1f}%)")
    print(f"{'─'*40}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    print(f"\nGenerating reward tracking plot (window={args.window}) …")
    plot_rewards(stats, env, agent, window=args.window, save_path=args.out)

if __name__ == "__main__":
    main()