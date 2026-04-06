"""
week-02/src/policy_improvement.py

Demonstrates the Policy Improvement step (S&B §4.2) on the 4×4 GridWorld.

Covers
──────
  1. Q-value computation     — Q(s,a) for every (state, action) pair
  2. Greedy improvement      — new_policy = argmax_a Q(s,a)
  3. Policy-stable detection — did any state's greedy action change?
  4. Tie-breaking modes      — "first", "uniform", "random"
  5. One full eval → improve cycle and proof of the Improvement Theorem
  6. Rich figure with 7 panels saved to week-02/policy_improvement.png

The Policy Improvement Theorem (S&B Theorem 4.2)
─────────────────────────────────────────────────
  Let π' be the greedy policy w.r.t. v_π.  Then for all s:

      v_{π'}(s) ≥ v_π(s)

  Proof sketch: the greedy action satisfies
      q_π(s, π'(s)) ≥ v_π(s)   (by definition of argmax)

  Unrolling this inequality one step at a time and applying the
  Bellman expectation equation yields v_{π'} ≥ v_π everywhere.

  Equality holds iff π and π' are both already optimal.

Usage:
    python week-02/src/policy_improvement.py

Commit: feat(w02): add policy_improvement() — greedy step
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as mgridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.dirname(__file__))
from gridworld import GridWorld, ACTION_NAMES, SB_GROUND_TRUTH

# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#0f0f14"
PANEL   = "#1a1a26"
PANEL2  = "#13131e"
MUTED   = "#555568"
GRID_LN = "#2a2a3a"
TEXT    = "#c8c8dc"
TDIM    = "#6a6a80"
ACCENT  = "#7C6AF7"
GREEN   = "#5ED4A0"
RED     = "#F76A6A"
AMBER   = "#F7C26A"
BLUE    = "#6AB8F7"


# ─────────────────────────────────────────────────────────────────────────────
# Drawing primitives  (self-contained — no viz.py import needed)
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax, grid=False):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(MUTED)
    if grid:
        ax.grid(color=GRID_LN, lw=0.4, ls="--", alpha=0.7)


def _draw_cells(ax, values, gw, norm, cmap="RdYlGn",
                labels=None, fontsize=10, alpha=1.0):
    """
    Draw coloured grid cells.
    values  : (n_states,) array used for colour mapping.
    labels  : (n_states,) string array for cell text.  Defaults to values.
    """
    cm = plt.get_cmap(cmap)
    for s in range(gw.n_states):
        r, c    = gw.to_rc(s)
        nv      = float(norm(values[s]))
        face    = cm(nv)
        is_term = gw.is_terminal(s)

        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.48, r - 0.48), 0.96, 0.96,
            boxstyle="square,pad=0",
            facecolor=(*face[:3], face[3] * alpha),
            edgecolor=AMBER if is_term else MUTED,
            linewidth=2.0 if is_term else 0.5,
        ))

        txt = (f"{values[s]:.1f}" if labels is None else labels[s])
        brt = nv
        col = "#111118" if 0.30 < brt < 0.72 else TEXT
        ax.text(c, r, txt, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=col)

    ax.set_xlim(-0.52, gw.size - 0.48)
    ax.set_ylim(gw.size - 0.48, -0.52)
    ax.set_xticks(range(gw.size)); ax.set_yticks(range(gw.size))
    ax.set_xticklabels(range(gw.size), color=TDIM, fontsize=7)
    ax.set_yticklabels(range(gw.size), color=TDIM, fontsize=7)
    ax.set_aspect("equal")


def _arrows(ax, policy, gw, colour=TEXT, alpha=0.9, scale=0.34):
    """Overlay greedy arrows from a policy matrix."""
    dir_map = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
    for s in range(gw.n_states):
        r, c = gw.to_rc(s)
        if gw.is_terminal(s):
            ax.text(c, r, "★", ha="center", va="center",
                    fontsize=14, color=AMBER, alpha=0.9)
            continue
        for a in range(gw.n_actions):
            if policy[s, a] > 0:
                dc, dr = dir_map[a]
                ax.annotate(
                    "", xy=(c + dc * scale, r + dr * scale),
                    xytext=(c, r),
                    arrowprops=dict(arrowstyle="-|>", color=colour,
                                   alpha=alpha, lw=1.4, mutation_scale=10),
                )


def _cbar(fig, ax, norm, cmap="RdYlGn", label=""):
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                      ax=ax, fraction=0.044, pad=0.04, shrink=0.85)
    cb.ax.tick_params(colors=TEXT, labelsize=7)
    cb.set_label(label, color=TDIM, fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grid(label, arr, size, fmt="{:7.2f}"):
    print(f"\n{label}")
    print("  " + "─" * (size * 9))
    for r in range(size):
        row = arr[r * size:(r + 1) * size]
        print("  " + "  ".join(fmt.format(v) for v in row))
    print("  " + "─" * (size * 9))


def _policy_grid(label, policy, gw):
    print(f"\n{label}")
    size = gw.size
    print("  " + "─" * (size * 6))
    for r in range(size):
        row_str = "  "
        for c in range(size):
            s = gw.to_state(r, c)
            if gw.is_terminal(s):
                row_str += "  T  "
            else:
                # show all actions with nonzero prob
                acts = [ACTION_NAMES[a] for a in range(gw.n_actions)
                        if policy[s, a] > 1e-9]
                row_str += f"{''.join(acts):^5}"
        print(row_str)
    print("  " + "─" * (size * 6))


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(gw, V_rand, pi_rand, V_imp, pi_imp, Q, stable,
                 save_path="policy_improvement.png"):
    """
    Seven-panel figure documenting the full improvement step.

    Row 0  [A] V under π_rand   [B] Q-table heatmap   [C] Improvement ΔV
    Row 1  [D] π_rand arrows    [E] π_imp arrows       [F] Changed-state mask
    Row 2  [G] Q-value bar for a chosen state (state 6)
    """
    fig = plt.figure(figsize=(16, 12), facecolor=BG, constrained_layout=True)
    fig.suptitle(
        "Week 02 — Policy Improvement Step  (S&B §4.2, GridWorld 4×4, γ=1)",
        color=TEXT, fontsize=12, fontweight="bold",
    )

    gs_top = mgridspec.GridSpec(2, 3, figure=fig,
                                hspace=0.42, wspace=0.35,
                                left=0.04, right=0.97,
                                top=0.92, bottom=0.36)
    gs_bot = mgridspec.GridSpec(1, 2, figure=fig,
                                left=0.04, right=0.97,
                                top=0.30, bottom=0.06,
                                wspace=0.35)

    axA = fig.add_subplot(gs_top[0, 0])
    axB = fig.add_subplot(gs_top[0, 1])
    axC = fig.add_subplot(gs_top[0, 2])
    axD = fig.add_subplot(gs_top[1, 0])
    axE = fig.add_subplot(gs_top[1, 1])
    axF = fig.add_subplot(gs_top[1, 2])
    axG = fig.add_subplot(gs_bot[0, 0])
    axH = fig.add_subplot(gs_bot[0, 1])

    for ax in (axA, axB, axC, axD, axE, axF, axG, axH):
        _style(ax)

    label_kw = dict(transform=None, fontsize=9, fontweight="bold",
                    color=ACCENT, va="top")

    # ── A: V under random policy ─────────────────────────────────────────────
    norm_v = Normalize(vmin=V_rand.min() - 1e-9, vmax=V_rand.max() + 1e-9)
    _draw_cells(axA, V_rand, gw, norm_v)
    _cbar(fig, axA, norm_v, label="V(s)")
    axA.set_title("A — V(s) under uniform random π", color=TEXT, fontsize=9, pad=6)

    # ── B: Q-table as a flat heatmap ─────────────────────────────────────────
    # Flatten Q to shape (n_states * n_actions,) displayed as a grid
    Q_flat   = Q.flatten()                          # length 64
    norm_q   = Normalize(vmin=Q_flat.min() - 1e-9, vmax=Q_flat.max() + 1e-9)
    Q_img    = Q.T                                  # shape (4, 16): rows=actions, cols=states
    axB.imshow(Q_img, cmap="RdYlGn", norm=norm_q, aspect="auto")
    axB.set_xticks(range(gw.n_states))
    axB.set_xticklabels(range(gw.n_states), fontsize=6, color=TDIM)
    axB.set_yticks(range(gw.n_actions))
    axB.set_yticklabels([ACTION_NAMES[a] for a in range(gw.n_actions)],
                        fontsize=9, color=TEXT)
    axB.set_xlabel("State s", color=TEXT, fontsize=8)
    axB.set_ylabel("Action a", color=TEXT, fontsize=8)
    # annotate each cell with Q value
    for a in range(gw.n_actions):
        for s in range(gw.n_states):
            nv  = float(norm_q(Q[s, a]))
            col = "#111118" if 0.28 < nv < 0.72 else TEXT
            axB.text(s, a, f"{Q[s,a]:.0f}",
                     ha="center", va="center", fontsize=5.5,
                     fontweight="bold", color=col)
    _cbar(fig, axB, norm_q, label="Q(s,a)")
    axB.set_title("B — Q(s,a) for all states × actions", color=TEXT,
                  fontsize=9, pad=6)

    # ── C: Improvement ΔV = V_imp − V_rand ───────────────────────────────────
    delta_V  = V_imp - V_rand
    norm_dv  = Normalize(vmin=delta_V.min() - 1e-9, vmax=delta_V.max() + 1e-9)
    _draw_cells(axC, delta_V, gw, norm_dv, cmap="YlOrRd")
    _cbar(fig, axC, norm_dv, cmap="YlOrRd", label="ΔV(s)")
    axC.set_title("C — Improvement ΔV(s) = V_improved − V_random",
                  color=TEXT, fontsize=9, pad=6)

    # ── D: Arrows of original (uniform) policy ───────────────────────────────
    _draw_cells(axD, V_rand, gw, norm_v, alpha=0.5, labels=np.array(
        ["" for _ in range(gw.n_states)]))
    _arrows(axD, pi_rand, gw, colour=BLUE, alpha=0.7, scale=0.30)
    _cbar(fig, axD, norm_v, label="V(s)")
    axD.set_title("D — Uniform random policy π₀\n(all four arrows, equal weight)",
                  color=TEXT, fontsize=9, pad=6)

    # ── E: Arrows of improved policy ─────────────────────────────────────────
    _draw_cells(axE, V_imp, gw,
                Normalize(vmin=V_imp.min()-1e-9, vmax=V_imp.max()+1e-9),
                alpha=0.55,
                labels=np.array([f"{v:.0f}" for v in V_imp]))
    _arrows(axE, pi_imp, gw, colour=GREEN, alpha=0.88, scale=0.32)
    _cbar(fig, axE,
          Normalize(vmin=V_imp.min()-1e-9, vmax=V_imp.max()+1e-9),
          label="V_improved(s)")
    axE.set_title("E — Improved greedy policy π'\n(argmax Q(s,a))",
                  color=TEXT, fontsize=9, pad=6)

    # ── F: States where policy changed ───────────────────────────────────────
    changed = np.zeros(gw.n_states)
    for s in range(gw.n_states):
        if gw.is_terminal(s):
            continue
        old_best = set(np.where(pi_rand[s] > 1e-9)[0])
        new_best = set(np.where(pi_imp[s]  > 1e-9)[0])
        changed[s] = 0.0 if old_best == new_best else 1.0

    n_changed = int(changed.sum())
    norm_ch = Normalize(vmin=-0.1, vmax=1.1)
    _draw_cells(axF, changed, gw, norm_ch, cmap="RdYlGn_r",
                labels=np.array(
                    ["T" if gw.is_terminal(s)
                     else ("changed" if changed[s] else "same")
                     for s in range(gw.n_states)]),
                fontsize=7)
    axF.set_title(
        f"F — States where greedy action changed  ({n_changed} of "
        f"{gw.n_states - len(gw.terminal_states)} non-terminal)",
        color=TEXT, fontsize=9, pad=6)

    policy_stable_str = "✓ policy stable" if stable else "✗ policy changed"
    axF.text(0.5, -0.10, policy_stable_str,
             transform=axF.transAxes, ha="center",
             fontsize=9, fontweight="bold",
             color=GREEN if stable else RED)

    # ── G: Q-value bar chart for a focus state ───────────────────────────────
    focus = 6   # state 6, row=1 col=2 — non-terminal, interesting Q spread
    q6    = Q[focus]
    best  = q6.max()
    cols  = [GREEN if abs(q - best) < 1e-4 else ACCENT for q in q6]
    bars  = axG.bar(range(gw.n_actions), q6, color=cols,
                    edgecolor=PANEL2, lw=0.5, width=0.6)
    axG.set_xticks(range(gw.n_actions))
    axG.set_xticklabels([f"{ACTION_NAMES[a]}\n(a={a})" for a in range(gw.n_actions)],
                        color=TEXT, fontsize=9)
    axG.set_ylabel("Q(s=6, a)", color=TEXT, fontsize=8)
    axG.axhline(best, color=GREEN, lw=1.0, ls="--", alpha=0.6)
    axG.set_title(
        f"G — Q-values at state 6  (row=1, col=2)\n"
        f"Greedy action = {ACTION_NAMES[int(np.argmax(q6))]}  "
        f"(Q={best:.2f})",
        color=TEXT, fontsize=9, pad=6,
    )
    for bar, q in zip(bars, q6):
        axG.text(bar.get_x() + bar.get_width() / 2,
                 q - 0.3, f"{q:.2f}",
                 ha="center", va="top", fontsize=8,
                 color="#111118" if abs(q - best) < 1e-4 else TEXT)
    _style(axG, grid=True)

    # ── H: Improvement theorem proof trace ───────────────────────────────────
    # Show per-state: q_π(s, π'(s)) and v_π(s) — the former must be ≥ latter
    q_pi_prime = np.array([
        Q[s, int(np.argmax(pi_imp[s]))] if not gw.is_terminal(s) else 0.0
        for s in range(gw.n_states)
    ])
    x   = np.arange(gw.n_states)
    w   = 0.38
    axH.bar(x - w / 2, V_rand, width=w, color=BLUE,
            edgecolor=PANEL2, lw=0.4, label="v_π(s)")
    axH.bar(x + w / 2, q_pi_prime, width=w, color=GREEN,
            edgecolor=PANEL2, lw=0.4, label="q_π(s, π'(s))")

    # Mark any state where the theorem is satisfied (should be ALL of them)
    for s in range(gw.n_states):
        if not gw.is_terminal(s):
            ok = q_pi_prime[s] >= V_rand[s] - 1e-6
            if not ok:
                axH.axvline(s, color=RED, lw=1.5, alpha=0.7)   # violation!

    axH.axhline(0, color=MUTED, lw=0.6, ls="--")
    axH.set_xticks(x)
    axH.set_xticklabels(x, fontsize=6, color=TDIM)
    axH.set_xlabel("State s", color=TEXT, fontsize=8)
    axH.set_ylabel("Value", color=TEXT, fontsize=8)
    axH.set_title(
        "H — Policy Improvement Theorem check\n"
        "q_π(s, π'(s)) ≥ v_π(s)  must hold for every state",
        color=TEXT, fontsize=9, pad=6,
    )
    axH.legend(fontsize=8, facecolor=PANEL2, labelcolor=TEXT, framealpha=0.8)
    _style(axH, grid=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Figure saved → {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  Week 02 — Policy Improvement  (S&B §4.2)")
    print("  GridWorld 4×4, γ=1.0")
    print("=" * 64)

    gw = GridWorld(size=4, gamma=1.0)

    # ── Step 0: Evaluate the uniform random policy ───────────────────────────
    print("\n[ Step 0 ]  Evaluate uniform random policy …")
    pi_rand          = gw.uniform_random_policy()
    V_rand, deltas   = gw.iterative_policy_evaluation(
        pi_rand, theta=1e-6, track_delta=True)
    print(f"  Converged in {len(deltas)} sweeps")
    _grid("V(s) — uniform random π:", V_rand, gw.size)

    # ── Step 1: Compute the Q-table ──────────────────────────────────────────
    print("\n[ Step 1 ]  Compute Q(s,a) for every (state, action) pair …")
    Q = gw.q_table(V_rand)
    print(f"  Q shape: {Q.shape}  (n_states × n_actions)")

    # Print a few spot-checks
    print("\n  Q-values for selected states:")
    print(f"  {'State':<8}  {'↑':>7}  {'↓':>7}  {'←':>7}  {'→':>7}  {'greedy':>8}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}")
    for s in [1, 4, 5, 6, 9, 10, 14]:
        q   = Q[s]
        g_a = int(np.argmax(q))
        print(f"  s={s:<6}  {q[0]:>7.2f}  {q[1]:>7.2f}  {q[2]:>7.2f}"
              f"  {q[3]:>7.2f}  {ACTION_NAMES[g_a]:>8}")

    # ── Step 2: Policy improvement ───────────────────────────────────────────
    print("\n[ Step 2 ]  Run policy_improvement(V_rand) …")
    pi_imp, stable, Q2 = gw.policy_improvement(V_rand, old_policy=pi_rand)
    print(f"  policy_stable = {stable}")
    print(f"  (False means some state got a better greedy action → keep iterating)")

    # Count changed states
    n_changed = sum(
        1 for s in range(gw.n_states)
        if not gw.is_terminal(s)
        and set(np.where(pi_rand[s] > 1e-9)[0]) !=
            set(np.where(pi_imp[s]  > 1e-9)[0])
    )
    n_nonterminal = gw.n_states - len(gw.terminal_states)
    print(f"  States with changed greedy action: {n_changed} / {n_nonterminal}")

    _policy_grid("π_rand  (uniform — all four arrows):", pi_rand, gw)
    _policy_grid("π_imp   (greedy — single best arrow):", pi_imp, gw)

    # ── Step 3: Evaluate improved policy ─────────────────────────────────────
    print("\n[ Step 3 ]  Evaluate improved policy …")
    V_imp, d_imp = gw.iterative_policy_evaluation(
        pi_imp, theta=1e-6, track_delta=True)
    print(f"  Converged in {len(d_imp)} sweeps  "
          f"(was {len(deltas)} sweeps for random policy)")
    _grid("V(s) — improved π':", V_imp, gw.size)

    # ── Step 4: Verify the Improvement Theorem ───────────────────────────────
    print("\n[ Step 4 ]  Verify Policy Improvement Theorem: v_π'(s) ≥ v_π(s) …")
    improvements = V_imp - V_rand
    violations   = (improvements < -1e-6).sum()
    print(f"  Max improvement  : {improvements.max():+.4f}")
    print(f"  Min improvement  : {improvements.min():+.4f}")
    print(f"  Violations (< 0) : {violations}  "
          f"{'✓ PASS — theorem holds' if violations == 0 else '✗ FAIL'}")
    print(f"  Mean V:  {V_rand[~np.array([gw.is_terminal(s) for s in range(gw.n_states)])].mean():.2f}"
          f"  →  "
          f"{V_imp[~np.array([gw.is_terminal(s) for s in range(gw.n_states)])].mean():.2f}"
          f"  (non-terminal states)")

    # ── Step 5: Tie-breaking modes ───────────────────────────────────────────
    print("\n[ Step 5 ]  Demonstrate tie_breaking modes …")
    for mode in ("first", "uniform", "random"):
        pi_tb, _, _ = gw.policy_improvement(V_rand, tie_breaking=mode)
        # Show row 0 (has many ties because all cells point toward terminal 0)
        row0_actions = {s: [ACTION_NAMES[a] for a in range(gw.n_actions)
                            if pi_tb[s, a] > 1e-9]
                        for s in range(4)}
        print(f"  tie_breaking='{mode}':  row-0 choices = "
              + ", ".join(f"s{s}→{''.join(v)}" for s, v in row0_actions.items()))

    # ── Step 6: Second improvement step ──────────────────────────────────────
    print("\n[ Step 6 ]  Apply a second improvement step …")
    pi_imp2, stable2, _ = gw.policy_improvement(V_imp, old_policy=pi_imp)
    print(f"  policy_stable after 2nd improvement = {stable2}")
    if stable2:
        print("  → Both π and V are optimal.  "
              "Policy Iteration would stop here.")

    V_imp2, _ = gw.iterative_policy_evaluation(pi_imp2, theta=1e-6)
    print(f"  Mean V after 2nd step: "
          f"{V_imp2[~np.array([gw.is_terminal(s) for s in range(gw.n_states)])].mean():.4f}")

    # ── Figure ───────────────────────────────────────────────────────────────
    print("\nGenerating figure …")
    build_figure(gw, V_rand, pi_rand, V_imp, pi_imp, Q, stable,
                 save_path="policy_improvement.png")


if __name__ == "__main__":
    main()