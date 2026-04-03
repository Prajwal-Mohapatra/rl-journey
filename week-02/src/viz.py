"""
week-02/src/viz.py

Standalone visualisation module for GridWorld value functions.
Imports GridWorld from gridworld.py — run this file directly to
generate all heatmap figures for the week-02 commit.

Figures produced
────────────────
  1. value_heatmap.png      — 6-panel deep-dive on V(s)
  2. sweep_evolution.png    — how V(s) evolves sweep-by-sweep during evaluation
  3. policy_comparison.png  — V(s) side-by-side for 4 different policies

Usage
─────
  python week-02/src/viz.py

Commit: feat(w02): visualize V(s) as heatmap
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as mgridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

# ── import the MDP class from the same package ───────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from gridworld_policy_evaluation import GridWorld, ACTION_NAMES, SB_GROUND_TRUTH

# ─────────────────────────────────────────────────────────────────────────────
# Shared palette  (matches rl_loop.py / gridworld.py)
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#0f0f14"
PANEL    = "#1a1a26"
PANEL2   = "#13131e"
MUTED    = "#555568"
GRID_LN  = "#2a2a3a"
TEXT     = "#c8c8dc"
TEXT_DIM = "#6a6a80"
ACCENT   = "#7C6AF7"   # purple
GREEN    = "#5ED4A0"   # success / positive
RED      = "#F76A6A"   # failure / negative
AMBER    = "#F7C26A"   # terminal / highlight
BLUE     = "#6AB8F7"   # info


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, grid: bool = False) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(MUTED)
    if grid:
        ax.grid(color=GRID_LN, lw=0.4, ls="--", alpha=0.7)


def _make_norm(V: np.ndarray) -> TwoSlopeNorm | Normalize:
    """
    Use TwoSlopeNorm when V spans negative → positive values so that
    zero always maps to the neutral mid-colour.  Fall back to Normalize
    for all-negative or all-positive arrays (e.g. uniform random policy).
    """
    vmin, vmax = float(V.min()), float(V.max())
    if vmin < -1e-6 and vmax > 1e-6:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return Normalize(vmin=vmin - 1e-9, vmax=vmax + 1e-9)


def _cell_text_colour(norm_val: float, cmap_name: str = "RdYlGn") -> str:
    """Pick dark/light text so it stays legible on any heatmap cell."""
    # mid-range colours (yellow zone of RdYlGn) need dark text
    return "#111118" if 0.30 < norm_val < 0.72 else TEXT


def _draw_grid_cells(
    ax: plt.Axes,
    V: np.ndarray,
    gw: GridWorld,
    norm: Normalize,
    cmap: str = "RdYlGn",
    show_values: bool = True,
    fontsize: int = 11,
    alpha: float = 1.0,
) -> None:
    """
    Core cell-drawing routine used by every heatmap panel.
    Draws coloured cells, optional value labels, and terminal outlines.
    """
    cmap_obj = plt.get_cmap(cmap)
    for s in range(gw.n_states):
        r, c = gw.to_rc(s)
        val       = V[s]
        nval      = float(norm(val))
        face      = cmap_obj(nval)
        is_term   = gw.is_terminal(s)

        rect = mpatches.FancyBboxPatch(
            (c - 0.48, r - 0.48), 0.96, 0.96,
            boxstyle="square,pad=0",
            facecolor=(*face[:3], face[3] * alpha),
            edgecolor=AMBER if is_term else MUTED,
            linewidth=2.2 if is_term else 0.5,
        )
        ax.add_patch(rect)

        if show_values:
            txt_col = _cell_text_colour(nval, cmap)
            ax.text(c, r, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=txt_col)

    ax.set_xlim(-0.52, gw.size - 0.48)
    ax.set_ylim(gw.size - 0.48, -0.52)   # row 0 at top
    ax.set_xticks(range(gw.size))
    ax.set_yticks(range(gw.size))
    ax.set_xticklabels([f"col {c}" for c in range(gw.size)],
                       color=TEXT_DIM, fontsize=7)
    ax.set_yticklabels([f"row {r}" for r in range(gw.size)],
                       color=TEXT_DIM, fontsize=7)
    ax.set_aspect("equal")
    ax.imshow(np.zeros((gw.size, gw.size)),   # invisible — just sets extent
              extent=[-0.5, gw.size - 0.5, gw.size - 0.5, -0.5],
              cmap="Greys", alpha=0.0)


def _add_colorbar(fig: plt.Figure, ax: plt.Axes,
                  norm: Normalize, cmap: str = "RdYlGn",
                  label: str = "V(s)") -> None:
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                      ax=ax, fraction=0.044, pad=0.04, shrink=0.85)
    cb.ax.tick_params(colors=TEXT, labelsize=7)
    cb.set_label(label, color=TEXT_DIM, fontsize=7)


def _add_policy_arrows(
    ax: plt.Axes,
    V: np.ndarray,
    gw: GridWorld,
    scale: float = 0.34,
    colour: str = TEXT,
    alpha: float = 0.85,
) -> None:
    """
    Overlay matplotlib quiver arrows showing the greedy policy.
    Ties (multiple equally-good actions) get one arrow per direction.
    Terminals get a gold "★".
    """
    # (Δcol, −Δrow) because imshow y-axis is flipped: row↑ = y↓
    quiver_dir = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

    for s in range(gw.n_states):
        r, c = gw.to_rc(s)

        if gw.is_terminal(s):
            ax.text(c, r, "★", ha="center", va="center",
                    fontsize=14, color=AMBER, alpha=0.9)
            continue

        # Q(s,a) for each action
        q_vals = np.array([
            sum(p * (rwd + (0. if dn else gw.gamma * V[sp]))
                for p, sp, rwd, dn in gw.P[s][a])
            for a in range(gw.n_actions)
        ])
        best = q_vals.max()
        for a in range(gw.n_actions):
            if abs(q_vals[a] - best) < 1e-4:
                dc, dr = quiver_dir[a]
                ax.annotate(
                    "", xy=(c + dc * scale, r + dr * scale),
                    xytext=(c, r),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=colour, alpha=alpha,
                        lw=1.4, mutation_scale=10,
                    ),
                )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — 6-panel V(s) deep dive
# ─────────────────────────────────────────────────────────────────────────────

def _panel_plain_heatmap(ax, V, gw, title, cmap="RdYlGn"):
    """Panel A: plain coloured cells + values."""
    norm = _make_norm(V)
    _style_ax(ax)
    _draw_grid_cells(ax, V, gw, norm, cmap=cmap, show_values=True, fontsize=11)
    _add_colorbar(ax.figure, ax, norm, cmap=cmap)
    ax.set_title(title, color=TEXT, fontsize=9, pad=6)


def _panel_heatmap_arrows(ax, V, gw, title):
    """Panel B: heatmap + greedy policy arrows."""
    norm = _make_norm(V)
    _style_ax(ax)
    _draw_grid_cells(ax, V, gw, norm, show_values=False, alpha=0.75)
    _add_policy_arrows(ax, V, gw)
    _add_colorbar(ax.figure, ax, norm)
    ax.set_title(title, color=TEXT, fontsize=9, pad=6)


def _panel_q_values(ax, V, gw, title):
    """
    Panel C: each cell is subdivided into 4 triangular quadrants,
    one per action, coloured by Q(s,a).  Gives an at-a-glance view
    of how Q-values vary within a state.
    """
    _style_ax(ax)
    ax.set_title(title, color=TEXT, fontsize=9, pad=6)
    ax.set_xlim(-0.5, gw.size - 0.5)
    ax.set_ylim(gw.size - 0.5, -0.5)
    ax.set_aspect("equal")

    # Compute all Q(s,a)
    Q_all = np.zeros((gw.n_states, gw.n_actions))
    for s in range(gw.n_states):
        for a in range(gw.n_actions):
            Q_all[s, a] = sum(
                p * (rwd + (0. if dn else gw.gamma * V[sp]))
                for p, sp, rwd, dn in gw.P[s][a]
            )

    norm  = Normalize(vmin=Q_all.min() - 1e-9, vmax=Q_all.max() + 1e-9)
    cmap  = plt.get_cmap("RdYlGn")

    # Triangle vertices for each action (up, down, left, right)
    # relative to cell centre (0,0), cell half-width = 0.48
    hw = 0.46
    tri_verts = {
        0: [( 0,  0), (-hw, -hw), ( hw, -hw)],   # up    → top triangle
        1: [( 0,  0), (-hw,  hw), ( hw,  hw)],   # down  → bottom triangle
        2: [( 0,  0), (-hw, -hw), (-hw,  hw)],   # left  → left triangle
        3: [( 0,  0), ( hw, -hw), ( hw,  hw)],   # right → right triangle
    }
    # label offsets (col_delta, row_delta)
    label_pos = {0: (0, -0.26), 1: (0, 0.26), 2: (-0.26, 0), 3: (0.26, 0)}

    for s in range(gw.n_states):
        row, col = gw.to_rc(s)
        is_term  = gw.is_terminal(s)

        for a in range(gw.n_actions):
            q   = Q_all[s, a]
            nv  = float(norm(q))
            clr = (0.12, 0.12, 0.18, 1.0) if is_term else cmap(nv)

            verts = [(col + dc, row + dr) for dc, dr in tri_verts[a]]
            tri   = plt.Polygon(verts, facecolor=clr,
                                edgecolor=PANEL2, linewidth=0.4)
            ax.add_patch(tri)

            if not is_term:
                ldc, ldr = label_pos[a]
                txt_col  = _cell_text_colour(nv)
                ax.text(col + ldc, row + ldr,
                        f"{q:.0f}",
                        ha="center", va="center",
                        fontsize=5.5, color=txt_col)

        # terminal star
        if is_term:
            ax.text(col, row, "★", ha="center", va="center",
                    fontsize=14, color=AMBER)
        else:
            # thin cross dividers between triangles
            ax.plot([col - hw, col + hw], [row, row],
                    color=PANEL2, lw=0.4)
            ax.plot([col, col], [row - hw, row + hw],
                    color=PANEL2, lw=0.4)

    ax.set_xticks(range(gw.size))
    ax.set_yticks(range(gw.size))
    ax.set_xticklabels(range(gw.size), color=TEXT_DIM, fontsize=7)
    ax.set_yticklabels(range(gw.size), color=TEXT_DIM, fontsize=7)
    _add_colorbar(ax.figure, ax, norm, label="Q(s,a)")

    # action legend
    legend_els = [mpatches.Patch(color=MUTED, label=f"{ACTION_NAMES[a]} = action {a}")
                  for a in range(gw.n_actions)]
    ax.legend(handles=legend_els, fontsize=5.5, facecolor=PANEL2,
              labelcolor=TEXT, framealpha=0.8, loc="upper right",
              ncol=2, columnspacing=0.5, handlelength=0.8)


def _panel_gradient_flow(ax, V, gw, title):
    """
    Panel D: value gradient as a flow field.
    Arrows show the direction of steepest value ascent ∇V(s),
    thickness encodes magnitude |∇V|.
    """
    _style_ax(ax)
    ax.set_title(title, color=TEXT, fontsize=9, pad=6)

    norm = _make_norm(V)
    _draw_grid_cells(ax, V, gw, norm, show_values=True,
                     fontsize=8, alpha=0.45)

    s = gw.size
    grid_V = V.reshape(s, s).astype(float)

    # Numerical gradient (finite differences, central where possible)
    # Note: row axis is "down" so ∂V/∂row points toward lower-value rows
    grad_row, grad_col = np.gradient(grid_V)

    for r in range(s):
        for c in range(s):
            state = gw.to_state(r, c)
            if gw.is_terminal(state):
                continue
            dr = -grad_row[r, c]   # flip: imshow row↓ = value-space row↑
            dc =  grad_col[r, c]
            mag = np.sqrt(dr**2 + dc**2)
            if mag < 0.01:
                continue
            dr_n, dc_n = dr / mag, dc / mag
            scale = min(0.38, 0.08 + 0.06 * mag)
            lw    = 0.6 + 1.4 * min(1.0, mag / 5.0)

            ax.annotate(
                "",
                xy=(c + dc_n * scale, r + dr_n * scale),
                xytext=(c - dc_n * scale * 0.3, r - dr_n * scale * 0.3),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ACCENT, alpha=0.82,
                    lw=lw, mutation_scale=8,
                ),
            )

    _add_colorbar(ax.figure, ax, norm)


def _panel_state_index(ax, gw, title):
    """
    Panel E: reference grid showing state indices and (row, col) coordinates.
    Useful for debugging transitions.
    """
    _style_ax(ax)
    ax.set_title(title, color=TEXT, fontsize=9, pad=6)
    ax.set_xlim(-0.5, gw.size - 0.5)
    ax.set_ylim(gw.size - 0.5, -0.5)
    ax.set_aspect("equal")

    for s in range(gw.n_states):
        r, c    = gw.to_rc(s)
        is_term = gw.is_terminal(s)
        face    = "#2a2240" if is_term else PANEL
        edge    = AMBER    if is_term else MUTED
        lw      = 2.0      if is_term else 0.5

        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.48, r - 0.48), 0.96, 0.96,
            boxstyle="square,pad=0",
            facecolor=face, edgecolor=edge, linewidth=lw,
        ))
        ax.text(c, r - 0.15, f"s={s}", ha="center", va="center",
                fontsize=8, fontweight="bold",
                color=AMBER if is_term else TEXT)
        ax.text(c, r + 0.18, f"({r},{c})", ha="center", va="center",
                fontsize=6, color=TEXT_DIM)

    ax.set_xticks(range(gw.size))
    ax.set_yticks(range(gw.size))
    ax.set_xticklabels([f"col {i}" for i in range(gw.size)],
                       color=TEXT_DIM, fontsize=7)
    ax.set_yticklabels([f"row {i}" for i in range(gw.size)],
                       color=TEXT_DIM, fontsize=7)
    legend_els = [
        mpatches.Patch(facecolor="#2a2240", edgecolor=AMBER, lw=2, label="terminal"),
        mpatches.Patch(facecolor=PANEL,     edgecolor=MUTED, lw=0.5, label="non-terminal"),
    ]
    ax.legend(handles=legend_els, fontsize=7, facecolor=PANEL2,
              labelcolor=TEXT, framealpha=0.8, loc="lower right")


def _panel_value_bar(ax, V, gw, title):
    """
    Panel F: horizontal bar chart of V(s), sorted by value.
    Colour matches the heatmap scale; terminals annotated.
    """
    _style_ax(ax, grid=True)
    ax.set_title(title, color=TEXT, fontsize=9, pad=6)

    order  = np.argsort(V)[::-1]         # highest value first
    ys     = np.arange(gw.n_states)
    norm   = _make_norm(V)
    cmap   = plt.get_cmap("RdYlGn")

    bars = ax.barh(
        ys,
        V[order],
        color=[cmap(norm(V[s])) for s in order],
        edgecolor=PANEL2, linewidth=0.3, height=0.72,
    )

    for i, s in enumerate(order):
        label_x = V[s] - 0.4 if V[s] < 0 else 0.1
        ha      = "right"    if V[s] < 0 else "left"
        marker  = " ★" if gw.is_terminal(s) else ""
        ax.text(label_x, i, f"s{s}{marker}",
                ha=ha, va="center", fontsize=6.5, color=TEXT_DIM)

    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks([])
    ax.set_xlabel("V(s)", color=TEXT, fontsize=8)
    ax.xaxis.label.set_color(TEXT)

    legend_els = [mpatches.Patch(color=AMBER, label="terminal (★)")]
    ax.legend(handles=legend_els, fontsize=7, facecolor=PANEL2,
              labelcolor=TEXT, framealpha=0.8)


def build_heatmap_figure(
    gw: GridWorld,
    V: np.ndarray,
    save_path: str = "value_heatmap.png",
) -> None:
    """
    6-panel figure:
      A — plain value heatmap with numbers
      B — heatmap + greedy policy arrows
      C — per-cell Q-value triangles (one triangle per action)
      D — value gradient flow field
      E — state index reference grid
      F — sorted horizontal bar chart of V(s)
    """
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle(
        "Week 02 — V(s) Heatmap Deep-Dive  "
        f"(GridWorld {gw.size}×{gw.size}, γ={gw.gamma})",
        color=TEXT, fontsize=12, fontweight="bold", y=0.995,
    )

    gs = mgridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.44, wspace=0.38,
        left=0.05, right=0.97, top=0.95, bottom=0.05,
    )

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    labels = ["A", "B", "C", "D", "E", "F"]
    for ax, lbl in zip(axes, labels):
        ax.text(-0.08, 1.04, lbl, transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=ACCENT, va="top")

    _panel_plain_heatmap (axes[0], V, gw, "A — V(s) values")
    _panel_heatmap_arrows(axes[1], V, gw, "B — V(s) + greedy policy π*(s)")
    _panel_q_values      (axes[2], V, gw, "C — Q(s,a) per action (triangles)")
    _panel_gradient_flow (axes[3], V, gw, "D — value gradient ∇V(s)")
    _panel_state_index   (axes[4], gw,    "E — state index reference")
    _panel_value_bar     (axes[5], V, gw, "F — V(s) ranked")

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — sweep-by-sweep evolution of V(s)
# ─────────────────────────────────────────────────────────────────────────────

def build_sweep_figure(
    gw: GridWorld,
    save_path: str = "sweep_evolution.png",
    snapshots: tuple[int, ...] = (1, 2, 5, 15, 40, 167),
) -> None:
    """
    Run policy evaluation internally and snapshot V(s) at chosen sweep numbers.
    Shows how the value function propagates outward from the terminals.
    """
    pi = gw.uniform_random_policy()

    # Re-run evaluation with per-sweep snapshots
    V       = np.zeros(gw.n_states, dtype=float)
    frames  = {}
    theta   = 1e-6
    max_sw  = max(snapshots) + 10

    for sweep in range(1, max_sw + 1):
        delta = 0.0
        for s in range(gw.n_states):
            v_old = V[s]
            v_new = 0.0
            for a in range(gw.n_actions):
                pi_sa = pi[s, a]
                for prob, sp, r, done in gw.P[s][a]:
                    v_new += pi_sa * prob * (r + (0. if done else gw.gamma * V[sp]))
            V[s]  = v_new
            delta = max(delta, abs(v_old - v_new))

        if sweep in snapshots:
            frames[sweep] = V.copy()

        if delta < theta and sweep >= max(snapshots):
            break

    n_frames = len(snapshots)
    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 2.8, 3.4),
                             facecolor=BG, constrained_layout=True)
    fig.suptitle(
        "Sweep-by-sweep evolution of V(s)  — uniform random policy, γ=1",
        color=TEXT, fontsize=11, fontweight="bold", y=1.01,
    )

    # Common norm across all frames so colours are comparable
    all_vals = np.concatenate(list(frames.values()))
    norm     = Normalize(vmin=all_vals.min() - 1e-9, vmax=all_vals.max() + 1e-9)

    for ax, sw in zip(axes, snapshots):
        _style_ax(ax)
        Vsw = frames.get(sw, V)
        _draw_grid_cells(ax, Vsw, gw, norm, show_values=True,
                         fontsize=8, alpha=1.0)
        ax.set_title(f"Sweep {sw}", color=TEXT, fontsize=9, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colourbar on the right
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="RdYlGn"),
                      ax=axes.tolist(), fraction=0.02, pad=0.02, shrink=0.85)
    cb.ax.tick_params(colors=TEXT, labelsize=7)
    cb.set_label("V(s)", color=TEXT_DIM, fontsize=7)

    _save(fig, save_path)
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_figure(
    gw: GridWorld,
    save_path: str = "policy_comparison.png",
) -> None:
    """
    Evaluate four different policies and plot V(s) side-by-side:
      1. Uniform random
      2. Always-right
      3. Always-down
      4. Greedy (one step of improvement from uniform V)
    Demonstrates how much policy choice matters even before full optimisation.
    """
    uniform_pi = gw.uniform_random_policy()
    V_uniform, _ = gw.iterative_policy_evaluation(uniform_pi)

    # Always-right policy
    right_actions = np.full(gw.n_states, 3, dtype=int)   # action 3 = right
    right_pi = gw.deterministic_policy(right_actions)
    V_right, _ = gw.iterative_policy_evaluation(right_pi)

    # Always-down policy
    down_actions = np.full(gw.n_states, 1, dtype=int)    # action 1 = down
    down_pi = gw.deterministic_policy(down_actions)
    V_down, _ = gw.iterative_policy_evaluation(down_pi)

    # Greedy improvement from V_uniform
    greedy_pi = gw.greedy_policy_from_V(V_uniform)
    V_greedy, _ = gw.iterative_policy_evaluation(greedy_pi)

    policies = [
        (V_uniform, "Uniform random",    "π(a|s) = ¼ for all a"),
        (V_right,   "Always right →",    "π(→|s) = 1 for all s"),
        (V_down,    "Always down ↓",     "π(↓|s) = 1 for all s"),
        (V_greedy,  "Greedy (1 step)",   "argmax_a Q(s,a) from V_uniform"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), facecolor=BG,
                             constrained_layout=True)
    fig.suptitle(
        "V(s) under four different policies  — GridWorld 4×4, γ=1",
        color=TEXT, fontsize=11, fontweight="bold", y=1.01,
    )

    # Common norm so all four heatmaps share the same colour scale
    all_V = np.concatenate([p[0] for p in policies])
    norm  = Normalize(vmin=all_V.min() - 1e-9, vmax=all_V.max() + 1e-9)

    for ax, (V_p, title, subtitle) in zip(axes, policies):
        _style_ax(ax)
        _draw_grid_cells(ax, V_p, gw, norm, show_values=True,
                         fontsize=9, alpha=1.0)
        _add_policy_arrows(ax, V_p, gw, scale=0.30,
                           colour="#ffffff", alpha=0.55)
        mean_v = V_p[~np.array([gw.is_terminal(s) for s in range(gw.n_states)])].mean()
        ax.set_title(f"{title}\n{subtitle}", color=TEXT,
                     fontsize=8, pad=5, linespacing=1.5)
        ax.text(0.5, -0.08, f"mean V (non-terminal): {mean_v:.1f}",
                transform=ax.transAxes, ha="center",
                fontsize=7, color=TEXT_DIM)
        ax.set_xticks([])
        ax.set_yticks([])

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="RdYlGn"),
                      ax=axes.tolist(), fraction=0.015, pad=0.02, shrink=0.85)
    cb.ax.tick_params(colors=TEXT, labelsize=7)
    cb.set_label("V(s)", color=TEXT_DIM, fontsize=7)

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Week 02 — V(s) Heatmap Visualisation")
    print("=" * 60)

    gw = GridWorld(size=4, gamma=1.0)
    pi = gw.uniform_random_policy()

    print("\nRunning policy evaluation (uniform random, γ=1) …")
    V, deltas = gw.iterative_policy_evaluation(pi, theta=1e-6, track_delta=True)
    print(f"  Converged in {len(deltas)} sweeps")
    print(f"  Max |error vs S&B|: {np.max(np.abs(V - SB_GROUND_TRUTH)):.2e}")

    print("\nGenerating figures …")
    build_heatmap_figure  (gw, V, save_path="value_heatmap.png")
    build_sweep_figure    (gw,    save_path="sweep_evolution.png")
    build_comparison_figure(gw,   save_path="policy_comparison.png")


if __name__ == "__main__":
    main()