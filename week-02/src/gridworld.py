"""
week-02/src/gridworld.py

GridWorld MDP — Sutton & Barto Example 4.1
Implements the full transition model P[s][a], iterative policy evaluation,
and the policy improvement step (greedy one-step lookahead).

Grid layout (4×4, row-major):
     0   1   2   3
     4   5   6   7
     8   9  10  11
    12  13  14  15

Terminal states : 0 (top-left) and 15 (bottom-right)
Actions         : 0=up, 1=down, 2=left, 3=right
Reward          : -1 on every non-terminal transition, 0 from terminals
Discount γ      : 1.0  (undiscounted episodic task, per S&B §4.1)

Public API
──────────
  GridWorld.iterative_policy_evaluation(policy)  → V, deltas
  GridWorld.policy_improvement(V, old_policy)    → new_policy, stable, Q
  GridWorld.q_values(s, V)                       → Q[n_actions]
  GridWorld.q_table(V)                           → Q[n_states, n_actions]

S&B ground-truth V under uniform random policy (γ=1):
     0.0  -14.0  -20.0  -22.0
   -14.0  -18.0  -20.0  -20.0
   -20.0  -20.0  -18.0  -14.0
   -22.0  -20.0  -14.0    0.0

Usage:
    python week-02/src/gridworld.py

Commits:
    feat(w02): GridWorld MDP class with transition model
    feat(w02): iterative policy evaluation on GridWorld
    feat(w02): add policy_improvement() — greedy step
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ACTION_NAMES  = {0: "↑", 1: "↓", 2: "←", 3: "→"}
ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # (Δrow, Δcol)

# S&B ground-truth values for uniform random policy on 4×4 grid (γ=1)
SB_GROUND_TRUTH = np.array([
     0.,  -14.,  -20.,  -22.,
   -14.,  -18.,  -20.,  -20.,
   -20.,  -20.,  -18.,  -14.,
   -22.,  -20.,  -14.,    0.,
])


# ─────────────────────────────────────────────────────────────────────────────
# GridWorld MDP
# ─────────────────────────────────────────────────────────────────────────────

class GridWorld:
    """
    Finite, tabular MDP representing the 4×4 GridWorld from S&B Example 4.1.

    The core data structure is the transition model:

        P[s][a] = [(prob, next_state, reward, done), ...]

    For this deterministic environment every list has exactly one entry
    with prob=1.0, but the structure supports stochastic variants without
    any change to the planning algorithms that consume it.

    Parameters
    ----------
    size  : int   Grid is size×size.  Default 4.
    gamma : float Discount factor.    Default 1.0 (undiscounted).
    """

    def __init__(self, size: int = 4, gamma: float = 1.0) -> None:
        self.size            = size
        self.n_states        = size * size
        self.n_actions       = 4
        self.gamma           = gamma
        self.terminal_states = {0, self.n_states - 1}
        self._build_transitions()

    # ── Geometry helpers ─────────────────────────────────────────────────────

    def to_rc(self, state: int) -> tuple[int, int]:
        """Flat index → (row, col)."""
        return divmod(state, self.size)

    def to_state(self, row: int, col: int) -> int:
        """(row, col) → flat index."""
        return row * self.size + col

    def is_terminal(self, state: int) -> bool:
        return state in self.terminal_states

    # ── Transition model ─────────────────────────────────────────────────────

    def _build_transitions(self) -> None:
        """
        Populate self.P[s][a] = [(prob, s', r, done)].

        Rules:
          • Terminal states are absorbing: any action → same state, r=0, done=True.
          • Moving into a wall: agent stays in place (wall-bounce).
          • All other transitions: move one cell, r=-1, done=(s' is terminal).
        """
        s = self.size
        self.P: dict[int, dict[int, list]] = {}

        for state in range(self.n_states):
            self.P[state] = {}
            row, col = self.to_rc(state)

            for action in range(self.n_actions):
                # ── absorbing terminal ──────────────────────────────────────
                if self.is_terminal(state):
                    self.P[state][action] = [(1.0, state, 0.0, True)]
                    continue

                # ── compute candidate next cell ─────────────────────────────
                dr, dc  = ACTION_DELTAS[action]
                nr      = max(0, min(s - 1, row + dr))   # clamp → wall-bounce
                nc      = max(0, min(s - 1, col + dc))
                s_prime = self.to_state(nr, nc)
                done    = self.is_terminal(s_prime)

                self.P[state][action] = [(1.0, s_prime, -1.0, done)]

    # ── Policy evaluation ────────────────────────────────────────────────────

    def iterative_policy_evaluation(
        self,
        policy: np.ndarray,
        theta: float = 1e-6,
        max_iter: int = 10_000,
        track_delta: bool = False,
    ) -> tuple[np.ndarray, list[float]]:
        """
        Evaluate a policy by iterating the Bellman expectation equation until
        convergence.

        Bellman expectation backup (one sweep over all states):

            V(s) ← Σ_a π(a|s) · Σ_{s',r} P(s',r|s,a) · [r + γ·V(s')]

        This is the in-place (Gauss-Seidel) variant: V[s] is updated
        immediately within the sweep, so later states in the same pass
        see the freshly updated values of earlier states.  This converges
        faster than the synchronous (Jacobi) variant while reaching the
        same fixed point.

        Parameters
        ----------
        policy      : np.ndarray  shape (n_states, n_actions)
                      policy[s, a] = π(a|s), rows must sum to 1.
        theta       : float  Convergence threshold on max|ΔV|.
        max_iter    : int    Safety cap on sweep count.
        track_delta : bool   If True, record max-delta per sweep for plotting.

        Returns
        -------
        V      : np.ndarray  shape (n_states,)  converged state-value estimates
        deltas : list[float] max|ΔV| per sweep  (empty if track_delta=False)
        """
        V      = np.zeros(self.n_states, dtype=float)
        deltas = []

        for iteration in range(max_iter):
            delta = 0.0

            for s in range(self.n_states):
                v_old = V[s]      # save before update (for delta tracking)
                v_new = 0.0

                # Bellman expectation backup
                for a in range(self.n_actions):
                    pi_sa = policy[s, a]
                    if pi_sa == 0.0:
                        continue                  # skip zero-probability actions

                    for prob, s_prime, reward, done in self.P[s][a]:
                        bootstrap = 0.0 if done else self.gamma * V[s_prime]
                        v_new    += pi_sa * prob * (reward + bootstrap)

                V[s]  = v_new
                delta = max(delta, abs(v_old - v_new))

            if track_delta:
                deltas.append(delta)

            if delta < theta:
                break

        return V, deltas

    # ── Convenience: build common policies ──────────────────────────────────

    def uniform_random_policy(self) -> np.ndarray:
        """π(a|s) = 1/|A| for all s, a."""
        return np.ones((self.n_states, self.n_actions)) / self.n_actions

    def deterministic_policy(self, action_array: np.ndarray) -> np.ndarray:
        """
        Convert a (n_states,) array of chosen actions into a one-hot policy matrix.
        """
        policy = np.zeros((self.n_states, self.n_actions))
        for s, a in enumerate(action_array):
            policy[s, a] = 1.0
        return policy

    # ── Q-value computation ──────────────────────────────────────────────────

    def q_values(self, s: int, V: np.ndarray) -> np.ndarray:
        """
        Compute Q(s, a) for every action from state s under value function V.

        One-step Bellman lookahead:
            Q(s, a) = Σ_{s',r} P(s',r | s,a) · [r + γ · V(s')]

        Terminal states always return all-zero Q-values (no future value).

        Parameters
        ----------
        s : int          State index.
        V : np.ndarray   Current value function, shape (n_states,).

        Returns
        -------
        Q : np.ndarray   shape (n_actions,)
        """
        if self.is_terminal(s):
            return np.zeros(self.n_actions)

        Q = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            for prob, s_prime, reward, done in self.P[s][a]:
                bootstrap = 0.0 if done else self.gamma * V[s_prime]
                Q[a]     += prob * (reward + bootstrap)
        return Q

    def q_table(self, V: np.ndarray) -> np.ndarray:
        """
        Build the full Q-table for every (state, action) pair.

        Returns
        -------
        Q : np.ndarray  shape (n_states, n_actions)
        """
        return np.stack([self.q_values(s, V) for s in range(self.n_states)])

    # ── Policy improvement ───────────────────────────────────────────────────

    def policy_improvement(
        self,
        V: np.ndarray,
        old_policy: np.ndarray | None = None,
        tie_breaking: str = "first",
    ) -> tuple[np.ndarray, bool, np.ndarray]:
        """
        Greedy policy improvement — S&B §4.2.

        For each non-terminal state s, compute Q(s,a) for all actions and
        assign probability 1.0 to whichever action(s) maximise Q.

        The Policy Improvement Theorem guarantees that the new policy π' is
        at least as good as the old policy π:

            v_{π'}(s) ≥ v_π(s)   for all s ∈ S

        with equality everywhere iff π was already optimal.

        Parameters
        ----------
        V            : np.ndarray  Current value function, shape (n_states,).
        old_policy   : np.ndarray | None
                       Previous policy, shape (n_states, n_actions).
                       If supplied, policy_stable is computed.
                       If None, policy_stable is always returned as False.
        tie_breaking : str  How to break ties among equally-good actions.
                       "first"    — pick the lowest-index action (deterministic).
                       "uniform"  — distribute probability equally (stochastic).
                       "random"   — pick one tied action at random each call.

        Returns
        -------
        new_policy    : np.ndarray  shape (n_states, n_actions)
                        Improved (or confirmed-optimal) policy.
        policy_stable : bool
                        True iff the greedy action is unchanged in every state.
                        When True, the current V is the optimal value function
                        and the current policy is optimal — iteration can stop.
        Q             : np.ndarray  shape (n_states, n_actions)
                        Full Q-table computed during improvement (free to return).
        """
        assert tie_breaking in ("first", "uniform", "random"), \
            f"Unknown tie_breaking: {tie_breaking!r}"

        Q          = self.q_table(V)
        new_policy = np.zeros((self.n_states, self.n_actions))
        policy_stable = True

        for s in range(self.n_states):
            if self.is_terminal(s):
                # Terminal states are absorbing; policy is irrelevant but
                # must be a valid probability distribution.
                new_policy[s, 0] = 1.0
                continue

            q_s    = Q[s]
            best_q = q_s.max()
            # All actions whose Q-value is within numerical tolerance of best
            tol     = max(1e-9, 1e-6 * abs(best_q))
            best_as = np.where(q_s >= best_q - tol)[0]

            # ── fill new_policy[s] based on tie_breaking rule ────────────────
            if tie_breaking == "first":
                new_policy[s, best_as[0]] = 1.0

            elif tie_breaking == "uniform":
                new_policy[s, best_as] = 1.0 / len(best_as)

            elif tie_breaking == "random":
                chosen = np.random.choice(best_as)
                new_policy[s, chosen] = 1.0

            # ── check policy stability ───────────────────────────────────────
            if old_policy is not None:
                old_best = set(np.where(old_policy[s] > 0)[0])
                new_best = set(best_as.tolist())
                if old_best != new_best:
                    policy_stable = False
            else:
                policy_stable = False   # no reference → can't confirm stable

        return new_policy, policy_stable, Q

    # ── backward-compat alias ────────────────────────────────────────────────

    def greedy_policy_from_V(self, V: np.ndarray) -> np.ndarray:
        """Alias for policy_improvement(); returns only the new policy."""
        policy, _, _ = self.policy_improvement(V)
        return policy

    # ── Validation ───────────────────────────────────────────────────────────

    def validate_against_sb(self, V: np.ndarray, tol: float = 0.5) -> bool:
        """
        Check converged V (uniform policy, γ=1) against S&B Example 4.1 table.
        Returns True if max absolute error < tol.
        """
        max_err = float(np.max(np.abs(V - SB_GROUND_TRUTH)))
        return max_err < tol

    # ── Dunder ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"GridWorld(size={self.size}, n_states={self.n_states}, "
                f"gamma={self.gamma})")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

BG      = "#0f0f14"
PANEL   = "#1a1a26"
MUTED   = "#555568"
TEXT    = "#c8c8dc"
TEXT_DIM= "#888899"
ACCENT  = "#7C6AF7"
GREEN   = "#5ED4A0"
RED     = "#F76A6A"
AMBER   = "#F7C26A"


def _ax_style(ax) -> None:
    """Apply consistent dark styling to an axes."""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)


def plot_value_heatmap(
    ax: plt.Axes,
    V: np.ndarray,
    gw: GridWorld,
    title: str = "State values V(s)",
    cmap: str = "RdYlGn",
    annotate: bool = True,
) -> None:
    """
    Draw the value function as a coloured grid with numeric annotations.
    Terminal cells are outlined in gold.
    """
    grid = V.reshape(gw.size, gw.size)
    vmin, vmax = grid.min(), grid.max()
    norm = Normalize(vmin=vmin - 1e-9, vmax=vmax + 1e-9)

    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="equal")

    if annotate:
        for s in range(gw.n_states):
            r, c = gw.to_rc(s)
            val   = grid[r, c]
            # pick text colour for legibility over the heatmap cell
            brightness = norm(val)
            txt_col    = "#0f0f14" if 0.35 < brightness < 0.75 else TEXT
            ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=txt_col)

    # outline terminal cells
    for s in gw.terminal_states:
        r, c = gw.to_rc(s)
        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.48, r - 0.48), 0.96, 0.96,
            boxstyle="square,pad=0", linewidth=2.0,
            edgecolor=AMBER, facecolor="none",
        ))

    ax.set_xticks(range(gw.size))
    ax.set_yticks(range(gw.size))
    ax.set_xticklabels(range(gw.size), color=TEXT_DIM, fontsize=7)
    ax.set_yticklabels(range(gw.size), color=TEXT_DIM, fontsize=7)
    ax.set_title(title, color=TEXT, fontsize=9, pad=7)

    plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 fraction=0.046, pad=0.03).ax.tick_params(
                     colors=TEXT, labelsize=7)


def plot_policy_arrows(
    ax: plt.Axes,
    V: np.ndarray,
    gw: GridWorld,
    title: str = "Greedy policy π*(s)",
) -> None:
    """
    Overlay the greedy policy as Unicode arrows on a light value heatmap.
    Cells with multiple equally-good actions show all tied arrows.
    """
    grid = V.reshape(gw.size, gw.size)
    norm = Normalize(vmin=grid.min() - 1e-9, vmax=grid.max() + 1e-9)
    ax.imshow(grid, cmap="RdYlGn", norm=norm, aspect="equal", alpha=0.55)

    for s in range(gw.n_states):
        r, c = gw.to_rc(s)

        if gw.is_terminal(s):
            ax.text(c, r, "T", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=AMBER)
            continue

        # collect Q(s,a) for all actions
        q_vals = []
        for a in range(gw.n_actions):
            q = sum(
                prob * (rwd + (0.0 if done else gw.gamma * V[s_prime]))
                for prob, s_prime, rwd, done in gw.P[s][a]
            )
            q_vals.append(q)

        q_vals  = np.array(q_vals)
        best_q  = q_vals.max()
        # show all actions within 1e-4 of optimal (handles ties)
        best_as = [a for a in range(gw.n_actions)
                   if abs(q_vals[a] - best_q) < 1e-4]
        arrows  = " ".join(ACTION_NAMES[a] for a in best_as)

        ax.text(c, r, arrows, ha="center", va="center",
                fontsize=13, color=TEXT)

    # terminal outlines
    for s in gw.terminal_states:
        r, c = gw.to_rc(s)
        ax.add_patch(mpatches.FancyBboxPatch(
            (c - 0.48, r - 0.48), 0.96, 0.96,
            boxstyle="square,pad=0", linewidth=2.0,
            edgecolor=AMBER, facecolor="none",
        ))

    ax.set_xticks(range(gw.size))
    ax.set_yticks(range(gw.size))
    ax.set_xticklabels(range(gw.size), color=TEXT_DIM, fontsize=7)
    ax.set_yticklabels(range(gw.size), color=TEXT_DIM, fontsize=7)
    ax.set_title(title, color=TEXT, fontsize=9, pad=7)


def plot_convergence(
    ax: plt.Axes,
    deltas: list[float],
    theta: float = 1e-6,
) -> None:
    """
    Plot max|ΔV| per sweep on a log scale, with a convergence threshold line.
    """
    sweeps = np.arange(1, len(deltas) + 1)
    ax.semilogy(sweeps, deltas, color=ACCENT, lw=1.8, label="max|ΔV|")
    ax.axhline(theta, color=RED, lw=1.0, ls="--",
               label=f"θ = {theta:.0e}")

    # mark convergence point
    conv_idx = next((i for i, d in enumerate(deltas) if d < theta), len(deltas) - 1)
    ax.axvline(conv_idx + 1, color=GREEN, lw=0.9, ls=":",
               label=f"converged @ sweep {conv_idx + 1}")

    ax.set_xlabel("Sweep", color=TEXT, fontsize=8)
    ax.set_ylabel("max |ΔV|", color=TEXT, fontsize=8)
    ax.set_title("Convergence of policy evaluation", color=TEXT, fontsize=9, pad=7)
    ax.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TEXT, framealpha=0.7)
    ax.grid(color="#2a2a3a", lw=0.4, ls="--", alpha=0.7)


def plot_error_vs_truth(
    ax: plt.Axes,
    V: np.ndarray,
    gw: GridWorld,
) -> None:
    """
    Bar chart: per-state absolute error |V(s) − V_SB(s)|.
    Only meaningful for the 4×4 uniform-policy evaluation.
    """
    errors = np.abs(V - SB_GROUND_TRUTH)
    states = np.arange(gw.n_states)
    colours = [AMBER if gw.is_terminal(s) else ACCENT for s in states]

    ax.bar(states, errors, color=colours, edgecolor=PANEL, linewidth=0.4)
    ax.set_xlabel("State", color=TEXT, fontsize=8)
    ax.set_ylabel("|V(s) − V_SB(s)|", color=TEXT, fontsize=8)
    ax.set_title("Per-state error vs. S&B ground truth", color=TEXT,
                 fontsize=9, pad=7)
    ax.set_xticks(states)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(-0.6, gw.n_states - 0.4)
    legend_els = [
        mpatches.Patch(color=ACCENT, label="non-terminal"),
        mpatches.Patch(color=AMBER,  label="terminal"),
    ]
    ax.legend(handles=legend_els, fontsize=7, facecolor=PANEL,
              labelcolor=TEXT, framealpha=0.7)
    ax.grid(color="#2a2a3a", lw=0.4, ls="--", axis="y", alpha=0.7)


def build_figure(
    gw: GridWorld,
    V: np.ndarray,
    deltas: list[float],
    save_path: str = "week-02/policy_evaluation.png",
) -> None:
    """
    Five-panel figure:
      [0] Value heatmap  — V under uniform random policy
      [1] Policy arrows  — greedy π derived from V
      [2] Convergence    — max|ΔV| per sweep (log scale)
      [3] Error bars     — |V(s) − V_SB(s)| per state
      [4] Transition map — rendered P table for a chosen state
    """
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.suptitle(
        "Week 02 — GridWorld MDP: Iterative Policy Evaluation  "
        f"(γ={gw.gamma}, θ=1e-6, {len(deltas)} sweeps to converge)",
        color=TEXT, fontsize=12, fontweight="bold", y=0.99,
    )

    # Layout: 2 rows × 3 cols; heatmap + policy share the left column pair
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.48, wspace=0.38,
                  left=0.05, right=0.97, top=0.93, bottom=0.07)

    ax_heat  = fig.add_subplot(gs[0, 0])
    ax_pol   = fig.add_subplot(gs[1, 0])
    ax_conv  = fig.add_subplot(gs[0, 1])
    ax_err   = fig.add_subplot(gs[1, 1])
    ax_trans = fig.add_subplot(gs[:, 2])   # tall right panel

    for ax in [ax_heat, ax_pol, ax_conv, ax_err, ax_trans]:
        _ax_style(ax)

    # Panel 0: value heatmap
    plot_value_heatmap(ax_heat, V, gw,
                       title="V(s) — uniform random policy")

    # Panel 1: greedy policy arrows
    plot_policy_arrows(ax_pol, V, gw,
                       title="Greedy π*(s) derived from V")

    # Panel 2: convergence
    plot_convergence(ax_conv, deltas)

    # Panel 3: per-state error
    plot_error_vs_truth(ax_err, V, gw)

    # Panel 4: transition table for state 5 (centre-ish, interesting)
    _draw_transition_table(ax_trans, gw, state=5)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Figure saved → {save_path}")
    plt.show()
    plt.close(fig)


def _draw_transition_table(ax: plt.Axes, gw: GridWorld, state: int = 5) -> None:
    """
    Render P[state][a] as a neat table, and draw the grid with arrows
    showing where each action leads from `state`.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Transition model  P[s={state}][a]",
                 color=TEXT, fontsize=9, pad=7)

    # ── mini grid at top of panel ────────────────────────────────────────────
    cell = 0.09          # cell size in axes-fraction units
    x0, y0 = 0.05, 0.55  # grid origin (bottom-left of bottom-left cell)

    for s in range(gw.n_states):
        r, c = gw.to_rc(s)
        # y increases upward in axes coords, grid row 0 is at top
        bx = x0 + c * cell
        by = y0 + (gw.size - 1 - r) * cell

        # cell background
        is_src  = s == state
        is_term = gw.is_terminal(s)
        face    = ACCENT if is_src else (AMBER + "55" if is_term else PANEL)
        edge    = AMBER  if is_term else (ACCENT if is_src else MUTED)
        lw      = 1.8    if (is_src or is_term) else 0.5

        rect = mpatches.FancyBboxPatch(
            (bx, by), cell * 0.92, cell * 0.92,
            boxstyle="square,pad=0",
            facecolor=face, edgecolor=edge, linewidth=lw,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)

        label = str(s)
        col   = "#0f0f14" if is_src else TEXT_DIM
        ax.text(bx + cell * 0.46, by + cell * 0.46, label,
                ha="center", va="center", fontsize=6,
                color=col, transform=ax.transAxes)

    # ── arrows showing next-states for each action ───────────────────────────
    row_s, col_s = gw.to_rc(state)
    arrow_props  = dict(arrowstyle="-|>", color=GREEN,
                        lw=1.2, mutation_scale=8)

    for a in range(gw.n_actions):
        _, s_prime, _, _ = gw.P[state][a][0]
        r2, c2 = gw.to_rc(s_prime)

        # centre of source cell
        sx = x0 + col_s * cell + cell * 0.46
        sy = y0 + (gw.size - 1 - row_s) * cell + cell * 0.46
        # centre of target cell
        tx = x0 + c2 * cell + cell * 0.46
        ty = y0 + (gw.size - 1 - r2) * cell + cell * 0.46

        if s_prime == state:
            # wall-bounce: draw a tiny curved annotation
            ax.annotate("", xy=(tx + 0.005, ty + 0.03),
                        xytext=(sx, sy),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=RED,
                                        lw=1.0, mutation_scale=7,
                                        connectionstyle="arc3,rad=0.6"))
        else:
            ax.annotate("", xy=(tx, ty), xytext=(sx, sy),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=arrow_props)

    # ── text table below the mini grid ───────────────────────────────────────
    headers = ["Action", "s′", "Reward", "Done", "Prob"]
    col_xs  = [0.04, 0.28, 0.46, 0.65, 0.82]
    row_h   = 0.048
    y_start = 0.48

    # header row
    for hdr, cx in zip(headers, col_xs):
        ax.text(cx, y_start, hdr, fontsize=7.5, fontweight="bold",
                color=ACCENT, transform=ax.transAxes, va="top")

    ax.plot([0.03, 0.97], [y_start - 0.012, y_start - 0.012],
            color=MUTED, lw=0.5, transform=ax.transAxes, clip_on=False)

    # data rows
    for i, a in enumerate(range(gw.n_actions)):
        prob, s_prime, reward, done = gw.P[state][a][0]
        y = y_start - row_h * (i + 1)
        row_bg = "#22223a" if i % 2 == 0 else PANEL
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.03, y - 0.01), 0.94, row_h * 0.92,
            boxstyle="square,pad=0", facecolor=row_bg, edgecolor="none",
            transform=ax.transAxes,
        ))
        vals = [f"{ACTION_NAMES[a]} ({a})", str(s_prime),
                f"{reward:+.1f}", str(done), f"{prob:.1f}"]
        for val, cx in zip(vals, col_xs):
            col = GREEN if val.startswith("+") else (RED if "-" in val else TEXT)
            ax.text(cx, y + 0.010, val, fontsize=7.5, color=col,
                    transform=ax.transAxes, va="center")

    # caption
    ax.text(0.5, 0.02,
            f"Green arrows = transitions from s={state}  |  "
            "Red curved = wall-bounce",
            ha="center", fontsize=6.5, color=TEXT_DIM,
            transform=ax.transAxes)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _print_grid(label: str, arr: np.ndarray, size: int, fmt: str = "{:7.1f}") -> None:
    print(f"\n{label}")
    print("  " + "─" * (size * 8))
    for r in range(size):
        row = arr[r * size: (r + 1) * size]
        print("  " + "  ".join(fmt.format(v) for v in row))
    print("  " + "─" * (size * 8))


def main() -> None:
    print("=" * 60)
    print("  Week 02 — GridWorld MDP")
    print("  Iterative Policy Evaluation (S&B Example 4.1)")
    print("=" * 60)

    gw = GridWorld(size=4, gamma=1.0)
    print(f"\n{gw}")
    print(f"  Terminal states : {gw.terminal_states}")
    print(f"  |S| = {gw.n_states},  |A| = {gw.n_actions}")

    # ── Inspect the transition model ─────────────────────────────────────────
    print("\n── Transition model for state 5 (row=1, col=1) ──")
    print(f"  {'Action':<10}  {'s′':>4}  {'Reward':>7}  {'Done':>5}  {'Prob':>5}")
    print(f"  {'─'*10}  {'─'*4}  {'─'*7}  {'─'*5}  {'─'*5}")
    for a in range(gw.n_actions):
        prob, s_prime, reward, done = gw.P[5][a][0]
        print(f"  {ACTION_NAMES[a]+' ('+str(a)+')':<10}  "
              f"{s_prime:>4}  {reward:>+7.1f}  {str(done):>5}  {prob:>5.1f}")

    # ── Policy evaluation: uniform random ────────────────────────────────────
    print("\n── Evaluating uniform random policy (γ=1.0) ──")
    pi_uniform = gw.uniform_random_policy()
    V, deltas  = gw.iterative_policy_evaluation(pi_uniform, theta=1e-6,
                                                  track_delta=True)

    print(f"  Converged in {len(deltas)} sweeps  "
          f"(final max|ΔV| = {deltas[-1]:.2e})")

    _print_grid("V(s) — uniform random policy:", V, gw.size)

    # ── Validate against S&B ground truth ────────────────────────────────────
    max_err = float(np.max(np.abs(V - SB_GROUND_TRUTH)))
    ok      = gw.validate_against_sb(V)
    print(f"\n  S&B ground truth check: {'✓ PASS' if ok else '✗ FAIL'}")
    print(f"  Max absolute error vs. S&B table: {max_err:.6f}")

    _print_grid("S&B ground truth:", SB_GROUND_TRUTH, gw.size)

    # ── One step of greedy improvement ───────────────────────────────────────
    print("\n── Greedy policy derived from V ──")
    greedy_pi  = gw.greedy_policy_from_V(V)
    best_a     = np.argmax(greedy_pi, axis=1)
    arrow_grid = np.array([ACTION_NAMES[a] for a in best_a]).reshape(gw.size, gw.size)
    print("  (T = terminal state)\n")
    for r in range(gw.size):
        row_str = "  "
        for c in range(gw.size):
            s = gw.to_state(r, c)
            row_str += ("T " if gw.is_terminal(s) else arrow_grid[r, c] + " ")
        print(row_str)

    # ── Compare: does the greedy policy perform better? ──────────────────────
    V_greedy, d_greedy = gw.iterative_policy_evaluation(
        greedy_pi, theta=1e-6, track_delta=True
    )
    print(f"\n── V under greedy policy (converged in {len(d_greedy)} sweeps) ──")
    _print_grid("V(s) — greedy policy:", V_greedy, gw.size)
    print(f"\n  Improvement: mean V went from {V.mean():.2f} → {V_greedy.mean():.2f}")

    # ── Build the figure ─────────────────────────────────────────────────────
    print("\nGenerating visualisation …")
    build_figure(gw, V, deltas,
                 save_path="week-02/policy_evaluation.png")

    print("\nDone.  Git commands:")
    print("  git add week-02/src/gridworld.py week-02/policy_evaluation.png")
    print("  git commit -m 'feat(w02): iterative policy evaluation on GridWorld'")


if __name__ == "__main__":
    main()