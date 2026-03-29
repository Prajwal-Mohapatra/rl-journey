"""
week-01/src/rl_loop.py

Hand-rolled RL loop with a custom 1D corridor environment.
No Gymnasium. No external RL libraries. Pure Python + NumPy.

Environment: LinearCorridor
  - A 1D grid of N cells, positions 0 .. N-1
  - Agent starts at position 0
  - Goal is position N-1 (rightmost cell)
  - Actions: 0 = move left, 1 = move right
  - Reward: +10 on reaching goal, -0.1 on every other step
  - Episode ends when agent reaches goal or max_steps is exceeded

Agent: EpsilonGreedyAgent
  - Maintains a Q-table: Q[state][action]
  - Selects actions via epsilon-greedy policy
  - Updates Q-values via the Q-learning rule:
      Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

Usage:
  python week-01/src/rl_loop.py

Commit: feat(w01): hand-rolled RL loop with custom 1D env
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional
import os


# ---------------------------------------------------------------------------
# 1. Custom Environment
# ---------------------------------------------------------------------------

class LinearCorridor:
    """
    A minimal 1D corridor MDP — no Gymnasium dependency whatsoever.

    State space : integers {0, 1, ..., n_cells-1}
    Action space: {0: left, 1: right}
    Dynamics    : deterministic by default; set slip_prob > 0 for stochastic
    Reward      : +10 at goal, -0.1 otherwise
    Termination : goal reached OR max_steps exceeded
    """

    LEFT  = 0
    RIGHT = 1
    ACTION_NAMES = {LEFT: "←", RIGHT: "→"}

    def __init__(self, n_cells: int = 10, max_steps: int = 200,
                 slip_prob: float = 0.0, goal_reward: float = 10.0,
                 step_penalty: float = -0.1):
        assert n_cells >= 2, "Need at least 2 cells."
        assert 0.0 <= slip_prob < 1.0

        self.n_cells     = n_cells
        self.max_steps   = max_steps
        self.slip_prob   = slip_prob    # probability action is reversed
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty

        # Public attributes updated by reset() / step()
        self.state       : int  = 0
        self.steps_taken : int  = 0
        self.done        : bool = False

        # Spaces (mirror Gymnasium convention for familiarity)
        self.observation_space = SimpleDiscreteSpace(n_cells)
        self.action_space      = SimpleDiscreteSpace(2)

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> int:
        """Reset to start state. Returns initial observation."""
        if seed is not None:
            np.random.seed(seed)
        self.state       = 0
        self.steps_taken = 0
        self.done        = False
        return self.state

    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Apply action, return (next_state, reward, done, info).
        Mirrors the classic Gym 4-tuple API (pre Gymnasium truncated flag).
        """
        assert not self.done, "Episode has ended. Call reset() first."
        assert action in (self.LEFT, self.RIGHT), f"Invalid action: {action}"

        # Stochastic slip: flip the action with probability slip_prob
        if self.slip_prob > 0 and np.random.rand() < self.slip_prob:
            action = 1 - action   # reverse direction

        # Apply movement, clamp to corridor
        if action == self.RIGHT:
            self.state = min(self.state + 1, self.n_cells - 1)
        else:
            self.state = max(self.state - 1, 0)

        self.steps_taken += 1

        # Reward and termination
        at_goal = self.state == self.n_cells - 1
        timeout = self.steps_taken >= self.max_steps

        reward = self.goal_reward if at_goal else self.step_penalty
        self.done = at_goal or timeout

        info = {
            "at_goal": at_goal,
            "timeout": timeout,
            "steps":   self.steps_taken,
        }
        return self.state, reward, self.done, info

    # ------------------------------------------------------------------
    def render(self) -> str:
        """Return a simple ASCII render of the current state."""
        cells = ["·"] * self.n_cells
        cells[0]               = "S"   # start
        cells[self.n_cells - 1] = "G"  # goal
        cells[self.state]       = "A"  # agent (overwrites S or G if on them)
        return " ".join(cells) + f"  (step {self.steps_taken})"

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"LinearCorridor(n_cells={self.n_cells}, "
                f"slip_prob={self.slip_prob}, max_steps={self.max_steps})")


class SimpleDiscreteSpace:
    """Tiny stand-in for gym.spaces.Discrete — no dependency needed."""
    def __init__(self, n: int):
        self.n = n

    def sample(self) -> int:
        return np.random.randint(self.n)

    def __contains__(self, x: int) -> bool:
        return 0 <= x < self.n


# ---------------------------------------------------------------------------
# 2. Agent
# ---------------------------------------------------------------------------

class EpsilonGreedyAgent:
    """
    Tabular Q-learning agent with epsilon-greedy exploration.

    Q-update rule (off-policy TD):
      Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    Epsilon schedule: linear decay from epsilon_start → epsilon_min
    over the first `epsilon_decay_episodes` episodes, then held at min.
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay_episodes: int = 400):

        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma

        self.epsilon_start          = epsilon_start
        self.epsilon_min            = epsilon_min
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon                = epsilon_start

        # Q-table: shape (n_states, n_actions), initialised to 0
        self.Q = np.zeros((n_states, n_actions))

        # Stats
        self.episode_count = 0

    # ------------------------------------------------------------------
    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)   # explore
        return int(np.argmax(self.Q[state]))            # exploit

    # ------------------------------------------------------------------
    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> float:
        """
        One Q-learning update step.
        Returns the TD error for logging.
        """
        best_next = 0.0 if done else float(np.max(self.Q[next_state]))
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        return td_error

    # ------------------------------------------------------------------
    def end_episode(self):
        """Call once per episode to decay epsilon."""
        self.episode_count += 1
        frac = min(1.0, self.episode_count / self.epsilon_decay_episodes)
        self.epsilon = (self.epsilon_start
                        + frac * (self.epsilon_min - self.epsilon_start))

    # ------------------------------------------------------------------
    def greedy_policy(self) -> np.ndarray:
        """Return the current greedy action for every state."""
        return np.argmax(self.Q, axis=1)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"EpsilonGreedyAgent(alpha={self.alpha}, gamma={self.gamma}, "
                f"epsilon={self.epsilon:.3f})")


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainingStats:
    """Lightweight container for per-episode metrics."""
    rewards:    list = field(default_factory=list)
    lengths:    list = field(default_factory=list)
    td_errors:  list = field(default_factory=list)   # mean |TD| per episode
    epsilons:   list = field(default_factory=list)
    successes:  list = field(default_factory=list)   # 1 if goal reached


def train(env: LinearCorridor,
          agent: EpsilonGreedyAgent,
          n_episodes: int = 500,
          verbose_every: int = 100) -> TrainingStats:
    """
    The core RL loop — pure Python, no library magic.

    For each episode:
      1. Reset environment → get initial state
      2. Loop until done:
           a. Agent selects action (epsilon-greedy)
           b. Environment transitions → (next_state, reward, done, info)
           c. Agent updates Q-table (TD learning)
      3. Decay epsilon
      4. Record stats
    """
    stats = TrainingStats()

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        ep_td_errors = []
        done = False

        # ---- inner loop: one episode --------------------------------
        while not done:
            action                          = agent.select_action(state)
            next_state, reward, done, info  = env.step(action)
            td_error                        = agent.update(state, action,
                                                           reward, next_state,
                                                           done)
            ep_td_errors.append(abs(td_error))
            total_reward += reward
            state = next_state
        # ---- end of episode -----------------------------------------

        agent.end_episode()

        stats.rewards.append(total_reward)
        stats.lengths.append(env.steps_taken)
        stats.td_errors.append(float(np.mean(ep_td_errors)))
        stats.epsilons.append(agent.epsilon)
        stats.successes.append(1 if info["at_goal"] else 0)

        if ep % verbose_every == 0 or ep == 1:
            win_rate = np.mean(stats.successes[-verbose_every:]) * 100
            avg_r    = np.mean(stats.rewards[-verbose_every:])
            print(f"  Episode {ep:>4d} | "
                  f"avg reward: {avg_r:+6.2f} | "
                  f"win rate: {win_rate:5.1f}% | "
                  f"ε={agent.epsilon:.3f} | "
                  f"steps: {env.steps_taken:>3d}")

    return stats


# ---------------------------------------------------------------------------
# 4. Evaluation (greedy, no exploration)
# ---------------------------------------------------------------------------

def evaluate(env: LinearCorridor, agent: EpsilonGreedyAgent,
             n_episodes: int = 20, render: bool = True) -> dict:
    """
    Run the learned greedy policy. Epsilon is forced to 0.
    Returns a dict of aggregate metrics.
    """
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0   # pure greedy

    successes, lengths, rewards = [], [], []

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False

        if render and ep <= 3:
            print(f"\n── Eval episode {ep} ──")
            print(f"  {env.render()}")

        while not done:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if render and ep <= 3:
                print(f"  {env.render()}  action={LinearCorridor.ACTION_NAMES[action]}")

        successes.append(1 if info["at_goal"] else 0)
        lengths.append(env.steps_taken)
        rewards.append(total_reward)

    agent.epsilon = saved_epsilon  # restore

    results = {
        "success_rate": float(np.mean(successes)),
        "mean_steps":   float(np.mean(lengths)),
        "mean_reward":  float(np.mean(rewards)),
    }
    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"  Success rate : {results['success_rate']*100:.1f}%")
    print(f"  Mean steps   : {results['mean_steps']:.1f}")
    print(f"  Mean reward  : {results['mean_reward']:.2f}")
    return results


# ---------------------------------------------------------------------------
# 5. Visualisation
# ---------------------------------------------------------------------------

def smooth(x, window: int = 20) -> np.ndarray:
    """Simple moving-average smoother."""
    if len(x) < window:
        return np.array(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_training(stats: TrainingStats, agent: EpsilonGreedyAgent,
                  env: LinearCorridor, save_path: str = "week-01/training_results.png"):
    """Four-panel training dashboard."""

    fig = plt.figure(figsize=(14, 9), facecolor="#0f0f14")
    fig.suptitle("Week 01 — Hand-rolled RL Loop on LinearCorridor",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.08)

    ax_reward  = fig.add_subplot(gs[0, :2])   # wide: reward curve
    ax_epsilon = fig.add_subplot(gs[0, 2])    # epsilon decay
    ax_steps   = fig.add_subplot(gs[1, 0])    # episode length
    ax_qtable  = fig.add_subplot(gs[1, 1])    # Q-table heatmap
    ax_policy  = fig.add_subplot(gs[1, 2])    # greedy policy

    ACCENT  = "#7C6AF7"
    ACCENT2 = "#F76A6A"
    GREEN   = "#5ED4A0"
    MUTED   = "#555568"
    TEXT    = "#c8c8dc"

    for ax in [ax_reward, ax_epsilon, ax_steps, ax_qtable, ax_policy]:
        ax.set_facecolor("#1a1a26")
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(MUTED)

    eps = np.arange(1, len(stats.rewards) + 1)

    # --- Panel 1: Reward curve ---
    ax_reward.plot(eps, stats.rewards, color=MUTED, lw=0.5, alpha=0.4, label="raw")
    sm = smooth(stats.rewards, 30)
    sm_eps = eps[len(eps) - len(sm):]
    ax_reward.plot(sm_eps, sm, color=ACCENT, lw=2.0, label="30-ep avg")
    ax_reward.axhline(0, color=MUTED, lw=0.5, ls="--")
    ax_reward.set_title("Episode reward", color=TEXT, fontsize=9)
    ax_reward.set_xlabel("Episode", color=TEXT, fontsize=8)
    ax_reward.set_ylabel("Total reward", color=TEXT, fontsize=8)
    ax_reward.legend(fontsize=8, facecolor="#1a1a26",
                     labelcolor=TEXT, framealpha=0.5)

    # --- Panel 2: Epsilon decay ---
    ax_epsilon.plot(eps, stats.epsilons, color=ACCENT2, lw=1.5)
    ax_epsilon.set_title("Epsilon (exploration)", color=TEXT, fontsize=9)
    ax_epsilon.set_xlabel("Episode", color=TEXT, fontsize=8)
    ax_epsilon.set_ylabel("ε", color=TEXT, fontsize=8)
    ax_epsilon.set_ylim(-0.05, 1.05)

    # --- Panel 3: Episode length ---
    ax_steps.plot(eps, stats.lengths, color=MUTED, lw=0.5, alpha=0.4)
    sm_len = smooth(stats.lengths, 30)
    sm_eps2 = eps[len(eps) - len(sm_len):]
    ax_steps.plot(sm_eps2, sm_len, color=GREEN, lw=2.0)
    optimal = env.n_cells - 1   # shortest path: walk straight right
    ax_steps.axhline(optimal, color=TEXT, lw=0.8, ls="--", alpha=0.6,
                     label=f"optimal ({optimal} steps)")
    ax_steps.set_title("Steps per episode", color=TEXT, fontsize=9)
    ax_steps.set_xlabel("Episode", color=TEXT, fontsize=8)
    ax_steps.set_ylabel("Steps", color=TEXT, fontsize=8)
    ax_steps.legend(fontsize=7, facecolor="#1a1a26",
                    labelcolor=TEXT, framealpha=0.5)

    # --- Panel 4: Q-table heatmap ---
    im = ax_qtable.imshow(agent.Q.T, aspect="auto", cmap="RdYlGn",
                          vmin=agent.Q.min(), vmax=agent.Q.max())
    ax_qtable.set_title("Q-table  (rows: actions)", color=TEXT, fontsize=9)
    ax_qtable.set_xlabel("State", color=TEXT, fontsize=8)
    ax_qtable.set_yticks([0, 1])
    ax_qtable.set_yticklabels(["← left", "→ right"], color=TEXT, fontsize=8)
    cbar = fig.colorbar(im, ax=ax_qtable, pad=0.02)
    cbar.ax.tick_params(colors=TEXT, labelsize=7)

    # --- Panel 5: Greedy policy ---
    policy = agent.greedy_policy()
    colors = [GREEN if a == LinearCorridor.RIGHT else ACCENT2 for a in policy]
    bars = ax_policy.bar(range(env.n_cells), [1] * env.n_cells,
                         color=colors, edgecolor=MUTED, lw=0.5)
    ax_policy.set_ylim(0, 1.6)
    for i, (a, bar) in enumerate(zip(policy, bars)):
        ax_policy.text(i, 1.1,
                       LinearCorridor.ACTION_NAMES[a],
                       ha="center", va="bottom",
                       color=TEXT, fontsize=10, fontweight="bold")
    ax_policy.set_title("Greedy policy (learned)", color=TEXT, fontsize=9)
    ax_policy.set_xlabel("State", color=TEXT, fontsize=8)
    ax_policy.set_yticks([])
    ax_policy.text(0, 1.45, "S", ha="center", color=ACCENT2, fontsize=8)
    ax_policy.text(env.n_cells - 1, 1.45, "G", ha="center",
                   color=GREEN, fontsize=8)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
                exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nPlot saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Week 01 — Hand-rolled RL Loop")
    print("  Environment : LinearCorridor (custom, no Gymnasium)")
    print("  Agent       : Q-learning with epsilon-greedy exploration")
    print("=" * 60)

    # ---- Hyperparameters ----
    N_CELLS       = 10       # corridor length
    N_EPISODES    = 500      # training episodes
    ALPHA         = 0.1      # learning rate
    GAMMA         = 0.99     # discount factor
    EPS_START     = 1.0      # initial exploration
    EPS_MIN       = 0.05     # final exploration
    EPS_DECAY_EP  = 350      # episodes over which to decay epsilon
    SLIP_PROB     = 0.0      # set >0 for a stochastic environment

    # ---- Build env + agent ----
    env   = LinearCorridor(n_cells=N_CELLS, max_steps=200,
                           slip_prob=SLIP_PROB)
    agent = EpsilonGreedyAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=ALPHA, gamma=GAMMA,
        epsilon_start=EPS_START,
        epsilon_min=EPS_MIN,
        epsilon_decay_episodes=EPS_DECAY_EP,
    )

    print(f"\n{env}")
    print(f"{agent}\n")

    # ---- Training ----
    print("Training...")
    stats = train(env, agent, n_episodes=N_EPISODES, verbose_every=100)

    # ---- Evaluation ----
    print("\nEvaluating learned policy (greedy)...")
    evaluate(env, agent, n_episodes=20, render=True)

    # ---- Print learned Q-table ----
    print("\nLearned Q-table (rows = states, cols = [left, right]):")
    header = f"{'State':>6}  {'Q(left)':>10}  {'Q(right)':>10}  {'Policy':>8}"
    print(header)
    print("-" * len(header))
    for s in range(env.n_cells):
        policy_arrow = LinearCorridor.ACTION_NAMES[agent.greedy_policy()[s]]
        print(f"  {s:>4}   {agent.Q[s,0]:>+10.4f}   {agent.Q[s,1]:>+10.4f}"
              f"   {policy_arrow:>6}")

    # ---- Plot ----
    print("\nGenerating training dashboard...")
    plot_training(stats, agent, env,
                  save_path="week-01/training_results.png")


if __name__ == "__main__":
    main()