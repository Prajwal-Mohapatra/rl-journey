# Week 02 — Bellman Equation Derivations

## 0. Notation

| Symbol | Meaning |
|--------|---------|
| `S` | Finite set of states |
| `A(s)` | Set of actions available in state `s` |
| `P(s',r \| s,a)` | Probability of transitioning to `s'` with reward `r` given `(s,a)` |
| `π(a\|s)` | Policy — probability of taking action `a` in state `s` |
| `γ ∈ [0,1)` | Discount factor |
| `Gₜ` | Return from time `t`: `Gₜ = Rₜ₊₁ + γRₜ₊₂ + γ²Rₜ₊₃ + …` |
| `vπ(s)` | State-value function under policy `π` |
| `qπ(s,a)` | Action-value function under policy `π` |
| `v*(s)` | Optimal state-value function |
| `q*(s,a)` | Optimal action-value function |

---

## 1. The Return

The agent's goal is to maximise the **expected discounted return**:

```
Gₜ = Rₜ₊₁ + γRₜ₊₂ + γ²Rₜ₊₃ + …
   = Σ_{k=0}^{∞}  γᵏ Rₜ₊ₖ₊₁
```

A critical recursive identity — used in every derivation below:

```
Gₜ = Rₜ₊₁ + γ Gₜ₊₁
```

This single line is why bootstrapping works: the return at time `t` depends
on the return at time `t+1`, which the agent can *estimate* from V(sₜ₊₁).

---

## 2. State-Value Function  vπ(s)

### Definition

```
vπ(s) = Eπ[Gₜ | Sₜ = s]
```

The expected return when starting in state `s` and following policy `π`
thereafter.

### Bellman Expectation Equation for vπ

**Step 1 — expand the definition using the recursive return:**

```
vπ(s) = Eπ[Gₜ | Sₜ = s]
       = Eπ[Rₜ₊₁ + γ Gₜ₊₁ | Sₜ = s]
```

**Step 2 — separate the immediate reward from the future:**

```
       = Eπ[Rₜ₊₁ | Sₜ = s]  +  γ Eπ[Gₜ₊₁ | Sₜ = s]
```

**Step 3 — expand the expectation explicitly over actions and transitions.
The policy chooses `a`, the environment picks `(s', r)`:**

```
       = Σ_a π(a|s)  Σ_{s',r}  P(s',r|s,a) · r
       + γ Σ_a π(a|s) Σ_{s',r}  P(s',r|s,a) · Eπ[Gₜ₊₁ | Sₜ₊₁ = s']
```

**Step 4 — recognise the inner expectation as vπ(s'):**

```
Eπ[Gₜ₊₁ | Sₜ₊₁ = s'] = vπ(s')
```

**Step 5 — combine into the Bellman expectation equation:**

```
vπ(s) = Σ_a π(a|s)  Σ_{s',r}  P(s',r|s,a) · [r + γ vπ(s')]
```

This is equation **(4.4)** in S&B. It says: the value of a state is the
sum over all actions (weighted by the policy) of all transitions (weighted
by the environment dynamics) of the immediate reward plus the discounted
value of the next state.

### Implemented in `gridworld.py`

```python
# iterative_policy_evaluation — inner loop
for a in range(self.n_actions):
    pi_sa = policy[s, a]                        # π(a|s)
    for prob, s_prime, reward, done in self.P[s][a]:
        bootstrap = 0.0 if done else self.gamma * V[s_prime]
        v_new += pi_sa * prob * (reward + bootstrap)
#              ^^^^^^^^ π(a|s) · P(s',r|s,a) · [r + γ V(s')]
```

---

## 3. Action-Value Function  qπ(s,a)

### Definition

```
qπ(s,a) = Eπ[Gₜ | Sₜ = s, Aₜ = a]
```

The expected return when starting in state `s`, taking action `a` first,
then following policy `π`.

### Bellman Expectation Equation for qπ

**Step 1 — expand Gₜ:**

```
qπ(s,a) = Eπ[Rₜ₊₁ + γ Gₜ₊₁ | Sₜ = s, Aₜ = a]
```

**Step 2 — the action `a` is fixed; only the environment randomness remains:**

```
        = Σ_{s',r}  P(s',r|s,a) · [r + γ Eπ[Gₜ₊₁ | Sₜ₊₁ = s']]
```

**Step 3 — the inner expectation is vπ(s'), which we can expand further:**

```
Eπ[Gₜ₊₁ | Sₜ₊₁ = s'] = vπ(s')
                        = Σ_{a'} π(a'|s') qπ(s',a')
```

**Step 4 — substitute:**

```
qπ(s,a) = Σ_{s',r}  P(s',r|s,a) · [r + γ Σ_{a'} π(a'|s') qπ(s',a')]
```

### Relationship between vπ and qπ

These two equations connect the two value functions:

```
vπ(s)   = Σ_a  π(a|s) qπ(s,a)          [average over actions]

qπ(s,a) = Σ_{s',r} P(s',r|s,a) [r + γ vπ(s')]   [average over transitions]
```

Substituting the first into the second recovers the Bellman equation for vπ,
and vice versa. They are two views of the same fixed point.

### Implemented in `gridworld.py`

```python
# q_values(s, V) — one-step lookahead
for a in range(self.n_actions):
    for prob, s_prime, reward, done in self.P[s][a]:
        bootstrap = 0.0 if done else self.gamma * V[s_prime]
        Q[a] += prob * (reward + bootstrap)
# = Σ_{s',r} P(s',r|s,a) · [r + γ V(s')]
```

---

## 4. Optimal Value Functions

### Optimal state-value function

```
v*(s) = max_π vπ(s)   for all s ∈ S
```

There always exists at least one policy `π*` that achieves `v*` everywhere
simultaneously (S&B Theorem 3.1).

### Bellman Optimality Equation for v*

The optimal value function satisfies:

```
v*(s) = max_a  Σ_{s',r}  P(s',r|s,a) · [r + γ v*(s')]
```

**Derivation:**

```
v*(s) = max_π vπ(s)
       = max_a qπ*(s,a)                       [best policy takes best action]
       = max_a Eπ*[Rₜ₊₁ + γ v*(Sₜ₊₁) | Sₜ=s, Aₜ=a]
       = max_a Σ_{s',r} P(s',r|s,a) · [r + γ v*(s')]
```

The key step: under `π*`, once we pick the optimal action `a`, every future
action is also optimal, so the future return is `v*(s')`.

This is equation **(4.1)** in S&B.

### Bellman Optimality Equation for q*

```
q*(s,a) = Σ_{s',r} P(s',r|s,a) · [r + γ max_{a'} q*(s',a')]
```

Once we have `q*`, the optimal policy is simply:

```
π*(s) = argmax_a q*(s,a)
```

No knowledge of the transition model `P` is needed at execution time — just
look up `argmax` in the Q-table. This is why Q-learning (Week 6) is so
powerful: it learns `q*` directly from experience.

### Implemented in `gridworld.py` — greedy improvement

```python
# policy_improvement() — greedy step w.r.t. V ≈ v*
Q = self.q_table(V)          # Q(s,a) = Σ P(s',r|s,a)[r + γ V(s')]
best_a = np.argmax(Q[s])     # argmax_a Q(s,a)
new_policy[s, best_a] = 1.0
```

---

## 5. Bellman Equations as Linear Systems

For a **fixed policy** `π` and a finite state space, the Bellman expectation
equations form a **system of |S| linear equations in |S| unknowns**:

```
vπ = Rπ + γ Pπ vπ
```

where:
- `vπ` is a column vector of length `|S|`
- `Rπ(s) = Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) · r`   (expected immediate reward)
- `Pπ(s,s') = Σ_a π(a|s) P(s'|s,a)`              (policy-weighted transition matrix)

### Closed-form solution

```
(I - γ Pπ) vπ = Rπ
vπ = (I - γ Pπ)⁻¹ Rπ
```

This matrix inverse exists because `γ < 1` makes `(I - γ Pπ)` strictly
diagonally dominant (all eigenvalues of `γ Pπ` have magnitude < 1).

**Why we don't use it in practice:**  
Inverting a `|S| × |S|` matrix costs **O(|S|³)**. For the 4×4 GridWorld
that's fine (16³ = 4096 operations), but for Atari (10^6+ states) or
continuous spaces it's completely intractable. Iterative policy evaluation
costs O(|S|² · |A|) per sweep and only needs O(|S|) memory — far more
scalable.

### The iterative method as a contraction

Iterative policy evaluation repeatedly applies the **Bellman operator** `Tπ`:

```
(Tπ v)(s) = Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) · [r + γ v(s')]
```

`Tπ` is a **contraction mapping** under the sup-norm (∞-norm):

```
‖Tπ u - Tπ v‖∞ ≤ γ ‖u - v‖∞
```

By the **Banach fixed-point theorem**, repeated application of a contraction
converges to a unique fixed point — which is exactly `vπ`. The convergence
rate is geometric: after `k` sweeps the error is at most `γᵏ · ‖v₀ - vπ‖∞`.

For `γ = 1.0` (our GridWorld), the contraction factor is 1 — not strictly
a contraction. Convergence is still guaranteed for episodic tasks because
the terminal states anchor the system, but it takes 167 sweeps rather than
the ~20 needed with `γ = 0.9`.

---

## 6. Bellman Equations on the 4×4 GridWorld

### Concrete example: state 5 (row 1, col 1)

Under uniform random policy (`π(a|s) = 0.25` for all `a`):

```
vπ(5) = 0.25 · [P(1|5,↑)(-1 + vπ(1))   +   P(9|5,↓)(-1 + vπ(9))
              + P(4|5,←)(-1 + vπ(4))   +   P(6|5,→)(-1 + vπ(6))]

       = 0.25 · [(-1 + vπ(1)) + (-1 + vπ(9)) + (-1 + vπ(4)) + (-1 + vπ(6))]

       = 0.25 · [-4 + vπ(1) + vπ(9) + vπ(4) + vπ(6)]

       = 0.25 · [-4 + (-14) + (-20) + (-14) + (-20)]

       = 0.25 · [-4 + (-68)]

       = 0.25 · (-72)

       = -18.0   ✓  matches S&B table
```

All four actions are available (no wall-bounce), transition probability is
1.0 for each, and all transitions carry reward `r = -1`.

### Wall-bounce example: state 3 (row 0, col 3, top-right corner)

State 3 has walls to the North and East, so those actions loop back to
state 3 itself:

```
Transitions from state 3 under each action:
  ↑: stays at s=3  (wall)      → next_state=3,  reward=-1
  ↓: moves to s=7              → next_state=7,  reward=-1
  ←: moves to s=2              → next_state=2,  reward=-1
  →: stays at s=3  (wall)      → next_state=3,  reward=-1
```

Bellman equation:

```
vπ(3) = 0.25 · [(-1 + vπ(3)) + (-1 + vπ(7))
              + (-1 + vπ(2)) + (-1 + vπ(3))]

       = 0.25 · [-4 + 2·vπ(3) + vπ(7) + vπ(2)]
```

Rearranging:

```
vπ(3) - 0.25 · 2 · vπ(3) = 0.25 · [-4 + vπ(7) + vπ(2)]
0.5 · vπ(3) = 0.25 · [-4 + (-20) + (-20)]
0.5 · vπ(3) = 0.25 · (-44) = -11
vπ(3) = -22.0   ✓  matches S&B table
```

The wall-bounce makes state 3 worse than its non-corner neighbours: two of
its four actions waste a step and earn `-1` without making progress.

---

## 7. Policy Improvement Theorem

### Statement (S&B Theorem 4.2)

Let `π` and `π'` be any pair of deterministic policies such that for all
`s ∈ S`:

```
qπ(s, π'(s)) ≥ vπ(s)
```

Then `π'` must be at least as good as `π`:

```
vπ'(s) ≥ vπ(s)   for all s ∈ S
```

### Proof

Starting from the hypothesis `qπ(s, π'(s)) ≥ vπ(s)`:

```
vπ(s) ≤ qπ(s, π'(s))
       = Eπ'[Rₜ₊₁ + γ vπ(Sₜ₊₁) | Sₜ = s]
       ≤ Eπ'[Rₜ₊₁ + γ qπ(Sₜ₊₁, π'(Sₜ₊₁)) | Sₜ = s]
       = Eπ'[Rₜ₊₁ + γ Eπ'[Rₜ₊₂ + γ vπ(Sₜ₊₂)] | Sₜ = s]
       = Eπ'[Rₜ₊₁ + γ Rₜ₊₂ + γ² vπ(Sₜ₊₂) | Sₜ = s]
       ≤ …   (unroll k more steps)
       ≤ Eπ'[Rₜ₊₁ + γ Rₜ₊₂ + γ² Rₜ₊₃ + … | Sₜ = s]
       = vπ'(s)
```

Each unrolling applies the hypothesis one step further into the future.
The limit holds because `γᵏ → 0` as `k → ∞` (for `γ < 1`), so the
telescoping sum converges to `vπ'(s)`. ∎

### Why the greedy policy satisfies the hypothesis

The greedy policy `π'(s) = argmax_a qπ(s,a)` satisfies, by definition:

```
qπ(s, π'(s)) = max_a qπ(s,a)
             ≥ Σ_a π(a|s) qπ(s,a)    [max ≥ average]
             = vπ(s)
```

The last step uses the relationship `vπ(s) = Σ_a π(a|s) qπ(s,a)`.

So the greedy step *always* satisfies the theorem's condition, and
`vπ' ≥ vπ` is therefore always guaranteed after improvement.

### Verified in our code

```
[ Step 4 ]  Verify Policy Improvement Theorem: v_π'(s) ≥ v_π(s) …
  Max improvement  : +19.0000
  Min improvement  : +0.0000   (terminals — already 0)
  Violations (< 0) : 0  ✓ PASS — theorem holds
  Mean V: -18.29  →  -2.00  (non-terminal states)
```

---

## 8. Convergence of Policy Iteration

### The algorithm

```
1. Initialise V(s) = 0 for all s,  π(s) = arbitrary
2. Repeat:
     a. Policy Evaluation:  iterate Bellman until ‖ΔV‖∞ < θ
     b. Policy Improvement: π' ← greedy w.r.t. V
     c. If policy_stable: break
3. Return π, V
```

### Why it terminates

- There are at most `|A|^|S|` deterministic policies (finite).
- Each improvement step produces a strictly better policy unless we're
  already at optimal (by the Improvement Theorem).
- A finite set with a strictly improving sequence must terminate.

For our 4×4 GridWorld: `4^16 = 4,294,967,296` possible policies, but Policy
Iteration finds the optimum in **2 improvement steps** in practice.

### Convergence speed vs. Value Iteration

| Method | Sweeps (GridWorld) | Cost per sweep |
|--------|--------------------|----------------|
| Policy Iteration eval (step 1) | 167 | O(\|S\|² \|A\|) |
| Policy Iteration eval (step 2) | 4 | O(\|S\|² \|A\|) |
| Value Iteration (Week 3) | ~20 | O(\|S\|² \|A\|) |

Policy Iteration does fewer total sweeps because the second (and subsequent)
evaluations converge fast — the improved policy's value function is nearly
correct from the previous iteration's V.

---

## 9. Key Equations — Quick Reference

```
# Bellman Expectation (state-value)
vπ(s) = Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ vπ(s')]

# Bellman Expectation (action-value)
qπ(s,a) = Σ_{s',r} P(s',r|s,a) [r + γ Σ_{a'} π(a'|s') qπ(s',a')]

# Connecting vπ and qπ
vπ(s) = Σ_a π(a|s) qπ(s,a)
qπ(s,a) = Σ_{s',r} P(s',r|s,a) [r + γ vπ(s')]

# Bellman Optimality (state-value)
v*(s) = max_a Σ_{s',r} P(s',r|s,a) [r + γ v*(s')]

# Bellman Optimality (action-value)
q*(s,a) = Σ_{s',r} P(s',r|s,a) [r + γ max_{a'} q*(s',a')]

# Optimal policy from q*
π*(s) = argmax_a q*(s,a)

# Policy Improvement condition
qπ(s, π'(s)) ≥ vπ(s)  ⟹  vπ'(s) ≥ vπ(s)  for all s
```

