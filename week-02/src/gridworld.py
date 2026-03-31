# week-02/src/gridworld.py
import numpy as np

class GridWorld:
    """
    4x4 GridWorld MDP from S&B Example 4.1.
    States: 0–15 (row-major). States 0 and 15 are terminal.
    Actions: 0=up, 1=down, 2=left, 3=right
    Reward: -1 on every non-terminal transition.
    """
    def __init__(self, size=4, gamma=1.0):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.gamma = gamma
        self.terminal_states = {0, self.n_states - 1}
        self._build_transitions()

    def _build_transitions(self):
        """P[s][a] = list of (prob, next_state, reward, done)"""
        s = self.size
        moves = [(-s, 0), (s, 0), (0, -1), (0, 1)]  # up,down,left,right
        self.P = {}
        for state in range(self.n_states):
            self.P[state] = {}
            row, col = divmod(state, s)
            for action, (dr, dc) in enumerate(moves):
                if state in self.terminal_states:
                    self.P[state][action] = [(1.0, state, 0, True)]
                    continue
                nr, nc = row + dr, col + dc
                # Clamp to grid boundaries
                nr = max(0, min(s - 1, nr))
                nc = max(0, min(s - 1, nc))
                next_state = nr * s + nc
                done = next_state in self.terminal_states
                self.P[state][action] = [(1.0, next_state, -1.0, done)]

    def iterative_policy_evaluation(self, policy, theta=1e-6):
        """Evaluate a given policy until convergence."""
        V = np.zeros(self.n_states)
        while True:
            delta = 0
            for s in range(self.n_states):
                v = V[s]
                new_v = 0
                for a, prob in enumerate(policy[s]):
                    for trans_prob, s_prime, reward, _ in self.P[s][a]:
                        new_v += prob * trans_prob * (reward + self.gamma * V[s_prime])
                V[s] = new_v
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        return V

if __name__ == "__main__":
    gw = GridWorld()
    # Uniform random policy
    uniform_policy = np.ones((gw.n_states, gw.n_actions)) / gw.n_actions
    V = gw.iterative_policy_evaluation(uniform_policy)
    print("State values under uniform random policy:")
    print(V.reshape(4, 4).round(1))