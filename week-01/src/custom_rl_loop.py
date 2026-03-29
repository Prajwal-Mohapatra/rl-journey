"""
Week 1/2: Hand-Rolled 1D Environment & RL Loop
Environment: 1D LineWorld (Corridor)

The Goal: The agent starts in the middle of a corridor. 
It must learn to go RIGHT to reach the Win (+1) state, 
while avoiding the LEFT end which is a Loss (-1) state.
"""

import numpy as np
import random
import time

# --- 1. The 1D Environment ---

class LineWorldEnv:
    def __init__(self, size: int = 7):
        self.size = size
        self.start_state = size // 2  # Start in the middle
        self.state = self.start_state
        self.action_space = [0, 1]    # 0: LEFT, 1: RIGHT

    def reset(self) -> int:
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        """Applies the action and returns (next_state, reward, done)."""
        if action == 0: # LEFT
            self.state = max(0, self.state - 1)
        elif action == 1: # RIGHT
            self.state = min(self.size - 1, self.state + 1)

        # Rewards and terminal states
        if self.state == 0:
            return self.state, -1.0, True   # Fell off the left edge (Loss)
        elif self.state == self.size - 1:
            return self.state, 1.0, True    # Reached the right edge (Win)
        else:
            return self.state, 0.0, False   # Neutral step

    def render(self):
        """Prints the 1D corridor to the terminal."""
        corridor = ["-"] * self.size
        corridor[0] = "X" # Trap/Loss
        corridor[-1] = "G" # Goal/Win
        
        # Place the agent if it hasn't fallen in the trap or reached the goal
        if self.state != 0 and self.state != self.size - 1:
            corridor[self.state] = "A"
            
        print(f"\r[{''.join(corridor)}]", end="", flush=True)


# --- 2. The Agent & Training Loop ---

def train_1d_agent(episodes: int = 50):
    env = LineWorldEnv(size=7)
    
    # Q-Table: Rows = States (0 to 6), Cols = Actions (0: Left, 1: Right)
    # We initialize with zeros.
    q_table = np.zeros((env.size, len(env.action_space)))
    
    # Hyperparameters
    alpha = 0.1      # Learning rate
    gamma = 0.9      # Discount factor
    epsilon = 0.9    # Initial exploration rate
    epsilon_decay = 0.95
    epsilon_min = 0.01

    print("Starting Training... Watch the Q-Table evolve!")
    print("-" * 50)
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # 1. Epsilon-Greedy Action Selection
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.action_space) # Explore
            else:
                action = np.argmax(q_table[state])       # Exploit
                
            # 2. Take Action
            next_state, reward, done = env.step(action)
            
            # 3. Q-Learning Update (Bellman Equation)
            # Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
            max_future_q = 0.0 if done else np.max(q_table[next_state])
            td_target = reward + gamma * max_future_q
            td_error = td_target - q_table[state, action]
            
            q_table[state, action] += alpha * td_error
            
            # Move to next state
            state = next_state
            
        # Decay epsilon at the end of each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Print the Q-table every 10 episodes to visualize learning
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1} | Epsilon: {epsilon:.2f}")
            print("State | Q(Left) | Q(Right)")
            print("-" * 26)
            for s in range(1, env.size - 1): # Skip terminal states in print
                print(f"  {s}   |  {q_table[s, 0]:5.2f}  |  {q_table[s, 1]:5.2f}")
            print("-" * 26)

    return env, q_table

def evaluate_1d_agent(env: LineWorldEnv, q_table: np.ndarray):
    """Watches the trained agent navigate without exploring."""
    print("\nEvaluating Trained Agent (Greedy Policy)...")
    state = env.reset()
    done = False
    
    env.render()
    time.sleep(0.5)
    
    while not done:
        # Pure exploitation (no epsilon)
        action = np.argmax(q_table[state])
        
        state, reward, done = env.step(action)
        env.render()
        time.sleep(0.5)
        
    if reward > 0:
        print("\nSuccess! Reached the Goal.")
    else:
        print("\nFailed! Fell into the trap.")

if __name__ == "__main__":
    trained_env, learned_q_table = train_1d_agent(episodes=60)
    evaluate_1d_agent(trained_env, learned_q_table)