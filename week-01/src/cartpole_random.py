#import dependancies
import gymnasium as gym
import matplotlib.pyplot as plt

#define run_random_agent function
def run_random_agent(n_episodes=100):
    env = gym.make("CartPole-v1")
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        episode_rewards.append(total_reward)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}: reward={total_reward:.1f}")

    env.close()
    return episode_rewards

if __name__ == "__main__":
    rewards = run_random_agent()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Random agent on CartPole-v1")
    plt.savefig("random_agent_rewards.png")
    plt.show()
