import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


def main(env, agent):
    N_ITERATIONS = 1000
    N_STEPS = 200

    episode_rewards = []
    moving_avg_rewards = deque(maxlen=50)

    for episode in range(N_ITERATIONS):
        memory = []
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False

        ep_reward = 0
        ep_len = 0

        for _ in range(N_STEPS):
            if done:
                break

            action, log_prob = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            memory.append([
                state,
                torch.tensor([[action]]),
                torch.tensor([reward], dtype=torch.float32),
                next_state,
                torch.tensor([done], dtype=torch.float32),
                torch.tensor([log_prob], dtype=torch.float32)
            ])
            state = next_state
            ep_reward += reward
            ep_len += 1

        episode_rewards.append(ep_reward)
        moving_avg_rewards.append(ep_reward)

        advantage = agent.cal_advantage(memory)
        loss = agent.train(memory, advantage)

        avg_reward = np.mean(moving_avg_rewards)
        print(f"Episode {episode:3d} | Loss: {loss: .4f} | Reward: {ep_reward:3.0f} | Avg(50): {avg_reward: .2f} | Len: {ep_len}")

    # 결과 시각화 
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'), label='Moving Avg (50)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid()
    plt.title('PPO Training Performance')
    plt.show()