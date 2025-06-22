import numpy as np
import matplotlib.pyplot as plt
import pickle
from env import FootballBettingEnv
from read_sample import match_batch_generator

# Constants
INITIAL_AMOUNT = 100

# ---- Hyperparameters ----
EPISODES = 1000
EPSILON = 1.0        # initial exploration rate
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
ALPHA = 0.1
GAMMA = 0.9
BATCH_SIZE = 5
NUM_BINS = 5

# ---- Discretization Bins ----
obs_bins = [np.linspace(1, 10, NUM_BINS) for _ in range(6)]

# ---- Q-table ----
q_shape = (NUM_BINS,) * 6 + (2,)  # 6 obs features, 2 actions
Q = np.random.uniform(low=0, high=1, size=q_shape)

# ---- Helpers ----
def discretize_obs(obs):
    state_idx = []
    for i in range(len(obs)):
        bin_idx = np.digitize(obs[i], obs_bins[i]) - 1
        bin_idx = max(0, min(NUM_BINS - 1, bin_idx))
        state_idx.append(bin_idx)
    return tuple(state_idx)

# ---- Metrics Tracking ----
episode_rewards = []

# ---- Training Loop ----
batch_gen = match_batch_generator("example.csv", batch_size=BATCH_SIZE)

for episode_num, batch_data in enumerate(batch_gen, start=1):
    if episode_num > EPISODES:
        break

    env = FootballBettingEnv(data=batch_data, initial_amount=INITIAL_AMOUNT)
    obs, _ = env.reset()
    state = discretize_obs(obs)
    done = False

    # Balance tracking at the end of the episode


    while not done:
        # ε-greedy action selection
        if np.random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_obs, reward, done, _, info = env.step(action)
        next_state = discretize_obs(next_obs)

        # Q-learning update
        best_future_q = np.max(Q[next_state])
        Q[state + (action,)] += ALPHA * (reward + GAMMA * best_future_q - Q[state + (action,)])

        state = next_state

    # ✅ Net profit/loss at end of episode
    final_reward = info["balance"] - INITIAL_AMOUNT
    episode_rewards.append(final_reward)

    # Decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    print(f"🎯 Episode {episode_num} | Reward: {final_reward:.2f} | Epsilon: {EPSILON:.4f}")

# ---- Save Q-table ----
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)
print("✅ Q-table saved to 'q_table.pkl'")

# ---- Plot Performance ----
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label='Episode Reward')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Episode Rewards")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("q_learning_rewards.png")
plt.show()
