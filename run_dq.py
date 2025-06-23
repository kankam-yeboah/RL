import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env import FootballBettingEnv
from read_sample import match_batch_generator
from analysis import plot_rewards, plot_P_N, plot_P_N_flow, plot_P_N_flow_average, cumulative_rewards_loss

# Constants
INITIAL_AMOUNT = 100

# --- Hyperparameters ---
EPISODES = 10000
GAMMA = 0.95
ALPHA = 0.0001
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
CONFIDENCE_THRESHOLD = 0.9 # ✅ NEW
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Q-network ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- Setup ---
obs_size = 4
n_actions = 2
policy_net = QNetwork(obs_size, n_actions).to(DEVICE)
optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)
epsilon = EPSILON_START
episode_rewards = []

batch_gen = match_batch_generator("example.csv", batch_size=3)

# --- Training Loop with Q-Learning + Confidence Threshold ---
for episode_num, batch_data in enumerate(batch_gen, start=1):
    if episode_num > EPISODES:
        break

    env = FootballBettingEnv(data=batch_data, initial_amount=INITIAL_AMOUNT)
    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32).flatten()
    done = False
    total_reward = 0

    while not done:
        skip_step = False

        # ε-greedy action selection with confidence threshold
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                q_values = policy_net(state_tensor)
                max_q = torch.max(q_values).item()
                if max_q < CONFIDENCE_THRESHOLD:
                    skip_step = True  # ✅ Don't bet
                else:
                    action = int(torch.argmax(q_values).item())

        if skip_step:
            # ✅ Skip betting this game
            env.current_game += 1
            if env.current_game >= len(env.data):
                break
            obs = env._get_obs()
            obs = np.array(obs, dtype=np.float32).flatten()
            continue

        # Take action
        next_obs, reward, done, _, info = env.step(action)
        next_obs = np.array(next_obs, dtype=np.float32).flatten()

        # Compute target Q-value
        with torch.no_grad():
            next_state_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            q_next_max = torch.max(policy_net(next_state_tensor)).item() if not done else 0.0

        target_q = reward + GAMMA * q_next_max

        # Predicted Q-value
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        q_pred = policy_net(state_tensor)[0, action]

        # Loss and optimization
        loss = (q_pred - target_q) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs
        total_reward += reward

    final_reward = info["balance"] - INITIAL_AMOUNT
    episode_rewards.append(final_reward)

    if epsilon > EPSILON_MIN:
        epsilon *= EPSILON_DECAY

    print(f"✅ Episode {episode_num} | Final Balance: {info['balance']:.2f} | Net: {final_reward:.2f} | ε={epsilon:.3f}")

# Save model
torch.save(policy_net.state_dict(), "q_learning_model_with_threshold.pth")
print("🎉 Q-Learning model saved to q_learning_model_with_threshold.pth")

# Plot
plot_rewards(episode_rewards)
plot_P_N(episode_rewards)
plot_P_N_flow(episode_rewards)
plot_P_N_flow_average(episode_rewards)
cumulative_rewards_loss(episode_rewards)
