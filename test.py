import torch
import torch.nn as nn
import numpy as np
from env import FootballBettingEnv
from read_sample import match_batch_generator

# --- Constants ---
MODEL_PATH = "dqn_model.pth"
TEST_DATA_PATH = "test_sample.csv"  # small test CSV
EPISODES = 10  # Number of batches to test (1 = one batch of 34)

# --- Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Q-network (same architecture used during training) ---
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

# --- Load Model ---
obs_size = 7
n_actions = 2
policy_net = QNetwork(obs_size, n_actions).to(DEVICE)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
policy_net.eval()
print("✅ Loaded model from", MODEL_PATH)

# --- Test Loop ---
batch_gen = match_batch_generator(TEST_DATA_PATH, batch_size=34)

for episode_num, batch_data in enumerate(batch_gen, start=1):
    if episode_num > EPISODES:
        break

    env = FootballBettingEnv(data=batch_data, initial_amount=100)
    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32).flatten()
    done = False
    total_reward = 0

    print(f"\n🧪 Test Episode {episode_num} Starts")

    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            q_values = policy_net(state_tensor)
            action = int(torch.argmax(q_values).item())

        obs, reward, done, _, info = env.step(action)
        obs = np.array(obs, dtype=np.float32).flatten()
        total_reward += reward
        env.render(reward=reward, selected_odds=info['selected_odds'])

    final_balance = info["balance"]
    print(f"🎯 Final Balance: {final_balance:.2f} | Net Gain/Loss: {final_balance - 100:.2f}")
