# Plot graphs
import matplotlib.pyplot as plt
import numpy as np

# Helper functions
# -------------------------------------------------------------
def process_rewards(episode_rewards):
    # Convert to numpy array for filtering
    rewards = np.array(episode_rewards)
    episodes = np.arange(1, len(rewards) + 1)

    # Positive profit episodes
    positive_mask = rewards > 0
    positive_episodes = episodes[positive_mask]
    positive_rewards = rewards[positive_mask]

    # Negative profit episodes
    negative_mask = rewards < 0
    negative_episodes = episodes[negative_mask]
    negative_rewards = rewards[negative_mask]

    return positive_episodes, positive_rewards, negative_episodes, negative_rewards

# -------------------------------------------------------------


def cumulative_rewards_loss(episode_rewards):
    # Cumulative tracking of profit and loss as episodes progress
    cumulative_profits = []
    cumulative_losses = []
    total_profit = 0
    total_loss = 0

    for r in episode_rewards:
        if r > 0:
            total_profit += r
        elif r < 0:
            total_loss += abs(r)
        cumulative_profits.append(total_profit)
        cumulative_losses.append(total_loss)

    episodes = np.arange(1, len(episode_rewards) + 1)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, cumulative_profits, label="Cumulative Profit", color='green')
    plt.plot(episodes, cumulative_losses, label="Cumulative Loss", color='red')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Value")
    plt.title("Cumulative Profit and Loss Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("./chart/cumulative_profit_loss.png")
    plt.tight_layout()
    plt.show()



def plot_rewards(episode_rewards):
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards Over Time")
    plt.grid()
    plt.savefig("./chart/rewards_over_time.png")
    # plt.show()


def plot_P_N(episode_rewards):
    """
    Plot the number of positive and negative rewards over episodes.
    """
    positive_rewards = [r for r in episode_rewards if r > 0]
    negative_rewards = [r for r in episode_rewards if r < 0]

    plt.figure(figsize=(10, 5))
    plt.bar(['Positive Rewards', 'Negative Rewards'], [len(positive_rewards), len(negative_rewards)], color=['green', 'red'])
    plt.title("Positive vs Negative Rewards")
    plt.ylabel("Count")
    plt.grid(axis='y')
    plt.savefig("./chart/positive_negative_rewards.png")
    plt.show()

def plot_P_N_flow(episode_rewards):
    positive_episodes, positive_rewards, negative_episodes, negative_rewards = process_rewards(episode_rewards)

    # Plot: Profit-only Episodes
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(positive_episodes, positive_rewards, color='green')
    plt.title("Episodes with Net Profit > 0")
    plt.xlabel("Episode")
    plt.ylabel("Net Profit")
    plt.grid(True)

    # Plot: Loss-only Episodes
    plt.subplot(1, 2, 2)
    plt.plot(negative_episodes, negative_rewards, color='red')
    plt.title("Episodes with Net Loss < 0")
    plt.xlabel("Episode")
    plt.ylabel("Net Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./chart/positive_negative_flow.png")
    # plt.show()


def plot_P_N_flow_average(episode_rewards):
    positive_episodes, positive_rewards, negative_episodes, negative_rewards = process_rewards(episode_rewards)


    def moving_average(data, window_size=10):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # Compute moving averages
    positive_ma = moving_average(positive_rewards)
    positive_ma_episodes = positive_episodes[:len(positive_ma)]

    negative_ma = moving_average(negative_rewards)
    negative_ma_episodes = negative_episodes[:len(negative_ma)]

    # Plot with moving average lines
    plt.figure(figsize=(12, 5))

    # Positive profit plot
    plt.subplot(1, 2, 1)
    plt.plot(positive_ma_episodes, positive_ma, color='darkgreen', linewidth=2, label='Moving Average')
    plt.title("Episodes with Net Profit > 0")
    plt.xlabel("Episode")
    plt.ylabel("Net Profit")
    plt.legend()
    plt.grid(True)

    # Negative profit plot
    plt.subplot(1, 2, 2)
    plt.plot(negative_ma_episodes, negative_ma, color='darkred', linewidth=2, label='Moving Average')
    plt.title("Episodes with Net Loss < 0")
    plt.xlabel("Episode")
    plt.ylabel("Net Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./chart/positive_negative_flow_average.png")
    # plt.show()
