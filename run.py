from env import FootballBettingEnv

from read_sample import match_batch_generator

batch_gen = match_batch_generator("example.csv", batch_size=34)

for episode_num, batch_data in enumerate(batch_gen, start=1):
    env = FootballBettingEnv(data=batch_data, initial_amount=100)
    obs, _ = env.reset()
    done = False

    while not done:
        action = 1 if obs[3] > obs[4] else 0  # Sample policy
        obs, reward, done, _, info = env.step(action)
        env.render(reward=reward, selected_odds=info['selected_odds'])

    print(f"✅ Episode {episode_num} completed | Final Balance: {info['balance']:.2f}")
