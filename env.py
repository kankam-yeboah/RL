import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class FootballBettingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data, initial_amount=100):
        super(FootballBettingEnv, self).__init__()
        self.data = data
        self.total_games = len(data)
        self.initial_amount = initial_amount

        # State: home, draw, away odds + BTTS yes/no odds + gameweek
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)

        # Action: 0 = BTTS No, 1 = BTTS Yes
        self.action_space = spaces.Discrete(2)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_game = 0
        self.balance = self.initial_amount
        self.risk_value = 5
        self.unit = self.initial_amount / self.risk_value
        self.done = False

        return self._get_obs(), {}

    def _get_obs(self):
        game = self.data[self.current_game]
        obs = np.array([
            game['btts_yes_odds'],
            game['btts_no_odds'],
            game['home_team_id'],
            game['away_team_id'],
        ], dtype=np.float32)
        return obs

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        if self.current_game >= len(self.data):
            self.done = True
            return np.zeros(4, dtype=np.float32), 0.0, self.done, False, {
                "balance": self.balance,
                "step": self.current_game,
                "reason": "out_of_games"
            }

        game = self.data[self.current_game]
        actual_btts = game['btts_result']  # 1 = Yes, 0 = No
        yes_odds = game['btts_yes_odds']
        no_odds = game['btts_no_odds']
        selected_odds = yes_odds if action == 1 else no_odds

        reward = 0.0
        if self.balance > 0:
            self.balance -= self.unit

            if action == actual_btts:
                win = self.unit * selected_odds
                self.balance += win
                # Reward shaped based on profit margin
                reward = min((win - self.unit) / self.unit, 1.0)
                self.unit = round(self.balance / (self.risk_value - 1), 2)
            else:
                # Lose the unit bet
                reward = -1.0  # Penalize wrong decision
                self.unit = round(self.balance / (self.risk_value + 2), 2)

        obs = self._get_obs()
        self.current_game += 1
        self.done = self.balance <= 0 or self.current_game >= len(self.data)
        if self.done:
            obs = np.zeros(4, dtype=np.float32)

        return obs, reward, self.done, False, {
            "balance": self.balance,
            "unit": self.unit,
            "step": self.current_game,
            "selected_odds": selected_odds,
        }

    def render(self, reward, selected_odds=None, unit=None, action=None):
        action_map = {0: "BTTS No", 1: "BTTS Yes"}
        print(f"Game {self.current_game}, Balance: {self.balance}, Selected Odds: {selected_odds}, Reward: {reward}, Unit: {self.unit}, Action: {action_map.get(action, action)}")

    def close(self):
        pass
