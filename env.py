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
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

        # Action: 0 = BTTS No, 1 = BTTS Yes, 2 = Skip
        self.action_space = spaces.Discrete(3)

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
            self.balance
        ], dtype=np.float32)
        return obs

    def step(self, action):

        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
        
        # ✅ Exit early if no more games available
        if self.current_game >= len(self.data):
            self.done = True
            return np.zeros(5, dtype=np.float32), 0.0, self.done, False, {
                "balance": self.balance,
                "step": self.step_count,
                "reason": "out_of_games"
            }

        game = self.data[self.current_game]
        actual_btts = game['btts_result']  # 1 for Yes, 0 for No

        selected_odds = game['btts_yes_odds'] if action == 1 else game['btts_no_odds']

        # Always risk one unit
        reward = 0.0
        if self.balance > 0:
            if action == actual_btts:
                self.balance -= self.unit
                # ✅ Calculate win based on selected odds
                win = self.unit * selected_odds
                self.balance += win
                reward = np.clip(win - self.unit, -1, 1)

                # if you win the step risk more amount
                self.unit = round(self.balance / (self.risk_value - 1),2) if self.balance / (self.risk_value - 1) > 0 else 1

            elif action == 2:  # Skip
                reward = 0.0

            else:
                self.balance -= self.unit
                self.unit = round(self.balance / (self.risk_value + 2),2) if self.balance / (self.risk_value + 1) < self.initial_amount else 5
                # reward = 0 # -self.unit This is for when how much you lose is important, but here we just want to track the balance
                reward = np.clip(-self.unit, -1,1)

        # ✅ Prepare observation BEFORE incrementing current_game
        obs = self._get_obs()

        self.current_game += 1

        # ✅ Force episode end if budget exhausted or step limit reached
        self.done = self.balance <= 0 or self.current_game >= len(self.data)

        if self.done:
            obs = np.zeros(5, dtype=np.float32)

        return obs, reward, self.done, False, {
            "balance": self.balance,
            "unit": self.unit,
            "step": self.current_game,
            "selected_odds": selected_odds,
        }

    def render(self, reward, selected_odds=None, unit=None, action=None):
        action_map = {0: "BTTS No", 1: "BTTS Yes", 2: "Skip"}
        print(f"Game {self.current_game}, Balance: {self.balance}, Selected Odds: {selected_odds}, Reward: {reward}, Unit: {self.unit}, Action: {action_map.get(action, action)}")

    def close(self):
        pass
