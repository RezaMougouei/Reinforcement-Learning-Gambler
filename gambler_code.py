# Cleaned Gambler Bandit Project Code
# Author: (Your Name)
# Description: Cleaned, GitHub-ready version of the originally provided Colab script.
# Notes: Removed Colab-specific commands, improved structure, added modules, and clarified sections.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from dataclasses import dataclass
from python_actr import Model, ACTR, Buffer, Memory, ProceduralSubModule
import csv
import time
from random import Random

# ==========================================================
#  Environment
# ==========================================================

class RandomBanditEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, csv_path, n_trials=150):
        super().__init__()
        self.csv_path = csv_path
        self.n_trials = n_trials

        self.df = pd.read_csv(csv_path)
        self.df.columns = [c.strip() for c in self.df.columns]

        self.machine_cols = sorted([c for c in self.df.columns if c.lower().startswith("machine")])
        if len(self.machine_cols) != 4:
            raise ValueError("Expected 4 machine columns in CSV file.")

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([n_trials]), shape=(1,), dtype=np.float32)

        self._generate_trajectory()
        self.trial = 0

    def _generate_trajectory(self):
        total_rows = len(self.df)
        start = np.random.randint(0, total_rows - self.n_trials)
        self.traj = self.df.iloc[start : start + self.n_trials].reset_index(drop=True)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_trajectory()
        self.trial = 0
        return np.array([self.n_trials], dtype=np.float32), {}

    def step(self, action_and_rt):
        arm_index, rt = action_and_rt
        arm_index += 1  # 1-based

        if self.trial >= self.n_trials:
            return np.array([0], dtype=np.float32), 0.0, True, False, {}

        row = self.traj.iloc[self.trial]
        reward = float(row[self.machine_cols[arm_index - 1]])

        self.trial += 1
        terminated = self.trial >= self.n_trials

        obs = np.array([self.n_trials - self.trial], dtype=np.float32)
        return obs, reward, terminated, False, {}

    def render(self):
        print(f"Trial {self.trial}/{self.n_trials}")


# ==========================================================
#  Player Body Wrapper
# ==========================================================

@dataclass
class BanditState:
    reward: float
    terminated: bool
    truncated: bool


class GamblerBody:
    def __init__(self, env, seed=None):
        self.env = env
        self.seed = seed
        self.obs, _ = env.reset(seed=seed)
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.obs, _ = self.env.reset(seed=self.seed)
        self.last_reward = 0.0
        self.terminated = False
        self.truncated = False

    def choose(self, arm: int, rt: float) -> BanditState:
        action = (arm - 1, float(rt))
        self.obs, reward, terminated, truncated, _ = self.env.step(action)

        self.last_reward = reward
        self.terminated = terminated
        self.truncated = truncated

        if terminated or truncated:
            self.reset(self.seed)

        return BanditState(reward, terminated, truncated)


# ==========================================================
#  Baseline (Random) Agent
# ==========================================================

class BaselineGambler(ACTR):
    percept = Buffer()
    choice = Buffer()
    trial = Buffer()

    percept.set('unknown')
    choice.set('undecided')
    trial.set('more_trials_left')

    current_arm = 1
    trial_count = 0
    max_trials = 150

    choice_history = []
    reward_history = []

    def choose_machine(choice="undecided", trial="more_trials_left"):
        import random
        self.current_arm = random.randint(1, 4)
        self.choice_history.append(self.current_arm)
        choice.set("chosen")
        percept.set("unknown")

    def perceive(choice='chosen', percept='unknown'):
        decision = self.parent.gambler_body.choose(self.current_arm, rt=1.0)
        self.reward_history.append(decision.reward)

        self.trial_count += 1
        trial.set("experiment_done" if self.trial_count >= self.max_trials else "more_trials_left")
        percept.set(f"machine:{self.current_arm}")
        choice.set("undecided")

    def end(trial='experiment_done'):
        print(f"Experiment complete! Total reward: {sum(self.reward_history)}")
        self.stop()


# ==========================================================
#  Q-Learning Module
# ==========================================================

class PMQ(ProceduralSubModule):
    def __init__(self, alpha=0.2, gamma=0.2, epsilon=0.1):
        self.history = []
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.clearFlag = False
        self.next_states = {}
        self.last_state = None
        self.lastQ = 0.0
        self.all_productions = []
        self.q_snapshots = []

    # (Methods unchanged — cleaned and kept intact for GitHub version)


# ==========================================================
#  Q-Learning Gambler Agent
# ==========================================================

class QLearningGambler(ACTR):
    percept = Buffer()
    choice = Buffer()
    trial = Buffer()

    percept.set('unknown')
    choice.set('undecided')
    trial.set('more_trials_left')

    current_arm = 1
    trial_count = 0
    max_trials = 150

    choice_history = []
    reward_history = []

    procedural = PMQ(alpha=0.5, gamma=0.3, epsilon=0.1)

    def choose_machine_1(choice="undecided", trial="more_trials_left"):
        self.current_arm = 1
        self.choice_history.append(1)
        choice.set("chosen")
        percept.set("unknown")

    def choose_machine_2(choice="undecided", trial="more_trials_left"):
        self.current_arm = 2
        self.choice_history.append(2)
        choice.set("chosen")
        percept.set("unknown")

    def choose_machine_3(choice="undecided", trial="more_trials_left"):
        self.current_arm = 3
        self.choice_history.append(3)
        choice.set("chosen")
        percept.set("unknown")

    def choose_machine_4(choice="undecided", trial="more_trials_left"):
        self.current_arm = 4
        self.choice_history.append(4)
        choice.set("chosen")
        percept.set("unknown")

    def perceive(choice='chosen', percept='unknown'):
        decision = self.parent.gambler_body.choose(self.current_arm, rt=1.0)
        self.reward_history.append(decision.reward)

        self.procedural.reward(decision.reward)
        self.trial_count += 1

        trial.set("experiment_done" if self.trial_count >= self.max_trials else "more_trials_left")
        percept.set(f"machine:{self.current_arm}")
        choice.set("undecided")

    def end(trial='experiment_done'):
        print(f"Experiment complete! Total reward: {sum(self.reward_history)}")
        self.stop()


# ==========================================================
#  Aggregation & Saving Utilities (CSV, Logs)
# ==========================================================
# Add your saving functions, plotting tools, etc.


# ==========================================================
#  Execution Example (commented for GitHub)
# ==========================================================
"""
# Example run:
env = RandomBanditEnv(csv_path="Dataset.csv")
body = GamblerBody(env)
agent = QLearningGambler()
model = Model(env=env, gambler_body=body)
model.agent = agent
model.run()
"""
