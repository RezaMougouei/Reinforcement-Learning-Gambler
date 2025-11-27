# Gambler Bandit Project

This repository contains a full implementation of a **4‑arm restless bandit task** simulated using two ACT‑R agents:

* **Baseline Gambler** (random policy)
* **Q‑Learning Gambler** (reinforcement‑learning‑augmented ACT‑R agent)

The environment reads from a CSV file with four drifting reward distributions (“Machine 1–4”) and simulates 150 trials per run.

---

## 📁 Project Structure

```
project/
├── cleaned_gambler_code.py       # Main implementation (environment + agents)
├── README.md                      # Project documentation
├── data/                          # Your CSV dataset (not included)
├── outputs/                       # Generated logs + CSV files
└── plots/                         # Any graphs you produce externally
```

---

## 🎮 Environment: RandomBanditEnv

A custom Gymnasium environment that:

* Loads reward schedules from a CSV file
* Generates a random 150‑trial trajectory per episode
* Returns reward values for the chosen machine
* Tracks episode progression

Machines must be named like:

```
Machine 1,
Machine 2,
Machine 3,
Machine 4
```

---

## 🧠 Agents

### **BaselineGambler (ACT‑R random policy)**

* Chooses a machine uniformly at random
* Logs arm choices and reward history
* Stops after 150 trials

### **QLearningGambler (ACT‑R + PMQ module)**

* Four production rules correspond to choosing Machines 1–4
* PMQ module performs:

  * Q‑value updates
  * State/action matching
  * Epsilon‑greedy selection
* Stores Q‑value snapshots across all trials for analysis

---

## 📊 Output Files

The code can generate:

* `aggregated_results.csv` — mean reward and arm choice proportions
* `q_values_over_time.csv` — average Q‑values by arm over trials
* `q_value_evolution.txt` — human‑readable Q‑table evolution

These are useful for plotting:

* Learning curves
* Reward trajectories
* Q‑value convergence

---

## ▶️ Running the Model

### Minimal example

```python
from cleaned_gambler_code import RandomBanditEnv, GamblerBody, QLearningGambler
from python_actr import Model

env = RandomBanditEnv(csv_path="your_dataset.csv")
body = GamblerBody(env)
agent = QLearningGambler()

model = Model(env=env, gambler_body=body)
model.agent = agent
model.run()
```

---

## 📦 Requirements

Add this to a `requirements.txt` file:

```
gymnasium
python_actr
numpy
pandas
```

---

## 🔍 Notes

* This repository is cleaned for GitHub and removes Colab‑specific commands.
* CSV data is not included
