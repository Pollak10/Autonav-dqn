# AutoNav — Autonomous Highway Driving Agent

AutoNav is a reinforcement learning agent that learns to drive on a simulated multi lane highway. It controls a car in real time deciding when to change lanes, accelerate, or brake and improves through trial and error over hundreds of training episodes. The goal is to travel as far and as fast as possible without crashing into other vehicles.

The agent uses a Deep Q-Network (DQN) trained in the `highway-env` Gymnasium environment, a lightweight 2D highway simulation that runs on any laptop without a GPU or game engine.

---

## What it does

- Simulates a car navigating a 4-lane highway with 30 other vehicles
- Learns a driving policy from scratch using only reward signals no labeled data, no human demonstrations
- Trains for 500 episodes and evaluates every 50 with a greedy (no-exploration) policy
- Tracks collision rate, survival time, and reward over training
- Saves the best performing model and lets you watch it drive after training
- Outputs a 4 panel results dashboard saved to `autonav_results.png`

---

## Algorithm

**Deep Q-Network (DQN)** with three enhancements:

**Dueling architecture** — the network splits into two streams: one estimates how good the current state is (value), the other estimates how much better each action is relative to average (advantage). They combine to produce Q-values. This helps in highway driving because many situations have similar value regardless of which lane you're in the dueling structure handles that better than a standard single stream network.

**Experience replay** — transitions (state, action, reward, next state) are stored in a buffer of 20,000 entries and sampled randomly during training. This breaks the temporal correlation between consecutive steps, which would otherwise destabilise learning.

**Double DQN** — the online network selects the best next action, but the target network evaluates it. This prevents the overestimation of Q-values that standard DQN is prone to, especially early in training when estimates are noisy.

**Epsilon-greedy exploration** — the agent starts by taking random actions (ε=1.0) and gradually shifts toward exploiting what it has learned (ε decays to 0.05 over training). This ensures the agent explores enough of the state space before committing to a policy.

DQN was chosen over policy gradient methods (like PPO or A3C) because the action space is small and discrete (5 actions), the environment is relatively low dimensional, and DQN converges reliably in these conditions with less hyperparameter sensitivity.

---

## Environment and Dataset

Reinforcement learning does not use a static dataset. The agent generates its own experience by interacting with the environment.

| Property | Value |
|---|---|
| Environment | `highway-v0` (highway-env Gymnasium) |
| Source | https://github.com/Farama-Foundation/HighwayEnv |
| Observation type | Kinematics matrix |
| Observation shape | 5 × 5 (5 vehicles × 5 features) |
| Flattened state size | 25 |
| Action space | 5 discrete actions |
| Episodes | 500 training episodes |
| Transitions stored | Up to 20,000 in replay buffer |

**Observation features (per vehicle):**

| Feature | Description | Type |
|---|---|---|
| `presence` | Whether this vehicle slot is occupied | Float (0 or 1) |
| `x` | Longitudinal position relative to ego car | Float (normalised) |
| `y` | Lateral position relative to ego car | Float (normalised) |
| `vx` | Longitudinal velocity relative to ego car | Float (normalised) |
| `vy` | Lateral velocity relative to ego car | Float (normalised) |

The first row always describes the ego vehicle (the car being controlled). Rows 2–5 describe the four nearest other vehicles. All values are normalised.

**Actions:**

| Index | Action |
|---|---|
| 0 | LANE_LEFT |
| 1 | IDLE |
| 2 | LANE_RIGHT |
| 3 | FASTER |
| 4 | SLOWER |

**Reward function:**

| Component | Value |
|---|---|
| High speed reward | +0.4 per step at target speed |
| Lane change reward | +0.1 |
| Collision penalty | -2.0 (episode ends) |

**Preprocessing:**
The observation matrix is flattened from 5×5 to a 25 element vector before being passed to the network. No further normalisation is applied since `highway-env` normalises observations internally.

---

## Stack

| Library | Role |
|---|---|
| `gymnasium` | Standard RL environment interface |
| `highway-env` | 2D highway driving simulation |
| `torch` (PyTorch) | Neural network definition, training, gradient updates |
| `numpy` | Array operations, replay buffer handling |
| `matplotlib` | Training results dashboard |

---

## How it works

```
Environment (highway-v0)
        │
        │  observation (5×5 matrix, flattened to 25 values)
        ▼
DQN Policy Network (25 → 256 → 256 → split)
        │                            │
   Value stream                Advantage stream
   V(s): 256→128→1             A(s,a): 256→128→5
        │                            │
        └──────────── Q(s,a) ────────┘
                          │
               Epsilon-greedy action selection
                          │
                     Action (0–4)
                          │
                          ▼
               Environment steps forward
                          │
        (reward, next_state, done)
                          │
                    Replay Buffer
                          │
             Sample random mini-batch (64)
                          │
              Double DQN loss (Huber)
                          │
              Backprop + gradient clip
                          │
            Soft update target network (τ=0.005)
```

Training runs for 500 episodes. Every 50 episodes the agent is evaluated greedily across 10 episodes (no random exploration) to measure true policy performance. The best scoring checkpoint is saved to `autonav_model.pth`.

---

## Running it

**Requirements:** Python 3.9+

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/autonav-dqn.git
cd autonav-dqn

# Install dependencies
pip install -r requirements.txt

# Train and run
python autonav_dqn.py
```

No API key, no GPU required. Training 500 episodes takes approximately 5–15 minutes on a standard laptop CPU.

**Expected output during training:**
```
========================================================
  AutoNav — Autonomous Highway Driving Agent (DQN)
========================================================

Environment : highway-v0
State size  : 25  (5 vehicles × 5 features)
Action size : 5   (LANE_LEFT / IDLE / LANE_RIGHT / FASTER / SLOWER)

Training on: cpu
Episodes: 500  |  Max steps: 200  |  Memory: 20,000

Ep   10/500  |  Reward:   8.34  |  10-ep avg:   6.21  |  ε: 0.951  |  Loss: 0.0000
Ep   20/500  |  Reward:  12.41  |  10-ep avg:  10.87  |  ε: 0.905  |  Loss: 0.1243
...
  --- Evaluation at episode 50 ---
  Mean reward   : 14.32 (±3.21)
  Mean survival : 87.4 steps
  Collision rate: 60.0%
...
  --- Evaluation at episode 500 ---
  Mean reward   : 28.91 (±4.12)
  Mean survival : 168.3 steps
  Collision rate: 10.0%
```

After training completes, you are prompted to watch the agent drive in a rendered window.

---

## Results

The agent shows clear improvement over 500 training episodes. Collision rate drops from around 70–80% in early episodes (random exploration) to below 20% by episode 400. Mean survival time increases from under 50 steps to over 150 steps, and episode reward roughly triples from baseline.

The evaluation curve (greedy policy, no exploration) shows a steeper improvement than the raw training curve because the training curve includes early random exploration episodes that drag down the average.

Feature importance is implicit in the network weights, but qualitatively the agent learns to prioritise the `x` and `vx` features of nearby vehicles it reacts to cars directly ahead slowing down and generally prefers the faster lanes when safe.

---

## Limitations and potential improvements

The most significant limitation is training time versus environment complexity. 500 episodes is enough to learn a reasonable policy but not a robust one — the agent still crashes in edge cases like sudden lane merges from multiple vehicles simultaneously. DQN also does not generalise across different traffic densities; a model trained with 30 vehicles struggles when tested with 50.

Things that would improve the agent:

- **More training** — 2,000–5,000 episodes would produce a significantly more robust policy. The current limit keeps runtime practical for the assignment
- **PPO instead of DQN** — Proximal Policy Optimization tends to converge faster and more stably in continuous control tasks; even with discrete actions it often outperforms DQN with less tuning
- **Richer observation** — adding vehicle type, traffic density, or road curvature features would help generalisation
- **Curriculum learning** — start training with fewer vehicles and gradually increase density, rather than throwing the agent into dense traffic from the start
- **CARLA integration** — replacing `highway-env` with CARLA would provide a photorealistic 3D simulation at the cost of significantly higher hardware requirements and setup complexity

---

## References

Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., & Zaremba, W. (2016). *OpenAI Gym*. https://arxiv.org/abs/1606.01540

Farama Foundation. (2023). *highway-env: A collection of environments for autonomous driving and tactical decision-making tasks* [Software]. https://github.com/Farama-Foundation/HighwayEnv

Leurent, E. (2018). *An environment for autonomous driving decision-making*. https://github.com/Farama-Foundation/HighwayEnv

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529–533. https://doi.org/10.1038/nature14236

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*. https://pytorch.org

Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning, 48*, 1995–2003. https://arxiv.org/abs/1511.06581
