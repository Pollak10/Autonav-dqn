"""
AutoNav — Autonomous Highway Driving Agent
Deep Q-Network (DQN) agent trained on the highway-env simulation environment.

Requirements:
    pip install gymnasium highway-env torch numpy matplotlib

Usage:
    python autonav_dqn.py
"""

import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym
import highway_env  

CONFIG = {
    "episodes":          500,
    "max_steps":         200,
    "batch_size":        64,
    "gamma":             0.99,
    "lr":                1e-3,
    "tau":               0.005,

    "eps_start":         1.0,
    "eps_end":           0.05,
    "eps_decay":         0.995,

    "memory_size":       20_000,
    "min_memory":        1_000,

    "target_update":     10,

    "eval_interval":     50,
    "eval_episodes":     10,

    "lanes":             4,
    "vehicles_count":    30,
    "obs_vehicles":      5,
    "seed":              42,
    "save_path":         "autonav_model.pth",
    "plot_path":         "autonav_results.png",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env(render: bool = False) -> gym.Env:
    """
    Create the highway-env driving environment.

    Observation: 5 x 5 matrix — ego vehicle + 4 nearest vehicles,
                 each described by [presence, x, y, vx, vy].
    Actions:     0=LANE_LEFT  1=IDLE  2=LANE_RIGHT  3=FASTER  4=SLOWER
    """
    render_mode = "human" if render else None
    env = gym.make("highway-v0", render_mode=render_mode)

    env.unwrapped.config.update({
        "lanes_count":           CONFIG["lanes"],
        "vehicles_count":        CONFIG["vehicles_count"],
        "observation": {
            "type":              "Kinematics",
            "vehicles_count":    CONFIG["obs_vehicles"],
            "features":          ["presence", "x", "y", "vx", "vy"],
            "normalize":         True,
            "absolute":          False,
        },
        "action": {
            "type":              "DiscreteMetaAction",
        },
        "duration":              CONFIG["max_steps"],
        "collision_reward":      -2.0,
        "high_speed_reward":      0.4,
        "lane_change_reward":     0.1,
        "reward_speed_range":    [20, 30],
        "simulation_frequency":   15,
        "policy_frequency":        5,
    })

    env.reset(seed=CONFIG["seed"])
    return env


def get_state_size(env: gym.Env) -> int:
    obs, _ = env.reset()
    return int(np.prod(obs.shape))


def get_action_size(env: gym.Env) -> int:
    return env.action_space.n

class ReplayMemory:
    """Fixed-size circular buffer storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer   = []
        self.pos      = 0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """
    Dueling DQN architecture.

    Separates the value stream (how good is this state?) from the
    advantage stream (how much better is each action relative to average?).
    This improves stability in environments where many actions have
    similar value — common in driving scenarios.
    """

    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features  = self.features(x)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class DQNAgent:
    """
    DQN agent with:
      - Epsilon-greedy exploration
      - Experience replay
      - Separate target network (soft updates via tau)
      - Dueling network architecture
    """

    def __init__(self, state_size: int, action_size: int):
        self.state_size  = state_size
        self.action_size = action_size
        self.epsilon     = CONFIG["eps_start"]

        self.policy_net = DQN(state_size, action_size).to(DEVICE)

        self.target_net = DQN(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=CONFIG["lr"])
        self.memory    = ReplayMemory(CONFIG["memory_size"])
        self.steps     = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_t = torch.FloatTensor(state.flatten()).unsqueeze(0).to(DEVICE)
            q_vals  = self.policy_net(state_t)
            return int(q_vals.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(
            state.flatten(), action, reward, next_state.flatten(), done
        )

    def learn(self) -> float | None:
        if len(self.memory) < CONFIG["min_memory"]:
            return None

        batch      = self.memory.sample(CONFIG["batch_size"])
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions)
            target_q     = rewards + CONFIG["gamma"] * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._soft_update()
        self.steps += 1

        return loss.item()

    def _soft_update(self):
        """Blend target network weights toward policy network weights."""
        tau = CONFIG["tau"]
        for target_p, policy_p in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):
            target_p.data.copy_(tau * policy_p.data + (1 - tau) * target_p.data)

    def decay_epsilon(self):
        self.epsilon = max(CONFIG["eps_end"],
                           self.epsilon * CONFIG["eps_decay"])

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)
        print(f"  Model saved to {path}")

    def load(self, path: str):
        self.policy_net.load_state_dict(
            torch.load(path, map_location=DEVICE)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

def evaluate_agent(agent: DQNAgent, env: gym.Env, n_episodes: int) -> dict:
    """Run the agent greedily (no exploration) and return performance stats."""
    rewards, survived, collisions = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps        = 0
        crashed      = False

        for _ in range(CONFIG["max_steps"]):
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps        += 1

            if info.get("crashed", False):
                crashed = True

            if terminated or truncated:
                break

        rewards.append(total_reward)
        survived.append(steps)
        collisions.append(int(crashed))

    return {
        "mean_reward":    np.mean(rewards),
        "std_reward":     np.std(rewards),
        "mean_steps":     np.mean(survived),
        "collision_rate": np.mean(collisions),
    }

def train(agent: DQNAgent, env: gym.Env):
    """Main training loop."""
    episode_rewards  = []
    episode_losses   = []
    eval_rewards     = []
    eval_episodes    = []
    collision_rates  = []

    print(f"Training on: {DEVICE}")
    print(f"Episodes: {CONFIG['episodes']}  |  "
          f"Max steps: {CONFIG['max_steps']}  |  "
          f"Memory: {CONFIG['memory_size']:,}\n")

    best_eval_reward = -np.inf

    for episode in range(1, CONFIG["episodes"] + 1):
        obs, _       = env.reset()
        total_reward = 0.0
        ep_losses    = []

        for _ in range(CONFIG["max_steps"]):
            action                            = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            agent.remember(obs, action, reward, next_obs,
                           float(terminated or truncated))

            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            total_reward += reward
            obs           = next_obs

            if terminated or truncated:
                break

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(ep_losses) if ep_losses else 0.0)

        if episode % 10 == 0:
            recent_mean = np.mean(episode_rewards[-10:])
            print(f"Ep {episode:4d}/{CONFIG['episodes']}  |  "
                  f"Reward: {total_reward:7.2f}  |  "
                  f"10-ep avg: {recent_mean:7.2f}  |  "
                  f"ε: {agent.epsilon:.3f}  |  "
                  f"Loss: {episode_losses[-1]:.4f}")

        if episode % CONFIG["eval_interval"] == 0:
            print(f"\n  --- Evaluation at episode {episode} ---")
            stats = evaluate_agent(agent, env, CONFIG["eval_episodes"])
            eval_rewards.append(stats["mean_reward"])
            eval_episodes.append(episode)
            collision_rates.append(stats["collision_rate"])

            print(f"  Mean reward   : {stats['mean_reward']:.2f} "
                  f"(±{stats['std_reward']:.2f})")
            print(f"  Mean survival : {stats['mean_steps']:.1f} steps")
            print(f"  Collision rate: {stats['collision_rate'] * 100:.1f}%\n")

            if stats["mean_reward"] > best_eval_reward:
                best_eval_reward = stats["mean_reward"]
                agent.save(CONFIG["save_path"])

    return episode_rewards, episode_losses, eval_rewards, eval_episodes, collision_rates

def plot_results(episode_rewards, episode_losses,
                 eval_rewards, eval_episodes, collision_rates):

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("AutoNav — DQN Highway Driving Agent Training Results",
                 fontsize=15, fontweight="bold")

    episodes = np.arange(1, len(episode_rewards) + 1)

    ax = axes[0, 0]
    ax.plot(episodes, episode_rewards, alpha=0.3, color="#1565C0", linewidth=0.8)
    window = 20
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards,
                               np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(episode_rewards) + 1),
                smoothed, color="#1565C0", linewidth=2,
                label=f"{window}-ep moving avg")
    ax.set_title("Episode Reward over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(episodes, episode_losses, alpha=0.4, color="#B71C1C", linewidth=0.8)
    if len(episode_losses) >= window:
        smoothed_loss = np.convolve(episode_losses,
                                    np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(episode_losses) + 1),
                smoothed_loss, color="#B71C1C", linewidth=2,
                label=f"{window}-ep moving avg")
    ax.set_title("Training Loss (Huber)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.plot(eval_episodes, eval_rewards,
            marker="o", color="#2E7D32", linewidth=2, markersize=6)
    ax.set_title("Evaluation Mean Reward (greedy policy)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Reward")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    collision_pct = [r * 100 for r in collision_rates]
    bar_colors    = ["#C62828" if r > 50 else "#F57F17" if r > 20 else "#2E7D32"
                     for r in collision_pct]
    bars = ax.bar(eval_episodes, collision_pct, color=bar_colors,
                  width=CONFIG["eval_interval"] * 0.7, alpha=0.85)
    ax.set_title("Collision Rate at Evaluation Points")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")

    legend_patches = [
        mpatches.Patch(color="#C62828", label="> 50%"),
        mpatches.Patch(color="#F57F17", label="20–50%"),
        mpatches.Patch(color="#2E7D32", label="< 20%"),
    ]
    ax.legend(handles=legend_patches, title="Collision rate", fontsize=9)

    plt.tight_layout()
    plt.savefig(CONFIG["plot_path"], dpi=150, bbox_inches="tight")
    print(f"\nResults chart saved to {CONFIG['plot_path']}")
    plt.show()

def demo(agent: DQNAgent):
    """Load best saved model and watch it drive for 3 episodes."""
    if not os.path.exists(CONFIG["save_path"]):
        print("No saved model found. Train first.")
        return

    agent.load(CONFIG["save_path"])
    agent.epsilon = 0.0

    print("\nWatching trained agent drive (close the window to continue)...")
    env = make_env(render=True)

    for ep in range(1, 4):
        obs, _ = env.reset()
        total  = 0.0

        for _ in range(CONFIG["max_steps"]):
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            env.render()
            if terminated or truncated:
                break

        crashed = "CRASHED" if info.get("crashed") else "survived"
        print(f"  Demo episode {ep}: reward={total:.2f}  ({crashed})")

    env.close()

def main():
    print("\n" + "=" * 56)
    print("  AutoNav — Autonomous Highway Driving Agent (DQN)")
    print("=" * 56 + "\n")

    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    env        = make_env(render=False)
    state_size = get_state_size(env)
    action_size = get_action_size(env)

    print(f"Environment : highway-v0")
    print(f"State size  : {state_size}  ({CONFIG['obs_vehicles']} vehicles × 5 features)")
    print(f"Action size : {action_size}  (LANE_LEFT / IDLE / LANE_RIGHT / FASTER / SLOWER)\n")

    agent = DQNAgent(state_size, action_size)

    (episode_rewards, episode_losses,
     eval_rewards, eval_episodes,
     collision_rates) = train(agent, env)

    env.close()
    print("\n" + "=" * 56)
    print("  Training Complete")
    print("=" * 56)
    print(f"  Best evaluation reward : {max(eval_rewards):.2f}")
    print(f"  Final collision rate   : {collision_rates[-1] * 100:.1f}%")
    print(f"  Final epsilon          : {agent.epsilon:.4f}")

    plot_results(episode_rewards, episode_losses,
                 eval_rewards, eval_episodes, collision_rates)

    answer = input("\nWatch the trained agent drive? (y/n): ").strip().lower()
    if answer == "y":
        demo(agent)

    print("\nDone.")


if __name__ == "__main__":
    main()
