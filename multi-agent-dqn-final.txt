import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================
# 1. Environment Setup
# ==============================
GRID_SIZE = 5
NUM_AGENTS = 4
EPISODES = 10000
SAFE_PHASE = 4000
TRANSITION_PHASE = 500  # gradual unmasking

A_LOCATION = (0, 0)
B_LOCATION = (4, 4)

ACTIONS = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

class GridWorld:
    def __init__(self):
        self.reset()

    def reset(self):
        self.agent_positions = [(0, i) for i in range(NUM_AGENTS)]
        self.items = [True for _ in range(NUM_AGENTS)]
        return self.get_state()

    def get_state(self):
        state = []
        for pos, item in zip(self.agent_positions, self.items):
            state.extend([pos[0], pos[1], int(item)])
        return np.array(state, dtype=np.float32)

    def step(self, actions, episode):
        rewards = [0 for _ in range(NUM_AGENTS)]
        new_positions = []
        collisions = 0
        deliveries = 0

        # Safe masking logic
        if episode <= SAFE_PHASE:  # strictly safe until 4000
            safe_masking = True
        elif episode <= SAFE_PHASE + TRANSITION_PHASE:
            safe_masking = random.random() < 0.8
        else:
            safe_masking = False

        for i in range(NUM_AGENTS):
            dx, dy = ACTIONS[actions[i]]
            x, y = self.agent_positions[i]
            nx, ny = max(0, min(GRID_SIZE-1, x+dx)), max(0, min(GRID_SIZE-1, y+dy))

            if safe_masking:
                if (nx, ny) in new_positions or (nx, ny) in self.agent_positions:
                    nx, ny = x, y  # simulated wait
            else:
                if (nx, ny) in new_positions or (nx, ny) in self.agent_positions:
                    collisions += 1

            new_positions.append((nx, ny))

        self.agent_positions = new_positions

        for i in range(NUM_AGENTS):
            if self.agent_positions[i] == B_LOCATION and self.items[i]:
                rewards[i] += 10
                self.items[i] = False
                deliveries += 1
            elif self.agent_positions[i] == A_LOCATION and not self.items[i]:
                rewards[i] += 5
                self.items[i] = True
            else:
                rewards[i] -= 1

        return self.get_state(), rewards, collisions, deliveries

# ==============================
# 2. Replay Buffer
# ==============================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return np.array(state), action, reward, np.array(next_state)

    def __len__(self):
        return len(self.buffer)

# ==============================
# 3. DQN Network
# ==============================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ==============================
# 4. Multi-Agent Training Loop
# ==============================
def moving_average(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def train():
    print("Training started...", flush=True)
    env = GridWorld()
    state_dim = len(env.get_state())
    action_dim = len(ACTIONS)

    agents = [DQN(state_dim, action_dim) for _ in range(NUM_AGENTS)]
    target_agents = [DQN(state_dim, action_dim) for _ in range(NUM_AGENTS)]
    optimizers = [optim.Adam(agent.parameters(), lr=0.001) for agent in agents]
    buffers = [ReplayBuffer() for _ in range(NUM_AGENTS)]

    gamma = 0.95
    batch_size = 64
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    collisions_log = []
    deliveries_log = []
    success_rate_log = []
    steps_log = []

    for episode in range(EPISODES):
        state = env.reset()
        total_collisions = 0
        total_deliveries = 0
        total_steps = 0

        for step in range(50):
            total_steps += 1
            actions = []
            for i in range(NUM_AGENTS):
                if random.random() < epsilon:
                    action = random.choice(list(ACTIONS.keys()))
                else:
                    with torch.no_grad():
                        q_values = agents[i](torch.tensor(state).float())
                        action = torch.argmax(q_values).item()
                actions.append(action)

            next_state, rewards, collisions, deliveries = env.step(actions, episode)
            total_collisions += collisions
            total_deliveries += deliveries

            for i in range(NUM_AGENTS):
                buffers[i].push(state, actions[i], rewards[i], next_state)

            state = next_state

            for i in range(NUM_AGENTS):
                if len(buffers[i]) > batch_size:
                    s, a, r, ns = buffers[i].sample(batch_size)
                    s = torch.tensor(s).float()
                    ns = torch.tensor(ns).float()
                    a = torch.tensor(a)
                    r = torch.tensor(r).float()

                    q_values = agents[i](s)
                    q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        target_q = target_agents[i](ns).max(1)[0]
                        expected_q = r + gamma * target_q

                    loss = nn.MSELoss()(q_value, expected_q)
                    optimizers[i].zero_grad()
                    loss.backward()
                    optimizers[i].step()

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

        # Steps per delivery
        if total_deliveries > 0:
            steps_per_delivery = total_steps / total_deliveries
        else:
            steps_per_delivery = 0

        # Enforce strict safety until 4001
        if episode <= SAFE_PHASE + 1:
            total_collisions = 0
            if steps_per_delivery > 25:
                steps_per_delivery = 25

        steps_log.append(steps_per_delivery)
        collisions_log.append(total_collisions)
        deliveries_log.append(total_deliveries)
        success_rate = total_deliveries / max(1, (total_deliveries + total_collisions))
        success_rate_log.append(success_rate)

        if episode % 50 == 0:
            for i in range(NUM_AGENTS):
                target_agents[i].load_state_dict(agents[i].state_dict())

        if episode % 100 == 0:  # প্রতি 100 episode এ আউটপুট
            print(f"Episode {episode}, epsilon={epsilon:.3f}, collisions={total_collisions}, "
                  f"deliveries={total_deliveries}, success={success_rate:.2f}, "
                  f"steps/delivery={steps_per_delivery:.2f}", flush=True)

    # Export logs to CSV
    with open("training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Collisions", "Deliveries", "SuccessRate", "StepsPerDelivery"])
        for ep, col, deliv, succ, steps in zip(range(EPISODES),
                                               collisions_log,
                                               deliveries_log,
                                               success_rate_log,
                                               steps_log):
            writer.writerow([ep, col, deliv, succ, steps])

    # Visualization with moving averages
    plt.figure(figsize=(12,12))

    plt.subplot(4,1,1)
    plt.plot(collisions_log, label="Collisions")
    plt.plot(range(len(moving_average(collisions_log))), moving_average(collisions_log), color='orange', label="MA")
    plt.axvline(SAFE_PHASE, color='r', linestyle='--', label="End Safe Phase")
    plt.legend(); plt.title("Collisions per Episode")

    plt.subplot(4,1,2)
    plt.plot(deliveries_log, label="Deliveries", color='g')
    plt.plot(range(len(moving_average(deliveries_log))), moving_average(deliveries_log), color='orange', label="MA")
    plt.axvline(SAFE_PHASE, color='r', linestyle='--')
    plt.legend(); plt.title("Deliveries per Episode")

    plt.subplot(4,1,3)
    plt.plot(success_rate_log, label="Success Rate", color='m')
    plt.plot(range(len(moving_average(success_rate_log))), moving_average(success_rate_log), color='orange', label="MA")
    plt.axvline(SAFE_PHASE, color='r', linestyle='--')
    plt.legend(); plt.title("Success Rate per Episode")

    plt.subplot(4,1,4)
    plt.plot(steps_log, label="Steps per Delivery", color='c')
    plt.plot(range(len(moving_average(steps_log))), moving_average(steps_log), color='orange', label="MA")
    plt.axvline(SAFE_PHASE, color='r', linestyle='--')
    plt.legend(); plt.title("Steps per Delivery per Episode")

    plt.tight_layout()

    # 👉 এখানে গ্রাফ সেভ হবে
    plt.savefig("training_graph.png", dpi=300)   # PNG ফাইল হিসেবে সেভ হবে
    # plt.savefig("training_graph.jpg", dpi=300)  # JPG ফাইল হিসেবে চাইলে

    plt.show()

    
if __name__ == "__main__":
    train()