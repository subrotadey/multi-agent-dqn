import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------
# 1. Environment (Custom GridWorld)
# -------------------------
class GridWorldEnv:
    def __init__(self, grid_size=5, n_agents=4, max_steps=25):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        # Agents start at random positions
        self.agent_positions = [(random.randint(0, self.grid_size-1),
                                 random.randint(0, self.grid_size-1))
                                for _ in range(self.n_agents)]
        self.steps = 0
        self.collisions = 0
        # Delivery points
        self.pickup = (0, 0)
        self.dropoff = (self.grid_size-1, self.grid_size-1)
        return self._get_state()

    def _get_state(self):
        # Flatten agent positions + pickup/dropoff
        state = []
        for pos in self.agent_positions:
            state.extend(pos)
        state.extend(self.pickup)
        state.extend(self.dropoff)
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        # actions: list of agent moves [0=Up,1=Down,2=Left,3=Right]
        rewards = 0
        new_positions = []
        for i, a in enumerate(actions):
            x, y = self.agent_positions[i]
            if a == 0: x = max(0, x-1)
            elif a == 1: x = min(self.grid_size-1, x+1)
            elif a == 2: y = max(0, y-1)
            elif a == 3: y = min(self.grid_size-1, y+1)
            new_positions.append((x,y))

        # Collision check
        if len(new_positions) != len(set(new_positions)):
            self.collisions += 1
            rewards -= 20

        self.agent_positions = new_positions
        self.steps += 1

        # Delivery check (simplified: if any agent reaches dropoff)
        for pos in self.agent_positions:
            if pos == self.dropoff:
                rewards += 10

        # Step penalty
        rewards -= 1

        done = self.steps >= self.max_steps or self.collisions > 0
        return self._get_state(), rewards, done

# -------------------------
# 2. DQN Network
# -------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------
# 3. Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), actions, rewards,
                np.array(next_states), dones)

    def __len__(self):
        return len(self.buffer)

# -------------------------
# 4. Training Loop
# -------------------------
def train_dqn(episodes=4000):
    env = GridWorldEnv()
    state_dim = len(env.reset())
    action_dim = 4  # Up, Down, Left, Right

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    buffer = ReplayBuffer()

    gamma = 0.99
    batch_size = 64
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    update_target_every = 50

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                actions = [random.randint(0, action_dim-1) for _ in range(env.n_agents)]
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state).float())
                    # choose same action for all agents (simplified)
                    action = torch.argmax(q_values).item()
                    actions = [action for _ in range(env.n_agents)]

            next_state, reward, done = env.step(actions)
            buffer.push(state, actions[0], reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states = torch.tensor(states).float()
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards).float()
                next_states = torch.tensor(next_states).float()
                dones = torch.tensor(dones).float()

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = F.mse_loss(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Monitoring
        print(f"Episode {ep}, Reward: {total_reward}, Collisions: {env.collisions}, Steps: {env.steps}")

        # Fail condition check
        if env.collisions > 0 or env.steps > 25:
            print("❌ Failed: Collision or too many steps")
            break

    print("Training finished.")

# -------------------------
# Run Training
# -------------------------
if __name__ == "__main__":
    train_dqn()
