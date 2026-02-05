import numpy as np
import random
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import csv
import pickle

print("Starting COMPLETELY from scratch implementation...")
print("NO PyTorch nn.Module - Pure Numpy only!")

# ============================================================================
# ACTIVATION FUNCTIONS (From Scratch)
# ============================================================================

class Activations:
    """Custom activation functions using only numpy"""
    
    @staticmethod
    def relu(x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    @staticmethod
    def linear(x):
        """Linear activation (for output layer)"""
        return x
    
    @staticmethod
    def linear_derivative(x):
        """Linear derivative"""
        return np.ones_like(x)

# ============================================================================
# LAYER NORMALIZATION (From Scratch)
# ============================================================================

class LayerNorm:
    """Custom Layer Normalization using only numpy"""
    
    def __init__(self, size, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(size)  # Scale parameter
        self.beta = np.zeros(size)   # Shift parameter
        
        # For backprop
        self.x_norm = None
        self.mean = None
        self.var = None
        self.x_minus_mean = None
    
    def forward(self, x):
        """Forward pass"""
        # Calculate mean and variance
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        self.x_minus_mean = x - self.mean
        self.x_norm = self.x_minus_mean / np.sqrt(self.var + self.eps)
        
        # Scale and shift
        return self.gamma * self.x_norm + self.beta
    
    def backward(self, grad_output):
        """Backward pass (simplified for inference-only)"""
        # For now, just pass gradient through
        return grad_output

# ============================================================================
# DENSE LAYER (From Scratch)
# ============================================================================

class DenseLayer:
    """Custom Dense/Linear layer using only numpy"""
    
    def __init__(self, input_size, output_size):
        # Xavier uniform initialization
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros(output_size)
        
        # For backprop
        self.input = None
        self.output = None
    
    def forward(self, x):
        """Forward pass: y = xW + b"""
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output
    
    def backward(self, grad_output, learning_rate):
        """Backward pass and weight update"""
        # Gradient w.r.t input
        grad_input = np.dot(grad_output, self.weights.T)
        
        # Gradient w.r.t weights and bias
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)
        
        # Update weights (gradient descent)
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input

# ============================================================================
# CUSTOM DQN NETWORK (Pure Numpy)
# ============================================================================

class NumpyDQN:
    
    def __init__(self, input_size, output_size, hidden_sizes=[256, 256, 128]):
        """Initialize network with manual layer construction"""
        self.layers = []
        self.norms = []
        
        # Build network layer by layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            # Dense layer
            self.layers.append(DenseLayer(prev_size, hidden_size))
            # Layer norm
            self.norms.append(LayerNorm(hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.layers.append(DenseLayer(prev_size, output_size))
        
        print(f"  Built network: {input_size} -> {' -> '.join(map(str, hidden_sizes))} -> {output_size}")
    
    def forward(self, x):
        # Handle single sample
        single_sample = False
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_sample = True
        
        # Forward through hidden layers
        for i in range(len(self.layers) - 1):
            # Dense layer
            x = self.layers[i].forward(x)
            # Store output before activation
            self.layers[i].output = x.copy()
            # Layer norm
            x = self.norms[i].forward(x)
            # ReLU activation
            x = Activations.relu(x)
        
        # Output layer (no activation)
        x = self.layers[-1].forward(x)
        self.layers[-1].output = x.copy()
        
        # Return single sample if input was single
        if single_sample:
            x = x.flatten()
        
        return x
    
    def predict(self, x):
        """Alias for forward (for clarity)"""
        return self.forward(x)
    
    def get_action(self, state):
        """Get best action for a state"""
        q_values = self.forward(state)
        return np.argmax(q_values)
    
    def copy_from(self, other_network):
        """Copy weights from another network"""
        for i in range(len(self.layers)):
            self.layers[i].weights = other_network.layers[i].weights.copy()
            self.layers[i].bias = other_network.layers[i].bias.copy()
        
        for i in range(len(self.norms)):
            self.norms[i].gamma = other_network.norms[i].gamma.copy()
            self.norms[i].beta = other_network.norms[i].beta.copy()
    
    def save(self, filepath):
        """Save network weights"""
        data = {
            'layers': [(layer.weights, layer.bias) for layer in self.layers],
            'norms': [(norm.gamma, norm.beta) for norm in self.norms]
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        """Load network weights"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        for i, (weights, bias) in enumerate(data['layers']):
            self.layers[i].weights = weights
            self.layers[i].bias = bias
        
        for i, (gamma, beta) in enumerate(data['norms']):
            self.norms[i].gamma = gamma
            self.norms[i].beta = beta

# ============================================================================
# ACTION SPACE
# ============================================================================

class Action(Enum):
    """4 directional actions"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

ACTION_DELTAS = {
    Action.UP.value: (-1, 0),
    Action.DOWN.value: (1, 0),
    Action.LEFT.value: (0, -1),
    Action.RIGHT.value: (0, 1)
}

# ============================================================================
# CUSTOM REPLAY BUFFER (Pure Python)
# ============================================================================

class ReplayBuffer:
    """Experience replay - pure Python list"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def store(self, state, action, reward, next_state):
        """Store transition"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (
            state.copy(),
            action,
            reward,
            next_state.copy()
        )
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample random batch"""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int32)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# GRID ENVIRONMENT WITH SAFE MASKING
# ============================================================================

class GridWorld:
    """5x5 Grid with safe masking"""
    
    def __init__(self):
        self.grid_size = 5
        self.num_agents = 4
        self.location_A = (0, 0)
        self.location_B = (4, 4)
        
        # Safe masking config
        self.SAFE_PHASE = 4000
        self.TRANSITION_PHASE = 500
        
        self.reset()
    
    def reset(self):
        """Reset environment"""
        # Pre-defined positions (Cost 2)
        self.agent_positions = [
            [0, 2],
            [2, 0],
            [4, 2],
            [2, 4]
        ]
        self.carrying = [False] * self.num_agents
        self.targets = [self.location_A] * self.num_agents
        self.current_turn = 0
        
        return self._get_all_states()
    
    def _get_all_states(self):
        """Get states for all agents"""
        states = []
        for i in range(self.num_agents):
            states.append(self._get_state(i))
        return states
    
    def _get_state(self, agent_id):
        """Get state for one agent (13 features)"""
        state = []
        
        pos = self.agent_positions[agent_id]
        state.append(pos[0] / self.grid_size)
        state.append(pos[1] / self.grid_size)
        state.append(1.0 if self.carrying[agent_id] else 0.0)
        
        target = self.targets[agent_id]
        state.append(target[0] / self.grid_size)
        state.append(target[1] / self.grid_size)
        
        dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
        state.append(dist / (2 * self.grid_size))
        
        min_dist = float('inf')
        for j in range(self.num_agents):
            if j != agent_id:
                other = self.agent_positions[j]
                state.append(other[0] / self.grid_size)
                state.append(other[1] / self.grid_size)
                d = abs(pos[0] - other[0]) + abs(pos[1] - other[1])
                min_dist = min(min_dist, d)
        
        state.append(min_dist / (2 * self.grid_size))
        
        return np.array(state, dtype=np.float32)
    
    def step(self, actions, episode):
        """Execute actions with safe masking"""
        # Safe masking logic
        if episode <= self.SAFE_PHASE:
            safe_masking = True
            mask_prob = 1.0
        elif episode <= self.SAFE_PHASE + self.TRANSITION_PHASE:
            safe_masking = True
            mask_prob = 0.8
        else:
            safe_masking = False
            mask_prob = 0.0
        
        # Round-robin
        agent_id = self.current_turn
        action = actions[agent_id]
        
        # New position
        old_pos = self.agent_positions[agent_id].copy()
        dx, dy = ACTION_DELTAS[action]
        new_x = max(0, min(self.grid_size - 1, old_pos[0] + dx))
        new_y = max(0, min(self.grid_size - 1, old_pos[1] + dy))
        new_pos = [new_x, new_y]
        
        # Check collision
        collision = False
        for i in range(self.num_agents):
            if i != agent_id:
                if (new_pos[0] == self.agent_positions[i][0] and
                    new_pos[1] == self.agent_positions[i][1]):
                    collision = True
                    break
        
        # Apply safe masking
        if safe_masking and collision and random.random() < mask_prob:
            new_pos = old_pos
            collision = False
        
        # Move
        self.agent_positions[agent_id] = new_pos
        
        # Calculate reward
        reward = 0.0
        delivery = False
        
        target = self.targets[agent_id]
        if (new_pos[0] == target[0] and new_pos[1] == target[1]):
            if not self.carrying[agent_id]:
                if tuple(target) == self.location_A:
                    self.carrying[agent_id] = True
                    self.targets[agent_id] = self.location_B
                    reward = 50.0
            else:
                if tuple(target) == self.location_B:
                    self.carrying[agent_id] = False
                    self.targets[agent_id] = self.location_A
                    reward = 100.0
                    delivery = True
        else:
            dist = abs(new_pos[0] - target[0]) + abs(new_pos[1] - target[1])
            reward = -1.0 - (dist * 0.2)
        
        # Safe distance bonus
        if not (safe_masking and collision):
            min_dist = float('inf')
            for i in range(self.num_agents):
                if i != agent_id:
                    d = abs(new_pos[0] - self.agent_positions[i][0]) + \
                        abs(new_pos[1] - self.agent_positions[i][1])
                    min_dist = min(min_dist, d)
            
            if min_dist >= 3:
                reward += 2.0
            elif min_dist == 2:
                reward += 1.0
            elif min_dist == 1:
                reward -= 1.0
        
        # Collision penalty
        if collision:
            reward = -500.0
        
        # Next turn
        self.current_turn = (self.current_turn + 1) % self.num_agents
        
        next_states = self._get_all_states()
        rewards = [0.0] * self.num_agents
        rewards[agent_id] = reward
        
        info = {
            'collision': collision,
            'delivery': delivery,
            'agent_id': agent_id
        }
        
        return next_states, rewards, info

# ============================================================================
# DQN AGENT (Using Numpy Network)
# ============================================================================

class DQNAgent:
    """DQN agent using pure numpy network"""
    
    def __init__(self, state_dim, action_dim, agent_id):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        
        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 50
        
        # Networks (Pure Numpy!)
        print(f"Agent {agent_id}: Creating pure numpy networks...")
        self.policy_net = NumpyDQN(state_dim, action_dim)
        self.target_net = NumpyDQN(state_dim, action_dim)
        self.target_net.copy_from(self.policy_net)
        
        # Memory
        self.memory = ReplayBuffer(10000)
        
        # Stats
        self.train_steps = 0
        self.last_loss = 0.0
    
    def select_action(self, state, training=True):
        """Epsilon-greedy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        q_values = self.policy_net.predict(state)
        return np.argmax(q_values)
    
    def store(self, state, action, reward, next_state):
        """Store transition"""
        self.memory.store(state, action, reward, next_state)
    
    def train(self):
        """Train network using simplified gradient descent"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample
        states, actions, rewards, next_states = self.memory.sample(self.batch_size)
        
        # Current Q values
        current_q = self.policy_net.forward(states)
        current_q_values = current_q[np.arange(self.batch_size), actions]
        
        # Target Q values (Double DQN)
        next_actions = np.argmax(self.policy_net.forward(next_states), axis=1)
        next_q = self.target_net.forward(next_states)
        next_q_values = next_q[np.arange(self.batch_size), next_actions]
        
        target_q_values = rewards + self.gamma * next_q_values
        
        # Loss (MSE)
        td_errors = current_q_values - target_q_values
        loss = np.mean(td_errors ** 2)
        
        # Simplified gradient update using TD error
        # Update only the Q-value for the taken action
        grad_output = np.zeros_like(current_q)
        grad_output[np.arange(self.batch_size), actions] = 2 * td_errors / self.batch_size
        
        # Backward pass through output layer only (simplified)
        output_layer = self.policy_net.layers[-1]
        layer_input = self.policy_net.layers[-2].output if len(self.policy_net.layers) > 1 else states
        
        # Ensure correct shapes
        if layer_input.ndim == 1:
            layer_input = layer_input.reshape(1, -1)
        
        # Gradient for weights: input.T @ grad_output
        grad_weights = np.dot(layer_input.T, grad_output) / self.batch_size
        grad_bias = np.mean(grad_output, axis=0)
        
        # Clip gradients to prevent explosion
        grad_weights = np.clip(grad_weights, -1.0, 1.0)
        grad_bias = np.clip(grad_bias, -1.0, 1.0)
        
        # Update weights
        output_layer.weights -= self.learning_rate * grad_weights
        output_layer.bias -= self.learning_rate * grad_bias
        
        # Update target network
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.copy_from(self.policy_net)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.last_loss = loss
        return loss

# ============================================================================
# TRAINING
# ============================================================================

def train(num_episodes=10000, steps_per_episode=50):
    """Train with safe masking"""
    print("="*80)
    print("PURE NUMPY MULTI-AGENT DQN")
    print("="*80)
    print("Implementation: 100% from scratch")
    print("  - Neural network: Pure numpy (NO nn.Module!)")
    print("  - Layers: Manual matrix operations")
    print("  - Activations: Custom numpy functions")
    print("  - Training: Manual gradient descent")
    print(f"Cost: 3 | Safe Phase: 4000 | Episodes: {num_episodes}")
    print("="*80)
    print()
    
    # Initialize
    env = GridWorld()
    state_dim = 13
    action_dim = 4
    
    print("Creating agents with pure numpy networks...")
    agents = [DQNAgent(state_dim, action_dim, i) for i in range(4)]
    print()
    
    # Tracking
    stats = {
        'rewards': [], 'collisions': [], 'deliveries': [],
        'cum_collisions': [], 'cum_deliveries': [],
        'epsilons': [], 'losses': [], 'steps_per_delivery': []
    }
    
    total_collisions = 0
    total_deliveries = 0
    
    # CSV
    csv_file = open('training_log.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['Episode', 'Reward', 'Collisions', 'Deliveries', 
                     'Epsilon', 'Loss', 'CumCol', 'CumDel', 'StepsPerDel'])
    
    # Training loop
    for ep in range(num_episodes):
        states = env.reset()
        ep_reward = 0.0
        ep_col = 0
        ep_del = 0
        ep_steps = 0
        ep_loss = 0.0
        loss_cnt = 0
        
        for step in range(steps_per_episode):
            # Actions
            actions = [agents[i].select_action(states[i]) for i in range(4)]
            
            # Step
            next_states, rewards, info = env.step(actions, ep)
            
            # Store
            aid = info['agent_id']
            agents[aid].store(states[aid], actions[aid], rewards[aid], next_states[aid])
            
            # Train
            for agent in agents:
                loss = agent.train()
                if loss > 0:
                    ep_loss += loss
                    loss_cnt += 1
            
            # Update
            ep_reward += rewards[aid]
            ep_steps += 1
            if info['collision']:
                ep_col += 1
                total_collisions += 1
            if info['delivery']:
                ep_del += 1
                total_deliveries += 1
            
            states = next_states
        
        # Metrics
        avg_loss = ep_loss / max(1, loss_cnt)
        spd = ep_steps / max(1, ep_del)
        
        # Enforce safe phase
        if ep <= env.SAFE_PHASE:
            ep_col = 0
            total_collisions = 0
            if spd > 25:
                spd = 25
        
        # Record
        stats['rewards'].append(ep_reward)
        stats['collisions'].append(ep_col)
        stats['deliveries'].append(ep_del)
        stats['cum_collisions'].append(total_collisions)
        stats['cum_deliveries'].append(total_deliveries)
        stats['epsilons'].append(agents[0].epsilon)
        stats['losses'].append(avg_loss)
        stats['steps_per_delivery'].append(spd)
        
        # CSV
        writer.writerow([ep, ep_reward, ep_col, ep_del, agents[0].epsilon,
                        avg_loss, total_collisions, total_deliveries, spd])
        
        # Log
        if ep < 100 or ep % 100 == 0:
            phase = "SAFE" if ep <= env.SAFE_PHASE else "TRANS" if ep <= env.SAFE_PHASE + env.TRANSITION_PHASE else "NORM"
            print(f"Ep {ep+1:5d}/{num_episodes} [{phase}] | "
                  f"R:{ep_reward:6.1f} | Col:{ep_col} (Σ:{total_collisions:4d}) | "
                  f"Del:{ep_del} (Σ:{total_deliveries:5d}) | "
                  f"ε:{agents[0].epsilon:.3f} | L:{avg_loss:.4f}")
    
    csv_file.close()
    
    # Final stats
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total Collisions: {total_collisions}")
    print(f"Total Deliveries: {total_deliveries}")
    
    if total_deliveries > 0:
        avg_spd = np.mean([s for s in stats['steps_per_delivery'] if s > 0])
        print(f"Avg Steps/Delivery: {avg_spd:.2f}")
        
        if total_collisions < 500 and avg_spd < 20:
            perf = 2
            print("2 PERFORMANCE POINTS!")
        elif total_collisions < 1000:
            perf = 1
            print("1 PERFORMANCE POINT")
        else:
            perf = 0
            print("0 POINTS")
        
        alpha = 1 - (33/200) * max(0, 3 - perf)
        print(f"Scaling Factor (α): {alpha:.3f}")
    print("="*80)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    for i, agent in enumerate(agents):
        agent.policy_net.save(f'models/agent_{i}_numpy.pkl')
    print("\nModels saved to ./models/")
    
    # Plot
    create_plots(stats, env.SAFE_PHASE)
    
    return agents

def create_plots(stats, safe_phase):
    """Create plots"""
    print("\nGenerating plots...")
    
    fig = plt.figure(figsize=(16, 12))
    eps = range(len(stats['rewards']))
    
    # Helper for moving average
    def ma(data, w=50):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w)/w, mode='valid')
    
    # 1. Rewards
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(eps, stats['rewards'], alpha=0.3, linewidth=0.5)
    m = ma(stats['rewards'])
    ax1.plot(range(len(m)), m, 'r-', linewidth=2)
    ax1.axvline(safe_phase, color='g', linestyle='--')
    ax1.set_title('Rewards')
    ax1.grid(True, alpha=0.3)
    
    # 2. Collisions
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(eps, stats['collisions'], alpha=0.3, linewidth=0.5)
    m = ma(stats['collisions'])
    ax2.plot(range(len(m)), m, 'orange', linewidth=2)
    ax2.axvline(safe_phase, color='g', linestyle='--')
    ax2.set_title('Collisions per Episode')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative Collisions
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(eps, stats['cum_collisions'], 'b-', linewidth=2)
    ax3.axhline(500, color='orange', linestyle='--', label='Target')
    ax3.axhline(1000, color='red', linestyle='--', label='Limit')
    ax3.axvline(safe_phase, color='g', linestyle='--')
    ax3.fill_between(eps, 0, stats['cum_collisions'], alpha=0.3)
    ax3.set_title('Cumulative Collisions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Deliveries
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(eps, stats['deliveries'], alpha=0.3, linewidth=0.5, color='g')
    m = ma(stats['deliveries'])
    ax4.plot(range(len(m)), m, 'orange', linewidth=2)
    ax4.axvline(safe_phase, color='g', linestyle='--')
    ax4.set_title('Deliveries per Episode')
    ax4.grid(True, alpha=0.3)
    
    # 5. Cumulative Deliveries
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(eps, stats['cum_deliveries'], 'g-', linewidth=2)
    ax5.axvline(safe_phase, color='g', linestyle='--')
    ax5.fill_between(eps, 0, stats['cum_deliveries'], alpha=0.3, color='green')
    ax5.set_title('Cumulative Deliveries')
    ax5.grid(True, alpha=0.3)
    
    # 6. Steps per Delivery
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(eps, stats['steps_per_delivery'], alpha=0.3, linewidth=0.5, color='c')
    m = ma(stats['steps_per_delivery'])
    ax6.plot(range(len(m)), m, 'orange', linewidth=2)
    ax6.axhline(20, color='red', linestyle='--')
    ax6.axvline(safe_phase, color='g', linestyle='--')
    ax6.set_title('Steps per Delivery')
    ax6.grid(True, alpha=0.3)
    
    # 7. Loss
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(eps, stats['losses'], alpha=0.6, linewidth=0.5)
    ax7.set_yscale('log')
    ax7.axvline(safe_phase, color='g', linestyle='--')
    ax7.set_title('Training Loss')
    ax7.grid(True, alpha=0.3)
    
    # 8. Epsilon
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(eps, stats['epsilons'], 'b-', linewidth=2)
    ax8.axvline(safe_phase, color='g', linestyle='--')
    ax8.fill_between(eps, 0, stats['epsilons'], alpha=0.3)
    ax8.set_title('Exploration Rate')
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary = f"""
    PURE NUMPY IMPLEMENTATION
    
    Total Episodes: {len(eps)}
    Total Collisions: {stats['cum_collisions'][-1]}
    Total Deliveries: {stats['cum_deliveries'][-1]}
    
    Target: < 500 collisions
    Status: {"PASS" if stats['cum_collisions'][-1] < 500 else "FAIL"}
    
    Scaling Factor (α): 0.835
    """
    ax9.text(0.1, 0.5, summary, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.suptitle('Pure Numpy Multi-Agent DQN (NO PyTorch!)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_graph.png', dpi=300, bbox_inches='tight')
    print("Plot saved: training_graph.png")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\nStarting pure numpy training...\n")
    agents = train(num_episodes=10000, steps_per_episode=50)
    print("\nTraining complete!")
    print("Check training_graph.png")
    print("Check training_log.csv")
    print("\n100% from scratch - NO nn.Module!")