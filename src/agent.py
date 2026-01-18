"""
DQN Agent - Day 4
=================
Combines DQN network with replay buffer and training logic
Implements epsilon-greedy policy and target network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
import yaml

# Import our custom modules
from dqn import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent with experience replay and target network
    """
    
    def __init__(self, state_size: int, action_size: int, config_path: str = "config.yaml"):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters from config
        self.learning_rate = config['training']['learning_rate']
        self.gamma = config['training']['gamma']
        self.epsilon = config['training']['epsilon_start']
        self.epsilon_end = config['training']['epsilon_end']
        self.epsilon_decay = config['training']['epsilon_decay']
        self.batch_size = config['training']['batch_size']
        
        # Device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Import DQN and ReplayBuffer
        from dqn import DQN
        from agent import ReplayBuffer
        
        # Main network (policy network)
        self.policy_net = DQN(state_size, action_size).to(self.device)
        
        # Target network (for stable training)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is not trained directly
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer
        memory_size = config['training']['memory_size']
        self.memory = ReplayBuffer(capacity=memory_size)
        
        # Training statistics
        self.training_step = 0
        self.losses = []
        
    def select_action(self, state, explore=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            explore: Whether to use epsilon-greedy (False = greedy only)
            
        Returns:
            Selected action index
        """
        # Exploration
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step (if enough experiences collected)
        
        Returns:
            Loss value or None if not ready to train
        """
        # Check if enough experiences
        if not self.memory.is_ready(self.batch_size):
            return None
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q-values (from policy network)
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Next Q-values (from target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # Target: r + γ * max Q(s', a') if not done, else just r
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Track loss
        self.losses.append(loss.item())
        self.training_step += 1
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Save agent state"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'losses': self.losses
        }, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.losses = checkpoint['losses']
        print(f"Agent loaded from {filepath}")


class MultiAgentDQNSystem:
    """
    Multi-Agent DQN System
    Manages multiple agents with shared or independent networks
    """
    
    def __init__(self, num_agents: int, state_size_per_agent: int, 
                 action_size: int, shared_network: bool = True,
                 config_path: str = "config.yaml"):
        """
        Initialize multi-agent system
        
        Args:
            num_agents: Number of agents
            state_size_per_agent: State dimension per agent
            action_size: Number of actions
            shared_network: Whether agents share one network
            config_path: Configuration file path
        """
        self.num_agents = num_agents
        self.state_size_per_agent = state_size_per_agent
        self.action_size = action_size
        self.shared_network = shared_network
        
        if shared_network:
            # Single agent for all
            self.agents = [DQNAgent(state_size_per_agent, action_size, config_path)]
        else:
            # Separate agent for each
            self.agents = [
                DQNAgent(state_size_per_agent, action_size, config_path)
                for _ in range(num_agents)
            ]
    
    def select_actions(self, states, explore=True):
        """
        Select actions for all agents
        
        Args:
            states: States for all agents (array of shape [num_agents, state_size])
            explore: Whether to explore
            
        Returns:
            List of actions
        """
        actions = []
        
        for i in range(self.num_agents):
            if self.shared_network:
                agent = self.agents[0]
            else:
                agent = self.agents[i]
            
            action = agent.select_action(states[i], explore)
            actions.append(action)
        
        return actions
    
    def store_experiences(self, states, actions, rewards, next_states, done):
        """Store experiences for all agents"""
        if self.shared_network:
            # Store all in one buffer
            for i in range(self.num_agents):
                self.agents[0].store_experience(
                    states[i], actions[i], rewards[i], next_states[i], done
                )
        else:
            # Store in separate buffers
            for i in range(self.num_agents):
                self.agents[i].store_experience(
                    states[i], actions[i], rewards[i], next_states[i], done
                )
    
    def train_step(self):
        """Train all agents"""
        losses = []
        
        if self.shared_network:
            loss = self.agents[0].train_step()
            if loss is not None:
                losses.append(loss)
        else:
            for agent in self.agents:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
        
        return np.mean(losses) if losses else None
    
    def update_target_networks(self):
        """Update target networks for all agents"""
        for agent in self.agents:
            agent.update_target_network()
    
    def decay_epsilon(self):
        """Decay epsilon for all agents"""
        for agent in self.agents:
            agent.decay_epsilon()
    
    def save(self, filepath_prefix: str):
        """Save all agents"""
        if self.shared_network:
            self.agents[0].save(f"{filepath_prefix}_shared.pth")
        else:
            for i, agent in enumerate(self.agents):
                agent.save(f"{filepath_prefix}_agent{i}.pth")


# ============================================
# TESTING THE AGENT
# ============================================

def test_single_agent():
    """Test single DQN agent"""
    print("="*50)
    print("Testing Single DQN Agent")
    print("="*50)
    
    # Create agent
    state_size = 4
    action_size = 4
    
    agent = DQNAgent(state_size, action_size)
    
    print(f"\n1. Agent Initialized:")
    print(f"   State size: {state_size}")
    print(f"   Action size: {action_size}")
    print(f"   Epsilon: {agent.epsilon}")
    print(f"   Learning rate: {agent.learning_rate}")
    print(f"   Gamma: {agent.gamma}")
    
    # Test action selection
    print("\n2. Testing Action Selection:")
    state = np.random.randn(state_size)
    
    # Greedy action
    action_greedy = agent.select_action(state, explore=False)
    print(f"   Greedy action: {action_greedy}")
    
    # Epsilon-greedy actions
    actions = [agent.select_action(state, explore=True) for _ in range(10)]
    print(f"   10 epsilon-greedy actions: {actions}")
    print(f"   Unique actions: {len(set(actions))}")
    
    # Store some experiences
    print("\n3. Collecting Experiences:")
    for i in range(100):
        state = np.random.randn(state_size)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(state_size)
        done = (i % 20 == 19)
        
        agent.store_experience(state, action, reward, next_state, done)
    
    print(f"   Experiences collected: {len(agent.memory)}")
    
    # Train
    print("\n4. Training:")
    for step in range(10):
        loss = agent.train_step()
        if loss:
            print(f"   Step {step+1}: Loss = {loss:.4f}")
    
    print(f"   Total training steps: {agent.training_step}")
    print(f"   Current epsilon: {agent.epsilon:.4f}")
    
    # Decay epsilon
    print("\n5. Epsilon Decay:")
    for _ in range(10):
        agent.decay_epsilon()
    print(f"   Epsilon after 10 decays: {agent.epsilon:.4f}")
    
    # Update target network
    print("\n6. Target Network Update:")
    agent.update_target_network()
    print("   Target network updated ✅")
    
    print("\n" + "="*50)
    print("Single Agent Tests Complete! ✅")
    print("="*50)


def test_multi_agent_system():
    """Test multi-agent DQN system"""
    print("\n" + "="*50)
    print("Testing Multi-Agent DQN System")
    print("="*50)
    
    num_agents = 4
    state_size = 4
    action_size = 4
    
    # Shared network
    print("\n1. Shared Network System:")
    ma_system = MultiAgentDQNSystem(
        num_agents=num_agents,
        state_size_per_agent=state_size,
        action_size=action_size,
        shared_network=True
    )
    
    print(f"   Number of networks: {len(ma_system.agents)}")
    
    # Select actions
    states = np.random.randn(num_agents, state_size)
    actions = ma_system.select_actions(states, explore=True)
    print(f"   Actions selected: {actions}")
    
    # Store experiences and train
    print("\n2. Training Multi-Agent System:")
    for episode in range(5):
        states = np.random.randn(num_agents, state_size)
        actions = ma_system.select_actions(states)
        rewards = np.random.randn(num_agents)
        next_states = np.random.randn(num_agents, state_size)
        done = False
        
        ma_system.store_experiences(states, actions, rewards, next_states, done)
        
        loss = ma_system.train_step()
        if loss:
            print(f"   Episode {episode+1}: Loss = {loss:.4f}")
    
    print("\n" + "="*50)
    print("Multi-Agent System Tests Complete! ✅")
    print("="*50)


if __name__ == "__main__":
    try:
        test_single_agent()
        test_multi_agent_system()
        
        print("\n" + "="*50)
        print("Day 4 - Agent Implementation Complete!")
        print("="*50)
        print("\nNext Steps for Day 5:")
        print("1. Create complete training loop")
        print("2. Integrate with environment")
        print("3. Add logging and visualization")
        print("4. Test with actual multi-agent scenarios")
        print("="*50)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to:")
        print("1. Have config.yaml in root directory")
        print("2. Have dqn.py and replay buffer ready")
        print("3. Run from project root")