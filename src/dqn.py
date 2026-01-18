"""
DQN Neural Network - Day 4
===========================
Deep Q-Network implementation using PyTorch
Input: Agent state (position, item, distance)
Output: Q-values for 4 actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DQN(nn.Module):
    """
    Deep Q-Network for Multi-Agent Coordination
    
    Architecture:
    - Input: state_size (e.g., 16 for 4 agents × 4 features)
    - Hidden Layer 1: 128 neurons (ReLU)
    - Hidden Layer 2: 64 neurons (ReLU)
    - Output: action_size (e.g., 4 for UP/DOWN/LEFT/RIGHT)
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list = [128, 64]):
        """
        Initialize DQN network
        
        Args:
            state_size: Dimension of input state
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Input layer to first hidden layer
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        
        # Hidden layers
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        
        # Output layer
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)
        
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        # First hidden layer with ReLU activation
        x = F.relu(self.fc1(state))
        
        # Second hidden layer with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Output layer (no activation - raw Q-values)
        q_values = self.fc3(x)
        
        return q_values
    
    def get_action(self, state, epsilon: float = 0.0):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state (numpy array or tensor)
            epsilon: Exploration rate (0-1)
            
        Returns:
            Selected action index
        """
        # Random exploration
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
        # Exploitation: Choose best action
        with torch.no_grad():
            # Convert state to tensor if needed
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            
            q_values = self.forward(state)
            action = q_values.argmax(dim=1).item()
            
        return action
    
    def save(self, filepath: str):
        """Save model weights"""
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights"""
        self.load_state_dict(torch.load(filepath))
        self.eval()
        print(f"Model loaded from {filepath}")


class MultiAgentDQN:
    """
    Wrapper for managing multiple agents with DQN
    Can use shared network or independent networks
    """
    
    def __init__(self, num_agents: int, state_size_per_agent: int, 
                 action_size: int, shared_network: bool = True):
        """
        Initialize multi-agent DQN
        
        Args:
            num_agents: Number of agents
            state_size_per_agent: State dimension per agent
            action_size: Number of actions
            shared_network: If True, all agents share one network
        """
        self.num_agents = num_agents
        self.state_size_per_agent = state_size_per_agent
        self.action_size = action_size
        self.shared_network = shared_network
        
        if shared_network:
            # Single network for all agents
            self.networks = [DQN(state_size_per_agent, action_size)]
        else:
            # Separate network for each agent
            self.networks = [
                DQN(state_size_per_agent, action_size) 
                for _ in range(num_agents)
            ]
    
    def get_actions(self, states, epsilon: float = 0.0):
        """
        Get actions for all agents
        
        Args:
            states: List or array of agent states
            epsilon: Exploration rate
            
        Returns:
            List of actions for each agent
        """
        actions = []
        
        for i in range(self.num_agents):
            if self.shared_network:
                network = self.networks[0]
            else:
                network = self.networks[i]
            
            # Get state for this agent
            if isinstance(states, np.ndarray):
                agent_state = states[i]
            else:
                agent_state = states[i]
            
            action = network.get_action(agent_state, epsilon)
            actions.append(action)
        
        return actions


# ============================================
# TESTING THE DQN NETWORK
# ============================================

def test_dqn_network():
    """Test DQN network functionality"""
    print("="*50)
    print("Testing DQN Neural Network")
    print("="*50)
    
    # Test 1: Single agent network
    print("\n1. Testing Single Agent DQN:")
    state_size = 4  # [x, y, has_item, distance]
    action_size = 4  # UP, DOWN, LEFT, RIGHT
    
    dqn = DQN(state_size, action_size)
    print(f"   Network created: {state_size} -> [128, 64] -> {action_size}")
    print(f"   Total parameters: {sum(p.numel() for p in dqn.parameters())}")
    
    # Test forward pass
    dummy_state = torch.randn(1, state_size)
    q_values = dqn(dummy_state)
    print(f"   Input shape: {dummy_state.shape}")
    print(f"   Output Q-values shape: {q_values.shape}")
    print(f"   Sample Q-values: {q_values.detach().numpy()[0]}")
    
    # Test 2: Action selection
    print("\n2. Testing Action Selection:")
    state = np.array([2.0, 3.0, 0.0, 5.0])  # [x, y, no_item, distance=5]
    
    # Greedy (epsilon=0)
    action_greedy = dqn.get_action(state, epsilon=0.0)
    print(f"   Greedy action (ε=0.0): {action_greedy}")
    
    # Epsilon-greedy (epsilon=0.5)
    actions = [dqn.get_action(state, epsilon=0.5) for _ in range(10)]
    print(f"   10 actions with ε=0.5: {actions}")
    print(f"   Variety of actions: {len(set(actions))} unique actions")
    
    # Test 3: Multi-agent network
    print("\n3. Testing Multi-Agent DQN:")
    num_agents = 4
    
    # Shared network approach
    ma_dqn_shared = MultiAgentDQN(
        num_agents=num_agents,
        state_size_per_agent=state_size,
        action_size=action_size,
        shared_network=True
    )
    print(f"   Shared network: 1 network for {num_agents} agents")
    
    # Test multi-agent action selection
    states = np.random.randn(num_agents, state_size)
    actions = ma_dqn_shared.get_actions(states, epsilon=0.0)
    print(f"   Actions for all agents: {actions}")
    
    # Independent networks approach
    ma_dqn_independent = MultiAgentDQN(
        num_agents=num_agents,
        state_size_per_agent=state_size,
        action_size=action_size,
        shared_network=False
    )
    print(f"   Independent: {num_agents} separate networks")
    
    # Test 4: Save and load
    print("\n4. Testing Save/Load:")
    import os
    os.makedirs("models", exist_ok=True)
    
    dqn.save("models/test_dqn.pth")
    
    # Load into new network
    dqn2 = DQN(state_size, action_size)
    dqn2.load("models/test_dqn.pth")
    
    # Verify same outputs
    q1 = dqn(dummy_state)
    q2 = dqn2(dummy_state)
    print(f"   Q-values match: {torch.allclose(q1, q2)}")
    
    print("\n" + "="*50)
    print("DQN Network Tests Complete! ✅")
    print("="*50)


def test_with_environment():
    """Test DQN with actual environment"""
    print("\n" + "="*50)
    print("Testing DQN with Environment")
    print("="*50)
    
    # Import environment
    import sys
    sys.path.append('src')
    from environment import MultiAgentGridWorld, Action
    
    # Create environment
    env = MultiAgentGridWorld()
    state = env.reset()
    
    print(f"\nEnvironment state shape: {state.shape}")
    print(f"State:\n{state}")
    
    # Create DQN for single agent
    state_size_per_agent = 4
    action_size = 4
    num_agents = 4
    
    # Option 1: Shared network (recommended)
    print("\n1. Using Shared Network:")
    ma_dqn = MultiAgentDQN(
        num_agents=num_agents,
        state_size_per_agent=state_size_per_agent,
        action_size=action_size,
        shared_network=True
    )
    
    # Get actions from DQN
    actions_indices = ma_dqn.get_actions(state, epsilon=0.5)
    print(f"   DQN action indices: {actions_indices}")
    
    # Convert to Action enum
    action_map = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    actions = [action_map[idx] for idx in actions_indices]
    print(f"   Converted actions: {[a.name for a in actions]}")
    
    # Execute in environment
    next_state, rewards, done, info = env.step(actions)
    print(f"   Rewards: {rewards}")
    print(f"   Info: {info}")
    
    env.render()
    
    # Run a few more steps
    print("\n2. Running 5 steps with DQN:")
    for step in range(5):
        actions_indices = ma_dqn.get_actions(state, epsilon=0.3)
        actions = [action_map[idx] for idx in actions_indices]
        
        next_state, rewards, done, info = env.step(actions)
        
        print(f"\nStep {step + 1}:")
        print(f"   Actions: {[a.name for a in actions]}")
        print(f"   Rewards: {rewards}")
        print(f"   Collisions: {info['total_collisions']}")
        
        state = next_state
        
        if done:
            print("   Episode done!")
            break
    
    env.render()
    
    print("\n" + "="*50)
    print("Environment Integration Test Complete! ✅")
    print("="*50)


if __name__ == "__main__":
    # Run tests
    test_dqn_network()
    
    # Test with environment
    try:
        test_with_environment()
    except ImportError:
        print("\nNote: Run from project root to test with environment")
        print("Command: python src/dqn.py")
    
    print("\n" + "="*50)
    print("Day 4 - DQN Network Complete!")
    print("="*50)
    print("\nNext Steps:")
    print("1. Implement Experience Replay Buffer")
    print("2. Create training loop")
    print("3. Add target network")
    print("="*50)