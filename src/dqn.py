"""
DQN Neural Network - FIXED VERSION
===================================
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
    DQN Network for single agent
    Input: state_size (e.g., 4: x, y, has_item, distance)
    Output: action_size Q-values (e.g., 4: UP, DOWN, LEFT, RIGHT)
    """
    
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural network layers
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)  # Output Q-values for actions
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: Tensor of shape (batch_size, state_size)
        
        Returns:
            Q-values: Tensor of shape (batch_size, action_size)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation on output (Q-values can be any real number)
    
    def get_action(self, state, epsilon: float = 0.0):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state (numpy array or tensor)
            epsilon: Exploration rate
        
        Returns:
            action: Selected action index (int)
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
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
            states: List or array of agent states (shape: [num_agents, state_size])
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
    assert 0 <= action_greedy < 4, f"Action {action_greedy} out of range!"
    
    # Epsilon-greedy (epsilon=0.5)
    actions = [dqn.get_action(state, epsilon=0.5) for _ in range(10)]
    print(f"   10 actions with ε=0.5: {actions}")
    print(f"   Variety of actions: {len(set(actions))} unique actions")
    assert all(0 <= a < 4 for a in actions), "Some actions out of range!"
    
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
    assert len(actions) == num_agents, "Wrong number of actions!"
    assert all(0 <= a < 4 for a in actions), "Some actions out of range!"
    
    # Independent networks approach
    ma_dqn_independent = MultiAgentDQN(
        num_agents=num_agents,
        state_size_per_agent=state_size,
        action_size=action_size,
        shared_network=False
    )
    print(f"   Independent: {num_agents} separate networks")
    
    actions_ind = ma_dqn_independent.get_actions(states, epsilon=0.3)
    print(f"   Actions (independent): {actions_ind}")
    assert all(0 <= a < 4 for a in actions_ind), "Some actions out of range!"
    
    # Test 4: Batch processing
    print("\n4. Testing Batch Processing:")
    batch_size = 32
    batch_states = torch.randn(batch_size, state_size)
    batch_q_values = dqn(batch_states)
    print(f"   Batch input shape: {batch_states.shape}")
    print(f"   Batch output shape: {batch_q_values.shape}")
    print(f"   Expected: (32, 4) - Got: {batch_q_values.shape}")
    assert batch_q_values.shape == (batch_size, action_size), "Wrong batch output shape!"
    
    print("\n" + "="*50)
    print("✅ DQN Network Tests Complete!")
    print("="*50)


def test_with_environment():
    """Test DQN with actual environment"""
    print("\n" + "="*50)
    print("Testing DQN with Environment")
    print("="*50)
    
    # Import environment
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    try:
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
        
        # Shared network (recommended)
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
        assert all(0 <= a < 4 for a in actions_indices), "Actions out of range!"
        
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
            assert all(0 <= a < 4 for a in actions_indices), f"Invalid actions: {actions_indices}"
            
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
        print("✅ Environment Integration Test Complete!")
        print("="*50)
        
    except ImportError as e:
        print(f"\n⚠️  Couldn't import environment: {e}")
        print("Run from project root to test with environment")


if __name__ == "__main__":
    # Run tests
    test_dqn_network()
    
    # Test with environment
    test_with_environment()
    
    print("\n" + "="*50)
    print("🎉 All DQN Tests Passed!")
    print("="*50)