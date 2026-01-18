"""
Experience Replay Buffer - Day 4
=================================
Stores and samples past experiences for DQN training
Breaks temporal correlation in training data
"""

import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple


# Named tuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN
    
    Stores transitions: (state, action, reward, next_state, done)
    Samples random mini-batches for training
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode is done
        """
        # Convert to numpy arrays for consistency
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)
        
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample random mini-batch from buffer
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        # Sample random experiences
        experiences = random.sample(self.buffer, batch_size)
        
        # Separate into components
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling"""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear all experiences from buffer"""
        self.buffer.clear()


class MultiAgentReplayBuffer:
    """
    Replay Buffer for Multi-Agent systems
    Can store either shared experiences or separate per agent
    """
    
    def __init__(self, num_agents: int, capacity: int = 10000, shared: bool = True):
        """
        Initialize multi-agent replay buffer
        
        Args:
            num_agents: Number of agents
            capacity: Buffer capacity (total or per agent)
            shared: If True, one shared buffer; if False, separate buffers
        """
        self.num_agents = num_agents
        self.shared = shared
        
        if shared:
            # Single shared buffer for all agents
            self.buffers = [ReplayBuffer(capacity)]
        else:
            # Separate buffer for each agent
            self.buffers = [ReplayBuffer(capacity) for _ in range(num_agents)]
    
    def push(self, states, actions, rewards, next_states, done):
        """
        Add multi-agent experience
        
        Args:
            states: States for all agents (list or array)
            actions: Actions for all agents
            rewards: Rewards for all agents
            next_states: Next states for all agents
            done: Episode done flag
        """
        if self.shared:
            # Store as single experience with all agent data
            self.buffers[0].push(states, actions, rewards, next_states, done)
        else:
            # Store separate experience for each agent
            for i in range(self.num_agents):
                self.buffers[i].push(
                    states[i], 
                    actions[i], 
                    rewards[i], 
                    next_states[i], 
                    done
                )
    
    def sample(self, batch_size: int, agent_id: int = 0):
        """
        Sample experiences
        
        Args:
            batch_size: Number of experiences to sample
            agent_id: Which agent's buffer to sample (if not shared)
            
        Returns:
            Sampled batch
        """
        if self.shared:
            return self.buffers[0].sample(batch_size)
        else:
            return self.buffers[agent_id].sample(batch_size)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer(s) ready for sampling"""
        if self.shared:
            return self.buffers[0].is_ready(batch_size)
        else:
            return all(buf.is_ready(batch_size) for buf in self.buffers)
    
    def __len__(self):
        """Return total experiences stored"""
        return sum(len(buf) for buf in self.buffers)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (Optional - Advanced)
    Samples experiences based on TD error priority
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        """
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
    
    def push(self, state, action, reward, next_state, done, priority: float = None):
        """
        Add experience with priority
        
        Args:
            priority: TD error or initial priority
        """
        super().push(state, action, reward, next_state, done)
        
        # Set max priority for new experience
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(priority if priority else max_priority)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample based on priorities with importance sampling
        
        Args:
            batch_size: Number of samples
            beta: Importance sampling weight (0 to 1)
            
        Returns:
            Sampled batch with importance weights and indices
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        # Separate components
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


# ============================================
# TESTING THE REPLAY BUFFER
# ============================================

def test_basic_replay_buffer():
    """Test basic replay buffer functionality"""
    print("="*50)
    print("Testing Basic Replay Buffer")
    print("="*50)
    
    # Create buffer
    buffer = ReplayBuffer(capacity=100)
    print(f"\n1. Buffer created with capacity: {buffer.capacity}")
    
    # Add some experiences
    print("\n2. Adding experiences:")
    for i in range(10):
        state = np.random.randn(4)
        action = np.random.randint(0, 4)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = (i == 9)  # Last one is terminal
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"   Buffer size: {len(buffer)}")
    print(f"   Is ready for batch_size=5? {buffer.is_ready(5)}")
    
    # Sample batch
    print("\n3. Sampling batch:")
    batch_size = 5
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    print(f"   States shape: {states.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Rewards shape: {rewards.shape}")
    print(f"   Next_states shape: {next_states.shape}")
    print(f"   Dones shape: {dones.shape}")
    print(f"   Sample actions: {actions}")
    print(f"   Sample rewards: {rewards}")
    
    # Test capacity limit
    print("\n4. Testing capacity limit:")
    for i in range(150):  # Add more than capacity
        buffer.push(
            np.random.randn(4),
            np.random.randint(0, 4),
            np.random.randn(),
            np.random.randn(4),
            False
        )
    print(f"   Buffer size after 150 additions: {len(buffer)}")
    print(f"   Max capacity respected: {len(buffer) <= buffer.capacity}")
    
    print("\n" + "="*50)
    print("Basic Replay Buffer Tests Complete! ✅")
    print("="*50)


def test_multi_agent_buffer():
    """Test multi-agent replay buffer"""
    print("\n" + "="*50)
    print("Testing Multi-Agent Replay Buffer")
    print("="*50)
    
    num_agents = 4
    state_size = 4
    
    # Test 1: Shared buffer
    print("\n1. Testing Shared Buffer:")
    shared_buffer = MultiAgentReplayBuffer(
        num_agents=num_agents,
        capacity=100,
        shared=True
    )
    
    # Add multi-agent experience
    states = np.random.randn(num_agents, state_size)
    actions = np.random.randint(0, 4, size=num_agents)
    rewards = np.random.randn(num_agents)
    next_states = np.random.randn(num_agents, state_size)
    done = False
    
    shared_buffer.push(states, actions, rewards, next_states, done)
    print(f"   Buffer size: {len(shared_buffer)}")
    
    # Add more experiences
    for _ in range(20):
        shared_buffer.push(
            np.random.randn(num_agents, state_size),
            np.random.randint(0, 4, size=num_agents),
            np.random.randn(num_agents),
            np.random.randn(num_agents, state_size),
            False
        )
    
    print(f"   Buffer size after 20 additions: {len(shared_buffer)}")
    
    # Sample
    if shared_buffer.is_ready(5):
        batch = shared_buffer.sample(5)
        print(f"   Sampled batch - states shape: {batch[0].shape}")
    
    # Test 2: Independent buffers
    print("\n2. Testing Independent Buffers:")
    independent_buffer = MultiAgentReplayBuffer(
        num_agents=num_agents,
        capacity=100,
        shared=False
    )
    
    # Add experiences
    for _ in range(20):
        independent_buffer.push(
            np.random.randn(num_agents, state_size),
            np.random.randint(0, 4, size=num_agents),
            np.random.randn(num_agents),
            np.random.randn(num_agents, state_size),
            False
        )
    
    print(f"   Total experiences stored: {len(independent_buffer)}")
    print(f"   Ready for sampling: {independent_buffer.is_ready(5)}")
    
    print("\n" + "="*50)
    print("Multi-Agent Buffer Tests Complete! ✅")
    print("="*50)


def test_with_environment():
    """Test replay buffer with actual environment"""
    print("\n" + "="*50)
    print("Testing Replay Buffer with Environment")
    print("="*50)
    
    try:
        import sys
        sys.path.append('src')
        from environment import MultiAgentGridWorld, Action
        
        # Create environment and buffer
        env = MultiAgentGridWorld()
        buffer = MultiAgentReplayBuffer(
            num_agents=4,
            capacity=1000,
            shared=True
        )
        
        # Collect some experiences
        print("\n1. Collecting 50 experiences:")
        state = env.reset()
        
        for step in range(50):
            # Random actions
            actions = [random.choice(Action.get_all_actions()) for _ in range(4)]
            next_state, rewards, done, info = env.step(actions)
            
            # Convert actions to indices
            action_map = {Action.UP: 0, Action.DOWN: 1, Action.LEFT: 2, Action.RIGHT: 3}
            action_indices = [action_map[a] for a in actions]
            
            # Store in buffer
            buffer.push(state, action_indices, rewards, next_state, done)
            
            state = next_state
            
            if done:
                state = env.reset()
        
        print(f"   Buffer size: {len(buffer)}")
        
        # Sample and inspect
        print("\n2. Sampling batch of 32:")
        if buffer.is_ready(32):
            states, actions, rewards, next_states, dones = buffer.sample(32)
            print(f"   States shape: {states.shape}")
            print(f"   Actions shape: {actions.shape}")
            print(f"   Rewards shape: {rewards.shape}")
            print(f"   Sample of rewards: {rewards[:5]}")
            print(f"   Number of done states: {dones.sum()}")
        
        print("\n" + "="*50)
        print("Environment Integration Test Complete! ✅")
        print("="*50)
        
    except ImportError as e:
        print(f"\nCouldn't import environment: {e}")
        print("Run from project root: python src/agent.py")


if __name__ == "__main__":
    # Run all tests
    test_basic_replay_buffer()
    test_multi_agent_buffer()
    test_with_environment()
    
    print("\n" + "="*50)
    print("Day 4 - Replay Buffer Complete!")
    print("="*50)
    print("\nNext Steps:")
    print("1. Combine DQN + Replay Buffer")
    print("2. Implement training loop")
    print("3. Add target network updates")
    print("="*50)