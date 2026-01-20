"""
Multi-Agent DQN Coordination - Environment
===========================================
5x5 Grid World Environment with 4 Agents
Task: Shuttle items from A to B, avoid collisions
"""

import numpy as np
from enum import Enum
from typing import List, Tuple, Dict
import random


# 1. MOVEMENT ENUM
class Action(Enum):
    """4 directional movements - NO WAIT action allowed"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    @staticmethod
    def get_all_actions():
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    
    @staticmethod
    def to_delta(action):
        """Convert action to grid delta (row, col)"""
        deltas = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1)
        }
        return deltas[action]


# 2. AGENT CLASS
class Agent:
    """Single agent that shuttles items from A to B"""
    
    def __init__(self, agent_id: int, start_pos: Tuple[int, int]):
        self.id = agent_id
        self.position = start_pos
        self.has_item = False
        self.total_deliveries = 0
        self.total_steps = 0
        
    def move(self, action: Action, grid_size: int = 5) -> Tuple[int, int]:
        """
        Move agent based on action
        Returns new position after applying boundary constraints
        """
        delta = Action.to_delta(action)
        new_row = max(0, min(grid_size - 1, self.position[0] + delta[0]))
        new_col = max(0, min(grid_size - 1, self.position[1] + delta[1]))
        return (new_row, new_col)
    
    def pickup_item(self):
        """Pick up item at location A"""
        self.has_item = True
    
    def dropoff_item(self):
        """Drop off item at location B"""
        self.has_item = False
        self.total_deliveries += 1
    
    def __repr__(self):
        return f"Agent{self.id}@{self.position} {'[ITEM]' if self.has_item else ''}"


# 3. GRID ENVIRONMENT
class MultiAgentGridWorld:
    """
    5x5 Grid World Environment
    - 4 agents shuttle items from A to B
    - Collision detection
    - Reward system
    """
    
    def __init__(self, grid_size: int = 5, num_agents: int = 4):
        self.grid_size = grid_size
        self.num_agents = num_agents
        
        # Define locations
        self.location_A = (0, 0)  # Pickup location (top-left)
        self.location_B = (4, 4)  # Dropoff location (bottom-right)
        
        # Initialize agents at different positions
        start_positions = [
            (0, 0), (0, 4), (4, 0), (4, 4)
        ]
        self.agents = [Agent(i, start_positions[i]) for i in range(num_agents)]
        
        # Tracking metrics
        self.total_steps = 0
        self.total_collisions = 0
        self.total_deliveries = 0
        
        # Training budgets
        self.max_steps = 1500
        self.max_collisions = 4
        
    def reset(self):
        """Reset environment to initial state"""
        # FIXED: Spread out starting positions
        start_positions = [
            (0, 0),  # Top-left (at A)
            (0, 4),  # Top-right  
            (4, 0),  # Bottom-left
            (4, 4)   # Bottom-right (at B)
        ]
        
        for i, agent in enumerate(self.agents):
            agent.position = start_positions[i]
            agent.has_item = False
            agent.total_deliveries = 0
            agent.total_steps = 0
        
        self.total_steps = 0
        self.total_collisions = 0
        self.total_deliveries = 0
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        Get current state representation
        Returns: numpy array of shape (num_agents, 4)
        Each agent state: [row, col, has_item, distance_to_target]
        """
        state = []
        for agent in self.agents:
            target = self.location_B if agent.has_item else self.location_A
            distance = abs(agent.position[0] - target[0]) + abs(agent.position[1] - target[1])
            
            agent_state = [
                agent.position[0],
                agent.position[1],
                1 if agent.has_item else 0,
                distance
            ]
            state.append(agent_state)
        
        return np.array(state, dtype=np.float32)
    
    def check_collision(self, positions: List[Tuple[int, int]]) -> bool:
        """
        Check if any two agents are at the same position
        Returns True if collision detected
        """
        return len(positions) != len(set(positions))
    
    def step(self, actions):
        new_positions = []
        rewards = [0.0] * self.num_agents
        old_positions = [agent.position for agent in self.agents]
        
        for agent, action in zip(self.agents, actions):
            new_pos = agent.move(action, self.grid_size)
            new_positions.append(new_pos)
        
        collision = self.check_collision(new_positions)
        
        if collision:
            self.total_collisions += 1
            rewards = [-2.0] * self.num_agents
        else:
            for agent, new_pos in zip(self.agents, new_positions):
                agent.position = new_pos
                agent.total_steps += 1
            
            for idx, agent in enumerate(self.agents):
                reward = -0.1
                
                if agent.position == self.location_A and not agent.has_item:
                    agent.pickup_item()
                    reward = 5.0  # OVERRIDE, not +=
                
                elif agent.position == self.location_B and agent.has_item:
                    agent.dropoff_item()
                    self.total_deliveries += 1
                    reward = 20.0  # OVERRIDE, not +=
                
                else:
                    target = self.location_B if agent.has_item else self.location_A
                    old_dist = abs(old_positions[idx][0] - target[0]) + abs(old_positions[idx][1] - target[1])
                    new_dist = abs(agent.position[0] - target[0]) + abs(agent.position[1] - target[1])
                    
                    if new_dist < old_dist:
                        reward = 0.5
                    elif new_dist > old_dist:
                        reward = -0.3
                
                rewards[idx] = reward
        
        self.total_steps += 1
        done = (self.total_steps >= self.max_steps or self.total_collisions >= self.max_collisions)
        
        info = {
            'total_steps': self.total_steps,
            'total_collisions': self.total_collisions,
            'total_deliveries': self.total_deliveries,
            'collision_this_step': collision
        }
        
        return self.get_state(), rewards, done, info
    
    def render(self):
        """Visual representation of the grid"""
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        
        # Mark locations
        grid[self.location_A] = 'A'
        grid[self.location_B] = 'B'
        
        # Mark agents
        for agent in self.agents:
            r, c = agent.position
            symbol = str(agent.id) if not agent.has_item else f"{agent.id}*"
            grid[r, c] = symbol
        
        print("\n=== Grid World ===")
        for row in grid:
            print(' '.join(row))
        print(f"Deliveries: {self.total_deliveries}, Collisions: {self.total_collisions}, Steps: {self.total_steps}")
        print()


# 4. TESTING
def test_environment():
    """Test basic environment functionality"""
    print("Testing Multi-Agent Grid Environment\n")
    
    env = MultiAgentGridWorld()
    state = env.reset()
    
    print("Initial State:")
    print(state)
    env.render()
    
    # Test random actions
    print("\nTesting 5 random steps:")
    for step in range(5):
        actions = [random.choice(Action.get_all_actions()) for _ in range(env.num_agents)]
        print(f"\nStep {step + 1}: Actions = {[a.name for a in actions]}")
        
        next_state, rewards, done, info = env.step(actions)
        
        print(f"Rewards: {rewards}")
        print(f"Info: {info}")
        env.render()
        
        if done:
            print("Episode finished!")
            break
    
    # Print agent statistics
    print("\n=== Agent Statistics ===")
    for agent in env.agents:
        print(agent)


if __name__ == "__main__":
    test_environment()
    
    print("\n" + "="*50)
    print("Environment Setup Complete!")
    print("="*50)