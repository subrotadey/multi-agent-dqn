"""
SOLUTION 1: Staggered Starting Positions
=========================================
Spread agents to avoid immediate collisions
"""

import numpy as np
from enum import Enum
from typing import List, Tuple, Dict
import random


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
    @staticmethod
    def to_delta(action):
        deltas = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1)
        }
        return deltas[action]


class Agent:
    def __init__(self, agent_id: int, start_pos: Tuple[int, int]):
        self.id = agent_id
        self.position = start_pos
        self.has_item = False
        self.total_deliveries = 0
        self.total_steps = 0
        
    def move(self, action: Action, grid_size: int = 5) -> Tuple[int, int]:
        delta = Action.to_delta(action)
        new_row = max(0, min(grid_size - 1, self.position[0] + delta[0]))
        new_col = max(0, min(grid_size - 1, self.position[1] + delta[1]))
        return (new_row, new_col)
    
    def pickup_item(self):
        self.has_item = True
    
    def dropoff_item(self):
        self.has_item = False
        self.total_deliveries += 1


class MultiAgentGridWorld:
    """
    FIXED: Better starting positions to avoid immediate collisions
    """
    
    def __init__(self, grid_size: int = 5, num_agents: int = 4):
        self.grid_size = grid_size
        self.num_agents = num_agents
        
        self.location_A = (0, 0)
        self.location_B = (4, 4)
        
        # SOLUTION 1A: Staggered positions - no overlap
        # Instead of all corners, spread them out
        start_positions = [
            (1, 1),  # Agent 0: Near A but not on it
            (1, 3),  # Agent 1: Right side
            (3, 1),  # Agent 2: Left side  
            (3, 3)   # Agent 3: Near B but not on it
        ]
        
        self.agents = [Agent(i, start_positions[i]) for i in range(num_agents)]
        
        self.total_steps = 0
        self.total_collisions = 0
        self.total_deliveries = 0
        
        # CRITICAL: Increase budgets for learning phase
        self.max_steps = 1500
        self.max_collisions = 50  # INCREASED from 4 to allow learning
        
    def reset(self):
        """Reset with safe starting positions"""
        # SOLUTION 1B: Random safe positions (no overlap)
        available_positions = [
            (1, 1), (1, 2), (1, 3),
            (2, 1), (2, 2), (2, 3),
            (3, 1), (3, 2), (3, 3)
        ]
        
        # Randomly sample 4 positions without replacement
        start_positions = random.sample(available_positions, self.num_agents)
        
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
        """Enhanced state with more information"""
        state = []
        for agent in self.agents:
            target = self.location_B if agent.has_item else self.location_A
            distance = abs(agent.position[0] - target[0]) + abs(agent.position[1] - target[1])
            
            agent_state = [
                agent.position[0] / 4.0,      # Normalized row
                agent.position[1] / 4.0,      # Normalized col
                1.0 if agent.has_item else 0.0,
                distance / 8.0                # Normalized distance
            ]
            state.append(agent_state)
        
        return np.array(state, dtype=np.float32)
    
    def check_collision(self, positions: List[Tuple[int, int]]) -> bool:
        """Check for position overlaps"""
        return len(positions) != len(set(positions))
    
    def step(self, actions):
        """
        SOLUTION 1C: Gentler collision handling
        """
        new_positions = []
        rewards = [0.0] * self.num_agents
        old_positions = [agent.position for agent in self.agents]
        
        # Calculate new positions
        for agent, action in zip(self.agents, actions):
            new_pos = agent.move(action, self.grid_size)
            new_positions.append(new_pos)
        
        # Check collisions
        collision = self.check_collision(new_positions)
        
        if collision:
            self.total_collisions += 1
            # SOLUTION: Smaller penalty, don't halt immediately
            collision_penalty = -2.0  # Reduced from -10
            rewards = [collision_penalty] * self.num_agents
            
            # Don't move agents on collision - stay in place
            new_positions = old_positions
        else:
            # Update positions
            for agent, new_pos in zip(self.agents, new_positions):
                agent.position = new_pos
                agent.total_steps += 1
            
            # Rewards
            for idx, agent in enumerate(self.agents):
                reward = -0.05  # Small step penalty
                
                # PICKUP
                if agent.position == self.location_A and not agent.has_item:
                    agent.pickup_item()
                    reward = 2.0  # Good reward
                
                # DELIVERY
                elif agent.position == self.location_B and agent.has_item:
                    agent.dropoff_item()
                    self.total_deliveries += 1
                    reward = 10.0  # Excellent reward
                
                # Progress shaping
                else:
                    target = self.location_B if agent.has_item else self.location_A
                    old_dist = abs(old_positions[idx][0] - target[0]) + abs(old_positions[idx][1] - target[1])
                    new_dist = abs(agent.position[0] - target[0]) + abs(agent.position[1] - target[1])
                    
                    if new_dist < old_dist:
                        reward = 0.3   # Decent progress reward
                    elif new_dist > old_dist:
                        reward = -0.3  # Small penalty
                
                rewards[idx] = reward
        
        self.total_steps += 1
        
        # Only end if step budget exceeded (let collisions happen during learning)
        done = self.total_steps >= self.max_steps
        
        info = {
            'total_steps': self.total_steps,
            'total_collisions': self.total_collisions,
            'total_deliveries': self.total_deliveries,
            'collision_this_step': collision
        }
        
        return self.get_state(), rewards, done, info
    
    def render(self):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        
        grid[self.location_A] = 'A'
        grid[self.location_B] = 'B'
        
        for agent in self.agents:
            r, c = agent.position
            symbol = str(agent.id) if not agent.has_item else f"{agent.id}*"
            if grid[r, c] in ['.']:
                grid[r, c] = symbol
            elif grid[r, c] == 'A':
                grid[r, c] = f"A{agent.id}"
            elif grid[r, c] == 'B':
                grid[r, c] = f"B{agent.id}"
        
        print("\n=== Grid World ===")
        for row in grid:
            print(' '.join(row))
        print(f"Deliveries: {self.total_deliveries}, Collisions: {self.total_collisions}, Steps: {self.total_steps}")


# Quick test
if __name__ == "__main__":
    env = MultiAgentGridWorld()
    state = env.reset()
    env.render()
    
    print("\n5 Random Steps:")
    for i in range(5):
        actions = [random.choice(list(Action)) for _ in range(4)]
        state, rewards, done, info = env.step(actions)
        print(f"Step {i+1}: Rewards={rewards}, Collisions={info['total_collisions']}")
        env.render()
        if done:
            break