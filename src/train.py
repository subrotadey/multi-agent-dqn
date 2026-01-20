"""
COMPLETE FIXED TRAINING SYSTEM
===============================
Fixes all issues with the original training
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import os
from datetime import datetime
import json
import sys

sys.path.insert(0, 'src')

from environment import MultiAgentGridWorld, Action
from agent import MultiAgentDQNSystem


# ============================================
# 1. FIXED ENVIRONMENT
# ============================================
class FixedEnvironment(MultiAgentGridWorld):
    """Fixed environment with better starting positions and rewards"""
    
    def reset(self):
        """Reset with SPREAD OUT initial positions"""
        # CRITICAL FIX: Agents start at corners, not clustered
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
    
    def step(self, actions):
        """Modified step with BETTER REWARD SHAPING"""
        new_positions = []
        rewards = [0.0] * self.num_agents
        
        # Store old positions for distance calculation
        old_positions = [agent.position for agent in self.agents]
        
        # Calculate new positions
        for agent, action in zip(self.agents, actions):
            new_pos = agent.move(action, self.grid_size)
            new_positions.append(new_pos)
        
        # Check collisions
        collision = self.check_collision(new_positions)
        
        if collision:
            self.total_collisions += 1
            # REDUCED penalty - less harsh than original -10
            rewards = [-2.0] * self.num_agents
            # Don't move agents if collision detected
        else:
            # Update positions
            for idx, (agent, new_pos) in enumerate(zip(self.agents, new_positions)):
                old_pos = agent.position
                agent.position = new_pos
                agent.total_steps += 1
                
                # === PICKUP DETECTION ===
                if agent.position == self.location_A and not agent.has_item:
                    agent.has_item = True
                    rewards[idx] += 5.0  # Reward for pickup
                    print(f"✅ Agent {agent.id} picked up item at {agent.position}!")
                
                # === DELIVERY DETECTION ===
                elif agent.position == self.location_B and agent.has_item:
                    agent.has_item = False
                    agent.total_deliveries += 1
                    self.total_deliveries += 1
                    rewards[idx] += 20.0  # Big reward for delivery
                    print(f"🎯 Agent {agent.id} delivered! Total: {self.total_deliveries}")
                
                # === DISTANCE-BASED REWARD SHAPING ===
                else:
                    # Determine target based on item status
                    if agent.has_item:
                        target = self.location_B  # Go to dropoff
                    else:
                        target = self.location_A  # Go to pickup
                    
                    # Calculate Manhattan distances
                    old_dist = abs(old_pos[0] - target[0]) + abs(old_pos[1] - target[1])
                    new_dist = abs(new_pos[0] - target[0]) + abs(new_pos[1] - target[1])
                    
                    # Reward for moving closer to target
                    if new_dist < old_dist:
                        rewards[idx] += 1.0  # Progress reward
                    elif new_dist > old_dist:
                        rewards[idx] -= 0.5  # Penalty for moving away
                
                # Small step penalty to encourage efficiency
                rewards[idx] -= 0.1
        
        self.total_steps += 1
        
        # Episode termination conditions
        done = (self.total_steps >= self.max_steps or 
                self.total_collisions >= self.max_collisions)
        
        info = {
            'total_steps': self.total_steps,
            'total_collisions': self.total_collisions,
            'total_deliveries': self.total_deliveries,
            'collision_this_step': collision
        }
        
        return self.get_state(), rewards, done, info


# ============================================
# 2. TRAINING LOGGER
# ============================================
class TrainingLogger:
    """Logger for tracking training metrics"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_deliveries = []
        self.episode_collisions = []
        self.episode_losses = []
        self.epsilon_history = []
        
        self.best_reward = float('-inf')
        self.best_deliveries = 0
        
    def log_episode(self, episode, total_reward, steps, deliveries, collisions, loss, epsilon):
        """Log metrics for an episode"""
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        self.episode_deliveries.append(deliveries)
        self.episode_collisions.append(collisions)
        self.episode_losses.append(loss if loss else 0)
        self.epsilon_history.append(epsilon)
        
        if deliveries > self.best_deliveries:
            self.best_deliveries = deliveries
        if total_reward > self.best_reward:
            self.best_reward = total_reward
    
    def print_summary(self, episode, window=100):
        """Print summary of recent performance"""
        if episode < window:
            return
        
        recent_rewards = self.episode_rewards[-window:]
        recent_deliveries = self.episode_deliveries[-window:]
        recent_collisions = self.episode_collisions[-window:]
        
        print(f"\n{'='*70}")
        print(f"Episode {episode} Summary (Last {window} episodes):")
        print(f"{'='*70}")
        print(f"  Avg Reward:      {np.mean(recent_rewards):8.2f}")
        print(f"  Avg Deliveries:  {np.mean(recent_deliveries):8.2f}")
        print(f"  Avg Collisions:  {np.mean(recent_collisions):8.2f}")
        print(f"  Best Deliveries: {self.best_deliveries}")
        print(f"  Current ε:       {self.epsilon_history[-1]:.4f}")
        print(f"{'='*70}\n")
    
    def plot_training_curves(self, save_path=None):
        """Generate and save training visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Metrics - Multi-Agent DQN', fontsize=16, fontweight='bold')
        
        # Rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Raw')
        if len(self.episode_rewards) >= 50:
            axes[0, 0].plot(self._moving_average(self.episode_rewards, 50), 
                          linewidth=2, label='MA(50)')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Deliveries
        axes[0, 1].plot(self.episode_deliveries, alpha=0.6, label='Raw')
        if len(self.episode_deliveries) >= 50:
            axes[0, 1].plot(self._moving_average(self.episode_deliveries, 50), 
                          linewidth=2, label='MA(50)')
        axes[0, 1].set_title('Deliveries per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Deliveries')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Collisions
        axes[0, 2].plot(self.episode_collisions, alpha=0.6, label='Raw')
        if len(self.episode_collisions) >= 50:
            axes[0, 2].plot(self._moving_average(self.episode_collisions, 50), 
                          linewidth=2, label='MA(50)')
        axes[0, 2].set_title('Collisions per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Collisions')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Steps
        axes[1, 0].plot(self.episode_steps, alpha=0.6, label='Raw')
        if len(self.episode_steps) >= 50:
            axes[1, 0].plot(self._moving_average(self.episode_steps, 50), 
                          linewidth=2, label='MA(50)')
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[1, 1].plot(self.episode_losses, alpha=0.6, label='Raw')
        if len(self.episode_losses) >= 50:
            axes[1, 1].plot(self._moving_average(self.episode_losses, 50), 
                          linewidth=2, label='MA(50)')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Epsilon
        axes[1, 2].plot(self.epsilon_history, linewidth=2, color='green')
        axes[1, 2].set_title('Exploration Rate (Epsilon)')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Epsilon')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Training curves saved to {save_path}")
        
        return fig
    
    def _moving_average(self, data, window):
        """Calculate moving average"""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def save_metrics(self, filepath):
        """Save all metrics to JSON"""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'episode_deliveries': self.episode_deliveries,
            'episode_collisions': self.episode_collisions,
            'episode_losses': self.episode_losses,
            'epsilon_history': self.epsilon_history,
            'best_reward': float(self.best_reward),
            'best_deliveries': int(self.best_deliveries)
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Metrics saved to {filepath}")


# ============================================
# 3. IMPROVED CONFIG
# ============================================
class ImprovedConfig:
    """Better hyperparameters for learning"""
    
    @staticmethod
    def save_improved_config():
        """Create optimized configuration file"""
        config = {
            'environment': {
                'grid_size': 5,
                'num_agents': 4,
                'max_steps': 1500,
                'max_collisions': 10  # Increased tolerance
            },
            'training': {
                'episodes': 1000,
                'learning_rate': 0.0005,  # Reduced for stability
                'gamma': 0.95,  # Slightly reduced discount
                'epsilon_start': 1.0,
                'epsilon_end': 0.05,  # Higher minimum exploration
                'epsilon_decay': 0.998,  # Slower decay
                'batch_size': 64,
                'memory_size': 10000
            },
            'model': {
                'hidden_layers': [128, 64],
                'activation': 'relu'
            }
        }
        
        with open('config_improved.yaml', 'w') as f:
            yaml.dump(config, f)
        print("✅ Improved config saved to config_improved.yaml")
        
        return config


# ============================================
# 4. MAIN TRAINER
# ============================================
class ImprovedTrainer:
    """Improved training with all fixes"""
    
    def __init__(self, config_path="config_improved.yaml"):
        # Create improved config if doesn't exist
        if not os.path.exists(config_path):
            ImprovedConfig.save_improved_config()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Use FIXED environment
        self.env = FixedEnvironment(
            grid_size=self.config['environment']['grid_size'],
            num_agents=self.config['environment']['num_agents']
        )
        
        # Override max steps/collisions
        self.env.max_steps = self.config['environment']['max_steps']
        self.env.max_collisions = self.config['environment']['max_collisions']
        
        # Agent system
        self.agent_system = MultiAgentDQNSystem(
            num_agents=4,
            state_size_per_agent=4,
            action_size=4,
            shared_network=True,
            config_path=config_path
        )
        
        # Logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = TrainingLogger(log_dir=f"logs/run_{timestamp}")
        
        self.num_episodes = self.config['training']['episodes']
        self.target_update_freq = 500  # Update target network frequently
        self.save_freq = 100
        
        self.action_map = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.global_step = 0
        
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("🚀 TRAINING MULTI-AGENT DQN")
        print("="*70)
        print(f"Episodes:         {self.num_episodes}")
        print(f"Max Steps:        {self.env.max_steps}")
        print(f"Max Collisions:   {self.env.max_collisions}")
        print(f"Learning Rate:    {self.config['training']['learning_rate']}")
        print(f"Epsilon Decay:    {self.config['training']['epsilon_decay']}")
        print("="*70 + "\n")
        
        for episode in range(1, self.num_episodes + 1):
            episode_reward, episode_steps, episode_info = self._run_episode(episode)
            
            # Calculate average loss
            recent_losses = self.agent_system.agents[0].losses[-episode_steps:] if self.agent_system.agents[0].losses else []
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            
            # Log episode
            self.logger.log_episode(
                episode=episode,
                total_reward=episode_reward,
                steps=episode_steps,
                deliveries=episode_info['deliveries'],
                collisions=episode_info['collisions'],
                loss=avg_loss,
                epsilon=self.agent_system.agents[0].epsilon
            )
            
            # Print progress
            if episode % 10 == 0:
                print(f"Ep {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Del: {episode_info['deliveries']:2d} | "
                      f"Col: {episode_info['collisions']:2d} | "
                      f"Steps: {episode_steps:4d} | "
                      f"ε: {self.agent_system.agents[0].epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f}")
            
            # Print summary
            if episode % 100 == 0:
                self.logger.print_summary(episode)
                self._save_checkpoint(episode)
        
        # Final save
        self._save_checkpoint('final')
        self.logger.plot_training_curves(
            save_path=os.path.join(self.logger.log_dir, 'training_curves.png')
        )
        self.logger.save_metrics(
            filepath=os.path.join(self.logger.log_dir, 'metrics.json')
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"Best Deliveries: {self.logger.best_deliveries}")
        print(f"Best Reward:     {self.logger.best_reward:.2f}")
        print(f"Logs saved to:   {self.logger.log_dir}")
        print("="*70 + "\n")
    
    def _run_episode(self, episode):
        """Run one training episode"""
        state = self.env.reset()
        total_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Select actions (with exploration)
            action_indices = self.agent_system.select_actions(state, explore=True)
            actions = [self.action_map[idx] for idx in action_indices]
            
            # Execute actions
            next_state, rewards, done, info = self.env.step(actions)
            
            # Store experiences
            self.agent_system.store_experiences(state, action_indices, rewards, next_state, done)
            
            # Train agent
            loss = self.agent_system.train_step()
            
            # Update target network periodically
            if self.global_step % self.target_update_freq == 0:
                self.agent_system.update_target_networks()
            
            # Update state
            state = next_state
            total_reward += sum(rewards)
            episode_steps += 1
            self.global_step += 1
            
            if done:
                break
        
        # Decay epsilon after each episode
        self.agent_system.decay_epsilon()
        
        episode_info = {
            'deliveries': self.env.total_deliveries,
            'collisions': self.env.total_collisions,
            'steps': self.env.total_steps
        }
        
        return total_reward, episode_steps, episode_info
    
    def _save_checkpoint(self, episode):
        """Save model checkpoint"""
        os.makedirs("models", exist_ok=True)
        filepath = f"models/dqn_episode_{episode}.pth"
        
        # Save only the shared agent
        torch.save({
            'policy_net': self.agent_system.agents[0].policy_net.state_dict(),
            'target_net': self.agent_system.agents[0].target_net.state_dict(),
            'optimizer': self.agent_system.agents[0].optimizer.state_dict(),
            'epsilon': self.agent_system.agents[0].epsilon,
            'episode': episode
        }, filepath)
        
        if episode != 'final':
            print(f"  💾 Checkpoint saved: episode {episode}")
    
    def evaluate(self, num_episodes=20):
        """Evaluate trained agent"""
        print("\n" + "="*70)
        print("📊 EVALUATING AGENT")
        print("="*70 + "\n")
        
        eval_deliveries = []
        eval_collisions = []
        eval_rewards = []
        eval_steps = []
        
        for ep in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Greedy actions only (no exploration)
                action_indices = self.agent_system.select_actions(state, explore=False)
                actions = [self.action_map[idx] for idx in action_indices]
                next_state, rewards, done, info = self.env.step(actions)
                
                state = next_state
                total_reward += sum(rewards)
            
            eval_deliveries.append(self.env.total_deliveries)
            eval_collisions.append(self.env.total_collisions)
            eval_rewards.append(total_reward)
            eval_steps.append(self.env.total_steps)
            
            print(f"Eval {ep+1:2d}: Deliveries={self.env.total_deliveries:2d}, "
                  f"Collisions={self.env.total_collisions:2d}, "
                  f"Steps={self.env.total_steps:4d}, "
                  f"Reward={total_reward:7.1f}")
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"Avg Deliveries: {np.mean(eval_deliveries):5.2f} ± {np.std(eval_deliveries):.2f}")
        print(f"Avg Collisions: {np.mean(eval_collisions):5.2f} ± {np.std(eval_collisions):.2f}")
        print(f"Avg Steps:      {np.mean(eval_steps):5.2f} ± {np.std(eval_steps):.2f}")
        print(f"Avg Reward:     {np.mean(eval_rewards):5.2f} ± {np.std(eval_rewards):.2f}")
        
        # Calculate success rate
        total_deliveries = sum(eval_deliveries)
        total_attempts = total_deliveries + sum(eval_collisions)
        success_rate = total_deliveries / max(1, total_attempts)
        print(f"Success Rate:   {success_rate:.2%}")
        print("="*70 + "\n")


# ============================================
# 5. MAIN FUNCTION
# ============================================
def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🔧 MULTI-AGENT DQN TRAINING SYSTEM")
    print("="*70)
    print("\nKey Features:")
    print("✅ Spread out starting positions (corners)")
    print("✅ Reduced collision penalty: -2.0")
    print("✅ Progress rewards: +1.0 for moving closer")
    print("✅ Pickup reward: +5.0")
    print("✅ Delivery reward: +20.0")
    print("✅ Slower epsilon decay: 0.998")
    print("✅ Lower learning rate: 0.0005")
    print("✅ Higher collision tolerance: 10")
    print("="*70 + "\n")
    
    try:
        trainer = ImprovedTrainer()
        trainer.train()
        trainer.evaluate(num_episodes=20)
        
        print("\n🎉 Training complete! Check logs/ folder for detailed results.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()