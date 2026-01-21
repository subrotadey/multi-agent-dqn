"""
SOLUTION 4: Fixed Training Loop
================================
Critical bug fixes in your training code
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from datetime import datetime
import json


class ImprovedTrainer:
    """
    Fixed trainer with proper action handling
    """
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Import after config is loaded
        import sys
        sys.path.insert(0, 'src')
        from environment import MultiAgentGridWorld, Action
        from agent import MultiAgentDQNSystem
        
        self.Action = Action
        
        # Create environment
        self.env = MultiAgentGridWorld()
        
        # Override with config
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
        
        # CRITICAL FIX: Action mapping
        self.action_map = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        
        # Training params
        self.num_episodes = self.config['training']['episodes']
        self.target_update_freq = self.config['training'].get('target_update_freq', 100)
        self.warm_start_steps = self.config['training'].get('warm_start_steps', 10000)
        
        self.global_step = 0
        
        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/run_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Metrics
        self.episode_rewards = []
        self.episode_deliveries = []
        self.episode_collisions = []
        self.episode_losses = []
        
        self.best_avg_deliveries = 0
        self.best_model_episode = 0
    
    def train(self):
        print("\n" + "="*70)
        print("🚀 IMPROVED TRAINING WITH FIXES")
        print("="*70)
        print(f"Episodes:         {self.num_episodes}")
        print(f"Warm Start:       {self.warm_start_steps} steps")
        print(f"Max Steps:        {self.env.max_steps}")
        print(f"Max Collisions:   {self.env.max_collisions}")
        print(f"Epsilon Decay:    {self.agent_system.agents[0].epsilon_decay}")
        print("="*70 + "\n")
        
        # Warm start
        if self.warm_start_steps > 0:
            print(f"🔥 Collecting {self.warm_start_steps} random experiences...")
            self._warm_start()
            print(f"✅ Buffer size: {len(self.agent_system.agents[0].memory)}\n")
        
        # Training loop
        for episode in range(1, self.num_episodes + 1):
            episode_reward, episode_info = self._run_episode()
            
            # Log
            self.episode_rewards.append(episode_reward)
            self.episode_deliveries.append(episode_info['deliveries'])
            self.episode_collisions.append(episode_info['collisions'])
            
            recent_losses = self.agent_system.agents[0].losses[-episode_info['steps']:] if self.agent_system.agents[0].losses else []
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            self.episode_losses.append(avg_loss)
            
            # Progress
            if episode % 20 == 0:
                avg_del = np.mean(self.episode_deliveries[-20:])
                avg_col = np.mean(self.episode_collisions[-20:])
                print(f"Ep {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Del: {episode_info['deliveries']:2d} (avg:{avg_del:.1f}) | "
                      f"Col: {episode_info['collisions']:2d} (avg:{avg_col:.1f}) | "
                      f"ε: {self.agent_system.agents[0].epsilon:.3f}")
            
            # Checkpoint
            if episode % 200 == 0:
                self._print_summary(episode)
                
                recent_deliveries = self.episode_deliveries[-100:]
                avg_deliveries = np.mean(recent_deliveries)
                
                if avg_deliveries > self.best_avg_deliveries:
                    self.best_avg_deliveries = avg_deliveries
                    self.best_model_episode = episode
                    self._save_checkpoint(f'best_ep{episode}')
                    print(f"  🏆 NEW BEST! Avg deliveries: {avg_deliveries:.2f}\n")
        
        # Final save
        self._save_checkpoint('final')
        self._save_plots()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print(f"Best Model: Episode {self.best_model_episode}")
        print(f"Best Avg Deliveries: {self.best_avg_deliveries:.2f}")
        print("="*70 + "\n")
    
    def _warm_start(self):
        """Collect initial experiences with random policy"""
        state = self.env.reset()
        steps = 0
        
        pbar = tqdm(total=self.warm_start_steps, desc="Warm Start")
        
        while steps < self.warm_start_steps:
            # CRITICAL FIX: Random action INDICES (0-3)
            action_indices = [np.random.randint(4) for _ in range(4)]
            
            # Convert to enum
            actions = [self.action_map[idx] for idx in action_indices]
            
            next_state, rewards, done, info = self.env.step(actions)
            
            # Store with INDICES not enums
            self.agent_system.store_experiences(
                state, action_indices, rewards, next_state, done
            )
            
            state = next_state
            steps += 1
            pbar.update(1)
            
            if done:
                state = self.env.reset()
        
        pbar.close()
    
    def _run_episode(self):
        """Run one training episode"""
        state = self.env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # CRITICAL FIX: Get action indices (0-3)
            action_indices = self.agent_system.select_actions(state, explore=True)
            
            # VALIDATION: Ensure indices are in range
            assert all(0 <= idx < 4 for idx in action_indices), \
                f"Invalid action indices: {action_indices}"
            
            # Convert to enum for environment
            actions = [self.action_map[idx] for idx in action_indices]
            
            next_state, rewards, done, info = self.env.step(actions)
            
            # Store with INDICES
            self.agent_system.store_experiences(
                state, action_indices, rewards, next_state, done
            )
            
            # Train
            self.agent_system.train_step()
            
            # Update target network
            if self.global_step % self.target_update_freq == 0:
                self.agent_system.update_target_networks()
            
            state = next_state
            total_reward += sum(rewards)
            self.global_step += 1
        
        # Decay epsilon
        self.agent_system.decay_epsilon()
        
        episode_info = {
            'deliveries': self.env.total_deliveries,
            'collisions': self.env.total_collisions,
            'steps': self.env.total_steps
        }
        
        return total_reward, episode_info
    
    def _print_summary(self, episode, window=100):
        if episode < window:
            return
        
        recent_deliveries = self.episode_deliveries[-window:]
        recent_collisions = self.episode_collisions[-window:]
        
        print(f"\n{'='*70}")
        print(f"Episode {episode} Summary (Last {window} episodes):")
        print(f"{'='*70}")
        print(f"  Avg Deliveries:    {np.mean(recent_deliveries):8.2f}")
        print(f"  Avg Collisions:    {np.mean(recent_collisions):8.2f}")
        print(f"  Best Avg Del:      {self.best_avg_deliveries:8.2f}")
        print(f"  Current ε:         {self.agent_system.agents[0].epsilon:.4f}")
        print(f"{'='*70}\n")
    
    def _save_checkpoint(self, episode):
        os.makedirs("models", exist_ok=True)
        filepath = f"models/dqn_episode_{episode}.pth"
        
        torch.save({
            'policy_net': self.agent_system.agents[0].policy_net.state_dict(),
            'target_net': self.agent_system.agents[0].target_net.state_dict(),
            'optimizer': self.agent_system.agents[0].optimizer.state_dict(),
            'epsilon': self.agent_system.agents[0].epsilon,
            'episode': episode
        }, filepath)
    
    def _save_plots(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Deliveries
        axes[0, 0].plot(self.episode_deliveries, alpha=0.6, label='Per Episode')
        if len(self.episode_deliveries) >= 50:
            ma = np.convolve(self.episode_deliveries, np.ones(50)/50, mode='valid')
            axes[0, 0].plot(range(49, len(self.episode_deliveries)), ma, 'r-', linewidth=2, label='MA-50')
        axes[0, 0].set_title('Deliveries per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rewards
        axes[0, 1].plot(self.episode_rewards, alpha=0.6)
        if len(self.episode_rewards) >= 50:
            ma = np.convolve(self.episode_rewards, np.ones(50)/50, mode='valid')
            axes[0, 1].plot(range(49, len(self.episode_rewards)), ma, 'r-', linewidth=2)
        axes[0, 1].set_title('Rewards per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Collisions
        axes[1, 0].plot(self.episode_collisions, alpha=0.6)
        if len(self.episode_collisions) >= 50:
            ma = np.convolve(self.episode_collisions, np.ones(50)/50, mode='valid')
            axes[1, 0].plot(range(49, len(self.episode_collisions)), ma, 'r-', linewidth=2)
        axes[1, 0].set_title('Collisions per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[1, 1].plot(self.episode_losses, alpha=0.6)
        if len(self.episode_losses) >= 50:
            ma = np.convolve(self.episode_losses, np.ones(50)/50, mode='valid')
            axes[1, 1].plot(range(49, len(self.episode_losses)), ma, 'r-', linewidth=2)
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/training_curves.png", dpi=150)
        print(f"✅ Plots saved to {self.log_dir}/training_curves.png")


def main():
    print("\n" + "="*70)
    print("🎯 FIXED MULTI-AGENT DQN TRAINER")
    print("="*70)
    print("\nKey Fixes:")
    print("✅ Staggered starting positions")
    print("✅ Increased collision budget for learning")
    print("✅ Better reward shaping")
    print("✅ Slower epsilon decay")
    print("✅ Fixed action index handling")
    print("="*70 + "\n")
    
    try:
        trainer = ImprovedTrainer(config_path="config.yaml")
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()