import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from datetime import datetime
import json
import sys

sys.path.insert(0, 'src')

from environment import MultiAgentGridWorld, Action
from agent import MultiAgentDQNSystem
from assignment_evaluator import AssignmentEvaluator


class AssignmentCompliantEnvironment(MultiAgentGridWorld):
    """Environment configured for assignment requirements"""
    
    def __init__(self, config):
        super().__init__(
            grid_size=config['environment']['grid_size'],
            num_agents=config['environment']['num_agents']
        )
        # Use config values
        self.max_steps = config['environment']['max_steps']
        self.max_collisions = config['environment']['max_collisions']
        self.reward_config = config.get('rewards', {})
    
    def reset(self):
        """Reset with spread out starting positions"""
        start_positions = [(0, 0), (0, 4), (4, 0), (4, 4)]
        
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
        """Step with assignment-compliant rewards"""
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
            collision_penalty = self.reward_config.get('collision', -10.0)
            rewards = [collision_penalty] * self.num_agents
        else:
            # Update positions
            for agent, new_pos in zip(self.agents, new_positions):
                agent.position = new_pos
                agent.total_steps += 1
            
            # Calculate rewards
            for idx, agent in enumerate(self.agents):
                step_penalty = self.reward_config.get('step', -0.1)
                reward = step_penalty
                
                # PICKUP
                if agent.position == self.location_A and not agent.has_item:
                    agent.pickup_item()
                    reward = self.reward_config.get('pickup', 1.0)
                
                # DELIVERY
                elif agent.position == self.location_B and agent.has_item:
                    agent.dropoff_item()
                    self.total_deliveries += 1
                    reward = self.reward_config.get('delivery', 10.0)
                
                # Progress shaping
                else:
                    target = self.location_B if agent.has_item else self.location_A
                    old_dist = abs(old_positions[idx][0] - target[0]) + abs(old_positions[idx][1] - target[1])
                    new_dist = abs(agent.position[0] - target[0]) + abs(agent.position[1] - target[1])
                    
                    if new_dist < old_dist:
                        reward = self.reward_config.get('progress_closer', 0.5)
                    elif new_dist > old_dist:
                        reward = self.reward_config.get('progress_away', -0.5)
                
                rewards[idx] = reward
        
        self.total_steps += 1
        
        done = (self.total_steps >= self.max_steps or 
                self.total_collisions >= self.max_collisions)
        
        info = {
            'total_steps': self.total_steps,
            'total_collisions': self.total_collisions,
            'total_deliveries': self.total_deliveries,
            'collision_this_step': collision
        }
        
        return self.get_state(), rewards, done, info


class AssignmentTrainer:
    """Trainer with assignment evaluation"""
    
    def __init__(self, config_path="config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create environment
        self.env = AssignmentCompliantEnvironment(self.config)
        
        # Agent system
        self.agent_system = MultiAgentDQNSystem(
            num_agents=4,
            state_size_per_agent=4,
            action_size=4,
            shared_network=True,
            config_path=config_path
        )
        
        # Assignment evaluator
        self.evaluator = AssignmentEvaluator(config_path)
        
        # Training params
        self.num_episodes = self.config['training']['episodes']
        self.target_update_freq = self.config['training'].get('target_update_freq', 200)
        self.warm_start_steps = self.config['training'].get('warm_start_steps', 5000)
        self.save_freq = 100
        
        self.action_map = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
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
        """Main training loop"""
        print("\n" + "="*70)
        print("🚀 ASSIGNMENT COMPLIANT TRAINING")
        print("="*70)
        print(f"Episodes:         {self.num_episodes}")
        print(f"Warm Start:       {self.warm_start_steps} steps")
        print(f"Max Steps:        {self.env.max_steps}")
        print(f"Max Collisions:   {self.env.max_collisions}")
        print(f"Collision Penalty: {self.config['rewards']['collision']}")
        print(f"Delivery Reward:  {self.config['rewards']['delivery']}")
        print("="*70 + "\n")
        
        # PHASE 1: Warm start
        if self.warm_start_steps > 0:
            print(f"🔥 WARM START: Collecting {self.warm_start_steps} experiences...")
            self._warm_start()
            print(f"✅ Warm start complete! Buffer size: {len(self.agent_system.agents[0].memory)}\n")
        
        # PHASE 2: Training
        for episode in range(1, self.num_episodes + 1):
            episode_reward, episode_info = self._run_episode()
            
            # Log metrics
            self.episode_rewards.append(episode_reward)
            self.episode_deliveries.append(episode_info['deliveries'])
            self.episode_collisions.append(episode_info['collisions'])
            
            recent_losses = self.agent_system.agents[0].losses[-episode_info['steps']:] if self.agent_system.agents[0].losses else []
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            self.episode_losses.append(avg_loss)
            
            # Print progress
            if episode % 10 == 0:
                print(f"Ep {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Del: {episode_info['deliveries']:2d} | "
                      f"Col: {episode_info['collisions']:2d} | "
                      f"Steps: {episode_info['steps']:4d} | "
                      f"ε: {self.agent_system.agents[0].epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f}")
            
            # Checkpoint
            if episode % 100 == 0:
                self._print_summary(episode)
                
                recent_deliveries = self.episode_deliveries[-100:]
                avg_deliveries = np.mean(recent_deliveries)
                
                if avg_deliveries > self.best_avg_deliveries:
                    self.best_avg_deliveries = avg_deliveries
                    self.best_model_episode = episode
                    self._save_checkpoint(f'best_ep{episode}')
                    print(f"  🏆 NEW BEST! Avg deliveries: {avg_deliveries:.2f}")
                
                self._save_checkpoint(episode)
        
        # Final save
        self._save_checkpoint('final')
        self._save_plots()
        self._save_metrics()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"Best Model: Episode {self.best_model_episode}")
        print(f"Best Avg Deliveries: {self.best_avg_deliveries:.2f}")
        print(f"Logs saved to: {self.log_dir}")
        print("="*70 + "\n")
        
        # Evaluate
        print("\n📊 EVALUATING BEST MODEL...")
        self._load_best_model()
        self.evaluate(num_episodes=20)
    
    def _warm_start(self):
        """Collect initial experiences"""
        state = self.env.reset()
        steps = 0
        
        pbar = tqdm(total=self.warm_start_steps, desc="Warm Start")
        
        while steps < self.warm_start_steps:
            actions = [np.random.randint(4) for _ in range(4)]
            action_enums = [self.action_map[a] for a in actions]
            
            next_state, rewards, done, info = self.env.step(action_enums)
            
            self.agent_system.store_experiences(state, actions, rewards, next_state, done)
            
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
        episode_steps = 0
        done = False
        
        while not done:
            action_indices = self.agent_system.select_actions(state, explore=True)
            actions = [self.action_map[idx] for idx in action_indices]
            
            next_state, rewards, done, info = self.env.step(actions)
            
            self.agent_system.store_experiences(state, action_indices, rewards, next_state, done)
            self.agent_system.train_step()
            
            if self.global_step % self.target_update_freq == 0:
                self.agent_system.update_target_networks()
            
            state = next_state
            total_reward += sum(rewards)
            episode_steps += 1
            self.global_step += 1
        
        self.agent_system.decay_epsilon()
        
        episode_info = {
            'deliveries': self.env.total_deliveries,
            'collisions': self.env.total_collisions,
            'steps': self.env.total_steps
        }
        
        return total_reward, episode_info
    
    def _print_summary(self, episode, window=100):
        """Print training summary"""
        if episode < window:
            return
        
        recent_rewards = self.episode_rewards[-window:]
        recent_deliveries = self.episode_deliveries[-window:]
        recent_collisions = self.episode_collisions[-window:]
        
        print(f"\n{'='*70}")
        print(f"Episode {episode} Summary (Last {window} episodes):")
        print(f"{'='*70}")
        print(f"  Avg Reward:        {np.mean(recent_rewards):8.2f}")
        print(f"  Avg Deliveries:    {np.mean(recent_deliveries):8.2f}")
        print(f"  Avg Collisions:    {np.mean(recent_collisions):8.2f}")
        print(f"  Best Avg Del:      {self.best_avg_deliveries:8.2f}")
        print(f"  Current ε:         {self.agent_system.agents[0].epsilon:.4f}")
        print(f"{'='*70}\n")
    
    def _save_checkpoint(self, episode):
        """Save model checkpoint"""
        os.makedirs("models", exist_ok=True)
        filepath = f"models/dqn_episode_{episode}.pth"
        
        torch.save({
            'policy_net': self.agent_system.agents[0].policy_net.state_dict(),
            'target_net': self.agent_system.agents[0].target_net.state_dict(),
            'optimizer': self.agent_system.agents[0].optimizer.state_dict(),
            'epsilon': self.agent_system.agents[0].epsilon,
            'episode': episode
        }, filepath)
    
    def _load_best_model(self):
        """Load best model"""
        filepath = f"models/dqn_episode_best_ep{self.best_model_episode}.pth"
        checkpoint = torch.load(filepath)
        self.agent_system.agents[0].policy_net.load_state_dict(checkpoint['policy_net'])
        print(f"✅ Loaded best model from episode {self.best_model_episode}")
    
    def _save_plots(self):
        """Save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(self.episode_deliveries, alpha=0.6)
        if len(self.episode_deliveries) >= 50:
            ma = np.convolve(self.episode_deliveries, np.ones(50)/50, mode='valid')
            axes[0, 0].plot(range(49, len(self.episode_deliveries)), ma, 'r-', linewidth=2)
        axes[0, 0].set_title('Deliveries per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.episode_rewards, alpha=0.6)
        if len(self.episode_rewards) >= 50:
            ma = np.convolve(self.episode_rewards, np.ones(50)/50, mode='valid')
            axes[0, 1].plot(range(49, len(self.episode_rewards)), ma, 'r-', linewidth=2)
        axes[0, 1].set_title('Rewards per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.episode_collisions, alpha=0.6)
        if len(self.episode_collisions) >= 50:
            ma = np.convolve(self.episode_collisions, np.ones(50)/50, mode='valid')
            axes[1, 0].plot(range(49, len(self.episode_collisions)), ma, 'r-', linewidth=2)
        axes[1, 0].set_title('Collisions per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].grid(True, alpha=0.3)
        
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
    
    def _save_metrics(self):
        """Save metrics to JSON"""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_deliveries': self.episode_deliveries,
            'episode_collisions': self.episode_collisions,
            'episode_losses': self.episode_losses,
            'best_avg_deliveries': float(self.best_avg_deliveries),
            'best_model_episode': int(self.best_model_episode)
        }
        
        with open(f"{self.log_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Metrics saved to {self.log_dir}/metrics.json")
    
    def evaluate(self, num_episodes=20):
        """Evaluate with assignment criteria"""
        self.agent_system.agents[0].policy_net.eval()
        
        print("\n" + "="*70)
        print("📊 FINAL EVALUATION (Greedy Policy)")
        print("="*70 + "\n")
        
        eval_results = []
        
        for ep in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action_indices = self.agent_system.select_actions(state, explore=False)
                actions = [self.action_map[idx] for idx in action_indices]
                next_state, rewards, done, info = self.env.step(actions)
                
                state = next_state
                total_reward += sum(rewards)
            
            eval_results.append({
                'deliveries': self.env.total_deliveries,
                'collisions': self.env.total_collisions,
                'steps': self.env.total_steps,
                'reward': total_reward
            })
            
            print(f"Eval {ep+1:2d}: Del={self.env.total_deliveries:2d}, "
                  f"Col={self.env.total_collisions:2d}, "
                  f"Steps={self.env.total_steps:4d}, "
                  f"Reward={total_reward:7.1f}")
        
        # Assignment evaluation
        final_eval = self.evaluator.evaluate_final(eval_results)
        self.evaluator.print_evaluation(final_eval)
        
        # Save evaluation
        with open(f"{self.log_dir}/final_evaluation.json", 'w') as f:
            json.dump(final_eval, f, indent=2)
        
        self.agent_system.agents[0].policy_net.train()
        
        return final_eval


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🎯 ASSIGNMENT COMPLIANT MULTI-AGENT DQN TRAINER")
    print("="*70)
    print("\nKey Features:")
    print("✅ Max steps: 1500")
    print("✅ Max collisions: 4")
    print("✅ Assignment evaluation criteria")
    print("✅ Performance points calculation")
    print("="*70 + "\n")
    
    try:
        trainer = AssignmentTrainer(config_path="config.yaml")
        trainer.train()
        
        print("\n🎉 Training complete! Check logs/ and models/ folders.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()