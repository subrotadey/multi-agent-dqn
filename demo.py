"""
Demo & Testing Script
==============================
Run quick tests and demonstrations
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, 'src')

from environment import MultiAgentGridWorld, Action
from dqn import DQN, MultiAgentDQN
from replay_buffer import ReplayBuffer
from agent import DQNAgent, MultiAgentDQNSystem


def test_full_pipeline():
    """Test complete pipeline integration"""
    print("\n" + "="*70)
    print("TESTING FULL PIPELINE")
    print("="*70 + "\n")
    
    # 1. Create environment
    print("1. Creating environment...")
    env = MultiAgentGridWorld(grid_size=5, num_agents=4)
    state = env.reset()
    print(f"   ✅ Environment created: {env.grid_size}x{env.grid_size} grid, {env.num_agents} agents")
    print(f"   State shape: {state.shape}")
    
    # 2. Create agent system
    print("\n2. Creating multi-agent DQN system...")
    ma_system = MultiAgentDQNSystem(
        num_agents=4,
        state_size_per_agent=4,
        action_size=4,
        shared_network=True,
        config_path="config.yaml"
    )
    print(f"   ✅ Agent system created with shared network")
    
    # 3. Test action selection
    print("\n3. Testing action selection...")
    action_indices = ma_system.select_actions(state, explore=True)
    print(f"   Selected actions (indices): {action_indices}")
    
    action_map = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    actions = [action_map[idx] for idx in action_indices]
    print(f"   Converted to actions: {[a.name for a in actions]}")
    
    # 4. Execute in environment
    print("\n4. Executing actions in environment...")
    next_state, rewards, done, info = env.step(actions)
    print(f"   Rewards: {rewards}")
    print(f"   Info: {info}")
    print(f"   Done: {done}")
    
    # 5. Store experience
    print("\n5. Storing experience in replay buffer...")
    ma_system.store_experiences(state, action_indices, rewards, next_state, done)
    print(f"   ✅ Experience stored")
    
    # 6. Collect more experiences
    print("\n6. Collecting 100 experiences...")
    for _ in range(100):
        action_indices = ma_system.select_actions(state, explore=True)
        actions = [action_map[idx] for idx in action_indices]
        next_state, rewards, done, info = env.step(actions)
        ma_system.store_experiences(state, action_indices, rewards, next_state, done)
        state = next_state
        if done:
            state = env.reset()
    print(f"   ✅ Collected 100 experiences")
    
    # 7. Train
    print("\n7. Training for 10 steps...")
    losses = []
    for step in range(10):
        loss = ma_system.train_step()
        if loss is not None:
            losses.append(loss)
            print(f"   Step {step+1}: Loss = {loss:.4f}")
    
    if losses:
        print(f"   ✅ Training successful, avg loss: {np.mean(losses):.4f}")
    else:
        print(f"   ⚠️  Not enough experiences for training yet")
    
    # 8. Update target network
    print("\n8. Updating target network...")
    ma_system.update_target_networks()
    print(f"   ✅ Target network updated")
    
    # 9. Epsilon decay
    print("\n9. Testing epsilon decay...")
    epsilon_before = ma_system.agents[0].epsilon
    ma_system.decay_epsilon()
    epsilon_after = ma_system.agents[0].epsilon
    print(f"   Epsilon: {epsilon_before:.4f} → {epsilon_after:.4f}")
    
    # 10. Save and load
    print("\n10. Testing save/load...")
    os.makedirs("models", exist_ok=True)
    ma_system.save("models/test_pipeline")
    print(f"   ✅ Model saved")
    
    print("\n" + "="*70)
    print("FULL PIPELINE TEST COMPLETE! ✅")
    print("="*70 + "\n")


def demo_random_agent():
    """Demo random agent performance"""
    print("\n" + "="*70)
    print("DEMO: Random Agent Baseline")
    print("="*70 + "\n")
    
    env = MultiAgentGridWorld()
    num_episodes = 10
    
    results = {
        'rewards': [],
        'deliveries': [],
        'collisions': [],
        'steps': []
    }
    
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Random actions
            actions = [np.random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]) 
                      for _ in range(env.num_agents)]
            next_state, rewards, done, info = env.step(actions)
            total_reward += sum(rewards)
            state = next_state
        
        results['rewards'].append(total_reward)
        results['deliveries'].append(env.total_deliveries)
        results['collisions'].append(env.total_collisions)
        results['steps'].append(env.total_steps)
        
        print(f"Episode {ep+1}: Reward={total_reward:.2f}, "
              f"Deliveries={env.total_deliveries}, "
              f"Collisions={env.total_collisions}, "
              f"Steps={env.total_steps}")
    
    print("\n" + "="*70)
    print("Random Agent Results:")
    print(f"  Avg Reward:     {np.mean(results['rewards']):.2f}")
    print(f"  Avg Deliveries: {np.mean(results['deliveries']):.2f}")
    print(f"  Avg Collisions: {np.mean(results['collisions']):.2f}")
    print(f"  Avg Steps:      {np.mean(results['steps']):.2f}")
    print("="*70 + "\n")
    
    return results


def demo_trained_agent(model_path):
    """Demo trained agent performance"""
    print("\n" + "="*70)
    print("DEMO: Trained Agent")
    print("="*70 + "\n")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Train the agent first using train.py")
        return
    
    # Create environment and agent
    env = MultiAgentGridWorld()
    ma_system = MultiAgentDQNSystem(
        num_agents=4,
        state_size_per_agent=4,
        action_size=4,
        shared_network=True,
        config_path="config.yaml"
    )
    
    # Load model
    checkpoint = torch.load(model_path)
    ma_system.agents[0].policy_net.load_state_dict(checkpoint['policy_net'])
    ma_system.agents[0].policy_net.eval()
    print(f"✅ Model loaded from {model_path}\n")
    
    # Run episodes
    num_episodes = 10
    results = {
        'rewards': [],
        'deliveries': [],
        'collisions': [],
        'steps': []
    }
    
    action_map = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Greedy actions (no exploration)
            action_indices = ma_system.select_actions(state, explore=False)
            actions = [action_map[idx] for idx in action_indices]
            next_state, rewards, done, info = env.step(actions)
            total_reward += sum(rewards)
            state = next_state
        
        results['rewards'].append(total_reward)
        results['deliveries'].append(env.total_deliveries)
        results['collisions'].append(env.total_collisions)
        results['steps'].append(env.total_steps)
        
        print(f"Episode {ep+1}: Reward={total_reward:.2f}, "
              f"Deliveries={env.total_deliveries}, "
              f"Collisions={env.total_collisions}, "
              f"Steps={env.total_steps}")
    
    print("\n" + "="*70)
    print("Trained Agent Results:")
    print(f"  Avg Reward:     {np.mean(results['rewards']):.2f}")
    print(f"  Avg Deliveries: {np.mean(results['deliveries']):.2f}")
    print(f"  Avg Collisions: {np.mean(results['collisions']):.2f}")
    print(f"  Avg Steps:      {np.mean(results['steps']):.2f}")
    
    # Calculate success rate
    total_deliveries = sum(results['deliveries'])
    total_attempts = total_deliveries + sum(results['collisions'])
    success_rate = total_deliveries / max(1, total_attempts)
    print(f"  Success Rate:   {success_rate:.2%}")
    print("="*70 + "\n")
    
    return results


def quick_training_test(num_episodes=50):
    """Quick training test"""
    print("\n" + "="*70)
    print(f"QUICK TRAINING TEST ({num_episodes} episodes)")
    print("="*70 + "\n")
    
    # Setup
    env = MultiAgentGridWorld()
    ma_system = MultiAgentDQNSystem(
        num_agents=4,
        state_size_per_agent=4,
        action_size=4,
        shared_network=True,
        config_path="config.yaml"
    )
    
    action_map = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    
    # Training metrics
    episode_rewards = []
    episode_deliveries = []
    
    # Train
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action_indices = ma_system.select_actions(state, explore=True)
            actions = [action_map[idx] for idx in action_indices]
            next_state, rewards, done, info = env.step(actions)
            
            ma_system.store_experiences(state, action_indices, rewards, next_state, done)
            ma_system.train_step()
            
            state = next_state
            total_reward += sum(rewards)
        
        ma_system.decay_epsilon()
        
        episode_rewards.append(total_reward)
        episode_deliveries.append(env.total_deliveries)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_deliveries = np.mean(episode_deliveries[-10:])
            print(f"Episode {episode+1:3d}: "
                  f"Avg Reward={avg_reward:7.2f}, "
                  f"Avg Deliveries={avg_deliveries:.2f}, "
                  f"ε={ma_system.agents[0].epsilon:.3f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(episode_rewards)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(episode_deliveries)
    axes[1].set_title('Deliveries per Episode')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Deliveries')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_training_test.png', dpi=150)
    print(f"\n✅ Training plot saved to quick_training_test.png")
    
    print("\n" + "="*70)
    print("QUICK TRAINING TEST COMPLETE!")
    print("="*70 + "\n")


def visualize_episode():
    """Visualize one episode step by step"""
    print("\n" + "="*70)
    print("VISUALIZING EPISODE")
    print("="*70 + "\n")
    
    env = MultiAgentGridWorld()
    state = env.reset()
    
    print("Initial state:")
    env.render()
    
    for step in range(20):
        actions = [np.random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]) 
                  for _ in range(env.num_agents)]
        
        print(f"\nStep {step+1}: Actions = {[a.name for a in actions]}")
        next_state, rewards, done, info = env.step(actions)
        
        print(f"Rewards: {rewards}")
        env.render()
        
        if done:
            print("\nEpisode finished!")
            break
        
        state = next_state
        
        import time
        time.sleep(0.5)


def main():
    """Main demo function"""
    print("\n" + "="*70)
    print("DAY 5 - DEMO & TESTING SCRIPT")
    print("="*70)
    
    while True:
        print("\nSelect an option:")
        print("1. Test full pipeline")
        print("2. Demo random agent baseline")
        print("3. Demo trained agent (requires trained model)")
        print("4. Quick training test (50 episodes)")
        print("5. Visualize episode")
        print("6. Run all tests")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-6): ").strip()
        
        if choice == '1':
            test_full_pipeline()
        elif choice == '2':
            demo_random_agent()
        elif choice == '3':
            model_path = input("Enter model path (or press Enter for default): ").strip()
            if not model_path:
                model_path = "models/dqn_episode_final.pth"
            demo_trained_agent(model_path)
        elif choice == '4':
            quick_training_test()
        elif choice == '5':
            visualize_episode()
        elif choice == '6':
            test_full_pipeline()
            demo_random_agent()
            quick_training_test()
        elif choice == '0':
            print("\nExiting...")
            break
        else:
            print("\n❌ Invalid choice, please try again")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()