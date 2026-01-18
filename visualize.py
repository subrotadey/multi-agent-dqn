import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.environment import MultiAgentGridWorld, Action
import random
import time

def visualize_grid(env):
    """Create a visual grid using matplotlib"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw grid
    for i in range(env.grid_size + 1):
        ax.plot([0, env.grid_size], [i, i], 'k-', linewidth=0.5)
        ax.plot([i, i], [0, env.grid_size], 'k-', linewidth=0.5)
    
    # Draw location A (pickup) - Green
    ax.add_patch(patches.Rectangle((0, env.grid_size-1), 1, 1, 
                                   facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax.text(0.5, env.grid_size-0.5, 'A\nPickup', ha='center', va='center', fontsize=12, weight='bold')
    
    # Draw location B (dropoff) - Red
    ax.add_patch(patches.Rectangle((4, 0), 1, 1, 
                                   facecolor='lightcoral', edgecolor='red', linewidth=2))
    ax.text(4.5, 0.5, 'B\nDropoff', ha='center', va='center', fontsize=12, weight='bold')
    
    # Draw agents
    colors = ['blue', 'orange', 'purple', 'brown']
    for agent in env.agents:
        col, row = agent.position[1], env.grid_size - 1 - agent.position[0]
        
        # Agent circle
        circle = plt.Circle((col + 0.5, row + 0.5), 0.3, 
                           color=colors[agent.id], alpha=0.7)
        ax.add_patch(circle)
        
        # Agent ID
        label = f"{agent.id}"
        if agent.has_item:
            label += "*"
        ax.text(col + 0.5, row + 0.5, label, ha='center', va='center', 
               fontsize=14, weight='bold', color='white')
    
    # Stats
    stats_text = f"Steps: {env.total_steps} | Deliveries: {env.total_deliveries} | Collisions: {env.total_collisions}"
    ax.text(2.5, -0.5, stats_text, ha='center', fontsize=11, weight='bold')
    
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig, ax

def animate_simulation(num_steps=20):
    """Run simulation with visual updates"""
    env = MultiAgentGridWorld()
    env.reset()
    
    plt.ion()  # Interactive mode on
    
    for step in range(num_steps):
        # Clear and redraw
        plt.clf()
        visualize_grid(env)
        plt.title(f"Multi-Agent Grid World - Step {step + 1}", fontsize=16, weight='bold')
        plt.pause(2)  # Pause for 0.5 seconds
        
        # Take random actions
        actions = [random.choice(Action.get_all_actions()) for _ in range(env.num_agents)]
        next_state, rewards, done, info = env.step(actions)
        
        print(f"\nStep {step + 1}:")
        print(f"Actions: {[a.name for a in actions]}")
        print(f"Rewards: {rewards}")
        print(f"Info: {info}")
        
        if done:
            print("\nEpisode finished!")
            break
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    print("Starting visualization...")
    animate_simulation(num_steps=30)