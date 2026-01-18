import os
import time
from src.environment import MultiAgentGridWorld, Action
import random

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def colored_render(env):
    """Render with colors in terminal"""
    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'green': '\033[92m',
        'red': '\033[91m',
        'blue': '\033[94m',
        'yellow': '\033[93m',
        'cyan': '\033[96m',
        'magenta': '\033[95m',
    }
    
    grid = [['.' for _ in range(env.grid_size)] for _ in range(env.grid_size)]
    
    # Mark locations
    grid[env.location_A[0]][env.location_A[1]] = f"{COLORS['green']}A{COLORS['reset']}"
    grid[env.location_B[0]][env.location_B[1]] = f"{COLORS['red']}B{COLORS['reset']}"
    
    # Mark agents with different colors
    agent_colors = ['blue', 'cyan', 'magenta', 'yellow']
    for agent in env.agents:
        r, c = agent.position
        color = COLORS[agent_colors[agent.id]]
        symbol = str(agent.id) if not agent.has_item else f"{agent.id}*"
        grid[r][c] = f"{color}{symbol}{COLORS['reset']}"
    
    # Print header
    print("\n" + "="*50)
    print("       🎮 MULTI-AGENT GRID WORLD 🎮")
    print("="*50)
    
    # Print grid with border
    print("\n  +" + "---"*env.grid_size + "+")
    for i, row in enumerate(grid):
        print(f"{i} | " + "  ".join(row) + " |")
    print("  +" + "---"*env.grid_size + "+")
    print("    " + "  ".join([str(i) for i in range(env.grid_size)]))
    
    # Print stats
    print("\n" + "="*50)
    print(f"📊 Steps: {env.total_steps} | 📦 Deliveries: {env.total_deliveries} | ⚠️  Collisions: {env.total_collisions}")
    print("="*50)
    
    # Print agent details
    print("\n🤖 Agent Status:")
    for agent in env.agents:
        status = "✅ Has Item" if agent.has_item else "❌ No Item"
        print(f"  Agent {agent.id}: Position {agent.position} | {status} | Deliveries: {agent.total_deliveries}")
    print()

def run_animation(num_steps=50, delay=0.5):
    """Run animated simulation"""
    env = MultiAgentGridWorld()
    env.reset()
    
    for step in range(num_steps):
        clear_screen()
        colored_render(env)
        
        # Take random actions
        actions = [random.choice(Action.get_all_actions()) for _ in range(env.num_agents)]
        print(f"🎯 Actions: {[a.name for a in actions]}")
        
        time.sleep(delay)
        
        next_state, rewards, done, info = env.step(actions)
        
        if done:
            clear_screen()
            colored_render(env)
            print("\n🏁 EPISODE FINISHED! 🏁")
            print(f"\n📈 Final Stats:")
            print(f"   Total Deliveries: {env.total_deliveries}")
            print(f"   Total Collisions: {env.total_collisions}")
            print(f"   Total Steps: {env.total_steps}")
            break
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    print("Starting terminal animation...")
    print("(Press Ctrl+C to stop)\n")
    time.sleep(2)
    run_animation(num_steps=100, delay=2)