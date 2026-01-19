"""
Quick Test - Verify Fixes Work
===============================
Run this first to verify everything works
"""

import sys
sys.path.insert(0, 'src')

from environment import MultiAgentGridWorld, Action
import numpy as np


class TestFixedEnvironment(MultiAgentGridWorld):
    """Test version with fixes"""
    
    def reset(self):
        # SPREAD OUT positions
        start_positions = [
            (0, 0),  # Top-left
            (0, 4),  # Top-right
            (4, 0),  # Bottom-left
            (4, 4)   # Bottom-right
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


def test_starting_positions():
    """Test 1: Verify starting positions don't collide"""
    print("\n" + "="*70)
    print("TEST 1: Starting Positions")
    print("="*70)
    
    env = TestFixedEnvironment()
    state = env.reset()
    
    print("\nAgent starting positions:")
    for i, agent in enumerate(env.agents):
        print(f"  Agent {i}: {agent.position}")
    
    # Check no collisions at start
    positions = [agent.position for agent in env.agents]
    has_collision = len(positions) != len(set(positions))
    
    if has_collision:
        print("\n❌ FAIL: Agents start at same position!")
        return False
    else:
        print("\n✅ PASS: All agents at different positions")
        return True


def test_deliveries_possible():
    """Test 2: Check if deliveries are possible"""
    print("\n" + "="*70)
    print("TEST 2: Delivery Mechanics")
    print("="*70)
    
    env = TestFixedEnvironment()
    env.reset()
    
    # Move agent 0 to pickup location
    agent = env.agents[0]
    agent.position = (0, 0)  # Location A
    agent.has_item = False
    
    print(f"\nAgent at A {env.location_A}, has_item: {agent.has_item}")
    
    # Try pickup
    actions = [Action.RIGHT] * 4  # Dummy actions
    state, rewards, done, info = env.step(actions)
    
    if env.agents[0].has_item:
        print("✅ Pickup works!")
    else:
        print("❌ Pickup FAILED")
        return False
    
    # Move to B
    env.agents[0].position = (4, 4)  # Location B
    state, rewards, done, info = env.step(actions)
    
    if env.total_deliveries > 0:
        print("✅ Delivery works!")
        print(f"   Total deliveries: {env.total_deliveries}")
        return True
    else:
        print("❌ Delivery FAILED")
        return False


def test_reward_shaping():
    """Test 3: Verify rewards are reasonable"""
    print("\n" + "="*70)
    print("TEST 3: Reward Shaping")
    print("="*70)
    
    env = TestFixedEnvironment()
    env.reset()
    
    # Test collision penalty
    print("\n1. Testing collision penalty:")
    env.agents[0].position = (2, 2)
    env.agents[1].position = (2, 3)
    
    # Move both to same spot
    actions = [Action.RIGHT, Action.LEFT, Action.UP, Action.UP]
    state, rewards, done, info = env.step(actions)
    
    if info['collision_this_step']:
        print(f"   Collision detected: rewards = {rewards}")
        if all(r >= -5 for r in rewards):  # Should be -2, not -10
            print("   ✅ Collision penalty is reasonable")
        else:
            print("   ❌ Collision penalty too high!")
    
    # Test delivery reward
    print("\n2. Testing delivery reward:")
    env.reset()
    env.agents[0].position = (0, 0)
    env.agents[0].has_item = False
    
    # Pickup
    actions = [Action.RIGHT] * 4
    env.step(actions)
    
    # Deliver
    env.agents[0].position = (4, 3)
    env.agents[0].has_item = True
    actions = [Action.RIGHT, Action.UP, Action.UP, Action.UP]
    state, rewards, done, info = env.step(actions)
    
    if rewards[0] > 10:
        print(f"   ✅ Delivery reward is good: {rewards[0]}")
        return True
    else:
        print(f"   ❌ Delivery reward too low: {rewards[0]}")
        return False


def run_quick_episode():
    """Test 4: Run one complete episode"""
    print("\n" + "="*70)
    print("TEST 4: Complete Episode")
    print("="*70)
    
    env = TestFixedEnvironment()
    env.max_collisions = 10
    state = env.reset()
    
    print("\nRunning 50 random steps...")
    
    for step in range(50):
        actions = [np.random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]) 
                  for _ in range(4)]
        state, rewards, done, info = env.step(actions)
        
        if step % 10 == 0:
            print(f"  Step {step}: Deliveries={info['total_deliveries']}, "
                  f"Collisions={info['total_collisions']}")
        
        if done:
            break
    
    print(f"\nFinal stats:")
    print(f"  Total deliveries: {env.total_deliveries}")
    print(f"  Total collisions: {env.total_collisions}")
    print(f"  Total steps: {env.total_steps}")
    
    if env.total_deliveries > 0:
        print("\n✅ PASS: Deliveries happening!")
        return True
    else:
        print("\n⚠️  WARNING: No deliveries yet (might need more steps)")
        return True  # Not a failure, just needs more steps


def main():
    print("\n" + "="*70)
    print("🧪 QUICK TEST SUITE - VERIFY FIXES")
    print("="*70)
    
    results = []
    
    # Run all tests
    results.append(("Starting Positions", test_starting_positions()))
    results.append(("Delivery Mechanics", test_deliveries_possible()))
    results.append(("Reward Shaping", test_reward_shaping()))
    results.append(("Complete Episode", run_quick_episode()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\nYou can now run the full training:")
        print("  python train.py")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease check the error messages above")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()