#!/bin/bash

# Day 5 Quick Start Script
# =======================
# Easy setup and run for training

set -e  # Exit on error

echo "=========================================="
echo "   Day 5: Multi-Agent DQN Training"
echo "=========================================="
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python --version
echo ""

# Check if in correct directory
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: config.yaml not found"
    echo "Please run this script from project root directory"
    exit 1
fi

if [ ! -d "src" ]; then
    echo "❌ Error: src/ directory not found"
    echo "Please run this script from project root directory"
    exit 1
fi

echo "✅ Directory structure looks good"
echo ""

# Check dependencies
echo "🔍 Checking dependencies..."
python -c "import torch; import numpy; import matplotlib; import yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Some dependencies missing"
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
else
    echo "✅ All dependencies installed"
fi
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p logs
echo "✅ Directories created"
echo ""

# Menu
echo "Select an option:"
echo "1. Quick test (5 episodes) - verify everything works"
echo "2. Short training (50 episodes) - ~5 minutes"
echo "3. Medium training (200 episodes) - ~20 minutes"
echo "4. Full training (1000 episodes) - ~1 hour"
echo "5. Demo trained agent"
echo "6. Run tests"
echo "0. Exit"
echo ""

read -p "Enter choice (0-6): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Running quick test (5 episodes)..."
        echo "This will verify all components work correctly"
        echo ""
        python -c "
import sys
sys.path.insert(0, 'src')
from train import MultiAgentTrainer

trainer = MultiAgentTrainer('config.yaml')
trainer.num_episodes = 5
trainer.train()
"
        ;;
    
    2)
        echo ""
        echo "🚀 Running short training (50 episodes)..."
        echo "Estimated time: 5 minutes"
        echo ""
        python -c "
import sys
sys.path.insert(0, 'src')
from train import MultiAgentTrainer

trainer = MultiAgentTrainer('config.yaml')
trainer.num_episodes = 50
trainer.save_freq = 25
trainer.train()
trainer.evaluate(num_episodes=5)
"
        ;;
    
    3)
        echo ""
        echo "🚀 Running medium training (200 episodes)..."
        echo "Estimated time: 20 minutes"
        echo ""
        python -c "
import sys
sys.path.insert(0, 'src')
from train import MultiAgentTrainer

trainer = MultiAgentTrainer('config.yaml')
trainer.num_episodes = 200
trainer.save_freq = 50
trainer.train()
trainer.evaluate(num_episodes=10)
"
        ;;
    
    4)
        echo ""
        echo "🚀 Running FULL training (1000 episodes)..."
        echo "Estimated time: 1 hour"
        echo "You can monitor progress in real-time"
        echo ""
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python src/train.py
        else
            echo "Training cancelled"
        fi
        ;;
    
    5)
        echo ""
        echo "🎬 Demo trained agent..."
        if [ ! -f "models/dqn_episode_final.pth" ]; then
            echo "❌ No trained model found at models/dqn_episode_final.pth"
            echo "Please train first using option 2, 3, or 4"
        else
            python demo.py
        fi
        ;;
    
    6)
        echo ""
        echo "🧪 Running tests..."
        python -c "
import sys
sys.path.insert(0, 'src')

print('Test 1: Environment')
from environment import MultiAgentGridWorld, Action
env = MultiAgentGridWorld()
state = env.reset()
print(f'✅ Environment OK - State shape: {state.shape}')

print('\nTest 2: DQN Network')
from dqn import DQN
dqn = DQN(4, 4)
import torch
test_input = torch.randn(1, 4)
output = dqn(test_input)
print(f'✅ DQN OK - Output shape: {output.shape}')

print('\nTest 3: Replay Buffer')
from replay_buffer import ReplayBuffer
buffer = ReplayBuffer(100)
import numpy as np
buffer.push(np.random.randn(4), 0, 1.0, np.random.randn(4), False)
print(f'✅ Replay Buffer OK - Size: {len(buffer)}')

print('\nTest 4: Agent System')
from agent import MultiAgentDQNSystem
ma_system = MultiAgentDQNSystem(4, 4, 4, True)
actions = ma_system.select_actions(state, explore=True)
print(f'✅ Agent System OK - Actions: {actions}')

print('\n✅ All tests passed!')
"
        ;;
    
    0)
        echo "Exiting..."
        exit 0
        ;;
    
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Done!"
echo "=========================================="

# Show results location
if [ -d "logs" ] && [ "$(ls -A logs)" ]; then
    latest_log=$(ls -t logs/ | head -1)
    echo ""
    echo "📊 Results saved to:"
    echo "   logs/$latest_log/"
    echo ""
    echo "To view training curves:"
    echo "   open logs/$latest_log/training_curves.png"
fi

if [ -d "models" ] && [ "$(ls -A models)" ]; then
    echo ""
    echo "💾 Models saved to:"
    echo "   models/"
    ls -lh models/
fi

echo ""
echo "Next steps:"
echo "1. Check training curves in logs/"
echo "2. Analyze metrics.json"
echo "3. Test trained model with option 5"
echo "4. Continue to Day 6 for optimization"
echo ""