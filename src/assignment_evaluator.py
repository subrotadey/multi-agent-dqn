"""
Assignment Performance Evaluator
=================================
Evaluates agent performance based on assignment criteria
"""

import numpy as np
import yaml


class AssignmentEvaluator:
    """
    Evaluates performance based on assignment requirements:
    - Performance Points (B): 2 points for >95% success, 1 point for >85%
    - Collision Budget: <500 collisions for 2 points
    - Steps per Delivery: <20 steps average
    - Scaling Factor: α = 1 - (33/200) × max(0, C - B)
    """
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.eval_config = self.config.get('evaluation', {})
        
        # Assignment criteria
        self.target_success_rate = self.eval_config.get('target_success_rate', 0.95)
        self.acceptable_success_rate = self.eval_config.get('acceptable_success_rate', 0.85)
        self.max_collisions_for_2pts = self.eval_config.get('max_collisions', 500)
        self.max_steps_per_delivery = self.eval_config.get('max_steps_per_delivery', 20)
    
    def calculate_performance_points(self, success_rate, total_collisions, avg_steps_per_delivery):
        """
        Calculate Performance Points (B)
        
        Returns:
            int: 0, 1, or 2 points
        """
        if success_rate >= self.target_success_rate and \
           total_collisions < self.max_collisions_for_2pts and \
           avg_steps_per_delivery < self.max_steps_per_delivery:
            return 2
        elif success_rate >= self.acceptable_success_rate:
            return 1
        else:
            return 0
    
    def calculate_scaling_factor(self, cost, performance_points):
        """
        Calculate scaling factor α
        
        Formula: α = 1 - (33/200) × max(0, C - B)
        
        Args:
            cost: Total cost (complexity)
            performance_points: Performance points (0, 1, or 2)
        
        Returns:
            float: Scaling factor
        """
        alpha = 1 - (33/200) * max(0, cost - performance_points)
        return alpha
    
    def evaluate_training(self, total_deliveries, total_attempts, 
                         total_collisions, total_steps):
        """
        Evaluate training performance
        
        Returns:
            dict: Evaluation results
        """
        success_rate = total_deliveries / max(1, total_attempts)
        avg_steps_per_delivery = total_steps / max(1, total_deliveries)
        
        performance_points = self.calculate_performance_points(
            success_rate, total_collisions, avg_steps_per_delivery
        )
        
        return {
            'success_rate': success_rate,
            'total_deliveries': total_deliveries,
            'total_collisions': total_collisions,
            'avg_steps_per_delivery': avg_steps_per_delivery,
            'performance_points': performance_points,
            'meets_95_target': success_rate >= self.target_success_rate,
            'meets_85_target': success_rate >= self.acceptable_success_rate,
            'collision_budget_ok': total_collisions < self.max_collisions_for_2pts,
            'steps_budget_ok': avg_steps_per_delivery < self.max_steps_per_delivery
        }
    
    def evaluate_final(self, eval_results_list):
        """
        Evaluate final performance over multiple evaluation episodes
        
        Args:
            eval_results_list: List of dicts from multiple eval episodes
        
        Returns:
            dict: Final evaluation
        """
        deliveries = [r['deliveries'] for r in eval_results_list]
        collisions = [r['collisions'] for r in eval_results_list]
        steps = [r['steps'] for r in eval_results_list]
        
        total_deliveries = sum(deliveries)
        total_collisions = sum(collisions)
        total_attempts = total_deliveries + total_collisions
        total_steps = sum(steps)
        
        return self.evaluate_training(
            total_deliveries, total_attempts, 
            total_collisions, total_steps
        )
    
    def print_evaluation(self, eval_dict):
        """Print formatted evaluation results"""
        print("\n" + "="*70)
        print("📊 ASSIGNMENT PERFORMANCE EVALUATION")
        print("="*70)
        
        print(f"\n🎯 Success Metrics:")
        print(f"   Success Rate:           {eval_dict['success_rate']:.2%}")
        print(f"   Total Deliveries:       {eval_dict['total_deliveries']}")
        print(f"   Total Collisions:       {eval_dict['total_collisions']}")
        print(f"   Avg Steps/Delivery:     {eval_dict['avg_steps_per_delivery']:.2f}")
        
        print(f"\n✅ Target Achievement:")
        status_95 = "✅ YES" if eval_dict['meets_95_target'] else "❌ NO"
        status_85 = "✅ YES" if eval_dict['meets_85_target'] else "❌ NO"
        status_col = "✅ YES" if eval_dict['collision_budget_ok'] else "❌ NO"
        status_steps = "✅ YES" if eval_dict['steps_budget_ok'] else "❌ NO"
        
        print(f"   >95% Success:           {status_95}")
        print(f"   >85% Success:           {status_85}")
        print(f"   <500 Collisions:        {status_col}")
        print(f"   <20 Steps/Delivery:     {status_steps}")
        
        print(f"\n🏆 Performance Points:")
        print(f"   Points Earned:          {eval_dict['performance_points']}/2")
        
        if eval_dict['performance_points'] == 2:
            print("   Grade: EXCELLENT ⭐⭐")
        elif eval_dict['performance_points'] == 1:
            print("   Grade: GOOD ⭐")
        else:
            print("   Grade: NEEDS IMPROVEMENT")
        
        print("="*70 + "\n")


# Example usage
def test_evaluator():
    """Test the evaluator"""
    evaluator = AssignmentEvaluator()
    
    # Example 1: Excellent performance
    print("Example 1: Excellent Performance")
    result = evaluator.evaluate_training(
        total_deliveries=950,
        total_attempts=1000,
        total_collisions=50,
        total_steps=15000
    )
    evaluator.print_evaluation(result)
    
    # Example 2: Good performance
    print("\nExample 2: Good Performance")
    result = evaluator.evaluate_training(
        total_deliveries=870,
        total_attempts=1000,
        total_collisions=130,
        total_steps=20000
    )
    evaluator.print_evaluation(result)
    
    # Example 3: Needs improvement
    print("\nExample 3: Needs Improvement")
    result = evaluator.evaluate_training(
        total_deliveries=700,
        total_attempts=1000,
        total_collisions=300,
        total_steps=25000
    )
    evaluator.print_evaluation(result)


if __name__ == "__main__":
    test_evaluator()