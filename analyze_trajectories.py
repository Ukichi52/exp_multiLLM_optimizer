#!/usr/bin/env python3
"""
analyze_trajectories.py - Analyze collected trajectories
"""
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analysis.strategy_analytics import StrategyAnalytics

# 1. 设置命令行参数解析
parser = argparse.ArgumentParser(description="Analyze collected trajectories")
parser.add_argument('--trajectory-dir', type=str, default='trajectories', help='Directory containing trajectory JSONs (default: trajectories)')
parser.add_argument('--output-stats', type=str, default='strategy_stats.json', help='Output path for the stats JSON (default: strategy_stats.json)')
args = parser.parse_args()

TRAJECTORY_DIR = args.trajectory_dir
OUTPUT_STATS = args.output_stats

print("=" * 80)
print("Trajectory Analysis")
print("=" * 80)
print()

# Create analytics
print(f"Loading trajectories from: {TRAJECTORY_DIR}/")
# 注意这里传入自定义的路径
analytics = StrategyAnalytics(trajectory_dir=TRAJECTORY_DIR)
analytics.load_trajectories()
print()

if not analytics.trajectories:
    print(f"⚠️  No trajectories found in {TRAJECTORY_DIR}/")
    print("   Please check your path or run baseline first.")
    sys.exit(1)

# Print report
analytics.print_report()

# Save detailed stats
print("\nSaving detailed statistics...")

recommendations = analytics.export_recommendations()

with open(OUTPUT_STATS, 'w') as f:
    json.dump({
        'total_trajectories': len(analytics.trajectories),
        'sub_stats': {
            sub_id: {
                'usage_count': stats['usage_count'],
                'avg_delta': stats['avg_delta'],
                'success_contribution': stats['success_contribution'],
                'best_position': analytics.get_sub_best_position(sub_id),
                'best_context': analytics.get_sub_best_context(sub_id)
            }
            for sub_id, stats in analytics.sub_stats.items()
        },
        'chain_stats': {
            name: {
                'usage_count': stats['usage_count'],
                'success_count': stats['success_count'],
                'success_rate': stats.get('success_rate', 0.0),
                'avg_score': stats.get('avg_score', 0.0)
            }
            for name, stats in analytics.chain_stats.items()
        },
        'recommendations': recommendations
    }, f, indent=2, default=str)

print(f"✅ Detailed stats saved to: {OUTPUT_STATS}")
print()
print("Next step: Run 'python run_adaptive.py' to use learned strategies")