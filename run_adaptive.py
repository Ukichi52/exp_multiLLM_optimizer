#!/usr/bin/env python3
"""
run_adaptive.py - Run adaptive batch experiment (Phase 4)
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.optimization.engine_factory import create_adaptive_engine
from src.utils.data_loader import load_dataset
from src.analysis.strategy_analytics import StrategyAnalytics

# 1. 设置命令行参数解析
parser = argparse.ArgumentParser(description="Run adaptive batch experiment (Phase 4)")
parser.add_argument('--start-index', type=int, default=30, help='Start index in dataset (default: 30)')
parser.add_argument('--batch-size', type=int, default=30, help='Number of queries to process (default: 30)')
parser.add_argument('--exploration-rate', type=float, default=0.2, help='Exploration rate (default: 0.2)')
parser.add_argument('--baseline-path', type=str, default=None, help='Specific baseline file to compare against (optional)')
parser.add_argument('--train-trajectory-dir', type=str, default='trajectories', help='Directory to load past trajectories for learning')
args = parser.parse_args()

START_INDEX = args.start_index
BATCH_SIZE = args.batch_size
EXPLORATION_RATE = args.exploration_rate

print("=" * 80)
print("Phase 4 - Adaptive Batch Experiment")
print("=" * 80)
print()

print(f"Configuration:")
print(f"  Start Index: {START_INDEX}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  End Index: {START_INDEX + BATCH_SIZE}")
print(f"  Exploration Rate: {EXPLORATION_RATE}")
print()

# Load analytics
print(f"Loading analytics from {args.train_trajectory_dir}/...")
analytics = StrategyAnalytics(trajectory_dir=args.train_trajectory_dir)
analytics.load_trajectories()
# analytics = StrategyAnalytics()
# analytics.load_trajectories()

if len(analytics.trajectories) < 10:
    print()
    print(f"⚠️  WARNING: Only {len(analytics.trajectories)} trajectories found!")
    print("   Recommendation: Run 'python run_baseline.py' first to collect more data.")
    print()
    response = input("Continue anyway? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)

print(f"Loaded {len(analytics.trajectories)} trajectories")
print()

# Show top strategies learned
print("Top 3 strategies learned:")
for i, (chain_name, rate) in enumerate(analytics.get_top_chains(3), 1):
    print(f"  {i}. {chain_name}: {rate:.1%} success rate")
print()

# Load dataset
print("Loading dataset...")
dataset = load_dataset()
print(f"Dataset size: {len(dataset)}")
print()

# Create adaptive engine
print("Creating adaptive engine (analytics-driven selection)...")
engine = create_adaptive_engine(
    analytics=analytics,
    exploration_rate=EXPLORATION_RATE
)
print()

# Run batch
print("Starting adaptive optimization...")
print("This will take approximately 30 minutes...")
print()

results = engine.optimize_batch(
    dataset=dataset,
    start_idx=START_INDEX,
    end_idx=START_INDEX + BATCH_SIZE,
    save_trajectories=True
)

# Summary
success_count = sum(1 for r in results if r.get('success', False))
avg_score = sum(r.get('final_score', 0.0) for r in results) / len(results)
avg_steps = sum(r.get('total_steps', 0) for r in results) / len(results)

print()
print("=" * 80)
print("ADAPTIVE RESULTS")
print("=" * 80)
print(f"Total Queries: {len(results)}")
print(f"Successes: {success_count} ({success_count/len(results)*100:.1f}%)")
print(f"Average Score: {avg_score:.3f}")
print(f"Average Steps: {avg_steps:.1f}")
print("=" * 80)
print()

# Save summary
import json
summary_path = f"batch_summary_{START_INDEX}_{START_INDEX + BATCH_SIZE}.json"
with open(summary_path, 'w') as f:
    json.dump({
        'phase': 'adaptive',
        'start_index': START_INDEX,
        'end_index': START_INDEX + BATCH_SIZE,
        'exploration_rate': EXPLORATION_RATE,
        'total': len(results),
        'success_count': success_count,
        'success_rate': success_count / len(results),
        'avg_score': avg_score,
        'avg_steps': avg_steps,
        'results': results
    }, f, indent=2)

print(f"✅ Summary saved to: {summary_path}")
print(f"📁 Trajectories saved to: trajectories/query_*.json")
print()

# 自动推导 Baseline 路径进行对比
if args.baseline_path:
    baseline_path = args.baseline_path
else:
    # 如果没指定，默认找上一个批次的 baseline 文件
    prev_start = max(0, START_INDEX - BATCH_SIZE)
    baseline_path = f"batch_summary_{prev_start}_{START_INDEX}.json"

if Path(baseline_path).exists():
    print("=" * 80)
    print(f"COMPARISON WITH BASELINE ({baseline_path})")
    print("=" * 80)
    
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    print(f"Baseline Success Rate: {baseline['success_rate']:.1%}")
    print(f"Adaptive Success Rate: {success_count / len(results):.1%}")
    print(f"Improvement: {(success_count / len(results) - baseline['success_rate']):.1%}")
    print()
    
    print(f"Baseline Avg Score: {baseline['avg_score']:.3f}")
    print(f"Adaptive Avg Score: {avg_score:.3f}")
    print(f"Improvement: {avg_score - baseline['avg_score']:+.3f}")
    print("=" * 80)
    print()
else:
    print(f"ℹ️ Could not find baseline file '{baseline_path}' for comparison.")

print("✅ Adaptive experiment completed!")