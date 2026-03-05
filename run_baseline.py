#!/usr/bin/env python3
"""
run_baseline.py - Run baseline batch experiment (Phase 3)
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.optimization.optimization_engine import create_optimization_engine
from src.utils.data_loader import load_dataset

# 1. 设置命令行参数解析
parser = argparse.ArgumentParser(description="Run baseline batch experiment (Phase 3)")
parser.add_argument('--start-index', type=int, default=0, help='Start index in dataset (default: 0)')
parser.add_argument('--batch-size', type=int, default=30, help='Number of queries to process (default: 30)')
args = parser.parse_args()

START_INDEX = args.start_index
BATCH_SIZE = args.batch_size

print("=" * 80)
print("Phase 3 - Baseline Batch Experiment")
print("=" * 80)
print()

print(f"Configuration:")
print(f"  Start Index: {START_INDEX}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  End Index: {START_INDEX + BATCH_SIZE}")
print()

# Load dataset
print("Loading dataset...")
dataset = load_dataset()
print(f"Dataset size: {len(dataset)}")
print()

# Create engine (baseline = rule-based)
print("Creating baseline engine (rule-based chain selection)...")
engine = create_optimization_engine()
print()

# Run batch
print("Starting batch optimization...")
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
print("BASELINE RESULTS")
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
        'phase': 'baseline',
        'start_index': START_INDEX,
        'end_index': START_INDEX + BATCH_SIZE,
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
print("Next step: Run 'python analyze_trajectories.py' to analyze results")