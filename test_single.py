#!/usr/bin/env python3
"""
test_single.py - Test single query optimization
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.optimization.optimization_engine import create_optimization_engine
from src.utils.data_loader import get_random_query

print("=" * 80)
print("Phase 4 - Single Query Test")
print("=" * 80)
print()

# Get a random query
print("Loading dataset...")
item = get_random_query(seed=23)

print(f"Image: {item['image_id']}")
print(f"Query: {item['query']}")
print()

# Create engine
print("Creating optimization engine...")
engine = create_optimization_engine()

# Optimize
print("Starting optimization...")
print()
result = engine.optimize_single_query(
    image_path=item['image_path'],
    query=item['query'],
    query_id="test_single",
    image_id=item['image_id'],
    save_trajectory=True
)

# Results
print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Success: {result['success']}")
print(f"Final Score: {result['final_score']:.3f}")
print(f"Total Steps: {result['total_steps']}")
print(f"Successful Context: {result.get('successful_context', 'N/A')}")

if result['success']:
    print()
    print(f"Initial: {item['query']}")
    print(f"Final:   {result['final_query'][:150]}...")
    if 'llm_tier' in result:
        print(f"LLM Tier: {result['llm_tier']}")

print("=" * 80)
print()
print("✅ Test completed!")
print(f"📁 Trajectory saved to: trajectories/test_single.json")