#!/bin/bash
# scripts/run_batch_experiment.sh
# Batch experiment script for Phase 4

set -e

echo "=========================================="
echo "Phase 4 Batch Experiment"
echo "=========================================="

# Configuration
PROJECT_ROOT="/data/heyuji/exp_multiLLM_optimizer"
BATCH_SIZE=30
START_INDEX=0

cd "$PROJECT_ROOT"

# Step 1: Run batch optimization
echo ""
echo "Step 1: Running batch optimization..."
echo "  - Batch size: $BATCH_SIZE"
echo "  - Start index: $START_INDEX"
echo ""

python main.py batch \
    --start-index $START_INDEX \
    --batch-size $BATCH_SIZE

# Step 2: Analyze trajectories
echo ""
echo "Step 2: Analyzing trajectories..."
echo ""

python main.py analyze \
    --trajectory-dir trajectories \
    --save-stats

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Trajectories: trajectories/*.json"
echo "  - Batch summary: batch_summary_${START_INDEX}_*.json"
echo "  - Strategy stats: strategy_stats.json"
echo ""