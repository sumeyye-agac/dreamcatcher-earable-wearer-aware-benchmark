#!/usr/bin/env bash

# Monitor balanced4 experiments progress

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║        Balanced 4-Class Experiments Monitor                   ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""

    echo "=== Process Status ==="
    if pgrep -f "train_baseline_balanced4" > /dev/null; then
        echo "✓ Baseline experiments running (PID: $(pgrep -f train_baseline_balanced4))"
    else
        echo "✗ Baseline experiments not running"
    fi

    if pgrep -f "train_kd_balanced4" > /dev/null; then
        echo "✓ KD experiments running (PID: $(pgrep -f train_kd_balanced4))"
    else
        echo "✗ KD experiments not running"
    fi

    if pgrep -f "eval_teacher_balanced4" > /dev/null; then
        echo "✓ Teacher evaluation running (PID: $(pgrep -f eval_teacher_balanced4))"
    else
        echo "✗ Teacher evaluation not running"
    fi

    echo ""
    echo "=== Recent Progress (last 10 steps) ==="
    tail -10 results/run_steps.csv | grep -E "balanced4" | cut -d',' -f1,2,3,4 | column -t -s',' || echo "No steps yet"

    echo ""
    echo "=== Completed Runs ==="
    if [ -d "results/runs" ]; then
        ls -1 results/runs/ | grep "balanced4" | while read run; do
            if [ -f "results/runs/$run/metrics.json" ]; then
                acc=$(grep -o '"acc": [0-9.]*' "results/runs/$run/metrics.json" | head -1 | cut -d' ' -f2)
                f1=$(grep -o '"f1_macro": [0-9.]*' "results/runs/$run/metrics.json" | head -1 | cut -d' ' -f2)
                echo "  ✓ $run (Acc: $acc, F1: $f1)"
            fi
        done
    fi
    [ ! -d "results/runs" ] || [ -z "$(ls -1 results/runs/ | grep balanced4)" ] && echo "  (none yet)"

    echo ""
    echo "Press Ctrl+C to exit. Refreshing every 10 seconds..."
    sleep 10
done
