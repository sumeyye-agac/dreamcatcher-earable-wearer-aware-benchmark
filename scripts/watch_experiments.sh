#!/bin/bash

while true; do
    clear
    echo "=================================="
    echo "DreamCatcher Experiment Monitor"
    echo "Updated: $(date '+%H:%M:%S')"
    echo "=================================="
    echo ""

    # Check running processes
    echo "🔄 Running Processes:"
    ps aux | grep -E "train_kd|eval_teacher" | grep -v grep | awk '{print "  - " $11 " " $15}' | sed 's/--student //' || echo "  No processes running"
    echo ""

    # Check completed runs
    echo "✅ Completed Runs:"
    grep "run_done" results/run_steps.csv | tail -10 | awk -F',' '{print "  - " $2 " (at " substr($1,12,8) ")"}' || echo "  None yet"
    echo ""

    # Check KD progress
    echo "⏳ KD Experiments Status:"
    for run in crnn_rbkd crnn_cbam_rbkdatt tinycnn_rbkd; do
        if [ -f "results/runs/$run/metrics.json" ]; then
            echo "  ✅ $run - COMPLETED"
        elif [ -d "results/runs/$run" ]; then
            last_log=$(grep "$run" results/run_steps.csv | tail -1 | awk -F',' '{print $3}')
            echo "  ⏳ $run - Running (last: $last_log)"
        else
            echo "  ❌ $run - Not started"
        fi
    done
    echo ""

    # Check recent log activity
    echo "📋 Recent Activity (last 10 entries):"
    tail -10 results/run_steps.csv | awk -F',' '{print "  " substr($1,12,8) " | " $2 " | " $3}' || echo "  No logs"
    echo ""

    echo "Refreshing every 10 seconds... (Press Ctrl+C to stop)"
    sleep 10
done
