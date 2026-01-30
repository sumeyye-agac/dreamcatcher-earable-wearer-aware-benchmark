#!/bin/bash
# Auto-launch CRNN_Improved_CBAM after tinycnn models complete
set -e

echo "============================================================"
echo "Auto-Launcher: CRNN_Improved_CBAM"
echo "Waiting for tinycnn_cbam models to complete..."
echo "============================================================"
echo ""

# Models to wait for
WAIT_MODELS=(
    "tinycnn_cbam_rr2_sk3"
    "tinycnn_cbam_rr4_sk3"
)

# Wait for models to finish
while true; do
    RUNNING=0
    for model in "${WAIT_MODELS[@]}"; do
        if ps aux | grep "src.training.train" | grep -q "run_name $model"; then
            RUNNING=$((RUNNING + 1))
        fi
    done

    if [ $RUNNING -eq 0 ]; then
        echo ""
        echo "============================================================"
        echo "TinyCNN models completed! Starting CRNN_Improved_CBAM..."
        echo "============================================================"
        echo ""
        break
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Still running: $RUNNING/2 tinycnn models"
    sleep 300  # Check every 5 minutes
done

# Change to project directory
cd /Users/sumeyyeagac/Desktop/GitHub_Sumeyye/dreamcatcher-earable-wearer-aware-benchmark

# Launch CRNN_Improved_CBAM
bash experiments/train_crnn_improved_cbam.sh

echo ""
echo "============================================================"
echo "CRNN_Improved_CBAM launched successfully!"
echo "============================================================"
echo ""
echo "All running models:"
ps aux | grep "src.training.train" | grep -v grep | grep -o "run_name [^ ]*" | awk '{print "  - " $2}'
echo ""
echo "Monitor: tail -f logs/crnn_improved_cbam.log"