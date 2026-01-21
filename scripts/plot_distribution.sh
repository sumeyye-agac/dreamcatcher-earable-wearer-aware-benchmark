#!/bin/bash
# Quick script to run class distribution visualization

cd "$(dirname "$0")" || exit

# Activate virtual environment
if [ -d "dream-env" ]; then
    source dream-env/bin/activate
else
    echo "Error: dream-env not found"
    exit 1
fi

# Run the visualization script
python src/evaluation/plot_class_distribution.py "$@"
