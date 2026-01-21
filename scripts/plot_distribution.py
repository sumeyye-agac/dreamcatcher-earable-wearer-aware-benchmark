#!/usr/bin/env python
"""
Quick runner script for class distribution visualization.
Usage: python plot_distribution.py [--confusion-matrix PATH] [--run-steps PATH] [--output-dir PATH]
"""

import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    repo_root = Path(__file__).parent
    script_path = repo_root / "src" / "evaluation" / "plot_class_distribution.py"

    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)

    # Run the visualization script with all arguments passed through
    result = subprocess.run([sys.executable, str(script_path)] + sys.argv[1:], cwd=str(repo_root))

    sys.exit(result.returncode)
