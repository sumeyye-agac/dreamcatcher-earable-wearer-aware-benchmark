#!/usr/bin/env bash
set -e

python3 -m src.evaluation.leaderboard_report --topk 10
python3 -m src.evaluation.pareto_plot --x cpu_latency_ms --y test_f1 --out results/plots/pareto_front.png --title "Pareto Frontier: test_f1 vs cpu_latency_ms"
python3 -m src.evaluation.pareto_plot --x params --y test_f1 --out results/plots/pareto_front_params.png --title "Pareto Frontier: test_f1 vs params"
