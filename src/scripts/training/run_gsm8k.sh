#!/usr/bin/env bash
# GSM8K benchmark - runs in background, outputs to log
# usage: bash scripts/run_gsm8k.sh

cd "$(dirname "$0")/.."
PYTHONPATH=. nohup python -u scripts/bench_gsm8k.py > logs/gsm8k_$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID=$!
echo "PID: $PID"
echo "log: logs/gsm8k_$(date +%Y%m%d_%H%M%S).log"
echo "watch: tail -f logs/gsm8k_*.log"
echo "kill: kill $PID"
echo $PID > logs/gsm8k.pid
