#!/usr/bin/env bash
cd ~/chronotope
sleep 30
while pgrep -f "studies/wi006_" > /dev/null; do sleep 30; done
PYTHONPATH=src .venv/bin/python studies/c1_balloon_drift.py --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:8000/v1 --thinking --reasoning-effort low --max-tokens 4096 --subset-thinking --window-s 32 --out runs/wi004_qwen38_think --max-calls 200 > runs/wi004_qwen38_think.log 2>&1
studies/pipeline_agents.sh
