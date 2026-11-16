#!/usr/bin/env bash
# Re-run WI-007 pipeline v2 after the first attempt crashed on a transient source edit (controller patch in flight).
cd ~/chronotope
while ! grep -q "pipeline v2 done" runs/pipeline.log; do sleep 20; done
for t in prl wi004 wi006; do rm -rf "runs/wi007_qwen3_vl_8b_instruct_$t" "runs/wi007_qwen3_vl_8b_instruct_$t.log"; done
echo "[$(date '+%F %T')] chain3: relaunching pipeline v2 (Qwen3-VL-8B, then UI-TARS native)" >> runs/pipeline.log
studies/pipeline_agents2.sh
