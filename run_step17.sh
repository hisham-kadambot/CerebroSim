#!/usr/bin/env bash
# Step 17: two more clot sizes for the AI model (40 % and 60 %), at all five levels: normal, low / high inflow
# (Step 9) and the stroke patients' blood pressures 118 / 146 mmHg (Step 10). Waits for the AI training (Step 14)
# to free the GPU. 40 % on the 0.1 mm grid (~6.5 h), then 60 % on the 0.05 mm grid like 70 % (~2 days).
# Skips a run whose results are complete, so it can be restarted after a stop.
#   bash run_step17.sh            (from Git Bash, in this folder)
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

while ! grep -q "^exit" ai/runs_step14.log 2>/dev/null; do sleep 60; done

done_already() { grep -q '"done": true' "results/$1/meta.json" 2>/dev/null; }

run() {  # run <case> <tag suffix> <beats> <dt-scale> <level arguments...>
    local tag="$1$2" case=$1 beats=$3 dts=$4
    shift 4
    if done_already "$tag"; then echo "$(date +%H:%M) skip $tag (done)"; return; fi
    echo "$(date +%H:%M) start $tag"
    uv run simulate.py "$case" "$@" --beats "$beats" --dt-scale "$dts" > "results/$tag.log" 2>&1 \
        && echo "$(date +%H:%M) done $tag" || echo "$(date +%H:%M) FAILED $tag (see results/$tag.log)"
}

levels() {  # levels <case> <beats>: the same time steps as Steps 9 and 10
    run "$1" ""          "$2" 1.0
    run "$1" "_flow70"   "$2" 1.0  --flow 0.7
    run "$1" "_flow130"  "$2" 0.75 --flow 1.3
    run "$1" "_map118"   "$2" 0.75 --map 118
    run "$1" "_map146"   "$2" 0.6  --map 146
}

levels narrow40 3
levels narrow60 2
echo "$(date +%H:%M) all done"
