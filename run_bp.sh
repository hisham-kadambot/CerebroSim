#!/usr/bin/env bash
# Step 10: every clot size at the stroke patients' blood pressures (Sen et al. 2014, West J Emerg Med 15:345):
# mean arterial pressure 118 mmHg (the 7 patients' average on arrival) and 146 mmHg (the highest, above the
# 185/110 limit for clot-dissolving drugs). The normal level (95 mmHg) is the calibrated healthy pressure: the
# existing runs. Cheapest first; skips a run whose results are complete, so it can be restarted after a stop.
#   bash run_bp.sh            (from Git Bash, in this folder; ~2 days in all)
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

done_already() { grep -q '"done": true' "results/$1/meta.json" 2>/dev/null; }

run() {  # run <case> <map mmHg> <beats> <dt-scale>
    local tag="$1_map$2"
    if done_already "$tag"; then echo "$(date +%H:%M) skip $tag (done)"; return; fi
    echo "$(date +%H:%M) start $tag"
    uv run simulate.py "$1" --map "$2" --beats "$3" --dt-scale "$4" > "results/$tag.log" 2>&1 \
        && echo "$(date +%H:%M) done $tag" || echo "$(date +%H:%M) FAILED $tag (see results/$tag.log)"
}

# faster blood needs a shorter time step (keeps the lattice speed down): 0.75 x at 118 mmHg, 0.6 x at 146 mmHg
for c in healthy narrow30 narrow50 occluded; do
    run $c 118 3 0.75
    run $c 146 3 0.6
done
for c in narrow70 narrow90; do
    run $c 118 2 0.75
    run $c 146 2 0.6
done
echo "$(date +%H:%M) all done"
