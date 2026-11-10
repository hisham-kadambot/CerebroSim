#!/usr/bin/env bash
# Step 9: every clot size x inflow level the explorer offers, one after another on the GPU, cheapest first.
# Skips a run whose results are complete (meta.json "done": true), so it can be restarted after a stop.
#   bash run_all.sh            (from Git Bash, in this folder; ~40 h in all)
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

done_already() { grep -q '"done": true' "results/$1/meta.json" 2>/dev/null; }

run() {  # run <case> <flow> <beats> <dt-scale>
    local tag=$1
    [ "$2" != "1.0" ] && tag="$1_flow$(awk "BEGIN{print int($2 * 100 + 0.5)}")"
    if done_already "$tag"; then echo "$(date +%H:%M) skip $tag (done)"; return; fi
    echo "$(date +%H:%M) start $tag"
    uv run simulate.py "$1" --flow "$2" --beats "$3" --dt-scale "$4" > "results/$tag.log" 2>&1 \
        && echo "$(date +%H:%M) done $tag" || echo "$(date +%H:%M) FAILED $tag (see results/$tag.log)"
}

# the clot sizes at the sourced inflow
run narrow30 1.0 3 1.0
run narrow50 1.0 3 1.0
# low and high inflow for the 0.1 mm cases (high flow: 0.75 x the time step keeps the lattice speed down)
for c in healthy narrow30 narrow50 occluded; do
    run $c 0.7 3 1.0
    run $c 1.3 3 0.75
done
# the 0.05 mm cases: ~6 h (normal, low) and ~8 h (high) each
run narrow90 1.0 2 1.0
for c in narrow70 narrow90; do
    run $c 0.7 2 1.0
    run $c 1.3 2 0.75
done
echo "$(date +%H:%M) all done"
