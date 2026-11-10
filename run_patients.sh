#!/usr/bin/env bash
# Step 11c: the patients' own vessels (patient.py) at the normal blood pressure, same inlet pressure and brain
# resistances as the story artery. Skips a run whose results are complete.
#   bash run_patients.sh P1_pre P1_post        (from Git Bash, in this folder; the cases in patients/cases.json)
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
done_already() { grep -q '"done": true' "results/$1/meta.json" 2>/dev/null; }
for c in "$@"; do
    if done_already "$c"; then echo "$(date +%H:%M) skip $c (done)"; continue; fi
    echo "$(date +%H:%M) start $c"
    uv run simulate.py "$c" --beats 3 > "results/$c.log" 2>&1 \
        && echo "$(date +%H:%M) done $c" || echo "$(date +%H:%M) FAILED $c (see results/$c.log)"
done
echo "$(date +%H:%M) all done"
