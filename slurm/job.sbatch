#!/bin/bash
# One CerebroSim job on a GPU node, in the image (slurm/config.sh: RUNTIME, IMAGE, MOUNTS). Submitted by
# submit_all.sh, which passes the partition, resources and log file.
#   job.sbatch sim                 one simulation: the line of slurm/runs.txt given by the array index
#   job.sbatch <command ...>       any command, run in the repository (e.g. python3 wss.py)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
REPO=$PWD
source slurm/config.sh
export PYTHONUNBUFFERED=1 WARP_CACHE_PATH="$REPO/.warp_cache"   # Warp's compiled kernels, shared by all jobs

if [ "$1" = sim ]; then
    read -r tag run_case args <<< "$(grep -vE '^[[:space:]]*(#|$)' slurm/runs.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")"
    if grep -q '"done": true' "results/$tag/meta.json" 2>/dev/null; then echo "$tag: done already"; exit 0; fi
    echo "$(date '+%F %H:%M') $tag on $(hostname)"
    cmd="python3 simulate.py $run_case $args > results/$tag.log 2>&1"
else
    cmd="$*"
fi

case $RUNTIME in
    pyxis)
        mounts="$REPO:$REPO"
        for m in ${MOUNTS//,/ }; do mounts+=",$m:$m"; done
        srun --container-image="$IMAGE" --container-mounts="$mounts" --container-workdir="$REPO" bash -c "$cmd" ;;
    apptainer)
        apptainer exec --nv --bind "$REPO${MOUNTS:+,$MOUNTS}" --pwd "$REPO" "$IMAGE" bash -c "$cmd" ;;
    none)
        bash -c "$cmd" ;;
    *)
        echo "unknown RUNTIME '$RUNTIME' in slurm/config.sh" >&2; exit 2 ;;
esac
echo "$(date '+%F %H:%M') finished"
