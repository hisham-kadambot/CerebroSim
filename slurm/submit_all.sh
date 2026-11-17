#!/bin/bash
# Submit CerebroSim's pipeline to Slurm as chained jobs (each starts when the one before it ends):
#   1. the healthy run (calibrates the inlet)  2. every other run in slurm/runs.txt, MAX_PARALLEL at once
#   3. wall shear stress  4. the AI training set  5. the MLP (and GeoTransolver with --geotransolver)
# Runs whose results are complete are skipped, so it is safe to submit again after a failure or a time-out.
#   bash slurm/submit_all.sh                 everything
#   bash slurm/submit_all.sh sims            the simulations only
#   bash slurm/submit_all.sh ai              wall shear stress, training set and training (results already there)
#   ... --geotransolver                      also train GeoTransolver (~3 h)
# Logs: results/slurm/; each simulation's own log: results/<run>.log. Watch with: squeue --me
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/config.sh
what=all geo=0
for a in "$@"; do
    case $a in sims|ai|all) what=$a ;; --geotransolver) geo=1 ;; *) echo "unknown argument: $a" >&2; exit 2 ;; esac
done
mkdir -p results/slurm
[ "$RUNTIME" = none ] || [ "${IMAGE#*#}" != "$IMAGE" ] || [ -e "$IMAGE" ] || { echo "IMAGE not found: $IMAGE (slurm/config.sh)" >&2; exit 2; }

common=(--parsable --partition="$PARTITION" --gpus=1 --cpus-per-task="$CPUS" --mem="$MEM")
[ -n "$ACCOUNT" ] && common+=(--account="$ACCOUNT")
[ -n "$QOS" ] && common+=(--qos="$QOS")
after=()
submit() {  # submit <name> <time> <dependency type> <sbatch options and job arguments ...>
    local name=$1 time=$2 dep=$3; shift 3
    local id
    id=$(sbatch "${common[@]}" --job-name="cs_$name" --time="$time" "${after[@]/#/--dependency=$dep:}" "$@")
    id=${id%%;*}                # "id;cluster" on multi-cluster set-ups
    echo "submitted $name: job $id" >&2
    after=("$id")
}

if [ "$what" != ai ]; then
    n=$(grep -cvE '^[[:space:]]*(#|$)' slurm/runs.txt)
    submit calibrate "$TIME_SIM" afterok --array=1 --output=results/slurm/%x_%a_%A.out slurm/job.sbatch sim
    submit sims "$TIME_SIM" afterok --array=2-"$n"%"$MAX_PARALLEL" --output=results/slurm/%x_%a_%A.out \
        slurm/job.sbatch sim
fi
if [ "$what" != sims ]; then
    # afterany: shear stress and training use whichever runs finished, even if one failed
    submit wss "$TIME_WSS" afterany --output=results/slurm/%x_%j.out slurm/job.sbatch python3 wss.py
    submit dataset "$TIME_DATA" afterok --output=results/slurm/%x_%j.out slurm/job.sbatch "cd ai && python3 dataset.py"
    data=("${after[@]}")
    submit train_mlp "$TIME_TRAIN" afterok --output=results/slurm/%x_%j.out \
        slurm/job.sbatch "cd ai && python3 train.py --model mlp --hours 1"
    if [ "$geo" = 1 ]; then
        after=("${data[@]}")
        submit train_geotransolver "$TIME_TRAIN" afterok --output=results/slurm/%x_%j.out \
            slurm/job.sbatch "cd ai && python3 train.py --model geotransolver --hours 3"
    fi
fi
echo "watch: squeue --me    logs: results/slurm/"
