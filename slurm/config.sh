# Slurm settings for CerebroSim: edit for your cluster (sinfo, scontrol show partition). Read by every script here.

# how jobs run the image: pyxis (srun --container-image, NVIDIA's Slurm plugin), apptainer, or none (the nodes'
# own Python with the packages in the Dockerfile)
RUNTIME=pyxis
# pyxis: an Enroot squashfs file or a registry image (registry#image:tag); apptainer: a .sif file
IMAGE="$PWD/cerebrosim.sqsh"
# folders the jobs must see, comma separated (the repository is always added); e.g. your shared storage
MOUNTS=""

PARTITION=gpu
ACCOUNT=""              # --account, if your cluster needs one
QOS=""                  # --qos, if your cluster needs one
CPUS=8
MEM=64G
TIME_SIM=24:00:00       # one simulation (A100: ~1-12 h; the longest, 60-90 % clots at 146 mmHg, need the most)
TIME_WSS=02:00:00
TIME_DATA=04:00:00
TIME_TRAIN=05:00:00
MAX_PARALLEL=8          # simulations running at once (one GPU each)
