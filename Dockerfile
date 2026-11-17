# CerebroSim's image: NVIDIA's PhysicsNeMo container plus NVIDIA Warp and the few packages the scripts use.
# One image for the explorer (docker run) and the Slurm jobs (Pyxis/Enroot or Apptainer); the code and results are
# mounted from shared storage, not baked in.
#   docker build -t cerebrosim .
FROM nvcr.io/nvidia/physicsnemo/physicsnemo:26.08

RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
RUN PKGS='warp-lang>=1.18,<1.19 scikit-image imageio-ffmpeg pyyaml uv' && \
    (pip install --no-cache-dir $PKGS || pip install --no-cache-dir --break-system-packages $PKGS) && \
    (python3 -c "import torch_geometric" 2>/dev/null || pip install --no-cache-dir torch_geometric \
     || pip install --no-cache-dir --break-system-packages torch_geometric)

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
