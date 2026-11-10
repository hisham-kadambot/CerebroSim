# CerebroSim

GPU blood-flow simulation of a blocked brain artery, with an AI surrogate model and an interactive 3D explorer.

CerebroSim simulates pulsing blood flow through a middle cerebral artery (MCA) segment, healthy or narrowed by a
clot (30 % to fully blocked), at normal and stroke-level blood pressures. It shows how the clot changes pressure,
speed, wall shear stress and the blood reaching the brain tissue.

- **Solver:** a sparse D3Q27 lattice Boltzmann method on the GPU, written in [NVIDIA Warp](https://github.com/NVIDIA/warp)
  and validated against exact pipe-flow solutions and a stenosis correlation.
- **Rendering:** ray-traced glass artery, blood particles, cut planes, in Warp; MP4 export with NVENC.
- **Explorer:** a browser control panel driving a live renderer. Pick the clot and blood pressure, play the heartbeat,
  slice the artery, compare with healthy and click to read the blood pressure and speed.
- **AI surrogate:** [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) models (GeoTransolver and an MLP
  baseline) trained on the simulations. They predict the flow for any clot size and blood pressure in seconds
  instead of an hour.

> **Not for clinical use.** This is a research and teaching model: an idealised vessel with typical adult sizes, a
> Newtonian blood model, rigid walls and no collateral circulation.

## Requirements

- Windows 11 (Linux should work for the solver; the AI environment is pinned to Windows wheels).
- An NVIDIA GPU (developed on an RTX 3090, 24 GB). The fine 0.05 mm runs need about 14 GB.
- [uv](https://docs.astral.sh/uv/). Every script declares its own dependencies, so there's no manual install.
  - Solver and explorer: `warp-lang>=1.18,<1.19`, numpy, scipy, scikit-image, matplotlib, pillow.
  - AI (`ai/pyproject.toml`): PhysicsNeMo 2.2.2, torch 2.12 + CUDA 12.6, torch_scatter.

## Quick start

```
uv run validate.py                     # ~13 min: the solver against exact answers
uv run simulate.py healthy --beats 4   # ~21 min: calibrates the inlet (results/healthy)
uv run simulate.py occluded            # ~16 min
uv run explore.py                      # http://127.0.0.1:8003
```

Simulation results aren't in the repository (they're tens of GB). The scripts make them in `results/`.

## Pipeline

| Step | Command | What it does |
|---|---|---|
| 1 | `uv run segment.py` | Extracts the artery's centreline and radius from the source 3D story model. Needs that project; its output `geometry/segment.json` is included, so you can skip this step |
| 2 | `uv run vessel.py` | Builds the life-size vessel (typical adult radii, smooth junction, flow extensions) for each clot size, voxelised on the GPU. Writes `geometry/vessel.json` and `geometry/physics.json`, which includes the sources |
| 3 | `uv run validate.py` | Validates `lbm.py`: steady and pulsing pipe flow, open pipes and a 70 % narrowing against the Young–Tsai correlation and a finer grid |
| 4 | `uv run simulate.py <case>` | Pulsatile flow over several heartbeats. Cases: `healthy`, `narrow30` … `narrow90`, `occluded`. Options: `--map <mmHg>`, `--flow <factor>`, `--beats`, `--dt-scale`. `uv run simulate.py report` writes the summary |
| 5 | `uv run visualise.py` | Ray-traced stills: wall pressure, change from healthy, particles, cut planes |
| 6 | `uv run demo.py` | The demonstration video (1080p30, NVENC). `--stills` gives one still per shot; `--serve` runs a viewer on port 8002 |
| 8 | `uv run explore.py` | The interactive explorer (port 8003) |
| 9–10 | `bash run_all.sh`, `bash run_bp.sh` | Every clot size at 70 % and 130 % flow, and at the stroke blood pressures of 118 and 146 mmHg |
| — | `uv run wss.py` | Wall shear stress for every finished run. `--test` checks it on Poiseuille pipes |

Faster blood needs a shorter time step: use `--dt-scale 0.75` at 118 mmHg and `0.6` at 146 mmHg, as the run
scripts do. Narrowings of 60 % and more run on a 0.05 mm grid.

## The explorer

`uv run explore.py`, then open http://127.0.0.1:8003.

- **Simulation:** the clot (healthy … fully blocked) and blood pressure (normal 95, stroke 118, severe 146 mmHg).
- **AI estimate:** any clot size (0–100 %) and blood pressure (70–145 mmHg), from the trained surrogate, shown
  against the nearest simulation.
- **Heartbeat:** play or pause, slow motion, and a slider to jump to any phase.
- **Visualization:** blood particles; wall coloured by pressure, shear stress or change; lengthwise or cross-section
  slices coloured by speed, pressure or change.
- **Compare:** side by side with healthy, or the pressure change as a colour map.
- **Camera:** whole, clot and junction views; orbit; drag to turn, scroll to zoom, click to probe.
- **Results:** flows (inlet, MCA, ACA), pressures before and after the clot, the pressure drop, top speed, brain
  tissue blood flow per 100 g (normal / penumbra / core, from CT-perfusion thresholds), and a pressure chart over
  the heartbeat.

To use it from other computers, open the port once (PowerShell as administrator):
`New-NetFirewallRule -DisplayName "CerebroSim explorer 8003" -Direction Inbound -Protocol TCP -LocalPort 8003 -Action Allow -Profile Any`

## AI surrogate (PhysicsNeMo)

Run these from `ai/`:

```
uv run check_env.py                                   # PhysicsNeMo, torch and the GPU
uv run dataset.py                                     # training data from the finished simulations
uv run train.py --model mlp --hours 1                 # the MLP baseline (used by the explorer)
uv run train.py --model geotransolver --hours 3       # GeoTransolver
uv run train.py --model mlp --eval                    # test a trained model on the held-out runs
uv run estimate.py 45 110                             # one estimate: 45 % clot at 110 mmHg
uv run estimate.py --serve                            # the explorer's AI service (port 8004, started by explore.py)
```

Each training example is one cell of the artery at one moment of the beat. Inputs are its position, wall distance,
place along the branch, local and healthy radius, direction and branch, plus the clot size, pressure level and
phase. Outputs are pressure and the three velocity components. The 50 % clot runs are held out to test clot sizes
the model hasn't seen.

## Data and limitations

- The vessel is the story model's MCA path with typical adult radii, not a patient scan. Flows and pressures are
  calibrated to published values; the sources are listed in `geometry/physics.json`.
- The patient-specific option (`patient.py`) needs your own MRA segmentations in `patients/<id>/pre.stl` and
  `post.stl`. No patient data is included in this repository.
- No collateral vessels are modelled, so a fully blocked artery's territory gets no flow.
- The 90 % clot's throat is only a few grid cells across, so its numbers are approximate.

## Licence

MIT. See [LICENSE](LICENSE).
