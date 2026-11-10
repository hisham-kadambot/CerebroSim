# Step 2 report: clean vessel, clot cases, physical set-up

## Geometry

- Path: the story's centrelines (Step 1), life size (168.2 mm per model unit), junction at the origin.
- Radii (Decision 1b, typical adult): trunk 4.0 mm; blocked branch 2.2 -> 1.9 mm over its first 24 mm; sibling 2.0 mm.
- Bends: the story's path was drawn for thin vessels; with adult radii some bends were tighter than the tube (wall folding on the inside). The path is relaxed only there, to curvature x radius <= 0.5: trunk moved at most 1.17 mm, blocked moved at most 0.04 mm, sibling moved at most 1.20 mm; max curvature x radius now trunk 0.65, blocked 0.48, sibling 0.49.
- Junction: smooth union, blend 0.6 mm. Flow extensions: 3 diameters, straight, cut flat (inlet before the trunk, outlets after both branches).
- Clot: at the story's position, 10.15 mm past the junction; body 2.12 mm (story) + 4.0 mm cosine ramps; severity = diameter reduction.

## Checks (0.1 mm voxels)

| case | fluid voxels | volume (mm³) | connected pieces | inlet / blocked outlet / sibling outlet piece | min diameter at the clot (mm) | SDF time |
|---|---|---|---|---|---|---|
| healthy | 0.62 M | 620 | 1 | 1 / 1 / 1 | 2.03 | 240 ms |
| narrow50 | 0.60 M | 604 | 1 | 1 / 1 / 1 | 1.02 | 164 ms |
| narrow70 | 0.60 M | 600 | 1 | 1 / 1 / 1 | 0.60 | 163 ms |
| occluded | 0.60 M | 597 | 2 | 2 / 1 / 2 | 0.00 | 163 ms |

- Healthy volume: voxels 620 mm³ vs tube formula 615 mm³ (+0.8 %; the formula double-counts the junction overlap and ignores the blend).
- Healthy diameters on the grid vs targets: trunk median 3.99 mm (target 4.0), sibling median 1.99 mm (target 2.0).
- Connectivity (18-neighbour, as a D3Q19 lattice sees it): the full occlusion must split the lumen in two (inlet + sibling vs the blocked branch beyond the clot); every other case is one piece.
- Clearance between walls that must not touch (adult radii on the story's path):
  - trunk vs trunk: +8.80 mm
  - trunk vs blocked: +4.95 mm
  - trunk vs sibling: +5.05 mm
  - blocked vs trunk: +4.95 mm
  - blocked vs blocked: +4.88 mm
  - blocked vs sibling: +0.73 mm
  - sibling vs trunk: +5.05 mm
  - sibling vs blocked: +0.73 mm
  - sibling vs sibling: +3.87 mm

## Physical set-up (`geometry/physics.json`)

- Blood: Newtonian, density 1060 kg/m³, viscosity 3.5 mPa·s.
- Heart period 1.0 s. Inlet: mean 240 mL/min, synthetic waveform with pulsatility index 0.83.
- Outlets: resistances tuned in Step 4 so the healthy MCA carries ~135 mL/min.
- Derived: mean velocity trunk 0.32 m/s, MCA 0.59 m/s (peak 0.97); Reynolds trunk 386, MCA 394 (peak 644): laminar; Womersley trunk 2.8, MCA 1.5.

Sources: listed in `geometry/physics.json` (`sources`).

## Files

- `vessel.py`: `build(case)` and `sdf_grid(geometry, dx)` for later steps (any resolution).
- `geometry/vessel.json`, `geometry/physics.json`.
- `reports/step2_vessel.png`, `step2_diameters.png`, `step2_clot_sections.png`.
