# Step 1 report: the blocked artery segment

Source: the stroke story's vessel mesh (`stroke_in_3d` `StoryScene.vessel_mesh`). Units: model units x 168.2 mm (StoryScriptController's anatomical reference).

## Scale check

- Brain shell at 168.2 mm per unit: 144 x 92 x 165 mm (width x height x length). A typical adult brain is ~140 mm wide and ~165-170 mm long, so the scale holds.

## Segment

- Vessel piece 4 of the mesh holds the clot. Junction at flow coordinate 0.2726.
- Clot face 7.49 mm past the junction along the blocked branch; clot body 2.12 mm long (story's _ClotStart..._ClotEnd).

| branch | role | mesh vertices | length (mm) | diameter mean (mm) | min-max (mm) | wall fit error (mm, median abs) |
|---|---|---|---|---|---|---|
| trunk | inlet trunk (likely the end of the internal carotid) | 284 | 21.2 | 2.04 | 1.26-3.22 | 0.259 |
| blocked | blocked branch (likely the middle cerebral artery, M1 onward) | 221 | 41.7 | 1.19 | 0.66-1.98 | 0.157 |
| sibling | sibling branch at the junction (likely the anterior cerebral artery, A1 stub) | 121 | 9.1 | 1.62 | 1.30-2.07 | 0.239 |

## Diameters vs typical adult anatomy

The story's vessels were modelled for display: they follow a plausible path at the right overall scale, but they are thinner than real arteries and their width wanders (the mesh has ~17 vertices per ring).

| branch | measured mean (mm) | typical adult (approx., mm) | measured / typical |
|---|---|---|---|
| trunk | 2.04 | 3.75-4.85 | 0.47 |
| blocked | 1.19 | 1.7-2.4 | 0.58 |
| sibling | 1.62 | 1.5-2.3 | 0.85 |

Flow resistance scales with radius^-4 (Poiseuille), so simulating the measured radii would make the flow unrealistic. Decision for Step 2: keep the story's centreline path, and give each branch a smooth, typical adult radius (sourced values), or keep the measured radii.

## Consistency

- trunk: centreline length 0.1259 units vs flow-coordinate span 0.1222 units (+3 %)
- blocked: centreline length 0.2479 units vs flow-coordinate span 0.2561 units (-3 %)
- sibling: centreline length 0.0540 units vs flow-coordinate span 0.0502 units (+7 %)

## Files

- `geometry/segment.json`: centreline points and radius per station (every 0.002 units = 0.34 mm) for each branch, junction, clot, scale.
- `reports/step1_context.png`: the segment inside the whole vessel tree.
- `reports/step1_segment.png`: close-up over the story's mesh, with the fitted walls.
- `reports/step1_diameter.png`: diameter along each branch.
