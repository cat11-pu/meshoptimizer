# Skinning-aware simplification validation

This directory validates `meshopt_simplifySkinned`, an experimental extension of the
edge simplifier that treats bone weights as hard constraints.

## What the new API guarantees

`meshopt_simplifySkinned` wraps the existing edge-collapse pipeline and adds two rules:

1. **Bone boundary lock.** An edge is never collapsed when its endpoints reference
   incompatible bone sets: either one endpoint uses a bone the other one does not,
   or the weight of a shared bone differs by more than `bone_weight_limit`. The check
   is transitive across collapse passes, because each surviving vertex carries a
   weight envelope of every source vertex it represents.
2. **No invented weights.** The simplifier never moves or averages surviving
   vertices, so every output vertex keeps the exact 4 bone indices/weights of one of
   the input vertices. The optional `vertex_remap` reports which source vertex each
   surviving vertex stands for (`~0u` for vertices absent from the result).

All pre-existing simplification entry points keep their signatures and behavior.

## Files

- `skinned_meshes.h` — three procedural skinned meshes with built-in animation rigs:
  - `arm_cylinder` — forearm/upper-arm sleeve with an elbow boundary
  - `cloth_sheet` — 2x2 bone panel with intersecting seams
  - `leg_knee` — hard two-bone knee seam on a tapered cylinder
- `models/*.obj` — rest-pose OBJ exports of the three meshes (bone weights in `# w` comments)
- `skinned_simplify_test.cpp` — 5-LOD validation harness and metrics
- `dump_lod.cpp` / `visualize_replay.py` — binary export and replay rendering
- `results/` — generated CSV, text report and PNG comparisons

## Metrics

For every mesh and each of the five LOD targets (25%, 50%, 65%, 80%, 90% triangle
reduction), the harness reports:

- **reduction** — actual triangle reduction vs. the source mesh
- **weight RMS / max** — bone-weight error seen when sampling the LOD surface
- **inherit error** — must be exactly 0; weights are never averaged into new values
- **seam retention** — fraction of bone-boundary edges whose endpoints stay distinct;
  any merge that does occur is independently verified to stay within `bone_weight_limit`
- **animation RMS / max** — per-source-vertex error when replaying a full bend cycle,
  sampling the animated LOD surface with barycentric interpolation

## Run

```sh
make run      # metrics -> results/lod_report.txt and results/lod_metrics.csv
make images   # replay comparison PNGs -> results/*_replay.png
```
