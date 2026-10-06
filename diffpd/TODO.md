# diffpd TODO / Notes

## Done

- [x] Pin vertices rendered bigger and yellow
- [x] Collider collision using projected closest surface point (`ContactPointMode::Surface`)
- [x] Support more than one collider at once
- [x] Add smoothness with subdivision to cloth in final simulation render
- [x] Gradient correction for `AnimatedColliderVelocityMode::DecomposedRigid` (backward pass differentiates animated colliders; `MaterialPointDiff` still unsupported there)
- [x] Triangle mesh collider (forward + backward)
  - [x] Curvature correction: not needed (flat, shape operator = 0, `inv_r = 0`)
  - [x] Rotation correction derived per closest feature (face / edge / vertex)
  - [x] Best-fit triangle selection: closest feature via `trimesh_closest`, sign and normal from its pseudo-normal

## Open

- [ ] Verify gradients (FD check via `fd_check_contact_*`) for animated colliders: sphere (`sphere_collider_animation.json`) and mesh (`tri_collider_animation.json`), analytic vs. mesh A/B
- [ ] Mesh collider performance: BVH / per-triangle culling (`trimesh_closest` is brute force; only a mesh-level AABB reject today)
- [ ] Add substeps to animated colliders (currently asserts `frame_substeps == 1` and `dt == 1/kColliderAnimFPS`)
- [ ] Config save/load UI: button to save current config under a given name + dropdown to pick a saved config
- [ ] Collider collision using some form of CCD (continuous collision detection)
- [ ] Cloth self-contact
- [ ] Try derive gradient correction for: contact_point_mode == ContactPointMode::Particle (approximate mode; may just stay documented as approximate)

### Mesh collider limitations (known)

- Particle-only contact: sharp mesh edges/vertices can pierce between cloth particles
- Velocity-level contact only, no positional push-out
- Normal jumps across facets (piecewise-constant)
