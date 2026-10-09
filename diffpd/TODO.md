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
  - [x] Remove smoothed cloth surface
  - [x] After simulation display where computation time was spent in percentage (for forward: contact detection, elastic local step, contact local step, gloabl step) 

## Open

- [ ] Verify gradients (FD check via `fd_check_contact_*`) for animated colliders: sphere (`sphere_collider_animation.json`) 
      and mesh (`tri_collider_animation.json`), analytic vs. mesh A/B
- [ ] Mesh collider performance: BVH / per-triangle culling (`trimesh_closest` is brute force; only a mesh-level AABB reject today)
- [ ] Add substeps to animated colliders (currently asserts `frame_substeps == 1` and `dt == 1/kColliderAnimFPS`)
- [ ] Config save/load UI: button to save current config under a given name + dropdown to pick a saved config
- [ ] Collider collision using some form of CCD (continuous collision detection)
- [ ] Cloth self-contact
- [ ] Try derive gradient correction for: contact_point_mode == ContactPointMode::Particle (approximate mode; may just stay documented as approximate)
- [ ] Forward convergence: check using the "real" energy, not just the relative difference between consecutive iterations  
- [ ] Adjoint convergence: check true residual, not `‖z_new − z‖` (or use Krylov + `L` preconditioner)
- [ ] Curvature correction: use `Q_i(f_i − m v_c)` from `f`, not `m(v⁺ − v_c)` (exact only if forward converged)
- [ ] Active set: record forward `d_n < 0` on tape, don't recompute in backward
- [ ] Generalize the parameters gradient computation and FD check
- [ ] Have a way to compute a single value for the general convergence over the complete forward simulation 
- [ ] After simulation display where computation time was spent in percentage for backward step

## Papers / Theoretical

- [ ] Check in: https://users.cs.utah.edu/~ladislav/li18damping/li18damping.pdf the sentence:
    This variational integration formulation usually provides a more stable numerical 
    solution compared to nonlinear root finding [KYT∗ 06,MTGG11]. This is because the 
    minimization problem could at least find a local minimum, which could be a reasonable 
    solution, while the root-finding could simply fail.

- [ ] in the notes in PD refactorize all derivations to follow a single notations and write 
      schematically all the formulas with clear derivations for each version of the PD 
      at the end of the notes so do not pollute to much


### Mesh collider limitations (known)

- Particle-only contact: sharp mesh edges/vertices can pierce between cloth particles
- Velocity-level contact only, no positional push-out
- Normal jumps across facets (piecewise-constant)
