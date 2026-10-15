# diffpd TODO / Notes

## Done

- [x] Pin vertices rendered bigger and yellow
- [x] Collider collision using projected closest surface point (`ContactPointMode::Surface`)
- [x] Support more than one collider at once
- [x] Add smoothness with subdivision to cloth in final simulation render
- [x] Gradient correction for `AnimatedColliderVelocityMode::DecomposedRigid` (backward pass differentiates animated colliders; `MaterialPointDiff` 
      still unsupported there)
- [x] Triangle mesh collider (forward + backward)
  - [x] Curvature correction: not needed (flat, shape operator = 0, `inv_r = 0`)
  - [x] Rotation correction derived per closest feature (face / edge / vertex)
  - [x] Best-fit triangle selection: closest feature via `trimesh_closest`, sign and normal from its pseudo-normal
- [x] Remove smoothed cloth surface
- [x] After simulation display where computation time was spent in percentage (for forward: contact detection, elastic local step, 
      contact local step, gloabl step) 
- [x] Have a way to compute a single value for the general convergence over the complete forward simulations
- [x] Add substeps to animated colliders (currently asserts `frame_substeps == 1` and `dt == 1/kColliderAnimFPS`)
- [x] Per-type stiffness (stretch, shear, bend) with gradients (`dphi/dk_stretch`, `dphi/dk_shear`, `dphi/dk_bend`; FD check per parameter)
- [x] After simulation display where computation time was spent in percentage for backward step

## Open

- [ ] Verify gradients (FD check via `fd_check_contact_*`) for animated colliders: 
      - sphere (`sphere_collider_animation.json`) 
      - mesh (`tri_collider_animation.json`), analytic vs. mesh A/B
- [ ] Mesh collider performance: BVH / per-triangle culling (`trimesh_closest` is brute force; only a mesh-level AABB reject today)
- [ ] Config save/load UI: button to save current config under a given name + dropdown to pick a saved config
- [!] Collider collision using some form of CCD (continuous collision detection)
- [ ] Cloth self-contact
- [ ] Add unified-per-vertex-mass gradient computation
- [ ] Try derive gradient correction for: contact_point_mode == ContactPointMode::Particle (approximate mode; may just stay documented as approximate)
- [ ] Adjoint convergence: check true residual, not `‖z_new − z‖` (or use Krylov + `L` preconditioner)
- [ ] Curvature correction: use `Q_i(f_i − m v_c)` from `f`, not `m(v⁺ − v_c)` (exact only if forward converged)
- [ ] Active set: record forward `d_n < 0` on tape, don't recompute in backward
- [ ] Generalize the parameters gradient computation and FD check
- [ ] Early-stop the forward and adjoint iterations when `est_rel < kConvergenceTol` 
      (calibration: ~60–80 iterations suffice vs 300; cuts the global-solve cost)
- [ ] Deforming collider meshes (skinned person: per-frame vertex positions baked from Blender)
  - [ ] Blender exporter: armature-baked mesh sequence + loader (extend `blender_export.py` / `load_collider_animation`)
  - [ ] Contact velocity from interpolated vertex velocities (replaces the rigid `v + ω × (x − pivot)`)
  - [ ] Backward: generalize the rotation correction `J_s` with the vertex-velocity terms
  - [ ] Requires the BVH item above (thousands of triangles)
  - [ ] Generalised attachment: bind pinned vertices to the nearest body vertex with barycentric weights 
        (replaces the hardcoded `collider_hip` waist attachment)
- [ ] Friction (really big): Coulomb / dry-frictional contact in the PD solve (Ly et al., *Projective Dynamics with Dry Frictional Contact*), 
      friction coefficient as a differentiable parameter; forward, backward, FD check
- [ ] Add a single forward only execution mode with rewatch player at the end that skips eveything that regards backward sim 
      in the initial config UI and final viewr panel  

## Investigations / Bugs / Errors

- [ ] Investigate why `ContactRecheckMode::Every` gives per-type stiffness gradients 0.2–1% off from FD in long-contact runs 
      (exact with `Once`, unified sum unaffected): compare analytic vs FD per step with the contact set frozen to the final recorded 
      set, to see whether contacts merged mid-iteration (`merge_detected_contacts`) break the backward pass's fixed-set assumption.

## Papers / Theoretical

- [ ] Check in: https://users.cs.utah.edu/~ladislav/li18damping/li18damping.pdf the sentence:
      This variational integration formulation usually provides a more stable numerical 
      solution compared to nonlinear root finding [KYT∗ 06,MTGG11]. This is because the 
      minimization problem could at least find a local minimum, which could be a reasonable 
      solution, while the root-finding could simply fail.

- [ ] in the notes in PD refactorize all derivations to follow a single notations and write 
      schematically all the formulas with clear derivations for each version of the PD 
      at the end of the notes so do not pollute to much

## Ideas to Explore / Furhther Checks

- [ ] smooth-out (like in render smooth) the triangulated surface to have better gradients by interpolating 
      normals of neighborhood
- [ ] Skinned-body substeps: LBS once per track frame, then vertices lerped linearly (chord) with constant per-vertex 
      velocity V=(X1-X0)*60 over the bracket. Alternatives: slerp/lerp bone transforms per substep then LBS (follows 
      the arc, costs one LBS per substep), or higher-order interpolation of the vertices.
- [ ] Skinned-body BVH refit quality: boxes get loose as the pose drifts from the one the tree was built on. Track the sum of 
      node surface areas against its value at build time and rebuild (`bvh_build`) above ~1.5x; only worth adding if 
      query times show it matters.

### Mesh collider limitations (known)

- Particle-only contact: sharp mesh edges/vertices can pierce between cloth particles
- Velocity-level contact only, no positional push-out
- Normal jumps across facets (piecewise-constant)
