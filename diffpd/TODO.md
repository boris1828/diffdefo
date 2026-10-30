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
- [x] Add a single forward only execution mode with rewatch player at the end that skips eveything that 
      regards backward sim in the initial config UI and final viewr panel  
- [x] auto smooth rendering of the tri-mesh with 60° degrees angles like in blender
- [x] Deforming collider meshes (skinned person: per-frame vertex positions baked from Blender)
  - [x] Blender exporter: armature-baked mesh sequence + loader (extend `blender_export.py` / `load_collider_animation`)
  - [x] Contact velocity from interpolated vertex velocities (replaces the rigid `v + ω × (x − pivot)`)
  - [x] Backward: generalize the rotation correction `J_s` with the vertex-velocity terms
  - [x] Requires the BVH item above (thousands of triangles)
- [x] Generalised attachment: bind pinned vertices to the nearest body vertex with barycentric weights 
      (replaces the hardcoded `collider_hip` waist attachment)
- [X] Add another way of generating initial cloth geometry: 
      - start with the perfect implicit surface
      - spawn veritces with a certain density (edges of the implicit surface have ad-hoc methods)
      - triangulate these vertices
- [X] Add ARAP / corotational energy


## Open

- [ ] Verify gradients (FD check via `fd_check_contact_*`) for animated colliders: 
      - sphere (`sphere_collider_animation.json`) 
      - mesh (`tri_collider_animation.json`), analytic vs. mesh A/B
- [ ] Config save/load UI: button to save current config under a given name + dropdown to pick a saved config
- [!] Collider collision using some form of CCD (continuous collision detection)
- [ ] Cloth self-contact
- [ ] Intersecting colliders: in `detect_contacts`, among the colliders a particle penetrates drop those whose closest surface point lies inside
      another collider (hidden surface, e.g. the inner part of a capsule crossing another), take the first remaining (fallback: first in list);
      same rule in `merge_detected_contacts`. Still one contact per particle. If particles still stick in the crease, do real multi-contact
      (forward: project onto 2 half-spaces; backward: rank-2 projector + both curvature terms, new FD checks)
- [ ] Add unified-per-vertex-mass gradient computation
- [ ] Adjoint convergence: check true residual, not `‖z_new − z‖` (or use Krylov + `L` preconditioner)
- [ ] Contact-free adjoint via preconditioned CG instead of the lagged `z = L⁻¹(h²ΔA z + rhs)` loop (Zeng et al., *Fast and Reliable 
      Gradients for Deformables Across Frictional Contact Regimes*, §5.4.1): solve `(L − h²ΔA) z = rhs` with
      `Eigen::ConjugateGradient<SparseMat, Lower|Upper, CholPrecond>`, `CholPrecond` wrapping `obj.solver` (Cholesky of `L`)
    - [ ] Assemble `A_adj = L − h²ΔA` per backward step (same pattern as `L`; precompute spring → entry map, reuse `gamma`)
    - [ ] `CholPrecond` wrapper class + new `compute_adjoint_vector_free`, tolerance 1e-8 on the true relative residual, warm start with the previous `z`
    - [ ] Fall back to the old loop when CG fails (`cg.info() != Success`, residual above tolerance, or NaN). Compression (`‖e‖ < l`) only 
          voids the SPD guarantee, it rarely makes the matrix indefinite (needs roughly `(l/‖e‖ − 1)·h²k/m ≳ 1`), so don't fall back on it 
          alone; log the fraction of steps with `min ‖e‖/l < 1` and their CG iteration counts instead
    - [ ] Validate: `z` matches the old loop at tight tolerance (~1e-8), FD check on `dphi/dk` still passes, compare iteration counts
- [ ] Curvature correction: use `Q_i(f_i − m v_c)` from `f`, not `m(v⁺ − v_c)` (exact only if forward converged)
- [ ] Active set: record forward `d_n < 0` on tape, don't recompute in backward
- [ ] Generalize the parameters gradient computation and FD check
- [ ] Early-stop the forward and adjoint iterations when `est_rel < kConvergenceTol` 
      (calibration: ~60–80 iterations suffice vs 300; cuts the global-solve cost)
- [ ] In skinned mesh have a collision mesh (the mesh that the cloth actually collides with) and a rendering mesh (the rendered mesh)
      with collision mesh subset of rendering mesh, so i can render the full body with the cost of a small collision mesh. 
- [ ] Add Re-run Backward button in the final simulation with possibility to change the number of adjoint iterations in case the number was too low 
      and it didn't converge (right now if I want to do it I have to re run everything)
- [ ] Show the bending stiffnes gradient

- [ ] Friction (really big): Coulomb / dry-frictional contact in the PD solve (Ly et al., *Projective Dynamics with Dry Frictional Contact*), 
      friction coefficient as a differentiable parameter; forward, backward, FD check

- [ ] Acceleration possibilities:
      - Chebyshev: A chebyshev semi-iterative approach for accelerating projective and position-based dynamics.
      - Anderson:  Anderson Acceleration for Geometry Optimization and Physics Simulation.
      - L-BFGS:    Quasi-Newton Methods for Real-Time Simulation of Hyperelastic Materials.


## Cloth pattern generation (clothcreate FORMAT.md v4 seams: `count`, `level`, same-shape seams, curved sides)

Done in this order, each step has its own test in `--meshgen-check` (synthetic pattern files written to the temp dir) and `--regression` must stay identical.
Today: one edge per side, no level, `sew()` in `tri_surface.h` (one edge = one `boundary_id` list, 20% length tolerance), `settle_seams` closes all links at once.

1. [x] **Parse the new seam fields.** `PatternSeam` gets `count_a/count_b` (default 1) and `level` (default 1); loader validates: shape index, edge index, `count` in [1, n_segments], `level` in 1..4, the two sides never share a segment (also inside one shape), no wrap on open shapes. Test: defaults for old files, every bad input throws with a clear message, the current shirt still loads unchanged.
2. [x] **Equal vertex counts by construction.** Before sampling, each seam fixes `N = round(mean side length / h)` intervals for both sides; a one-segment side gets N uniform intervals, a multi-segment side splits N among its segments in proportion to length (each >= 1, corners stay vertices). `PatternChart` takes a per-segment interval count (default `round(L/h)` as today, so unseamed edges and `--regression` do not change). Errors (shown in the panel): the two sides differ in length by more than 10%, a side has more segments than N (lower h), one segment is in two seams that need different N (name the segment). Test: 2 curves vs 1 curve gives N+1 vertices on both sides, corners are still samples, every error case throws, a file without seams meshes exactly as before.
3. [x] **Resolve a side to an ordered vertex chain and pair by index.** `count` consecutive edges (wrapping last -> 0 on closed shapes) concatenated from the `boundary` lists, shared corner vertices once, skipping zero-length segments. `sew()` pairs chain a[i] with chain b[i] (b reversed when `reversed`) and throws if the two chains have different lengths (safety net: never merge two vertices onto one, never leave one free). Test: count 1 equals today's links (the shirt's 22), count 2, wrap-around (`edge = n-1, count = 2`), reversed, a deliberately mismatched chain throws.
4. [x] **Curved sides.** Check step 2-3 on Bezier sides (arc vs arc of equal length, arc of 2 curves vs 1 curve): the vertex of each pair differs in arclength by at most about h/2, and the link lengths are small once the pieces are placed face to face (arclength is taken on the meshed chain, i.e. the flattened curve).
5. [x] **Several seams per vertex and same-shape seams.** Corner vertices shared by two seams, a vertex linked to more than one vertex, `a.shape == b.shape` (closing a sleeve tube; sides may touch at a corner, `a != b` guard already drops the self link). Test: tube of one piece sewn to itself, links dedupe, no self link, simulation of the sewn tube stays finite.
6. [x] **Seam levels in the data.** `TriSurface::seams`, `SimMesh::seams` and `SeamEnergy::links` carry the level (parallel `level` vector); `SeamEnergy` gets `max_level` (links above it are left out of the stiffness triplets and of the force). Test: triplet count per `max_level`, `max_level = 4` identical to today.
7. [x] **Settle seams level by level.** `settle_seams` loops level 1..max in one call from the previous pose: stage L has `max_level = L`, links of levels < L at rest (pull 0), links of level L ramp pull 1 -> 0, later levels off. Test: a 3-piece synthetic pattern (coplanar / overlapping pieces, two levels): after each stage the longest link of that level is small and the earlier levels stay closed; label shows the level.
8. [ ] **Viewer.** Seam ribbons coloured per level (palette of 4), the settle label and the "Seams settled" line show the level reached (`level L/N`). Check by eye on the shirt and on the 3-piece file.
9. [ ] **End to end on real files.** A new pattern with sleeves (curved armholes, `count` 2 vs 1, a tube closing seam, levels 1-2): settle seams -> settle cloth -> real run, then the sewn-pattern `dphi/dk` finite-difference check (no colliders) on it. Update `CLAUDE.md` 5a and `PATTERN_SHAPES_PLAN.md` to the v4 seam format.

Open (not blocking the steps above): per-level time/iterations in the UI (today one "Seams Time" for all), self-collision between pieces that end up overlapping, whether 10% is the right length tolerance between the two sides of a seam (today > 20% throws), and a segment sewn to two different seams (step 2 rejects it unless both need the same N).

## Investigations / Bugs / Errors

- [ ] Investigate why `ContactRecheckMode::Every` gives per-type stiffness gradients 0.2–1% off from FD in long-contact runs 
      (exact with `Once`, unified sum unaffected): compare analytic vs FD per step with the contact set frozen to the final recorded 
      set, to see whether contacts merged mid-iteration (`merge_detected_contacts`) break the backward pass's fixed-set assumption.
- [ ] Investigate the ~40–75 self-intersecting body triangles around the pelvis cap: where cloth particles near them can get the wrong 
      inside/outside sign (pseudo-normal vs ray parity disagree), producing spurious or missed contacts and inverted normals at the waist, 
      and whether cleaning the cap in Blender removes it.

## Papers / Theoretical

- [ ] Sparse inverted global step (Zeng, Luo, Shi & Zhang 2025, *Fast But Accurate: A Real-Time Hyperelastic Simulator with Robust 
      Frictional Contact*, §4 "Sparse Inverse Solution"): since the global matrix `A` is constant, Cholesky-factor it 
      once (`A = L Lᵀ`, after a nested-dissection reordering) and explicitly precompute `S = L⁻¹`, which stays sparse (its pattern is only 
      each node's ancestors in the elimination tree), so every global step becomes `x = Sᵀ S b`, two parallel sparse mat-vecs instead of 
      sequential triangular solves. Check whether it fits our `obj.solver` solves (forward and adjoint).

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

- [ ] formulates the forward contact step as a QP or QCQP and extracts gradients by implicitly differentiating the 
      optimality (KKT) conditions (like:Differentiable cloth simulation for inverse problems.)

### DiffCloth implementations ideas:

- [ ] Iteration count. The solve stops early once ‖Δx‖/N < threshold, with at most −log₁₀(threshold)·150 iterations. 4
      If it doesn’t converge, it reverts to the best iterate seen.
- [ ] Collider test: three samples along the predicted path. For each collider, three samples are tested in 
      order: x_n, then x_n + ½h·v_pred, then x_n + h·v_pred (= s_n).
- [ ] Contact margins. A contact is triggered by proximity, not penetration, and every threshold is hard-coded.
- [ ] internal forces are the negative gradients of three quadratic Projective Dynamics energies: a per-triangle strain 
      energy ½·k_stretch·A_rest·‖F − R(F)‖² (deformation gradient pulled to its nearest rotation via SVD), a per-interior-edge 
      cotangent-Laplacian bending energy ½·k_bend·(3/(A₀+A₁))·‖Σ w_i x_i − p‖² that keeps the rest curvature magnitude, and 
      stiff attachment springs ½·10⁴·‖x_i − x_fixed‖² on pinned vertices, with no distance springs.
- [ ] Uses a lumped, diagonal mass matrix: each vertex gets one third of the rest area of every triangle it belongs to, 
      multiplied by a single fabric density.
- [ ] In the hat experiments they use splines to model the trajectory of the pulling forces that positions the 
      hat on the head
- [ ] In the real-to-sim flag example they show single nodes trajectory in 3D from the real and guessed sim 
      before and after optimization to show that the optimized sim get closer to the real one

### Mesh collider limitations (known)

- Particle-only contact: sharp mesh edges/vertices can pierce between cloth particles
- Velocity-level contact only, no positional push-out
- Normal jumps across facets (piecewise-constant)
