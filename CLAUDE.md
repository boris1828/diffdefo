# DiffXPBD

A research project implementing **differentiable XPBD** (eXtended Position-Based Dynamics) physics simulation. The goal is to compute analytic gradients through the simulator for inverse-physics problems: given a target trajectory, recover the simulation parameters (compliance, initial positions) that produce it.

## What it does

Simulates cloth and chain objects with distance constraints, then differentiates through the simulation to compute:
- `dL/d_compliance` — gradient of a loss w.r.t. constraint stiffness parameters
- `dL/d_x0` — gradient of a loss w.r.t. initial positions

These gradients enable parameter optimization (inverse simulation) and visualization of where stiffness matters most across the trajectory.

## Two implementations

| File | Language | Role |
|---|---|---|
| `src/main.cpp` | C++17 + Eigen | Primary implementation with hand-derived analytic Jacobians |
| `jax_impl.py` | Python + JAX | Reference implementation using JAX autodiff |
| `tester.py` | Python | Cross-validates both impls numerically (RTOL=1e-6, ATOL=1e-9) |
| `compare.ipynb` | Jupyter | Plots loss-scan results from the C++ exe |

The JAX impl is the ground truth — if they disagree, the C++ math is wrong.

## Architecture (C++ impl)

All code lives in a single file `src/main.cpp`. Sections in order:

1. **Types** — `Real=double`, `Vec3`, `Mat3`, `Mat6`, `PointsX` (N×3 matrix), `SparseMat`
2. **Object** — holds positions `x`, velocities `v`, inverse masses `w`, and a list of `DistanceConstraint`
3. **Factories** (`make::chain`, `make::cloth`) — build chain (1D) or cloth (2D grid) with structural + shear constraints
4. **DistanceConstraint** — computes XPBD correction; also accumulates `ddeltax_dx` (6×6 position Jacobian) and `dx_dalpha` (6×1 compliance Jacobian) during `compute_correction`
5. **Colliders** — `Halfspace` and `Sphere`; each implements `project(x) -> {x', J, active}` where `J = dx'/dx` (3×3 Jacobian)
6. **Jacobian assembly** — `assemble_system_jacobian` builds the sparse `dx+/dx-` step Jacobian (I + constraint contributions) post-multiplied by the collision Jacobian; `assemble_compliance_jacobian` builds `dx/d_alpha`
7. **SimulationTape** — records `{jacobians[], compliance_jac[], positions[]}` for every step
8. **XPBD step** — `XPBD_step_jacobi_1iter` runs one Jacobi iteration (all constraints in parallel), resolves collisions, then records to tape. `XPBD_step_gauss_seidel` is the non-differentiable multi-iter version.
9. **Loss functions** — `mse_final_position`, `mse_full_trajectory`, `mse_frames_trajectory`; return `LossGradients` with per-step `dphi/dx` and `dphi/dv`
10. **Adjoint pass** — `backward_explicit_adjoint` propagates adjoints backward through the tape; `backward_implicit_adjoint` is an alternative implicit formulation
11. **Gradient** — `compute_dphi_dcompliance` contracts the adjoint with `compliance_jac` at each step
12. **Config** — `Config` + `load_config` parse `src/param.conf` (key = value, `#` comments)
13. **Optimizer** — GD, momentum, ADAM (scalar update, applied to the single scalar compliance)
14. **Experiments** — dispatched from `main()` based on `experiment` field in config

## Config file (`src/param.conf`)

Controls everything. Key fields:

```
sim_rate  = 144          # simulation Hz
n_seconds = 4
gravity   = (0.0, -9.81, 0.0)

compliance        = 0.003    # guess (what we optimize)
target_compliance = 0.001    # ground truth

obj        = cloth(10, 10, false)   # chain(N) or cloth(W,H) [, pin=true]
colliders  = [ sphere((4.5, -5, 4.5), 3) ]   # or halfspace(origin, normal) or []

experiment = compliance_optimization(200)
optimizer  = momentum(5e-9, 0.9)
loss       = mse_frames_trajectory(24)

export_obj = true   # dumps .obj frames to animation/
fps        = 24
```

Available experiments: `forward_simulation`, `single_step_jacobian(step)`, `compliance_gradient`, `x0_gradient`, `compliance_optimization(iters)`, `loss_scan_compliance(min, max, steps)`

## Build

```
cmake -S . -B build
cmake --build build --config Release
# binary: build/bin/Release/xpbd.exe
```

C++17, Eigen vendored at `external/eigen` (header-only). Polyscope is vendored but commented out.

## Run

```
# use default config
./build/bin/Release/xpbd.exe

# use custom config
./build/bin/Release/xpbd.exe path/to/config.conf
```

Output is printed to stdout in a parseable format (`label = [...]` for vectors/positions, `label = [...]` for matrices).

## Test

```
python tester.py          # runs all 14 cross-validation cases
python tester.py src/param.conf   # runs one config and compares C++ vs JAX
```

Tester writes a temp config file, runs both executables, parses their stdout, and checks `np.allclose(rtol=1e-6, atol=1e-9)`.

## Key invariants

- XPBD uses **1 Jacobi iteration** (all constraints solved simultaneously, corrections summed) — this is what makes it differentiable with a closed-form step Jacobian.
- Compliance `alpha` is the per-constraint stiffness; `alpha_tilde = alpha / dt²` is the regularized form used in the XPBD update.
- Pinned particles have `w=0`; they are skipped in prediction, collision resolution, and velocity update.
- Only one collider is supported at a time (asserted in main).
- The JAX impl always wins disputes — if C++ and JAX disagree, fix the C++ math.
