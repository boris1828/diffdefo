# DiffXPBD

Differentiable XPBD (Extended Position-Based Dynamics) for cloth/chain. Two implementations: **C++/Eigen** (forward + hand-written adjoint) and **JAX** (autodiff via `grad`). Same simulation, same config, comparable output.

## Examples of Forward Simulations

| 10×10 cloth, free fall | Chain + ground collision | 5×5 cloth + ground collision |
| :---------------------: | :----------------------: | :---------------------------: |
|    ![](media/ex1.gif)    |     ![](media/ex2.gif)     |       ![](media/ex3.gif)       |

## Contents

```
src/main.cpp     C++ sim + adjoint (gravity, ground collider, distance constraints, Jacobi 1-iter)
src/param.conf   parameters (sim_rate, gravity, object, compliance, ...)
jax_impl.py      same solver in JAX, automatic gradient w.r.t. compliance
compare.ipynb    numerical comparison C++ vs JAX
animation/       per-frame .obj output (target_*.obj, guess_*.obj) — loadable in animator.blend
external/eigen   header-only, vendored
CMakeLists.txt   C++ build
docs/            theory notes (PDF)
```

## Theory

[docs/InversePhysics.pdf](docs/InversePhysics.pdf) — notes covering the implicit BDF1 simulator, the adjoint method for gradients, Baraff-Witkin cloth, descent / primal-dual methods, and the XPBD adjoint derivation this repo implements.

## Build (C++)

Requires CMake ≥ 3.16 and a C++17 compiler (MSVC / clang / gcc).

```bash
cmake -S . -B build
cmake --build build --config Release
./build/bin/xpbd              # Linux/macOS
build\bin\Release\xpbd.exe    # Windows
```

## Run JAX

```bash
pip install jax numpy
python jax_impl.py
```

Prints target/guess final positions, loss, and `dL/dcompliance`.

## Config (`src/param.conf`)

One `key = value` per line; `#`, `;`, `//` start comments. Both impls read the same file (pass a path as the first CLI arg, relative to the project root, otherwise the standard path `src/param.conf` is used).

```
sim_rate          = 312                 # integration substeps per second
n_seconds         = 4                   # simulated duration in integer seconds
gravity           = (0.0, -9.81, 0.0)
fps               = 24                  # .obj export rate
target_compliance = 0.0005              # compliance of the ground-truth "target" sim
compliance        = 0.0001              # compliance of the "guess" sim
target_offset     = (0.0, 0.0, 0.0)     # initial position offset of the target object
offset            = (0.0, 0.0, 0.0)     # initial position offset of the guess object
obj               = cloth(10, 10)       # chain(N) | cloth(W, H)
ground_ori        = (0.0, -5.0, 0.0)    # a point on the ground plane (halfspace collider)
ground_normal     = (0.0, 1.0, 0.0)     # ground plane normal
export_obj        = true                # write per-frame target_*.obj / guess_*.obj into animation/
experiment        = compliance_optimization(50)
optimizer         = momentum(1e-8, 0.8)
loss              = mse_frames_trajectory(24)
```

Field notes:

- **experiment** — what to run:
  - `compliance_gradient` — `dL/dcompliance`
  - `x0_gradient` — `dL/d(initial positions)`.
  - `single_step_jacobian(step)` — the per-step Jacobian `dx⁺/dx⁻` at update `step`.
  - `compliance_optimization(iters)` — gradient-descent fit of compliance to the target for `iters` steps (uses `optimizer`).
- **optimizer** — descent rule, only for optimization experiments:
  - `GD(lr)`
  - `momentum(lr, beta)`
  - `ADAM(lr, beta1, beta2, epsilon)`
- **loss** — trajectory-matching error:
  - `mse_final_position` (only final frame)
  - `mse_full_trajectory` (every single step)
  - `mse_frames_trajectory(fps)` (frames sampled at `fps`)

```

```
