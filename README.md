# DiffXPBD

Differentiable XPBD (Extended Position-Based Dynamics) for cloth/chain. Two implementations: **C++/Eigen** (forward + hand-written adjoint) and **JAX** (autodiff via `grad`). Same simulation, same config, comparable output.

## Examples

| 10×10 cloth, free fall | Chain + ground collision | 5×5 cloth + ground collision |
|:---:|:---:|:---:|
| ![](media/ex1.gif) | ![](media/ex2.gif) | ![](media/ex3.gif) |

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

```
sim_rate          = 312                 # steps per second
n_seconds         = 10
gravity           = (0.0, -9.81, 0.0)
fps               = 24                  # exported frames per second
target_compliance = 0.0001
compliance        = 0.0002
target_offset     = (0.0, 3.0, 0.0)
offset            = (0.0, 3.0, 0.0)
obj               = cloth(5, 5)         # or chain(N)
ground_ori        = (0.0, 0.0, 0.0)
ground_normal     = (0.0, 1.0, 0.0)
export_obj        = true
```
