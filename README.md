# DiffXPBD

Differentiable XPBD (Extended Position-Based Dynamics) for cloth/chain. Two implementations: **C++/Eigen** (forward + hand-written adjoint) and **JAX** (autodiff via `grad`). Same simulation, same config, comparable output.

## Contents

```
src/main.cpp     C++ sim + adjoint (gravity, ground collider, distance constraints, Jacobi 1-iter)
src/param.conf   parameters (sim_rate, gravity, object, compliance, ...)
jax_impl.py      same solver in JAX, automatic gradient w.r.t. compliance
compare.ipynb    numerical comparison C++ vs JAX
animation/       per-frame .obj output (target_*.obj, guess_*.obj) — loadable in animator.blend
external/eigen   header-only, vendored
CMakeLists.txt   C++ build
```

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
sim_rate          = 312       # steps per second
n_seconds         = 10
gravity           = (0,-9.81,0)
fps               = 24        # exported frames per second
target_compliance = 1e-4      # ground truth
compliance        = 2e-4      # guess (optimized via gradient)
obj               = cloth(5,5)   # or chain(N)
ground_normal     = (0,1,0)
export_obj        = true
```

## Notes

- **Hardcoded** paths in [src/main.cpp:994](src/main.cpp#L994) and [jax_impl.py:258](jax_impl.py#L258) — replace with relative paths before running outside my machine.
- Polyscope (interactive viewer) is optionally linked in CMake but commented out; to enable it, uncomment in [CMakeLists.txt](CMakeLists.txt) and clone `polyscope` into `external/`.
- To view the `.obj` frames in Blender: open `animation/animator.blend`.
