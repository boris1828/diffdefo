# DiffXPBD

Differentiable XPBD (Extended Position-Based Dynamics) per cloth/chain. Due implementazioni: **C++/Eigen** (forward + adjoint manuale) e **JAX** (autodiff via `grad`). Stessa simulazione, stesso config, output confrontabile.

## Contenuto

```
src/main.cpp     C++ sim + adjoint (gravity, ground collider, distance constraints, Jacobi 1-iter)
src/param.conf   parametri (sim_rate, gravity, oggetto, compliance, ...)
jax_impl.py      stesso solver in JAX, gradiente automatico vs compliance
compare.ipynb    confronto numerico C++ vs JAX
animation/       output .obj per frame (target_*.obj, guess_*.obj) — caricabili in animator.blend
external/eigen   header-only, vendored
CMakeLists.txt   build C++
```

## Build (C++)

Richiede CMake ≥ 3.16 e un compilatore C++17 (MSVC / clang / gcc).

```bash
cmake -S . -B build
cmake --build build --config Release
./build/bin/xpbd        # Linux/macOS
build\bin\Release\xpbd.exe   # Windows
```

## Run JAX

```bash
pip install jax numpy
python jax_impl.py
```

Stampa posizioni finali target/guess, loss, e `dL/dcompliance`.

## Config (`src/param.conf`)

```
sim_rate          = 312       # passi/secondo
n_seconds         = 10
gravity           = (0,-9.81,0)
fps               = 24        # frame esportati al secondo
target_compliance = 1e-4      # ground truth
compliance        = 2e-4      # guess (ottimizzato via gradiente)
obj               = cloth(5,5)   # oppure chain(N)
ground_normal     = (0,1,0)
export_obj        = true
```

## Note

- Path **hardcoded** in [src/main.cpp:994](src/main.cpp#L994) e [jax_impl.py:258](jax_impl.py#L258) — sostituiscili con percorsi relativi prima dell'uso fuori dalla mia macchina.
- Polyscope (viewer interattivo) è linkato opzionalmente nel CMake ma commentato; per usarlo, decommenta in [CMakeLists.txt](CMakeLists.txt) e clona `polyscope` in `external/`.
- Per visualizzare gli `.obj` in Blender: apri `animation/animator.blend`.
