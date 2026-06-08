import os
import re
import sys
import subprocess
import tempfile
import numpy as np

PROJ_ROOT  = os.path.dirname(os.path.abspath(__file__))
CPP_EXE    = os.path.join(PROJ_ROOT, "build", "bin", "Release", "xpbd.exe")
JAX_SCRIPT = os.path.join(PROJ_ROOT, "jax_impl.py")

RTOL, ATOL = 1e-6, 1e-9

# ----------------
#     PARSERS
# ----------------

def parse_gradient(out, label):
    m = re.search(rf"{re.escape(label)}\s*=\s*\[(.*?)\]", out, re.DOTALL)
    assert m, f"missing '{label} = [...]'"
    return np.array([float(t) for t in m.group(1).split(",") if t.strip()])

def parse_positions(out, label):
    m = re.search(rf"{re.escape(label)}\s*=\s*\[(.*?)\]", out, re.DOTALL)
    assert m, f"missing '{label} = [...]'"
    triples = re.findall(r"\(([^)]*)\)", m.group(1))
    return np.array([[float(v) for v in t.split(",")] for t in triples])

def parse_matrix(out, label):
    m = re.search(rf"{re.escape(label)}\s*=\s*\[(.*)\]", out, re.DOTALL)
    assert m, f"missing '{label} = [...]'"
    rows = re.findall(r"\[([^\[\]]+)\]", m.group(1))
    return np.array([[float(v) for v in r.split(",")] for r in rows])

def parse_loss(out, label):
    m = re.search(r"loss:\s*([-\d.eE+]+)", out)
    assert m, "missing 'loss: ...'"
    return np.array(float(m.group(1)))

# what to compare for each experiment
FIELDS = {
    "compliance_gradient":  [("pos_final", parse_positions), ("pos_guess", parse_positions),
                             ("dL_dalpha", parse_gradient)],
    "x0_gradient":          [("pos_final", parse_positions), ("pos_guess", parse_positions),
                             ("loss", parse_loss), ("dL_dx0", parse_gradient)],
    "single_step_jacobian": [("J", parse_matrix)],
}

# ----------------
#   CONFIG / RUN
# ----------------

BASE = {
    "sim_rate":          "312",
    "n_seconds":         "2",
    "fps":               "24",
    "gravity":           "(0.0, -9.81, 0.0)",
    "target_compliance": "0.0001",
    "compliance":        "0.0002",   # differs -> non-trivial gradients
    "target_offset":     "(0.0, 0.0, 0.0)",
    "offset":            "(0.0, 0.0, 0.0)",
    "ground_ori":        "(0.0, -1000.0, 0.0)",
    "ground_normal":     "(0.0, 1.0, 0.0)",
    "export_obj":        "false",
}

FAR, CONTACT = "(0.0, -1000.0, 0.0)", "(0.0, -2.0, 0.0)"
X0_OFF       = "(0.01, 0.0, 0.02)"

def case(obj, experiment, loss, ground=FAR, offset="(0.0, 0.0, 0.0)"):
    return {"obj": obj, "experiment": experiment, "loss": loss,
            "ground_ori": ground, "offset": offset}

CASES = [
    case("chain(10)",  "compliance_gradient",       "mse_final_position"),
    case("chain(10)",  "compliance_gradient",       "mse_full_trajectory"),
    case("chain(10)",  "compliance_gradient",       "mse_frames_trajectory(24)"),
    case("chain(10)",  "compliance_gradient",       "mse_frames_trajectory(24)", ground=CONTACT),
    case("cloth(4,4)", "compliance_gradient",       "mse_final_position"),
    case("cloth(4,4)", "compliance_gradient",       "mse_frames_trajectory(24)", ground=CONTACT),
    case("chain(10)",  "x0_gradient",               "mse_final_position",        offset=X0_OFF),
    case("chain(10)",  "x0_gradient",               "mse_full_trajectory",       offset=X0_OFF),
    case("chain(10)",  "x0_gradient",               "mse_frames_trajectory(24)", ground=CONTACT, offset=X0_OFF),
    case("cloth(4,4)", "x0_gradient",               "mse_frames_trajectory(24)", offset=X0_OFF),
    case("cloth(3,3)", "x0_gradient",               "mse_final_position",        ground=CONTACT, offset=X0_OFF),
    case("chain(6)",   "single_step_jacobian(50)",  "mse_frames_trajectory(24)"),
    case("chain(6)",   "single_step_jacobian(500)", "mse_frames_trajectory(24)", ground=CONTACT),
    case("cloth(3,3)", "single_step_jacobian(80)",  "mse_frames_trajectory(24)"),
]

def write_temp_config(params):
    fd, path = tempfile.mkstemp(suffix=".conf")
    with os.fdopen(fd, "w") as f:
        for k, v in params.items():
            f.write(f"{k} = {v}\n")
    return path

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip()
                           else f"exit {r.returncode}")
    return r.stdout

# ----------------
#      MAIN
# ----------------

def run_case(c):
    """Returns (ok, detail). Raises nothing."""
    path = write_temp_config({**BASE, **c})
    try:
        cpp = run([CPP_EXE, path])
        jax = run([sys.executable, JAX_SCRIPT, path])
    except RuntimeError as e:
        return False, f"ERROR: {e}"
    finally:
        os.remove(path)

    exp = c["experiment"].split("(")[0]
    worst_field, worst_diff = None, 0.0
    for label, parser in FIELDS[exp]:
        a, b = parser(cpp, label), parser(jax, label)
        if a.shape != b.shape:
            return False, f"{label}: shape {a.shape} vs {b.shape}"
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        if not np.allclose(a, b, rtol=RTOL, atol=ATOL):
            return False, f"{label}: max abs diff {d:.3e}"
        if d > worst_diff:
            worst_field, worst_diff = label, d
    return True, f"max {worst_diff:.1e}" + (f" ({worst_field})" if worst_field else "")

def main():
    if not os.path.exists(CPP_EXE):
        print(f"C++ binary not found: {CPP_EXE}\nBuild it first: cmake --build build --config Release")
        sys.exit(2)

    failures = []
    for c in CASES:
        name = f'{c["obj"]:<10} | {c["experiment"]:<26} | {c["loss"]}'
        ok, detail = run_case(c)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}    {detail}")
        if not ok:
            failures.append(f"{name}: {detail}")

    print("=" * 60)
    print(f"{len(CASES)} cases: {len(CASES) - len(failures)} passed, {len(failures)} failed")
    for f in failures:
        print("  -", f)
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
