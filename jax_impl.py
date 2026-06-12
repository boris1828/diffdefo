import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, lax
import os
import re
import sys

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))

jax.config.update("jax_enable_x64", True)

# State:       positions (N, 2), velocities (N, 2), inverse masses (N,)
# Constraints: pairs of indices (M, 2), rest lengths (M,), compliances (M,)

# ----------------
#     OBJECT
# ----------------

def make_chain(n_particles, spacing=1.0):
    x = jnp.stack([jnp.arange(n_particles) * spacing,
                   jnp.zeros(n_particles), 
                   jnp.zeros(n_particles)], axis=1)
    v = jnp.zeros_like(x)
    w = jnp.ones(n_particles)
    w = w.at[0].set(0.0) 
    
    pairs = jnp.stack([jnp.arange(n_particles - 1),
                       jnp.arange(1, n_particles)], axis=1)
    rest = jnp.full(n_particles - 1, spacing)
    compliance = jnp.full(n_particles - 1, 1e-6)
    
    return x, v, w, pairs, rest, compliance

def make_cloth(width, height, spacing=1.0, compliance=1e-6):

    ii, jj = jnp.meshgrid(jnp.arange(width), jnp.arange(height), indexing='ij')
    x = jnp.stack([
        ii.flatten() * spacing,   
        jnp.zeros(width * height),
        jj.flatten() * spacing,   
    ], axis=1)

    v = jnp.zeros_like(x)

    w = jnp.ones(width * height)
    def idx(i, j):
        return i * height + j
    w = w.at[idx(0, 0)].set(0.0)
    w = w.at[idx(width - 1, 0)].set(0.0)

    pair_list = []
    rest_list = []

    # structural: horizontal (along i) and vertical (along j)
    for i in range(width):
        for j in range(height - 1):
            pair_list.append((idx(i, j), idx(i, j + 1)))
            rest_list.append(spacing)
    for i in range(width - 1):
        for j in range(height):
            pair_list.append((idx(i, j), idx(i + 1, j)))
            rest_list.append(spacing)

    # shear: both diagonals of each cell
    diag = spacing * jnp.sqrt(2.0)
    for i in range(width - 1):
        for j in range(height - 1):
            pair_list.append((idx(i, j),     idx(i + 1, j + 1)))
            rest_list.append(diag)
            pair_list.append((idx(i + 1, j), idx(i, j + 1)))
            rest_list.append(diag)

    pairs      = jnp.array(pair_list, dtype=jnp.int32)
    rest       = jnp.array(rest_list, dtype=x.dtype)
    compliance = jnp.full(pairs.shape[0], compliance)

    return x, v, w, pairs, rest, compliance

def make_object(spec_str, compliance, spacing=1.0):
    name, args = parse_object_spec(spec_str)
    if name == "chain":
        assert len(args) == 1, f"chain expects 1 arg, got {len(args)}"
        x, v, w, pairs, rest, _ = make_chain(args[0], spacing)
    elif name == "cloth":
        assert len(args) == 2, f"cloth expects 2 args, got {len(args)}"
        x, v, w, pairs, rest, _ = make_cloth(args[0], args[1], spacing)
    else:
        raise ValueError(f"unknown object type: {name}")
    comp = jnp.full(pairs.shape[0], compliance)
    return x, v, w, pairs, rest, comp

# ----------------
#   CONSTRAINT
# ----------------

def project_distance(x_i, x_j, w_i, w_j, rest, compliance, lam, dt):
    delta = x_i - x_j
    dist = jnp.linalg.norm(delta)
    
    n = delta / jnp.where(dist > 1e-12, dist, 1.0)
    
    C = dist - rest
    alpha_tilde = compliance / (dt * dt)
    
    denom = w_i + w_j + alpha_tilde
    dlam  = (-C - alpha_tilde * lam) / denom
    
    x_i_new = x_i + dlam * w_i * n
    x_j_new = x_j - dlam * w_j * n
    lam_new = lam + dlam
    
    return x_i_new, x_j_new, lam_new

def solve_constraints_gauss_seidel(x, w, pairs, rest, compliance, lam, dt, n_iter):
    def one_constraint(carry, c_idx):
        x, lam = carry
        i, j   = pairs[c_idx, 0], pairs[c_idx, 1]
        x_i_new, x_j_new, lam_new = project_distance(
            x[i], x[j], w[i], w[j],
            rest[c_idx], compliance[c_idx], lam[c_idx], dt
        )
        x   = x.at[i].set(x_i_new).at[j].set(x_j_new)
        lam = lam.at[c_idx].set(lam_new)
        return (x, lam), None
    
    def one_iteration(carry, _):
        (x, lam), _ = lax.scan(one_constraint, carry, jnp.arange(pairs.shape[0]))
        return (x, lam), None
    
    (x, lam), _ = lax.scan(one_iteration, (x, lam), None, length=n_iter)
    return x, lam

def compute_distance_correction(x_i, x_j, w_i, w_j, rest, compliance, lam, dt):
    delta = x_i - x_j
    dist = jnp.linalg.norm(delta)
    
    n = delta / jnp.where(dist > 1e-12, dist, 1.0)
    
    C = dist - rest
    alpha_tilde = compliance / (dt * dt)
    
    denom = w_i + w_j + alpha_tilde
    dlam = (-C - alpha_tilde * lam) / denom
    
    dx_i =  dlam * w_i * n
    dx_j = -dlam * w_j * n
    
    return dx_i, dx_j, dlam

# ----------------
#   COLLIDER
# ----------------

def make_ground_collider(origin, normal):
    normal = jnp.asarray(normal, dtype=jnp.float64)
    normal = normal / jnp.linalg.norm(normal)
    return {
        "kind":   "halfspace",
        "origin": jnp.asarray(origin, dtype=jnp.float64),
        "normal": normal,
    }

def make_sphere_collider(center, radius):
    return {
        "kind":   "sphere",
        "center": jnp.asarray(center, dtype=jnp.float64),
        "radius": jnp.asarray(radius, dtype=jnp.float64),
    }

def apply_collider(x, w, collider):
    kind = collider["kind"]
    if kind == "halfspace": return _apply_halfspace(x, w, collider)
    if kind == "sphere":    return _apply_sphere(x, w, collider)
    raise ValueError(f"unknown collider kind '{kind}'")

def _apply_halfspace(x, w, collider):
    p0 = collider["origin"]
    n  = collider["normal"]

    signed_dist = (x - p0[None, :]) @ n
    penetration = jnp.minimum(signed_dist, 0.0)
    correction  = -penetration[:, None] * n[None, :]
    movable     = (w > 0).astype(x.dtype)[:, None]

    return x + correction * movable

def _apply_sphere(x, w, collider):
    c = collider["center"]
    r = collider["radius"]

    delta  = x - c[None, :]                              # (N, 3)
    d      = jnp.linalg.norm(delta, axis=1)              # (N,)
    safe_d = jnp.where(d > 1e-12, d, 1.0)                # avoid 0/0 (NaN-safe gradient)
    x_surf = c[None, :] + r * delta / safe_d[:, None]    # project onto the surface: c + r * n

    inside  = d < r
    movable = w > 0
    use     = (inside & movable)[:, None]

    return jnp.where(use, x_surf, x)

# ---- colliders field parser (mirrors the C++ ColliderSet) ----

def _split_top_level(s, delim=","):
    # split on `delim`, but only at top level (not inside parentheses)
    out, depth, cur = [], 0, ""
    for ch in s:
        if   ch == "(": depth += 1
        elif ch == ")": depth -= 1
        if ch == delim and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    out.append(cur)
    return out

def _collider_vec3(s):
    parts = [float(p) for p in s.replace("(", "").replace(")", "").split(",")]
    assert len(parts) == 3, f"collider vec3 needs 3 components, got: {s}"
    return jnp.array(parts)

def _make_collider(name, args):
    if name == "halfspace":
        assert len(args) == 2, f"halfspace expects 2 args (origin, normal), got {len(args)}"
        return make_ground_collider(_collider_vec3(args[0]), _collider_vec3(args[1]))
    if name == "sphere":
        assert len(args) == 2, f"sphere expects 2 args (center, radius), got {len(args)}"
        return make_sphere_collider(_collider_vec3(args[0]), float(args[1]))
    raise ValueError(f"unknown collider '{name}'")

class ColliderSet:
    def __init__(self, colliders):
        self.colliders = colliders          # list of collider dicts

    @staticmethod
    def parse(field):
        lb, rb = field.find("["), field.rfind("]")
        assert lb != -1 and rb > lb, f"colliders must be a [...] list, got: {field}"
        items = []
        for entry in _split_top_level(field[lb + 1:rb]):
            entry = entry.strip()
            if not entry:
                continue
            lp, rp = entry.find("("), entry.rfind(")")
            assert lp != -1 and rp > lp, f"collider must be name(...), got: {entry}"
            name = entry[:lp].strip()
            args = [a.strip() for a in _split_top_level(entry[lp + 1:rp]) if a.strip()]
            items.append(_make_collider(name, args))
        return ColliderSet(items)

    @staticmethod
    def from_cfg(cfg):
        return ColliderSet.parse(cfg["colliders"]) if "colliders" in cfg else ColliderSet([])

    def apply(self, x, w):
        for c in self.colliders:
            x = apply_collider(x, w, c)
        return x

    def __len__(self):
        return len(self.colliders)

# ----------------
#      XPBD
# ----------------

def solve_constraints_jacobi(x, w, pairs, rest, compliance, lam, dt, n_iter):
    n_particles = x.shape[0]
    
    def one_iteration(carry, _):
        x, lam = carry
        
        # 1. Compute all corrections in parallel against the SAME state.
        #    vmap over constraints. Inputs/outputs all have a leading constraint axis.
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        
        dx_i, dx_j, dlam = jax.vmap(compute_distance_correction)(
            x[i_idx], x[j_idx],
            w[i_idx], w[j_idx],
            rest, compliance, lam, jnp.full_like(rest, dt)
        )
        # Shapes: dx_i, dx_j -> (M, 3),  dlam -> (M,)
        
        # 2. Scatter-add corrections into x.
        #    Each particle accumulates contributions from every constraint it's in.
        dx = jnp.zeros_like(x)
        dx = dx.at[i_idx].add(dx_i)
        dx = dx.at[j_idx].add(dx_j)
        
        x_new   = x + dx
        lam_new = lam + dlam
        
        return (x_new, lam_new), None
    
    (x, lam), _ = lax.scan(one_iteration, (x, lam), None, length=n_iter)
    return x, lam

def xpbd_step(x, v, w, pairs, rest, compliance, dt, gravity, n_iter, colliders=None):
    movable = (w > 0).astype(x.dtype)[:, None]
    x_pred  = x + (v * dt + gravity * dt * dt) * movable

    lam = jnp.zeros(pairs.shape[0])

    x_new, _ = solve_constraints_jacobi(x_pred, w, pairs, rest, compliance, lam, dt, n_iter)

    if colliders is not None:
        x_new = colliders.apply(x_new, w)

    v_new = (x_new - x) / dt

    return x_new, v_new

# ----------------
#      CONFIG
# ----------------

def load_config(path):
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            # comments: # ; or //
            if not line or line[0] in "#;":
                continue
            if line.startswith("//"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            cfg[key.strip()] = value.strip()
    return cfg

def cfg_vec3(cfg, key):
    s = cfg[key].replace("(", "").replace(")", "")
    parts = [float(p) for p in s.split(",")]
    assert len(parts) == 3, f"vec3 key '{key}' needs 3 values, got: {cfg[key]}"
    return jnp.array(parts)

def parse_object_spec(spec):
    m = re.match(r"\s*(\w+)\s*\((.*)\)\s*$", spec)
    assert m, f"bad object spec: {spec}"
    name = m.group(1)
    args = [int(a) for a in m.group(2).split(",") if a.strip()]
    return name, args

def parse_experiment_spec(spec):
    # like parse_object_spec, but the argument list is optional: "name" or "name(args)".
    # Returns int view (args) and Real view (rargs) of each token, mirroring the C++ ObjectSpec.
    m = re.match(r"\s*(\w+)\s*(?:\((.*)\))?\s*$", spec)
    assert m, f"bad experiment spec: {spec}"
    name  = m.group(1)
    toks  = [t for t in (m.group(2) or "").split(",") if t.strip()]
    args  = [int(float(t)) for t in toks]
    rargs = [float(t) for t in toks]
    return name, args, rargs

# ----------------
#      LOSS
# ----------------

def make_loss(loss_spec, sim_rate, duration_s):
    # Returns loss_fn(sim_traj, target_traj) -> scalar.
    # Trajectories are the full per-step stacks, shape (n_steps, N, 3).
    name, args, _ = parse_experiment_spec(loss_spec)

    if name == "mse_final_position":
        def loss_fn(sim_traj, target_traj):
            return jnp.mean((sim_traj[-1] - target_traj[-1]) ** 2)

    elif name == "mse_full_trajectory":
        def loss_fn(sim_traj, target_traj):
            return jnp.mean((sim_traj - target_traj) ** 2)

    elif name == "mse_frames_trajectory":
        assert len(args) == 1, f"mse_frames_trajectory expects 1 arg (fps), got {len(args)}"
        fps         = args[0]
        n_frames    = fps * duration_s
        frame_steps = jnp.round(jnp.arange(n_frames) * sim_rate / fps).astype(jnp.int32)
        def loss_fn(sim_traj, target_traj):
            return jnp.mean((sim_traj[frame_steps] - target_traj[frame_steps]) ** 2)

    else:
        raise ValueError(f"unknown loss: {name}")

    return loss_fn

# ----------------
#      OUTPUT
# ----------------

def print_positions(label, P):
    rows = ["(" + ", ".join(f"{float(c):.16e}" for c in P[i]) + ")" for i in range(P.shape[0])]
    print(f"{label} = [" + ", ".join(rows) + " ]")

def print_vector(label, v):
    print(f"{label} = [" + ", ".join(f"{float(x):.16e}" for x in v) + "]")

def print_matrix(label, M):
    n_rows = M.shape[0]
    print(f"{label} = [")
    for r in range(n_rows):
        row = ", ".join(f"{float(x):.16e}" for x in M[r])
        print(f"[{row}]" + ("," if r + 1 < n_rows else ""))
    print("]")

def print_final_positions(target_final, guess_final):
    print("\n=== Final Positions ===")
    print_positions("pos_final", target_final)
    print_positions("pos_guess", guess_final)

# ----------------
#   COMPLIANCE
# ----------------

def _compliance_setup(cfg):
    sim_rate   = int(cfg["sim_rate"])
    fps        = int(cfg["fps"])
    duration_s = int(cfg["n_seconds"])

    target_compliance_val = float(cfg["target_compliance"])
    guess_compliance_val  = float(cfg["compliance"])

    gravity = cfg_vec3(cfg, "gravity")
    offset, target_offset = cfg_vec3(cfg, "offset"), cfg_vec3(cfg, "target_offset")
    colliders = ColliderSet.from_cfg(cfg)

    assert jnp.array_equal(offset, target_offset), "the offset must match in this implementation"

    dt      = 1.0 / float(sim_rate)
    n_iter  = 1
    n_steps = int(sim_rate * duration_s)
    loss_fn = make_loss(cfg.get("loss", f"mse_frames_trajectory({fps})"), sim_rate, duration_s)

    x0, _, w_, pairs, rest, _ = make_object(cfg["obj"], guess_compliance_val)
    x0 = x0 + offset[None, :]
    n_constraints = pairs.shape[0]

    def simulate(compliance):
        def body(carry, _):
            x, v = carry
            x, v = xpbd_step(x, v, w_, pairs, rest, compliance, dt, gravity, n_iter, colliders=colliders)
            return (x, v), x
        (_, _), trajectory = lax.scan(body, (x0, jnp.zeros_like(x0)), None, length=n_steps)
        return trajectory

    target_traj = simulate(jnp.full(n_constraints, target_compliance_val))

    def loss_of(compliance):
        return loss_fn(simulate(compliance), target_traj)

    return {
        "n_constraints":         n_constraints,
        "target_compliance_val": target_compliance_val,
        "guess_compliance_val":  guess_compliance_val,
        "simulate":              simulate,
        "target_traj":           target_traj,
        "loss_of":               loss_of,
    }

def run_compliance_experiment(config_path=os.path.join(_PROJ_ROOT, "src", "param.conf")):
    s = _compliance_setup(load_config(config_path))

    guess_compliance = jnp.full(s["n_constraints"], s["guess_compliance_val"])
    loss_value       = s["loss_of"](guess_compliance)
    dL_dc            = grad(s["loss_of"])(guess_compliance)
    guess_traj       = s["simulate"](guess_compliance)

    print_final_positions(s["target_traj"][-1], guess_traj[-1])

    print("\n=== Compliance gradient ===")
    print_vector("dL_dalpha", dL_dc)

    print(f"\ndL/dcompliance sum:  {dL_dc.sum():.16e}")
    print(f"dL/dcompliance mean: {dL_dc.mean():.16e}")

    return loss_value, dL_dc

def compliance_optimization_experiment(lr, iters, config_path=os.path.join(_PROJ_ROOT, "src", "param.conf")):
    s  = _compliance_setup(load_config(config_path))
    nc = s["n_constraints"]

    def loss_scalar(c):
        return s["loss_of"](jnp.full(nc, c))

    loss_and_grad = jax.value_and_grad(loss_scalar)

    compliance = s["guess_compliance_val"]
    for it in range(iters):
        loss_val, grad_val = loss_and_grad(compliance)
        compliance = compliance - lr * grad_val
        print(f"iter {it}  loss: {float(loss_val):.16e}  grad: {float(grad_val):.16e}  compliance: {float(compliance):.16e}")

# ----------------
#  STEP JACOBIAN
# ----------------

def step_jacobian_experiment(step_index, config_path=os.path.join(_PROJ_ROOT, "src", "param.conf")):

    cfg = load_config(config_path)

    sim_rate       = int(cfg["sim_rate"])
    duration_s     = int(cfg["n_seconds"])
    compliance_val = float(cfg["compliance"])

    gravity   = cfg_vec3(cfg, "gravity")
    offset    = cfg_vec3(cfg, "offset")
    colliders = ColliderSet.from_cfg(cfg)

    dt      = 1.0 / float(sim_rate)
    n_iter  = 1
    n_steps = int(sim_rate * duration_s)

    assert 1 <= step_index <= n_steps, f"step_index must be in [1, {n_steps}], got {step_index}"

    obj_spec = cfg["obj"]
    x0, _, w_, pairs, rest, _ = make_object(obj_spec, compliance_val)
    x0 = x0 + offset[None, :]
    compliance = jnp.full(pairs.shape[0], compliance_val)

    def body(carry, _):
        x, v = carry
        x, v = xpbd_step(x, v, w_, pairs, rest, compliance, dt, gravity, n_iter, colliders=colliders)
        return (x, v), None

    v0 = jnp.zeros_like(x0)
    (x_in, v_in), _ = lax.scan(body, (x0, v0), None, length=step_index - 1)

    def step_positions(x):
        x_out, _ = xpbd_step(x, v_in, w_, pairs, rest, compliance, dt, gravity, n_iter, colliders=colliders)
        return x_out

    P = x_in.shape[0]
    J = jax.jacobian(step_positions)(x_in)
    J = J.reshape(3 * P, 3 * P)

    print(f"=== d x^+ / d x^-  at update {step_index} / {n_steps} ===")
    print_matrix("J", J)

    return J, x_in, v_in

# ----------------
#   X0 GRADIENT
# ----------------

def x0_gradient_experiment(config_path=os.path.join(_PROJ_ROOT, "src", "param.conf")):
    cfg = load_config(config_path)

    sim_rate   = int(cfg["sim_rate"])
    fps        = int(cfg["fps"])
    duration_s = int(cfg["n_seconds"])

    target_compliance_val = float(cfg["target_compliance"])
    guess_compliance_val  = float(cfg["compliance"])

    gravity       = cfg_vec3(cfg, "gravity")
    offset        = cfg_vec3(cfg, "offset")
    target_offset = cfg_vec3(cfg, "target_offset")
    colliders     = ColliderSet.from_cfg(cfg)

    dt          = 1.0 / float(sim_rate)
    n_iter      = 1
    n_steps     = int(sim_rate * duration_s)

    loss_spec   = cfg.get("loss", f"mse_frames_trajectory({fps})")
    loss_fn     = make_loss(loss_spec, sim_rate, duration_s)

    obj_spec    = cfg["obj"]

    base_x0, _, w_, pairs, rest, _ = make_object(obj_spec, guess_compliance_val)
    n_constraints = pairs.shape[0]
    num_particles = base_x0.shape[0]

    target_compliance = jnp.full(n_constraints, target_compliance_val)
    guess_compliance  = jnp.full(n_constraints, guess_compliance_val)

    def simulate(compliance, x0):
        def body(carry, _):
            x, v = carry
            x, v = xpbd_step(x, v, w_, pairs, rest, compliance, dt, gravity, n_iter, colliders=colliders)
            return (x, v), x
        v0 = jnp.zeros_like(x0)
        (_, _), trajectory = lax.scan(body, (x0, v0), None, length=n_steps)
        return trajectory

    # target (fixed reference): target_offset + target_compliance
    x0_target   = base_x0 + target_offset[None, :]
    target_traj = simulate(target_compliance, x0_target)

    # the initial positions we differentiate w.r.t.
    x0_guess   = base_x0 + offset[None, :]
    guess_traj = simulate(guess_compliance, x0_guess)

    def loss_x0(x0):
        return loss_fn(simulate(guess_compliance, x0), target_traj)

    loss_value = loss_x0(x0_guess)
    dL_dx0     = grad(loss_x0)(x0_guess)   # (N, 3)
    flat       = dL_dx0.reshape(-1)        # 3N, (x, y, z) interleaved -> matches C++ flatten

    print_final_positions(target_traj[-1], guess_traj[-1])

    print(f"=== d loss / d x0  ({num_particles} particles, {flat.shape[0]} dims) ===")
    print(f"loss: {loss_value:.16e}")
    print_vector("dL_dx0", flat)

    return dL_dx0

# ----------------
#    DISPATCH
# ----------------

def forward_simulation_experiment(config_path=os.path.join(_PROJ_ROOT, "src", "param.conf")):
    cfg = load_config(config_path)

    sim_rate   = int(cfg["sim_rate"])
    duration_s = int(cfg["n_seconds"])
    gravity    = cfg_vec3(cfg, "gravity")
    colliders  = ColliderSet.from_cfg(cfg)
    dt         = 1.0 / float(sim_rate)
    n_iter     = 1
    n_steps    = int(sim_rate * duration_s)

    def forward_final(compliance_val, offset):
        x0, _, w_, pairs, rest, _ = make_object(cfg["obj"], compliance_val)
        x0   = x0 + offset[None, :]
        comp = jnp.full(pairs.shape[0], compliance_val)

        def body(carry, _):
            x, v = carry
            x, v = xpbd_step(x, v, w_, pairs, rest, comp, dt, gravity, n_iter, colliders=colliders)
            return (x, v), None

        (x_final, _), _ = lax.scan(body, (x0, jnp.zeros_like(x0)), None, length=n_steps)
        return x_final

    target_final = forward_final(float(cfg["target_compliance"]), cfg_vec3(cfg, "target_offset"))
    guess_final  = forward_final(float(cfg["compliance"]),        cfg_vec3(cfg, "offset"))

    print("\n=== Forward simulation ===")
    print_positions("pos_final", target_final)
    print_positions("pos_guess", guess_final)

    return target_final, guess_final

def run_experiment(config_path=os.path.join(_PROJ_ROOT, "src", "param.conf")):
    cfg = load_config(config_path)
    name, args, rargs = parse_experiment_spec(cfg["experiment"])
    if name == "forward_simulation":
        return forward_simulation_experiment(config_path)
    elif name == "compliance_gradient":
        return run_compliance_experiment(config_path)
    elif name == "single_step_jacobian":
        assert len(args) == 1, f"single_step_jacobian expects 1 arg (step_index), got {len(args)}"
        return step_jacobian_experiment(args[0], config_path)
    elif name == "x0_gradient":
        return x0_gradient_experiment(config_path)
    elif name == "compliance_optimization":
        assert len(args) == 2, f"compliance_optimization expects 2 args (lr, iters), got {len(args)}"
        return compliance_optimization_experiment(rargs[0], args[1], config_path)
    else:
        raise ValueError(f"unknown experiment: {name}")

if __name__ == "__main__":
    config_path = (os.path.join(_PROJ_ROOT, sys.argv[1]) 
                    if len(sys.argv) > 1
                    else os.path.join(_PROJ_ROOT, "src", "param.conf"))
    run_experiment(config_path)
