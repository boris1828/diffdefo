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

def apply_collider(x, w, collider):
    p0 = collider["origin"]
    n  = collider["normal"]

    signed_dist = (x - p0[None, :]) @ n
    penetration = jnp.minimum(signed_dist, 0.0)
    correction  = -penetration[:, None] * n[None, :]
    movable     = (w > 0).astype(x.dtype)[:, None]

    return x + correction * movable

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

def xpbd_step(x, v, w, pairs, rest, compliance, dt, gravity, n_iter, collider=None):
    movable = (w > 0).astype(x.dtype)[:, None]
    x_pred  = x + (v * dt + gravity * dt * dt) * movable

    lam = jnp.zeros(pairs.shape[0])

    x_new, _ = solve_constraints_jacobi(x_pred, w, pairs, rest, compliance, lam, dt, n_iter)

    if collider is not None:
        x_new = apply_collider(x_new, w, collider)

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
    # like parse_object_spec, but the argument list is optional: "name" or "name(args)"
    m = re.match(r"\s*(\w+)\s*(?:\((.*)\))?\s*$", spec)
    assert m, f"bad experiment spec: {spec}"
    name = m.group(1)
    args = [int(a) for a in (m.group(2) or "").split(",") if a.strip()]
    return name, args

# ----------------
#      LOSS
# ----------------

def make_loss(loss_spec, sim_rate, duration_s):
    # Returns loss_fn(sim_traj, target_traj) -> scalar.
    # Trajectories are the full per-step stacks, shape (n_steps, N, 3).
    name, args = parse_experiment_spec(loss_spec)

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
#   COMPL. GRAD
# ----------------

def run_compliance_experiment(config_path=os.path.join(_PROJ_ROOT, "src", "param.conf")):
    cfg = load_config(config_path)

    sim_rate   = int(cfg["sim_rate"])
    fps        = int(cfg["fps"])
    duration_s = int(cfg["n_seconds"])

    target_compliance_val = float(cfg["target_compliance"])
    guess_compliance_val  = float(cfg["compliance"])

    gravity       = cfg_vec3(cfg, "gravity")
    ground_origin = cfg_vec3(cfg, "ground_ori")
    ground_normal = cfg_vec3(cfg, "ground_normal")
    offset        = cfg_vec3(cfg, "offset")
    target_offset = cfg_vec3(cfg, "target_offset")
    ground        = make_ground_collider(origin=ground_origin, normal=ground_normal)

    assert jnp.array_equal(offset, target_offset), \
        "the offset must match in this implementation"
    
    dt          = 1.0 / float(sim_rate)
    n_iter      = 1
    n_steps     = int(sim_rate * duration_s)

    loss_spec   = cfg.get("loss", f"mse_frames_trajectory({fps})")
    loss_fn     = make_loss(loss_spec, sim_rate, duration_s)

    obj_spec    = cfg["obj"]

    x0, _, w_, pairs, rest, _ = make_object(obj_spec, guess_compliance_val)
    x0 = x0 + offset[None, :]
    n_constraints = pairs.shape[0]
    num_particles = x0.shape[0]

    def simulate(compliance, x0):
        def body(carry, _):
            x, v = carry
            x, v = xpbd_step(x, v, w_, pairs, rest, compliance, dt, gravity, n_iter, collider=ground)
            return (x, v), x
        v0 = jnp.zeros_like(x0)
        (_, _), trajectory = lax.scan(body, (x0, v0), None, length=n_steps)
        return trajectory

    # --- target  ---
    target_traj = simulate(jnp.full(n_constraints, target_compliance_val), x0)

    guess_traj = simulate(jnp.full(n_constraints, guess_compliance_val), x0)

    def loss(compliance):
        return loss_fn(simulate(compliance, x0), target_traj)

    guess_compliance = jnp.full(n_constraints, guess_compliance_val)
    loss_value = loss(guess_compliance)
    dL_dc      = grad(loss)(guess_compliance)

    def print_as_list(label, P, inline=False):
        M = P.shape[0]
        is_vec = (P.ndim > 1)
        def fmt(row):
            if is_vec: return "(" + ", ".join(f"{v:.8e}" for v in row) + ")"
            return f"{row:.8e}"
        if inline:
            print(f"{label} = [ " + ", ".join(fmt(P[i]) for i in range(M)) + " ]")
        else:
            print(f"{label} = [")
            for i in range(M - 1):
                print(f"  {fmt(P[i])},")
            print(f"  {fmt(P[M-1])}")
            print(f"]")

    print_as_list("pos_final", target_traj[-1], inline=True)
    print_as_list("pos_guess",  guess_traj[-1], inline=True)

    print(f"\nloss: {loss_value:.8e}")
    print(f"\n=== Compliance gradient ({n_constraints} constraints) ===")
    print_as_list("dL_dalpha", dL_dc, inline=True)

    print(f"\ndL/dcompliance sum:  {dL_dc.sum():.8e}")
    print(f"dL/dcompliance mean: {dL_dc.mean():.8e}")

    return loss_value, dL_dc

# ----------------
#  STEP JACOBIAN
# ----------------

def step_jacobian_experiment(step_index, config_path=os.path.join(_PROJ_ROOT, "src", "param.conf")):

    cfg = load_config(config_path)

    sim_rate       = int(cfg["sim_rate"])
    duration_s     = int(cfg["n_seconds"])
    compliance_val = float(cfg["compliance"])

    gravity       = cfg_vec3(cfg, "gravity")
    ground_origin = cfg_vec3(cfg, "ground_ori")
    ground_normal = cfg_vec3(cfg, "ground_normal")
    offset        = cfg_vec3(cfg, "offset")
    ground        = make_ground_collider(origin=ground_origin, normal=ground_normal)

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
        x, v = xpbd_step(x, v, w_, pairs, rest, compliance, dt, gravity, n_iter, collider=ground)
        return (x, v), None

    v0 = jnp.zeros_like(x0)
    (x_in, v_in), _ = lax.scan(body, (x0, v0), None, length=step_index - 1)

    def step_positions(x):
        x_out, _ = xpbd_step(x, v_in, w_, pairs, rest, compliance, dt, gravity, n_iter, collider=ground)
        return x_out

    P = x_in.shape[0]
    J = jax.jacobian(step_positions)(x_in)
    J = J.reshape(3 * P, 3 * P)

    print(f"=== d x^+ / d x^-  at update {step_index} / {n_steps} ===")
    print(f"Frobenius norm: {jnp.linalg.norm(J):.8e}")

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
    ground_origin = cfg_vec3(cfg, "ground_ori")
    ground_normal = cfg_vec3(cfg, "ground_normal")
    offset        = cfg_vec3(cfg, "offset")
    target_offset = cfg_vec3(cfg, "target_offset")
    ground        = make_ground_collider(origin=ground_origin, normal=ground_normal)

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
            x, v = xpbd_step(x, v, w_, pairs, rest, compliance, dt, gravity, n_iter, collider=ground)
            return (x, v), x
        v0 = jnp.zeros_like(x0)
        (_, _), trajectory = lax.scan(body, (x0, v0), None, length=n_steps)
        return trajectory

    # target (fixed reference): target_offset + target_compliance
    x0_target   = base_x0 + target_offset[None, :]
    target_traj = simulate(target_compliance, x0_target)

    # the initial positions we differentiate w.r.t.
    x0_guess = base_x0 + offset[None, :]

    def loss_x0(x0):
        return loss_fn(simulate(guess_compliance, x0), target_traj)

    loss_value = loss_x0(x0_guess)
    dL_dx0     = grad(loss_x0)(x0_guess)   # (N, 3)
    flat       = dL_dx0.reshape(-1)        # 3N, (x, y, z) interleaved -> matches C++ flatten

    print(f"=== d loss / d x0  ({num_particles} particles, {flat.shape[0]} dims) ===")
    print(f"loss: {loss_value:.8e}")
    print("dL_dx0 = [" + ", ".join(f"{float(v):.8e}" for v in flat) + "]")

    return dL_dx0

# ----------------
#    DISPATCH
# ----------------

def run_experiment(config_path=os.path.join(_PROJ_ROOT, "src", "param.conf")):
    cfg = load_config(config_path)
    name, args = parse_experiment_spec(cfg["experiment"])
    if name == "compliance_gradient":
        return run_compliance_experiment(config_path)
    elif name == "single_step_jacobian":
        assert len(args) == 1, f"single_step_jacobian expects 1 arg (step_index), got {len(args)}"
        return step_jacobian_experiment(args[0], config_path)
    elif name == "x0_gradient":
        return x0_gradient_experiment(config_path)
    else:
        raise ValueError(f"unknown experiment: {name}")

if __name__ == "__main__":
    config_path = (os.path.join(_PROJ_ROOT, sys.argv[1]) 
                    if len(sys.argv) > 1
                    else os.path.join(_PROJ_ROOT, "src", "param.conf"))
    run_experiment(config_path)
