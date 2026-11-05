#pragma once

// `diffpd --friction-check`: self-checks of the impulse method's Coulomb friction (update_contact_force_friction, forward only).
// Not a standalone header: it is #included once, just above main() in diffpd.cpp, after diffpd_ncp_check.h (it reuses its helpers).

namespace friction_check
{
using ncp_check::check;

inline Vec3 rnd(std::mt19937& g) { std::uniform_real_distribution<Real> u(-1.0, 1.0); return Vec3(u(g), u(g), u(g)); }

inline void check_function()
{
    std::printf("== update_contact_force_friction\n");
    const Vec3 n = Vec3::UnitY(), z = Vec3::Zero();
    const Real m = 2.0, mu = 0.5;

    check(update_contact_force_friction(Vec3(3, 1, 0), m, z, n, mu).isZero(0.0), "take-off (d_N > 0): r = 0");
    const Vec3 f_stick(0.4, -1.0, 0.2); // |d_T| = 0.447 <= mu * 1
    check((update_contact_force_friction(f_stick, m, z, n, mu) + f_stick).norm() < 1e-15, "stick (inside the cone): r = -d, so v+ has no relative motion");
    const Vec3 f_slip(3.0, -1.0, 0.0);
    const Vec3 r_slip = update_contact_force_friction(f_slip, m, z, n, mu);
    check((r_slip - Vec3(-mu, 1.0, 0.0)).norm() < 1e-15, "slip: r_N = -d_N, r_T = -mu r_N d_T/|d_T|");
    const Vec3 f_edge(0.5, -1.0, 0.0); // |d_T| = mu * r_N exactly
    check((update_contact_force_friction(f_edge, m, z, n, mu) - Vec3(-0.5, 1.0, 0.0)).norm() < 1e-15, "on the cone boundary: stick and slip agree");

    // moving obstacle: v_c shifts d, so a body already moving with the obstacle sticks to it
    const Vec3 v_c(2.0, 0.0, 0.0);
    check((update_contact_force_friction(Vec3(m * 2.0, -1.0, 0.0), m, v_c, n, mu) - Vec3(0, 1.0, 0)).norm() < 1e-15, "moving obstacle: no relative tangential velocity -> only r_N");

    // mu = 0 is the frictionless impulse, bit for bit
    std::mt19937 g(7);
    bool same = true;
    for (int i = 0; i < 1000; ++i)
    {
        const Vec3 f = rnd(g), v = rnd(g), nn = rnd(g).normalized();
        same = same && update_contact_force_friction(f, 1.3, v, nn, 0.0) == update_contact_force(f, 1.3, v, nn);
    }
    check(same, "mu = 0 equals update_contact_force on 1000 random inputs");

    // the cone: |r_T| <= mu r_N always, and r_T opposes d_T
    bool cone = true;
    for (int i = 0; i < 1000; ++i)
    {
        const Vec3 f = rnd(g) * 3.0, nn = rnd(g).normalized();
        const Vec3 r = update_contact_force_friction(f, 1.0, z, nn, 0.4);
        const Real rn = r.dot(nn);
        const Vec3 rt = r - rn * nn, dt = f - f.dot(nn) * nn;
        cone = cone && rt.norm() <= 0.4 * rn + 1e-12 && rt.dot(dt) <= 1e-12;
    }
    check(cone, "r is inside the Coulomb cone and r_T opposes d_T (1000 random)");
}

// A flat free cloth lying on the ground plane (every node moves alike, so the springs add nothing: the scheme has no error
// and the step is the exact scalar Coulomb recurrence below).
inline Tape run_flat(Real mu, int steps, const Vec3& gravity, const Vec3& v0, const Vec3& plane_velocity)
{
    AppConfig cfg = ncp_check::cloth_on_sphere(make_plane(Vec3(0, 0, 0), Vec3(0, 1, 0), plane_velocity));
    cfg.cloth.pin_mode = PinMode::NONE;
    cfg.cloth.origin   = Vec3(0.0, 0.0, 0.0);
    cfg.solver.gravity = gravity;
    cfg.solver.n_iters = 50;
    cfg.world.contact.friction = mu;
    WorldSetup world = prepare_world(cfg);
    world.collision.collect_backward = false;
    const Real dt = 1.0 / (cfg.solver.FPS * cfg.solver.frame_substeps);
    Object obj = build_run_cloth(cfg, Stiffness::uniform(cfg.stiffness));
    construct_pd_lhs(obj, dt);
    for (Index i = 0; i < obj.num_particles(); ++i) obj.v.segment<3>(3 * i) = v0;
    Tape tape;
    pd_contact(obj, world.collision, dt, gravity, cfg.solver.n_iters, steps, 1, tape, "friction_check", false, false);
    return tape;
}

// Independent scalar recurrence of the same law: w = relative tangential speed + h g_t, capped by mu h g_n.
inline Real expected_vx(Real mu, int n, Real g_t, Real g_n, Real v0, Real u, Real h)
{
    Real v = v0;
    const Real cap = mu * h * g_n;
    for (int i = 0; i < n; ++i)
    {
        const Real w = (v - u) + h * g_t;
        v = u + (std::abs(w) <= cap ? 0.0 : w - (w > 0 ? cap : -cap));
    }
    return v;
}

inline void check_plane(const char* name, Real mu, const Vec3& gravity, const Vec3& v0, Real u)
{
    const int steps = 30;
    const Real h = 1.0 / 60;
    const Tape tape = run_flat(mu, steps, gravity, v0, Vec3(u, 0, 0));
    Real err = 0.0, off_axis = 0.0;
    for (int n : { 1, 5, 13, 30 })
    {
        const PointsX& v = tape.velocities[n];
        err = std::max(err, (v.col(0).array() - expected_vx(mu, n, gravity.x(), -gravity.y(), v0.x(), u, h)).abs().maxCoeff());
        off_axis = std::max<Real>(off_axis, std::max(v.col(1).cwiseAbs().maxCoeff(), v.col(2).cwiseAbs().maxCoeff()));
    }
    std::printf("  [info] %s: vx(1 s/2) = %.6f, |vx - exact| %.2e, off-axis speed %.2e\n", name, tape.velocities[steps].col(0).mean(), err, off_axis);
    check(err < 1e-9 && off_axis < 1e-9, name);
}

inline void check_planes()
{
    std::printf("== cloth on a plane against the exact Coulomb recurrence\n");
    const Vec3 g0(0, -9.81, 0);
    const Real th = 30.0 * 3.14159265358979323846 / 180.0; // gravity of a 30 degree slope, tan = 0.577
    const Vec3 g_slope(9.81 * std::sin(th), -9.81 * std::cos(th), 0);
    check_plane("mu 0: a sliding cloth keeps its speed", 0.0, g0, Vec3(1, 0, 0), 0.0);
    check_plane("mu 0.5: a sliding cloth stops after mu g t = v0 and stays stopped", 0.5, g0, Vec3(1, 0, 0), 0.0);
    check_plane("slope 30 deg, mu 0.3 < tan: slides with a = g (sin - mu cos)", 0.3, g_slope, Vec3::Zero(), 0.0);
    check_plane("slope 30 deg, mu 0.7 > tan: sticks", 0.7, g_slope, Vec3::Zero(), 0.0);
    check_plane("plane moving at 1 m/s, mu 1: the cloth is dragged to the plane speed", 1.0, g0, Vec3::Zero(), 1.0);
}

inline void check_wiring()
{
    std::printf("== config\n");
    AppConfig c;
    check(*c.method_error() == '\0', "friction 0: no method error");
    c.world.contact.friction = 0.4;
    check(*c.method_error() != '\0', "friction > 0 with Guess+Target+Backward: error (Run disabled)");
    c.run_mode = RunMode::ForwardOnly;
    check(*c.method_error() == '\0', "friction > 0, forward only: ok");
    c.world.contact.model = ContactModel::Gap;
    check(*c.method_error() != '\0', "friction > 0 with the Gap model: error");

    AppConfig saved, loaded;
    saved.world.contact.friction = 0.37;
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "diffpd_friction_check.conf";
    const bool ok = save_config(path, saved) && load_config(path, loaded);
    std::filesystem::remove(path);
    check(ok && loaded.world.contact.friction == 0.37, "contact_friction survives save / load");
}
} // namespace friction_check

static int run_friction_check()
{
    ncp_check::failures = 0;
    friction_check::check_function();
    friction_check::check_planes();
    friction_check::check_wiring();
    std::printf(ncp_check::failures ? "friction-check: %d FAILED\n" : "friction-check: all ok\n", ncp_check::failures);
    return ncp_check::failures ? 1 : 0;
}
