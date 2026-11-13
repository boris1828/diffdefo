#pragma once

// `diffpd --friction-check`: self-checks of the impulse method's Coulomb friction (impulse_coulomb and the model / rule split of contact_impulse.h, forward only).
// Not a standalone header: it is #included once, just above main() in diffpd.cpp, after diffpd_ncp_check.h (it reuses its helpers).

namespace friction_check
{
using ncp_check::check;

inline Vec3 rnd(std::mt19937& g) { std::uniform_real_distribution<Real> u(-1.0, 1.0); return Vec3(u(g), u(g), u(g)); }

inline Vec3 coul(const Vec3& f, Real m, const Vec3& v_c, const Vec3& n, Real mu) { return impulse_coulomb(d_relvel(f, m, v_c), n, mu); }
inline Vec3 free_relvel(const Vec3& f, Real m, const Vec3& v_c, const Vec3& n) { return impulse_frictionless(d_n_relvel(f, m, v_c, n), n); }

// impulse_coulomb as it was before coulomb_state existed, verbatim: the refactor must not change a single bit.
inline Vec3 impulse_coulomb_reference(const Vec3& d, const Vec3& n, Real mu)
{
    const Real d_n = d.dot(n);
    if (d_n >= 0.0) return Vec3::Zero();
    const Vec3 d_t = d - d_n * n;
    const Real r_n = -d_n;
    if (d_t.norm() <= mu * r_n) return -d;
    return r_n * n - mu * r_n * d_t.normalized();
}

// coulomb_state: the one branch test shared by the rule (and, later, by its derivative in the backward pass).
inline void check_state(std::mt19937& g)
{
    using S = CoulombState;
    const Vec3 n = Vec3::UnitY();
    const Real mu = 0.5;
    check(coulomb_state(Vec3(3, 1, 0), n, mu) == S::TakeOff
       && coulomb_state(Vec3(0.4, -1, 0.2), n, mu) == S::Stick
       && coulomb_state(Vec3(3, -1, 0), n, mu) == S::Slip, "coulomb_state: the take-off / stick / slip examples above");
    check(coulomb_state(Vec3(0.5, -1, 0), n, mu) == S::Stick, "coulomb_state: the cone boundary |d_T| = mu r_n is Stick");
    check(coulomb_state(Vec3(2, 0, 0), n, mu) == S::TakeOff && coulomb_state(Vec3(0, 0, 0), n, mu) == S::TakeOff,
          "coulomb_state: d_n = 0 is TakeOff (also d = 0)");
    check(coulomb_state(Vec3(0, -1, 0), n, 0.0) == S::Stick && coulomb_state(Vec3(0, -1, 0), n, mu) == S::Stick,
          "coulomb_state: d_T = 0 is Stick for every mu, mu = 0 included (pressed straight onto a plane)");
    check(coulomb_state(Vec3(1e-9, -1, 0), n, 0.0) == S::Slip, "coulomb_state: mu = 0 with d_T != 0 is Slip");
    check(coulomb_state(Vec3(std::nan(""), -1, 0), n, mu) == S::Slip && impulse_coulomb(Vec3(std::nan(""), -1, 0), n, mu).hasNaN(),
          "coulomb_state: NaN d -> Slip, the impulse stays NaN (as before)");

    // Random inputs, plus the measure-zero cases a random draw never hits: d_T = 0 exactly and d on the cone boundary.
    bool bits = true, overloads = true, consistent = true;
    int count[3] = { 0, 0, 0 };
    for (int i = 0; i < 3000; ++i)
    {
        const Vec3 nn = rnd(g).normalized();
        const Real m = (i % 4 == 0) ? 0.0 : std::abs(rnd(g).x()); // every 4th mu = 0
        Vec3 d = rnd(g) * 3.0;
        if (i % 7 == 0) d = d.dot(nn) * nn;                                           // d_T = 0 (up to the rounding of d_n n)
        else if (i % 11 == 0) { const Vec3 t = (d - d.dot(nn) * nn).normalized(); d = -nn + m * t; } // on the boundary, up to rounding

        const Real d_n = d.dot(nn);
        const S    s   = coulomb_state(d, nn, m);
        const Vec3 r   = impulse_coulomb(d, nn, m);
        ++count[(int)s];

        bits      = bits && r == impulse_coulomb_reference(d, nn, m);
        overloads = overloads && s == coulomb_state(d_n, (d - d_n * nn).norm(), m);
        const Real r_n = r.dot(nn);
        const Vec3 r_t = r - r_n * nn;
        if (s == S::TakeOff) consistent = consistent && d_n >= 0.0 && r.isZero(0.0);
        if (s == S::Stick)   consistent = consistent && d_n < 0.0 && r == -d;
        if (s == S::Slip)    consistent = consistent && d_n < 0.0 && (d - d_n * nn).norm() > 0.0
                                                    && std::abs(r_t.norm() - m * r_n) <= 1e-12 * (1.0 + r_n);
    }
    std::printf("  [info] random draws: %d take-off, %d stick, %d slip\n", count[0], count[1], count[2]);
    check(bits, "impulse_coulomb is bit-identical to the pre-coulomb_state body (3000 inputs incl. d_T = 0 and boundary)");
    check(overloads, "coulomb_state(d, n, mu) == coulomb_state(d_n, |d_T|, mu)");
    check(consistent, "the state matches the impulse: take-off r = 0, stick r = -d, slip |r_T| = mu r_n with d_T != 0");
}

inline void check_function()
{
    std::printf("== impulse_coulomb (rule) and the model builders\n");
    const Vec3 n = Vec3::UnitY(), z = Vec3::Zero();
    const Real m = 2.0, mu = 0.5;

    check(coul(Vec3(3, 1, 0), m, z, n, mu).isZero(0.0), "take-off (d_N > 0): r = 0");
    const Vec3 f_stick(0.4, -1.0, 0.2); // |d_T| = 0.447 <= mu * 1
    check((coul(f_stick, m, z, n, mu) + f_stick).norm() < 1e-15, "stick (inside the cone): r = -d, so v+ has no relative motion");
    const Vec3 f_slip(3.0, -1.0, 0.0);
    const Vec3 r_slip = coul(f_slip, m, z, n, mu);
    check((r_slip - Vec3(-mu, 1.0, 0.0)).norm() < 1e-15, "slip: r_N = -d_N, r_T = -mu r_N d_T/|d_T|");
    const Vec3 f_edge(0.5, -1.0, 0.0); // |d_T| = mu * r_N exactly
    check((coul(f_edge, m, z, n, mu) - Vec3(-0.5, 1.0, 0.0)).norm() < 1e-15, "on the cone boundary: stick and slip agree");

    // moving obstacle: v_c shifts d, so a body already moving with the obstacle sticks to it
    const Vec3 v_c(2.0, 0.0, 0.0);
    check((coul(Vec3(m * 2.0, -1.0, 0.0), m, v_c, n, mu) - Vec3(0, 1.0, 0)).norm() < 1e-15, "moving obstacle: no relative tangential velocity -> only r_N");

    // mu = 0 is the frictionless impulse, bit for bit
    std::mt19937 g(7);
    bool same = true;
    for (int i = 0; i < 1000; ++i)
    {
        const Vec3 f = rnd(g), v = rnd(g), nn = rnd(g).normalized();
        same = same && coul(f, 1.3, v, nn, 0.0) == free_relvel(f, 1.3, v, nn);
    }
    check(same, "mu = 0 equals impulse_frictionless on 1000 random inputs");

    // the cone: |r_T| <= mu r_N always, and r_T opposes d_T
    bool cone = true;
    for (int i = 0; i < 1000; ++i)
    {
        const Vec3 f = rnd(g) * 3.0, nn = rnd(g).normalized();
        const Vec3 r = coul(f, 1.0, z, nn, 0.4);
        const Real rn = r.dot(nn);
        const Vec3 rt = r - rn * nn, dt = f - f.dot(nn) * nn;
        cone = cone && rt.norm() <= 0.4 * rn + 1e-12 && rt.dot(dt) <= 1e-12;
    }
    check(cone, "r is inside the Coulomb cone and r_T opposes d_T (1000 random)");

    check_state(g);

    // the two models only build d: gap-aware = relative velocity with the virtual obstacle velocity P_T v_c - (phi/h) n
    bool eq = true, scalar = true, disp = true;
    for (int i = 0; i < 1000; ++i)
    {
        const Vec3 f = rnd(g) * 3.0, vc = rnd(g), nn = rnd(g).normalized();
        const Real m_i = 1.3, phi_h = rnd(g).x();
        const Vec3 vc_t = vc - vc.dot(nn) * nn;
        eq     = eq     && (d_gap(f, m_i, vc, nn, phi_h) - d_relvel(f, m_i, vc_t - phi_h * nn)).norm() < 1e-13;
        scalar = scalar && std::abs(d_n_gap(f, m_i, nn, phi_h) - d_gap(f, m_i, vc, nn, phi_h).dot(nn)) < 1e-13
                        && std::abs(d_n_relvel(f, m_i, vc, nn) - d_relvel(f, m_i, vc).dot(nn)) < 1e-13;
        // the dispatcher: each model with each rule
        Contact c; c.normal = nn; c.v_c = vc; c.gap = phi_h * 0.01; // h = 0.01
        const Real h = 0.01;
        disp = disp && contact_impulse(c, true,  f, m_i, h, 0.0) == impulse_frictionless(d_n_gap(f, m_i, nn, c.gap / h), nn)
                    && contact_impulse(c, false, f, m_i, h, 0.0) == impulse_frictionless(d_n_relvel(f, m_i, vc, nn), nn)
                    && contact_impulse(c, true,  f, m_i, h, 0.4) == impulse_coulomb(d_gap(f, m_i, vc, nn, c.gap / h), nn, 0.4)
                    && contact_impulse(c, false, f, m_i, h, 0.4) == impulse_coulomb(d_relvel(f, m_i, vc), nn, 0.4);
    }
    check(eq, "d_gap(v_c, phi/h) = d_relvel(P_T v_c - (phi/h) n): only the tangential v_c enters (1000 random)");
    check(scalar, "scalar d_n_gap / d_n_relvel = n . (vector d) (1000 random)");
    check(disp, "contact_impulse picks model and rule as specified (gap/rel-vel x mu 0/0.4)");
}

// A flat free cloth lying on the ground plane (every node moves alike, so the springs add nothing: the scheme has no error
// and the step is the exact scalar Coulomb recurrence below).
inline Tape run_flat(Real mu, int steps, const Vec3& gravity, const Vec3& v0, const Vec3& plane_velocity,
                     ContactModel model = ContactModel::RelativeVelocity, Real y0 = 0.0)
{
    AppConfig cfg = ncp_check::cloth_on_sphere(make_plane(Vec3(0, 0, 0), Vec3(0, 1, 0), plane_velocity));
    cfg.cloth.pin_mode = PinMode::NONE;
    cfg.cloth.origin   = Vec3(0.0, y0, 0.0);
    cfg.solver.gravity = gravity;
    cfg.solver.n_iters = 50;
    cfg.world.contact.friction = mu;
    cfg.world.contact.model    = model;
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

inline void check_plane(const std::string& name, Real mu, const Vec3& gravity, const Vec3& v0, Real u, ContactModel model)
{
    const int steps = 30;
    const Real h = 1.0 / 60;
    const Tape tape = run_flat(mu, steps, gravity, v0, Vec3(u, 0, 0), model);
    Real err = 0.0, off_axis = 0.0;
    for (int n : { 1, 5, 13, 30 })
    {
        const PointsX& v = tape.velocities[n];
        err = std::max(err, (v.col(0).array() - expected_vx(mu, n, gravity.x(), -gravity.y(), v0.x(), u, h)).abs().maxCoeff());
        off_axis = std::max<Real>(off_axis, std::max(v.col(1).cwiseAbs().maxCoeff(), v.col(2).cwiseAbs().maxCoeff()));
    }
    std::printf("  [info] %s: vx(1 s/2) = %.6f, |vx - exact| %.2e, off-axis speed %.2e\n", name.c_str(), tape.velocities[steps].col(0).mean(), err, off_axis);
    check(err < 1e-9 && off_axis < 1e-9, name);
}

inline void check_planes()
{
    std::printf("== cloth on a plane against the exact Coulomb recurrence\n");
    const Vec3 g0(0, -9.81, 0);
    const Real th = 30.0 * 3.14159265358979323846 / 180.0; // gravity of a 30 degree slope, tan = 0.577
    const Vec3 g_slope(9.81 * std::sin(th), -9.81 * std::cos(th), 0);
    for (ContactModel model : { ContactModel::RelativeVelocity, ContactModel::Gap })
    {
        const std::string tag = model == ContactModel::Gap ? " [gap]" : " [rel-vel]";
        check_plane("mu 0: a sliding cloth keeps its speed" + tag, 0.0, g0, Vec3(1, 0, 0), 0.0, model);
        check_plane("mu 0.5: a sliding cloth stops after mu g t = v0 and stays stopped" + tag, 0.5, g0, Vec3(1, 0, 0), 0.0, model);
        check_plane("slope 30 deg, mu 0.3 < tan: slides with a = g (sin - mu cos)" + tag, 0.3, g_slope, Vec3::Zero(), 0.0, model);
        check_plane("slope 30 deg, mu 0.7 > tan: sticks" + tag, 0.7, g_slope, Vec3::Zero(), 0.0, model);
        check_plane("plane moving at 1 m/s, mu 1: the cloth is dragged to the plane speed" + tag, 1.0, g0, Vec3::Zero(), 1.0, model);
    }
}

// Gap model, a node that starts above the plane (phi > 0) and lands: the normal impulse is the arrest term (v+_n = -phi/h), and
// friction is charged against it, so the motion after the landing step is exact in closed form (rigid cloth again). Scalar reference below.
inline void check_gap_landing()
{
    std::printf("== gap model: a node landing from phi > 0 (the speculative contact the rel-velocity model cannot see)\n");
    const Real h = 1.0 / 60, g = -9.81, mu = 0.5, y0 = 0.02;
    const int steps = 30;
    const Tape tape = run_flat(mu, steps, Vec3(0, g, 0), Vec3(1, 0, 0), Vec3::Zero(), ContactModel::Gap, y0);
    Real y = y0, vy = 0.0, vx = 1.0, err = 0.0;
    int landing = -1;
    for (int n = 1; n <= steps; ++n)
    {
        const Real wy = vy + h * g;               // free vertical velocity
        const Real dn = wy + y / h;               // d_n / m with the gap term
        if (dn < 0.0)
        {
            if (landing < 0) landing = n;
            const Real cap = mu * (-dn);          // mu r_n / m
            vy = -y / h;
            vx = std::abs(vx) <= cap ? 0.0 : vx - (vx > 0 ? cap : -cap);
        }
        else vy = wy;
        y += h * vy;
        err = std::max({ err, (tape.velocities[n].col(0).array() - vx).abs().maxCoeff(), (tape.velocities[n].col(1).array() - vy).abs().maxCoeff(),
                         (tape.positions[n].col(1).array() - y).abs().maxCoeff() });
    }
    std::printf("  [info] lands at step %d, final vx %.6f, max |velocity, height - exact| %.2e\n", landing, vx, err);
    check(landing > 1 && err < 1e-9, "gap model, mu 0.5: landing from phi > 0 (arrest impulse, friction budget, rest on the surface) matches the scalar reference");
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
    check(*c.method_error() == '\0', "friction > 0, forward only, Gap model: ok");
    c.run_mode = RunMode::GuessTargetBackward;
    check(*c.method_error() != '\0', "friction > 0, Gap model, with backward: error");

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
    friction_check::check_gap_landing();
    friction_check::check_wiring();
    std::printf(ncp_check::failures ? "friction-check: %d FAILED\n" : "friction-check: all ok\n", ncp_check::failures);
    return ncp_check::failures ? 1 : 0;
}
