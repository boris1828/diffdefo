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

// coulomb_dr_dd / coulomb_dr_dn_pullback / coulomb_dr_dmu against central differences of impulse_coulomb, and the
// relative-velocity chain (dr/df = R, dr/dv_c = -m R). Inputs are kept 5% of |d| away from every branch seam: the
// derivative jumps there, so a difference quotient across one would be meaningless.
inline void check_derivatives(std::mt19937& g)
{
    using S = CoulombState;
    std::printf("== Coulomb rule derivatives against central differences (eps 1e-6)\n");
    const Real eps = 1e-6;
    auto seam_free = [](const Vec3& d, const Vec3& n, Real mu)
    {
        const Real d_n = d.dot(n), s = (d - d_n * n).norm(), sc = 0.05 * d.norm();
        return std::abs(d_n) > sc && s > sc && (d_n >= 0.0 || std::abs(s - mu * (-d_n)) > sc);
    };
    auto unit_perp = [](const Vec3& n) { return (std::abs(n.x()) < 0.9 ? Vec3::UnitX() : Vec3::UnitY()).cross(n).normalized(); };

    int count[3] = { 0, 0, 0 };
    Real err_R = 0, err_n = 0, err_mu = 0, err_f = 0, err_vc = 0, off_plane = 0;
    bool exact = true;
    for (int i = 0; i < 6000; ++i)
    {
        const Vec3 n  = rnd(g).normalized(), z = rnd(g), d = rnd(g) * 3.0;
        const Real mu = 1.2 * std::abs(rnd(g).x());
        if (!seam_free(d, n, mu)) continue;
        const S state = coulomb_state(d, n, mu);
        ++count[(int)state];

        const Mat3 R  = coulomb_dr_dd(d, n, mu);
        const Vec3 pn = coulomb_dr_dn_pullback(d, n, mu, z);
        const Vec3 pm = coulomb_dr_dmu(d, n, mu);
        if (state == S::TakeOff) exact = exact && R.isZero(0.0);
        if (state == S::Stick)   exact = exact && R == -Mat3::Identity();
        if (state != S::Slip)    exact = exact && pn.isZero(0.0) && pm.isZero(0.0);

        // dr/dd, column by column
        Mat3 R_fd;
        for (int j = 0; j < 3; ++j)
        {
            const Vec3 e = eps * Vec3::Unit(j);
            R_fd.col(j) = (impulse_coulomb(d + e, n, mu) - impulse_coulomb(d - e, n, mu)) / (2 * eps);
        }
        err_R = std::max(err_R, (R - R_fd).cwiseAbs().maxCoeff());

        // (dr/dn)^T z: n moves on the unit sphere, so test the two tangent directions b, z . dr/dn[b] = b . pullback
        const Vec3 b1 = unit_perp(n), b2 = n.cross(b1);
        for (const Vec3& b : { b1, b2 })
        {
            const Vec3 dr = (impulse_coulomb(d, (n + eps * b).normalized(), mu) - impulse_coulomb(d, (n - eps * b).normalized(), mu)) / (2 * eps);
            err_n = std::max(err_n, std::abs(z.dot(dr) - b.dot(pn)));
        }
        off_plane = std::max(off_plane, std::abs(n.dot(pn)));

        // dr/dmu (r is linear in mu in slip, constant otherwise)
        err_mu = std::max(err_mu, (pm - (impulse_coulomb(d, n, mu + eps) - impulse_coulomb(d, n, mu - eps)) / (2 * eps)).cwiseAbs().maxCoeff());

        // relative-velocity model chain: r(f, v_c) = rule(f - m v_c); pick v_c, then f so that d is the draw above
        const Real m = 0.5 + std::abs(rnd(g).x());
        const Vec3 v_c = rnd(g), f = d + m * v_c;
        auto r_rel = [&](const Vec3& ff, const Vec3& vv) { return impulse_coulomb(d_relvel(ff, m, vv), n, mu); };
        for (int j = 0; j < 3; ++j)
        {
            const Vec3 e = eps * Vec3::Unit(j);
            err_f  = std::max(err_f,  (R.col(j)      - (r_rel(f + e, v_c) - r_rel(f - e, v_c)) / (2 * eps)).cwiseAbs().maxCoeff());
            err_vc = std::max(err_vc, (-m * R.col(j) - (r_rel(f, v_c + e) - r_rel(f, v_c - e)) / (2 * eps)).cwiseAbs().maxCoeff());
        }
    }
    std::printf("  [info] seam-free draws: %d take-off, %d stick, %d slip; max |analytic - FD|: R %.1e, n %.1e, mu %.1e, f %.1e, v_c %.1e\n",
                count[0], count[1], count[2], err_R, err_n, err_mu, err_f, err_vc);
    const Real tol = 1e-7;
    check(count[0] >= 100 && count[1] >= 100 && count[2] >= 100, "every branch is exercised (>= 100 draws each)");
    check(exact, "take-off: R = 0, stick: R = -I exactly; dr/dn and dr/dmu vanish outside slip");
    check(err_R  < tol, "dr/dd (R, incl. the non-symmetric mu t n^T) matches central differences");
    check(err_n  < tol && off_plane < 1e-14, "(dr/dn)^T z matches central differences on the unit sphere, and is tangent to n");
    check(err_mu < tol, "dr/dmu matches central differences");
    check(err_f < tol && err_vc < tol, "relative-velocity chain: dr/df = R, dr/dv_c = -m R");

    // mu = 0 in slip (d_T != 0) is the frictionless derivative: R = -n n^T, pullback = -z_n d_T - d_n z_perp (normal_correction's term)
    Real lim_R = 0, lim_n = 0;
    for (int i = 0; i < 1000; ++i)
    {
        const Vec3 n = rnd(g).normalized(), z = rnd(g), d = rnd(g) * 3.0;
        const Real d_n = d.dot(n);
        if (coulomb_state(d, n, 0.0) != S::Slip) continue;
        const Vec3 d_t = d - d_n * n, z_perp = z - n * n.dot(z);
        lim_R = std::max(lim_R, (coulomb_dr_dd(d, n, 0.0) + n * n.transpose()).cwiseAbs().maxCoeff());
        lim_n = std::max(lim_n, (coulomb_dr_dn_pullback(d, n, 0.0, z) - (-n.dot(z) * d_t - d_n * z_perp)).cwiseAbs().maxCoeff());
    }
    check(lim_R < 1e-15 && lim_n < 1e-13, "mu = 0 slip: R = -n n^T and the n pullback is the frictionless -z_n d_T - d_n z_perp");
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
    check_derivatives(g);

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
inline AppConfig flat_cfg(Real mu, const Vec3& gravity, const Vec3& plane_velocity, ContactModel model, Real y0)
{
    AppConfig cfg = ncp_check::cloth_on_sphere(make_plane(Vec3(0, 0, 0), Vec3(0, 1, 0), plane_velocity));
    cfg.cloth.pin_mode = PinMode::NONE;
    cfg.cloth.origin   = Vec3(0.0, y0, 0.0);
    cfg.solver.gravity = gravity;
    cfg.solver.n_iters = 50;
    cfg.world.contact.friction = mu;
    cfg.world.contact.model    = model;
    return cfg;
}

inline Tape run_flat(Real mu, int steps, const Vec3& gravity, const Vec3& v0, const Vec3& plane_velocity,
                     ContactModel model = ContactModel::RelativeVelocity, Real y0 = 0.0)
{
    const AppConfig cfg = flat_cfg(mu, gravity, plane_velocity, model, y0);
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

// The backward's precompute on recorded friction runs: Contact::d must be the model's d at the converged v+ (d . n = d_n,
// active = pressing), and coulomb_state(d) must give the branch the scalar recurrence says the step is in. This is also the
// forward / backward branch agreement of the plan's subtlety 3, on scenes clear of the stick / slip seam.
inline void check_precompute()
{
    std::printf("== backward precompute on friction runs: Contact::d and its Coulomb branch\n");
    const Real th = 30.0 * 3.14159265358979323846 / 180.0;
    const Vec3 g0(0, -9.81, 0), g_slope(9.81 * std::sin(th), -9.81 * std::cos(th), 0);
    struct Scene { const char* name; Real mu; Vec3 g; Real v0, u; };
    const Scene scenes[] = {
        { "sliding cloth stops (slip, then stick)", 0.5, g0,      1.0, 0.0 },
        { "slope 30 deg, mu 0.3 (slip)",            0.3, g_slope, 0.0, 0.0 },
        { "slope 30 deg, mu 0.7 (stick)",           0.7, g_slope, 0.0, 0.0 },
        { "dragged by a 1 m/s plane (slip, then stick)", 1.0, g0, 0.0, 1.0 },
    };
    const int steps = 30;
    for (ContactModel model : { ContactModel::RelativeVelocity, ContactModel::Gap })
        for (const Scene& sc : scenes)
        {
            const Vec3 plane_v(sc.u, 0, 0);
            const Tape tape = run_flat(sc.mu, steps, sc.g, Vec3(sc.v0, 0, 0), plane_v, model);
            const AppConfig cfg = flat_cfg(sc.mu, sc.g, plane_v, model, 0.0);
            const Real h = 1.0 / (cfg.solver.FPS * cfg.solver.frame_substeps);
            Object obj = build_run_cloth(cfg, Stiffness::uniform(cfg.stiffness));
            construct_pd_lhs(obj, h);
            const Index dofs = obj.num_dofs();
            const bool gap = model == ContactModel::Gap;

            // the scalar recurrence of expected_vx, read per step: slip iff the relative tangential trial speed |w| exceeds the cap
            const Real cap = sc.mu * h * (-sc.g.y());
            Real v = sc.v0;
            int n_checked = 0, n_slip = 0, n_stick = 0;
            bool ok = true;
            Real err_dn = 0.0;
            for (int t = 1; t <= steps; ++t)
            {
                const Real w = (v - sc.u) + h * sc.g.x();
                v = sc.u + (std::abs(w) <= cap ? 0.0 : w - (w > 0 ? cap : -cap));
                if (std::abs(std::abs(w) - cap) < 1e-6) continue; // on the seam: either branch is right
                const CoulombState want = std::abs(w) > cap ? CoulombState::Slip : CoulombState::Stick;

                Contacts contacts = tape.contacts[t - 1];
                precompute_contacts_local_derivative(contacts, obj,
                    Eigen::Map<const Positions>(tape.positions[t].data(), dofs), Eigen::Map<const Velocities>(tape.velocities[t].data(), dofs),
                    Eigen::Map<const Positions>(tape.positions[t - 1].data(), dofs), Eigen::Map<const Velocities>(tape.velocities[t - 1].data(), dofs),
                    sc.g, h, gap);
                ok = ok && !contacts.empty();
                for (const Contact& c : contacts)
                {
                    const CoulombState s = coulomb_state(c.d, c.normal, sc.mu);
                    ok = ok && s == want && c.active == (s != CoulombState::TakeOff);
                    err_dn = std::max(err_dn, std::abs(c.d.dot(c.normal) - c.d_n) / (1.0 + std::abs(c.d_n)));
                }
                ++n_checked;
                (want == CoulombState::Slip ? n_slip : n_stick) += 1;
            }
            std::printf("  [info] %s [%s]: %d steps checked (%d slip, %d stick), max |d.n - d_n| %.1e\n",
                        sc.name, gap ? "gap" : "rel-vel", n_checked, n_slip, n_stick, err_dn);
            check(ok && n_checked >= steps - 2 && err_dn < 1e-12,
                  std::string("precompute: every contact's coulomb_state(d) is the expected branch, d . n = d_n: ") + sc.name + (gap ? " [gap]" : " [rel-vel]"));
        }
}

// split_by_contact with friction: free + con = z, con = -R^T z per active contact, untouched elsewhere; mu = 0 the old projector exactly.
inline void check_split()
{
    std::printf("== split_by_contact with friction: z_free = (I + R^T) z, z_con = -R^T z\n");
    std::mt19937 g(5);
    const Index n_part = 40;
    bool sums = true, formula = true, passive = true, frictionless = true;
    for (int rep = 0; rep < 50; ++rep)
    {
        const RealVecX z = RealVecX::Random(3 * n_part);
        Contacts contacts;
        for (Index p = 0; p < n_part; p += 2) // every other particle in contact, some of them inactive
        {
            Contact c; c.particle = (ParticleId)p; c.normal = rnd(g).normalized(); c.d = rnd(g) * 3.0; c.active = (p % 6 != 0);
            contacts.push_back(c);
        }
        const Real mu = 0.6;
        const ContactSplit s = split_by_contact(contacts, z, mu);
        sums = sums && (s.z_perp + s.Pz - z).cwiseAbs().maxCoeff() < 1e-14;
        std::vector<bool> touched(n_part, false);
        for (const Contact& c : contacts)
        {
            touched[c.particle] = c.active;
            const Vec3 zi = z.segment<3>(3 * c.particle), con = s.Pz.segment<3>(3 * c.particle);
            formula = formula && (c.active ? (con + coulomb_dr_dd(c.d, c.normal, mu).transpose() * zi).norm() < 1e-14 : con.isZero(0.0));
        }
        for (Index p = 0; p < n_part; ++p)
            if (!touched[p]) passive = passive && s.Pz.segment<3>(3 * p).isZero(0.0) && s.z_perp.segment<3>(3 * p) == z.segment<3>(3 * p);

        // mu = 0: exactly the old projector P = n n^T
        const ContactSplit f = split_by_contact(contacts, z, 0.0);
        for (const Contact& c : contacts)
        {
            if (!c.active) continue;
            const Vec3 zi = z.segment<3>(3 * c.particle), pn = c.normal * c.normal.dot(zi);
            frictionless = frictionless && f.Pz.segment<3>(3 * c.particle) == pn && f.z_perp.segment<3>(3 * c.particle) == zi - pn;
        }
    }
    check(sums, "z_free + z_con = z");
    check(formula, "z_con = -R^T z at active contacts (R = coulomb_dr_dd(c.d)), 0 at inactive ones");
    check(passive, "particles without an active contact pass through untouched");
    check(frictionless, "mu = 0: the old projector split, bit for bit");
}

// An OBJ from animation/ with every vertex scaled by s, as text for load_trimesh_obj(std::istream&).
inline std::string scaled_obj(const std::string& file, Real s)
{
    std::ifstream in(animation_path(file));
    ASSERT(in, "friction-check needs animation/" << file);
    std::ostringstream out;
    out << std::setprecision(17);
    std::string line;
    while (std::getline(in, line))
    {
        if (line.rfind("v ", 0) == 0)
        {
            std::istringstream ls(line.substr(2));
            Real x, y, z;
            ls >> x >> y >> z;
            out << "v " << s * x << ' ' << s * y << ' ' << s * z << "\n";
        }
        else out << line << "\n";
    }
    return out.str();
}

// The friction backward end to end: backward_pd_contact over a window of steps (from a warmed-up state in contact) against central
// differences of the forward, for dphi/dx0, dphi/dv0, dphi/dk, dphi/dmu and (planes) dphi/d(collider velocity). Relative-velocity
// model. Each scene also runs the backward with one ingredient removed, which must then fail: friction off (all scenes), the curvature
// term (static sphere, inv_r = 0) and the surface-velocity gradient (rotating sphere, dvc_dx = 0). Loss L = u . x_T.
inline void check_backward_gradients()
{
    std::printf("== friction backward (backward_pd_contact) vs central differences of the forward, relative-velocity model\n");
    const Vec3 g0(0, -9.81, 0);
    struct Scene
    {
        std::string name; AppConfig cfg; Vec3 v0; int warmup, steps; bool collider_velocity_fd; std::string ablate; bool need_stick;
        int iters = 3000;      // forward and adjoint iterations per step (converged: the FD then differentiates the true map)
        bool jitter = true;    // random 0.1 m/s per node at the start
        std::string mesh_text; // OBJ text of world.trimeshes[0] for a Mesh collider (mesh_id 0)
    };
    std::vector<Scene> scenes;
    scenes.push_back({ "plane, sliding stops (slip -> stick)", flat_cfg(0.5, g0, Vec3::Zero(), ContactModel::RelativeVelocity, 0.0), Vec3(1, 0, 0), 8, 8, true, "", true });
    scenes.push_back({ "plane moving at 1 m/s drags (slip -> stick)", flat_cfg(1.0, g0, Vec3(1, 0, 0), ContactModel::RelativeVelocity, 0.0), Vec3::Zero(), 2, 8, true, "", true });
    {
        const Collider sphere = make_sphere(Vec3(1.0, 0.35, 1.0), 0.65); // its top touches the cloth (y = 0.95) at once
        AppConfig c = ncp_check::cloth_on_sphere(sphere);
        c.cloth.pin_mode = PinMode::NONE;
        c.world.contact.friction = 0.3; // atan 0.3 = 17 deg: the flat top sticks, the flanks slip
        scenes.push_back({ "static sphere, mu 0.3 (curvature)", c, Vec3::Zero(), 15, 4, false, "inv_r", false }); // all slip: a stuck node has no curvature term (r = -d)
        Collider spin = sphere;
        spin.rotation_axis = RotationAxis::Z; spin.rotation_origin = sphere.sphere_center; spin.omega = 2.0; // the top moves at 1.3 m/s along -x
        AppConfig cr = c;
        cr.world.colliders = { spin };
        cr.world.contact.friction = 0.5;
        scenes.push_back({ "rotating sphere, mu 0.5 (surface velocity G)", cr, Vec3::Zero(), 15, 4, false, "dvc_dx", true });
    }
    {
        // Drapes: an 8 x 8 square (1.6 m, y = 0.95, x and z in [0, 1.6]) pinned at the two corners of its z = 0 edge, at rest 5 cm above
        // the collider's top, falls onto it for 2 s (120 steps), mu 0.4. Capsules lie along x under the middle; the meshes are the
        // animation/ icosphere (r 0.5, 80 triangles) and the capsule (scaled to r 0.3, half-length 0.3, turned onto x), so the
        // analytic and mesh capsules have the same shape.
        auto drape = [](const std::string& name, const Collider& col, const std::string& mesh_text, const std::string& ablate)
        {
            AppConfig c = ncp_check::cloth_on_sphere(col);
            c.cloth.width = c.cloth.height = 8; c.cloth.size_x = c.cloth.size_y = 1.6;
            c.cloth.pin_mode = PinMode::CORNERS;
            c.world.contact.friction = 0.4;
            c.solver.anderson_m = 5;
            Scene s{ name, c, Vec3::Zero(), 0, 120, false, ablate, true };
            s.iters = 300; s.jitter = false; s.mesh_text = mesh_text;
            return s;
        };
        const Vec3 cap_c(0.8, 0.6, 0.8), sph_c(0.8, 0.4, 0.8);
        const Collider capsule = make_capsule(cap_c - Vec3(0.3, 0, 0), cap_c + Vec3(0.3, 0, 0), 0.3);
        Collider roller = capsule;
        roller.rotation_axis = RotationAxis::X; roller.rotation_origin = cap_c; roller.omega = 3.0; // its top moves at 0.9 m/s along -z, towards the pins
        Collider mesh_sphere;
        mesh_sphere.type = ColliderType::Mesh; mesh_sphere.mesh_id = 0; mesh_sphere.rigid_pose.position = sph_c;
        Collider mesh_capsule = mesh_sphere;
        mesh_capsule.rigid_pose.position = cap_c;
        mesh_capsule.rigid_pose.rotation = Eigen::Quaternion<Real>(Eigen::AngleAxis<Real>(-0.5 * 3.14159265358979323846, Vec3::UnitZ())); // file axis y -> x
        scenes.push_back(drape("drape on a capsule (curvature, axis)", capsule, "", "inv_r"));
        scenes.push_back(drape("drape on a capsule spinning about its axis (G)", roller, "", "dvc_dx"));
        scenes.push_back(drape("drape on a mesh sphere", mesh_sphere, scaled_obj("tri_sphere.obj", 1.0), ""));
        scenes.push_back(drape("drape on a mesh capsule", mesh_capsule, scaled_obj("tri_capsule.obj", 0.3), ""));
    }

    for (const Scene& sc : scenes)
    {
        const auto t_start = std::chrono::steady_clock::now();
        const int iters = sc.iters;
        const AppConfig& cfg = sc.cfg;
        auto make_world = [&](const AppConfig& c)
        {
            WorldSetup w = prepare_world(c);
            if (!sc.mesh_text.empty())
            {
                TriMesh m;
                std::istringstream in(sc.mesh_text);
                ASSERT(load_trimesh_obj(in, sc.name, m), "friction-check: the mesh of '" << sc.name << "' does not load");
                w.collision.trimeshes = { std::move(m) };
            }
            return w;
        };
        const Vec3 g = cfg.solver.gravity;
        const Real h = 1.0 / (cfg.solver.FPS * cfg.solver.frame_substeps);
        const Real k0 = cfg.stiffness, mu0 = cfg.world.contact.friction;
        const Vec3 vcol0 = cfg.world.colliders[0].velocity;
        auto with = [&](Real k, Real mu, const Vec3& v_col)
        {
            AppConfig c = cfg;
            c.stiffness = k; c.world.contact.friction = mu; c.world.colliders[0].velocity = v_col;
            return c;
        };

        RealVecX xs, vs; // the warmed-up state the window starts from
        {
            WorldSetup w = make_world(cfg);
            w.collision.collect_backward = false;
            Object o = build_run_cloth(cfg, Stiffness::uniform(k0));
            construct_pd_lhs(o, h);
            // plus a random 0.1 m/s per node: a uniform cloth never stretches, and its dphi/dk would be 0 (an empty check)
            for (Index i = 0; i < o.num_particles(); ++i) o.v.segment<3>(3 * i) = sc.v0 + (sc.jitter ? 0.1 : 0.0) * Vec3::Random();
            Tape t;
            if (sc.warmup > 0) pd_contact(o, w.collision, h, g, iters, sc.warmup, 1, t, "friction_check", false, false, nullptr, -1.0);
            xs = o.x; vs = o.v;
        }
        const Index dofs = xs.size();
        const RealVecX u = RealVecX::Random(dofs);
        const Real L_scale = u.norm() * xs.norm();

        // The loss of a perturbed run, and its contact structure: per step, which particles touch which collider triangle in which Coulomb
        // branch. The forward is only piecewise smooth (a mesh normal jumps between triangles, a branch switch is a kink), so a central
        // difference is only taken where the + and - runs have the same structure (see `central`).
        struct Probe { Real loss; std::vector<std::uint64_t> structure; };
        auto probe = [&](const RealVecX& x, const RealVecX& v, Real k, Real mu, const Vec3& v_col)
        {
            const AppConfig c = with(k, mu, v_col);
            WorldSetup w = make_world(c);
            w.collision.collect_backward = false;
            Object o = build_run_cloth(c, Stiffness::uniform(k));
            construct_pd_lhs(o, h);
            o.x = x; o.v = v;
            Tape t;
            pd_contact(o, w.collision, h, g, iters, sc.steps, 1, t, "friction_check", false, false, nullptr, -1.0, false, sc.warmup);
            Probe p{ u.dot(o.x), {} };
            for (int s = 1; s <= sc.steps; ++s)
            {
                Contacts cs = t.contacts[s - 1];
                precompute_contacts_local_derivative(cs, o,
                    Eigen::Map<const Positions>(t.positions[s].data(), dofs), Eigen::Map<const Velocities>(t.velocities[s].data(), dofs),
                    Eigen::Map<const Positions>(t.positions[s - 1].data(), dofs), Eigen::Map<const Velocities>(t.velocities[s - 1].data(), dofs),
                    g, h, false);
                for (const Contact& cn : cs)
                {
                    const int state = cn.active ? (int)coulomb_state(cn.d, cn.normal, mu) : (int)CoulombState::TakeOff;
                    p.structure.push_back((std::uint64_t(s) << 40) ^ (std::uint64_t(cn.particle) << 20) ^ (std::uint64_t(cn.collider_id + 1) << 16)
                                          ^ (std::uint64_t(cn.triangle_id + 1) << 2) ^ std::uint64_t(state));
                }
            }
            return p;
        };
        // Central difference of the loss along a parameter offset s -> at(s), from step eps down by 10x until the + and - runs share their
        // contact structure (at most 4 tries); `eps` = 0 when no step did (the quotient would straddle a jump: not testable there).
        struct FD { Real value; Real eps; };
        auto central = [&](auto&& at, Real eps)
        {
            for (int k = 0; k < 4; ++k, eps *= 0.1)
            {
                const Probe p = at(eps), m = at(-eps);
                if (p.structure == m.structure) return FD{ (p.loss - m.loss) / (2 * eps), eps };
            }
            return FD{ 0.0, 0.0 };
        };

        // analytic: the recorded window and the real backward pass
        const AppConfig c0 = with(k0, mu0, vcol0);
        WorldSetup w0 = make_world(c0); // collect_backward stays on
        Object o0 = build_run_cloth(c0, Stiffness::uniform(k0));
        construct_pd_lhs(o0, h);
        o0.x = xs; o0.v = vs;
        Tape tape;
        pd_contact(o0, w0.collision, h, g, iters, sc.steps, 1, tape, "friction_check", false, false, nullptr, 1e-3, false, sc.warmup); // count > 1 mm deep after the solve
        Loss loss;
        loss.dloss_dx.assign(sc.steps + 1, RealVecX::Zero(dofs));
        loss.dloss_dx.back() = u;
        auto backward = [&](Tape& tp, Real mu_backward)
        {
            w0.collision.contact.friction = mu_backward;
            BackwardGradContact gr = backward_pd_contact(o0, w0.collision, tp, loss, g, iters, h);
            w0.collision.contact.friction = mu0;
            return gr;
        };
        const BackwardGradContact gr = backward(tape, mu0);

        // which branches the window contains (the precompute at each recorded step)
        int n_stick = 0, n_slip = 0, n_off = 0, first_touch = -1, n_face = 0, n_edge = 0, n_vertex = 0;
        std::vector<bool> touched(o0.num_particles(), false);
        for (int t = 1; t <= sc.steps; ++t)
        {
            Contacts cs = tape.contacts[t - 1];
            precompute_contacts_local_derivative(cs, o0,
                Eigen::Map<const Positions>(tape.positions[t].data(), dofs), Eigen::Map<const Velocities>(tape.velocities[t].data(), dofs),
                Eigen::Map<const Positions>(tape.positions[t - 1].data(), dofs), Eigen::Map<const Velocities>(tape.velocities[t - 1].data(), dofs),
                g, h, false);
            for (const Contact& c : cs)
            {
                const CoulombState s = c.active ? coulomb_state(c.d, c.normal, mu0) : CoulombState::TakeOff;
                (s == CoulombState::Stick ? n_stick : s == CoulombState::Slip ? n_slip : n_off) += 1;
                if (s == CoulombState::TakeOff) continue;
                if (first_touch < 0) first_touch = t;
                touched[c.particle] = true;
                if (cfg.world.colliders[0].type == ColliderType::Mesh)
                    (c.feature == ContactFeature::Face ? n_face : c.feature == ContactFeature::Edge ? n_edge : n_vertex) += 1;
            }
        }
        Real est = 0.0;
        for (const StepResidual& r : tape.forward_residual) est = std::max(est, r.est_rel);

        // central differences
        const Real eps = 1e-6;
        std::vector<std::pair<RealVecX, RealVecX>> dirs; // (e_x, e_v)
        for (int k = 0; k < 2; ++k) dirs.push_back({ RealVecX::Random(dofs), RealVecX::Random(dofs) });
        std::vector<std::pair<Real, Real>> fd_xv;
        std::vector<Real> fd_eps; // the step each quotient was taken with (0: no step kept the structure)
        for (const auto& [ex, ev] : dirs)
        {
            const FD fx = central([&](Real e) { return probe(xs + e * ex, vs, k0, mu0, vcol0); }, eps);
            const FD fv = central([&](Real e) { return probe(xs, vs + e * ev, k0, mu0, vcol0); }, eps);
            fd_xv.push_back({ fx.value, fv.value });
            fd_eps.push_back(fx.eps); fd_eps.push_back(fv.eps);
        }
        // dphi/dk is small on the planes: a wider step keeps FD rounding well under the tolerance (truncation ~1e-8)
        const FD fk  = central([&](Real e) { return probe(xs, vs, k0 + e, mu0, vcol0); }, 1e-4 * k0);
        const FD fmu = central([&](Real e) { return probe(xs, vs, k0, mu0 + e, vcol0); }, eps);
        const Real fd_k = fk.value, fd_mu = fmu.value;
        fd_eps.push_back(fk.eps); fd_eps.push_back(fmu.eps);
        Real fd_vc = 0.0;
        if (sc.collider_velocity_fd)
        {
            const FD fc = central([&](Real e) { return probe(xs, vs, k0, mu0, vcol0 + e * Vec3::UnitX()); }, eps);
            fd_vc = fc.value;
            fd_eps.push_back(fc.eps);
        }
        const bool fd_testable = std::none_of(fd_eps.begin(), fd_eps.end(), [](Real e) { return e == 0.0; });
        Real eps_min = 1.0, eps_max = 0.0;
        for (Real e : fd_eps) if (e > 0.0) { eps_min = std::min(eps_min, e); eps_max = std::max(eps_max, e); }

        auto rel = [&](Real an, Real fd) { return std::abs(an - fd) / std::max(std::abs(fd), 1e-6 * L_scale); };
        auto err_xv = [&](const BackwardGradContact& b)
        {
            Real e = 0.0;
            for (size_t k = 0; k < dirs.size(); ++k)
            {
                const Real scale = std::abs(fd_xv[k].first) + std::abs(fd_xv[k].second); // a stuck cloth's x_T barely depends on v0
                e = std::max({ e, std::abs(dirs[k].first.dot(b.dphi_dx) - fd_xv[k].first) / scale,
                                  std::abs(dirs[k].second.dot(b.dphi_dv) - fd_xv[k].second) / scale });
            }
            return e;
        };
        const Real e_xv = err_xv(gr), e_k = rel(gr.dphi_dk.sum(), fd_k), e_mu = rel(gr.dphi_dmu, fd_mu);
        Real e_vc = 0.0;
        if (sc.collider_velocity_fd)
        {
            Real an = 0.0;
            for (const Vec3& dv : gr.colliders[0].d_velocity) an += dv.x();
            e_vc = rel(an, fd_vc);
        }

        // ablations: the same backward with one ingredient removed must be clearly wrong
        const Real e_nofric = err_xv(backward(tape, 0.0));
        Real e_abl = 0.0;
        if (!sc.ablate.empty())
        {
            Tape t2 = tape;
            for (Contacts& cs : t2.contacts)
                for (Contact& c : cs) { if (sc.ablate == "inv_r") c.inv_r = 0.0; else c.dvc_dx.setZero(); }
            e_abl = err_xv(backward(t2, mu0));
        }

        Real adj = 0.0;
        for (Real r : gr.residual) adj = std::max(adj, r);
        const int n_touched = (int)std::count(touched.begin(), touched.end(), true);
        const bool is_mesh = cfg.world.colliders[0].type == ColliderType::Mesh;
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        std::printf("  [info] %s: %d stick / %d slip / %d off contact-steps, forward est_rel %.0e, adjoint residual %.0e\n",
                    sc.name.c_str(), n_stick, n_slip, n_off, est, adj);
        std::printf("         first contact at step %d of %d, %d of %d particles touched", first_touch, sc.steps, n_touched, (int)touched.size());
        if (is_mesh) std::printf(", mesh features face %d / edge %d / vertex %d", n_face, n_edge, n_vertex);
        int deep = 0, deep_max = 0;
        for (int n : tape.unresolved_contacts) { deep += n; deep_max = std::max(deep_max, n); }
        std::printf(", > 1 mm deep after the solve: %d particle-steps (max %d in a step), %.1f s\n", deep, deep_max, secs);
        std::printf("         rel err x0/v0 %.1e, k %.1e (fd %.3e), mu %.1e (fd %.3e), collider v %.1e; friction off %.1e, ablation (%s) %.1e\n",
                    e_xv, e_k, fd_k, e_mu, fd_mu, e_vc, e_nofric, sc.ablate.empty() ? "none" : sc.ablate.c_str(), e_abl);
        if (fd_testable) std::printf("         FD steps kept the contact structure at eps %.0e .. %.0e\n", eps_min, eps_max);
        else             std::printf("         FD: some quotient changed the contact structure down to eps 1e-9 (straddles a jump)\n");
        const Real tol = 1e-5;
        check(n_slip > 0 && (!sc.need_stick || n_stick > 0), std::string(sc.need_stick ? "the window has stick and slip contacts: " : "the window has slip contacts: ") + sc.name);
        // a drape starts in the air: it must land and spread over the collider. (A relative-velocity contact is a particle inside the
        // collider; inside a convex mesh the closest boundary point is on a face, so edge / vertex contacts need the Gap model or a thickness.)
        if (sc.warmup == 0)
            check(first_touch > 1 && n_touched >= 8 && (!is_mesh || n_face > 0),
                  "the cloth falls onto the collider (first contact after step 1, >= 8 particles touched): " + sc.name);
        check(fd_testable && e_xv < tol && e_k < tol && e_mu < tol && e_vc < tol,
              "dphi/dx0, dv0, dk, dmu" + std::string(sc.collider_velocity_fd ? ", dv_collider" : "") + " match FD: " + sc.name);
        check(e_nofric > 1e-3 && (sc.ablate.empty() || e_abl > 1e-3),
              "the frictionless backward" + std::string(sc.ablate.empty() ? "" : " and the one without " + sc.ablate) + " fail on it: " + sc.name);
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

// The app's own path with friction, as run_app does it: a target run, a guess run, the MSE Loss against the target, backward_pd_contact,
// and fd_check_contact_stiffness (the playback screen's FD Check). The drape on a capsule, 1 s. dphi/dmu by hand with the same loss.
inline void check_app_pipeline()
{
    std::printf("== the app's path with friction: target, guess, MSE loss, backward_pd_contact, fd_check_contact_stiffness\n");
    const Vec3 cap_c(0.8, 0.6, 0.8);
    AppConfig cfg = ncp_check::cloth_on_sphere(make_capsule(cap_c - Vec3(0.3, 0, 0), cap_c + Vec3(0.3, 0, 0), 0.3));
    cfg.cloth.width = cfg.cloth.height = 8; cfg.cloth.size_x = cfg.cloth.size_y = 1.6;
    cfg.cloth.pin_mode = PinMode::CORNERS;
    cfg.world.contact.friction = 0.4;
    cfg.solver.anderson_m = 5;
    cfg.stiffness = 1.0; cfg.target_stiffness = 1.5;
    check(*cfg.method_error() == '\0' && cfg.backward_enabled(), "the drape config with friction runs Guess+Target+Backward");

    WorldSetup world = prepare_world(cfg);
    const Real h = 1.0 / (cfg.solver.FPS * cfg.solver.frame_substeps), mu = cfg.world.contact.friction;
    const int steps = 60, iters = 300;
    auto build = [&](Real k) { return build_run_cloth(cfg, Stiffness::uniform(k)); };
    auto run = [&](Real k, Tape& tape)
    {
        Object o = build(k);
        construct_pd_lhs(o, h);
        pd_contact(o, world.collision, h, cfg.solver.gravity, iters, steps, 1, tape, "friction_check", false, false);
        return o;
    };
    Tape target, guess;
    run(cfg.target_stiffness, target);
    Object obj = run(cfg.stiffness, guess);
    const Loss loss(guess, target, 1);
    const BackwardGradContact gr = backward_pd_contact(obj, world.collision, guess, loss, cfg.solver.gravity, iters, h);
    const FDCheckResult fk = fd_check_contact_stiffness(world.collision, build, target, cfg.stiffness, h, cfg.solver.gravity, iters, steps, 1, 1,
                                                        gr.dphi_dk.sum(), 1e-6);
    auto loss_at_mu = [&](Real m)
    {
        world.collision.contact.friction = m;
        Tape t;
        run(cfg.stiffness, t);
        world.collision.contact.friction = mu;
        return Loss(t, target, 1).total;
    };
    const Real e = 1e-6, fd_mu = (loss_at_mu(mu + e) - loss_at_mu(mu - e)) / (2 * e);
    const Real err_mu = std::abs(gr.dphi_dmu - fd_mu) / std::abs(fd_mu);
    std::printf("  [info] loss %.6e; dphi/dk %.9e vs fd %.9e (rel %.1e); dphi/dmu %.9e vs fd %.9e (rel %.1e)\n",
                loss.total, gr.dphi_dk.sum(), fk.fd, fk.rel_err, gr.dphi_dmu, fd_mu, err_mu);
    check(loss.total > 0.0 && fk.rel_err < 1e-5 && err_mu < 1e-5, "MSE loss vs a stiffer target: dphi/dk (the app's FD Check) and dphi/dmu match");
}

inline void check_wiring()
{
    std::printf("== config\n");
    AppConfig c;
    check(*c.method_error() == '\0', "friction 0: no method error");
    c.world.contact.friction = 0.4;
    check(*c.method_error() == '\0' && c.backward_enabled(), "friction > 0 with Guess+Target+Backward, Rel. Velocity: ok (the friction backward runs)");
    c.run_mode = RunMode::ForwardOnly;
    check(*c.method_error() == '\0', "friction > 0, forward only: ok");
    c.world.contact.model = ContactModel::Gap;
    check(*c.method_error() == '\0', "friction > 0, forward only, Gap model: ok");
    c.run_mode = RunMode::GuessTargetBackward;
    check(*c.method_error() != '\0', "friction > 0, Gap model, with backward: error (no Gap friction backward yet)");
    c.world.contact.friction = 0.0;
    check(*c.method_error() == '\0', "friction 0, Gap model, with backward: ok (the frictionless Gap backward)");

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
    friction_check::check_precompute();
    friction_check::check_split();
    friction_check::check_backward_gradients();
    friction_check::check_gap_landing();
    friction_check::check_app_pipeline();
    friction_check::check_wiring();
    std::printf(ncp_check::failures ? "friction-check: %d FAILED\n" : "friction-check: all ok\n", ncp_check::failures);
    return ncp_check::failures ? 1 : 0;
}
