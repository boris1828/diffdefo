#pragma once

// `diffpd --ncp-check`: self-checks of the NCP contact method (solver/ncp/). Standalone: it does not run the other checks.
// Not a standalone header: it is #included once, just above main() in diffpd.cpp, after diffpd_checks.h.

#include "solver/impulse/forward.h" // only to compare with the impulse method
#include "solver/ncp/forward.h"

namespace ncp_check
{
inline int failures = 0;

inline void check(bool ok, const std::string& what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// One contact (3 rows) through ncp_functions.
struct One { Real phi_n, om_n, e_n; Eigen::Vector2d phi_f; Real om_f, e_f; };
inline One one_contact(Real dn, const Eigen::Vector2d& df, Real ln, const Eigen::Vector2d& lf, Real rho_n, Real rho_f, Real mu, Real two_eps2 = 1e-12)
{
    RealVecX delta(3), lambda(3), rn(1), rf(1), phi, om, e;
    delta << dn, df(0), df(1);
    lambda << ln, lf(0), lf(1);
    rn << rho_n;
    rf << rho_f;
    ncp_functions(delta, lambda, rn, rf, mu, two_eps2, phi, om, e);
    return { phi(0), om(0), e(0), Eigen::Vector2d(phi(1), phi(2)), om(1), e(1) };
}

inline void check_ncp_functions()
{
    std::printf("== NCP functions\n");
    const Eigen::Vector2d z(0, 0);

    // normal row: phi = 0 iff 0 <= delta, 0 <= lambda, delta * lambda = 0 (up to the smoothing)
    check(std::abs(one_contact(0.01, z, 0.0, z, 0.1, 0.1, 0.0).phi_n) < 1e-9, "separating (delta > 0, lambda = 0): phi = 0");
    check(std::abs(one_contact(0.0, z, 1.0, z, 0.1, 0.1, 0.0).phi_n) < 1e-9, "pressing (delta = 0, lambda > 0): phi = 0");
    check(one_contact(0.01, z, 1.0, z, 0.1, 0.1, 0.0).phi_n > 1e-3, "delta > 0 and lambda > 0: phi > 0");
    check(one_contact(-0.01, z, 0.0, z, 0.1, 0.1, 0.0).phi_n < -1e-3, "penetrating, no force: phi < 0");

    // Omega = d phi / d delta and E = d phi / d lambda of the normal row against central differences
    int bad = 0;
    for (Real dn : { -0.02, -0.001, 0.0005, 0.03 })
        for (Real ln : { 0.0, 0.5, 4.0 })
        {
            const Real rho = 0.05, h = 1e-7;
            const One c = one_contact(dn, z, ln, z, rho, rho, 0.0);
            const Real d_om = (one_contact(dn + h, z, ln, z, rho, rho, 0.0).phi_n - one_contact(dn - h, z, ln, z, rho, rho, 0.0).phi_n) / (2 * h);
            const Real d_e  = (one_contact(dn, z, ln + h, z, rho, rho, 0.0).phi_n - one_contact(dn, z, ln - h, z, rho, rho, 0.0).phi_n) / (2 * h);
            if (std::abs(d_om - c.om_n) > 1e-6 || std::abs(d_e - c.e_n) > 1e-6) ++bad;
        }
    check(bad == 0, "normal omega and E match central differences of phi (12 points)");

    // friction rows, contact with lambda_n = 1, mu = 0.5, rho_f = 0.1
    const One off = one_contact(0.0, Eigen::Vector2d(0.01, 0), 0.0, Eigen::Vector2d(0.3, 0.1), 0.1, 0.1, 0.5);
    check(off.om_f == 0.0 && off.e_f == 1.0 && (off.phi_f - Eigen::Vector2d(0.3, 0.1)).norm() < 1e-15, "no normal force: friction off, phi_f = lambda_f");
    const One stick = one_contact(0.0, z, 1.0, Eigen::Vector2d(0.2, 0.1), 0.1, 0.1, 0.5);
    check(stick.om_f == 1.0 && stick.phi_f.norm() < 1e-6, "stick (no slip, |lambda_f| < mu lambda_n): phi_f = 0");
    const One slip = one_contact(0.0, Eigen::Vector2d(0.01, 0), 1.0, Eigen::Vector2d(-0.5, 0), 0.1, 0.1, 0.5);
    check(slip.phi_f.norm() < 1e-6, "slip (lambda_f = -mu lambda_n along the slip): phi_f = 0");
    const One wrong = one_contact(0.0, Eigen::Vector2d(0.01, 0), 1.0, Eigen::Vector2d(0.2, 0), 0.1, 0.1, 0.5);
    check(wrong.phi_f.norm() > 1e-3, "friction force along the slip (wrong sign): phi_f != 0");
    const One zero_mu = one_contact(0.0, Eigen::Vector2d(0.01, 0), 1.0, Eigen::Vector2d(0.2, 0), 0.1, 0.1, 0.0);
    check(zero_mu.om_f == 0.0 && zero_mu.e_f == 1.0, "mu = 0: friction off");
    const One pole = one_contact(0.0, z, 1.0, Eigen::Vector2d(1.0, 0), 0.1, 0.1, 0.5); // |lambda_f| = 2 mu lambda_n: the pole of E
    std::printf("  [info] |lambda_f| = 2 mu lambda_n, no slip: e_f = %.3g (the paper's eq. 57 has a pole here)\n", pole.e_f);
    check(std::isfinite(pole.e_f) && std::isfinite(pole.phi_f.norm()), "the pole of E_f gives no NaN/inf");

    // preconditioner
    const Index m = 3;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(3 * m, 3 * m);
    const Eigen::MatrixXd W = A * A.transpose();
    RealVecX rn, rf;
    const Real h = 1.0 / 60;
    complementarity_scaling(W, h, rn, rf);
    bool ok = true;
    for (Index j = 0; j < m; ++j)
        ok = ok && std::abs(rn(j) - h * h * W(j, j)) < 1e-15 && std::abs(rf(j) - 0.5 * h * h * (W(m + 2 * j, m + 2 * j) + W(m + 2 * j + 1, m + 2 * j + 1))) < 1e-15;
    check(ok, "complementarity_scaling: rho_n = h^2 W_jj, rho_f = h^2/2 (W_f1 + W_f2)");
}

// A 10x10 cloth, its pinned corners, a sphere at (1, 0.3, 1) r = 0.5: the scene of the other contact checks.
inline AppConfig cloth_on_sphere(const Collider& sphere)
{
    AppConfig cfg;
    cfg.cloth.cloth_type = ClothType::Square; cfg.cloth.mesh_gen = MeshGen::Quad;
    cfg.cloth.width = cfg.cloth.height = 10; cfg.cloth.size_x = cfg.cloth.size_y = 2.0;
    cfg.cloth.pin_mode = PinMode::CORNERS;
    cfg.per_type_stiffness = false;
    cfg.world.colliders = { sphere };
    cfg.world.animation_file = ""; cfg.world.waist_attach_enabled = false;
    return cfg;
}

inline void check_contacts_and_delassus()
{
    std::printf("== contact wrapper, Delassus, Schur solve\n");
    const Real h = 1.0 / 60;
    const Vec3 center(1.0, 0.3, 1.0), v_sphere(0.0, 0.3, 0.0);
    const Real radius = 0.5;

    AppConfig cfg = cloth_on_sphere(make_sphere(center, radius, v_sphere));
    WorldSetup world = prepare_world(cfg);
    world.collision.collect_backward = false;
    Object obj = build_run_cloth(cfg, Stiffness::uniform(cfg.stiffness));
    construct_pd_lhs(obj, h);

    // every free particle at y = 0.6: the ones near the sphere's axis are inside it
    Positions x_tilde = obj.x;
    for (Index i = 0; i < obj.num_particles(); ++i) x_tilde(3 * i + 1) = 0.6;
    const Positions x_minus = x_tilde - h * v_sphere.replicate(obj.num_particles(), 1); // the particles move with the sphere
    const StepClock clock{ h, 0, 1, false };

    NcpSettings ncp; ncp.margin = 0.0;
    const NcpContacts c0 = detect_ncp_contacts(world.collision, x_tilde, clock, 0, x_minus, h, ncp);
    check(c0.m() > 0, "contacts found (" + std::to_string(c0.m()) + ")");
    ncp.margin = 0.1;
    const NcpContacts c1 = detect_ncp_contacts(world.collision, x_tilde, clock, 0, x_minus, h, ncp);
    check(c1.m() > c0.m(), "a margin adds contacts (" + std::to_string(c1.m()) + " > " + std::to_string(c0.m()) + ")");

    const Index m = c1.m();
    Eigen::MatrixXd HHt = Eigen::MatrixXd(c1.H * c1.H.transpose()) - Eigen::MatrixXd::Identity(3 * m, 3 * m);
    check(HHt.cwiseAbs().maxCoeff() < 1e-12, "H rows are orthonormal (unit normal and tangents, one contact per particle)");

    // normal gap = signed distance to the sphere at its end-of-step pose (it has moved by v h)
    const Vec3 center_end = center + h * v_sphere;
    const RealVecX delta = c1.H * x_tilde - c1.d;
    Real gap_err = 0.0, max_gap = -1e9, max_gap0 = -1e9;
    for (Index j = 0; j < m; ++j)
    {
        const Vec3 p = x_tilde.segment<3>(3 * c1.contacts[j].particle);
        gap_err = std::max(gap_err, std::abs(delta(j) - ((p - center_end).norm() - radius)));
        max_gap = std::max(max_gap, delta(j));
    }
    const RealVecX delta0 = c0.H * x_tilde - c0.d;
    for (Index j = 0; j < c0.m(); ++j) max_gap0 = std::max(max_gap0, delta0(j));
    check(gap_err < 1e-12, "delta_n = |x - c| - r on the sphere");
    check(max_gap0 < 0.0 && max_gap < 0.1, "margin 0: every gap < 0; margin 0.1: every gap < 0.1");

    // a particle moving with the collider does not slip: delta_f = 0
    Positions x_follow = x_minus;
    bool v_ok = true;
    for (const Contact& c : c1.contacts)
    {
        v_ok = v_ok && (c.v_c - v_sphere).norm() < 1e-12;
        x_follow.segment<3>(3 * c.particle) += h * c.v_c;
    }
    const RealVecX slip = c1.H * x_follow - c1.d;
    check(v_ok, "contact velocity v_c = the sphere's velocity");
    check(slip.segment(m, 2 * m).cwiseAbs().maxCoeff() < 1e-12, "moving with the collider: tangential slip = 0");

    // Delassus against a dense inverse of L
    const Eigen::MatrixXd W = delassus(cholesky_solver(obj), c1.H);
    const Eigen::MatrixXd Linv = Eigen::MatrixXd(obj.L).inverse();
    const Eigen::MatrixXd W_ref = Eigen::MatrixXd(c1.H) * Linv * Eigen::MatrixXd(c1.H).transpose();
    check((W - W_ref).cwiseAbs().maxCoeff() < 1e-9 * W_ref.cwiseAbs().maxCoeff(), "Delassus = H L^-1 H^T (dense inverse)");
    check((W - W.transpose()).cwiseAbs().maxCoeff() == 0.0 && W.diagonal().minCoeff() > 0.0, "Delassus symmetric, positive diagonal");

    // Schur solve, including negative E entries
    RealVecX omega = (RealVecX::Random(3 * m).array() + 1.0) * 0.5, e = RealVecX::Random(3 * m).array() * 0.5 + 1.0, rhs = RealVecX::Random(3 * m);
    e(0) = -0.7;
    const RealVecX x = solve_schur(W, omega, e, h, rhs);
    Eigen::MatrixXd D = (h * h) * omega.asDiagonal() * W * omega.asDiagonal();
    D += e.asDiagonal();
    check((D * x - rhs).norm() < 1e-8 * rhs.norm(), "Schur solve residual (with a negative E entry)");
}

// One particle (m = 1 kg, L = m I) on a static plane through the origin, x^- = 0: the closed-form step of the contact + friction problem.
// This tests the whole Newton iteration (functions, Delassus, Schur, algebra of the pseudocode) without a cloth.
//   free step x~ = h v^- + h^2 g; lambda_n = -m n.x~ / h^2 (needs n.x~ < 0), x~_t = x~ - (n.x~) n
//   stick (m |x~_t| / h^2 <= mu lambda_n): x = 0, lambda_f = -m x~_t / h^2
//   slip: x = (|x~_t| - h^2 mu lambda_n / m) t^,  lambda_f = -mu lambda_n t^,  t^ = x~_t / |x~_t|
inline void check_single_particle_newton()
{
    std::printf("== one particle on a plane: Newton iteration against the closed form\n");
    const Real h = 1.0 / 60, mass = 1.0;
    const Vec3 g(0.0, -9.81, 0.0);
    struct Case { const char* name; Real theta_deg; Vec3 v; Real mu; };
    const Case cases[] = {
        { "resting on a 10 deg slope, mu 0   ", 10.0, Vec3(0, 0, 0),   0.0 },
        { "resting on a 10 deg slope, mu 0.1 ", 10.0, Vec3(0, 0, 0),   0.1 },
        { "resting on a 10 deg slope, mu 0.3 ", 10.0, Vec3(0, 0, 0),   0.3 },
        { "landing flat while sliding, mu 0.3", 0.0,  Vec3(2, -1, 0),  0.3 },
        { "landing flat while sliding, mu 1.0", 0.0,  Vec3(2, -1, 0),  1.0 },
        { "45 deg slope, fast, mu 0.2        ", 45.0, Vec3(0, -3, 1),  0.2 },
        { "45 deg slope, fast, mu 2.0        ", 45.0, Vec3(0, -3, 1),  2.0 },
    };
    for (const Case& cs : cases)
    {
        const Real th = cs.theta_deg * 3.14159265358979323846 / 180.0;
        const Vec3 n(-std::sin(th), std::cos(th), 0.0);
        const Vec3 xt = h * cs.v + h * h * g; // free step
        const Real xn = n.dot(xt);
        if (xn >= 0.0) { check(false, std::string(cs.name) + ": scene does not penetrate"); continue; }

        NcpContacts c;
        c.mu = cs.mu;
        c.d  = RealVecX::Zero(3);
        Vec3 t1, t2;
        tangent_basis(n, t1, t2);
        std::vector<Triplet> trip;
        const Vec3 rows[3] = { n, t1, t2 };
        for (int r = 0; r < 3; ++r) for (int a = 0; a < 3; ++a) trip.emplace_back(r, a, rows[r](a));
        c.H.resize(3, 3);
        c.H.setFromTriplets(trip.begin(), trip.end());

        auto solve_L = [&](const auto& B) { using T = std::decay_t<decltype(B)>; return T(B / mass); };
        const NcpSystem sys = make_ncp_system(solve_L, c.H, h);
        RealVecX b = mass * xt, x = xt, lambda = RealVecX::Zero(3);
        int it = 0;
        NcpUpdate u;
        for (; it < 100; ++it)
        {
            u = ncp_iteration(solve_L, c, sys, 1e-12, h, b, x, lambda);
            if (u.dx < 1e-10 * xt.norm() && u.dlambda < 1e-10) { ++it; break; }
        }

        const Real lam_n = -mass * xn / (h * h);
        const Vec3 xt_t  = xt - xn * n;
        Vec3 x_exp, lf_exp;
        bool stick = mass * xt_t.norm() / (h * h) <= cs.mu * lam_n;
        if (stick) { x_exp = Vec3::Zero(); lf_exp = -mass * xt_t / (h * h); }
        else       { const Vec3 th_hat = xt_t.normalized(); x_exp = (xt_t.norm() - h * h * cs.mu * lam_n / mass) * th_hat; lf_exp = -cs.mu * lam_n * th_hat; }
        const Vec3 lf = lambda(1) * t1 + lambda(2) * t2;
        const Real ex = (x - x_exp).norm(), el = std::abs(lambda(0) - lam_n) + (lf - lf_exp).norm();
        const bool ok = it < 100 && ex <= 1e-4 * xt.norm() + 1e-8 && el <= 1e-4 * lam_n;
        std::printf("  [%s] %s %s: %d iterations, |x - exact| %.2e (|x~| %.2e), |lambda - exact| %.2e (lambda_n %.3g)\n",
                    ok ? "ok" : "FAIL", cs.name, stick ? "stick" : "slip ", it, ex, xt.norm(), el, lam_n);
        if (!ok) ++failures;
    }
}

// ---- forward runs (cloth scenes) ----

struct Run { Object obj; Tape tape; NcpLog log; };

inline Run run_ncp(const AppConfig& cfg, const NcpSettings& ncp, int steps, const Vec3& v0 = Vec3::Zero())
{
    WorldSetup world = prepare_world(cfg);
    world.collision.collect_backward = false;
    const Real dt = 1.0 / (cfg.solver.FPS * cfg.solver.frame_substeps);
    Run r;
    r.obj = build_run_cloth(cfg, Stiffness::uniform(cfg.stiffness));
    construct_pd_lhs(r.obj, dt);
    for (Index i = 0; i < r.obj.num_particles(); ++i) r.obj.v.segment<3>(3 * i) = v0;
    pd_ncp_contact(r.obj, world.collision, dt, cfg.solver.gravity, ncp, steps, cfg.solver.frame_substeps, r.tape, "ncp_check", false, false, nullptr, 0.0, 0, &r.log);
    return r;
}

inline Run run_impulse(const AppConfig& cfg, int steps)
{
    WorldSetup world = prepare_world(cfg);
    world.collision.collect_backward = false;
    const Real dt = 1.0 / (cfg.solver.FPS * cfg.solver.frame_substeps);
    Run r;
    r.obj = build_run_cloth(cfg, Stiffness::uniform(cfg.stiffness));
    construct_pd_lhs(r.obj, dt);
    pd_contact(r.obj, world.collision, dt, cfg.solver.gravity, cfg.solver.n_iters, steps, cfg.solver.frame_substeps, r.tape, "ncp_check_impulse", false, false);
    return r;
}

inline Real mean_vx(const Tape& t, size_t frame) { return t.velocities[frame].col(0).mean(); }

inline NcpSettings test_settings(Real mu, int iters = 40, Real tol = 1e-7)
{
    NcpSettings s;
    s.friction = mu; s.n_iters = iters; s.tol = tol;
    return s;
}

inline void check_cloth_on_sphere()
{
    std::printf("== cloth on a static sphere, mu 0: NCP against the impulse method\n");
    const Vec3 center(1.0, 0.3, 1.0);
    AppConfig cfg = cloth_on_sphere(make_sphere(center, 0.5));
    const int steps = 60;
    const auto t0 = std::chrono::steady_clock::now();
    Run a = run_ncp(cfg, test_settings(0.0, 600, 1e-9), steps); // both methods converged: the PD local-global loop contracts slowly, like the impulse one
    const Real secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    Run b = run_impulse(cfg, steps);

    const PointsX& xa = a.tape.positions.back();
    const PointsX& xb = b.tape.positions.back();
    check(xa.allFinite(), "NCP positions finite");
    const Real dmax = (xa - xb).rowwise().norm().maxCoeff(), dmean = (xa - xb).rowwise().norm().mean();
    // The two methods discretise contact differently (position level with exact non-penetration vs velocity level, which lets particles sink,
    // see "unresolved" above), so they differ at dt = 1/60; they must approach each other as the time step shrinks.
    AppConfig fine = cfg;
    fine.solver.frame_substeps = 4; // 1/240 s: same 1 s, same kColliderAnimFPS rule is not involved (no animation)
    const Run a4 = run_ncp(fine, test_settings(0.0, 600, 1e-9), steps * 4);
    const Run b4 = run_impulse(fine, steps * 4);
    const Real dmean4 = (a4.tape.positions.back() - b4.tape.positions.back()).rowwise().norm().mean();
    std::printf("  [info] final positions NCP vs impulse: dt 1/60: max %.3e m, mean %.3e m; dt 1/240: mean %.3e m\n", dmax, dmean, dmean4);
    check(dmean4 < 0.6 * dmean, "the difference between the methods shrinks with the time step");

    int max_it = 0, total_it = 0, contact_steps = 0;
    Real worst_gap = 0.0, worst_lambda = 0.0, worst_active_gap = 0.0, worst_depth = 0.0;
    for (size_t t = 0; t < a.log.iterations.size(); ++t)
    {
        max_it = std::max(max_it, a.log.iterations[t]);
        total_it += a.log.iterations[t];
        const Index m = a.log.n_contacts[t];
        if (m == 0) continue;
        ++contact_steps;
        for (Index j = 0; j < m; ++j)
        {
            worst_gap    = std::min(worst_gap, a.log.delta[t](j));
            worst_lambda = std::min(worst_lambda, a.log.lambda[t](j));
            if (a.log.lambda[t](j) > 1e-6) worst_active_gap = std::max(worst_active_gap, std::abs(a.log.delta[t](j)));
        }
    }
    for (size_t t = 0; t < a.tape.positions.size(); ++t)
        for (Index i = 0; i < a.tape.positions[t].rows(); ++i)
            worst_depth = std::min(worst_depth, (a.tape.positions[t].row(i).transpose() - center).norm() - 0.5);
    std::printf("  [info] %d steps with contacts, iterations: mean %.1f max %d, %.2f s total; worst gap %.2e, worst lambda_n %.2e, worst |gap| of a pressing contact %.2e, deepest particle %.2e\n",
                contact_steps, double(total_it) / steps, max_it, secs, worst_gap, worst_lambda, worst_active_gap, worst_depth);
    check(contact_steps > 10, "the cloth touches the sphere");
    check(worst_gap > -1e-5, "gap >= 0 at the last iterate (no penetration of the contact set)");
    check(worst_lambda > -1e-6, "lambda_n >= 0");
    check(worst_active_gap < 1e-5, "pressing contacts have gap 0");
    check(max_it < 600, "converges before the iteration cap");
    size_t pressing = 0;
    for (const Contacts& c : a.tape.contacts) pressing += c.size();
    check(pressing > 0, "tape.contacts holds the pressing contacts (" + std::to_string(pressing) + ")");
}

inline void check_friction()
{
    std::printf("== friction on a plane\n");
    const int steps = 60;
    // a cloth sliding on the ground (v = 1 m/s along x): mu 0 keeps the speed, mu > 0 stops it and it stays stopped
    AppConfig cfg = cloth_on_sphere(make_plane(Vec3(0, 0, 0), Vec3(0, 1, 0)));
    cfg.cloth.pin_mode = PinMode::NONE;
    cfg.cloth.origin = Vec3(0.0, 0.01, 0.0);
    const Run free_slide = run_ncp(cfg, test_settings(0.0), steps, Vec3(1, 0, 0));
    const Run rubbing    = run_ncp(cfg, test_settings(0.5), steps, Vec3(1, 0, 0));
    std::printf("  [info] mean vx at 0.5 s / 1 s: mu 0: %.3f / %.3f, mu 0.5: %.3f / %.3f\n",
                mean_vx(free_slide.tape, 30), mean_vx(free_slide.tape, steps), mean_vx(rubbing.tape, 30), mean_vx(rubbing.tape, steps));
    check(mean_vx(free_slide.tape, steps) > 0.9, "mu = 0: the cloth keeps sliding");
    check(std::abs(mean_vx(rubbing.tape, 30)) < 0.05 && std::abs(mean_vx(rubbing.tape, steps)) < 0.05, "mu = 0.5: it stops (mu g = 4.9 m/s^2 -> 0.2 s) and stays stopped");

    // a plane moving along x under a resting cloth: friction drags the cloth to the plane's speed (v_c enters the slip)
    AppConfig moving = cloth_on_sphere(make_plane(Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(1, 0, 0)));
    moving.cloth.pin_mode = PinMode::NONE;
    moving.cloth.origin = Vec3(0.0, 0.01, 0.0);
    const Run dragged = run_ncp(moving, test_settings(1.0), steps);
    const Run smooth  = run_ncp(moving, test_settings(0.0), steps);
    std::printf("  [info] moving plane (1 m/s), mean vx of the cloth at 1 s: mu 1: %.3f, mu 0: %.3f\n", mean_vx(dragged.tape, steps), mean_vx(smooth.tape, steps));
    check(mean_vx(dragged.tape, steps) > 0.9, "mu = 1: the cloth is dragged along with the moving plane");
    check(std::abs(mean_vx(smooth.tape, steps)) < 0.02, "mu = 0: the moving plane does not move the cloth");
}

inline void check_rotating_sphere()
{
    std::printf("== cloth on a rotating sphere (omega = 2 rad/s about y)\n");
    const Vec3 center(1.0, 0.3, 1.0);
    Collider sphere = make_sphere(center, 0.5);
    sphere.rotation_axis = RotationAxis::Y; sphere.rotation_origin = center; sphere.omega = 2.0;
    AppConfig cfg = cloth_on_sphere(sphere);
    cfg.cloth.pin_mode = PinMode::NONE;
    const int steps = 60;
    // angular speed about y of the particles pressing at the end
    auto spin = [&](const Run& r)
    {
        const Contacts& cs = r.tape.contacts.back();
        if (cs.empty()) return std::numeric_limits<Real>::quiet_NaN();
        Real num = 0.0, den = 0.0;
        for (const Contact& c : cs)
        {
            const Vec3 p = r.tape.positions.back().row(c.particle).transpose() - center, v = r.tape.velocities.back().row(c.particle).transpose();
            num += p.z() * v.x() - p.x() * v.z();
            den += p.x() * p.x() + p.z() * p.z();
        }
        return num / den;
    };
    const Run rough  = run_ncp(cfg, test_settings(1.0), steps);
    const Run smooth = run_ncp(cfg, test_settings(0.0), steps);
    std::printf("  [info] angular speed of the pressing particles: mu 1: %.3f, mu 0: %.3f rad/s (%zu / %zu pressing)\n",
                spin(rough), spin(smooth), rough.tape.contacts.back().size(), smooth.tape.contacts.back().size());
    check(spin(rough) > 0.5 * 2.0, "mu = 1: the cloth is carried round with the sphere");
    check(std::abs(spin(smooth)) < 0.25 * 2.0, "mu = 0: the sphere spins under the cloth");
}
} // namespace ncp_check

static int run_ncp_check()
{
    ncp_check::failures = 0;
    ncp_check::check_ncp_functions();
    ncp_check::check_contacts_and_delassus();
    ncp_check::check_single_particle_newton();
    ncp_check::check_cloth_on_sphere();
    ncp_check::check_friction();
    ncp_check::check_rotating_sphere();
    std::printf(ncp_check::failures ? "ncp-check: %d FAILED\n" : "ncp-check: all ok\n", ncp_check::failures);
    return ncp_check::failures ? 1 : 0;
}
