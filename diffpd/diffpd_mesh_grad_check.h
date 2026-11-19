#pragma once

// `diffpd --mesh-vertex-grad-check`: collision/mesh_pullback.h against central differences of the REAL contact query. A convex icosahedron
// under a random rigid pose, one particle in its thickness shell per trial, found at a face / an edge / a vertex. The loss is
// l = lambda_n . n + lambda_phi phi of the contact stamp_contacts produces (Gap model), every vertex coordinate perturbed in the rest frame,
// the mesh rebuilt (build_trimesh) and the contact stamped again. Also: the corner sum equals the rigid-collider gradient the backward pass
// already has (-(S lambda_n + lambda_phi grad phi)), a face check with the normal term removed must fail, and without the shell (thickness 0)
// an edge / vertex contact must report its skipped normal term while its gap term still matches.

#include "collision/detect.h"
#include "collision/mesh_pullback.h"
#include "diffpd_friction_check.h" // friction_check::scaled_obj
#include "diffpd_ncp_check.h"      // ncp_check::cloth_on_sphere
#include "solver/impulse/backward.h"

#include <cstdint>
#include <random>
#include <sstream>

namespace mesh_grad_check
{
inline TriMesh icosahedron(Real radius)
{
    const Real t = (1.0 + std::sqrt(5.0)) / 2.0;
    const Vec3 corners[12] = { Vec3(-1, t, 0), Vec3(1, t, 0), Vec3(-1, -t, 0), Vec3(1, -t, 0), Vec3(0, -1, t), Vec3(0, 1, t),
                               Vec3(0, -1, -t), Vec3(0, 1, -t), Vec3(t, 0, -1), Vec3(t, 0, 1), Vec3(-t, 0, -1), Vec3(-t, 0, 1) };
    std::vector<Vec3> V;
    for (const Vec3& c : corners) V.push_back(radius * c.normalized());
    const std::vector<std::array<int, 3>> T = { {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                                                {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1} };
    TriMesh m;
    ASSERT(build_trimesh("icosahedron", V, T, m), "mesh-vertex-grad-check: the icosahedron does not build");
    return m;
}

// The Gap contact of a particle at p_world, exactly as the backward pass sees it (stamp_contacts).
inline Contact stamp(const TriMesh& mesh, const RigidPose& pose, Real thickness, const Vec3& p_world)
{
    CollisionWorld w;
    w.trimeshes = { mesh };
    Collider col;
    col.type = ColliderType::Mesh; col.mesh_id = 0; col.rigid_pose = pose;
    w.colliders = { col };
    w.contact.model = ContactModel::Gap; w.contact.thickness = thickness;
    Contacts cs(1);
    cs[0].particle = 0; cs[0].collider_id = 0;
    StepClock clock;
    clock.frozen = true;
    Positions x(3);
    x = p_world;
    stamp_contacts(w, clock, 0, x, cs, 0, 1);
    return cs[0];
}

// What a contact touches, independent of which of the triangles sharing an edge / vertex reports it: the face, the sorted edge ends, the vertex.
inline std::array<int, 3> touched(const TriMesh& mesh, const Contact& c)
{
    const auto& t = mesh.triangles[c.triangle_id];
    if (c.feature == ContactFeature::Face) return { c.triangle_id, -1, -1 };
    if (c.feature == ContactFeature::Vertex) return { t[c.feature_index], -1, -2 };
    const int a = t[c.feature_index], b = t[(c.feature_index + 1) % 3];
    return { std::min(a, b), std::max(a, b), -3 };
}

inline void run_case(const char* name, ContactFeature want, Real thickness, bool shell, std::mt19937_64& rng, int& failures)
{
    const TriMesh mesh = icosahedron(0.1);
    const int n_tri = (int)mesh.triangles.size();
    auto uni = [&](Real a, Real b) { return std::uniform_real_distribution<Real>(a, b)(rng); };
    auto unit_random = [&]() { return Vec3(uni(-1, 1), uni(-1, 1), uni(-1, 1)).normalized(); };

    std::vector<Real> err, err_no_normal, err_rigid;
    int trials = 0, wrong_feature = 0, moved_feature = 0;
    bool all_skipped_reported = true, all_gap_ok = true;
    while ((int)err.size() < 60 && trials++ < 5000)
    {
        RigidPose pose;
        pose.position = Vec3(uni(-.2, .2), uni(-.2, .2), uni(-.2, .2));
        pose.rotation = Eigen::Quaternion<Real>(Eigen::AngleAxis<Real>(uni(0, 6.28), unit_random()));

        // a particle in the shell above the wanted feature (rest frame; the mesh is convex, so the closest feature is the one constructed)
        const int t = std::uniform_int_distribution<int>(0, n_tri - 1)(rng), e = std::uniform_int_distribution<int>(0, 2)(rng);
        const auto& tri = mesh.triangles[t];
        const Real dd = uni(0.2, 0.9) * 0.01;
        Vec3 p_rest;
        if (want == ContactFeature::Face)
        {
            Vec3 b(uni(.15, 1), uni(.15, 1), uni(.15, 1));
            b /= b.sum();
            p_rest = b[0] * mesh.vertices[tri[0]] + b[1] * mesh.vertices[tri[1]] + b[2] * mesh.vertices[tri[2]] + dd * mesh.face_normal[t];
        }
        else if (want == ContactFeature::Edge)
        {
            const Vec3 a = mesh.vertices[tri[e]], b = mesh.vertices[tri[(e + 1) % 3]];
            const Real s = uni(.25, .75), w = uni(.2, .8);
            p_rest = a + s * (b - a) + dd * (w * mesh.face_normal[t] + (1 - w) * mesh.face_normal[mesh.edge_neighbor[t][e]]).normalized();
        }
        else
            p_rest = mesh.vertices[tri[e]] + dd * (mesh.vertex_pseudo_normal[tri[e]] + 0.15 * unit_random()).normalized();
        const Vec3 p = pose.position + pose.rotation * p_rest;

        const Contact c = stamp(mesh, pose, thickness, p);
        if (c.feature != want || c.triangle_id < 0) { ++wrong_feature; continue; }

        const Vec3 lambda_n = Vec3(uni(-1, 1), uni(-1, 1), uni(-1, 1));
        const Real lambda_phi = uni(-1, 1);
        // central differences of l over every vertex coordinate (rest frame); the contact must keep its triangle and feature
        auto loss_at = [&](const std::vector<Vec3>& V, bool& same)
        {
            TriMesh m;
            ASSERT(build_trimesh("perturbed", V, mesh.triangles, m), "mesh-vertex-grad-check: the perturbed mesh does not build");
            const Contact cp = stamp(m, pose, thickness, p);
            same = same && cp.feature == c.feature && touched(mesh, cp) == touched(mesh, c);
            return lambda_n.dot(cp.normal) + lambda_phi * cp.gap;
        };
        const Real eps = 1e-6;
        std::vector<Vec3> fd(mesh.vertices.size(), Vec3::Zero());
        bool same = true;
        for (size_t v = 0; v < mesh.vertices.size(); ++v)
            for (int k = 0; k < 3; ++k)
            {
                std::vector<Vec3> Vp = mesh.vertices, Vm = mesh.vertices;
                Vp[v][k] += eps; Vm[v][k] -= eps;
                fd[v][k] = (loss_at(Vp, same) - loss_at(Vm, same)) / (2 * eps);
            }
        if (!same) { ++moved_feature; continue; }

        auto max_diff = [&](const std::vector<Vec3>& a, const std::vector<Vec3>& b)
        {
            Real d = 0.0, s = 1e-12;
            for (size_t v = 0; v < a.size(); ++v) { d = std::max(d, (a[v] - b[v]).cwiseAbs().maxCoeff()); s = std::max(s, b[v].cwiseAbs().maxCoeff()); }
            return d / s;
        };
        std::vector<Vec3> g(mesh.vertices.size(), Vec3::Zero());
        const bool normal_done = mesh_vertex_pullback(mesh, pose, c, lambda_n, lambda_phi, g);
        if (shell)
        {
            err.push_back(max_diff(g, fd));
            Vec3 sum = Vec3::Zero();
            for (const Vec3& gv : g) sum += pose.rotation * gv;
            const Vec3 lam_t = lambda_n - c.normal * c.normal.dot(lambda_n);
            const Vec3 rigid = -(apply_shape_operator(c, lam_t) + lambda_phi * c.gap_grad);
            err_rigid.push_back((sum - rigid).norm() / std::max(rigid.norm(), 1e-12));
            if (want == ContactFeature::Face)
            {
                std::vector<Vec3> g0(mesh.vertices.size(), Vec3::Zero());
                mesh_vertex_pullback(mesh, pose, c, Vec3::Zero(), lambda_phi, g0); // the normal term removed: must fail
                err_no_normal.push_back(max_diff(g0, fd));
            }
            all_skipped_reported = all_skipped_reported && normal_done;
        }
        else // no shell: a pseudo-normal contact, so only its gap term is differentiated (lambda_n = 0 in the loss)
        {
            const Vec3 zero = Vec3::Zero();
            std::vector<Vec3> g1(mesh.vertices.size(), Vec3::Zero()), fd1(mesh.vertices.size(), Vec3::Zero());
            const bool done = mesh_vertex_pullback(mesh, pose, c, zero, lambda_phi, g1);
            for (size_t v = 0; v < mesh.vertices.size(); ++v)
                for (int k = 0; k < 3; ++k)
                {
                    std::vector<Vec3> Vp = mesh.vertices, Vm = mesh.vertices;
                    Vp[v][k] += eps; Vm[v][k] -= eps;
                    auto phi_at = [&](const std::vector<Vec3>& V) { TriMesh m; build_trimesh("perturbed", V, mesh.triangles, m); return stamp(m, pose, thickness, p).gap; };
                    fd1[v][k] = lambda_phi * (phi_at(Vp) - phi_at(Vm)) / (2 * eps);
                }
            const Real e1 = max_diff(g1, fd1);
            err.push_back(e1);
            all_skipped_reported = all_skipped_reported && !done;
            all_gap_ok = all_gap_ok && e1 < 1e-5;
        }
    }
    const Real worst = err.empty() ? 1.0 : *std::max_element(err.begin(), err.end());
    bool ok = (int)err.size() == 60 && worst < 1e-5 && all_skipped_reported && all_gap_ok;
    std::printf("  %-34s %zu contacts (%d samples had another feature, %d changed feature under the FD step), vs FD max %.2e", name, err.size(), wrong_feature, moved_feature, worst);
    if (shell && !err_rigid.empty())
    {
        const Real worst_rigid = *std::max_element(err_rigid.begin(), err_rigid.end());
        std::printf(", corner sum vs rigid gradient %.2e", worst_rigid);
        ok = ok && worst_rigid < 1e-9;
        if (want == ContactFeature::Face && !err_no_normal.empty())
        {
            const Real least = *std::min_element(err_no_normal.begin(), err_no_normal.end());
            std::printf(", without the normal term min error %.2e", least);
            ok = ok && least > 1e-3; // every face case must notice the missing tilt term
        }
    }
    else std::printf(", normal term reported skipped: %s", all_skipped_reported ? "yes" : "NO");
    std::printf("  %s\n", ok ? "ok" : "FAIL");
    failures += !ok;
}
// The same gradient through a whole simulation: an 8 x 8 cloth falls 120 steps onto the animation/ icosphere (r 0.5, 80 triangles), Gap model with
// a 1 cm thickness shell, loss L = u . x_T (random u, final frame). backward_pd_contact's mesh_vertices against central differences of the forward
// along three directions of vertex motion (random, inflate, steepest at the most influential vertex), plus the exact identity
// sum_v g_v = sum_s ColliderGrad::d_position[s]. Needs converged solves; as in friction-check a quotient is only taken where the + and - runs
// share their contact structure, since the forward is piecewise smooth.
inline void run_e2e_case(Real mu, int& failures)
{
    const Vec3 sph_c(0.8, 0.4, 0.8);
    Collider mesh_sphere;
    mesh_sphere.type = ColliderType::Mesh; mesh_sphere.mesh_id = 0; mesh_sphere.rigid_pose.position = sph_c;
    AppConfig cfg = ncp_check::cloth_on_sphere(mesh_sphere);
    cfg.cloth.width = cfg.cloth.height = 8; cfg.cloth.size_x = cfg.cloth.size_y = 1.6;
    cfg.cloth.pin_mode = PinMode::CORNERS;
    cfg.solver.anderson_m = 5;
    cfg.world.contact.model = ContactModel::Gap; cfg.world.contact.margin = 0.01; cfg.world.contact.thickness = 0.01;
    cfg.world.contact.friction = mu;
    const Vec3 g = cfg.solver.gravity;
    const Real h = 1.0 / (cfg.solver.FPS * cfg.solver.frame_substeps);
    const int steps = 120, iters = 300;

    TriMesh mesh0;
    {
        std::istringstream in(friction_check::scaled_obj("tri_sphere.obj", 1.0));
        ASSERT(load_trimesh_obj(in, "mesh-vertex-grad-check", mesh0), "mesh-vertex-grad-check: tri_sphere.obj does not load");
    }
    auto make_world = [&](const TriMesh& m) { WorldSetup w = prepare_world(cfg); w.collision.trimeshes = { m }; return w; };
    auto simulate = [&](WorldSetup& w, Object& o, Tape& t, Real unresolved)
    {
        o = build_run_cloth(cfg, Stiffness::uniform(cfg.stiffness));
        construct_pd_lhs(o, h);
        pd_contact(o, w.collision, h, g, iters, steps, 1, t, "mesh_vertex_grad_check", false, false, nullptr, unresolved);
    };

    // analytic: the real backward over the whole drape
    WorldSetup w0 = make_world(mesh0);
    Object o0; Tape tape;
    simulate(w0, o0, tape, 1e-3);
    const Index dofs = o0.num_dofs();
    const RealVecX u = RealVecX::Random(dofs);
    Loss loss;
    loss.dloss_dx.assign(steps + 1, RealVecX::Zero(dofs));
    loss.dloss_dx.back() = u;
    const BackwardGradContact grad = backward_pd_contact(o0, w0.collision, tape, loss, g, iters, h);
    const std::vector<Vec3>& gv = grad.mesh_vertices.at(0);
    Real est = 0.0;
    for (const StepResidual& r : tape.forward_residual) est = std::max(est, r.est_rel);
    int n_touched = 0, steepest = 0;
    for (size_t v = 0; v < gv.size(); ++v)
    {
        n_touched += gv[v].squaredNorm() > 0.0;
        if (gv[v].norm() > gv[steepest].norm()) steepest = (int)v;
    }

    Vec3 sum_v = Vec3::Zero(), sum_c = Vec3::Zero();
    for (const Vec3& x : gv) sum_v += x;
    for (const Vec3& x : grad.colliders[0].d_position) sum_c += x;
    const Real e_sum = (sum_v - sum_c).norm() / std::max(sum_c.norm(), 1e-12);

    // the loss of a perturbed mesh, and the contact structure of its run (per step: which particles touch which triangle in which Coulomb branch)
    struct Probe { Real loss; std::vector<std::uint64_t> structure; };
    auto probe = [&](const TriMesh& m)
    {
        WorldSetup w = make_world(m);
        w.collision.collect_backward = false;
        Object o; Tape t;
        simulate(w, o, t, -1.0);
        Probe p{ u.dot(o.x), {} };
        for (int s = 1; s <= steps; ++s)
        {
            Contacts cs = t.contacts[s - 1];
            precompute_contacts_local_derivative(cs, o,
                Eigen::Map<const Positions>(t.positions[s].data(), dofs), Eigen::Map<const Velocities>(t.velocities[s].data(), dofs),
                Eigen::Map<const Positions>(t.positions[s - 1].data(), dofs), Eigen::Map<const Velocities>(t.velocities[s - 1].data(), dofs),
                g, h, true);
            for (const Contact& cn : cs)
            {
                const int state = cn.active ? (mu > 0.0 ? (int)coulomb_state(cn.d, cn.normal, mu) : (int)CoulombState::Slip) : (int)CoulombState::TakeOff;
                if (state == (int)CoulombState::TakeOff) continue; // r = 0, R = 0: detecting it or not changes nothing
                // the feature only where the normal jumps with it: inside the surface; in the thickness shell the normal is continuous
                const bool inside = cn.gap + cfg.world.contact.thickness < 0.0;
                const std::uint64_t where = inside ? ((std::uint64_t(cn.triangle_id + 1) << 2) ^ (std::uint64_t(3 * (int)cn.feature + cn.feature_index) << 58)) : 0;
                p.structure.push_back((std::uint64_t(s) << 40) ^ (std::uint64_t(cn.particle) << 20) ^ where ^ std::uint64_t(state));
            }
        }
        return p;
    };
    auto displaced = [&](const std::vector<Vec3>& dir, Real e)
    {
        std::vector<Vec3> V = mesh0.vertices;
        for (size_t v = 0; v < V.size(); ++v) V[v] += e * dir[v];
        TriMesh m;
        ASSERT(build_trimesh("mesh-vertex-grad-check", V, mesh0.triangles, m), "mesh-vertex-grad-check: the displaced mesh does not build");
        return m;
    };
    struct FD { Real value, eps; }; // eps = 0: no step kept the contact structure
    auto central = [&](const std::vector<Vec3>& dir, Real eps)
    {
        for (int k = 0; k < 4; ++k, eps *= 0.1)
        {
            const Probe p = probe(displaced(dir, eps)), m = probe(displaced(dir, -eps));
            if (p.structure == m.structure) return FD{ (p.loss - m.loss) / (2 * eps), eps };
        }
        return FD{ 0.0, 0.0 };
    };

    std::vector<std::pair<const char*, std::vector<Vec3>>> dirs(3);
    dirs[0].first = "random per vertex"; dirs[1].first = "inflate (radial)"; dirs[2].first = "steepest, one vertex";
    for (size_t v = 0; v < gv.size(); ++v)
    {
        dirs[0].second.push_back(Vec3::Random());
        dirs[1].second.push_back(2.0 * mesh0.vertices[v]); // the icosphere is centred on its origin, r = 0.5
        dirs[2].second.push_back(v == (size_t)steepest ? Vec3(gv[v] / gv[v].norm()) : Vec3::Zero().eval());
    }

    std::printf("  mu %.1f: %d of %zu vertices touched, %d pseudo-normal contacts, forward est_rel max %.1e, sum_v g vs sum d_position %.1e\n",
                mu, n_touched, gv.size(), grad.mesh_pseudo_normal_contacts, est, e_sum);
    bool ok = e_sum < 1e-9 && n_touched > 0;
    const Real floor = 1e-6 * u.norm() * o0.x.norm();
    for (const auto& [name, dir] : dirs)
    {
        Real an = 0.0;
        for (size_t v = 0; v < gv.size(); ++v) an += gv[v].dot(dir[v]);
        const FD fd = central(dir, 1e-5);
        const Real err = fd.eps > 0.0 ? std::abs(an - fd.value) / std::max(std::abs(fd.value), floor) : 1.0;
        std::printf("    %-22s analytic % .6e  fd % .6e (eps %.0e)  rel err %.2e\n", name, an, fd.value, fd.eps, err);
        ok = ok && err < 1e-3;
    }
    std::printf("    %s\n", ok ? "ok" : "FAIL");
    failures += !ok;
}
} // namespace mesh_grad_check

static int run_mesh_grad_check()
{
    std::mt19937_64 rng(3);
    int failures = 0;
    std::printf("== mesh vertex gradient (collision/mesh_pullback.h) vs central differences of the real contact query\n");
    mesh_grad_check::run_case("face, shell 1 cm",          ContactFeature::Face,   0.01, true,  rng, failures);
    mesh_grad_check::run_case("edge, shell 1 cm",          ContactFeature::Edge,   0.01, true,  rng, failures);
    mesh_grad_check::run_case("vertex, shell 1 cm",        ContactFeature::Vertex, 0.01, true,  rng, failures);
    mesh_grad_check::run_case("edge, no shell (gap term)", ContactFeature::Edge,   0.0,  false, rng, failures);
    mesh_grad_check::run_case("vertex, no shell (gap term)", ContactFeature::Vertex, 0.0, false, rng, failures);
    std::printf("== mesh vertex gradient through a drape (backward_pd_contact::mesh_vertices) vs central differences of the forward\n");
    mesh_grad_check::run_e2e_case(0.4, failures);
    mesh_grad_check::run_e2e_case(0.0, failures);
    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
