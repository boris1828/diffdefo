#pragma once

// `diffpd --mesh-vertex-grad-check`: collision/mesh_pullback.h against central differences of the REAL contact query. A convex icosahedron
// under a random rigid pose, one particle in its thickness shell per trial, found at a face / an edge / a vertex. The loss is
// l = lambda_n . n + lambda_phi phi of the contact stamp_contacts produces (Gap model), every vertex coordinate perturbed in the rest frame,
// the mesh rebuilt (build_trimesh) and the contact stamped again. Also: the corner sum equals the rigid-collider gradient the backward pass
// already has (-(S lambda_n + lambda_phi grad phi)), a face check with the normal term removed must fail, and without the shell (thickness 0)
// an edge / vertex contact must report its skipped normal term while its gap term still matches.

#include "collision/detect.h"
#include "collision/mesh_pullback.h"
#include "solver/impulse/backward.h"

#include <random>

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
    std::printf(failures ? "FAILED\n" : "all ok\n");
    return failures ? 1 : 0;
}
