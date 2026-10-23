#pragma once

#include "diffpd_core.h"

// ----------------
//   ARAP TRIANGLES
// ----------------

// As-rigid-as-possible membrane energy per triangle, E = (w/2) ||F - R(F)||^2 with F = X Ghat (3x2), R the polar
// rotation, w = k * A0. Forward only: the backward methods assert. Same contract as SpringEnergy (see ElasticEnergies
// in diffpd_types.h); `pinned` is Object::mesh.pinned_rest.
struct ArapEnergy
{
    using Mat32 = Eigen::Matrix<Real, 3, 2>;

    struct Tri
    {
        SpringKind kind;
        Real       w;        // k * A0
        bool       has_pinned; // any of the 3 vertices pinned
        ParticleId dof[3];   // >= 0: particle id, < 0: pinned, pinned_rest index -(dof+1)
        Mat32      ghat;     // rest stencil: F = [x0 x1 x2] * ghat
        Mat3       gg;       // ghat * ghat^T
    };
    std::vector<Tri> tris;

    size_t triplet_count() const { return 27 * tris.size(); }

    // One triangle from its vertices' SimMesh::Vertex::dof and 3D rest positions (isometric to its flat rest shape).
    // Skipped if all three are pinned.
    void add_triangle(const ParticleId (&dof)[3], const Vec3 (&pos)[3], SpringKind kind, Real k)
    {
        if (dof[0] < 0 && dof[1] < 0 && dof[2] < 0) return;

        const Vec3 d1 = pos[1] - pos[0], d2 = pos[2] - pos[0];
        const Vec3 e1 = d1.normalized();
        const Vec3 e2 = d1.cross(d2).cross(d1).normalized(); // in-plane, perpendicular to e1
        Eigen::Matrix<Real, 2, 2> Dm;
        Dm << d1.norm(), d2.dot(e1),
              0.0,       d2.dot(e2);
        ASSERT(std::abs(Dm.determinant()) > 1e-18, "ARAP: degenerate rest triangle");

        Eigen::Matrix<Real, 3, 2> top;
        top << -1, -1,
                1,  0,
                0,  1;

        Tri t;
        t.kind = kind;
        t.w    = k * 0.5 * std::abs(Dm.determinant());
        for (int a = 0; a < 3; ++a) t.dof[a] = dof[a];
        t.has_pinned = dof[0] < 0 || dof[1] < 0 || dof[2] < 0;
        t.ghat = top * Dm.inverse();
        t.gg   = t.ghat * t.ghat.transpose();
        tris.push_back(t);
    }

    // scale * K, K = sum_t w (ghat ghat^T) (x) I3 on the free dofs.
    void add_stiffness_triplets(std::vector<Triplet>& out, Real scale) const
    {
        for (const Tri& t : tris)
            for (int a = 0; a < 3; ++a)
                for (int c = 0; c < 3; ++c)
                {
                    if (t.dof[a] < 0 || t.dof[c] < 0) continue;
                    for (int d = 0; d < 3; ++d)
                        out.emplace_back(3 * t.dof[a] + d, 3 * t.dof[c] + d, scale * t.w * t.gg(a, c));
                }
    }

    // b += scale * w * (R ghat^T) on the free vertices, minus the constant pinned-vertex coupling w gg_ac x_c.
    void add_forces(const RealVecX& x, const std::vector<Vec3>& pinned, RealVecX& b, Real scale) const
    {
        for (const Tri& t : tris)
        {
            Mat3 X;
            for (int a = 0; a < 3; ++a)
                X.col(a) = t.dof[a] >= 0 ? Vec3(x.segment<3>(3 * t.dof[a])) : pinned[-(t.dof[a] + 1)];

            const Mat3 F = polar_R(X * t.ghat) * t.ghat.transpose();
            for (int a = 0; a < 3; ++a)
            {
                if (t.dof[a] < 0) continue;
                Vec3 f = F.col(a);
                if (t.has_pinned)
                    for (int c = 0; c < 3; ++c)
                        if (t.dof[c] < 0) f -= t.gg(a, c) * X.col(c);
                b.segment<3>(3 * t.dof[a]) += scale * t.w * f;
            }
        }
    }

    void precompute_local_derivative(const RealVecX&, const std::vector<Vec3>&)
    { ASSERT(tris.empty(), "ARAP backward pass is not implemented"); }

    void add_jacobian_product(const RealVecX&, RealVecX&) const
    { ASSERT(tris.empty(), "ARAP backward pass is not implemented"); }

    void add_param_gradient(const RealVecX&, const RealVecX&, const RealVecX&, const std::vector<Vec3>&, Real, Real, Stiffness&) const
    { ASSERT(tris.empty(), "ARAP backward pass is not implemented"); }

    // R = S (S^T S)^(-1/2), S 3x2, closed form: A = S^T S, s = sqrt(det A) = ||f_u x f_v||, t = sqrt(tr A + 2s),
    // R = S adj(A + sI) / (s t). shortcut: s t clamped at 1e-24, a collapsed triangle gives a meaningless R; revisit with the backward pass.
    static Mat32 polar_R(const Mat32& S)
    {
        const Real 
            a11 = S.col(0).squaredNorm(), 
            a22 = S.col(1).squaredNorm(), 
            a12 = S.col(0).dot(S.col(1));
        const Real s   = S.col(0).cross(S.col(1)).norm();
        const Real inv = 1.0 / std::max(s * std::sqrt(a11 + a22 + 2.0 * s), 1e-24);
        Eigen::Matrix<Real, 2, 2> adj;
        adj << 
            a22 + s, -a12,
            -a12,    a11 + s;
        return S * adj * inv;
    }
};
