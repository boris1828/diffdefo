#pragma once

#include "core/diffpd_core.h"

// ----------------
//   ARAP TRIANGLES
// ----------------

// As-rigid-as-possible membrane energy per triangle, E = (w/2) ||F - R(F)||^2 with F = X Ghat (3x2), R the polar
// rotation, w = k * A0. Backward per arap3.pdf Sec. 4-5 (Gam, Delta L y, dphi/dk). Same contract as SpringEnergy (see ElasticEnergies
// in diffpd_types.h); `pinned` is Object::mesh.pinned_rest.
struct ArapEnergy
{
    using Mat32 = Eigen::Matrix<Real, 3, 2>;
    using Mat6  = Eigen::Matrix<Real, 6, 6>;

    struct Tri
    {
        SpringKind kind;
        Real       w;        // k * A0
        bool       has_pinned; // any of the 3 vertices pinned
        ParticleId dof[3];   // >= 0: particle id, < 0: pinned, pinned_rest index -(dof+1)
        Mat32      ghat;     // rest stencil: F = [x0 x1 x2] * ghat
        Mat3       gg;       // ghat * ghat^T
        Real       A0;       // rest area (w = k A0, so dphi/dk = A0 dphi/dw)
        Mat32      R;        // cached by precompute_local_derivative: rotation at x+
        Mat6       Gam;      // cached by precompute_local_derivative: w dp/dq
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
        t.A0   = 0.5 * std::abs(Dm.determinant());
        t.w    = k * t.A0;
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

    // Caches R (= p_i) and Gam = w dp/dq (6x6, vec order S11 S21 S31 S12 S22 S32) at x (= x+ of the step being differentiated):
    // Gam = w (V (x) Ubar) Lambda (V (x) Ubar)^T, Ubar = [u1 u2 u1 x u2] (arap3.pdf, Sec. 4). No h^2 inside, like Constraint::gamma.
    void precompute_local_derivative(const RealVecX& x, const std::vector<Vec3>& pinned)
    {
        for (Tri& t : tris)
        {
            Mat3 X;
            for (int a = 0; a < 3; ++a)
                X.col(a) = t.dof[a] >= 0 ? Vec3(x.segment<3>(3 * t.dof[a])) : pinned[-(t.dof[a] + 1)];

            const Mat32 S = X * t.ghat;
            t.R = polar_R(S);

            Eigen::JacobiSVD<Mat32> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const Real s1 = std::max(svd.singularValues()(0), 1e-12), s2 = std::max(svd.singularValues()(1), 1e-12);
            Mat3 Ubar = svd.matrixU();
            Ubar.col(2) = Ubar.col(0).cross(Ubar.col(1));

            const Real c = 1.0 / (s1 + s2);
            Mat6 Lambda = Mat6::Zero();
            Lambda(1, 1) = c;  Lambda(1, 3) = -c;
            Lambda(3, 1) = -c; Lambda(3, 3) = c;
            Lambda(2, 2) = 1.0 / s1;
            Lambda(5, 5) = 1.0 / s2;

            Mat6 K; // kron(V, Ubar)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) K.block<3, 3>(3 * i, 3 * j) = svd.matrixV()(i, j) * Ubar;
            t.Gam = t.w * (K * Lambda * K.transpose());
        }
    }

    // out += (sum_t G^T Gam G) v: dS = [v_a v_b v_c] Ghat (pinned columns 0), dR = mat(Gam vec dS), out_a += (dR Ghat^T)_a.
    void add_jacobian_product(const RealVecX& v, RealVecX& out) const
    {
        for (const Tri& t : tris)
        {
            Mat3 Y = Mat3::Zero();
            for (int a = 0; a < 3; ++a)
                if (t.dof[a] >= 0) Y.col(a) = v.segment<3>(3 * t.dof[a]);

            const Mat32 dS = Y * t.ghat;
            Eigen::Matrix<Real, 6, 1> dR = t.Gam * Eigen::Map<const Eigen::Matrix<Real, 6, 1>>(dS.data());
            const Mat3 f = Eigen::Map<const Mat32>(dR.data()) * t.ghat.transpose();
            for (int a = 0; a < 3; ++a)
                if (t.dof[a] >= 0) out.segment<3>(3 * t.dof[a]) += f.col(a);
        }
    }

    // grad[kind] += dphi/dk for this step = A0 <Z Ghat, h (R - X- Ghat) - h2_eff V+ Ghat>_F (arap3.pdf, Sec. 5, written like the
    // spring term so the damping h2_eff works unchanged). Needs precompute_local_derivative (R) for this step.
    void add_param_gradient(const RealVecX& z_perp, const RealVecX& x_minus, const RealVecX& v_plus,
                            const std::vector<Vec3>& pinned, Real h, Real h2_eff, Stiffness& grad) const
    {
        for (const Tri& t : tris)
        {
            Mat3 Z = Mat3::Zero(), Xm, Vp = Mat3::Zero();
            for (int a = 0; a < 3; ++a)
            {
                if (t.dof[a] >= 0)
                {
                    Z.col(a)  = z_perp.segment<3>(3 * t.dof[a]);
                    Xm.col(a) = x_minus.segment<3>(3 * t.dof[a]);
                    Vp.col(a) = v_plus.segment<3>(3 * t.dof[a]);
                }
                else Xm.col(a) = pinned[-(t.dof[a] + 1)];
            }
            const Mat32 q = h * (t.R - Xm * t.ghat) - h2_eff * (Vp * t.ghat);
            grad[t.kind] += t.A0 * (Z * t.ghat).cwiseProduct(q).sum();
        }
    }

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
