#pragma once

#include "diffpd_core.h"

// ----------------
//  HINGE BENDING
// ----------------

// Quadratic (cotangent) bending over two triangles sharing an edge, E = (w/2) ||G x - p||^2 with n_v = 4, r = 1:
// S = sum_a ghat_a x_a (a 3-vector, the discrete mean-curvature normal), p = l0 S/||S||, w = k A0 ||e0||^2 (Bergou et al. 2006, rescaled).
// Flat rest: l0 = 0, p = 0.
// SCALING: w = k A0 ||e0||^2 (e0 = the shared hinge edge), not Bergou's plain k A0. ghat ~ 1/A0 makes the stiffness-matrix
// entries ~ k cot^2 / A0 ~ k / h^2, i.e. ~1/h^2 times the membrane (ARAP) ones, so a plain k had to be ~1e-6 and changed with the
// mesh resolution. The extra ||e0||^2 ~ h^2 cancels that: K entries ~ k cot^2 (||e0||^2/A0) ~ k, comparable to ARAP's k and
// roughly independent of h. Energy = (k ||e0||^2 / 2 A0) ||sum cot x||^2. Units of k: like the ARAP modulus [N/m], not a bending modulus. Forward only: the backward methods assert. Same contract as SpringEnergy (see
// ElasticEnergies in diffpd_types.h); `pinned` is Object::mesh.pinned_rest.
struct HingeEnergy
{
    using Mat4 = Eigen::Matrix<Real, 4, 4>;

    struct Elem
    {
        SpringKind kind;
        Real       w;        // k * A0 * ||e0||^2
        Real       l0;       // rest ||S||
        ParticleId dof[4];   // hinge edge v0, v1, then tips v2 (triangle 0), v3 (triangle 1); < 0: pinned, index -(dof+1)
        Real       ghat[4];
        Mat4       gg;       // ghat ghat^T
    };
    std::vector<Elem> elems;

    size_t triplet_count() const { return 48 * elems.size(); }

    // One hinge from its vertices' SimMesh::Vertex::dof and 3D rest positions (edge v0-v1, tips v2 and v3).
    // Skipped if all four are pinned.
    // flat_rest: force l0 = 0 (the hinge rests flat whatever the rest geometry, e.g. a cloth cut from a flat sheet).
    void add_hinge(const ParticleId (&dof)[4], const Vec3 (&pos)[4], SpringKind kind, Real k, bool flat_rest = false)
    {
        if (dof[0] < 0 && dof[1] < 0 && dof[2] < 0 && dof[3] < 0) return;

        const Vec3 e0 = pos[1] - pos[0], e1 = pos[2] - pos[0], e2 = pos[3] - pos[0];
        const Vec3 e3 = pos[2] - pos[1], e4 = pos[3] - pos[1];
        auto cot = [](const Vec3& a, const Vec3& b)
        {
            const Real s = a.cross(b).norm();
            ASSERT(s > 1e-12, "hinge: degenerate rest triangle");
            return a.dot(b) / s;
        };
        const Real c02 = cot(e0, e1), c03 = cot(e0, e2), c12 = cot(-e0, e3), c13 = cot(-e0, e4);
        const Real A0  = 0.5 * (e0.cross(e1).norm() + e0.cross(e2).norm());

        Elem e;
        e.kind = kind;
        e.w    = k * A0 * e0.squaredNorm(); // see SCALING above
        const Real g[4] = { -(c12 + c13) / A0, -(c02 + c03) / A0, (c02 + c12) / A0, (c03 + c13) / A0 };
        Vec3 S = Vec3::Zero();
        for (int a = 0; a < 4; ++a)
        {
            e.dof[a]  = dof[a];
            e.ghat[a] = g[a];
            S += g[a] * pos[a];
        }
        e.l0 = flat_rest ? 0.0 : S.norm();
        for (int a = 0; a < 4; ++a)
            for (int c = 0; c < 4; ++c) e.gg(a, c) = g[a] * g[c];
        elems.push_back(e);
    }

    // scale * K, K = sum_e w (ghat ghat^T) (x) I3 on the free dofs.
    void add_stiffness_triplets(std::vector<Triplet>& out, Real scale) const
    {
        for (const Elem& e : elems)
            for (int a = 0; a < 4; ++a)
                for (int c = 0; c < 4; ++c)
                {
                    if (e.dof[a] < 0 || e.dof[c] < 0) continue;
                    for (int d = 0; d < 3; ++d)
                        out.emplace_back(3 * e.dof[a] + d, 3 * e.dof[c] + d, scale * e.w * e.gg(a, c));
                }
    }

    // b += scale * w * (ghat_a p) on the free vertices, minus the constant pinned-vertex coupling w gg_ac x_c.
    void add_forces(const RealVecX& x, const std::vector<Vec3>& pinned, RealVecX& b, Real scale) const
    {
        for (const Elem& e : elems)
        {
            Vec3 X[4];
            for (int a = 0; a < 4; ++a)
                X[a] = e.dof[a] >= 0 ? Vec3(x.segment<3>(3 * e.dof[a])) : pinned[-(e.dof[a] + 1)];

            Vec3 p = Vec3::Zero();
            if (e.l0 > 0.0) // flat rest (l0 == 0): p = 0. shortcut: also fires on rounding noise (p ~ 1e-17), harmless
            {
                const Vec3 S = e.ghat[0] * X[0] + e.ghat[1] * X[1] + e.ghat[2] * X[2] + e.ghat[3] * X[3];
                const Real n = S.norm();
                if (n > 1e-12) p = (e.l0 / n) * S;
            }

            for (int a = 0; a < 4; ++a)
            {
                if (e.dof[a] < 0) continue;
                Vec3 f = e.ghat[a] * p;
                for (int c = 0; c < 4; ++c)
                    if (e.dof[c] < 0) f -= e.gg(a, c) * X[c];
                b.segment<3>(3 * e.dof[a]) += scale * e.w * f;
            }
        }
    }

    void precompute_local_derivative(const RealVecX&, const std::vector<Vec3>&)
    { ASSERT(elems.empty(), "Hinge backward pass is not implemented"); }

    void add_jacobian_product(const RealVecX&, RealVecX&) const
    { ASSERT(elems.empty(), "Hinge backward pass is not implemented"); }

    void add_param_gradient(const RealVecX&, const RealVecX&, const RealVecX&, const std::vector<Vec3>&, Real, Real, Stiffness&) const
    { ASSERT(elems.empty(), "Hinge backward pass is not implemented"); }
};
