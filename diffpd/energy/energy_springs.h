#pragma once

#include "core/diffpd_core.h"

// ----------------
//     SPRINGS
// ----------------

// Distance springs, the first elastic energy behind the contract described above ElasticEnergies
// (diffpd_types.h). `pinned` is always Object::mesh.pinned_rest.

enum class SpringType : std::uint8_t
{
    Spring2,   // both endpoints are free particles
    Spring1,   // one free particle anchored to a pinned vertex
};

struct Constraint
{
    SpringType type;
    SpringKind kind;
    Real       k; // stiffness
    Real       l; // rest length
    Mat3       gamma;  // local Jacobian factor: (k*l/||e||) * P_perp
    Vec3       p_star; // normalized spring correction: l * (e / ||e||)
    Vec3       e;      // edge vector

    union
    {
        struct
        {
            ParticleId i1;
            ParticleId i2;
        } spring2;

        struct
        {
            ParticleId i;
            Index      pinned_index; // index into Object::mesh.pinned_rest: the anchor's position
        } spring1;
    };

    static Constraint make(SpringType type, SpringKind kind, Real k, Real l)
    {
        Constraint c;
        c.type   = type;
        c.kind   = kind;
        c.k      = k;
        c.l      = l;
        c.gamma  = Mat3::Zero();
        c.p_star = Vec3::Zero();
        c.e      = Vec3::Zero();
        return c;
    }

    static Constraint makeSpring2(SpringKind kind, Real k, Real l, ParticleId i1, ParticleId i2)
    {
        Constraint c = make(SpringType::Spring2, kind, k, l);
        c.spring2.i1 = i1;
        c.spring2.i2 = i2;
        return c;
    }

    static Constraint makeSpring1(SpringKind kind, Real k, Real l, ParticleId i, Index pinned_index)
    {
        Constraint c = make(SpringType::Spring1, kind, k, l);
        c.spring1.i = i;
        c.spring1.pinned_index = pinned_index;
        return c;
    }
};

using Constraints = std::vector<Constraint>;

struct SpringEnergy
{
    Constraints constraints;

    size_t triplet_count() const { return 12 * constraints.size(); }

    // One spring between two mesh vertices, given their SimMesh::Vertex::dof (>= 0: particle id, < 0: pinned,
    // pinned_rest index -(dof+1)) and rest positions. Skipped if both are pinned.
    void add_edge(ParticleId dof_a, ParticleId dof_b, const Vec3& pos_a, const Vec3& pos_b, SpringKind kind, Real k)
    {
        const Real l  = (pos_a - pos_b).norm();
        const bool pa = dof_a < 0, pb = dof_b < 0;
        if (pa && pb) return;
        if (!pa && !pb)
        {
            constraints.push_back(Constraint::makeSpring2(kind, k, l, dof_a, dof_b));
        }
        else
        {
            const ParticleId free   = pa ? dof_b : dof_a;
            const ParticleId anchor = pa ? dof_a : dof_b;
            constraints.push_back(Constraint::makeSpring1(kind, k, l, free, Index(-(anchor + 1))));
        }
    }

    // scale * K, K = sum_i k_i G_i^T G_i on the free dofs, in emission order.
    void add_stiffness_triplets(std::vector<Triplet>& out, Real scale) const
    {
        for (const Constraint& c : constraints)
        {
            if (c.type == SpringType::Spring2)
            {
                const Index i1 = c.spring2.i1;
                const Index i2 = c.spring2.i2;
                for (int d = 0; d < 3; ++d)
                {
                    out.emplace_back(3*i1+d, 3*i1+d, +scale * c.k);
                    out.emplace_back(3*i2+d, 3*i2+d, +scale * c.k);
                    out.emplace_back(3*i1+d, 3*i2+d, -scale * c.k);
                    out.emplace_back(3*i2+d, 3*i1+d, -scale * c.k);
                }
            }
            else // Spring1
            {
                const Index i = c.spring1.i;
                for (int d = 0; d < 3; ++d)
                    out.emplace_back(3*i+d, 3*i+d, +scale * c.k);
            }
        }
    }

    // Adds each constraint's elastic force contribution to `b` in place, scaled by `scale`.
    void add_forces(const RealVecX& x, const std::vector<Vec3>& pinned, RealVecX& b, Real scale) const
    {
        for (const Constraint& c : constraints)
        {
            if (c.type == SpringType::Spring2)
            {
                const Index i1 = c.spring2.i1;
                const Index i2 = c.spring2.i2;
                const Vec3 e   = x.segment<3>(3*i1) - x.segment<3>(3*i2);
                const Vec3 p   = c.l * e / e.norm();
                b.segment<3>(3*i1) += scale * c.k * p;
                b.segment<3>(3*i2) -= scale * c.k * p;
            }
            else // SpringType::Spring1
            {
                const Vec3& xbar = pinned[c.spring1.pinned_index];
                const Index i    = c.spring1.i;
                const Vec3 e     = x.segment<3>(3*i) - xbar;
                const Vec3 p     = c.l * e / e.norm();
                b.segment<3>(3*i) += scale * c.k * (xbar + p);
            }
        }
    }

    // Caches e, p_star and gamma = k * dp/de at x (= x+ of the step being differentiated).
    void precompute_local_derivative(const RealVecX& x, const std::vector<Vec3>& pinned)
    {
        for (Constraint& c : constraints)
        {
            Vec3 e;

            if (c.type == SpringType::Spring2)
            {
                const Index i1 = c.spring2.i1;
                const Index i2 = c.spring2.i2;
                e = x.segment<3>(3*i1) - x.segment<3>(3*i2);
            }
            else // SpringType::Spring1
            {
                const Index i = c.spring1.i;
                e = x.segment<3>(3*i) - pinned[c.spring1.pinned_index];
            }

            const Real e_norm = e.norm();

            if (e_norm < 1e-9)
            {
                c.gamma  = Mat3::Zero();
                c.p_star = Vec3::Zero();
                c.e      = Vec3::Zero();
                continue;
            }

            const Vec3 e_hat  = e / e_norm;
            const Mat3 P_perp = Mat3::Identity() - (e_hat * e_hat.transpose());

            c.e      = e;
            c.p_star = c.l * e_hat;
            c.gamma  = (c.k * c.l / e_norm) * P_perp;
        }
    }

    // out += (sum_i G_i^T Gamma_i G_i) v. Needs precompute_local_derivative for this step.
    void add_jacobian_product(const RealVecX& v, RealVecX& out) const
    {
        for (const Constraint& c : constraints)
        {
            if (c.type == SpringType::Spring2)
            {
                const Index i1 = c.spring2.i1;
                const Index i2 = c.spring2.i2;
                const Vec3 d   = c.gamma * (v.segment<3>(3*i1) - v.segment<3>(3*i2));
                out.segment<3>(3*i1) += d;
                out.segment<3>(3*i2) -= d;
            }
            else // SpringType::Spring1
            {
                const Index i = c.spring1.i;
                out.segment<3>(3*i) += c.gamma * v.segment<3>(3*i);
            }
        }
    }

    // grad[kind] += dphi/dk for this step. Needs precompute_local_derivative (p_star) for this step.
    void add_param_gradient(const RealVecX& z_perp, const RealVecX& x_minus, const RealVecX& v_plus,
                            const std::vector<Vec3>& pinned, Real h, Real h2_eff, Stiffness& grad) const
    {
        for (const Constraint& c : constraints)
        {
            if (c.type == SpringType::Spring2)
            {
                const Index i1 = c.spring2.i1;
                const Index i2 = c.spring2.i2;

                const Vec3 e_minus = x_minus.segment<3>(3*i1) - x_minus.segment<3>(3*i2);
                const Vec3 dv      = v_plus.segment<3>(3*i1)  - v_plus.segment<3>(3*i2);
                const Vec3 dz      = z_perp.segment<3>(3*i1)  - z_perp.segment<3>(3*i2);

                grad[c.kind] += dz.dot(h * (c.p_star - e_minus) - h2_eff * dv);
            }
            else // Spring1
            {
                const Index i = c.spring1.i;

                const Vec3 e_minus = x_minus.segment<3>(3*i) - pinned[c.spring1.pinned_index];
                const Vec3 vi      = v_plus.segment<3>(3*i);
                const Vec3 zi      = z_perp.segment<3>(3*i);

                grad[c.kind] += zi.dot(h * (c.p_star - e_minus) - h2_eff * vi);
            }
        }
    }
};
