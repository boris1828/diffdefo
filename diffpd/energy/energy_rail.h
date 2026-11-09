#pragma once

// Rail: particles ("rings") that slide on a thin straight bar. Energy per ring E = k/2 (x - a)^T P (x - a), P = I - d d^T (d = unit bar
// direction, a = a point on the bar): a penalty on the offset ACROSS the bar, free along it (no friction, the force is always perpendicular to d).
// Linear, like SeamEnergy: constant stiffness block k P, constant right-hand side k P a, no local step (Gamma = 0), no parameter gradient, so the
// backward pass sees it only through K (L and C). One stiffness k for the whole energy; a load f across the bar moves a ring by f/k.
// P is stored per ring, so another projector (nn^T = plane, I = soft pin) would only need another add_* helper.
// NOT of the form L_s (x) I3: the block is diag(0, k, k) for a bar along x, so the Scalar / ScalarAvx2 global solves are wrong with it (they solve y and z
// with the x block, and their pattern guard cannot see it): kronecker() is false with rings, and construct_pd_lhs then forces GlobalSolve::Full.
// Same energy contract as SpringEnergy (see ElasticEnergies in energy/elastic_energies.h).

#include "core/diffpd_core.h"

struct RailEnergy
{
    struct Ring { ParticleId dof; Mat3 P; Vec3 Pa; }; // Pa = P a, the whole right-hand side of the ring

    std::vector<Ring> rings;
    Real k = 0;

    size_t triplet_count() const { return 9 * rings.size(); }
    bool   kronecker() const { return rings.empty(); } // no rings: contributes nothing, the default solve stays valid

    // dof < 0 (pinned ring) is skipped: a pinned vertex cannot move. `dir` need not be unit length, but must not be zero.
    void add_rail(ParticleId dof, const Vec3& anchor, const Vec3& dir)
    {
        if (dof < 0) return;
        ASSERT(dir.squaredNorm() > 0.0, "RailEnergy: zero bar direction");
        const Vec3 d = dir.normalized();
        const Mat3 P = Mat3::Identity() - d * d.transpose();
        rings.push_back({ dof, P, P * anchor });
    }

    void add_stiffness_triplets(std::vector<Triplet>& out, Real scale) const
    {
        for (const Ring& r : rings)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    out.emplace_back(3 * r.dof + i, 3 * r.dof + j, scale * k * r.P(i, j));
    }

    void add_forces(const RealVecX&, const std::vector<Vec3>&, RealVecX& b, Real scale) const
    {
        for (const Ring& r : rings) b.segment<3>(3 * r.dof) += scale * k * r.Pa;
    }
    void precompute_local_derivative(const RealVecX&, const std::vector<Vec3>&) {}
    void add_jacobian_product(const RealVecX&, RealVecX&) const {}
    void add_param_gradient(const RealVecX&, const RealVecX&, const RealVecX&, const std::vector<Vec3>&, Real, Real, Stiffness&) const {}
};
