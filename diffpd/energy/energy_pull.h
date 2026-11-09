#pragma once

// Constant forces on single particles (the "pulls" of a pattern file, applied at the end of "Settle seams"). A constant force is the linear
// potential -f.x: no stiffness, so nothing goes into K (L and its factorisation do not change) and the only effect is f added to the right-hand
// side, like gravity. `ramp` (0..1, set between steps) scales every force so it can be faded in and out.
// Initialisation only: it is never differentiated, so the three backward methods assert that no force is stored.

#include "core/diffpd_core.h"

struct PullEnergy
{
    std::vector<std::pair<ParticleId, Vec3>> forces; // free particle -> force [N]; entries on the same particle add up
    Real ramp = 0;

    size_t triplet_count() const { return 0; }
    bool   kronecker() const { return true; }

    void add_pull(ParticleId dof, const Vec3& f)
    {
        if (dof >= 0) forces.emplace_back(dof, f); // a pinned vertex cannot move
    }

    void add_stiffness_triplets(std::vector<Triplet>&, Real) const {}

    void add_forces(const RealVecX&, const std::vector<Vec3>&, RealVecX& b, Real scale) const
    {
        if (ramp == 0) return;
        for (const auto& [dof, f] : forces) b.segment<3>(3 * dof) += scale * ramp * f;
    }

    void precompute_local_derivative(const RealVecX&, const std::vector<Vec3>&) { ASSERT(forces.empty(), "PullEnergy is for initialisation only: no backward pass with pulls"); }
    void add_jacobian_product(const RealVecX&, RealVecX&) const { ASSERT(forces.empty(), "PullEnergy is for initialisation only: no backward pass with pulls"); }
    void add_param_gradient(const RealVecX&, const RealVecX&, const RealVecX&, const std::vector<Vec3>&, Real, Real, Stiffness&) const
    {
        ASSERT(forces.empty(), "PullEnergy is for initialisation only: no backward pass with pulls");
    }
};
