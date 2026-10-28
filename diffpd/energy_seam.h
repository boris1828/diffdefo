#pragma once

// Sewing: rest-length-0 springs between pairs of free particles (energy k/2 ||x_a - x_b||^2, linear, so the stiffness matrix stays constant
// and there is no local step). Same energy contract as SpringEnergy (see ElasticEnergies in diffpd_types.h), but the stiffness is a fixed
// constant, not a parameter: no SpringKind slot, no dphi/dk term. The adjoint still sees it through K (L and C), so the gradients of the
// other parameters stay exact. Pinned vertices are not supported (shortcut: a link with a pinned end is dropped; the pattern cloth has no pins).

#include "diffpd_core.h"

struct SeamEnergy
{
    std::vector<std::pair<ParticleId, ParticleId>> links;
    Real k = 0;

    size_t triplet_count() const { return 12 * links.size(); }

    void add_link(ParticleId a, ParticleId b)
    {
        if (a >= 0 && b >= 0 && a != b) links.emplace_back(a, b);
    }

    void add_stiffness_triplets(std::vector<Triplet>& out, Real scale) const
    {
        for (const auto& [a, b] : links)
            for (int d = 0; d < 3; ++d)
            {
                out.emplace_back(3*a+d, 3*a+d, +scale * k);
                out.emplace_back(3*b+d, 3*b+d, +scale * k);
                out.emplace_back(3*a+d, 3*b+d, -scale * k);
                out.emplace_back(3*b+d, 3*a+d, -scale * k);
            }
    }

    // rest length 0: p = 0, Gamma = 0, nothing to add
    void add_forces(const RealVecX&, const std::vector<Vec3>&, RealVecX&, Real) const {}
    void precompute_local_derivative(const RealVecX&, const std::vector<Vec3>&) {}
    void add_jacobian_product(const RealVecX&, RealVecX&) const {}
    void add_param_gradient(const RealVecX&, const RealVecX&, const RealVecX&, const std::vector<Vec3>&, Real, Real, Stiffness&) const {}
};
