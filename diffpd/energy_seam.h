#pragma once

// Sewing: rest-length-0 springs between pairs of free particles (energy k/2 ||x_a - x_b||^2, linear, so the stiffness matrix stays constant
// and there is no local step). Same energy contract as SpringEnergy (see ElasticEnergies in diffpd_types.h), but the stiffness is a fixed
// constant, not a parameter: no SpringKind slot, no dphi/dk term. The adjoint still sees it through K (L and C), so the gradients of the
// other parameters stay exact. "Settle seams" (settle_cloth) first closes the links gently: energy k/2 ||x_a - x_b - pull*d0||^2 with d0 the
// initial offset and pull going 1 -> 0, still linear (only the right-hand side moves, K is unchanged); pull = 0 in every normal run. Pinned vertices are not supported (shortcut: a link with a pinned end is dropped; the pattern cloth has no pins).

#include "diffpd_core.h"

struct SeamEnergy
{
    std::vector<std::pair<ParticleId, ParticleId>> links;
    Real k = 0;
    std::vector<Vec3> offset; // per link, d0 = x_a - x_b when set_offsets() was called (settle seams); empty = none
    Real pull = 0;            // fraction of d0 the links still hold open, 1 -> 0

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

    void set_offsets(const RealVecX& x)
    {
        offset.resize(links.size());
        for (size_t i = 0; i < links.size(); ++i) offset[i] = x.segment<3>(3 * links[i].first) - x.segment<3>(3 * links[i].second);
    }

    Real max_gap(const RealVecX& x) const // longest link
    {
        Real g = 0;
        for (const auto& [a, b] : links) g = std::max(g, (x.segment<3>(3 * a) - x.segment<3>(3 * b)).norm());
        return g;
    }

    // rest vector pull*d0 (0 in a normal run): p does not depend on x, so Gamma = 0 and the backward pass has nothing to add
    void add_forces(const RealVecX&, const std::vector<Vec3>&, RealVecX& b, Real scale) const
    {
        if (pull == 0 || offset.empty()) return;
        for (size_t i = 0; i < links.size(); ++i)
        {
            const Vec3 f = scale * k * pull * offset[i];
            b.segment<3>(3 * links[i].first)  += f;
            b.segment<3>(3 * links[i].second) -= f;
        }
    }
    void precompute_local_derivative(const RealVecX&, const std::vector<Vec3>&) {}
    void add_jacobian_product(const RealVecX&, RealVecX&) const {}
    void add_param_gradient(const RealVecX&, const RealVecX&, const RealVecX&, const std::vector<Vec3>&, Real, Real, Stiffness&) const {}
};
