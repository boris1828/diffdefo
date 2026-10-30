#pragma once

// Sewing: rest-length-0 springs between pairs of free particles (energy k/2 ||x_a - x_b||^2, linear, so the stiffness matrix stays constant
// and there is no local step). Same energy contract as SpringEnergy (see ElasticEnergies in diffpd_types.h), but the stiffness is a fixed
// constant, not a parameter: no SpringKind slot, no dphi/dk term. The adjoint still sees it through K (L and C), so the gradients of the
// other parameters stay exact.
// Every link has a level 1..4 (pattern file `level`). "Settle seams" (settle_cloth) closes them level by level: stage L keeps only the links
// of level <= L (`max_level`, left out of K and of the force above it; the matrix is rebuilt per stage) and closes the links of level `pull_level`
// gently: energy k/2 ||x_a - x_b - pull*d0||^2 with d0 the offset when the stage started and pull going 1 -> 0, still linear (only the
// right-hand side moves, K is unchanged). The links of the earlier levels rest at 0 and hold. pull = 0 and max_level = 4 in every normal run.
// Pinned vertices are not supported (shortcut: a link with a pinned end is dropped; the pattern cloth has no pins).

#include "core/diffpd_core.h"

struct SeamEnergy
{
    std::vector<std::pair<ParticleId, ParticleId>> links;
    std::vector<int> level;   // per link, 1..4 (parallel to links)
    Real k = 0;
    std::vector<Vec3> offset; // per link, d0 = x_a - x_b when set_offsets() was called (settle seams); empty = none
    Real pull = 0;            // fraction of d0 the links of `pull_level` still hold open, 1 -> 0
    int  pull_level = 0;      // 0 = none
    int  max_level = 4;       // links of a higher level are not part of the energy

    size_t triplet_count() const { return 12 * links.size(); }

    void add_link(ParticleId a, ParticleId b, int link_level = 1)
    {
        if (a < 0 || b < 0 || a == b) return;
        links.emplace_back(a, b);
        level.push_back(link_level);
    }

    int highest_level() const // 0 = no links
    {
        int m = 0;
        for (int l : level) m = std::max(m, l);
        return m;
    }

    void add_stiffness_triplets(std::vector<Triplet>& out, Real scale) const
    {
        for (size_t i = 0; i < links.size(); ++i)
        {
            if (level[i] > max_level) continue;
            const auto [a, b] = links[i];
            for (int d = 0; d < 3; ++d)
            {
                out.emplace_back(3*a+d, 3*a+d, +scale * k);
                out.emplace_back(3*b+d, 3*b+d, +scale * k);
                out.emplace_back(3*a+d, 3*b+d, -scale * k);
                out.emplace_back(3*b+d, 3*a+d, -scale * k);
            }
        }
    }

    void set_offsets(const RealVecX& x)
    {
        offset.resize(links.size());
        for (size_t i = 0; i < links.size(); ++i) offset[i] = x.segment<3>(3 * links[i].first) - x.segment<3>(3 * links[i].second);
    }

    Real max_gap(const RealVecX& x, int up_to_level = 4) const // longest link of level <= up_to_level
    {
        Real g = 0;
        for (size_t i = 0; i < links.size(); ++i)
            if (level[i] <= up_to_level) g = std::max(g, (x.segment<3>(3 * links[i].first) - x.segment<3>(3 * links[i].second)).norm());
        return g;
    }

    // rest vector pull*d0 for the links of pull_level (0 in a normal run): p does not depend on x, so Gamma = 0 and the backward pass has nothing to add
    void add_forces(const RealVecX&, const std::vector<Vec3>&, RealVecX& b, Real scale) const
    {
        if (pull == 0 || offset.empty()) return;
        for (size_t i = 0; i < links.size(); ++i)
        {
            if (level[i] != pull_level || level[i] > max_level) continue;
            const Vec3 f = scale * k * pull * offset[i];
            b.segment<3>(3 * links[i].first)  += f;
            b.segment<3>(3 * links[i].second) -= f;
        }
    }
    void precompute_local_derivative(const RealVecX&, const std::vector<Vec3>&) {}
    void add_jacobian_product(const RealVecX&, RealVecX&) const {}
    void add_param_gradient(const RealVecX&, const RealVecX&, const RealVecX&, const std::vector<Vec3>&, Real, Real, Stiffness&) const {}
};
