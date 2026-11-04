#pragma once

// The global system of Projective Dynamics, shared by every contact method: the elastic force, the system matrix
// L = M + h^2 K (and its damped factor) and the right-hand side of one global step. Nothing here knows about contact.
// Depends on cloth/ only.

#include "cloth/object.h"

// ----------------
//      SOLVER
// ----------------

// Adds every elastic energy's force contribution to `b` in place, scaled by `scale`.
// No allocation: `b` is the caller's buffer, sized to obj.num_dofs().
inline void add_elastic_forces(const Object& obj, const RealVecX& x, RealVecX& b, Real scale = 1.0)
{
    obj.energies.for_each([&](const auto& e) { e.add_forces(x, obj.mesh.pinned_rest, b, scale); });
}

// ----------------
// VELOCITY SOLVER
// ----------------

inline void construct_pd_lhs(Object& obj, Real dt)
{
    const Index n3 = obj.num_dofs();
    const Real  h2 = dt * dt;

    size_t n_elastic = 0;
    obj.energies.for_each([&](const auto& e) { n_elastic += e.triplet_count(); });

    std::vector<Triplet> triplets_L, triplets_C;
    triplets_L.reserve(n3 + n_elastic);
    triplets_C.reserve(     n_elastic);

    // M (mass diagonal, not divided by h²)
    for (Index i = 0; i < n3; ++i)
        triplets_L.emplace_back(i, i, obj.mass(i));

    // h² · K  (elastic contribution)
    obj.energies.for_each([&](const auto& e)
    {
        e.add_stiffness_triplets(triplets_L, h2);
        e.add_stiffness_triplets(triplets_C, h2);
    });

    obj.L.resize(n3, n3);
    obj.L.setFromTriplets(triplets_L.begin(), triplets_L.end());

    obj.C.resize(n3, n3);
    obj.C.setFromTriplets(triplets_C.begin(), triplets_C.end());

    // Laplacian damping: D = alpha*M + beta*K with K = C/h^2, so
    //   C_d = C + h*D = (1 + beta/h)*C + h*alpha*M,   L_d = M + C_d.
    // L and C above stay undamped (they build b_tilde and its derivative); only the solve and the
    // free force f = b_tilde - C_d*v use the damped pair. alpha = beta = 0 gives C_d == C, L_d == L.
    std::vector<Triplet> triplets_M;
    triplets_M.reserve(n3);
    for (Index i = 0; i < n3; ++i)
        triplets_M.emplace_back(i, i, obj.mass(i));
    SparseMat M_sp(n3, n3);
    M_sp.setFromTriplets(triplets_M.begin(), triplets_M.end());

    obj.C_d = (1.0 + obj.damp_beta / dt) * obj.C + (dt * obj.damp_alpha) * M_sp;
    const SparseMat L_d = M_sp + obj.C_d;

    obj.solver = std::make_unique<Cholesky>();
    obj.solver->compute(L_d);
    ASSERT(obj.solver->info() == Eigen::Success, "Cholesky factorization of velocity LHS failed");
}

inline RealVecX construct_velocity_rhs(
    const Object&    obj,
    const RealVecX&  b_inertia,
    const RealVecX&  x_k,
    const Positions& x_minus,
    Real             h)
{
    const Real h2 = h * h;
    RealVecX   b  = b_inertia;
    add_elastic_forces(obj, x_k, b, h2);
    return (1.0 / h) * (b - obj.L * x_minus);
}
