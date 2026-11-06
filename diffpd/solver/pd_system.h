#pragma once

// The global system of Projective Dynamics, shared by every contact method: the elastic force, the system matrix
// L = M + h^2 K (and its damped factor) and the right-hand side of one global step. Nothing here knows about contact.
// Depends on cloth/ only.

#include "cloth/object.h"
#include "solver/llt_avx2.h"

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

    obj.solver_scalar.reset();
    obj.scalar_perm.clear(); obj.scalar_dinv.clear();
    if (obj.global_solve == GlobalSolve::ScalarAvx2 && !avx2_fma_available())
    {
        static bool warned = false;
        if (!warned) WARNING("global solve Scalar AVX2: this CPU/build has no AVX2+FMA kernel, using Scalar");
        warned = true;
        obj.global_solve = GlobalSolve::Scalar;
    }
    if (obj.global_solve != GlobalSolve::Full)
    {
        for (Index i = 0; i < n3 / 3; ++i)
            ASSERT(obj.mass(3*i) == obj.mass(3*i + 1) && obj.mass(3*i) == obj.mass(3*i + 2), "global solve Scalar: anisotropic mass at particle " << i << "; use Full");
        // L_d = L_s (x) I3: take the x-coordinate block (rows and columns 3i). Every energy so far has this form (--global-solve-check).
        std::vector<Triplet> ts;
        for (int k = 0; k < L_d.outerSize(); ++k)
            for (SparseMat::InnerIterator it(L_d, k); it; ++it)
                if (it.row() % 3 == 0 && it.col() % 3 == 0) ts.emplace_back(it.row() / 3, it.col() / 3, it.value());
        SparseMat L_s(n3 / 3, n3 / 3);
        L_s.setFromTriplets(ts.begin(), ts.end());
        ASSERT(L_d.nonZeros() == 3 * L_s.nonZeros(), "global solve Scalar: the system couples coordinates (not L_s (x) I3); use Full"); // shortcut: checks the pattern and the mass, not that the three stiffness blocks are equal
        obj.solver_scalar = std::make_unique<Cholesky>();
        obj.solver_scalar->compute(L_s);
        ASSERT(obj.solver_scalar->info() == Eigen::Success, "Cholesky factorization of the scalar block failed");
        const SparseMat& F = obj.solver_scalar->matrixL().nestedExpression();
        ASSERT(F.isCompressed(), "scalar factor is not compressed");
        for (Index j = 0; j < F.cols(); ++j) ASSERT(F.innerIndexPtr()[F.outerIndexPtr()[j]] == j, "scalar factor: diagonal is not first in column " << j);
        if (obj.global_solve == GlobalSolve::ScalarAvx2)
        {
            const auto& P = obj.solver_scalar->permutationP().indices(); // (P b)[P[i]] = b[i]
            obj.scalar_perm.assign(P.data(), P.data() + P.size());
            obj.scalar_dinv.resize(F.cols());
            for (Index j = 0; j < F.cols(); ++j) obj.scalar_dinv[j] = 1.0 / F.valuePtr()[F.outerIndexPtr()[j]];
        }
    }
}

// Solves y = L_d^-1 g with the factor of K right-hand sides stored row-major (N x K): the triangular factor is walked once for all K
// columns (3 separate solves walk it 3 times, Eigen's own multi-column solve does too). y holds P*g and is overwritten by L^-T L^-1 y.
template <int K>
inline void llt_solve_inplace(const SparseMat& L, Real* y)
{
    const Index N = L.cols();
    const int* outer = L.outerIndexPtr(); const int* inner = L.innerIndexPtr(); const Real* val = L.valuePtr();
    for (Index j = 0; j < N; ++j) // L y = b
    {
        const int p0 = outer[j];
        Real yj[K]; const Real inv = 1.0 / val[p0];
        for (int c = 0; c < K; ++c) { yj[c] = y[j*K + c] * inv; y[j*K + c] = yj[c]; }
        for (int p = p0 + 1; p < outer[j+1]; ++p)
        {
            const Real l = val[p]; Real* yi = y + (Index)inner[p] * K;
            for (int c = 0; c < K; ++c) yi[c] -= l * yj[c];
        }
    }
    for (Index j = N - 1; j >= 0; --j) // L^T x = y
    {
        const int p0 = outer[j];
        Real acc[K]; for (int c = 0; c < K; ++c) acc[c] = y[j*K + c];
        for (int p = p0 + 1; p < outer[j+1]; ++p)
        {
            const Real l = val[p]; const Real* xi = y + (Index)inner[p] * K;
            for (int c = 0; c < K; ++c) acc[c] -= l * xi[c];
        }
        const Real inv = 1.0 / val[p0];
        for (int c = 0; c < K; ++c) y[j*K + c] = acc[c] * inv;
    }
}

// The global solve of every PD iteration (forward and adjoint): L_d^-1 g, by obj.global_solve.
inline RealVecX solve_global(const Object& obj, const RealVecX& g)
{
    if (obj.global_solve == GlobalSolve::Full) return obj.solver->solve(g);
#if DIFFPD_AVX2_KERNEL
    if (obj.global_solve == GlobalSolve::ScalarAvx2) // construct_pd_lhs only leaves this mode when the CPU can run it
    {
        const Index N = g.size() / 3;
        Eigen::Matrix<Real, Eigen::Dynamic, 4, Eigen::RowMajor> y(N, 4); // rows (x y z pad), permuted
        for (Index o = 0; o < N; ++o) { Real* d = y.data() + 4 * obj.scalar_perm[o]; d[0] = g[3*o]; d[1] = g[3*o + 1]; d[2] = g[3*o + 2]; d[3] = 0.0; }
        llt_solve_inplace_avx2(obj.solver_scalar->matrixL().nestedExpression(), obj.scalar_dinv.data(), y.data());
        RealVecX x(g.size());
        for (Index o = 0; o < N; ++o) { const Real* s = y.data() + 4 * obj.scalar_perm[o]; x[3*o] = s[0]; x[3*o + 1] = s[1]; x[3*o + 2] = s[2]; }
        return x;
    }
#endif
    using RowMajor3 = Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>;
    const Cholesky& ch = *obj.solver_scalar;
    RowMajor3 y = ch.permutationP() * Eigen::Map<const RowMajor3>(g.data(), g.size() / 3, 3); // g = (x0 y0 z0 x1 ...) is N x 3 row-major already
    llt_solve_inplace<3>(ch.matrixL().nestedExpression(), y.data());
    y = ch.permutationPinv() * y;
    return Eigen::Map<const RealVecX>(y.data(), g.size());
}

// L_x_minus = obj.L * x_minus: constant over the iterations of a step (x_minus is the position at its start), so the caller computes it once.
inline RealVecX construct_velocity_rhs(
    const Object&    obj,
    const RealVecX&  b_inertia,
    const RealVecX&  x_k,
    const RealVecX&  L_x_minus,
    Real             h)
{
    const Real h2 = h * h;
    RealVecX   b  = b_inertia;
    add_elastic_forces(obj, x_k, b, h2);
    return (1.0 / h) * (b - L_x_minus);
}
