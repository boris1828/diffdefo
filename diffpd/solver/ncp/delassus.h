#pragma once

// The dense pieces of the constraint (Schur) system: the Delassus operator W = H L^-1 H^T and the solve of
// h^2 Omega W Omega + E. `solve_L` is any callable applying L^-1 to a dense matrix or vector (a Cholesky factor of the
// undamped L = M + h^2 K, see cholesky_solver).

#include "cloth/object.h"
#include "core/diffpd_core.h"

#include <type_traits>

// L^-1 through obj.solver. W = H L^-1 H^T is the paper's, so L must be the undamped matrix: with damping obj.solver factors L_d instead.
inline auto cholesky_solver(const Object& obj)
{
    ASSERT(obj.solver, "NCP: the system matrix is not factored (construct_pd_lhs first)");
    ASSERT(obj.damp_alpha == 0.0 && obj.damp_beta == 0.0, "NCP: damping is not supported (obj.solver factors the damped matrix)");
    return [&obj](const auto& B)
    {
        using T = std::decay_t<decltype(B)>;
        return T(obj.solver->solve(B));
    };
}

// shortcut: dense 3m x 3m, 3m solves; fine for hundreds of contacts. Upgrade: the sparse inverse S = L^-1 (paper sec. 4) with W = H S^T S H^T.
template <class SolveL>
inline Eigen::MatrixXd delassus(const SolveL& solve_L, const SparseMat& H)
{
    const Eigen::MatrixXd Ht = Eigen::MatrixXd(H.transpose());
    const Eigen::MatrixXd X  = solve_L(Ht);
    const Eigen::MatrixXd W  = H * X;
    return 0.5 * (W + W.transpose()); // symmetric by construction; removes round-off
}

// (h^2 Omega W Omega + E) x = rhs.
// shortcut: dense PartialPivLU, O((3m)^3) per iteration because Omega and E change every iteration, so nothing is factored once. It is
// used instead of LDLT because the matrix can be indefinite (see ncp_functions). Alternatives: LDLT while E >= 0, or the paper's
// Conjugate Residual on the matrix-free product (W is dense but its action is one Cholesky solve of L).
inline RealVecX solve_schur(const Eigen::MatrixXd& W, const RealVecX& omega, const RealVecX& e, Real h, const RealVecX& rhs)
{
    Eigen::MatrixXd D = (h * h) * W.cwiseProduct(omega * omega.transpose());
    D.diagonal() += e;
    return D.partialPivLu().solve(rhs);
}
