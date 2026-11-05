#pragma once

// One non-smooth Newton iteration of the constrained local-global step (the body of the k-loop of the paper's Algorithm 3/4),
// in position space, on the contact set of NcpContacts.

#include "solver/ncp/contacts.h"
#include "solver/ncp/delassus.h"
#include "solver/ncp/ncp_functions.h"

// What is fixed for the whole step once the contacts are known.
struct NcpSystem
{
    Eigen::MatrixXd W;      // Delassus operator H L^-1 H^T
    RealVecX rho_n, rho_f;  // complementarity scaling
};

template <class SolveL>
inline NcpSystem make_ncp_system(const SolveL& solve_L, const SparseMat& H, Real h)
{
    NcpSystem s;
    s.W = delassus(solve_L, H);
    complementarity_scaling(s.W, h, s.rho_n, s.rho_f);
    return s;
}

// Size of the last update, for the early exit: dx = |x_new - x| (absolute: the caller picks the scale, x may be ~0), dlambda = |dlambda| / |lambda|.
struct NcpUpdate { Real dx = 0.0, dlambda = 0.0; };

// b: right-hand side of the local step (M x_tilde + h^2 elastic(x)); x, lambda: the iterate, updated in place.
//   J = Omega H;  g = b + h^2 J^T lambda;  eta = J x - phi
//   (h^2 Omega W Omega + E) dlambda = eta - J L^-1 g       (= the paper's (1/h^2)(Omega W Omega + E/h^2)^-1 (eta - J L^-1 g))
//   lambda += dlambda;  x = L^-1 (b + h^2 J^T lambda)
// Omega and E are diagonal, so J is never formed: J v = omega o (H v), J^T v = H^T (omega o v).
template <class SolveL>
inline NcpUpdate ncp_iteration(const SolveL& solve_L, const NcpContacts& c, const NcpSystem& sys, Real two_eps2, Real h,
                               const RealVecX& b, RealVecX& x, RealVecX& lambda)
{
    const Real h2 = h * h;
    const RealVecX Hx    = c.H * x;
    const RealVecX delta = Hx - c.d;
    RealVecX phi, omega, e;
    ncp_functions(delta, lambda, sys.rho_n, sys.rho_f, c.mu, two_eps2, phi, omega, e);

    const RealVecX g   = b + h2 * (c.H.transpose() * omega.cwiseProduct(lambda));
    const RealVecX eta = omega.cwiseProduct(Hx) - phi;
    const RealVecX JLg = omega.cwiseProduct(c.H * solve_L(g));
    const RealVecX dlambda = solve_schur(sys.W, omega, e, h, eta - JLg);

    lambda += dlambda;
    const RealVecX rhs   = b + h2 * (c.H.transpose() * omega.cwiseProduct(lambda));
    const RealVecX x_new = solve_L(rhs);

    NcpUpdate u;
    u.dx      = (x_new - x).norm();
    u.dlambda = dlambda.norm() / std::max(lambda.norm(), Real(1e-12));
    x = x_new;
    return u;
}
