#pragma once

// One non-smooth Newton iteration of the constrained local-global step (the body of the k-loop of the paper's Algorithm 3/4),
// in position space, on the contact set of NcpContacts.

#include "solver/ncp/contacts.h"
#include "solver/ncp/delassus.h"
#include "solver/ncp/ncp_functions.h"

// What is fixed for the whole step once the contacts are known.
struct NcpSystem
{
    Eigen::MatrixXd W;      // Delassus operator H L^-1 H^T (empty in mass-split mode)
    RealVecX rho_n, rho_f;  // complementarity scaling
    const SparseMat* A = nullptr; // mass-split mode when set: the system matrix M + h^2 K (obj.L), for the residual r = g - A x
    RealVecX minv, w;       // mass-split mode: 1/mass per dof; diag(H M^-1 H^T) per row
};

// Mass-split Schur complement (Zeng et al. SA Posters '25): W = H M^-1 H^T. One contact per particle and orthonormal rows make it
// exactly diagonal, so there is nothing to factor. `obj` must outlive the returned system (it keeps a pointer to obj.L).
inline NcpSystem make_ncp_system_diag(const Object& obj, const SparseMat& H, Real h)
{
    NcpSystem s;
    s.A    = &obj.L;
    s.minv = obj.mass.cwiseInverse();
    const SparseMat H2 = H.cwiseAbs2();
    s.w    = H2 * s.minv;
    const Index m = H.rows() / 3;
    s.rho_n.resize(m);
    s.rho_f.resize(m);
    for (Index j = 0; j < m; ++j)
    {
        s.rho_n(j) = h * h * s.w(j);
        s.rho_f(j) = 0.5 * h * h * (s.w(m + 2 * j) + s.w(m + 2 * j + 1));
    }
    return s;
}

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

    const RealVecX g = b + h2 * (c.H.transpose() * omega.cwiseProduct(lambda));
    RealVecX dlambda;
    if (sys.A) // mass split: J A^-1 g ~ J (x + M^-1 r) with r = g - A x, so  (h^2 w omega^2 + e) dlambda = -phi - omega H M^-1 r
    {
        const RealVecX r   = g - (*sys.A) * x;
        const RealVecX rhs = -phi - omega.cwiseProduct(c.H * sys.minv.cwiseProduct(r));
        dlambda.resize(rhs.size());
        for (Index i = 0; i < rhs.size(); ++i)
        {
            Real den = h2 * omega(i) * omega(i) * sys.w(i) + e(i);
            if (std::abs(den) < 1e-12) den = 1e-12; // shortcut: the pole of E_f (e << 0) can cancel the diagonal; the dense LU pivoted, this only avoids 0/0
            dlambda(i) = rhs(i) / den;
        }
    }
    else
    {
        const RealVecX eta = omega.cwiseProduct(Hx) - phi;
        const RealVecX JLg = omega.cwiseProduct(c.H * solve_L(g));
        dlambda = solve_schur(sys.W, omega, e, h, eta - JLg);
    }

    lambda += dlambda;
    const RealVecX rhs   = b + h2 * (c.H.transpose() * omega.cwiseProduct(lambda));
    const RealVecX x_new = solve_L(rhs);

    NcpUpdate u;
    u.dx      = (x_new - x).norm();
    u.dlambda = dlambda.norm() / std::max(lambda.norm(), Real(1e-12));
    x = x_new;
    return u;
}
