#pragma once

// The non-smooth part of the NCP method: the complementarity preconditioner and the smoothed Fischer-Burmeister residuals with
// the diagonal Newton blocks Omega (d phi / d delta) and E (d phi / d lambda). Pure functions on vectors; rows are laid out like
// NcpContacts::H (m normal rows, then m tangent pairs). Position-level friction: delta_f is a slip displacement [m], rho_f = h^2 W.

#include "core/diffpd_core.h"

// rho_n = h^2 W_jj, rho_f = h^2 (W_f1,f1 + W_f2,f2) / 2: the Delassus diagonal turns a force into a length, so both sides of
// each complementarity equation are lengths.
inline void complementarity_scaling(const Eigen::MatrixXd& W, Real h, RealVecX& rho_n, RealVecX& rho_f)
{
    const Index m = W.rows() / 3;
    rho_n.resize(m);
    rho_f.resize(m);
    for (Index j = 0; j < m; ++j)
    {
        rho_n(j) = h * h * W(j, j);
        rho_f(j) = 0.5 * h * h * (W(m + 2 * j, m + 2 * j) + W(m + 2 * j + 1, m + 2 * j + 1));
    }
}

// phi: the residuals; omega, e: diagonals of Omega and E (3m each; the two tangent rows of a contact share a value).
// Normal: FB(delta, rho lambda) = delta + rho lambda - sqrt(delta^2 + rho^2 lambda^2 + 2 eps^2).
// Friction (paper eq. 55-57): off (phi = lambda_f) unless lambda_n > 0 and mu > 0; on, phi_f = delta_f + e lambda_f, which vanishes
// for stick (delta_f = 0, |lambda_f| < mu lambda_n) and for slip (lambda_f = -mu lambda_n delta_f / |delta_f|).
// shortcut: e's denominator is not positive for |lambda_f| > 2 mu lambda_n, so E can be negative and the Schur matrix indefinite (the
// paper copes with 10 Conjugate Residual iterations); the caller must use an LU, not a Cholesky.
inline void ncp_functions(const RealVecX& delta, const RealVecX& lambda, const RealVecX& rho_n, const RealVecX& rho_f, Real mu, Real two_eps2,
                          RealVecX& phi, RealVecX& omega, RealVecX& e)
{
    const Index m = rho_n.size();
    phi.resize(3 * m);
    omega.resize(3 * m);
    e.resize(3 * m);
    for (Index j = 0; j < m; ++j)
    {
        const Real dn = delta(j), ln = lambda(j), rn = rho_n(j), rf = rho_f(j);
        const Real s  = std::sqrt(dn * dn + rn * rn * ln * ln + two_eps2);
        phi(j)   = dn + rn * ln - s;
        omega(j) = 1.0 - dn / s;
        e(j)     = rn * (1.0 - rn * ln / s);

        const Index f = m + 2 * j;
        const Eigen::Vector2d df(delta(f), delta(f + 1)), lf(lambda(f), lambda(f + 1));
        Real om_f, e_f;
        Eigen::Vector2d phi_f;
        if (ln <= 0.0 || mu == 0.0)
        {
            om_f  = 0.0;
            e_f   = 1.0;
            phi_f = lf;
        }
        else
        {
            const Real a    = df.norm();
            const Real beta = rf * (mu * ln - lf.norm());
            const Real t    = std::sqrt(a * a + beta * beta + two_eps2);
            const Real den  = a + mu * rf * ln - t;
            e_f   = std::abs(den) <= 1e-12 * (a + mu * rf * ln + t) ? 0.0 : rf * (t - beta) / den; // den ~ 0: the stick limit
            om_f  = 1.0;
            phi_f = om_f * df + e_f * lf;
        }
        omega(f) = omega(f + 1) = om_f;
        e(f)     = e(f + 1)     = e_f;
        phi(f)     = phi_f(0);
        phi(f + 1) = phi_f(1);
    }
}
