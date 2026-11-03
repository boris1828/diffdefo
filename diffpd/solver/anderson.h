#pragma once
#include "core/diffpd_core.h"

// Anderson acceleration (Peng et al. 2018, Alg. 1, beta = 1) of a fixed-point iteration v <- G(v), with a residual
// safeguard in place of the paper's energy check: an AA iterate whose residual did not drop is replaced by the last
// plain iterate and the history is cleared. m = 0: disabled, apply() returns at once.
struct Anderson
{
    Anderson(int m, Index n) : m(m) { if (m > 0) { D.resize(n, m); E.resize(n, m); } }

    // The map G changed (e.g. contacts were added): the stored differences describe another map.
    void reset() { pushed = 0; has_prev = false; last_was_aa = false; }

    // On entry g = G(v) for the current iterate v; on exit g = the next iterate.
    void apply(const RealVecX& v, RealVecX& g)
    {
        if (m == 0) return;
        RealVecX f = g - v; // residual F = G(v) - v
        const Real e = f.norm();
        if (last_was_aa && e >= e_prev) { g = g_prev; reset(); return; } // safeguard
        e_prev = e;
        if (has_prev)
        {
            const int c = pushed++ % m; // ring buffer: column order does not matter for the least squares
            D.col(c) = f - f_prev;
            E.col(c) = g - g_prev;
        }
        f_prev = std::move(f); g_prev = g; has_prev = true;
        const int cols = std::min(pushed, m);
        last_was_aa = cols > 0;
        if (!last_was_aa) return;
        const auto Dk = D.leftCols(cols); // theta = argmin |F - D theta|, normal equations (m x m)
        const Eigen::VectorXd theta = (Dk.transpose() * Dk).completeOrthogonalDecomposition().solve(Dk.transpose() * f_prev);
        g -= E.leftCols(cols) * theta;
    }

private:
    int             m;
    Eigen::MatrixXd D, E;          // last m differences of F and of G
    RealVecX        f_prev, g_prev;
    int             pushed = 0;
    bool            has_prev = false, last_was_aa = false;
    Real            e_prev = 0.0;
};
