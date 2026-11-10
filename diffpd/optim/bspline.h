#pragma once

// A hand path as a clamped uniform cubic B-spline: K control points over the horizon, p = B theta (+ the fixed start), so the gradient
// pull-back is B^T. Depends only on core/. Positions are in diffpd axes, one per track frame (feed them to make_translation_track).

#include "core/diffpd_core.h"

struct BSplinePath
{
    int             K;
    Vec3            start;     // control point 0: not a parameter, every path begins here
    Vec3            free_axes; // 1 = this axis of control points 1..K-1 is a parameter, 0 = it stays at start (e.g. z: motion in the curtain's plane)
    Eigen::MatrixXd B;         // frames x K, rows sum to 1; row 0 = e_0 and the last row = e_{K-1} (clamped ends)
    std::vector<Vec3> base;    // control points as they are on the masked axes (default: all at start); overwrite to hold a masked axis at a known path instead

    BSplinePath(int K_, int frames, const Vec3& start_, const Vec3& free_axes_ = Vec3(1.0, 1.0, 0.0)) : K(K_), start(start_), free_axes(free_axes_)
    {
        ASSERT(K >= 4 && frames >= 2, "BSplinePath: a cubic needs K >= 4 control points and 2 frames");
        base.assign(K, start);
        std::vector<Real> U(K + 4, 0.0); // open uniform knots: 4 at each end, K - 4 equally spaced inside
        for (int i = 0; i < K + 4; ++i) U[i] = i < 4 ? 0.0 : (i >= K ? 1.0 : Real(i - 3) / (K - 3));
        B = Eigen::MatrixXd::Zero(frames, K);
        for (int k = 0; k < frames; ++k)
        {
            const Real t = Real(k) / (frames - 1);
            if (k == frames - 1) { B(k, K - 1) = 1.0; continue; } // the half-open spans of the recursion miss t = 1
            for (int j = 0; j < K; ++j) B(k, j) = basis(U, j, 3, t);
        }
    }

    int num_free() const { return (int)(free_axes[0] != 0.0) + (int)(free_axes[1] != 0.0) + (int)(free_axes[2] != 0.0); }
    int num_params() const { return (K - 1) * num_free(); }

    // theta from K control points (c[0] is ignored: it is `start`); layout: control point major, free axes in x, y, z order
    RealVecX theta_from_controls(const std::vector<Vec3>& c) const
    {
        RealVecX theta(num_params());
        int n = 0;
        for (int j = 1; j < K; ++j)
            for (int a = 0; a < 3; ++a)
                if (free_axes[a] != 0.0) theta[n++] = c[j][a];
        return theta;
    }

    std::vector<Vec3> positions(const RealVecX& theta) const
    {
        ASSERT(theta.size() == num_params(), "BSplinePath::positions: theta has " << theta.size() << " entries, expected " << num_params());
        std::vector<Vec3> c = base; // masked axes keep their base value
        c[0] = start;
        int n = 0;
        for (int j = 1; j < K; ++j)
            for (int a = 0; a < 3; ++a)
                if (free_axes[a] != 0.0) c[j][a] = theta[n++];
        std::vector<Vec3> p(B.rows(), Vec3::Zero());
        for (int k = 0; k < (int)B.rows(); ++k)
            for (int j = 0; j < K; ++j) p[k] += B(k, j) * c[j];
        return p;
    }

    // dL/dtheta = B^T dL/dp, from the gradient per frame in the path's own (diffpd) axes; masked axes drop out
    RealVecX pullback(const std::vector<Vec3>& dL_dp) const
    {
        ASSERT((int)dL_dp.size() == B.rows(), "BSplinePath::pullback: " << dL_dp.size() << " frames, expected " << B.rows());
        RealVecX g = RealVecX::Zero(num_params());
        for (int k = 0; k < (int)B.rows(); ++k)
        {
            int n = 0;
            for (int j = 1; j < K; ++j)
                for (int a = 0; a < 3; ++a)
                    if (free_axes[a] != 0.0) g[n++] += B(k, j) * dL_dp[k][a];
        }
        return g;
    }

private:
    // Cox-de Boor: value of the degree-p basis function i of knot vector U at t
    static Real basis(const std::vector<Real>& U, int i, int p, Real t)
    {
        if (p == 0) return (U[i] <= t && t < U[i + 1]) ? 1.0 : 0.0;
        Real a = 0.0, b = 0.0;
        if (U[i + p] > U[i])         a = (t - U[i]) / (U[i + p] - U[i]) * basis(U, i, p - 1, t);
        if (U[i + p + 1] > U[i + 1]) b = (U[i + p + 1] - t) / (U[i + p + 1] - U[i + 1]) * basis(U, i + 1, p - 1, t);
        return a + b;
    }
};
