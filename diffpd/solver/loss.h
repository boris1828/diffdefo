#pragma once

// Mean squared position difference between a guess and a target tape, and its gradient per frame.

#include "solver/tape.h"

struct Loss
{
    Real                  total = 0.0;
    std::vector<RealVecX> dloss_dx;

    Loss() = default; // for derived losses that fill total / dloss_dx themselves

    Loss(const Tape& guess, const Tape& target, int sample_every, int target_first = 0) // target_first: target frame that guess frame 0 is compared with (a chunk)
    {
        ASSERT((int)guess.positions.size() + target_first <= (int)target.positions.size(),
               "tape size mismatch: " << guess.positions.size() << " (+ offset " << target_first << ") vs " << target.positions.size());
        ASSERT(sample_every > 0, "sample_every must be positive");

        const int   n_frames = (int)guess.positions.size();
        const int   n_steps  = n_frames - 1;
        const Index dofs     = 3 * guess.positions[0].rows();

        total = 0.0;
        dloss_dx.resize(n_frames, RealVecX::Zero(dofs));

        for (int t = 0; t < n_frames; ++t)
        {
            const bool sampled = (t == n_steps) || ((n_steps - t) % sample_every == 0);
            if (!sampled) continue;

            const RealVecX xg = Eigen::Map<const RealVecX>(guess.positions[t].data(),  dofs);
            const RealVecX xt = Eigen::Map<const RealVecX>(target.positions[target_first + t].data(), dofs);

            const RealVecX diff = xg - xt;
            total       += diff.squaredNorm() / Real(dofs);
            dloss_dx[t]  = (2.0 / Real(dofs)) * diff;
        }
    }
};

// Gather loss, no target: over the last `window` steps, every free particle should be within r0 of the vertical axis through the origin (x only, a one-sided
// squared hinge: any gathered cloth scores 0) and at rest (lam_rest * speed^2). Gradient is 0 outside the window. Also reports the open fraction (|x| <= r_eval)
// and the rms speed over the window.
struct GatherLoss : Loss
{
    Real gather = 0.0, rest = 0.0, open_frac = 0.0, speed_rms = 0.0;

    GatherLoss() = default;
    GatherLoss(const Tape& tape, Real r0, int window, Real lam_rest, Real dt, Real r_eval = 0.15)
    {
        const int n_frames = (int)tape.positions.size(), n = (int)tape.positions[0].rows();
        ASSERT(window >= 1 && window < n_frames, "GatherLoss: window must be in [1, steps)");
        dloss_dx.assign(n_frames, RealVecX::Zero(3 * n));
        const Real w = 1.0 / (Real(n) * window);
        Real inside = 0.0, speed2 = 0.0;
        for (int t = n_frames - window; t < n_frames; ++t)
        {
            const PointsX& x = tape.positions[t];
            for (int i = 0; i < n; ++i)
            {
                const Real over = std::abs(x(i, 0)) - r0;
                if (over > 0.0) { gather += w * over * over; dloss_dx[t][3 * i] += 2.0 * w * over * (x(i, 0) > 0.0 ? 1.0 : -1.0); }
                inside += std::abs(x(i, 0)) <= r_eval ? 1.0 : 0.0;
            }
            const PointsX v = (x - tape.positions[t - 1]) / dt;
            speed2 += v.squaredNorm();
            const PointsX g = (2.0 * lam_rest * w / dt) * v; // d/dx_t of lam_rest * w * |v|^2, minus the same on x_{t-1}
            dloss_dx[t]     += Eigen::Map<const RealVecX>(g.data(), 3 * n);
            dloss_dx[t - 1] -= Eigen::Map<const RealVecX>(g.data(), 3 * n);
        }
        rest      = lam_rest * speed2 * w;
        open_frac = inside * w;
        speed_rms = std::sqrt(speed2 * w);
        total     = gather + rest;
    }
};
