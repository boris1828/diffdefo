#pragma once

// Mean squared position difference between a guess and a target tape, and its gradient per frame.

#include "solver/tape.h"

struct Loss
{
    Real                  total;
    std::vector<RealVecX> dloss_dx;

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
