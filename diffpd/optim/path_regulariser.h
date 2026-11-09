#pragma once

// Costs on a hand path alone (no simulation), in metres and seconds, on the keyframe positions p_k (diffpd axes, `fps` apart):
//   speed   lambda_v * integral |p'|^2 dt     short, direct paths
//   accel   lambda_a * integral |p''|^2 dt    smooth paths, bounds the force an arm would need
//   bound   lambda_b * sum_k dist(p_k, box)^2 stay inside the workspace
// Written on positions, not on theta, so the weights mean the same for any parametrisation: add dR/dp to the simulation's dL/dp before the pull-back.
// Depends only on core/.

#include "core/diffpd_core.h"
#include <limits>

struct PathRegulariser
{
    Real speed = 0.0, accel = 0.0, bound = 0.0; // the weights lambda; 0 switches a term off
    Vec3 box_min = Vec3::Constant(-std::numeric_limits<Real>::infinity());
    Vec3 box_max = Vec3::Constant(std::numeric_limits<Real>::infinity());
    Real fps     = 60.0;

    // Returns the cost and adds its gradient to `grad` (same size as `p`).
    Real add(const std::vector<Vec3>& p, std::vector<Vec3>& grad) const
    {
        const int n = (int)p.size();
        ASSERT((int)grad.size() == n, "PathRegulariser::add: grad has " << grad.size() << " entries, the path " << n);
        Real cost = 0.0;
        for (int k = 0; k + 1 < n && speed != 0.0; ++k) // sum |p_k+1 - p_k|^2 * fps  (= sum |v|^2 dt with v = fps * diff)
        {
            const Vec3 d = p[k + 1] - p[k];
            cost += speed * fps * d.squaredNorm();
            grad[k + 1] += 2.0 * speed * fps * d;
            grad[k]     -= 2.0 * speed * fps * d;
        }
        for (int k = 1; k + 1 < n && accel != 0.0; ++k) // sum |p_k+1 - 2 p_k + p_k-1|^2 * fps^3  (= sum |a|^2 dt with a = fps^2 * second difference)
        {
            const Vec3 s = p[k + 1] - 2.0 * p[k] + p[k - 1];
            const Real w = accel * fps * fps * fps;
            cost += w * s.squaredNorm();
            grad[k + 1] += 2.0 * w * s;
            grad[k]     -= 4.0 * w * s;
            grad[k - 1] += 2.0 * w * s;
        }
        for (int k = 0; k < n && bound != 0.0; ++k)
            for (int a = 0; a < 3; ++a)
            {
                const Real over = std::max(0.0, p[k][a] - box_max[a]), under = std::max(0.0, box_min[a] - p[k][a]);
                cost += bound * (over * over + under * under);
                grad[k][a] += 2.0 * bound * (over - under);
            }
        return cost;
    }
};
