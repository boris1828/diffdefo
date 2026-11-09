#pragma once

// Adam on any flat parameter vector (hand-path control points, log stiffness, ...); knows nothing about the simulation. Depends only on core/.

#include "core/diffpd_core.h"

struct Adam
{
    Real lr0      = 0.05;
    Real lr_decay = 0.98;  // lr at step t = lr0 * lr_decay^t
    Real beta1    = 0.9;
    Real beta2    = 0.99;  // low, so one gradient spike is forgotten within ~100 steps
    Real eps      = 1e-20; // tiny: gradients here can be ~1e-9
    int  t        = 0;
    RealVecX m, v;

    // One update of `theta` in place, descending along `grad` (dL/dtheta).
    void step(RealVecX& theta, const RealVecX& grad)
    {
        ASSERT(theta.size() == grad.size(), "Adam::step: theta has " << theta.size() << " entries, grad " << grad.size());
        if (m.size() != theta.size()) { m = RealVecX::Zero(theta.size()); v = m; }
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad.cwiseProduct(grad);
        const Real lr = lr0 * std::pow(lr_decay, t);
        ++t;
        const Real c1 = 1.0 - std::pow(beta1, t), c2 = 1.0 - std::pow(beta2, t); // bias correction
        theta -= lr * (m / c1).cwiseQuotient(((v / c2).cwiseSqrt().array() + eps).matrix());
    }
};
