#pragma once

// Parameters of the NCP contact method (Zeng et al., "Fast But Accurate"). Plain data; nothing else in solver/ reads it.

#include "core/diffpd_core.h"

// Which operator the contact-multiplier update uses (solver/ncp/newton.h). Full: W = H A^-1 H^T, dense. MassSplit: W = H M^-1 H^T
// (diagonal for one contact per particle) with the elastic part as an explicit residual; same fixed point, cheaper iterations.
enum class NcpSchur { Full = 0, MassSplit = 1 };

struct NcpSettings
{
    NcpSchur schur = NcpSchur::Full;
    Real friction  = 0.3;    // Coulomb coefficient mu, one value for every contact (colliders have no friction field yet)
    Real margin    = 0.005;  // detection margin epsilon [m]: particles up to this far outside the surface at x_tilde join the set (their constraint is simply inactive)
    Real smoothing = 1e-12;  // 2 eps^2 of the smoothed Fischer-Burmeister function (the paper's value; a backward pass would need the same one)
    int  n_iters   = 20;     // cap on the local-global iterations of one step
    Real tol       = 1e-4;   // early exit: relative change of x and of lambda below this (after at least 2 iterations)
};
