#pragma once

// How a run is stepped: time step, iteration counts, gravity and damping. Plain data, no behaviour.

#include "core/diffpd_core.h"

struct SolverSettings
{
    Vec3 gravity = Vec3::UnitY() * -9.81;

    // Laplacian damping D = alpha*M + beta*K (see Object::damp_alpha). When disabled, alpha/beta are
    // passed as 0 so the damped matrices equal the undamped ones.
    bool damping_enabled = false;
    Real damping_alpha   = 0.0;    // 1/s
    Real damping_beta    = 0.001;  // s

    int FPS             = 60;
    int frame_substeps  = 1;
    int secs            = 4;
    int n_iters         = 300; // forward PD global-local iterations per step
    int n_iters_adjoint = 300; // backward adjoint-vector iterations per step
    int anderson_m      = 0;   // Anderson acceleration history of the forward and adjoint iterations (solver/anderson.h); 0 = off
};
