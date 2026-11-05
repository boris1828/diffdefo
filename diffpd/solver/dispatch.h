#pragma once

// The one place that knows both contact methods: runs the forward simulation of the method chosen in SolverSettings.
// Settle and the FD check call pd_contact directly (a rest pose and the gradient check are impulse-method things).

#include "solver/impulse/forward.h"
#include "solver/ncp/forward.h"
#include "solver/settings.h"

inline void simulate_forward(Object& obj, CollisionWorld& world, const SolverSettings& s, int n_steps, Tape& tape, const std::string& prefix,
                             bool export_obj, bool verbose, const StepCallback& on_step, Real unresolved_penetration_threshold)
{
    const Real dt = 1.0 / (s.FPS * s.frame_substeps);
    switch (s.contact_method)
    {
    case ContactMethod::Impulse:
        pd_contact(obj, world, dt, s.gravity, s.n_iters, n_steps, s.frame_substeps, tape, prefix, export_obj, verbose, on_step, unresolved_penetration_threshold);
        break;
    case ContactMethod::Ncp:
        pd_ncp_contact(obj, world, dt, s.gravity, s.ncp, n_steps, s.frame_substeps, tape, prefix, export_obj, verbose, on_step, unresolved_penetration_threshold);
        break;
    }
}
