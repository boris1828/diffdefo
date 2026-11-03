#pragma once

// Central-difference checks of the adjoint gradients: re-simulate with a perturbed x0 / v0 / stiffness and compare.

#include "app/fd_types.h"
#include "solver/forward.h"
#include "solver/loss.h"

#include <functional>
#include <iostream>

// ----------------
//    FD CHECK
// ----------------

enum class FDTarget { X0, V0 };

// FDCheckResult and FDCheckRunner are defined in app/fd_types.h (shared with the viewer).

inline FDCheckResult fd_check_contact_direction(
    CollisionWorld& world,
    const std::function<Object()>& build_obj,
    const Tape&     target_tape,
    Real            dt,
    const Vec3&     gravity,
    int             n_iters,
    int             n_steps,
    int             frame_substeps,
    int             sample_every,
    const RealVecX& direction,
    FDTarget        target,
    const RealVecX& analytic_grad,
    Real            eps = 1e-6)
{
    auto run_loss = [&](Real delta) -> Real
    {
        Object obj = build_obj();
        if (target == FDTarget::X0) obj.x += delta * direction;
        else                        obj.v += delta * direction;

        init_pd_velocity(obj, dt);
        Tape tape;
        pd_contact(obj, world, dt, gravity, n_iters, n_steps, frame_substeps, tape, "fd_check", /*export_obj=*/false);
        return Loss(tape, target_tape, sample_every).total;
    };

    const Real loss_plus  = run_loss(+eps);
    const Real loss_minus = run_loss(-eps);
    const Real fd         = (loss_plus - loss_minus) / (2.0 * eps);
    const Real analytic   = direction.dot(analytic_grad);
    const Real rel_err    = std::abs(fd - analytic)
        / std::max({ std::abs(fd), std::abs(analytic), Real(1e-12) });

    return { fd, analytic, rel_err };
}

inline RealVecX uniform_axis_direction(Index num_dofs, int axis)
{
    RealVecX d = RealVecX::Zero(num_dofs);
    for (Index i = axis; i < num_dofs; i += 3) d(i) = 1.0;
    return d;
}

inline void fd_check_contact_offsets(
    CollisionWorld& world,
    const std::function<Object()>& build_obj,
    const Tape&     target_tape,
    Real            dt,
    const Vec3&     gravity,
    int             n_iters,
    int             n_steps,
    int             frame_substeps,
    int             sample_every,
    Index           num_dofs,
    const RealVecX& dphi_dx0,
    const RealVecX& dphi_dv0,
    Real            eps = 1e-6)
{
    static const char* axis_name[3] = { "x", "y", "z" };
    std::cout << "FD check (uniform offset over all particles, eps=" << eps << "):\n";
    for (int axis = 0; axis < 3; ++axis)
    {
        const RealVecX dir = uniform_axis_direction(num_dofs, axis);

        const FDCheckResult rx = fd_check_contact_direction(
            world, build_obj, target_tape, dt, gravity, n_iters, n_steps, frame_substeps,
            sample_every, dir, FDTarget::X0, dphi_dx0, eps);
        const FDCheckResult rv = fd_check_contact_direction(
            world, build_obj, target_tape, dt, gravity, n_iters, n_steps, frame_substeps,
            sample_every, dir, FDTarget::V0, dphi_dv0, eps);

        std::cout << "  ." << axis_name[axis]
                   << "  dx0: fd=" << rx.fd << " analytic=" << rx.analytic << " rel_err=" << rx.rel_err
                   << "  |  dv0: fd=" << rv.fd << " analytic=" << rv.analytic << " rel_err=" << rv.rel_err
                   << "\n";
    }
}

inline FDCheckResult fd_check_contact_stiffness(
    CollisionWorld& world,
    const std::function<Object(Real)>& build_obj_k,
    const Tape&     target_tape,
    Real            k0,
    Real            dt,
    const Vec3&     gravity,
    int             n_iters,
    int             n_steps,
    int             frame_substeps,
    int             sample_every,
    Real            analytic_dphi_dk,
    Real            eps = 1e-6,
    const StepCallback& on_step = nullptr)
{
    // `pass` = 0 for +eps, 1 for -eps: on_step sees both runs as one loop of 2*n_steps steps, so
    // callers get continuous progress (step in [0, 2*n_steps)) across the pair.
    auto run_loss = [&](Real delta, int pass) -> Real
    {
        Object obj = build_obj_k(k0 + delta);
        init_pd_velocity(obj, dt);
        Tape tape;
        const StepCallback pass_on_step = on_step
            ? StepCallback([&](int step, int n) { return on_step(pass * n + step, 2 * n); })
            : nullptr;
        pd_contact(obj, world, dt, gravity, n_iters, n_steps, frame_substeps, tape, "fd_check", /*export_obj=*/false,
                   /*verbose=*/false, pass_on_step);
        // on_step may abort early, leaving a truncated tape; bail with a sentinel instead of
        // crashing Loss's tape-length assert.
        if ((int)tape.positions.size() != n_steps + 1) return 0.0;
        return Loss(tape, target_tape, sample_every).total;
    };

    const Real loss_plus  = run_loss(+eps, 0);
    const Real loss_minus = run_loss(-eps, 1);
    const Real fd         = (loss_plus - loss_minus) / (2.0 * eps);
    const Real rel_err    = std::abs(fd - analytic_dphi_dk)
        / std::max({ std::abs(fd), std::abs(analytic_dphi_dk), Real(1e-12) });

    return { fd, analytic_dphi_dk, rel_err };
}
