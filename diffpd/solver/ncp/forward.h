#pragma once

// pd_ncp_contact: the forward simulation with the NCP contact method (Zeng et al., "Fast But Accurate"), recorded on a Tape.
// Position-space local-global iterations; contacts and friction are multipliers solved together with the global step by a
// non-smooth Newton method (solver/ncp/newton.h). Forward only: the tape cannot feed the backward pass.
// The outer step loop mirrors solver/impulse/forward.h on purpose (neither method includes the other).

#include "cloth/obj_output.h"
#include "cloth/waist.h"
#include "collision/detect.h"
#include "core/timer.h"
#include "solver/ncp/newton.h"
#include "solver/pd_system.h"
#include "solver/tape.h"

#include <iostream>

// Optional per-step record of the last iterate, outside Tape: what the checks read now and a backward pass would need.
struct NcpLog
{
    std::vector<RealVecX> lambda;     // 3m multipliers per step, rows as NcpContacts::H (normal block, then tangent pairs)
    std::vector<RealVecX> delta;      // 3m gaps / slips H x - d at the last iterate
    std::vector<int>      iterations; // local-global iterations used (early exit)
    std::vector<int>      n_contacts; // detected contacts m (incl. the inactive ones inside the margin)
};

// tape.contacts[t] holds only the contacts with lambda_n > 0 (pressing), not everything detected.
inline void pd_ncp_contact(Object& obj, CollisionWorld& world, Real dt, const Vec3& gravity, const NcpSettings& ncp, int n_steps, int frame_substeps, Tape& tape,
                            const std::string& prefix, bool export_obj = true, bool verbose = false, const StepCallback& on_step = nullptr,
                            Real unresolved_penetration_threshold = 0.0, int first_step = 0, NcpLog* log = nullptr)
{
    ASSERT(!world.collect_backward, "NCP is forward only: set world.collect_backward = false");
    if (!world.animations.empty() || std::any_of(world.colliders.begin(), world.colliders.end(), [](const Collider& c) { return c.type == ColliderType::Skinned; }))
        ASSERT(std::abs(dt * frame_substeps - 1.0 / kColliderAnimFPS) < 1e-9,
               "animated colliders need FPS = " << kColliderAnimFPS << " (one sim frame per track frame; use frame_substeps to refine)");
    const auto solve_L = cholesky_solver(obj); // also asserts: factored, undamped

    tape.clear();
    const StepClock clock{ dt, first_step, frame_substeps, false };
    tape.clock = clock;
    tape.has_backward_data = false;
    tape.record(obj);
    if (export_obj) write_obj_frame(obj, 0, prefix);
    advance_world(world, obj, clock.substeps, first_step);

    TIMER_START("loop");
    for (int step = 0; step < n_steps; ++step)
    {
        advance_world(world, obj, clock.substeps, clock.end_state(step));

        RealVecX x_tilde = obj.x + dt * obj.v;
        const Vec3 dg = (dt * dt) * gravity;
        for (Index i = 0; i < obj.num_particles(); ++i)
            x_tilde.segment<3>(3 * i) += dg;
        const RealVecX b_inertia = obj.mass.cwiseProduct(x_tilde);

        obj.prev_x = obj.x;
        TIMER_START("detect");
        const NcpContacts contacts = detect_ncp_contacts(world, x_tilde, clock, step, obj.prev_x, dt, ncp);
        TIMER_END("detect");
        const Index m = contacts.m();
        NcpSystem sys;
        TIMER_START("delassus");
        if (m > 0) sys = make_ncp_system(solve_L, contacts.H, dt);
        TIMER_END("delassus");

        RealVecX x = x_tilde, lambda = RealVecX::Zero(3 * m);
        StepResidual res{};
        Real prev_delta = -1.0;
        int k = 0;
        for (; k < ncp.n_iters;)
        {
            TIMER_START("elastic");
            RealVecX b = b_inertia;
            add_elastic_forces(obj, x, b, dt * dt);
            TIMER_END("elastic");
            const RealVecX x_old = x;
            Real dlambda = 0.0;
            TIMER_START("ncp"); // contact multipliers + global solve
            if (m > 0) dlambda = ncp_iteration(solve_L, contacts, sys, ncp.smoothing, dt, b, x, lambda).dlambda;
            else       x = solve_L(b);
            TIMER_END("ncp");
            ++k;

            const RealVecX dv = (x - x_old) / dt;
            res        = make_step_residual(dv, (x - obj.prev_x) / dt, prev_delta, dt, gravity);
            prev_delta = dv.norm();
            // friction only switches on once lambda_n > 0 (iteration 2 on), so never stop before the second iteration
            if (k >= 2 && res.rel < ncp.tol && dlambda < ncp.tol) break;
        }
        if (verbose && res.rel >= ncp.tol) std::cout << "  step " << step << ": NCP not converged after " << k << " iterations, rel = " << res.rel << "\n";

        obj.x = x;
        obj.v = (x - obj.prev_x) / dt;
        tape.forward_residual.push_back(res);

        Contacts pressing;
        for (Index j = 0; j < m; ++j)
            if (lambda(j) > 0.0) pressing.push_back(contacts.contacts[j]);
        tape.record_contacts(pressing);
        if (log)
        {
            log->lambda.push_back(lambda);
            log->delta.push_back(m > 0 ? RealVecX(contacts.H * x - contacts.d) : RealVecX());
            log->iterations.push_back(k);
            log->n_contacts.push_back((int)m);
        }

        const Real contact_time = clock.contact_time(step);
        tape.unresolved_contacts.push_back((int)detect_contacts(world, obj.x, contact_time, unresolved_penetration_threshold).size());
        tape.record(obj);
        if (export_obj && step % frame_substeps == 0) write_obj_frame(obj, (step / frame_substeps) + 1, prefix);
        bool keep_going = true;
        if (on_step)
        {
            TIMER_START("viewer_callback"); // live-window drawing: excluded from the total
            keep_going = on_step(step, n_steps);
            TIMER_END("viewer_callback");
        }
        if (!keep_going) break;
        if (step % 10 == 0)
            std::cout << "step " << step << "/" << n_steps << "  contacts=" << m << " pressing=" << pressing.size()
                      << " iterations=" << k << " unresolved=" << tape.unresolved_contacts.back() << "\n";
    }
    TIMER_END("loop");
}
