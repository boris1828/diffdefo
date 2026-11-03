#pragma once

// pd_contact: the forward simulation with contact (Projective Dynamics, velocity-implicit, recorded on a Tape).

#include "anderson.h"
#include "cloth/obj_output.h"
#include "cloth/waist.h"
#include "collision/detect.h"
#include "core/timer.h"
#include "solver/tape.h"
#include "solver/velocity_solver.h"

#include <iostream>
#include <sstream>

// ----------------
//       PD
// ----------------

// Convergence measures (StepResidual) of the last iteration, whose velocity change is `dv` and result `v`;
// `prev_delta` is the previous iteration's ||dv|| (< 0 if there was none).
inline StepResidual make_step_residual(const RealVecX& dv, const RealVecX& v, Real prev_delta, Real h, const Vec3& gravity)
{
    const Index n     = v.size() / 3;
    const Real  delta = dv.norm();
    Real vmax = 0.0;
    for (Index i = 0; i < n; ++i)
        vmax = std::max(vmax, Vec3(dv.segment<3>(3 * i)).norm());

    StepResidual r;
    r.rel  = delta / std::max(v.norm() + std::sqrt((Real)n) * h * gravity.norm(), Real(1e-12));
    r.vmax = vmax;
    r.rho  = prev_delta > 0.0 ? delta / prev_delta : std::numeric_limits<Real>::quiet_NaN();
    r.est_rel = r.rel;
    if (!std::isnan(r.rho))
    {
        const Real rho = std::min(r.rho, kMaxContraction);
        r.est_rel = r.rel * rho / (1.0 - rho);
    }
    return r;
}

inline void pd_contact(Object& obj, CollisionWorld& world, Real dt, const Vec3& gravity, int n_iters, int n_steps, int frame_substeps, Tape& tape, const std::string& prefix, bool export_obj = true, bool verbose = false, const StepCallback& on_step = nullptr, Real unresolved_penetration_threshold = 0.0, bool frozen = false, int first_step = 0)
{
    // `frozen`: every collider (animated, skinned, analytic) and the waist pins stay at their frame-0 pose with zero
    // velocity for the whole run - used to settle the cloth into a rest pose (settle_cloth). The track/FPS
    // preconditions don't apply since no pose is ever advanced.
    // Preconditions, checked once up front so the body below stays assertion-free.
    auto validate = [&]()
    {
        const bool has_skinned = std::any_of(world.colliders.begin(), world.colliders.end(),
                                              [](const Collider& c) { return c.type == ColliderType::Skinned; });
        if (world.animations.empty() && !has_skinned) return;
        ASSERT(std::abs(dt * frame_substeps - 1.0 / kColliderAnimFPS) < 1e-9,
               "animated world.colliders need FPS = " << kColliderAnimFPS << " (one sim frame per track frame; "
               "use frame_substeps to refine) — got dt*frame_substeps=" << dt * frame_substeps);
    };
    if (!frozen) validate();

    tape.clear();
    const StepClock clock{ dt, first_step, frame_substeps, frozen };
    tape.clock = clock; // the backward pass and the viewer pose the world from this same clock
    tape.has_backward_data = world.collect_backward;
    tape.record(obj);
    if (export_obj) write_obj_frame(obj, 0, prefix);

    // Poses animated colliders (+ waist attachment) from the imported track at physics state `step`.
    // Called for state 0 up front, then once per step (end-of-step state, matching x_tilde) so contact
    // detection/velocity basis see the right pose.
    auto stamp_animated_colliders = [&](int state) { advance_world(world, obj, clock.substeps, state); };

    stamp_animated_colliders(first_step);

    size_t total_added_in_iters = 0; // contacts found mid-iteration (merge_detected_contacts), whole run
    TIMER_START("loop");
    for (int step = 0; step < n_steps; ++step)
    {
        if (!frozen) stamp_animated_colliders(clock.end_state(step));

        RealVecX x_tilde = obj.x + dt * obj.v;
        const Vec3 dg = (dt * dt) * gravity;
        for (Index i = 0; i < obj.num_particles(); ++i)
            x_tilde.segment<3>(3*i) += dg;

        const Real     contact_time = clock.contact_time(step);
        const RealVecX b_inertia    = obj.mass.cwiseProduct(x_tilde);
        
        TIMER_START("detect");
        Contacts contacts = detect_contacts(world, x_tilde, contact_time, world.contact.detect_threshold());
        TIMER_END("detect");

        obj.prev_x = obj.x;
        obj.x      = x_tilde;

        stamp_contacts(world, clock, step, obj.prev_x, contacts, 0, contacts.size());

        // Rechecks stop halfway so the final contact set still gets n_iters/2 iterations to converge (the backward
        // pass assumes the fixed point). Gap contact never rechecks: its margin replaces them.
        const int recheck_end    = n_iters / 2;
        const int recheck_period = world.contact.recheck_period(recheck_end);

        Anderson aa(frozen ? 0 : anderson_m, obj.v.size()); // never while settling
        int  contacts_added_in_iters = 0;
        Real prev_delta = -1.0; // ||dv|| of the previous iteration, for the contraction ratio
        for (int k = 0; k < n_iters; ++k)
        {
            // Re-test the current iterate and grow the contact set (merge_detected_contacts). Runs
            // before the solve so the last iteration — the one the tape records — sees the final set.
            if (k > 0 && k < recheck_end && recheck_period > 0 && k % recheck_period == 0)
            {
                TIMER_START("detect");
                const int added = merge_detected_contacts(world, contacts, obj.x, x_tilde, contact_time);
                stamp_contacts(world, clock, step, obj.prev_x, contacts, contacts.size() - added, contacts.size());
                contacts_added_in_iters += added;
                if (added > 0) aa.reset();
                TIMER_END("detect");
            }

            TIMER_START("elastic");
            const RealVecX b_tilde = construct_velocity_rhs(obj, b_inertia, obj.x, obj.prev_x, dt);
            TIMER_END("elastic");

            TIMER_START("contact");
            const RealVecX f = b_tilde - obj.C_d * obj.v; // damped Jacobi split (C_d == C when undamped)
            RealVecX g = b_tilde;
            for (Contact& c : contacts)
            {
                const Vec3 f_i = f.segment<3>(3 * c.particle);
                const Real m_i = obj.mass(3 * c.particle);
                g.segment<3>(3 * c.particle) += update_contact_force(f_i, m_i, c.v_c, c.normal, c.gap / dt);
            }
            TIMER_END("contact");

            TIMER_START("global");
            RealVecX v_hat = obj.solver->solve(g);
            if (k >= n_iters - 2) // only the last two deltas are needed (contraction ratio)
            {
                const RealVecX dv = v_hat - obj.v;
                if (k == n_iters - 1)
                {
                    const StepResidual res = make_step_residual(dv, v_hat, prev_delta, dt, gravity);
                    tape.forward_residual.push_back(res);
                    if (verbose && res.est_rel > kConvergenceTol)
                        std::cout << "  step " << step << " iter " << k+1 << " est_error = " << res.est_rel << "\n";
                }
                prev_delta = dv.norm();
            }

            // k = 0 evaluates G at x~ (not x- + h v-), another map; the last iteration stays plain so v+ solves the global system
            if (k > 0 && k + 1 < n_iters) aa.apply(obj.v, v_hat);
            obj.v = v_hat;
            obj.x = obj.prev_x + dt * obj.v;
            TIMER_END("global");
        }

        tape.record_contacts(contacts); // after the solve: carries the converged per-contact v_c
        // Re-detect against the converged x: how many particles are still penetrating a collider
        // after this step's iterations, as opposed to `contacts` above (detected pre-solve, against
        // x_tilde). A small positive threshold treats negligible residual penetration as resolved.
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
        total_added_in_iters += contacts_added_in_iters;
        if (step % 10 == 0)
            std::cout << "step " << step << "/" << n_steps
                       << "  contacts=" << contacts.size()
                       << " (added in iters=" << contacts_added_in_iters << ")"
                       << " unresolved=" << tape.unresolved_contacts.back() << "\n";
    }
    TIMER_END("loop");

    size_t total_contacts = 0;
    for (const Contacts& c : tape.contacts) total_contacts += c.size();
    size_t total_unresolved = 0;
    for (int u : tape.unresolved_contacts) total_unresolved += u;
    const Real unresolved_pct = total_contacts > 0
        ? 100.0 * (Real)total_unresolved / (Real)total_contacts
        : 0.0;
    std::cout << "[" << prefix << "] total contacts=" << total_contacts
               << " (added in iters=" << total_added_in_iters << ")"
               << " total unresolved=" << total_unresolved
               << " (" << unresolved_pct << "%)\n";
}
