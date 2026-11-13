#pragma once

// backward_pd_contact: the adjoint pass through a recorded contact simulation (dphi/dx0, dphi/dv0, dphi/dk).

#include "solver/impulse/anderson.h"
#include "cloth/waist.h"
#include "collision/detect.h"
#include "core/timer.h"
#include "solver/loss.h"
#include "solver/tape.h"
#include "solver/impulse/contact_impulse.h"
#include "solver/pd_system.h"

#include <iostream>

// ----------------
//    BACKWARD
// ----------------

inline void precompute_constraints_local_derivative(Object& obj, const Positions& x)
{
    obj.energies.for_each([&](auto& e) { e.precompute_local_derivative(x, obj.mesh.pinned_rest); });
}

// Applies the assembled local elastic Jacobian ΔA (Σ_i Gᵢᵀ Γᵢ Gᵢ, every energy) to a vector — the same
// pattern used by both the position- and velocity-space adjoint RHS.
inline RealVecX apply_spring_jacobian(const Object& obj, const RealVecX& v)
{
    RealVecX out = RealVecX::Zero(v.rows());
    obj.energies.for_each([&](const auto& e) { e.add_jacobian_product(v, out); });
    return out;
}

inline void precompute_contacts_local_derivative(
    Contacts&          contacts,
    const Object&      obj,
    const Positions&   x_plus,
    const Velocities&  v_plus,
    const Positions&   x_minus,
    const Velocities&  v_minus,
    const Vec3&        gravity,
    Real               h,
    bool               gap_model)
{
    RealVecX x_tilde = x_minus + h * v_minus;
    const Vec3 dg    = h * h * gravity;
    for (Index i = 0; i < obj.num_particles(); ++i)
        x_tilde.segment<3>(3*i) += dg;

    const RealVecX b_inertia = obj.mass.cwiseProduct(x_tilde);
    const RealVecX b_tilde   = construct_velocity_rhs(obj, b_inertia, x_plus, obj.L * x_minus, h);
    const RealVecX f         = b_tilde - obj.C_d * v_plus; // damped free force (C_d == C when undamped)

    for (Contact& c : contacts)
    {
        const Vec3 f_i = f.segment<3>(3 * c.particle);
        const Real m_i = obj.mass(3 * c.particle);
        const Real d_n = contact_d_n(c, gap_model, f_i, m_i, h); // the forward pass's own test
        c.active = (d_n < 0.0);
        c.d_n    = d_n;
        c.d      = contact_d(c, gap_model, f_i, m_i, h); // friction: the derivatives rebuild state, t, kappa from (d, n, mu)
    }
}

struct ContactSplit
{
    RealVecX z_perp; // (I-P) z : normal component removed at active contacts
    RealVecX Pz;     // P z     : only the normal component, at active contacts
};

// Splits a vector into its active-contact-normal and perpendicular parts:
// v = (I-P) v + P v, where P = block-diag(n nᵀ) over active contacts.
inline ContactSplit split_by_contact(const Contacts& contacts, const RealVecX& v)
{
    ContactSplit s{ v, RealVecX::Zero(v.rows()) };
    for (const Contact& c : contacts)
    {
        if (!c.active) continue;
        const Vec3 vi = v.segment<3>(3 * c.particle);
        const Vec3 pn = c.normal * c.normal.dot(vi);
        s.Pz.segment<3>(3 * c.particle)     = pn;
        s.z_perp.segment<3>(3 * c.particle) = vi - pn;
    }
    return s;
}

inline RealVecX construct_backward_contact_rhs(
    const Object&   obj,
    const Contacts& contacts,
    const RealVecX& z,
    const RealVecX& dloss_dv,
    const RealVecX& dloss_dv_t,
    Real            h)
{

    const Real h2 = h * h;
    const ContactSplit split = split_by_contact(contacts, z);

    RealVecX b = h2 * apply_spring_jacobian(obj, split.z_perp);
    if (!contacts.empty())
        b += obj.C_d * split.Pz; // damped: the adjoint's P-coupling uses C_d (C_d == C when undamped)

    return b + dloss_dv + dloss_dv_t;
}

inline RealVecX compute_adjoint_vector_contact(
    Object&           obj,
    Contacts&         contacts,
    const Positions&  x_plus,
    const Velocities& v_plus,
    const Positions&  x_minus,
    const Velocities& v_minus,
    const RealVecX&   dloss_dv,
    const RealVecX&   dloss_dv_t,
    const Vec3&       gravity,
    Real              h,
    bool              gap_model,
    int               n_iters_adjoint,
    bool              verbose,
    Real&             residual_out)
{
    precompute_constraints_local_derivative(obj, x_plus);
    precompute_contacts_local_derivative(contacts, obj, x_plus, v_plus, x_minus, v_minus, gravity, h, gap_model);

    RealVecX z = RealVecX::Zero(obj.num_dofs());
    Anderson aa(anderson_m, z.size()); // linear map, active set fixed for the step: AA acts like GMRES
    Real rel_residual = 0.0;
    for (int k = 0; k < n_iters_adjoint; ++k)
    {
        const RealVecX b_back = construct_backward_contact_rhs(obj, contacts, z, dloss_dv, dloss_dv_t, h);
        RealVecX       z_new  = solve_global(obj, b_back);
        rel_residual = (z_new - z).norm() / std::max(z_new.norm(), Real(1e-12));
        if (k + 1 < n_iters_adjoint) aa.apply(z, z_new); // last iteration plain, as in the forward
        z = z_new;
    }
    residual_out = rel_residual;
    if (verbose && n_iters_adjoint > 0 && rel_residual > kConvergenceTol)
    {
        std::cout << "  iter " << n_iters_adjoint << " rel_delta = " << rel_residual << "\n";
    }

    return z;
}

// dphi/dk per SpringKind for one step: each spring's term lands in its own kind's slot, so the
// unified-stiffness gradient is just the sum of the three.
inline Stiffness compute_gradient_stiffness_contact(
    const Object&     obj,
    const Contacts&   contacts,
    const RealVecX&   z,
    const Positions&  x_minus,
    const Velocities& v_plus,
    Real              h)
{
    const RealVecX z_perp = split_by_contact(contacts, z).z_perp;
    // dC_d/dk = (h^2 + h*beta) * sum_i G_i^T G_i: the beta part is the damping force's sensitivity to k
    // (beta = 0, undamped: h^2 as before). b_tilde itself is undamped, so the p* - e^- term is unchanged.
    const Real     h2     = h * h + h * obj.damp_beta;

    Stiffness grad;
    obj.energies.for_each([&](const auto& e)
    { e.add_param_gradient(z_perp, x_minus, v_plus, obj.mesh.pinned_rest, h, h2, grad); });
    return grad;
}

// ----------------
//  BACKWARD: CONTACT CORRECTIONS
// ----------------
// An active contact adds  r_i = -n n^T (f_i - m_i v_c)  to the velocity-solve RHS, where the normal n and the
// contact-point velocity v_c are both evaluated at the predicted position x~ = x- + h v- + h^2 g. So the adjoint
// needs two corrections, one per way the RHS moves with x~. Each function returns its contribution to dphi/dx~
// (sign included); the caller adds it to dphi_dx and, times h (dx~/dv- = h I), to dphi_dv. (Speculative contact:
// the geometry is evaluated at x- instead, so it goes to dphi_dx only.)
// Common notation: z_n = n . z_i is the normal part of the adjoint, the part the contact projection removes.

// (1) Normal correction, n = n(x~): dn/dx~ is the collider's shape operator (curvature), inv_r (I - n n^T) for a
// sphere, also projected off the axis for a cylinder / capsule body (an Arc scales that part by R/rho), and 0 for planes and meshes (inv_r = 0).
//   dphi/dx~ = -inv_r P_axis [ d_n z_perp + z_n m (v+ - v_c) ]
inline Vec3 normal_correction(const Contact& c, const Vec3& z_perp_i, const Vec3& v_plus_i, Real z_n, Real m_i, bool speculative)
{
    // Q_i d_i = m_i Q_i (v+ - v_c); the old constraint makes n.(v+ - v_c) = 0 so Q drops out, the speculative one
    // (n.v+ = -gap/h) does not (only projected there, so the default path stays bit-identical).
    Vec3 w = v_plus_i - c.v_c;
    if (speculative) w -= c.normal * c.normal.dot(w);
    const Vec3 term = c.d_n * z_perp_i + z_n * m_i * w;
    return -c.inv_r * (term - c.axis_scale * c.axis * c.axis.dot(term)); // axis = 0 unless cylinder / capsule body / Arc; axis_scale = 1 except Arc
}

// (2) Velocity correction, v_c = v_c(x~): the forward pass stores G = dv_c/dx~ in Contact::dvc_dx, so for every collider
//   dphi/dx~ = +m z_n G^T n
// G is [w]_x J_s for a rotating rigid collider (v_c = v + w x (q - pivot), q the surface point, J_s = dq/dx~, see
// rigid_velocity_gradient in collision/shapes.h; then G^T n = -J_s^T (w x n)), the blend of the vertex velocity
// gradients for a skinned body, and zero if v_c does not depend on x~ (no rotation, speculative contact).

// dL/d(pose) and dL/d(v_c) of one collider, per global state (clock.end_state, the state its step is posed and detected at).
// Translation only: the collider's orientation is not differentiated.
struct ColliderGrad
{
    std::vector<Vec3> d_position; // dL/dc_s: the collider moved rigidly by c_s at state s (normal, surface point and gap all shift with it)
    std::vector<Vec3> d_velocity; // dL/dv_c,s: its velocity at state s, as the contact impulse sees it (0 with Gap contact)
};

struct BackwardGradContact
{
    RealVecX dphi_dv; // dphi/dv0 — gradient of loss w.r.t. initial velocity
    RealVecX dphi_dx; // dphi/dx0 — gradient of loss w.r.t. initial position
    Stiffness dphi_dk; // dphi/dk per SpringKind (sum = gradient w.r.t. a unified stiffness)
    Real     dphi_dalpha = 0.0; // dphi/d(alpha) — Laplacian damping, mass-proportional (0 unless damping on)
    Real     dphi_dbeta  = 0.0; // dphi/d(beta)  — Laplacian damping, K-proportional    (0 unless damping on)

    // Adjoint-solve convergence diagnostic (mirrors Tape::forward_residual), indexed in forward
    // chronological order even though computed in reverse.
    std::vector<Real> residual;

    std::vector<ColliderGrad> colliders; // one per world.colliders entry, sized first_step + n_steps + 1, zero where nothing touched it
};

inline BackwardGradContact backward_pd_contact(
    Object&     obj,
    CollisionWorld& world,
    Tape&       tape,
    const Loss& loss,
    const Vec3& gravity,
    int         n_iters_adjoint,
    Real        h,
    const StepCallback& on_step = nullptr,
    bool        verbose = false)
{
    const int   n_steps = (int)tape.positions.size() - 1;
    const Index dofs    = obj.num_dofs();

    // Preconditions, checked once up front so the body below stays assertion-frees
    auto validate = [&]()
    {
        ASSERT(tape.has_backward_data, "backward_pd_contact: the tape was recorded with world.collect_backward = false (forward only, e.g. friction > 0)");
        ASSERT((int)loss.dloss_dx.size() == n_steps + 1,
               "loss gradient size " << loss.dloss_dx.size()
               << " != tape size "   << tape.positions.size());
        ASSERT((int)tape.contacts.size() == n_steps,
               "contacts tape size " << tape.contacts.size()
               << " != n_steps "     << n_steps);
        for (const Collider& c : world.colliders)
        {
            const bool rotation_enabled = !c.animated && c.rotation_axis != RotationAxis::None && c.omega != 0.0;
            if (rotation_enabled)
                ASSERT(c.type == ColliderType::Sphere || c.type == ColliderType::Cylinder || c.type == ColliderType::Capsule,
                       "Rotation curvature correction is only implemented for Sphere, Cylinder, and Capsule world.colliders");
            if (c.animated)
                ASSERT(c.type == ColliderType::Sphere || c.type == ColliderType::Capsule || c.type == ColliderType::Mesh || c.type == ColliderType::Arc,
                       "Rotation curvature correction is only implemented for Sphere, Capsule, Mesh and Arc world.colliders");
        }
    };
    validate();

    std::vector<Real> residual(n_steps, 0.0);

    RealVecX dphi_dv = RealVecX::Zero(dofs); // a_t = dL/dv^+_t, seeded from later steps
    RealVecX dphi_dx = RealVecX::Zero(dofs); // b_t = dL/dx^+_t, seeded from later steps
    Stiffness dphi_dk;                       // dphi/dk per SpringKind, accumulated across all steps
    Real     dphi_dalpha = 0.0;              // dphi/d(alpha), accumulated across all steps
    Real     dphi_dbeta  = 0.0;              // dphi/d(beta),  accumulated across all steps

    const StepClock& clock = tape.clock; // the one the forward pass recorded
    ColliderGrad zero_grad;
    zero_grad.d_position.assign(clock.first_step + n_steps + 1, Vec3::Zero());
    zero_grad.d_velocity = zero_grad.d_position;
    std::vector<ColliderGrad> collider_grad(world.colliders.size(), zero_grad);
    for (int t = n_steps; t >= 1; --t)
    {
        // Restamp animated colliders + waist attachment to this step's state; shared with the
        // forward pass so collider pose and pinned positions can't drift out of sync.
        advance_world(world, obj, clock.substeps, clock.end_state(t - 1));

        const Positions  x_plus_t   = Eigen::Map<const Positions>(tape.positions[t].data(),       dofs);
        const Velocities v_plus_t   = Eigen::Map<const Velocities>(tape.velocities[t].data(),     dofs);
        const Positions  x_plus_tm1 = Eigen::Map<const Positions>(tape.positions[t - 1].data(),   dofs);
        const Velocities v_plus_tm1 = Eigen::Map<const Velocities>(tape.velocities[t - 1].data(), dofs);
        Contacts         contacts_t = tape.contacts[t - 1];

        const RealVecX b_t = dphi_dx + loss.dloss_dx[t];

        Real step_residual = 0.0;
        const RealVecX z = compute_adjoint_vector_contact(
            obj, contacts_t, x_plus_t, v_plus_t, x_plus_tm1, v_plus_tm1,
            dphi_dv, h * b_t, gravity, h, world.contact.gap(), n_iters_adjoint, verbose, step_residual);
        residual[t - 1] = step_residual;

        dphi_dk += compute_gradient_stiffness_contact(obj, contacts_t, z, x_plus_tm1, v_plus_t, h);

        const ContactSplit split   = split_by_contact(contacts_t, z);
        const RealVecX&     z_perp = split.z_perp; // (I-P) z

        // Damping parameters enter only through C_d (b_tilde is undamped): dC_d/dalpha = h M,
        // dC_d/dbeta = h K = C/h, so dphi/dalpha = -h z_perp^T M v+, dphi/dbeta = -(1/h) z_perp^T C v+.
        dphi_dalpha -= h * z_perp.dot(obj.mass.cwiseProduct(v_plus_t));
        dphi_dbeta  -= z_perp.dot(obj.C * v_plus_t) / h;

        dphi_dv = obj.mass.cwiseProduct(z_perp);
        dphi_dx = h * apply_spring_jacobian(obj, z_perp) - (obj.C * z_perp) / h + b_t;

        // Corrections for the terms of a contact that depend on x~: its normal, its surface velocity v_c and, with Gap contact, its gap.
        for (const Contact& c : contacts_t)
        {
            if (!c.active) continue;
            const Index i   = c.particle;
            const Real  m_i = obj.mass(3 * i);
            const Real  z_n = c.normal.dot(z.segment<3>(3 * i));

            const Vec3 normal_term   = normal_correction(c, z_perp.segment<3>(3 * i), v_plus_t.segment<3>(3 * i), z_n, m_i, world.contact.gap());
            const Vec3 velocity_term = m_i * z_n * (c.dvc_dx.transpose() * c.normal); // zero with Gap contact (v_c = 0)
            const Vec3 d             = normal_term + velocity_term;

            dphi_dx.segment<3>(3 * i) += d;
            Vec3 d_geometry = normal_term; // what moving the collider rigidly by c changes: n, q and the gap depend on x~ - c, so d/dc = -d/dx~
            if (world.contact.gap())
            {
                // The geometry sits at x-, so nothing goes to dphi/dv-; the gap term -m phi(x-)/h n adds this instead.
                const Vec3 gap_term = (m_i / h) * z_n * c.gap_grad;
                dphi_dx.segment<3>(3 * i) -= gap_term;
                d_geometry -= gap_term;
            }
            else
                dphi_dv.segment<3>(3 * i) += h * d; // x~ = x- + h v- + h^2 g, so v- moves the contact h times as far as x- does

            if (c.collider_id >= 0) // v_c enters the impulse as d = f - m v_c, so dphi/dv_c = m z_n n (Gap contact has v_c = 0 and no such term)
            {
                ColliderGrad& g = collider_grad[c.collider_id];
                g.d_position[clock.end_state(t - 1)] -= d_geometry;
                if (!world.contact.gap()) g.d_velocity[clock.end_state(t - 1)] += m_i * z_n * c.normal;
            }
        }

        if (on_step && !on_step(t, n_steps)) break;
        if (t % 10 == 0) std::cout << "backward (contact) step " << t << "/" << n_steps << "\n";
    }

    dphi_dx += loss.dloss_dx[0];

    return { std::move(dphi_dv), std::move(dphi_dx), dphi_dk, dphi_dalpha, dphi_dbeta, std::move(residual), std::move(collider_grad) };
}
