// ----------------------------------------------------------------------------------------------
// LEGACY / REFERENCE DUMP — NOT PART OF THE BUILD (not listed in CMakeLists.txt).
//
// This file holds the classic position-based Projective Dynamics forward/backward path (no
// contact) that used to live in diffpd.cpp, removed once the project committed exclusively to the
// velocity-based contact PD path (pd_contact / backward_pd_contact). Kept here verbatim, in case
// any of this needs to be looked up or resurrected later.
//
// Everything below depended only on Object/Constraint/Tape/Loss (diffpd_types.h) plus a handful of
// diffpd.cpp helpers that are still in active use and were NOT removed: add_elastic_forces,
// apply_spring_jacobian, precompute_constraints_local_derivative. Those three are shared with the
// contact path (construct_velocity_rhs, construct_backward_contact_rhs, compute_adjoint_vector_contact
// all call into them) and still live in diffpd.cpp.
//
// To resurrect: paste the relevant piece(s) back into diffpd.cpp (this file assumes the same
// #include "diffpd_types.h" and namespace-free style) and re-add to CMakeLists.txt if compiling
// this file standalone rather than merging it back in.
// ----------------------------------------------------------------------------------------------

#include "diffpd_types.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <string>
#include <functional>

// ----------------
//   BackwardGrad<ParamGrad> — return type of backward_pd()
// ----------------

template <typename ParamGrad>
struct BackwardGrad
{
    RealVecX  free_grad;  // dphi/dx for free particles (per-step: dx_minus; full: dx0)
    ParamGrad param_grad; // dphi/dtheta — shape determined by ParamGrad
};

// ----------------
//   construct_lhs — position-space LHS (L = M/h^2 + sum_i k_i G_i^T G_i), used only by init_pd
// ----------------

void construct_lhs(Object& obj, Real dt)
{
    const Index n3 = obj.num_dofs();
    const Real  h2 = dt * dt;

    std::vector<Triplet> triplets;
    triplets.reserve(n3 + 12 * Index(obj.constraints.size()));

    // M / h²
    for (Index i = 0; i < n3; ++i)
        triplets.emplace_back(i, i, obj.mass(i) / h2);

    // sum_i k_i G_i^T G_i
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            for (int d = 0; d < 3; ++d)
            {
                triplets.emplace_back(3*i1+d, 3*i1+d, +c.k);
                triplets.emplace_back(3*i2+d, 3*i2+d, +c.k);
                triplets.emplace_back(3*i1+d, 3*i2+d, -c.k);
                triplets.emplace_back(3*i2+d, 3*i1+d, -c.k);
            }
        }
        else // SpringType::Spring1
        {
            const Index i = c.spring1.i;
            for (int d = 0; d < 3; ++d)
                triplets.emplace_back(3*i+d, 3*i+d, +c.k);
        }
    }

    obj.L.resize(n3, n3);
    obj.L.setFromTriplets(triplets.begin(), triplets.end());

    obj.solver = std::make_unique<Cholesky>();
    obj.solver->compute(obj.L);
    ASSERT(obj.solver->info() == Eigen::Success, "Cholesky factorization of L failed");
}

// ----------------
//   construct_backward_rhs — non-contact adjoint RHS, used only by compute_adjoint_vector
// ----------------
//
// Assumes apply_spring_jacobian(obj, v) (still in diffpd.cpp) is available.

RealVecX construct_backward_rhs(
    const Object&   obj,
    const RealVecX& z,
    const RealVecX& dloss_dx,
    const RealVecX& dloss_dx_t)
{
    return apply_spring_jacobian(obj, z) + dloss_dx + dloss_dx_t;
}

// ----------------
//   compute_adjoint_vector — non-contact adjoint solve, used only by backward_pd_step
// ----------------
//
// Assumes precompute_constraints_local_derivative(obj, x) (still in diffpd.cpp) is available.

RealVecX compute_adjoint_vector(
    Object&          obj,
    const Positions& x_plus,
    const RealVecX&  dloss_dx,
    const RealVecX&  dloss_dx_t,
    int              n_iters_adjoint)
{
    precompute_constraints_local_derivative(obj, x_plus);

    RealVecX z = RealVecX::Zero(obj.num_dofs());
    for (int k = 0; k < n_iters_adjoint; ++k)
    {
        const RealVecX b_back = construct_backward_rhs(obj, z, dloss_dx, dloss_dx_t);
        z = obj.solver->solve(b_back);
    }
    return z;
}

// ----------------
//   compute_gradient_pinned_vertices — dphi/dxbar for anchor positions (example backward_pd callback)
// ----------------

Vec3 compute_gradient_pinned_vertices(const Object& obj, const RealVecX& z)
{
    Vec3 grad = Vec3::Zero();
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring1)
        {
            const Vec3 zi = z.segment<3>(3 * c.spring1.i);
            grad += c.k * zi - c.gamma * zi; // (k·I − Γ)·z_i
        }
    }
    return grad;
}

// ----------------
//   compute_gradient_stiffness — dphi/dk, non-contact version (example backward_pd callback).
//   NOT the same as compute_gradient_stiffness_contact, which is still in diffpd.cpp and in use.
// ----------------

Real compute_gradient_stiffness(const Object& obj, const RealVecX& z)
{
    Real grad = 0;
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Vec3 zi1 = z.segment<3>(3 * c.spring2.i1);
            const Vec3 zi2 = z.segment<3>(3 * c.spring2.i2);
            grad += (zi1 - zi2).dot(c.p_star - c.e);
        }
        else // Spring1
        {
            const Vec3 zi = z.segment<3>(3 * c.spring1.i);
            grad += zi.dot(c.p_star - c.e);
        }
    }
    return grad;
}

// ----------------
//   AdjointStep / backward_pd_step — per-step non-contact adjoint, used only by backward_pd
// ----------------

struct AdjointStep
{
    RealVecX free_grad;
    RealVecX z;
};

AdjointStep backward_pd_step(
    Object&          obj,
    const Positions& x_plus,
    const RealVecX&  dloss_dx,
    const RealVecX&  dloss_dx_t,
    Real             dt,
    int              n_iters_adjoint)
{
    RealVecX z = compute_adjoint_vector(obj, x_plus, dloss_dx, dloss_dx_t, n_iters_adjoint);
    RealVecX free_grad = obj.mass.cwiseProduct(z) / (dt * dt);
    return { std::move(free_grad), std::move(z) };
}

// ----------------
//   pd_step / init_pd / pd — position-space forward PD (no contact)
// ----------------
//
// Assumes add_elastic_forces(obj, x, b, scale) (still in diffpd.cpp) and write_obj_frame (still in
// diffpd.cpp) are available.

void pd_step(Object& obj, Real dt, const Vec3& gravity, int n_iters)
{
    const Real h2 = dt * dt;

    obj.prev_x = obj.x;
    RealVecX x_tilde = obj.x + dt * obj.v;
    const Vec3 dg = h2 * gravity;
    for (Index i = 0; i < obj.num_particles(); ++i)
        x_tilde.segment<3>(3*i) += dg;

    const RealVecX b_inertia = obj.mass.cwiseProduct(x_tilde) / h2;

    obj.x = x_tilde;

    RealVecX b(obj.num_dofs()); // reused every iteration below; same size each time, so no realloc
    for (int k = 0; k < n_iters; ++k)
    {
        b = b_inertia;
        add_elastic_forces(obj, obj.x, b);
        obj.x = obj.solver->solve(b);
    }

    obj.v = (obj.x - obj.prev_x) / dt;
}

void init_pd(Object& obj, Real dt)
{
    construct_lhs(obj, dt);
}

void pd(Object& obj, Real dt, const Vec3& gravity, int n_iters, int n_steps, int frame_substeps, Tape& tape, const std::string& prefix)
{
    tape.clear();
    tape.record(obj);
    write_obj_frame(obj, 0);

    for (int step = 0; step < n_steps; ++step)
    {
        pd_step(obj, dt, gravity, n_iters);
        tape.record(obj);
        if (step % frame_substeps == 0) write_obj_frame(obj, (step / frame_substeps) + 1, prefix);

        if (step % 10 == 0) std::cout << "step " << step << "/" << n_steps << "\n";
    }
}

// ----------------
//   backward_pd<ParamGrad, F> — generic non-contact backward loop, never called from main()
// ----------------

template <typename ParamGrad, typename F>
BackwardGrad<ParamGrad> backward_pd(
    Object&     obj,
    const Tape& tape,
    const Loss& loss,
    int         n_iters_adjoint,
    Real        dt,
    ParamGrad   init_param_grad,
    F&&         accumulate_param_grad)
{
    const int   n_steps = (int)tape.positions.size() - 1;
    const Index dofs    = obj.num_dofs();

    ASSERT((int)loss.dloss_dx.size() == n_steps + 1,
           "loss gradient size " << loss.dloss_dx.size()
           << " != tape size " << tape.positions.size());

    RealVecX  adj        = RealVecX::Zero(dofs);
    ParamGrad param_grad = std::move(init_param_grad);

    for (int t = n_steps; t >= 1; --t)
    {
        const Positions x_plus = Eigen::Map<const Positions>(tape.positions[t].data(), dofs);
        auto [free_grad, z]    = backward_pd_step(obj, x_plus, adj, loss.dloss_dx[t], dt, n_iters_adjoint);
        adj = std::move(free_grad);
        accumulate_param_grad(obj, z, param_grad);

        if (t % 10 == 0) std::cout << "backward step " << t << "/" << n_steps << "\n";
    }

    return { std::move(adj), std::move(param_grad) };
}

// Built-in callbacks (see CLAUDE.md's original description of backward_pd):
//   anchor positions: accumulate_param_grad = [](auto& obj, auto& z, Vec3& acc) { acc += compute_gradient_pinned_vertices(obj, z); };
//   uniform stiffness: accumulate_param_grad = [](auto& obj, auto& z, Real& acc) { acc += compute_gradient_stiffness(obj, z); };
