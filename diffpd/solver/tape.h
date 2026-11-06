#pragma once

// The recording of a forward run (positions, velocities, contacts, convergence residuals) that the backward pass and
// the viewer read. Plain data plus record(); no solver code.

#include "cloth/object.h"
#include "collision/world.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

// A step counts as converged when its relative residual (change between the last two iterations) is below this.
inline constexpr Real kConvergenceTol = 1e-4;

// Forward-solve convergence measures for one step, all taken from the last iteration's change dv = v_K - v_{K-1}.
struct StepResidual
{
    Real rel;     // ||dv|| / (||v|| + sqrt(N) h |g|): relative change, floored by the velocity gravity adds in one step
                  // so near-rest steps aren't inflated by a tiny ||v||
    Real vmax;    // max_i ||dv_i||  [m/s]: per-particle worst case, so a few unsettled contact particles aren't diluted
    Real rho;     // contraction ratio ||dv_K|| / ||dv_{K-1}|| (NaN if undefined); near 1 = stagnation / limit cycle
    Real est_rel; // rho-corrected remaining error rel * rho/(1-rho), rho clamped to [0, kMaxContraction]; = rel if rho
                  // undefined. The headline measure: tracks the true error to ~1.2x where rel alone is ~7x too small
};
inline constexpr Real kMaxContraction = 0.99;

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

// Per-step convergence residuals for one guess/target run (forward + adjoint), for the playback
// screen's convergence graphs. All three are sized n_steps, forward chronological order.
struct ResidualHistory
{
    std::vector<StepResidual> target_forward;
    std::vector<StepResidual> guess_forward;
    std::vector<Real> backward_adjoint;
};

// ----------------
//       TAPE
// ----------------

struct Tape
{
    std::vector<PointsX> positions;   // positions: one Nx3 matrix per timestep
    std::vector<PointsX> velocities;  // velocities: one Nx3 matrix per timestep
    std::vector<Contacts> contacts;   // contacts[t] = contacts detected at the start of step t (size == n_steps)
                                       // -> contacts[t].size() is "particles colliding" that frame.

    // Forward-solve convergence diagnostic: one StepResidual per step (from the last iteration).
    std::vector<StepResidual> forward_residual;

    // How many particles are still penetrating a collider after step t's iterations converge
    // (re-detected against the converged x, not the pre-solve x_tilde in contacts[t]). Ideally 0 —
    // a nonzero value means the solve didn't fully resolve contact within n_iters that step.
    // Empty when the run did not compute it (WorldSpec::compute_unresolved off).
    std::vector<int> unresolved_contacts;

    // The step -> state/time mapping this tape was recorded with (set by pd_contact). Animated colliders, skinned
    // bodies and the waist are indexed by global step, so the backward pass and the viewer pose them from this clock;
    // clock.first_step != 0 for a run resumed mid-trajectory (a chunk).
    StepClock clock;

    // False if pd_contact ran with world.collect_backward = false: the contacts then lack the data backward_pd_contact needs.
    bool has_backward_data = true;

    void clear()
    {
        positions.clear();
        velocities.clear();
        contacts.clear();
        forward_residual.clear();
        unresolved_contacts.clear();
    }

    void record(const Object& obj)
    {
        const Index n = obj.num_particles();
        PointsX pos = Eigen::Map<const PointsX>(obj.x.data(), n, 3);
        PointsX vel = Eigen::Map<const PointsX>(obj.v.data(), n, 3);
        positions.push_back(pos);
        velocities.push_back(vel);
    }

    void record_contacts(const Contacts& c) { contacts.push_back(c); }
};

// Called after each step/iteration of a watchable loop (pd_contact, backward_pd_contact,
// fd_check_contact_stiffness). Returning false aborts the loop early (e.g. window closed).
using StepCallback = std::function<bool(int step, int n_steps)>;
