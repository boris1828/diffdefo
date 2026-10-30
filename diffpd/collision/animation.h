#pragma once

// Baked per-frame collider tracks (from Blender): the track data, and the pose / velocity of a collider at a
// physics step. The .json loader is load_collider_animation in diffpd.cpp. Depends only on collision/collider.h.

#include "collision/collider.h"

// ----------------
//  ANIMATED COLLIDERS
// ----------------
// A collider's base fields can be stamped every step from a per-frame track baked from Blender
// (an animation .json in diffpd/animation) instead of derived analytically from velocity/omega. See Collider::animated.

// Rigid velocity (linear + angular) of an animated collider over one bracket [frame, frame+1] of the
// baked track — constant across that whole bracket, so it's precomputed once per frame (see
// ColliderAnimation::velocity_basis) rather than recomputed per step.
struct AnimatedColliderVelocityBasis
{
    Vec3 linear_velocity = Vec3::Zero();
    Vec3 omega_vec       = Vec3::Zero();
};

struct ColliderFrame
{
    Vec3 position;                    // world-space capsule midpoint / sphere center
    Eigen::Quaternion<Real> rotation; // world-space orientation of the source bone
};

struct ColliderAnimation
{
    std::string name;                 // matches the JSON's colliders_metadata key, for diagnostics
    ColliderType type      = ColliderType::Capsule;
    Real         radius      = 0.0;
    int          mesh_id     = -1;          // mesh only: index into trimeshes
    Real         half_length = 0.0;         // capsule only
    Vec3         axis_local  = Vec3::UnitY(); // capsule only; local axis direction before `rotation`
    std::vector<ColliderFrame> frames;      // frames[k] = pose at track frame k, kColliderAnimFPS fixed rate

    // velocity_basis[k] = compute_animated_collider_velocity_basis(*this, k, ...) — see
    // precompute_velocity_basis. Empty until that's called; indexed and clamped the same way as `frames`.
    std::vector<AnimatedColliderVelocityBasis> velocity_basis;
};

inline RigidPose to_diffpd_pose(const ColliderFrame& f)
{
    return { blender_to_diffpd(f.position), blender_to_diffpd_rotation(f.rotation) };
}

// The diffpd-space (position, orientation) of a track at integer `frame`, clamped so an overrun sim
// holds the last pose instead of crashing.
inline RigidPose collider_animation_pose_at(const ColliderAnimation& anim, int frame)
{
    return to_diffpd_pose(anim.frames[std::clamp(frame, 0, (int)anim.frames.size() - 1)]);
}

// How a physics step maps to the state and time the world is posed at. pd_contact builds one per run and records
// it on the Tape; backward_pd_contact and the viewer read it back, so every pass poses colliders, skinned
// bodies and waist pins identically (the adjoint needs forward and backward to see the same state per step).
struct StepClock
{
    Real dt         = 1.0 / kColliderAnimFPS;
    int  first_step = 0;     // global step of state 0 (a run resumed mid-trajectory, a chunk, starts later)
    int  substeps   = 1;     // physics steps per track frame (= frame_substeps: the sim runs at kColliderAnimFPS)
    bool frozen     = false; // settle: every collider stays at its frame-0 pose

    // Local step `s` (0-based) ends in global state first_step + s + 1: where x_tilde sits, so where contact
    // is detected and the world must be posed. The backward step t (1-based) is the same state, end_state(t - 1).
    int  end_state(int s) const { return first_step + s + 1; }
    Real contact_time(int s) const { return frozen ? 0.0 : end_state(s) * dt; }
};

// Where physics state `step` falls on a track: integer `frame` plus a fraction `alpha` in [0,1) of
// the way to frame + 1. With substeps == 1 this is always { step, 0 }.
struct AnimSample { int frame; Real alpha; };

inline AnimSample anim_sample_at_step(int substeps, int step)
{
    return { step / substeps, Real(step % substeps) / substeps };
}

// The raw (Blender-space) track pose at physics state `step`: lerp position / slerp rotation between
// the two bracketing frames — exactly the constant v/omega rigid motion the velocity basis assumes over a
// bracket, so the stamped geometry and the contact velocity agree at every substep. alpha == 0 returns
// the stored frame untouched. Clamped like collider_animation_pose_at.
inline ColliderFrame collider_frame_at_step(const ColliderAnimation& anim, int substeps, int step)
{
    const AnimSample     s    = anim_sample_at_step(substeps, step);
    const int            last = (int)anim.frames.size() - 1;
    const ColliderFrame& f0   = anim.frames[std::clamp(s.frame, 0, last)];
    if (s.alpha == 0.0) return f0;
    const ColliderFrame& f1   = anim.frames[std::clamp(s.frame + 1, 0, last)];
    return { f0.position + s.alpha * (f1.position - f0.position), f0.rotation.slerp(s.alpha, f1.rotation) };
}

// The diffpd-space pose of a track at physics state `step` (interpolated, see collider_frame_at_step).
inline RigidPose collider_animation_pose_at_step(const ColliderAnimation& anim, int substeps, int step)
{
    return to_diffpd_pose(collider_frame_at_step(anim, substeps, step));
}

// Stamps the track's pose at physics state `step` onto c's base fields.
inline void apply_collider_at_step(Collider& c, const ColliderAnimation& anim, int substeps, int step)
{
    const ColliderFrame f = collider_frame_at_step(anim, substeps, step);

    const Vec3 position = blender_to_diffpd(f.position);

    if (c.type == ColliderType::Capsule)
    {
        const Vec3 world_axis = blender_to_diffpd(f.rotation * anim.axis_local).normalized();
        c.capsule_p0     = position - anim.half_length * world_axis;
        c.capsule_p1     = position + anim.half_length * world_axis;
        c.capsule_radius = anim.radius;
    }
    else if (c.type == ColliderType::Sphere)
    {
        c.sphere_center = position;
        c.sphere_radius = anim.radius;
    }
    else if (c.type == ColliderType::Mesh)
    {
        c.mesh_pose = to_diffpd_pose(f);
    }
}

// Extracts linear/angular velocity from the two frames bracketing `frame` (quaternion log-map for
// omega) so points later use v + omega x (x - pivot). Guards quaternion double-cover (negate
// q[frame+1] if dot < 0, for the shortest arc) and the small-angle 0/0 case (omega = 0).
inline AnimatedColliderVelocityBasis compute_animated_collider_velocity_basis(
    const ColliderAnimation& anim, int frame, Real dt)
{
    const RigidPose pose0 = collider_animation_pose_at(anim, frame);
    const RigidPose pose1 = collider_animation_pose_at(anim, frame + 1);

    AnimatedColliderVelocityBasis basis;
    basis.linear_velocity = (pose1.position - pose0.position) / dt;

    Eigen::Quaternion<Real> q1 = pose1.rotation;
    if (pose0.rotation.dot(q1) < 0.0) q1.coeffs() = -q1.coeffs(); // shortest-arc fix (double cover)
    const Eigen::Quaternion<Real> dq = (q1 * pose0.rotation.conjugate()).normalized();

    static constexpr Real kEps = 1e-12;
    const Real            sin_half = dq.vec().norm();
    if (sin_half > kEps)
    {
        const Real angle  = 2.0 * std::atan2(sin_half, dq.w());
        basis.omega_vec   = (dq.vec() / sin_half) * (angle / dt);
    }
    return basis;
}

// Fills anim.velocity_basis[k] for every frame k, using the track's own bake period (1/kColliderAnimFPS)
// as the bracket duration — not the sim's physics dt, which may subdivide a bracket into several
// steps. Call once per run (prepare_world does this right after loading the tracks).
inline void precompute_velocity_basis(ColliderAnimation& anim)
{
    const Real anim_dt = 1.0 / kColliderAnimFPS;
    anim.velocity_basis.resize(anim.frames.size());
    for (int k = 0; k < (int)anim.frames.size(); ++k)
        anim.velocity_basis[k] = compute_animated_collider_velocity_basis(anim, k, anim_dt);
}

// The precomputed basis of the bracket physics state `step` lies in (clamped like `frames`).
inline const AnimatedColliderVelocityBasis& animated_collider_basis_at_step(const ColliderAnimation& anim, int substeps, int step)
{
    const int frame = std::clamp(anim_sample_at_step(substeps, step).frame, 0, (int)anim.velocity_basis.size() - 1);
    return anim.velocity_basis[frame];
}

// Velocity of `world_point` on an animated collider at physics state `step` — the only thing that
// varies per contact. The bracket's basis is constant; only the collider's current pose moves inside
// it, to the interpolated pose at alpha (see collider_frame_at_step): v + omega x (x - pivot), with the
// pivot at the current (interpolated) position.
inline Vec3 animated_collider_point_velocity(const ColliderAnimation& anim, int substeps, int step, const Vec3& world_point)
{
    const AnimatedColliderVelocityBasis& basis = animated_collider_basis_at_step(anim, substeps, step);
    const RigidPose                      pose  = collider_animation_pose_at_step(anim, substeps, step);
    return basis.linear_velocity + basis.omega_vec.cross(world_point - pose.position);
}
