#pragma once

// Particle-vs-collider detection: loops the world's colliders and the particles, asks each shape for the closest
// point (collision/shapes.h) and collects the penetrating ones as Contacts. Takes stacked positions (3n), never a
// cloth Object.

#include "collision/animation.h"
#include "collision/shapes.h"

// Each particle contacts at most one collider: the first (in list order) it's found penetrating.
// `claimed` tracks assigned particles; `pre_claimed` optionally seeds it to exclude particles the
// caller already holds a contact for. `penetration_threshold` shrinks the surface inward by that much
// before testing (0.0 = exact surface, used by the forward solve; pd_contact's "unresolved" re-check
// passes a small positive slack so negligible residual penetration doesn't count). A negative threshold is a
// margin: particles within that distance outside the surface count too (speculative contact).
inline Contacts detect_contacts(const CollisionWorld& world, const Positions& x, Real time, Real penetration_threshold = 0.0,
                                const std::vector<bool>* pre_claimed = nullptr)
{
    const Index n = x.rows() / 3; // x is the stacked positions of one object: the only thing detection needs from it
    Contacts contacts;
    std::vector<bool> claimed = pre_claimed ? *pre_claimed : std::vector<bool>(n, false);
    ASSERT((Index)claimed.size() == n, "detect_contacts: pre_claimed has " << claimed.size() << " entries for " << n << " particles");

    for (int ci = 0; ci < (int)world.colliders.size(); ++ci)
    {
        const Collider& collider = world.colliders[ci];
        if (collider.type == ColliderType::None) continue;

        const ColliderPose pose = collider_pose_at(collider, time);
        AABB               box  = collider_aabb(world, collider, pose); // valid only for finite shapes (Sphere/Capsule/Mesh/Skinned)
        const Vec3 grow = Vec3::Constant(std::max(0.0, -penetration_threshold)); // a margin grows the box too
        box.min -= grow; box.max += grow;

        for (Index i = 0; i < n; ++i)
        {
            if (claimed[i]) continue;

            const Vec3 pos = x.segment<3>(3*i);

            // Cheap reject before the real distance test; skipped for Cylinder/Plane (no finite bound).
            if (box.valid && !aabb_contains(box, pos)) continue;

            Contact c;
            if (contact_geometry(world, collider, ci, pose, static_cast<ParticleId>(i), pos, c) < -penetration_threshold)
            {
                contacts.push_back(c);
                claimed[i] = true;
            }
        }
    }

    return contacts;
}

// Active-set growth inside pd_contact's iteration loop: catches particles the springs drag into a
// collider mid-step (e.g. cloth "following" a limb from behind), which the once-per-step detection
// against x_tilde misses. Union only — existing contacts are never removed or re-targeted. New
// penetrators are found at `x_test` (current iterate) but their geometry is evaluated at `x_geom`
// (= x_tilde), keeping the backward pass's n = n(x_tilde) assumption exact. Appends new contacts at
// the back and returns how many, so the caller can post-process just those.
inline int merge_detected_contacts(const CollisionWorld& world, Contacts& contacts, const Positions& x_test, const Positions& x_geom, Real time)
{
    std::vector<bool> claimed(x_test.rows() / 3, false);
    for (const Contact& c : contacts) claimed[c.particle] = true;

    const Contacts fresh = detect_contacts(world, x_test, time, 0.0, &claimed);
    for (const Contact& hit : fresh)
    {
        const Collider& collider = world.colliders[hit.collider_id];
        Contact c;
        contact_geometry(world, collider, hit.collider_id, collider_pose_at(collider, time),
                         hit.particle, x_geom.segment<3>(3 * hit.particle), c);
        contacts.push_back(c);
    }
    return (int)fresh.size();
}

// Stamps what the solve and the backward pass read from contacts[begin, end), found at `step` of `clock`; x_minus is the
// step's start-of-step positions. Relative-velocity contact: v_c at the surface point (the collider's motion), and, if
// world.collect_backward, dvc_dx = G = dv_c/dx~ (the backward pass's velocity correction).
// Speculative contact instead re-evaluates the whole contact geometry (normal, inv_r, axis, surface point) and the gap
// phi(x-) at x-, against the collider's end-of-step pose (the pose detection used), and leaves v_c = 0: the collider's
// motion is already in the gap. The constraint phi(x-) + h v+.n >= 0 is then the first-order expansion of phi(x+) >= 0
// at x-, and nothing in it depends on x~ (x~ only picks which particles are contacts).
inline void stamp_contacts(const CollisionWorld& world, const StepClock& clock, int step, const Positions& x_minus,
                           Contacts& contacts, size_t begin, size_t end)
{
    const Real contact_time = clock.contact_time(step);

    if (world.contact.gap())
    {
        for (size_t j = begin; j < end; ++j)
        {
            Contact& c = contacts[j];
            const Collider& collider = world.colliders[c.collider_id];
            const Vec3 pos = x_minus.segment<3>(3 * c.particle);
            Contact at_x_minus;
            const Real gap = contact_geometry(world, collider, c.collider_id, collider_pose_at(collider, contact_time), c.particle, pos, at_x_minus);
            c = at_x_minus;
            c.gap = gap;
            // grad phi = (x - closest point) / phi off the surface (exact for every shape); the normal on it
            if (world.collect_backward)
                c.gap_grad = std::abs(gap) > 1e-12 ? Vec3((pos - c.surface_point) / gap) : c.normal;
        }
        return;
    }

    // Velocity of an animated collider is a precomputed, static property of its track (ColliderAnimation::velocity_basis),
    // evaluated at the end-of-step state the collider was stamped at, at the cached closest surface point.
    auto contact_velocity = [&](const Contact& c) -> Vec3
    {
        if (clock.frozen) return Vec3::Zero();
        const Collider& collider = world.colliders[c.collider_id];
        // Skinned body: the barycentric blend of the contact triangle's vertex velocities.
        if (collider.type == ColliderType::Skinned)
            return skinned_surface_velocity(world, collider.skinned_id, c.triangle_id, c.bary);
        if (!collider.animated)
            return collider_point_velocity(collider, c.surface_point, contact_time);
        return animated_collider_point_velocity(world.animations[collider.anim_id], clock.substeps, clock.end_state(step), c.surface_point);
    };

    auto velocity_gradient = [&](const Contact& c) -> Mat3
    {
        if (clock.frozen) return Mat3::Zero();
        const Collider& collider = world.colliders[c.collider_id];
        if (collider.type == ColliderType::Skinned)
            return c.triangle_id >= 0 ? skinned_velocity_gradient(world, collider.skinned_id, c.triangle_id, c.feature, c.feature_index)
                                      : Mat3::Zero().eval();
        const Vec3 omega = collider.animated
            ? animated_collider_basis_at_step(world.animations[collider.anim_id], clock.substeps, clock.end_state(step)).omega_vec
            : collider_omega_vec(collider);
        return rigid_velocity_gradient(collider, c, omega);
    };

    for (size_t j = begin; j < end; ++j)
    {
        contacts[j].v_c = contact_velocity(contacts[j]);
        if (world.collect_backward) contacts[j].dvc_dx = velocity_gradient(contacts[j]);
    }
}
