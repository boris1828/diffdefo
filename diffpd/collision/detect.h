#pragma once

// Particle-vs-collider detection: loops the world's colliders and the particles, asks each shape for the closest
// point (collision/shapes.h) and collects the penetrating ones as Contacts. Takes stacked positions (3n), never a
// cloth Object.

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
