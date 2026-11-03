#pragma once

// Pinned cloth vertices that follow a moving body, and the one call that poses everything that moves in a step.
// Depends on cloth/object.h and collision/.

#include "cloth/object.h"
#include "collision/world.h"

// ----------------
//  WAIST ATTACHMENT
// ----------------
// Rigidly attaches every pinned cloth vertex to one animated collider instead of a fixed rest
// position — e.g. a waistband following the hip through a walk cycle.


// Call once after the Object is built: bakes each pinned vertex's offset relative to the track's
// frame-0 pose, and marks obj as attached to it.
inline void bake_waist_attachment(Object& obj, int anim_id, const std::vector<ColliderAnimation>& anims)
{
    const RigidPose pose0 = collider_animation_pose_at(anims[anim_id], 0);
    obj.waist.local_offset.resize(obj.mesh.pinned_rest.size());
    for (size_t k = 0; k < obj.mesh.pinned_rest.size(); ++k)
        obj.waist.local_offset[k] = pose0.rotation.conjugate() * (obj.mesh.pinned_rest[k] - pose0.position);
    obj.waist.anim_id = anim_id;
}

// The binding that rides a pin at frame-0 world position `p0` on the given bones/weights (up to
// PinBinding::kMaxInfluences; zero weights are skipped). Weights are used as given (callers pass a
// partition of unity, e.g. {pelvis: 1}).
inline PinBinding make_pin_binding(const SkinnedMesh& body, const Vec3& p0,
                                   const std::vector<std::pair<int, Real>>& bones_and_weights)
{
    PinBinding b;
    b.weight = { 0.0, 0.0, 0.0, 0.0 };
    int slot = 0;
    for (const auto& [bone, w] : bones_and_weights)
    {
        if (w == 0.0 || slot >= PinBinding::kMaxInfluences) continue;
        const AffineTransform& S0 = body.skin_at(0, bone);
        b.bone[slot]   = bone;
        b.weight[slot] = w;
        b.q[slot]      = S0.R.inverse() * (p0 - S0.t); // S0^-1 p0
        ++slot;
    }
    return b;
}

// Call once after the Object is built (pins at their frame-0 positions): attaches every pinned vertex
// to skinned body `skinned_id`, rigidly to bone `bone` for now (weight 1). To use per-vertex weights
// instead (e.g. copied from the nearest body vertices), change only the list given to make_pin_binding.
inline void bake_waist_attachment_skinned(Object& obj, const CollisionWorld& world, int skinned_id, int bone)
{
    const SkinnedMesh& body = world.skinned_meshes[skinned_id];
    obj.waist.binding.resize(obj.mesh.pinned_rest.size());
    for (size_t k = 0; k < obj.mesh.pinned_rest.size(); ++k)
        obj.waist.binding[k] = make_pin_binding(body, obj.mesh.pinned_rest[k], { { bone, 1.0 } });
    obj.waist.skinned_id = skinned_id;
}

// A pin's world position at physics state `step`.
inline Vec3 pin_position_at_step(const SkinnedMesh& body, const PinBinding& pin, int substeps, int step)
{
    Vec3 p = Vec3::Zero();
    for (int k = 0; k < PinBinding::kMaxInfluences; ++k)
        if (pin.weight[k] != 0.0) p += pin.weight[k] * skin_matrix_at_step(body, pin.bone[k], substeps, step).apply(pin.q[k]);
    return p;
}

// Re-poses just mesh.pinned_rest from the attached body's pose at physics state `step` — the part
// rendering needs; shared by update_waist_attachment and the playback viewer (which has no live Object).
inline void update_waist_attachment_mesh(SimMesh& mesh, const WaistAttachment& waist,
                                          const CollisionWorld& world, int substeps, int step)
{
    if (waist.anim_id >= 0)
    {
        const RigidPose pose = collider_animation_pose_at_step(world.animations[waist.anim_id], substeps, step);
        for (size_t k = 0; k < mesh.pinned_rest.size(); ++k)
            mesh.pinned_rest[k] = pose.position + pose.rotation * waist.local_offset[k];
    }
    else if (waist.skinned_id >= 0)
    {
        const SkinnedMesh& body = world.skinned_meshes[waist.skinned_id];
        for (size_t k = 0; k < mesh.pinned_rest.size(); ++k)
            mesh.pinned_rest[k] = pin_position_at_step(body, waist.binding[k], substeps, step);
    }
}

// Per-step: re-poses mesh.pinned_rest (which the energies read their anchors from) from the attached
// body's pose at physics state `step`. No-op if waist attachment isn't enabled.
inline void update_waist_attachment(Object& obj, const CollisionWorld& world, int substeps, int step)
{
    if (!obj.waist.active()) return;

    update_waist_attachment_mesh(obj.mesh, obj.waist, world, substeps, step);
}

// One step's pose for everything that moves: the world's colliders and skinned bodies, then the cloth's pins.
// Shared by pd_contact (forward) and backward_pd_contact, which get `substeps` and `step` from the same StepClock,
// so the two can never drift apart on how a step index is turned into a pose: both must see exactly the same
// collider/anchor state at a given step for the adjoint to be correct. No-op for the pins if waist attachment isn't enabled.
inline void advance_world(CollisionWorld& world, Object& obj, int substeps, int step)
{
    world.advance_to(substeps, step);
    update_waist_attachment(obj, world, substeps, step);
}

// Config-screen / run-start check: how many of `obj`'s particles (free ones, then pinned ones) lie inside
// a skinned body at its state 0. Contact has no push-out, so a cloth that starts inside a body stays stuck.
// Uses world's skinned bodies, stamping them at state 0.
inline void count_cloth_inside_skinned(const Object& obj, CollisionWorld& world, int& free_inside, int& pinned_inside)
{
    free_inside = pinned_inside = 0;
    for (int id = 0; id < (int)world.skinned_meshes.size(); ++id)
    {
        world.stamp_skinned_body(id, 1, 0);
        const SkinnedMesh&  body = world.skinned_meshes[id];
        const SkinnedState& st   = world.skinned_states[id];
        const auto inside = [&](const Vec3& p)
        { return mesh_closest(st.x, body.rest.triangles, st.normals, st.bvh, p).dist < 0.0; };
        for (Index i = 0; i < obj.num_particles(); ++i)
            free_inside += inside(obj.x.segment<3>(3 * i));
        for (const Vec3& p : obj.mesh.pinned_rest)
            pinned_inside += inside(p);
    }
}
