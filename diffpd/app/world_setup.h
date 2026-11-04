#pragma once

// What a run (or a settle) collides with, built from the config, and the waist attachment of a fresh cloth.

#include "app/config.h"
#include "cloth/waist.h"
#include "collision/animation_io.h"
#include "solver/impulse/anderson.h"

// ----------------
//      MAIN
// ----------------


// What one run (or settle) collides with, plus which track / skinned body the waist attaches to.
struct WorldSetup
{
    CollisionWorld collision;
    int waist_attach_anim_id = -1, waist_attach_skinned_id = -1;
};

// Attaches obj's pinned vertices to the rigid hip track (anim_id >= 0) or to the pelvis of a skinned body
// (skinned_id >= 0), baking from their current positions, then poses them at state 0. No-op if neither.
inline void attach_waist(Object& obj, const WorldSetup& world)
{
    const CollisionWorld& cw = world.collision;
    if (world.waist_attach_anim_id >= 0)
        bake_waist_attachment(obj, world.waist_attach_anim_id, cw.animations);
    else if (world.waist_attach_skinned_id >= 0)
        bake_waist_attachment_skinned(obj, cw, world.waist_attach_skinned_id,
                                      find_bone(cw.skinned_meshes[world.waist_attach_skinned_id], kWaistAttachmentBoneName));
    else
        return;
    update_waist_attachment(obj, cw, 1, 0);
}

// Builds everything a run (or a settle) collides with from the spec: the manual colliders, the animation track + its
// velocity basis, the skinned bodies, the contact options, and which of them the waist attaches to.
inline WorldSetup prepare_world(const WorldSpec& cfg)
{
    WorldSetup      world;
    CollisionWorld& cw = world.collision;

    // the collider list is UI-managed (cfg.colliders); animated colliders are appended separately and aren't
    // editable from the config screen.
    cw.colliders             = cfg.colliders;
    cw.contact               = cfg.contact;
    if (!cfg.animation_file.empty())
        load_collider_animation(animation_path(cfg.animation_file), cw);
    for (ColliderAnimation& anim : cw.animations)
        precompute_velocity_basis(anim);

    // Resolve the hardcoded collider/bone names once per run: the rigid hip track, else the pelvis of a
    // skinned body. Missing warns and disables.
    if (cfg.waist_attach_enabled)
    {
        for (int i = 0; i < (int)cw.animations.size(); ++i)
            if (cw.animations[i].name == kWaistAttachmentColliderName) { world.waist_attach_anim_id = i; break; }
        if (world.waist_attach_anim_id < 0)
            for (int i = 0; i < (int)cw.skinned_meshes.size(); ++i)
                if (find_bone(cw.skinned_meshes[i], kWaistAttachmentBoneName) >= 0) { world.waist_attach_skinned_id = i; break; }
        if (world.waist_attach_anim_id < 0 && world.waist_attach_skinned_id < 0)
            WARNING("Waist Attachment enabled but no '" << kWaistAttachmentColliderName << "' collider animation or '"
                    << kWaistAttachmentBoneName << "' skinned-body bone was loaded; disabling for this run.");
    }
    return world;
}

// The same from the whole config; also hands the Anderson history size to the solver.
inline WorldSetup prepare_world(const AppConfig& cfg)
{
    anderson_m = cfg.solver.anderson_m;
    return prepare_world(cfg.world);
}
