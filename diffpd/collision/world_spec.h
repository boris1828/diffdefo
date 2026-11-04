#pragma once

// What a run collides with, before it is loaded: the collider list and contact options, the animation track and
// whether the cloth's pins follow it. prepare_world (app/world_setup.h) turns it into a CollisionWorld.

#include "collision/world.h"

#include <string>
#include <vector>

struct WorldSpec
{
    // UI-managed list; starts empty, "+"/"-" grows/shrinks it (empty disables contact against
    // manually-placed colliders — animated colliders loaded from Blender are separate and unaffected).
    std::vector<Collider> colliders = {};

    // Contact model, margin and recheck (re-test the current iterate inside pd_contact's solver loop and add newly
    // penetrating particles to the step's contact set, see merge_detected_contacts in collision/detect.h).
    ContactOptions contact;

    // Minimum penetration depth (world units) for the post-solve "unresolved contacts" stat
    // (Tape::unresolved_contacts) to count a particle as still colliding; shallower residual
    // penetration is treated as resolved. Does not affect the forward solve's own contact detection.
    // Config-screen picker restricts this to one of 1e-1/1e-2/1e-3/1e-4 (see unresolved_threshold_field).
    Real unresolved_contact_threshold = 1e-2;

    // Animated-collider track loaded for the run: a filename in diffpd/animation, or "" for none.
    std::string animation_file = "fem_skinned_body_stand_to_walk.json";

    // Rigidly attaches pinned cloth vertices to the hip collider instead of a fixed rest position.
    // Only takes effect if the selected animation has that collider.
    bool waist_attach_enabled = true;
};
