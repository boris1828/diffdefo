#pragma once

// Loading a baked collider animation (.json) into a CollisionWorld; the bodies are in collision/animation_io.cpp.

#include "collision/world.h"

#include <string>
#include <vector>

// Name of the collider (Blender empty / animation .json entry) that Waist Attachment pins to.
inline constexpr const char* kWaistAttachmentColliderName = "collider_hip";

// Bone of a skinned body that Waist Attachment pins to (rigidly, weight 1); see bake_waist_attachment_skinned.
inline constexpr const char* kWaistAttachmentBoneName = "pelvis";

// Replaces world.animations and the mesh / skinned-body registries from an animation .json, and appends one
// Collider (+ ColliderAnimation) per supported entry to world.colliders; the real pose is applied later by
// apply_collider_at_step, not by this loader.
void load_collider_animation(const std::string& path, CollisionWorld& world);

// Animation .json files live in COLLIDER_ANIM_DIR_DEFAULT (diffpd/animation); AppConfig::animation_file
// names one of them by filename, or is empty for "None".
// Sorted filenames (with extension) of every *.json in that folder; empty if it can't be read.
std::vector<std::string> list_animation_files();
std::string              animation_path(const std::string& filename);

// What the config screen needs to show the selected animation before a run (display-only).
struct AnimationPreview
{
    CollisionWorld        world;                     // colliders: the animated ones posed at frame 0, plus the assets they index
    bool                  has_waist = false;         // true if the animation has a waist-attachment target: the
                                                     // hip collider (rigid) or a pelvis bone (skinned body)
    bool                  waist_snaps_origin = false; // rigid hip: the cloth origin snaps to waist_default_origin;
                                                      // skinned body: the cloth is placed by hand, no snap
    Vec3                  waist_default_origin = Vec3::Zero(); // the hip collider's frame-0 position
    bool                  has_skinned = false;       // the animation is a skinned body (state 0 is stamped)
};
AnimationPreview load_animation_preview(const std::string& filename); // empty filename -> empty preview
