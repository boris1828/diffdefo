#pragma once

// Collider shapes (Sphere, Cylinder, Plane, Capsule, Mesh, Skinned), their constant translation / axis-locked
// rotation, and the pose at a time. Depends only on core/. The geometry a Mesh or Skinned collider points at
// lives in the CollisionWorld (collision/world.h), not here.

#include "core/diffpd_core.h"

// ----------------
//   CONVENTIONS
// ----------------

// The track's own fixed bake rate — a property of the imported data. The sim must run at this FPS
// (one sim frame == one track frame); frame_substeps then splits each track frame into
// StepClock::substeps physics steps. compute_animated_collider_velocity_basis is
// defined per bracket of this duration, not per physics step.
inline constexpr Real kColliderAnimFPS = 60.0;

// Blender is Z-up, diffpd is Y-up: a -90deg turn about X (Blender Z -> diffpd Y, Blender Y -> -diffpd Z).
// A proper rotation, so safe for both positions and directions.
inline Vec3 blender_to_diffpd(const Vec3& v)
{
    return Vec3(v.x(), v.z(), -v.y());
}

// Same -90deg-about-X change of basis as blender_to_diffpd(), applied to an orientation via conjugation.
inline Eigen::Quaternion<Real> blender_to_diffpd_rotation(const Eigen::Quaternion<Real>& q)
{
    static const Eigen::Quaternion<Real> kBasisChange(std::sqrt(Real(0.5)), -std::sqrt(Real(0.5)), 0.0, 0.0);
    return (kBasisChange * q * kBasisChange.conjugate()).normalized();
}

// None, Mesh and Skinned come last so existing Sphere/Cylinder/Plane/Capsule indices (GuiToggleGroup's
// active index in the config screen) don't shift. Mesh and Skinned are animation-only, not
// UI-selectable. Skinned = a linear-blend-skinned triangle body, see SkinnedMesh.
enum class ColliderType { Sphere, Cylinder, Plane, Capsule, None, Mesh, Skinned };

// Locks rotation to a world axis, so it's a simple 2D spin (no axis-angle machinery needed).
enum class RotationAxis { X, Y, Z, None };

struct RigidPose
{
    Vec3 position                   = Vec3::Zero();
    Eigen::Quaternion<Real> rotation = Eigen::Quaternion<Real>::Identity();
};

struct Collider
{
    ColliderType type = ColliderType::Sphere;

    // Sphere: center + radius
    Vec3 sphere_center = Vec3(0.0, -10.0, 0.0);
    Real sphere_radius = 1.0;

    // Infinite cylinder: point on the axis + unit axis direction + radius
    Vec3 cylinder_origin = Vec3::Zero();
    Vec3 cylinder_axis   = Vec3::UnitY();
    Real cylinder_radius = 1.0;

    // Plane: point on the plane + unit outward normal
    Vec3 plane_origin = Vec3::Zero();
    Vec3 plane_normal = Vec3::UnitY();

    // Capsule: segment endpoints + radius (cylindrical body capped by two hemispheres)
    Vec3 capsule_p0     = Vec3::Zero();
    Vec3 capsule_p1     = Vec3::UnitY();
    Real capsule_radius = 1.0;

    // Mesh: index into `trimeshes` + current rigid pose (stamped per step from the animation track)
    int       mesh_id   = -1;
    RigidPose mesh_pose;

    // Skinned: index into `skinned_meshes`. Not `animated` (yet): that flag means a rigid
    // ColliderAnimation track, which a skinned body doesn't have.
    int skinned_id = -1;

    Vec3 velocity = Vec3::Zero(); // shared constant translation velocity (m/s), whichever shape is active

    // Shared constant rotation, applied on top of the translation above.
    RotationAxis rotation_axis   = RotationAxis::None;
    Vec3         rotation_origin = Vec3::Zero(); // point the rotation axis passes through
    Real         omega           = 0.0;          // signed angular velocity (rad/s)

    // If true, base fields are overwritten every step from a baked per-frame track (see
    // ColliderAnimation) instead of velocity/omega. Participates in forward contact but not yet
    // the backward pass (no way to differentiate a baked trajectory); not editable in the UI.
    bool animated = false;
    int  anim_id  = -1; // index into the AppConfig-level std::vector<ColliderAnimation>
};

inline Collider make_sphere(Vec3 center, Real radius, Vec3 velocity = Vec3::Zero())
{
    Collider c;
    c.type          = ColliderType::Sphere;
    c.sphere_center = center;
    c.sphere_radius = radius;
    c.velocity      = velocity;
    return c;
}

inline Collider make_cylinder(Vec3 origin, Vec3 axis, Real radius, Vec3 velocity = Vec3::Zero())
{
    Collider c;
    c.type            = ColliderType::Cylinder;
    c.cylinder_origin = origin;
    c.cylinder_axis   = axis;
    c.cylinder_radius = radius;
    c.velocity        = velocity;
    return c;
}

inline Collider make_plane(Vec3 origin, Vec3 normal, Vec3 velocity = Vec3::Zero())
{
    Collider c;
    c.type         = ColliderType::Plane;
    c.plane_origin = origin;
    c.plane_normal = normal;
    c.velocity     = velocity;
    return c;
}

inline Collider make_capsule(Vec3 p0, Vec3 p1, Real radius, Vec3 velocity = Vec3::Zero())
{
    Collider c;
    c.type           = ColliderType::Capsule;
    c.capsule_p0     = p0;
    c.capsule_p1     = p1;
    c.capsule_radius = radius;
    c.velocity       = velocity;
    return c;
}

inline Real collider_surface_radius(const Collider& c)
{
    switch (c.type)
    {
        case ColliderType::Sphere:   return c.sphere_radius;
        case ColliderType::Cylinder: return c.cylinder_radius;
        case ColliderType::Capsule:  return c.capsule_radius;
        default:                     return 0.0; // Plane has no radius
    }
}

inline Vec3 rotation_axis_vector(RotationAxis axis)
{
    switch (axis)
    {
        case RotationAxis::X: return Vec3::UnitX();
        case RotationAxis::Y: return Vec3::UnitY();
        case RotationAxis::Z: return Vec3::UnitZ();
        default:              return Vec3::Zero();
    }
}

inline Vec3 collider_transform_point(const Collider& c, const Vec3& p, Real time)
{
    Vec3 out = p;
    if (c.rotation_axis != RotationAxis::None && c.omega != 0.0)
    {
        const Eigen::AngleAxis<Real> R(c.omega * time, rotation_axis_vector(c.rotation_axis));
        out = c.rotation_origin + R * (p - c.rotation_origin);
    }
    return out + c.velocity * time;
}

inline Vec3 collider_transform_direction(const Collider& c, const Vec3& d, Real time)
{
    if (c.rotation_axis == RotationAxis::None || c.omega == 0.0) return d;
    const Eigen::AngleAxis<Real> R(c.omega * time, rotation_axis_vector(c.rotation_axis));
    return R * d;
}

inline Vec3 collider_point_velocity(const Collider& c, const Vec3& x, Real time)
{
    if (c.rotation_axis == RotationAxis::None || c.omega == 0.0) return c.velocity;
    const Vec3 omega_vec     = c.omega * rotation_axis_vector(c.rotation_axis);
    const Vec3 current_pivot = collider_transform_point(c, c.rotation_origin, time);
    return c.velocity + omega_vec.cross(x - current_pivot);
}

struct ColliderPose
{
    Vec3 sphere_center;
    Vec3 cylinder_origin, cylinder_axis;
    Vec3 plane_origin,    plane_normal;
    Vec3 capsule_p0,      capsule_p1;
};

// Cheap per-particle reject before the real distance test. Only meaningful for finite shapes
// (Sphere, Capsule, Mesh); Cylinder/Plane are unbounded so `valid` stays false and the AABB is skipped.
struct AABB
{
    Vec3 min   = Vec3::Zero();
    Vec3 max   = Vec3::Zero();
    bool valid = false;
};


inline ColliderPose collider_pose_at(const Collider& c, Real time)
{
    ColliderPose pose{};
    switch (c.type)
    {
        case ColliderType::Sphere:
            pose.sphere_center = collider_transform_point(c, c.sphere_center, time);
            break;
        case ColliderType::Cylinder:
            pose.cylinder_origin = collider_transform_point(c, c.cylinder_origin, time);
            pose.cylinder_axis   = collider_transform_direction(c, c.cylinder_axis, time);
            break;
        case ColliderType::Plane:
            pose.plane_origin = collider_transform_point(c, c.plane_origin, time);
            pose.plane_normal = collider_transform_direction(c, c.plane_normal, time);
            break;
        case ColliderType::Capsule:
            pose.capsule_p0 = collider_transform_point(c, c.capsule_p0, time);
            pose.capsule_p1 = collider_transform_point(c, c.capsule_p1, time);
            break;
        case ColliderType::None:
        case ColliderType::Mesh:    // pose lives in Collider::mesh_pose, not ColliderPose
        case ColliderType::Skinned: // deformed vertices, not a rigid pose
            break;
    }
    return pose;
}
