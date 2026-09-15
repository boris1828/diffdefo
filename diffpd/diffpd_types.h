#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <functional>

#define WARNING(message) \
    do { \
        std::ostringstream _oss; \
        _oss << message; \
        std::fprintf(stderr, "[WARNING] %s:%d: %s\n", \
                     __FILE__, __LINE__, _oss.str().c_str()); \
    } while (0)

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::ostringstream _oss; \
            _oss << message; \
            std::fprintf(stderr, "[ASSERT] %s:%d: (%s) %s\n", \
                         __FILE__, __LINE__, #condition, _oss.str().c_str()); \
            std::abort(); \
        } \
    } while (0)

// ----------------
//      TYPES
// ----------------

using Real         = double;
using Index        = Eigen::Index;
using VertexId     = int;
using ParticleId   = int;
using ConstraintId = int;

using Vec3 = Eigen::Matrix<Real, 3, 1>;

using Mat3 = Eigen::Matrix<Real, 3, 3>;

using RealVecX = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

using PointsX = Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>;

using SparseMat = Eigen::SparseMatrix<Real>;
using Triplet   = Eigen::Triplet<Real>;

using Cholesky = Eigen::SimplicialLLT<SparseMat>;

using MassDiag = RealVecX;

using Edge  = std::pair<VertexId, VertexId>;
using Edges = std::vector<Edge>;

constexpr Real DEFAULT_STIFFNESS = 1e3;

using Positions  = RealVecX;
using Velocities = RealVecX;
using RestMesh   = PointsX;

// ----------------
//      MESH
// ----------------

struct Object; // SimMesh::position() needs Object; defined below, method body after Object.

// Named SimMesh, not Mesh, to avoid colliding with raylib's global `struct Mesh`.
struct SimMesh
{
    struct Vertex
    {
        ParticleId dof;
    };

    std::vector<Vertex> vertices;              // output order == index order
    std::vector<Vec3>   pinned_rest;           // fixed positions for pinned verts
    std::vector<std::pair<Index,Index>> edges; // indices into `vertices`

    // Grid shape: vertices[] laid out row-major as i*height+j. Set by cloth()/skirt() so
    // renderers can reconstruct quad adjacency without re-deriving it from edges[].
    Index width  = 0;
    Index height = 0;

    // True if the i axis wraps into a closed ring (set by skirt(), false for cloth()) — tells
    // grid renderers to close the seam between the last and first column.
    bool wrap_i = false;

    Vec3 position(const Object& obj, Index vi) const; // defined after Object
};

// ----------------
//   CONSTRAINTS
// ----------------

enum class SpringType : std::uint8_t
{
    Spring2,   // both endpoints are free particles
    Spring1,   // one free particle anchored to a pinned vertex
};

struct Constraint
{
    SpringType type;
    Real       k; // stiffness
    Real       l; // rest length
    Mat3       gamma;  // local Jacobian factor: (k*l/||e||) * P_perp
    Vec3       p_star; // normalized spring correction: l * (e / ||e||)
    Vec3       e;      // edge vector

    union
    {
        struct
        {
            ParticleId i1;
            ParticleId i2;
        } spring2;

        struct
        {
            ParticleId i;
            Real       xbar[3];
            Index      pinned_index; // index into Object::mesh.pinned_rest this anchor tracks
        } spring1;
    };

    static Constraint make(SpringType type, Real k, Real l)
    {
        Constraint c;
        c.type   = type;
        c.k      = k;
        c.l      = l;
        c.gamma  = Mat3::Zero();
        c.p_star = Vec3::Zero();
        c.e      = Vec3::Zero();
        return c;
    }

    static Constraint makeSpring2(Real k, Real l, ParticleId i1, ParticleId i2)
    {
        Constraint c = make(SpringType::Spring2, k, l);
        c.spring2.i1 = i1;
        c.spring2.i2 = i2;
        return c;
    }

    static Constraint makeSpring1(Real k, Real l, ParticleId i, const Vec3& xbar, Index pinned_index)
    {
        Constraint c = make(SpringType::Spring1, k, l);
        c.spring1.i = i;
        c.spring1.xbar[0] = xbar.x();
        c.spring1.xbar[1] = xbar.y();
        c.spring1.xbar[2] = xbar.z();
        c.spring1.pinned_index = pinned_index;
        return c;
    }
};

using Constraints = std::vector<Constraint>;

struct Object
{
    Positions  x;        // current positions       (3n)
    Velocities v;        // current velocities      (3n)
    Positions  prev_x;   // x at start of step

    MassDiag    mass;    // diagonal of M,          (3n)
    Constraints constraints;

    SparseMat L;                       // L = M/h^2 + sum_i k_i G_i^T G_i   (SPD, constant)
    SparseMat C;                       // h^2 * K — elastic block; only set by construct_velocity_lhs
    std::unique_ptr<Cholesky> solver;  // factor of L; heap-allocated so Object stays moveable

    SimMesh mesh;

    // Waist attachment: if waist_attach_anim_id >= 0, pinned vertices follow that collider's
    // animated pose instead of a fixed rest position (see bake/update_waist_attachment).
    std::vector<Vec3> pin_local_offset;
    int                waist_attach_anim_id = -1; // index into a std::vector<ColliderAnimation>; -1 = disabled

    Object() = default;

    Object(const Object&)            = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&&)                 = default;
    Object& operator=(Object&&)      = default;

    Index num_particles() const { return x.rows() / 3; }
    Index num_dofs()      const { return x.rows();     }
};

inline Vec3 SimMesh::position(const Object& obj, Index vi) const
{
    const Vertex& vert = vertices[vi];
    if (vert.dof >= 0)
    {
        ASSERT(3 * vert.dof + 2 < obj.x.rows(), "mesh dof out of range for obj.x");
        return obj.x.segment<3>(3 * vert.dof);
    }
    return pinned_rest[ -(vert.dof + 1) ];
}

// ----------------
//    COLLIDER
// ----------------

// None is last so existing Sphere/Cylinder/Plane/Capsule indices (used as GuiToggleGroup's
// active index in the config screen) don't shift.
enum class ColliderType { Sphere, Cylinder, Plane, Capsule, None };

// Locks rotation to a world axis, so it's a simple 2D spin (no axis-angle machinery needed).
enum class RotationAxis { X, Y, Z, None };

// Which point's velocity stands in for "the collider's velocity" at a contact: the particle's
// own position (cheap, approximate under rotation) or the closest surface point (exact). Global,
// since it's a choice about the contact model, not a per-collider property.
enum class ContactPointMode { Particle, Surface };

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
// (Sphere, Capsule); Cylinder/Plane are unbounded so `valid` stays false and the AABB is skipped.
struct AABB
{
    Vec3 min   = Vec3::Zero();
    Vec3 max   = Vec3::Zero();
    bool valid = false;
};

inline AABB collider_aabb(const Collider& c, const ColliderPose& pose)
{
    AABB box;
    if (c.type == ColliderType::Sphere)
    {
        box.min   = pose.sphere_center - Vec3::Constant(c.sphere_radius);
        box.max   = pose.sphere_center + Vec3::Constant(c.sphere_radius);
        box.valid = true;
    }
    else if (c.type == ColliderType::Capsule)
    {
        box.min   = pose.capsule_p0.cwiseMin(pose.capsule_p1) - Vec3::Constant(c.capsule_radius);
        box.max   = pose.capsule_p0.cwiseMax(pose.capsule_p1) + Vec3::Constant(c.capsule_radius);
        box.valid = true;
    }
    return box; // Cylinder/Plane/None: valid = false, min/max unused
}

inline bool aabb_contains(const AABB& box, const Vec3& p)
{
    return (p.array() >= box.min.array()).all() && (p.array() <= box.max.array()).all();
}

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
            break;
    }
    return pose;
}

// ----------------
//  ANIMATED COLLIDERS
// ----------------
// A collider's base fields can be stamped every step from a per-frame track baked from Blender
// (collider_animation.json) instead of derived analytically from velocity/omega. See Collider::animated.

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
    Real         half_length = 0.0;         // capsule only
    Vec3         axis_local  = Vec3::UnitY(); // capsule only; local axis direction before `rotation`
    std::vector<ColliderFrame> frames;      // frames[s] = pose at step s (fixed 60fps, no substeps)
};

// Blender is Z-up, diffpd is Y-up: a -90deg turn about X (Blender Z -> diffpd Y, Blender Y -> -diffpd Z).
// A proper rotation, so safe for both positions and directions.
inline Vec3 blender_to_diffpd(const Vec3& v)
{
    return Vec3(v.x(), v.z(), -v.y());
}

// Stamps anim.frames[frame] onto c's base fields; frame is clamped so an overrun sim holds the
// last pose instead of crashing.
inline void apply_collider_frame(Collider& c, const ColliderAnimation& anim, int frame)
{
    frame = std::clamp(frame, 0, (int)anim.frames.size() - 1);
    const ColliderFrame& f = anim.frames[frame];

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
}

// ----------------
//  WAIST ATTACHMENT
// ----------------
// Rigidly attaches every pinned cloth vertex to one animated collider instead of a fixed rest
// position — e.g. a waistband following the hip through a walk cycle.

struct RigidPose
{
    Vec3 position;
    Eigen::Quaternion<Real> rotation;
};

// Same -90deg-about-X change of basis as blender_to_diffpd(), applied to an orientation via conjugation.
inline Eigen::Quaternion<Real> blender_to_diffpd_rotation(const Eigen::Quaternion<Real>& q)
{
    static const Eigen::Quaternion<Real> kBasisChange(std::sqrt(Real(0.5)), -std::sqrt(Real(0.5)), 0.0, 0.0);
    return (kBasisChange * q * kBasisChange.conjugate()).normalized();
}

// The diffpd-space (position, orientation) of a track at `frame`, clamped like apply_collider_frame.
inline RigidPose collider_animation_pose_at(const ColliderAnimation& anim, int frame)
{
    frame = std::clamp(frame, 0, (int)anim.frames.size() - 1);
    const ColliderFrame& f = anim.frames[frame];
    return { blender_to_diffpd(f.position), blender_to_diffpd_rotation(f.rotation) };
}

// How an animated collider's contact-point velocity is estimated from its baked track — see
// compute_animated_collider_velocity_basis below.
enum class AnimatedColliderVelocityMode { MaterialPointDiff, DecomposedRigid };

// Per-step data for "velocity of point X on this collider", computed once from the two bracketing
// frames and reused for every contact against that collider that step (not per-contact/iteration).
struct AnimatedColliderVelocityBasis
{
    RigidPose pose0;              // collider_animation_pose_at(anim, frame)     — used by both methods
    RigidPose pose1;              // collider_animation_pose_at(anim, frame + 1) — MaterialPointDiff only
    Vec3      linear_velocity = Vec3::Zero(); // DecomposedRigid only
    Vec3      omega_vec       = Vec3::Zero(); // DecomposedRigid only
};

// Precomputes per-step data from the two frames bracketing `frame`.
// MaterialPointDiff: finite-differences a point rigidly attached to the collider's local frame
// between pose(frame) and pose(frame+1) — exact for the discrete data, no small-angle assumption.
// DecomposedRigid: extracts linear/angular velocity from the same two frames (quaternion log-map
// for omega) so points later use v + omega x (x - pivot). Guards quaternion double-cover (negate
// q[frame+1] if dot < 0, for the shortest arc) and the small-angle 0/0 case (omega = 0).
inline AnimatedColliderVelocityBasis compute_animated_collider_velocity_basis(
    const ColliderAnimation& anim, int frame, Real dt, AnimatedColliderVelocityMode mode)
{
    AnimatedColliderVelocityBasis basis;
    basis.pose0 = collider_animation_pose_at(anim, frame);
    basis.pose1 = collider_animation_pose_at(anim, frame + 1);

    if (mode == AnimatedColliderVelocityMode::DecomposedRigid)
    {
        basis.linear_velocity = (basis.pose1.position - basis.pose0.position) / dt;

        Eigen::Quaternion<Real> q1 = basis.pose1.rotation;
        if (basis.pose0.rotation.dot(q1) < 0.0) q1.coeffs() = -q1.coeffs(); // shortest-arc fix (double cover)
        const Eigen::Quaternion<Real> dq = (q1 * basis.pose0.rotation.conjugate()).normalized();

        static constexpr Real kEps = 1e-12;
        const Real            sin_half = dq.vec().norm();
        if (sin_half > kEps)
        {
            const Real angle  = 2.0 * std::atan2(sin_half, dq.w());
            basis.omega_vec   = (dq.vec() / sin_half) * (angle / dt);
        }
    }
    return basis;
}

// Cheap per-point evaluation from an already-computed basis — the only thing that varies per contact.
inline Vec3 animated_collider_point_velocity_from_basis(const AnimatedColliderVelocityBasis& basis,
                                                          const Vec3& world_point, Real dt, AnimatedColliderVelocityMode mode)
{
    if (mode == AnimatedColliderVelocityMode::MaterialPointDiff)
    {
        const Vec3 local = basis.pose0.rotation.conjugate() * (world_point - basis.pose0.position);
        return ((basis.pose1.position + basis.pose1.rotation * local) - world_point) / dt;
    }
    return basis.linear_velocity + basis.omega_vec.cross(world_point - basis.pose0.position);
}

// Call once after the Object is built: bakes each pinned vertex's offset relative to the track's
// frame-0 pose, and marks obj as attached to it.
inline void bake_waist_attachment(Object& obj, int anim_id, const std::vector<ColliderAnimation>& anims)
{
    const RigidPose pose0 = collider_animation_pose_at(anims[anim_id], 0);
    obj.pin_local_offset.resize(obj.mesh.pinned_rest.size());
    for (size_t k = 0; k < obj.mesh.pinned_rest.size(); ++k)
        obj.pin_local_offset[k] = pose0.rotation.conjugate() * (obj.mesh.pinned_rest[k] - pose0.position);
    obj.waist_attach_anim_id = anim_id;
}

// Re-poses just mesh.pinned_rest from the collider track's pose at `frame` — the part rendering
// needs; shared by update_waist_attachment and the playback viewer (which has no live Object).
inline void update_waist_attachment_mesh(SimMesh& mesh, const std::vector<Vec3>& pin_local_offset,
                                          int anim_id, const std::vector<ColliderAnimation>& anims, int frame)
{
    if (anim_id < 0) return;
    const RigidPose pose = collider_animation_pose_at(anims[anim_id], frame);
    for (size_t k = 0; k < mesh.pinned_rest.size(); ++k)
        mesh.pinned_rest[k] = pose.position + pose.rotation * pin_local_offset[k];
}

// Per-step: re-poses mesh.pinned_rest and every Spring1's xbar from the attached collider's pose
// at `frame`. No-op if waist attachment isn't enabled.
inline void update_waist_attachment(Object& obj, const std::vector<ColliderAnimation>& anims, int frame)
{
    if (obj.waist_attach_anim_id < 0) return;

    update_waist_attachment_mesh(obj.mesh, obj.pin_local_offset, obj.waist_attach_anim_id, anims, frame);

    for (Constraint& c : obj.constraints)
    {
        if (c.type != SpringType::Spring1) continue;
        const Vec3& p = obj.mesh.pinned_rest[c.spring1.pinned_index];
        c.spring1.xbar[0] = p.x();
        c.spring1.xbar[1] = p.y();
        c.spring1.xbar[2] = p.z();
    }
}

// Builds one Collider + ColliderAnimation per entry in collider_animation.json's metadata; the
// real pose is applied later by apply_collider_frame, not by this loader.
std::vector<ColliderAnimation> load_collider_animation(const std::string& path, std::vector<Collider>& out_colliders);

// ----------------
//  CLOTH CONFIG
// ----------------

// How the cloth is anchored: NONE = free fall, CORNERS = two top corners, ROW = entire first row.
enum class PinMode { NONE, CORNERS, ROW };

// HORIZONTAL: flat in XZ, falls and swings under gravity. VERTICAL: in XY, j=0 row at top,
// hanging straight down from the start.
enum class HangingMode { HORIZONTAL, VERTICAL };

namespace ClothFlags
{
    constexpr uint8_t STRETCH = 1 << 0;
    constexpr uint8_t SHEAR   = 1 << 1;
    constexpr uint8_t BENDING = 1 << 2;
    constexpr uint8_t ALL     = STRETCH | SHEAR | BENDING;
}

// Which factory builds the sim object: an open sheet (cloth()) or a closed conical skirt (skirt()).
// AppConfig carries both shapes' params simultaneously; see build_cloth() for the dispatch.
enum class ClothType { Square, Skirt };

// Declared here (body in diffpd.cpp) so diffpd_viewer.cpp can rebuild a live preview without
// including diffpd.cpp, keeping it the only raylib-aware file.
Object cloth(
    Index       width,
    Index       height,
    Real        stiffness    = DEFAULT_STIFFNESS,
    Vec3        origin       = Vec3::Zero(),
    PinMode     pin_mode     = PinMode::CORNERS,
    HangingMode hanging_mode = HangingMode::HORIZONTAL,
    uint8_t     flags        = ClothFlags::ALL,
    Real        m_tot        = 1.0);

// A conical-frustum ("skirt") cloth: `num_rings` rings of `particles_per_ring` vertices, stacked
// along -Y from a pinned `radius_top` ring to `radius_bottom` at `height` below (interpolated).
// The ring direction wraps into a closed loop; height stays open (no caps), unlike cloth().
Object skirt(
    Index   particles_per_ring,
    Index   num_rings,
    Real    stiffness     = DEFAULT_STIFFNESS,
    Vec3    origin        = Vec3::Zero(),
    Real    radius_top    = 0.15,
    Real    radius_bottom = 0.3,
    Real    height        = 0.7,
    uint8_t flags         = ClothFlags::ALL,
    Real    m_tot         = 0.5);

// ----------------
//   APP CONFIG
// ----------------

// Pre-fills every shape's params so switching collider shape in the config screen never lands on
// Collider's own generic defaults.
inline Collider default_config_collider()
{
    Collider c;
    c.type            = ColliderType::Sphere;
    c.sphere_center   = Vec3(1.0, -1.0, 1.0);
    c.sphere_radius   = 0.5;
    c.cylinder_origin = Vec3(5.0, -6.0, 5.0);
    c.cylinder_axis   = Vec3::UnitX();
    c.cylinder_radius = 1.0;
    c.plane_origin    = Vec3(0.0, -1.4, 0.0);
    c.plane_normal    = Vec3::UnitY();
    c.capsule_p0      = Vec3(5.0, -10.0, 0.0);
    c.capsule_p1      = Vec3(5.0, -5.0, 0.0);
    c.capsule_radius  = 1.0;
    c.velocity        = Vec3::Zero();
    return c;
}

// Every knob main() used to hardcode, now editable live in the pre-run config screen.
struct AppConfig
{
    // Square uses width/height/pin_mode/hang_mode below; Skirt uses particles_per_ring/num_rings/
    // radius_top/radius_bottom/skirt_height.
    ClothType cloth_type = ClothType::Skirt;

    // cloth (shared)
    Real stiffness        = 1.0;
    Real target_stiffness = 2.0;
    Vec3 origin           = Vec3(0.0, 0.0, 0.0); // shared by both target and guess cloth
    bool flag_shear       = true;  // ClothFlags::SHEAR   (STRETCH always on, not stored)
    bool flag_bending     = true;  // ClothFlags::BENDING
    Real m_tot            = 0.04;

    // cloth (Square only)
    int         width     = 15;
    int         height    = 15;
    PinMode     pin_mode  = PinMode::ROW;
    HangingMode hang_mode = HangingMode::HORIZONTAL;

    // cloth (Skirt only)
    int  particles_per_ring = 24;
    int  num_rings          = 14;
    Real radius_top         = 0.15;
    Real radius_bottom      = 0.3;
    Real skirt_height       = 0.7;

    // UI-managed list; starts empty, "+"/"-" grows/shrinks it (empty disables contact against
    // manually-placed colliders — animated colliders loaded from Blender are separate and unaffected).
    std::vector<Collider> colliders = {};

    // Global contact-model choice, regardless of which collider shape is active.
    ContactPointMode contact_point_mode = ContactPointMode::Surface;

    // How an animated collider's contact velocity is estimated (see AnimatedColliderVelocityMode).
    AnimatedColliderVelocityMode animated_collider_velocity_mode = AnimatedColliderVelocityMode::DecomposedRigid;

    // Minimum penetration depth (world units) for the post-solve "unresolved contacts" stat
    // (Tape::unresolved_contacts) to count a particle as still colliding; shallower residual
    // penetration is treated as resolved. Does not affect the forward solve's own contact detection.
    // Config-screen picker restricts this to one of 1e-1/1e-2/1e-3/1e-4 (see unresolved_threshold_field).
    Real unresolved_contact_threshold = 1e-2;

    // Rigidly attaches pinned cloth vertices to the hip collider instead of a fixed rest position.
    bool waist_attach_enabled = true;

    // physics
    Vec3 gravity = Vec3::UnitY() * -9.81;

    // simulation / solver
    int FPS             = 60;
    int frame_substeps  = 1;
    int secs            = 4;
    int n_iters         = 300; // forward PD global-local iterations per step
    int n_iters_adjoint = 300; // backward adjoint-vector iterations per step

    // output
    bool record_on_run = false; // start the screen recorder automatically when Run is clicked

    // gradient check (dphi/dk vs. central finite differences)
    bool run_fd_check = false;
    // one flag per order of magnitude in kFDEpsilonValues (1e-2 .. 1e-10)
    bool fd_eps_selected[9] = { false, false, false, false, false, false, true, true, true };
};

// Dispatches to cloth() or skirt() per cfg.cloth_type. `stiffness`/`origin` are passed explicitly
// (not read from cfg) since callers need arbitrary target/guess/FD-perturbed values.
inline Object build_cloth(const AppConfig& cfg, Real stiffness, Vec3 origin)
{
    const uint8_t flags = ClothFlags::STRETCH
                         | (cfg.flag_shear   ? ClothFlags::SHEAR   : 0)
                         | (cfg.flag_bending ? ClothFlags::BENDING : 0);
    switch (cfg.cloth_type)
    {
        case ClothType::Skirt:
            return skirt(cfg.particles_per_ring, cfg.num_rings, stiffness, origin,
                         cfg.radius_top, cfg.radius_bottom, cfg.skirt_height, flags, cfg.m_tot);
        default: // ClothType::Square
            return cloth(cfg.width, cfg.height, stiffness, origin, cfg.pin_mode, cfg.hang_mode,
                         flags, cfg.m_tot);
    }
}

// FD-check epsilon choices, indexed like AppConfig::fd_eps_selected (0 = 1e-2 ... 8 = 1e-10).
inline constexpr Real kFDEpsilonValues[9] = { 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 };

// Loss/gradient summary for one guess/target pair, for playback-screen display. dphi_dx0/dphi_dv0
// are collapsed from per-particle vectors to their column sums (net gradient per world axis).
struct GradientSummary
{
    Real loss     = 0.0;
    Vec3 dphi_dx0 = Vec3::Zero();
    Vec3 dphi_dv0 = Vec3::Zero();
    Real dphi_dk  = 0.0;
};

// Per-step convergence residuals for one guess/target run (forward + adjoint), for the playback
// screen's convergence graphs. All three are sized n_steps, forward chronological order.
struct ResidualHistory
{
    std::vector<Real> target_forward;
    std::vector<Real> guess_forward;
    std::vector<Real> backward_adjoint;
};

// Result of one central-difference check of dphi/dk against the analytic gradient; shared here so
// the viewer can display it without knowing about Loss/pd_contact/Object construction.
struct FDCheckResult
{
    Real fd;
    Real analytic;
    Real rel_err;
};

// Runs a stiffness FD check per entry of `epsilons`. diffpd.cpp builds the actual closure; the
// viewer just calls it with whichever epsilons are selected.
using FDCheckRunner = std::function<std::vector<FDCheckResult>(const std::vector<Real>& epsilons)>;

// ----------------
//    CONTACT
// ----------------

struct Contact
{
    ParticleId particle;
    int        collider_id = -1; // index into the colliders vector this contact was found against
    Vec3       normal; // unit outward contact normal
    Real       inv_r;  // curvature scale (1/dist to the collider's center/axis); 0 for a plane
    bool       active; // set in the backward pass: true if the contact is pressing (d_n < 0)
    Real       d_n;    // set in the backward pass: f_i . normal, cached for the d(normal)/dx term
    Vec3       axis = Vec3::Zero(); // unit collider axis; zero unless the collider is a Cylinder
    Vec3       surface_point = Vec3::Zero(); // closest surface point, for ContactPointMode::Surface
    Vec3       v_c = Vec3::Zero(); // collider velocity at contact, cached so backward pass matches forward
};

using Contacts = std::vector<Contact>;

// ----------------
//       TAPE
// ----------------

struct Tape
{
    std::vector<PointsX> positions;   // positions: one Nx3 matrix per timestep
    std::vector<PointsX> velocities;  // velocities: one Nx3 matrix per timestep
    std::vector<Contacts> contacts;   // contacts[t] = contacts detected at the start of step t (size == n_steps)
                                       // -> contacts[t].size() is "particles colliding" that frame.

    // Forward-solve convergence diagnostic: relative step size at the last iteration of each step.
    std::vector<Real> forward_residual;

    // How many particles are still penetrating a collider after step t's iterations converge
    // (re-detected against the converged x, not the pre-solve x_tilde in contacts[t]). Ideally 0 —
    // a nonzero value means the solve didn't fully resolve contact within n_iters that step.
    std::vector<int> unresolved_contacts;

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
