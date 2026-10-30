#pragma once

#include "core/diffpd_core.h"
#include "energy/elastic_energies.h"
#include "meshgen/tri_surface.h"
#include "meshgen/pattern_io.h"

#include <filesystem>

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
    std::vector<std::pair<Index,Index>> seams; // vertex pairs sewn together (Pattern; drawn by the viewer)
    std::vector<int>                    seam_levels; // per seam: the level it is closed in (1..4)

    // Grid shape: vertices[] laid out row-major as i*height+j. Set by cloth()/skirt() so
    // renderers can reconstruct quad adjacency without re-deriving it from edges[].
    Index width  = 0;
    Index height = 0;

    // True if the i axis wraps into a closed ring (set by skirt(), false for cloth()) — tells
    // grid renderers to close the seam between the last and first column.
    bool wrap_i = false;

    // Triangle list into `vertices` (empty for grid meshes; renderers use it when non-empty). Set by tri_cloth(),
    // where width = height = 0.
    std::vector<std::array<Index, 3>> triangles;

    Vec3 position(const Object& obj, Index vi) const; // defined after Object
};

// One pinned vertex riding a skinned body: up to 4 (bone, weight) pairs, like a skinned vertex.
// q[k] = S_bone[k](0)^-1 * p0 is the vertex's frame-0 world position p0 expressed through bone k's own
// frame-0 skinning matrix S, so the vertex sits at sum_k weight[k] * S_bone[k](t) * q[k] at step t
// (== S(t) S(0)^-1 p0 for a single bone with weight 1). Unused slots have weight 0.
struct PinBinding
{
    static constexpr int kMaxInfluences = 4;

    std::array<int, kMaxInfluences>  bone   = { 0, 0, 0, 0 };
    std::array<Real, kMaxInfluences> weight = { 1.0, 0.0, 0.0, 0.0 };
    std::array<Vec3, kMaxInfluences> q      = { Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), Vec3::Zero() };
};

// How the cloth's pinned vertices follow a moving body ("waist attachment"): either a rigid animated
// collider track (anim_id >= 0; each pin keeps a fixed offset in the track's frame-0 pose) or a skinned
// body (skinned_id >= 0; one PinBinding per pinned vertex). Neither set = pins stay at their rest position.
// See bake_waist_attachment* / update_waist_attachment*.
struct WaistAttachment
{
    int                     anim_id = -1;   // rigid: index into a std::vector<ColliderAnimation>
    std::vector<Vec3>       local_offset;   // rigid: per pinned vertex, offset in the track's frame-0 pose
    int                     skinned_id = -1; // skinned: index into skinned_meshes
    std::vector<PinBinding> binding;         // skinned: per pinned vertex

    bool active() const { return anim_id >= 0 || skinned_id >= 0; }
};

struct Object
{
    Positions  x;        // current positions       (3n)
    Velocities v;        // current velocities      (3n)
    Positions  prev_x;   // x at start of step

    MassDiag        mass;     // diagonal of M,          (3n)
    ElasticEnergies energies;

    SparseMat L;                       // L = M/h^2 + sum_i k_i G_i^T G_i   (SPD, constant)
    SparseMat C;                       // h^2 * K — elastic block; only set by construct_velocity_lhs
    // Laplacian damping (Rayleigh with the PD Laplacian K), D = alpha*M + beta*K, set by build_cloth().
    // alpha = beta = 0 is "off": C_d == C and the factor below is of L itself. L and C stay UNDAMPED
    // (they build b_tilde and its derivative); C_d = C + h*D is only for the free force f and the
    // adjoint's P-coupling, and `solver` factors L_d = M + C_d.
    Real      damp_alpha = 0.0;        // 1/s, mass-proportional
    Real      damp_beta  = 0.0;        // s,   Laplacian-proportional
    SparseMat C_d;                     // C + h*D; only set by construct_velocity_lhs
    std::unique_ptr<Cholesky> solver;  // factor of L_d (== L when undamped); heap-allocated so Object stays moveable

    SimMesh mesh;

    // Pinned vertices follow a moving body instead of a fixed rest position, if active().
    WaistAttachment waist;

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

// None, Mesh and Skinned come last so existing Sphere/Cylinder/Plane/Capsule indices (GuiToggleGroup's
// active index in the config screen) don't shift. Mesh and Skinned are animation-only, not
// UI-selectable. Skinned = a linear-blend-skinned triangle body, see SkinnedMesh.
enum class ColliderType { Sphere, Cylinder, Plane, Capsule, None, Mesh, Skinned };

// Locks rotation to a world axis, so it's a simple 2D spin (no axis-angle machinery needed).
enum class RotationAxis { X, Y, Z, None };

// How often pd_contact re-tests the current iterate for newly penetrating particles:
// Once = only the pre-solve detection against x_tilde; Every = before every iteration;
// NTimes = n checks in total (the pre-solve one included), evenly spaced over the iterations.
enum class ContactRecheckMode { Once, Every, NTimes };

struct RigidPose
{
    Vec3 position                   = Vec3::Zero();
    Eigen::Quaternion<Real> rotation = Eigen::Quaternion<Real>::Identity();
};

// Which part of the closest triangle a mesh query landed on (selects the backward formula and which
// pseudo-normal gives the inside/outside sign).
enum class ContactFeature { Face, Edge, Vertex };

// Axis-aligned box of one BVH node (not the collider-level `AABB` further down, which carries a `valid` flag).
struct BvhBox
{
    Vec3 lo = Vec3::Zero();
    Vec3 hi = Vec3::Zero();
};

// One node of a Bvh. Leaf: count > 0, owns Bvh::tri_ids[first, first + count). Inner: left/right children.
struct BvhNode
{
    BvhBox box;
    int    left  = -1;
    int    right = -1;
    int    first = 0;
    int    count = 0;
};

// Bounding-volume hierarchy over a triangle mesh (algorithms in bvh.h). `nodes` is in pre-order, so every
// child has a higher index than its parent and refitting is one reverse loop. Empty until bvh_build.
struct Bvh
{
    std::vector<BvhNode> nodes;   // nodes[0] is the root
    std::vector<int>     tri_ids; // triangle ids, grouped per leaf
};

// Per-triangle / per-edge / per-vertex normals of a closed mesh. Recomputed from the current vertices
// (compute_tri_normals) whenever the mesh deforms; the pseudo-normals give a robust inside/outside sign.
struct TriNormals
{
    std::vector<Vec3>                face_normal;          // unit, per triangle
    std::vector<Vec3>                vertex_pseudo_normal; // corner-angle-weighted sum of adjacent face normals
    std::vector<std::array<Vec3, 3>> edge_pseudo_normal;   // [t][e]: edge from triangle vertex e to (e+1)%3
};

// Rigid triangle mesh in its rest (local) frame, built once at load and never modified.
// Triangles are welded and outward-wound; pseudo-normals give a robust inside/outside sign.
struct TriMesh : TriNormals
{
    std::string                    name;
    std::vector<Vec3>              vertices;
    std::vector<std::array<int, 3>> triangles;
    std::vector<Real>              face_area;
    std::vector<std::array<int, 3>> edge_neighbor;        // [t][e]: triangle across edge e (vertex e -> (e+1)%3)
    Vec3                           aabb_min = Vec3::Zero();
    Vec3                           aabb_max = Vec3::Zero();
    Bvh                            bvh;                   // over `vertices`; built by build_trimesh
};

// x' = R x + t.
struct AffineTransform
{
    Mat3 R = Mat3::Identity();
    Vec3 t = Vec3::Zero();

    Vec3 apply(const Vec3& x) const { return R * x + t; }
};

// A closed triangle body deformed by linear blend skinning (LBS), imported from a
// "diffpd_skinned_body" .json (animation/export_skinned_body.py). Everything is in diffpd's Y-up
// frame. Vertex i deforms to x_i(f) = sum_k weight[i][k] * skin_at(f, bone_index[i][k]).apply(rest.vertices[i]),
// where skin_at(f, b) maps the rest pose straight to bone b's pose at track frame f (bind pose
// already folded in). Frames run at kColliderAnimFPS, like the rigid collider tracks.
struct SkinnedMesh
{
    static constexpr int kMaxInfluences = 4;

    std::string name;
    TriMesh     rest; // rest geometry + topology (vertex order == bone_index/weight order)

    std::vector<std::string> bone_names;
    std::vector<int>         bone_parent; // -1 = root

    std::vector<std::array<int, kMaxInfluences>>  bone_index; // per vertex; unused slots weight 0
    std::vector<std::array<Real, kMaxInfluences>> weight;     // per vertex, sums to 1

    int                          n_frames = 0;
    std::vector<AffineTransform> skin; // [frame * n_bones() + bone]

    // Physics steps per track frame (= frame_substeps, since the sim runs at kColliderAnimFPS). Set once
    // per run in main(); turns a step index into a track time, like ColliderAnimation::steps_per_frame.
    int steps_per_frame = 1;

    int n_bones() const { return (int)bone_names.size(); }
    const AffineTransform& skin_at(int frame, int bone) const { return skin[(size_t)frame * n_bones() + bone]; }
};

// The body's deformed vertices at track frame `frame` (clamped to the track): plain LBS of the rest
// vertices, no interpolation between frames.
inline void skin_vertices(const SkinnedMesh& body, int frame, std::vector<Vec3>& out)
{
    frame = std::clamp(frame, 0, body.n_frames - 1);
    out.assign(body.rest.vertices.size(), Vec3::Zero());
    for (size_t i = 0; i < out.size(); ++i)
        for (int k = 0; k < SkinnedMesh::kMaxInfluences; ++k)
        {
            const Real w = body.weight[i][k];
            if (w != 0.0) out[i] += w * body.skin_at(frame, body.bone_index[i][k]).apply(body.rest.vertices[i]);
        }
}

// Index of the bone called `name`, or -1.
inline int find_bone(const SkinnedMesh& body, const std::string& name)
{
    const auto it = std::find(body.bone_names.begin(), body.bone_names.end(), name);
    return it == body.bone_names.end() ? -1 : (int)(it - body.bone_names.begin());
}

// Bone `bone`'s skinning matrix at physics state `step`, with the same step -> track-time mapping, end
// hold and linear interpolation between the two bracketing frames as stamp_skinned_body: a point carried
// by this matrix moves exactly like a body vertex skinned 100 % to the bone (chord, constant velocity).
inline AffineTransform skin_matrix_at_step(const SkinnedMesh& body, int bone, int step)
{
    const int  last  = body.n_frames - 1;
    const int  frame = std::clamp(step / body.steps_per_frame, 0, last);
    const Real alpha = Real(step % body.steps_per_frame) / body.steps_per_frame;
    const AffineTransform& a = body.skin_at(frame, bone);
    if (alpha == 0.0) return a;
    const AffineTransform& b = body.skin_at(std::min(frame + 1, last), bone);
    AffineTransform out;
    out.R = a.R + alpha * (b.R - a.R);
    out.t = a.t + alpha * (b.t - a.t);
    return out;
}

// The body's pose at the current physics step, written by stamp_skinned_body. Shared (global, one per
// SkinnedMesh) so the forward pass, the backward pass and the viewer all read one state.
struct SkinnedState
{
    int               bracket = -1;   // track frame f that X0 was computed for (X1 is frame f + 1); -1 = never stamped
    int               step    = -1;   // physics state last stamped
    std::vector<Vec3> X0, X1;         // deformed vertices at the two bracketing track frames
    std::vector<Vec3> x;              // current vertices: X0 + alpha (X1 - X0)
    std::vector<Vec3> v;              // per-vertex velocity (m/s), constant over the bracket
    TriNormals        normals;        // face/edge/vertex normals of x, refreshed on every stamp
    Bvh               bvh;            // over x: built on the first stamp, refit on every later one
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

// ----------------
//  ANIMATED COLLIDERS
// ----------------
// A collider's base fields can be stamped every step from a per-frame track baked from Blender
// (an animation .json in diffpd/animation) instead of derived analytically from velocity/omega. See Collider::animated.

// The track's own fixed bake rate — a property of the imported data. The sim must run at this FPS
// (one sim frame == one track frame); frame_substeps then splits each track frame into
// ColliderAnimation::steps_per_frame physics steps. compute_animated_collider_velocity_basis is
// defined per bracket of this duration, not per physics step.
inline constexpr Real kColliderAnimFPS = 60.0;

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

    // Physics steps per track frame (= frame_substeps, since the sim runs at kColliderAnimFPS). Set once
    // per run in main(); turns a step index into a track time — see anim_sample_at_step.
    int steps_per_frame = 1;

    // velocity_basis[k] = compute_animated_collider_velocity_basis(*this, k, ...) — see
    // precompute_velocity_basis. Empty until that's called; indexed and clamped the same way as `frames`.
    std::vector<AnimatedColliderVelocityBasis> velocity_basis;
};

// ----------------
//  COLLISION WORLD
// ----------------
// Everything a step collides with, handed explicitly to pd_contact / backward_pd_contact / the viewer (it used
// to be file-scope globals). Indices: Collider::anim_id -> animations, ::mesh_id -> trimeshes, ::skinned_id ->
// skinned_meshes / skinned_states (parallel). It is mutated while a run advances (advance_to poses the colliders
// and skinned bodies in place), so one world serves one run at a time; prepare_world builds a fresh one.
struct CollisionWorld
{
    std::vector<Collider>          colliders;
    std::vector<ColliderAnimation> animations;
    std::vector<TriMesh>           trimeshes;      // rest-frame meshes of Mesh colliders
    std::vector<SkinnedMesh>       skinned_meshes;
    std::vector<SkinnedState>      skinned_states; // parallel to skinned_meshes

    // Contact model options: the AppConfig fields of the same names, copied by prepare_world.
    ContactRecheckMode contact_recheck_mode  = ContactRecheckMode::Once;
    int                contact_recheck_count = 10;
    bool               speculative_contact   = false;
    Real               contact_margin        = 0.0;

    // Poses skinned body `id` at physics state `step`, with the same step -> track-time mapping and end
    // clamping as the rigid tracks (anim_sample_at_step): state `step` sits step / steps_per_frame frames
    // in, and past the last frame the pose is held (v = 0). The deformed vertices are linearly interpolated
    // between the two bracketing frames, so every vertex moves at the constant velocity
    // v = (X1 - X0) * kColliderAnimFPS over a bracket. LBS only runs when the bracket changes; the normals
    // and the BVH (built on the first stamp, refit afterwards) are refreshed on every call so closest-point
    // queries always see the current pose. Defined in diffpd.cpp (it needs the BVH/normal code in bvh.h/trimesh.h).
    void stamp_skinned_body(int id, int step);

    // Restamps every animated collider and skinned body (+ the waist attachment, if enabled) to physics
    // state `step`. Shared by pd_contact (forward) and backward_pd_contact so the two can never drift apart
    // on how a step index is turned into a pose: both must see exactly the same collider/anchor state at
    // a given step for the adjoint to be correct.
    void advance_to(Object& obj, int step);
};

inline AABB collider_aabb(const CollisionWorld& world, const Collider& c, const ColliderPose& pose)
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
    else if (c.type == ColliderType::Skinned)
    {
        // The BVH root already bounds the deformed vertices at the stamped state (empty before any stamp).
        const Bvh& bvh = world.skinned_states[c.skinned_id].bvh;
        if (!bvh.nodes.empty())
        {
            box.min   = bvh.nodes[0].box.lo;
            box.max   = bvh.nodes[0].box.hi;
            box.valid = true;
        }
    }
    else if (c.type == ColliderType::Mesh)
    {
        // World box around the posed local AABB's 8 corners (conservative under rotation).
        const TriMesh& m = world.trimeshes[c.mesh_id];
        box.min = box.max = c.mesh_pose.position + c.mesh_pose.rotation * m.aabb_min;
        for (int k = 0; k < 8; ++k)
        {
            const Vec3 corner(k & 1 ? m.aabb_max.x() : m.aabb_min.x(),
                              k & 2 ? m.aabb_max.y() : m.aabb_min.y(),
                              k & 4 ? m.aabb_max.z() : m.aabb_min.z());
            const Vec3 w = c.mesh_pose.position + c.mesh_pose.rotation * corner;
            box.min = box.min.cwiseMin(w);
            box.max = box.max.cwiseMax(w);
        }
        box.valid = true;
    }
    return box; // Cylinder/Plane/None: valid = false, min/max unused
}

inline bool aabb_contains(const AABB& box, const Vec3& p)
{
    return (p.array() >= box.min.array()).all() && (p.array() <= box.max.array()).all();
}

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

// Where physics state `step` falls on a track: integer `frame` plus a fraction `alpha` in [0,1) of
// the way to frame + 1. With steps_per_frame == 1 this is always { step, 0 }.
struct AnimSample { int frame; Real alpha; };

inline AnimSample anim_sample_at_step(const ColliderAnimation& anim, int step)
{
    return { step / anim.steps_per_frame, Real(step % anim.steps_per_frame) / anim.steps_per_frame };
}

// The raw (Blender-space) track pose at physics state `step`: lerp position / slerp rotation between
// the two bracketing frames — exactly the constant v/omega rigid motion the velocity basis assumes over a
// bracket, so the stamped geometry and the contact velocity agree at every substep. alpha == 0 returns
// the stored frame untouched. Clamped like collider_animation_pose_at.
inline ColliderFrame collider_frame_at_step(const ColliderAnimation& anim, int step)
{
    const AnimSample     s    = anim_sample_at_step(anim, step);
    const int            last = (int)anim.frames.size() - 1;
    const ColliderFrame& f0   = anim.frames[std::clamp(s.frame, 0, last)];
    if (s.alpha == 0.0) return f0;
    const ColliderFrame& f1   = anim.frames[std::clamp(s.frame + 1, 0, last)];
    return { f0.position + s.alpha * (f1.position - f0.position), f0.rotation.slerp(s.alpha, f1.rotation) };
}

// The diffpd-space pose of a track at physics state `step` (interpolated, see collider_frame_at_step).
inline RigidPose collider_animation_pose_at_step(const ColliderAnimation& anim, int step)
{
    return to_diffpd_pose(collider_frame_at_step(anim, step));
}

// Stamps the track's pose at physics state `step` onto c's base fields.
inline void apply_collider_at_step(Collider& c, const ColliderAnimation& anim, int step)
{
    const ColliderFrame f = collider_frame_at_step(anim, step);

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

// ----------------
//  WAIST ATTACHMENT
// ----------------
// Rigidly attaches every pinned cloth vertex to one animated collider instead of a fixed rest
// position — e.g. a waistband following the hip through a walk cycle.

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
inline const AnimatedColliderVelocityBasis& animated_collider_basis_at_step(const ColliderAnimation& anim, int step)
{
    const int frame = std::clamp(anim_sample_at_step(anim, step).frame, 0, (int)anim.velocity_basis.size() - 1);
    return anim.velocity_basis[frame];
}

// Velocity of `world_point` on an animated collider at physics state `step` — the only thing that
// varies per contact. The bracket's basis is constant; only the collider's current pose moves inside
// it, to the interpolated pose at alpha (see collider_frame_at_step): v + omega x (x - pivot), with the
// pivot at the current (interpolated) position.
inline Vec3 animated_collider_point_velocity(const ColliderAnimation& anim, int step, const Vec3& world_point)
{
    const AnimatedColliderVelocityBasis& basis = animated_collider_basis_at_step(anim, step);
    const RigidPose                      pose  = collider_animation_pose_at_step(anim, step);
    return basis.linear_velocity + basis.omega_vec.cross(world_point - pose.position);
}

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
inline Vec3 pin_position_at_step(const SkinnedMesh& body, const PinBinding& pin, int step)
{
    Vec3 p = Vec3::Zero();
    for (int k = 0; k < PinBinding::kMaxInfluences; ++k)
        if (pin.weight[k] != 0.0) p += pin.weight[k] * skin_matrix_at_step(body, pin.bone[k], step).apply(pin.q[k]);
    return p;
}

// Re-poses just mesh.pinned_rest from the attached body's pose at physics state `step` — the part
// rendering needs; shared by update_waist_attachment and the playback viewer (which has no live Object).
inline void update_waist_attachment_mesh(SimMesh& mesh, const WaistAttachment& waist,
                                          const CollisionWorld& world, int step)
{
    if (waist.anim_id >= 0)
    {
        const RigidPose pose = collider_animation_pose_at_step(world.animations[waist.anim_id], step);
        for (size_t k = 0; k < mesh.pinned_rest.size(); ++k)
            mesh.pinned_rest[k] = pose.position + pose.rotation * waist.local_offset[k];
    }
    else if (waist.skinned_id >= 0)
    {
        const SkinnedMesh& body = world.skinned_meshes[waist.skinned_id];
        for (size_t k = 0; k < mesh.pinned_rest.size(); ++k)
            mesh.pinned_rest[k] = pin_position_at_step(body, waist.binding[k], step);
    }
}

// Per-step: re-poses mesh.pinned_rest (which the energies read their anchors from) from the attached
// body's pose at physics state `step`. No-op if waist attachment isn't enabled.
inline void update_waist_attachment(Object& obj, const CollisionWorld& world, int step)
{
    if (!obj.waist.active()) return;

    update_waist_attachment_mesh(obj.mesh, obj.waist, world, step);
}

inline void CollisionWorld::advance_to(Object& obj, int step)
{
    for (Collider& c : colliders)
    {
        if (c.animated && c.anim_id >= 0 && c.anim_id < (int)animations.size())
            apply_collider_at_step(c, animations[c.anim_id], step);
        else if (c.type == ColliderType::Skinned && c.skinned_id >= 0 && c.skinned_id < (int)skinned_meshes.size())
            stamp_skinned_body(c.skinned_id, step);
    }
    update_waist_attachment(obj, *this, step);
}

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

// Config-screen / run-start check: how many of `obj`'s particles (free ones, then pinned ones) lie inside
// a skinned body at its state 0. Contact has no push-out, so a cloth that starts inside a body stays stuck.
// Uses world's skinned bodies, stamping them at state 0.
void count_cloth_inside_skinned(const Object& obj, CollisionWorld& world, int& free_inside, int& pinned_inside);

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

// Which factory builds the sim object: an open sheet (cloth()), a closed conical skirt (skirt()) or the pieces of a pattern file
// (Pattern: always a Tri mesh, see tri_surface_for()). AppConfig carries all shapes' params simultaneously; see build_cloth().
enum class ClothType { Square, Skirt, Pattern };

// Pattern contour files (clothcreate format, see pattern_io.h) live in PATTERNS_DIR_DEFAULT (diffpd/patterns).
#ifndef PATTERNS_DIR_DEFAULT
#define PATTERNS_DIR_DEFAULT "patterns"
#endif
inline std::string pattern_path(const std::string& file) { return (std::filesystem::path(PATTERNS_DIR_DEFAULT) / file).string(); }
inline std::vector<std::string> list_pattern_files() // *.json names, sorted
{
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(PATTERNS_DIR_DEFAULT, ec))
        if (e.is_regular_file() && e.path().extension() == ".json") out.push_back(e.path().filename().string());
    std::sort(out.begin(), out.end());
    return out;
}
// Size + modification time of a pattern file: editing the file invalidates the generated mesh and the settled pose.
inline std::string pattern_stamp(const std::string& file)
{
    std::error_code ec;
    const auto size = std::filesystem::file_size(pattern_path(file), ec);
    const auto time = std::filesystem::last_write_time(pattern_path(file), ec).time_since_epoch().count();
    return ec ? std::string("missing") : std::to_string(size) + "@" + std::to_string(time);
}

struct AppConfig;
inline bool pattern_ok(const AppConfig& cfg); // defined after tri_surface_for

// How the mesh is made: the (i,j) quad grid (stretch + shear + bend springs) or a generated triangle mesh
// (stretch springs on the triangle edges only, or ARAP). Orthogonal to ClothType: both shapes have both.
enum class MeshGen { Quad, Tri };

// Elastic energy of a Tri mesh: one Stretch spring per edge, or one ARAP element per triangle (forward only).
enum class TriEnergy { Springs, ARAP };


// Declared here (body in diffpd.cpp) so diffpd_viewer.cpp can rebuild a live preview without
// including diffpd.cpp, keeping it the only raylib-aware file.
Object cloth(
    Index       width,
    Index       height,
    Real        size_x, // metres spanned by `width` particles (x) and `height` particles (z or -y)
    Real        size_y,
    Stiffness   stiffness    = Stiffness::uniform(DEFAULT_STIFFNESS),
    Vec3        origin       = Vec3::Zero(),
    PinMode     pin_mode     = PinMode::CORNERS,
    HangingMode hanging_mode = HangingMode::HORIZONTAL,
    uint8_t     flags        = ClothFlags::ALL,
    Real        m_tot        = 1.0);

// A conical-frustum ("skirt") cloth: `num_rings` rings of `particles_per_ring` vertices, stacked
// along -Y from a pinned `radius_top` ring to `radius_bottom` at `height` below (interpolated).
// The ring direction wraps into a closed loop; height stays open (no caps), unlike cloth().
Object skirt(
    Index     particles_per_ring,
    Index     num_rings,
    Stiffness stiffness     = Stiffness::uniform(DEFAULT_STIFFNESS),
    Vec3      origin        = Vec3::Zero(),
    Real      radius_top    = 0.15,
    Real      radius_bottom = 0.3,
    Real      height        = 0.7,
    uint8_t   flags         = ClothFlags::ALL,
    Real      m_tot         = 0.5);

// A cloth built on a generated triangle mesh (see tri_surface.h): one Stretch spring per triangle edge, area-weighted
// masses, the vertices on `pinned_curves` (chart curve ids, may be empty) pinned. `s` is relative to `origin`.
Object tri_cloth(
    const meshgen::TriSurface& s,
    const std::vector<int>& pinned_curves,
    Stiffness stiffness,
    Vec3      origin,
    Real      m_tot,
    TriEnergy energy = TriEnergy::Springs,
    bool      flat_bend_rest = true, // ARAP only: hinge rest curvature l0 = 0 (flat sheet) or from the generated mesh
    Real      seam_k = 0);           // stiffness of the s.seams links (rest length 0)

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

// What a run does. GuessTargetBackward = target sim + guess sim + loss + adjoint (+ optional FD check),
// played back as an orange-vs-white comparison. ForwardOnly = a single forward sim with the guess
// parameters, played back on its own; nothing about the backward pass is computed or shown.
enum class RunMode { GuessTargetBackward = 0, ForwardOnly = 1 };

// A relaxed rest pose for the cloth, computed by the config screen's "Settle cloth" button (settle_cloth in
// diffpd.cpp): the cloth simulated for a few seconds with every collider frozen at its frame-0 pose. Used as the
// initial positions x0 of the real runs instead of the procedurally generated mesh (see build_run_cloth).
// Valid only for the config it was computed from: `key` is settle_key() of that config, and the config screen
// drops the pose the moment the key no longer matches.
struct SettledStart
{
    bool        valid = false;
    std::string key;
    RealVecX    x;                 // free-particle positions
    std::vector<Vec3> pins;        // pinned-vertex positions after the waist snap (empty = procedural ones)
    Real        max_speed = 0.0;   // fastest particle [m/s] when the settle stopped;
    bool        seams_only = false; // the "Settle seams" stage only: seams closed, no gravity; "Settle cloth" continues from it
    Real        seam_gap  = -1.0;  // longest sewn link when it stopped [m] (-1 = no seams)
    int         steps     = 0;     // steps it ran. The settle stops early only below the rest speed, so max_speed
                                   // above it means the time cap was hit while still moving.
};

// Every knob main() used to hardcode, now editable live in the pre-run config screen.
struct AppConfig
{
    RunMode run_mode = RunMode::GuessTargetBackward;
    bool    backward_enabled() const { return run_mode == RunMode::GuessTargetBackward; }

    // Square uses width/height/pin_mode/hang_mode below; Skirt uses particles_per_ring/num_rings/
    // radius_top/radius_bottom/skirt_height.
    ClothType cloth_type = ClothType::Skirt;

    // cloth (shared)
    Vec3 origin           = Vec3(0.0, 0.95, 0.0); // shared by both target and guess cloth
    Real m_tot            = 0.04;

    // stiffness: constraint types and mode are shared by target and guess; only the values differ.
    // Unified = one k for every spring (single dphi/dk); per-type = one k per SpringKind.
    bool      flag_shear                = true;  // ClothFlags::SHEAR   (STRETCH always on, not stored)
    bool      flag_bending              = true;  // ClothFlags::BENDING
    bool      per_type_stiffness        = true;
    Real      stiffness                 = 1.0;   // unified
    Real      target_stiffness          = 2.0;   // unified
    Stiffness stiffness_per_type        = { 2.0, 0.5, 0.3 };  // stretch, shear, bend
    Stiffness target_stiffness_per_type = { 4.0, 1.0, 0.6 };

    Stiffness guess_k()  const { return use_arap() ? Stiffness{ arap_stiffness, 0.0, arap_bend_stiffness }        : per_type_stiffness ? stiffness_per_type        : Stiffness::uniform(stiffness); }
    Stiffness target_k() const { return use_arap() ? Stiffness{ target_arap_stiffness, 0.0, target_arap_bend_stiffness } : per_type_stiffness ? target_stiffness_per_type : Stiffness::uniform(target_stiffness); }
    // Whether k is one value per slot (report dphi/dk_stretch, FD-check the Stretch slot alone) or one shared k (report the
    // sum, FD-perturb every slot together). ARAP always has its own ARAP k and Bend k, whatever per_type_stiffness says.
    bool per_type_k() const { return per_type_stiffness || use_arap(); }
    // The stiffness the cloth is settled with, for target and guess alike (the real start pose is data, not a function of
    // the guess): the target's, or the only one shown in forward-only mode.
    Stiffness settle_k() const { return backward_enabled() ? target_k() : guess_k(); }

    // cloth (Square only)
    int         width     = 15;
    int         height    = 15;
    Real        size_x    = 1.0; // metres along width / height
    Real        size_y    = 1.0;
    PinMode     pin_mode  = PinMode::ROW;
    HangingMode hang_mode = HangingMode::HORIZONTAL;

    // cloth (Skirt only)
    int  particles_per_ring = 40;
    int  num_rings          = 20;
    Real radius_top         = 0.20;
    Real radius_bottom      = 0.3;
    Real skirt_height       = 0.5;

    // cloth (Pattern only): every shape of the file, laid out as in the file (side by side, flat, vertical XY plane), one object of
    // disconnected pieces; no pinned vertices, no sewing. Scale = metres per image pixel.
    std::string pattern_file  = "tshirt_pattern.json";
    Real        pattern_scale = 0.001;
    Real        seam_stiffness = 200.0; // k of the rest-length-0 springs sewing the pattern's seams (not optimised, see energy_seam.h)

    // Mesh generation (MeshGen::Tri uses tri_*; see meshgen::Params). Values are clamped when used, see tri_surface_for().
    MeshGen mesh_gen         = MeshGen::Tri;
    Real    tri_h            = 0.03;
    Real    tri_beta         = 0.80;
    int     tri_smooth_iters = 30;
    int     tri_seed         = 0;
    TriEnergy tri_energy     = TriEnergy::Springs;
    Real    arap_stiffness        = 4.0; // ARAP surface modulus (rides in the Stretch slot); not comparable to a spring k
    Real    target_arap_stiffness = 8.0;
    Real    arap_bend_stiffness        = 0.01; // hinge bending between neighbouring triangles (Bend slot), ARAP only
    Real    target_arap_bend_stiffness = 0.02;
    bool    arap_flat_rest = true;  // hinge rest l0 = 0 everywhere (flat sheet) instead of the generated mesh's curvature

    bool use_tri() const { return mesh_gen == MeshGen::Tri || cloth_type == ClothType::Pattern; }
    bool use_arap() const { return use_tri() && tri_energy == TriEnergy::ARAP; }
    Real tri_edge() const { return std::max(tri_h, 0.02); } // edge length h of the generated mesh, floored
    // The cone chart rejects cylinders (|dr| < 1e-3), degenerate sizes and an h that leaves fewer than 6 samples on a ring;
    // the rectangle chart needs at least 3 segments per edge.
    bool tri_ok() const
    {
        if (!use_tri()) return true;
        if (cloth_type == ClothType::Pattern) return pattern_ok(*this);
        if (cloth_type == ClothType::Square) return std::min(size_x, size_y) >= 2.5 * tri_edge();
        const Real r_min = std::min(radius_top, radius_bottom);
        return std::abs(radius_top - radius_bottom) >= 1e-3 && r_min >= 0.02 && skirt_height >= 0.02
               && tri_h <= 2.0 * 3.14159265358979 * r_min / 6.0;
    }
    // Shear/bend springs exist on the quad grid only.
    bool shear_on()   const { return flag_shear   && !use_tri(); }
    bool bending_on() const { return flag_bending && !use_tri(); }

    // UI-managed list; starts empty, "+"/"-" grows/shrinks it (empty disables contact against
    // manually-placed colliders — animated colliders loaded from Blender are separate and unaffected).
    std::vector<Collider> colliders = {};

    // Re-test the current iterate inside pd_contact's solver loop and add newly penetrating
    // particles to the step's contact set (see merge_detected_contacts in diffpd.cpp). Once = the
    // set is fixed by the single pre-solve detection against x_tilde; NTimes uses contact_recheck_count (>= 2).
    // Ignored (always Once) with speculative_contact.
    ContactRecheckMode contact_recheck_mode  = ContactRecheckMode::Once;
    int                contact_recheck_count = 5;

    // Experiment: gap-aware ("speculative") contact, always with a fixed contact set (Once). The constraint becomes "x+ not inside the collider's end-of-step
    // pose" (phi(x-) + h v+.n >= 0) instead of "(v+ - v_c).n >= 0", so a particle detected short of the surface lands
    // on it instead of hovering, and one already inside is pushed out. See pd_contact.
    bool               speculative_contact   = false;
    // Speculative only: pre-solve detection also takes particles within this distance outside the surface (at x_tilde).
    // Safe there (a contact that would not reach the surface is inactive) and lets the frozen set catch what the solve drags in.
    Real               contact_margin        = 0.0;

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

    // physics
    Vec3 gravity = Vec3::UnitY() * -9.81;

    // Laplacian damping D = alpha*M + beta*K (see Object::damp_alpha). When disabled, alpha/beta are
    // passed as 0 so the damped matrices equal the undamped ones.
    bool damping_enabled = false;
    Real damping_alpha   = 0.0;    // 1/s
    Real damping_beta    = 0.001;  // s

    // simulation / solver
    int FPS             = 60;
    int frame_substeps  = 1;
    int secs            = 4;
    int n_iters         = 300; // forward PD global-local iterations per step
    int n_iters_adjoint = 300; // backward adjoint-vector iterations per step
    int anderson_m      = 0;   // Anderson acceleration history of the forward and adjoint iterations (anderson.h); 0 = off

    // output
    bool record_on_run = false; // start the screen recorder automatically when Run is clicked

    // "Settle cloth" settings (see settle_cloth): the run stops at settle_seconds, or earlier once no particle moves
    // faster than settle_rest_speed. Fixed dt = 1/60 s. Part of settle_key (they change the resulting pose).
    Real settle_seconds    = 10.0;  // time cap [s]
    Real settle_rest_speed = 0.0;   // stop threshold on the fastest particle [m/s]
    int  settle_iters      = 20;    // PD iterations per step
    Real seam_settle_seconds = 2.0; // "Settle seams" (Pattern with seams): the links close over this time, no gravity [s]
    int  seam_settle_iters   = 30;  // PD iterations per step of the seam stage

    // Settled rest pose (see SettledStart). Its validity is tied to settle_key(): any field the settle depends
    // on must be listed there, or editing it would leave a stale pose active.
    SettledStart settled;

    // gradient check (dphi/dk vs. central finite differences)
    bool run_fd_check = false;
    // one flag per order of magnitude in kFDEpsilonValues (1e-2 .. 1e-10)
    bool fd_eps_selected[9] = { false, false, false, false, false, false, true, true, true };
};

// The generated mesh for cfg's skirt, cached per thread on its parameters (generation is far costlier than a grid, and the
// target, guess, FD check, settle and the per-frame config preview all rebuild the cloth). Out-of-range values are clamped.
struct TriCache { std::string key; meshgen::TriSurface surface; meshgen::Report report; std::string error; }; // error: Pattern only, empty = ok

inline const TriCache& tri_surface_for(const AppConfig& cfg)
{
    thread_local TriCache cache;
    meshgen::Params p;
    p.h = cfg.tri_edge();
    p.beta = std::clamp(cfg.tri_beta, 0.5, 1.0);
    p.smooth_iters = std::clamp(cfg.tri_smooth_iters, 0, 30);
    p.seed = (uint64_t)(uint32_t)cfg.tri_seed;
    std::ostringstream k;
    k << std::setprecision(17) << (int)cfg.cloth_type << ',' << (int)cfg.hang_mode << ',' << cfg.size_x << ',' << cfg.size_y << ';'
      << cfg.radius_top << ',' << cfg.radius_bottom << ',' << cfg.skirt_height << ';'
      << (cfg.cloth_type == ClothType::Pattern ? cfg.pattern_file + pattern_stamp(cfg.pattern_file) : std::string()) << ',' << cfg.pattern_scale << ';'
      << p.h << ',' << p.beta << ',' << p.smooth_iters << ',' << p.seed;
    if (cache.key != k.str() && cfg.cloth_type == ClothType::Pattern)
    {
        // every shape of the file, around one shared origin so the pieces keep their layout; a bad file / scale / h is an error
        // message (and tri_ok() == false), remembered under the same key so it is not retried every frame
        cache.key = k.str();
        cache.error.clear();
        try
        {
            std::vector<meshgen::PatternSeam> seams;
            const std::vector<meshgen::PatternShape> shapes = load_pattern(pattern_path(cfg.pattern_file), &seams);
            const Vec3 centre = meshgen::pattern_center(shapes);
            std::vector<meshgen::TriSurface> pieces;
            std::vector<meshgen::Report> reports;
            std::vector<meshgen::PatternChart> charts;
            for (const meshgen::PatternShape& shape : shapes) charts.emplace_back(shape, cfg.pattern_scale, p.h / 50, centre);
            meshgen::plan_seam_intervals(charts, seams, p.h); // both sides of a seam get the same number of vertices
            for (const meshgen::PatternChart& chart : charts)
            {
                pieces.push_back(meshgen::generate(chart, p));
                reports.push_back(meshgen::validate(pieces.back(), chart, p));
            }
            cache.surface = meshgen::merge(pieces);
            cache.surface.seams = meshgen::sew(pieces, seams, &cache.surface.seam_level);
            cache.report  = meshgen::combine(reports);
        }
        catch (const std::runtime_error& e)
        {
            cache.surface = meshgen::TriSurface{};
            cache.report  = meshgen::Report{};
            cache.error   = e.what();
        }
    }
    else if (cache.key != k.str())
    {
        std::unique_ptr<meshgen::Chart> chart;
        if (cfg.cloth_type == ClothType::Skirt)
            chart = std::make_unique<meshgen::ConeFrustum>(cfg.radius_top, cfg.radius_bottom, cfg.skirt_height);
        else
            chart = std::make_unique<meshgen::Rectangle>(cfg.size_x, cfg.size_y, cfg.hang_mode == HangingMode::VERTICAL);
        cache.surface = meshgen::generate(*chart, p);
        cache.report  = meshgen::validate(cache.surface, *chart, p);
        cache.key     = k.str();
    }
    return cache;
}

inline bool pattern_ok(const AppConfig& cfg) { return tri_surface_for(cfg).error.empty(); }

// Chart curves of tri_surface_for(cfg) that are pinned: the skirt's top ring, the square's row / two corners / nothing, nothing for a pattern.
inline std::vector<int> tri_pinned_curves(const AppConfig& cfg)
{
    using R = meshgen::Rectangle;
    if (cfg.cloth_type == ClothType::Pattern) return {};
    if (cfg.cloth_type == ClothType::Skirt) return { 0 };
    if (cfg.pin_mode == PinMode::ROW)       return { R::Top, R::CornerTL, R::CornerTR };
    if (cfg.pin_mode == PinMode::CORNERS)   return { R::CornerTL, R::CornerTR };
    return {};
}

// Dispatches to cloth() or skirt() per cfg.cloth_type. `stiffness`/`origin` are passed explicitly
// (not read from cfg) since callers need arbitrary target/guess/FD-perturbed values.
inline Object build_cloth(const AppConfig& cfg, const Stiffness& stiffness, Vec3 origin)
{
    const uint8_t flags = ClothFlags::STRETCH
                         | (cfg.shear_on()   ? ClothFlags::SHEAR   : 0)
                         | (cfg.bending_on() ? ClothFlags::BENDING : 0);
    Object obj = cfg.use_tri()
        ? tri_cloth(tri_surface_for(cfg).surface, tri_pinned_curves(cfg), stiffness, origin, cfg.m_tot, cfg.tri_energy, cfg.arap_flat_rest, cfg.seam_stiffness)
        : cfg.cloth_type == ClothType::Skirt
        ? skirt(cfg.particles_per_ring, cfg.num_rings, stiffness, origin,
                cfg.radius_top, cfg.radius_bottom, cfg.skirt_height, flags, cfg.m_tot)
        : cloth(cfg.width, cfg.height, cfg.size_x, cfg.size_y, stiffness, origin, cfg.pin_mode, cfg.hang_mode,
                flags, cfg.m_tot);
    obj.damp_alpha = cfg.damping_enabled ? cfg.damping_alpha : 0.0;
    obj.damp_beta  = cfg.damping_enabled ? cfg.damping_beta  : 0.0;
    return obj;
}

// Everything a settled pose (SettledStart) depends on, serialized: cloth shape/mass/origin, stiffness, pins,
// colliders, animation + waist attachment, gravity, damping, contact model. Deliberately NOT in it: FPS, substeps,
// seconds, iteration counts, run mode, FD check and recording options (the settle has its own settings, which are in it).
inline std::string settle_key(const AppConfig& c)
{
    std::ostringstream s;
    s << std::setprecision(17);
    auto vec = [&](const Vec3& v) { s << v.x() << ',' << v.y() << ',' << v.z() << ';'; };
    s << (int)c.cloth_type << ';';
    if (c.cloth_type == ClothType::Pattern) s << c.pattern_file << pattern_stamp(c.pattern_file) << ',' << c.pattern_scale << ',' << c.seam_stiffness << ';';
    vec(c.origin);
    s << c.m_tot << ';' << c.shear_on() << c.bending_on() << c.per_type_stiffness << ';';
    const Stiffness k = c.settle_k();
    s << k.stretch << ',' << k.shear << ',' << k.bend << ';';
    s << c.width << ',' << c.height << ',' << c.size_x << ',' << c.size_y << ',' << (int)c.pin_mode << ',' << (int)c.hang_mode << ';';
    s << c.particles_per_ring << ',' << c.num_rings << ',' << c.radius_top << ',' << c.radius_bottom << ',' << c.skirt_height << ';';
    s << (int)c.mesh_gen << ',' << c.tri_h << ',' << c.tri_beta << ',' << c.tri_smooth_iters << ',' << c.tri_seed << ',' << (int)c.tri_energy << ',' << c.arap_stiffness << ',' << c.target_arap_stiffness << ',' << c.arap_bend_stiffness << ',' << c.target_arap_bend_stiffness << ',' << c.arap_flat_rest << ';';
    for (const Collider& k : c.colliders)
    {
        s << (int)k.type << ':' << k.sphere_radius << ',' << k.cylinder_radius << ',' << k.capsule_radius << ',' << k.omega << ','
          << (int)k.rotation_axis << ',' << k.animated << ':';
        vec(k.sphere_center); vec(k.cylinder_origin); vec(k.cylinder_axis); vec(k.plane_origin); vec(k.plane_normal);
        vec(k.capsule_p0); vec(k.capsule_p1); vec(k.velocity); vec(k.rotation_origin);
    }
    s << '|' << c.animation_file << '|' << c.waist_attach_enabled << ';';
    vec(c.gravity);
    s << c.damping_enabled << ',' << c.damping_alpha << ',' << c.damping_beta << ';';
    s << (int)c.contact_recheck_mode << ',' << c.contact_recheck_count << ',' << c.speculative_contact << ',' << c.contact_margin << ';';
    s << c.settle_seconds << ',' << c.settle_rest_speed << ',' << c.settle_iters << ',' << c.seam_settle_seconds << ',' << c.seam_settle_iters;
    return s.str();
}

// Moves the pinned vertices to `pins` (indexed like mesh.pinned_rest).
inline void set_pinned_positions(Object& obj, const std::vector<Vec3>& pins)
{
    obj.mesh.pinned_rest = pins;
}

// The cloth a real run starts from: the procedural mesh, with its free particles moved to the settled pose when one
// is active. Target and guess start from the same pose. Used by every run (target, guess, FD check) and the
// config-screen preview, so they always agree.
inline Object build_run_cloth(const AppConfig& cfg, const Stiffness& stiffness)
{
    Object obj = build_cloth(cfg, stiffness, cfg.origin);
    if (cfg.settled.valid)
    {
        ASSERT(cfg.settled.x.size() == obj.x.size(), "settled pose has " << cfg.settled.x.size() << " values, the cloth " << obj.x.size());
        obj.x = cfg.settled.x;
        if (!cfg.settled.pins.empty()) set_pinned_positions(obj, cfg.settled.pins);
    }
    return obj;
}

// A step counts as converged when its relative residual (change between the last two iterations) is below this.
inline constexpr Real kConvergenceTol = 1e-4;

// FD-check epsilon choices, indexed like AppConfig::fd_eps_selected (0 = 1e-2 ... 8 = 1e-10).
inline constexpr Real kFDEpsilonValues[9] = { 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 };

// Loss/gradient summary for one guess/target pair, for playback-screen display. dphi_dx0/dphi_dv0
// are collapsed from per-particle vectors to their column sums (net gradient per world axis).
struct GradientSummary
{
    Real loss     = 0.0;
    Vec3 dphi_dx0 = Vec3::Zero();
    Vec3 dphi_dv0 = Vec3::Zero();
    // {"dphi/dk", g} when stiffness is unified, else one {"dphi/dk_<kind>", g} per enabled spring
    // kind, stretch first. front() is the gradient the FD check verifies.
    std::vector<std::pair<std::string, Real>> dphi_dk;
};

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

// Per-step convergence residuals for one guess/target run (forward + adjoint), for the playback
// screen's convergence graphs. All three are sized n_steps, forward chronological order.
struct ResidualHistory
{
    std::vector<StepResidual> target_forward;
    std::vector<StepResidual> guess_forward;
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

// FD results of the current experiment, indexed like kFDEpsilonValues; an empty slot = not checked yet.
using FDCheckCache = std::array<std::optional<FDCheckResult>, 9>;

// Runs a stiffness FD check for every selected epsilon (indexed like kFDEpsilonValues) that isn't
// already in the experiment's FDCheckCache, storing results there. diffpd.cpp builds the actual
// closure (and owns the cache); the viewer just calls it with its checkbox state.
using FDCheckRunner = std::function<void(const bool (&selected)[9])>;

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
    Vec3       surface_point = Vec3::Zero(); // closest surface point, where v_c is evaluated
    Vec3       v_c = Vec3::Zero(); // collider velocity at contact, cached so backward pass matches forward
    // Speculative contact only (else 0): signed distance phi(x-) from the step-start position to the collider's
    // end-of-step pose, and its gradient d phi / d x- (backward pass). In that mode normal/inv_r/axis/surface_point
    // above are evaluated at x- too (not at x~).
    Real       gap      = 0.0;
    Vec3       gap_grad = Vec3::Zero();

    // Mesh / Skinned colliders only. feature_index is the edge/vertex slot (0..2) of triangle_id; unused for Face.
    int            triangle_id   = -1;
    ContactFeature feature       = ContactFeature::Face;
    int            feature_index = 0;
    Vec3           edge_dir      = Vec3::Zero(); // Edge contacts: unit world-space edge direction
    Vec3           bary          = Vec3::Zero(); // Skinned only: barycentric coords of surface_point in triangle_id
    Mat3           dvc_dx        = Mat3::Zero(); // Skinned only: d v_c / d x_tilde (3x3), cached like v_c for the backward pass
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

    // Forward-solve convergence diagnostic: one StepResidual per step (from the last iteration).
    std::vector<StepResidual> forward_residual;

    // How many particles are still penetrating a collider after step t's iterations converge
    // (re-detected against the converged x, not the pre-solve x_tilde in contacts[t]). Ideally 0 —
    // a nonzero value means the solve didn't fully resolve contact within n_iters that step.
    std::vector<int> unresolved_contacts;

    // Global physics step the tape's state 0 sits at (0 = a run from the start). Animated colliders, skinned bodies
    // and the waist are indexed by global step, so a run resumed mid-trajectory (a chunk) must pose them from here.
    int first_step = 0;

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
