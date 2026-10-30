#pragma once

// The collision module: colliders, animation tracks, meshes and skinned bodies (CollisionWorld), the contacts
// found against them (Contact) and the detection itself. Depends only on core/: detection takes stacked positions
// (3n), never a cloth Object, so any integrator can call it. The cloth-side glue (waist pins, advance_world)
// is in diffpd_types.h.

#include "collision/collider.h"
#include "collision/animation.h"
#include "collision/skinned_mesh.h"

// How often pd_contact re-tests the current iterate for newly penetrating particles:
// Once = only the pre-solve detection against x_tilde; Every = before every iteration;
// NTimes = n checks in total (the pre-solve one included), evenly spaced over the iterations.
enum class ContactRecheckMode { Once, Every, NTimes };

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
    // clamping as the rigid tracks (anim_sample_at_step): state `step` sits step / substeps frames
    // in, and past the last frame the pose is held (v = 0). The deformed vertices are linearly interpolated
    // between the two bracketing frames, so every vertex moves at the constant velocity
    // v = (X1 - X0) * kColliderAnimFPS over a bracket. LBS only runs when the bracket changes; the normals
    // and the BVH (built on the first stamp, refit afterwards) are refreshed on every call so closest-point
    // queries always see the current pose.
    void stamp_skinned_body(int id, int substeps, int step);

    // Restamps every animated collider and skinned body to physics state `step`. Both passes go through
    // advance_world (diffpd_types.h), which also moves the cloth's waist pins.
    void advance_to(int substeps, int step)
    {
        for (Collider& c : colliders)
        {
            if (c.animated && c.anim_id >= 0 && c.anim_id < (int)animations.size())
                apply_collider_at_step(c, animations[c.anim_id], substeps, step);
            else if (c.type == ColliderType::Skinned && c.skinned_id >= 0 && c.skinned_id < (int)skinned_meshes.size())
                stamp_skinned_body(c.skinned_id, substeps, step);
        }
    }
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

// Declared in CollisionWorld (see the comment there for the step -> pose mapping).
inline void CollisionWorld::stamp_skinned_body(int id, int substeps, int step)
{
    const SkinnedMesh& body = skinned_meshes[id];
    SkinnedState&      st   = skinned_states[id];

    const int  last  = body.n_frames - 1;
    const int  frame = std::clamp(step / substeps, 0, last);
    const Real alpha = Real(step % substeps) / substeps;
    const size_t n   = body.rest.vertices.size();

    if (st.bracket != frame)
    {
        skin_vertices(body, frame, st.X0);
        skin_vertices(body, std::min(frame + 1, last), st.X1);
        st.v.resize(n);
        for (size_t i = 0; i < n; ++i) st.v[i] = (st.X1[i] - st.X0[i]) * kColliderAnimFPS;
        st.bracket = frame;
    }

    st.x.resize(n);
    for (size_t i = 0; i < n; ++i) st.x[i] = st.X0[i] + alpha * (st.X1[i] - st.X0[i]);

    // Query geometry for the new pose: normals from the winding of x, BVH boxes refit around it. The
    // tree is built once (on the first stamped pose; the loader stamps state 0) and only refit after.
    compute_tri_normals(st.x, body.rest.triangles, body.rest.edge_neighbor, st.normals);
    if (st.bvh.nodes.empty()) bvh_build(st.bvh, st.x, body.rest.triangles);
    else                      bvh_refit(st.bvh, st.x, body.rest.triangles);
    st.step = step;
}

// Velocity of the point with barycentric coordinates `bary` in triangle `triangle` of skinned body `id`:
// the blend of its three vertex velocities at the stamped state (each vertex is constant over a bracket).
inline Vec3 skinned_surface_velocity(const CollisionWorld& world, int id, int triangle, const Vec3& bary)
{
    const SkinnedState&      st  = world.skinned_states[id];
    const std::array<int, 3>& tri = world.skinned_meshes[id].rest.triangles[triangle];
    return bary[0] * st.v[tri[0]] + bary[1] * st.v[tri[1]] + bary[2] * st.v[tri[2]];
}

// G = d v_c / d x_tilde for a contact on a skinned-body triangle (only the barycentrics depend on x_tilde):
//   Face: sum_k v_k g_k^T, g_k = n x (x_{k+2} - x_{k+1}) / (2A)   Edge: (v_b - v_a) e^T / L   Vertex: 0
inline Mat3 skinned_velocity_gradient(const CollisionWorld& world, int id, int triangle, ContactFeature feature, int index)
{
    const SkinnedState&       st  = world.skinned_states[id];
    const std::array<int, 3>& tri = world.skinned_meshes[id].rest.triangles[triangle];
    switch (feature)
    {
        case ContactFeature::Vertex:
            return Mat3::Zero();
        case ContactFeature::Edge:
        {
            const int  a = tri[index], b = tri[(index + 1) % 3];
            const Vec3 e = st.x[b] - st.x[a];
            const Real L = e.norm();
            if (L < 1e-14) return Mat3::Zero();
            return (st.v[b] - st.v[a]) * (e / L).transpose() / L;
        }
        case ContactFeature::Face:
        {
            const Vec3 cr     = (st.x[tri[1]] - st.x[tri[0]]).cross(st.x[tri[2]] - st.x[tri[0]]);
            const Real two_a  = cr.norm();
            if (two_a < 1e-14) return Mat3::Zero();
            const Vec3 n = cr / two_a;
            Mat3 G = Mat3::Zero();
            for (int k = 0; k < 3; ++k)
            {
                const Vec3 g = n.cross(st.x[tri[(k + 2) % 3]] - st.x[tri[(k + 1) % 3]]) / two_a;
                G += st.v[tri[k]] * g.transpose();
            }
            return G;
        }
    }
    return Mat3::Zero();
}

// Fills `out` (normal, inv_r, axis, surface_point) for particle `i` at `pos` against one collider
// pose, and returns the signed distance to the surface (< 0 = inside). Split from detect_contacts so
// merge_detected_contacts can test penetration at one position but evaluate geometry at another.
inline Real contact_geometry(const CollisionWorld& world, const Collider& collider, int ci, const ColliderPose& pose, ParticleId i, const Vec3& pos, Contact& out)
{
    Real dist = 0.0;
    if (collider.type == ColliderType::Sphere)
    {
        const Vec3 offset           = pos - pose.sphere_center;
        const Real dist_from_center = offset.norm();
        dist                        = dist_from_center - collider.sphere_radius;
        const Vec3 normal           = offset / dist_from_center;
        out = Contact{i, ci, normal, 1.0 / dist_from_center, false, 0.0};
        out.surface_point = pose.sphere_center + collider.sphere_radius * normal;
    }
    else if (collider.type == ColliderType::Cylinder)
    {
        const Vec3 axis   = pose.cylinder_axis.normalized();
        const Vec3 rel    = pos - pose.cylinder_origin;
        const Vec3 perp   = rel - rel.dot(axis) * axis;
        const Real rho    = perp.norm();
        dist              = rho - collider.cylinder_radius;
        const Vec3 normal = perp / rho;
        out = Contact{i, ci, normal, 1.0 / rho, false, 0.0};
        out.axis          = axis;
        out.surface_point = (pos - perp) + collider.cylinder_radius * normal; // pos - perp = closest point on axis
    }
    else if (collider.type == ColliderType::Plane)
    {
        const Vec3 normal = pose.plane_normal.normalized();
        dist              = (pos - pose.plane_origin).dot(normal);
        out = Contact{i, ci, normal, 0.0, false, 0.0};
        out.surface_point = pos - dist * normal;
    }
    else if (collider.type == ColliderType::Mesh)
    {
        // Query in the mesh's rest frame (only the point moves), then bring the results back to world.
        const RigidPose& mp  = collider.mesh_pose;
        const TriMeshHit hit = trimesh_closest(world.trimeshes[collider.mesh_id], mp.rotation.conjugate() * (pos - mp.position));
        dist              = hit.dist;
        out = Contact{i, ci, mp.rotation * hit.normal, 0.0, false, 0.0}; // flat: no curvature
        out.surface_point = mp.position + mp.rotation * hit.point;
        out.triangle_id   = hit.triangle;
        out.feature       = hit.feature;
        out.feature_index = hit.index;
        if (hit.feature == ContactFeature::Edge)
        {
            const TriMesh& mesh = world.trimeshes[collider.mesh_id];
            const auto&    tri  = mesh.triangles[hit.triangle];
            out.edge_dir = mp.rotation * (mesh.vertices[tri[(hit.index + 1) % 3]] - mesh.vertices[tri[hit.index]]).normalized();
        }
    }
    else if (collider.type == ColliderType::Skinned)
    {
        // The body is already in world coordinates: query the deformed mesh at the stamped state.
        const SkinnedMesh&  body = world.skinned_meshes[collider.skinned_id];
        const SkinnedState& st   = world.skinned_states[collider.skinned_id];
        const TriMeshHit    hit  = mesh_closest(st.x, body.rest.triangles, st.normals, st.bvh, pos);
        dist              = hit.dist;
        out = Contact{i, ci, hit.normal, 0.0, false, 0.0}; // flat: no curvature
        out.surface_point = hit.point;
        out.triangle_id   = hit.triangle;
        out.feature       = hit.feature;
        out.feature_index = hit.index;
        out.bary          = hit.bary;
        if (hit.triangle >= 0)
            out.dvc_dx = skinned_velocity_gradient(world, collider.skinned_id, hit.triangle, hit.feature, hit.index);
        if (hit.triangle >= 0 && hit.feature == ContactFeature::Edge)
        {
            const auto& tri = body.rest.triangles[hit.triangle];
            out.edge_dir = (st.x[tri[(hit.index + 1) % 3]] - st.x[tri[hit.index]]).normalized();
        }
    }
    else // ColliderType::Capsule
    {
        const Vec3 p0 = pose.capsule_p0;
        const Vec3 p1 = pose.capsule_p1;
        const Vec3 axis_vec = p1 - p0;
        const Real L  = axis_vec.norm();
        const Vec3 a  = axis_vec / L;
        Real t = a.dot(pos - p0);
        t = std::clamp(t, 0.0, L);
        const Vec3 closest = p0 + t * a;
        const Vec3 offset  = pos - closest;
        const Real dist_from_axis = offset.norm();
        dist              = dist_from_axis - collider.capsule_radius;
        const Vec3 normal = offset / dist_from_axis;
        out = Contact{i, ci, normal, 1.0 / dist_from_axis, false, 0.0};
        out.axis          = (t > 0.0 && t < L) ? a : Vec3::Zero(); // cylinder regime vs. sphere-cap regime
        out.surface_point = closest + collider.capsule_radius * normal;
    }
    return dist;
}

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
