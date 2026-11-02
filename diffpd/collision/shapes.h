#pragma once

// Per-shape collision maths: one struct per ColliderType, each owning everything the solver needs to know about that
// shape (its pose at a time, its bounding box, the closest-point query that fills a Contact, and the Jacobian of its
// surface point). Adding a shape means adding one struct and one case in visit_shape; collider_pose_at /
// collider_aabb / contact_geometry / rigid_velocity_gradient dispatch through it. `diffpd --shape-check` checks each query's normal, curvature and surface-point Jacobian against
// central differences. Depends only on collision/world.h.
//
// query(world, collider, ci, pose, i, pos, out) fills `out` (normal, inv_r, axis, surface_point, and the mesh
// fields) for particle `i` at `pos` and returns the signed distance to the surface (< 0 = inside).
// J_s(c, R) = d(surface point)/d(x_tilde) (3x3), R the collider's surface radius; rigid_velocity_gradient turns it into
// dv_c/dx_tilde of a rotating collider (the forward pass stores that in Contact::dvc_dx for the backward pass).

#include "collision/world.h"

// ----------------
//     SPHERE
// ----------------

struct SphereShape
{
    static void pose(const Collider& c, Real time, ColliderPose& p) { p.sphere_center = collider_transform_point(c, c.sphere_center, time); }
    static AABB aabb(const CollisionWorld&, const Collider& c, const ColliderPose& pose)
    {
        AABB box;
        box.min   = pose.sphere_center - Vec3::Constant(c.sphere_radius);
        box.max   = pose.sphere_center + Vec3::Constant(c.sphere_radius);
        box.valid = true;
        return box;
    }
    static Real query(const CollisionWorld&, const Collider& collider, int ci, const ColliderPose& pose, ParticleId i, const Vec3& pos, Contact& out)
    {
        Real dist = 0.0;
        const Vec3 offset           = pos - pose.sphere_center;
        const Real dist_from_center = offset.norm();
        dist                        = dist_from_center - collider.sphere_radius;
        const Vec3 normal           = offset / dist_from_center;
        out = Contact{i, ci, normal, 1.0 / dist_from_center, false, 0.0};
        out.surface_point = pose.sphere_center + collider.sphere_radius * normal;
        return dist;
    }
    // (R/r)(I - n n^T)
    static Mat3 J_s(const Contact& c, Real R) { return (R * c.inv_r) * (Mat3::Identity() - c.normal * c.normal.transpose()); }
};

// ----------------
//    CYLINDER
// ----------------

struct CylinderShape
{
    static void pose(const Collider& c, Real time, ColliderPose& p)
    {
        p.cylinder_origin = collider_transform_point(c, c.cylinder_origin, time);
        p.cylinder_axis   = collider_transform_direction(c, c.cylinder_axis, time);
    }
    static AABB aabb(const CollisionWorld&, const Collider&, const ColliderPose&) { return AABB{}; } // unbounded: valid = false
    static Real query(const CollisionWorld&, const Collider& collider, int ci, const ColliderPose& pose, ParticleId i, const Vec3& pos, Contact& out)
    {
        Real dist = 0.0;
        const Vec3 axis   = pose.cylinder_axis.normalized();
        const Vec3 rel    = pos - pose.cylinder_origin;
        const Vec3 perp   = rel - rel.dot(axis) * axis;
        const Real rho    = perp.norm();
        dist              = rho - collider.cylinder_radius;
        const Vec3 normal = perp / rho;
        out = Contact{i, ci, normal, 1.0 / rho, false, 0.0};
        out.axis          = axis;
        out.surface_point = (pos - perp) + collider.cylinder_radius * normal; // pos - perp = closest point on axis
        return dist;
    }
    // identity along the axis, (R/rho)(I - n n^T - a a^T) across it (shared with the capsule body)
    static Mat3 body_J_s(const Contact& c, Real R)
    {
        const Mat3 aat = c.axis * c.axis.transpose();
        return aat + (R * c.inv_r) * (Mat3::Identity() - c.normal * c.normal.transpose() - aat);
    }
    static Mat3 J_s(const Contact& c, Real R) { return body_J_s(c, R); }
};

// ----------------
//     PLANE
// ----------------

struct PlaneShape
{
    static void pose(const Collider& c, Real time, ColliderPose& p)
    {
        p.plane_origin = collider_transform_point(c, c.plane_origin, time);
        p.plane_normal = collider_transform_direction(c, c.plane_normal, time);
    }
    static AABB aabb(const CollisionWorld&, const Collider&, const ColliderPose&) { return AABB{}; } // unbounded: valid = false
    static Real query(const CollisionWorld&, const Collider&, int ci, const ColliderPose& pose, ParticleId i, const Vec3& pos, Contact& out)
    {
        Real dist = 0.0;
        const Vec3 normal = pose.plane_normal.normalized();
        dist              = (pos - pose.plane_origin).dot(normal);
        out = Contact{i, ci, normal, 0.0, false, 0.0};
        out.surface_point = pos - dist * normal;
        return dist;
    }
    // a rotating plane is not supported (asserted in backward_pd_contact)
    static Mat3 J_s(const Contact&, Real) { return Mat3::Zero(); }
};

// ----------------
//    CAPSULE
// ----------------

struct CapsuleShape
{
    static void pose(const Collider& c, Real time, ColliderPose& p)
    {
        p.capsule_p0 = collider_transform_point(c, c.capsule_p0, time);
        p.capsule_p1 = collider_transform_point(c, c.capsule_p1, time);
    }
    static AABB aabb(const CollisionWorld&, const Collider& c, const ColliderPose& pose)
    {
        AABB box;
        box.min   = pose.capsule_p0.cwiseMin(pose.capsule_p1) - Vec3::Constant(c.capsule_radius);
        box.max   = pose.capsule_p0.cwiseMax(pose.capsule_p1) + Vec3::Constant(c.capsule_radius);
        box.valid = true;
        return box;
    }
    static Real query(const CollisionWorld&, const Collider& collider, int ci, const ColliderPose& pose, ParticleId i, const Vec3& pos, Contact& out)
    {
        Real dist = 0.0;
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
        return dist;
    }
    // cylinder regime (axis set) or sphere-cap regime (axis zero)
    static Mat3 J_s(const Contact& c, Real R)
    {
        if (c.axis.squaredNorm() > 0.0) return CylinderShape::body_J_s(c, R);
        return SphereShape::J_s(c, R);
    }
};

// ----------------
//   TRIANGLE MESH
// ----------------

struct MeshShape
{
    static void pose(const Collider&, Real, ColliderPose&) {} // pose lives elsewhere (see below)
    static AABB aabb(const CollisionWorld& world, const Collider& c, const ColliderPose&)
    {
        AABB box;
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
        return box;
    }
    static Real query(const CollisionWorld& world, const Collider& collider, int ci, const ColliderPose&, ParticleId i, const Vec3& pos, Contact& out)
    {
        Real dist = 0.0;
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
        return dist;
    }
    // I - n n^T on a face, e e^T on an edge, 0 on a vertex
    static Mat3 J_s(const Contact& c, Real)
    {
        if (c.feature == ContactFeature::Face) return Mat3::Identity() - c.normal * c.normal.transpose();
        if (c.feature == ContactFeature::Edge) return c.edge_dir * c.edge_dir.transpose();
        return Mat3::Zero();
    }
};

// ----------------
//   SKINNED BODY
// ----------------

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

struct SkinnedShape
{
    static void pose(const Collider&, Real, ColliderPose&) {} // pose lives elsewhere (see below)
    static AABB aabb(const CollisionWorld& world, const Collider& c, const ColliderPose&)
    {
        AABB box;
        // The BVH root already bounds the deformed vertices at the stamped state (empty before any stamp).
        const Bvh& bvh = world.skinned_states[c.skinned_id].bvh;
        if (!bvh.nodes.empty())
        {
            box.min   = bvh.nodes[0].box.lo;
            box.max   = bvh.nodes[0].box.hi;
            box.valid = true;
        }
        return box;
    }
    static Real query(const CollisionWorld& world, const Collider& collider, int ci, const ColliderPose&, ParticleId i, const Vec3& pos, Contact& out)
    {
        Real dist = 0.0;
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
        if (hit.triangle >= 0 && hit.feature == ContactFeature::Edge)
        {
            const auto& tri = body.rest.triangles[hit.triangle];
            out.edge_dir = (st.x[tri[(hit.index + 1) % 3]] - st.x[tri[hit.index]]).normalized();
        }
        return dist;
    }
    // not a rigid motion: pd_contact fills Contact::dvc_dx from skinned_velocity_gradient instead
    static Mat3 J_s(const Contact&, Real) { return Mat3::Zero(); }
};

// ----------------
//     NONE
// ----------------

struct NoShape
{
    static void pose(const Collider&, Real, ColliderPose&) {} // pose lives elsewhere (see below)
    static AABB aabb(const CollisionWorld&, const Collider&, const ColliderPose&) { return AABB{}; } // unbounded: valid = false
    static Real query(const CollisionWorld&, const Collider&, int, const ColliderPose&, ParticleId, const Vec3&, Contact&)
    {
        ASSERT(false, "contact_geometry on a None collider");
        return 0.0;
    }
    static Mat3 J_s(const Contact&, Real) { return Mat3::Zero(); }
};

// Calls f(shape) with the shape struct of `type`; every f must return the same type.
template <class F>
inline decltype(auto) visit_shape(ColliderType type, F&& f)
{
    switch (type)
    {
        case ColliderType::Sphere:   return f(SphereShape{});
        case ColliderType::Cylinder: return f(CylinderShape{});
        case ColliderType::Plane:    return f(PlaneShape{});
        case ColliderType::Capsule:  return f(CapsuleShape{});
        case ColliderType::Mesh:     return f(MeshShape{});
        case ColliderType::Skinned:  return f(SkinnedShape{});
        case ColliderType::None:     break;
    }
    return f(NoShape{});
}

// [w]_x, so that skew(w) * v = w x v.
inline Mat3 skew(const Vec3& w)
{
    Mat3 m;
    m <<  0.0,    -w.z(),  w.y(),
          w.z(),   0.0,   -w.x(),
         -w.y(),   w.x(),  0.0;
    return m;
}

// G = dv_c/dx_tilde of a rigid collider rotating at `omega`: v_c = v + omega x (q - pivot) with q the contact's surface
// point, so only q depends on x_tilde and G = [omega]_x J_s. Zero if it does not rotate.
inline Mat3 rigid_velocity_gradient(const Collider& collider, const Contact& c, const Vec3& omega)
{
    if (omega.squaredNorm() == 0.0) return Mat3::Zero();
    const Real R = collider_surface_radius(collider);
    return skew(omega) * visit_shape(collider.type, [&](auto shape) { return shape.J_s(c, R); });
}

inline ColliderPose collider_pose_at(const Collider& c, Real time)
{
    ColliderPose pose{};
    visit_shape(c.type, [&](auto shape) { shape.pose(c, time, pose); });
    return pose;
}

inline AABB collider_aabb(const CollisionWorld& world, const Collider& c, const ColliderPose& pose)
{
    return visit_shape(c.type, [&](auto shape) { return shape.aabb(world, c, pose); });
}

inline bool aabb_contains(const AABB& box, const Vec3& p)
{
    return (p.array() >= box.min.array()).all() && (p.array() <= box.max.array()).all();
}

// Fills `out` for particle `i` at `pos` against one collider pose, and returns the signed distance to the surface
// (< 0 = inside). Split from detect_contacts so merge_detected_contacts can test penetration at one position but
// evaluate geometry at another.
inline Real contact_geometry(const CollisionWorld& world, const Collider& collider, int ci, const ColliderPose& pose, ParticleId i, const Vec3& pos, Contact& out)
{
    return visit_shape(collider.type, [&](auto shape) { return shape.query(world, collider, ci, pose, i, pos, out); });
}
