#pragma once

// A closed triangle body deformed by linear blend skinning, and the state it is posed into at a step.
// The .json reader is in collision/skinned_mesh_io.h.

#include "collision/trimesh.h"

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
inline AffineTransform skin_matrix_at_step(const SkinnedMesh& body, int bone, int substeps, int step)
{
    const int  last  = body.n_frames - 1;
    const int  frame = std::clamp(step / substeps, 0, last);
    const Real alpha = Real(step % substeps) / substeps;
    const AffineTransform& a = body.skin_at(frame, bone);
    if (alpha == 0.0) return a;
    const AffineTransform& b = body.skin_at(std::min(frame + 1, last), bone);
    AffineTransform out;
    out.R = a.R + alpha * (b.R - a.R);
    out.t = a.t + alpha * (b.t - a.t);
    return out;
}

// The body's pose at the current physics step, written by stamp_skinned_body. Shared (one per SkinnedMesh,
// CollisionWorld::skinned_states) so the forward pass, the backward pass and the viewer all read one state.
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
