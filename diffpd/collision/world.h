#pragma once

// The collision module's data: colliders, animation tracks, meshes and skinned bodies (CollisionWorld) and the
// contacts found against them (Contact). The per-shape maths is collision/shapes.h, the detection loop
// collision/detect.h (it takes stacked positions (3n), never a cloth Object, so any integrator can call it).
// Depends only on core/. The cloth-side glue (waist pins, advance_world) is in cloth/waist.h.

#include "collision/collider.h"
#include "collision/animation.h"
#include "collision/skinned_mesh.h"

// How often pd_contact re-tests the current iterate for newly penetrating particles:
// Once = only the pre-solve detection against x_tilde; Every = before every iteration;
// NTimes = n checks in total (the pre-solve one included), evenly spaced over the iterations.
enum class ContactRecheckMode { Once, Every, NTimes };

// RelativeVelocity: Ly et al., the constraint is (v+ - v_c).n >= 0 for the contacts found at x_tilde (optionally grown by rechecks).
// Gap: gap-aware ("speculative") contact with a fixed contact set: the constraint is "x+ not inside the collider's end-of-step
// pose" (phi(x-) + h v+.n >= 0), so a particle detected short of the surface lands on it instead of hovering, and one already
// inside is pushed out. See pd_contact.
enum class ContactModel { RelativeVelocity, Gap };

// Everything that selects how contacts are found and constrained; one value, copied from AppConfig into the CollisionWorld.
struct ContactOptions
{
    ContactModel       model = ContactModel::RelativeVelocity;
    Real               margin = 0.0; // Gap only: detection also takes particles this far outside the surface (at x_tilde); safe there (a contact that would not reach the surface is inactive) and lets the frozen set catch what the solve drags in
    ContactRecheckMode recheck = ContactRecheckMode::Once; // RelativeVelocity only: Gap always keeps its set fixed, as the backward pass assumes
    int                recheck_count = 5; // NTimes: total checks, the pre-solve one included (>= 2)
    Real               friction = 0.0; // Coulomb mu of the impulse method, every contact (forward only; needs RelativeVelocity). 0 = frictionless path

    bool gap() const { return model == ContactModel::Gap; }

    // detect_contacts' penetration_threshold: a negative one is the margin
    Real detect_threshold() const { return gap() ? -margin : 0.0; }

    // Iterations between rechecks before `recheck_end`: Every = 1, NTimes = recheck_end / n, Once (and Gap) = never (0).
    int recheck_period(int recheck_end) const
    {
        if (gap()) return 0;
        if (recheck == ContactRecheckMode::Every) return 1;
        if (recheck == ContactRecheckMode::NTimes) return std::max(1, recheck_end / std::max(2, recheck_count));
        return 0;
    }
};

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
    Mat3           dvc_dx        = Mat3::Zero(); // d v_c / d x_tilde (3x3), backward data cached like v_c: [omega]_x J_s for a rotating rigid collider,
                                                 // the blend of vertex velocity gradients for a skinned body; zero otherwise (and with speculative contact, v_c = 0)
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

    ContactOptions contact; // AppConfig::contact, copied by prepare_world

    // Fill the per-contact data only the backward pass reads (Contact::dvc_dx, Contact::gap_grad). Off for a forward-only
    // run; the Tape remembers it (Tape::has_backward_data) and backward_pd_contact refuses such a tape.
    bool collect_backward = true;

    // Poses skinned body `id` at physics state `step`, with the same step -> track-time mapping and end
    // clamping as the rigid tracks (anim_sample_at_step): state `step` sits step / substeps frames
    // in, and past the last frame the pose is held (v = 0). The deformed vertices are linearly interpolated
    // between the two bracketing frames, so every vertex moves at the constant velocity
    // v = (X1 - X0) * kColliderAnimFPS over a bracket. LBS only runs when the bracket changes; the normals
    // and the BVH (built on the first stamp, refit afterwards) are refreshed on every call so closest-point
    // queries always see the current pose.
    void stamp_skinned_body(int id, int substeps, int step);

    // Restamps every animated collider and skinned body to physics state `step`. Both passes go through
    // advance_world (cloth/waist.h), which also moves the cloth's waist pins.
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
