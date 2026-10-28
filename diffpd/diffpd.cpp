#include <map>
#include "diffpd_types.h"
#include "diffpd_viewer.h"
#include "diffpd_timer.h"
#include "anderson.h"
#include "trimesh.h"
#include "skinned_mesh.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <nlohmann/json.hpp>

#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <limits>
#include <functional>
#include <cmath>

namespace fs = std::filesystem;

std::string ANIM_DIR;

// ----------------
//      FILE
// ----------------

void clear_folder(const std::string& folder)
{
    namespace fs = std::filesystem;

    if (!fs::exists(folder))
    {
        fs::create_directories(folder);
        return;
    }

    ASSERT(fs::is_directory(folder), "clear_folder: not a directory: " << folder);

    for (const auto& entry : fs::directory_iterator(folder))
        if (entry.is_regular_file() && entry.path().extension() == ".obj")
            fs::remove(entry.path());
}

// ----------------
//    CONTACT
// ----------------

std::vector<Collider> colliders;                                 // default empty; populated in main()

// How often pd_contact grows the contact set inside its iteration loop (see merge_detected_contacts); overwritten in main().
ContactRecheckMode contact_recheck_mode  = ContactRecheckMode::Once;
int                contact_recheck_count = 10;
bool               speculative_contact   = false; // gap-aware contact (AppConfig::speculative_contact); overwritten in main()
Real               contact_margin        = 0.0;   // speculative only: detection distance outside the surface; overwritten in main()
int                anderson_m            = 0;     // Anderson history size, 0 = off (AppConfig::anderson_m); overwritten in main()

// Parallel to `colliders`: colliders[i].anim_id, when >= 0, indexes into this track list.
std::vector<ColliderAnimation> collider_animations;

// Rest-frame triangle meshes referenced by Mesh colliders. Replaced on every load_collider_animation
// call, so only the most recently loaded animation's meshes are valid.
std::vector<TriMesh> trimeshes;

// Skinned bodies referenced by Skinned colliders; refilled by load_collider_animation like `trimeshes`.
std::vector<SkinnedMesh>  skinned_meshes;
std::vector<SkinnedState> skinned_states; // parallel to skinned_meshes

// Declared in diffpd_types.h (see there for the step -> pose mapping).
void stamp_skinned_body(int id, int step)
{
    const SkinnedMesh& body = skinned_meshes[id];
    SkinnedState&      st   = skinned_states[id];

    const int  last  = body.n_frames - 1;
    const int  frame = std::clamp(step / body.steps_per_frame, 0, last);
    const Real alpha = Real(step % body.steps_per_frame) / body.steps_per_frame;
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
static Vec3 skinned_surface_velocity(int id, int triangle, const Vec3& bary)
{
    const SkinnedState&      st  = skinned_states[id];
    const std::array<int, 3>& tri = skinned_meshes[id].rest.triangles[triangle];
    return bary[0] * st.v[tri[0]] + bary[1] * st.v[tri[1]] + bary[2] * st.v[tri[2]];
}

// G = d v_c / d x_tilde for a contact on a skinned-body triangle (only the barycentrics depend on x_tilde):
//   Face: sum_k v_k g_k^T, g_k = n x (x_{k+2} - x_{k+1}) / (2A)   Edge: (v_b - v_a) e^T / L   Vertex: 0
static Mat3 skinned_velocity_gradient(int id, int triangle, ContactFeature feature, int index)
{
    const SkinnedState&       st  = skinned_states[id];
    const std::array<int, 3>& tri = skinned_meshes[id].rest.triangles[triangle];
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

// Name of the collider (Blender empty / animation .json entry) that Waist Attachment pins to.
constexpr const char* kWaistAttachmentColliderName = "collider_hip";

// Bone of a skinned body that Waist Attachment pins to (rigidly, weight 1); see bake_waist_attachment_skinned.
constexpr const char* kWaistAttachmentBoneName = "pelvis";

// The animation-owned globals (mesh registry, skinned bodies) outlive a load, so "Animation: None" must reset them too.
static void clear_animation_assets()
{
    trimeshes.clear();
    skinned_meshes.clear();
    skinned_states.clear();
}

// Parses an animation .json, appending one `animated=true` Collider per supported "colliders_metadata"
// entry and returning the matching frame tracks. Missing/unreadable file just warns and returns
// empty rather than failing, since this is an optional layer on the config-managed collider list.
// Entries with an unsupported "type" (anything but sphere/capsule/mesh), or a mesh that fails to
// load, are skipped with a warning. A mesh entry's "mesh_file" is relative to the .json's folder.
std::vector<ColliderAnimation> load_collider_animation(const std::string& path, std::vector<Collider>& out_colliders)
{
    std::vector<ColliderAnimation> anims;
    clear_animation_assets();

    std::ifstream in(path);
    if (!in.is_open())
    {
        WARNING("load_collider_animation: could not open " << path << " (skipping animated colliders)");
        return anims;
    }

    nlohmann::json j;
    try { in >> j; }
    catch (const std::exception& e)
    {
        WARNING("load_collider_animation: failed to parse " << path << ": " << e.what());
        return anims;
    }

    // A skinned-body file (animation/export_skinned_body.py) shares this folder and the Animation
    // dropdown but has no "colliders_metadata": it becomes one Skinned collider and no rigid tracks.
    if (j.is_object() && j.value("format", std::string()) == "diffpd_skinned_body")
    {
        SkinnedMesh body;
        if (load_skinned_body(j, fs::path(path).stem().string(), body))
        {
            Collider c;
            c.type       = ColliderType::Skinned;
            c.skinned_id = (int)skinned_meshes.size();
            skinned_meshes.push_back(std::move(body));
            skinned_states.emplace_back();
            stamp_skinned_body(c.skinned_id, 0); // so previews/viewers have a valid pose before any run
            out_colliders.push_back(c);
        }
        return anims;
    }
    if (!j.is_object() || !j.contains("colliders_metadata") || !j.contains("frames"))
    {
        WARNING("load_collider_animation: " << path << ": no \"colliders_metadata\"/\"frames\" (skipping)");
        return anims;
    }

    // name -> index into `anims`, so the per-frame loop below can find each track by name.
    std::unordered_map<std::string, int> name_to_anim_id;

    for (auto& [name, meta] : j.at("colliders_metadata").items())
    {
        const std::string type_str = meta.value("type", "capsule");
        if (type_str != "sphere" && type_str != "capsule" && type_str != "mesh")
        {
            WARNING("load_collider_animation: " << path << ": skipping '" << name
                    << "' (unsupported collider type '" << type_str << "')");
            continue;
        }

        ColliderAnimation anim;
        anim.name = name;
        anim.type = (type_str == "sphere") ? ColliderType::Sphere
                  : (type_str == "mesh")   ? ColliderType::Mesh
                                           : ColliderType::Capsule;

        if (anim.type == ColliderType::Mesh)
        {
            TriMesh mesh;
            if (meta.contains("mesh")) // inline OBJ text under the top-level "meshes" block
            {
                const std::string key = meta["mesh"].get<std::string>();
                if (!j.contains("meshes") || !j["meshes"].contains(key))
                {
                    WARNING("load_collider_animation: " << path << ": '" << name << "' references missing mesh '" << key << "'");
                    continue;
                }
                std::istringstream obj_text(j["meshes"][key].get<std::string>());
                if (!load_trimesh_obj(obj_text, key, mesh)) continue;
            }
            else
            {
                const fs::path mesh_path = fs::path(path).parent_path() / meta.value("mesh_file", "");
                if (!load_trimesh_obj(mesh_path.string(), mesh)) continue; // load_trimesh_obj already warned
            }
            std::cout << "[mesh] " << name << ": " << mesh.vertices.size() << " vertices, "
                      << mesh.triangles.size() << " triangles\n";
            anim.mesh_id = (int)trimeshes.size();
            trimeshes.push_back(std::move(mesh));
        }

        anim.radius      = meta.value("radius", 0.0);
        anim.half_length = meta.value("half_length", 0.0);
        const auto axis  = meta.value("axis_local", std::vector<Real>{0.0, 1.0, 0.0});
        anim.axis_local  = Vec3(axis[0], axis[1], axis[2]);

        Collider c;
        c.type     = anim.type;
        c.animated = true;
        c.anim_id  = (int)anims.size();
        c.mesh_id  = anim.mesh_id;

        name_to_anim_id[name] = (int)anims.size();
        anims.push_back(anim);
        out_colliders.push_back(c);
    }

    for (const auto& frame_entry : j.at("frames"))
    {
        for (auto& [name, pose] : frame_entry.at("colliders").items())
        {
            auto it = name_to_anim_id.find(name);
            if (it == name_to_anim_id.end()) continue; // no metadata entry for this name; ignore

            const auto pos = pose.at("position").get<std::vector<Real>>();
            const auto rot = pose.at("rotation").get<std::vector<Real>>(); // [x, y, z, w]

            ColliderFrame f;
            f.position = Vec3(pos[0], pos[1], pos[2]);
            f.rotation = Eigen::Quaternion<Real>(rot[3], rot[0], rot[1], rot[2]); // ctor is (w, x, y, z)

            anims[it->second].frames.push_back(f);
        }
    }

    return anims;
}

std::string animation_path(const std::string& filename)
{
    return (fs::path(COLLIDER_ANIM_DIR_DEFAULT) / filename).string();
}

std::vector<std::string> list_animation_files()
{
    std::vector<std::string> files;
    std::error_code          ec;
    for (const auto& entry : fs::directory_iterator(COLLIDER_ANIM_DIR_DEFAULT, ec))
        if (entry.is_regular_file() && entry.path().extension() == ".json")
            files.push_back(entry.path().filename().string());
    std::sort(files.begin(), files.end());
    return files;
}

AnimationPreview load_animation_preview(const std::string& filename)
{
    AnimationPreview preview;
    if (filename.empty()) return preview;

    std::vector<ColliderAnimation> anims = load_collider_animation(animation_path(filename), preview.colliders);
    for (Collider& c : preview.colliders)
        if (c.anim_id >= 0 && c.anim_id < (int)anims.size() && !anims[c.anim_id].frames.empty())
            apply_collider_at_step(c, anims[c.anim_id], 0);

    for (const ColliderAnimation& anim : anims)
        if (anim.name == kWaistAttachmentColliderName && !anim.frames.empty())
        {
            preview.has_waist            = true;
            preview.waist_snaps_origin   = true;
            preview.waist_default_origin = collider_animation_pose_at(anim, 0).position;
            break;
        }

    // A skinned body is attached through its pelvis bone, and the cloth is placed by hand (no origin snap).
    if (!skinned_meshes.empty())
    {
        preview.has_skinned = true;
        if (!preview.has_waist && find_bone(skinned_meshes[0], kWaistAttachmentBoneName) >= 0)
            preview.has_waist = true;
    }
    return preview;
}

void count_cloth_inside_skinned(const Object& obj, int& free_inside, int& pinned_inside)
{
    free_inside = pinned_inside = 0;
    for (int id = 0; id < (int)skinned_meshes.size(); ++id)
    {
        stamp_skinned_body(id, 0);
        const SkinnedMesh&  body = skinned_meshes[id];
        const SkinnedState& st   = skinned_states[id];
        const auto inside = [&](const Vec3& p)
        { return mesh_closest(st.x, body.rest.triangles, st.normals, st.bvh, p).dist < 0.0; };
        for (Index i = 0; i < obj.num_particles(); ++i)
            free_inside += inside(obj.x.segment<3>(3 * i));
        for (const Vec3& p : obj.mesh.pinned_rest)
            pinned_inside += inside(p);
    }
}

// Fills `out` (normal, inv_r, axis, surface_point) for particle `i` at `pos` against one collider
// pose, and returns the signed distance to the surface (< 0 = inside). Split from detect_contacts so
// merge_detected_contacts can test penetration at one position but evaluate geometry at another.
Real contact_geometry(const Collider& collider, int ci, const ColliderPose& pose, ParticleId i, const Vec3& pos, Contact& out)
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
        const TriMeshHit hit = trimesh_closest(trimeshes[collider.mesh_id], mp.rotation.conjugate() * (pos - mp.position));
        dist              = hit.dist;
        out = Contact{i, ci, mp.rotation * hit.normal, 0.0, false, 0.0}; // flat: no curvature
        out.surface_point = mp.position + mp.rotation * hit.point;
        out.triangle_id   = hit.triangle;
        out.feature       = hit.feature;
        out.feature_index = hit.index;
        if (hit.feature == ContactFeature::Edge)
        {
            const TriMesh& mesh = trimeshes[collider.mesh_id];
            const auto&    tri  = mesh.triangles[hit.triangle];
            out.edge_dir = mp.rotation * (mesh.vertices[tri[(hit.index + 1) % 3]] - mesh.vertices[tri[hit.index]]).normalized();
        }
    }
    else if (collider.type == ColliderType::Skinned)
    {
        // The body is already in world coordinates: query the deformed mesh at the stamped state.
        const SkinnedMesh&  body = skinned_meshes[collider.skinned_id];
        const SkinnedState& st   = skinned_states[collider.skinned_id];
        const TriMeshHit    hit  = mesh_closest(st.x, body.rest.triangles, st.normals, st.bvh, pos);
        dist              = hit.dist;
        out = Contact{i, ci, hit.normal, 0.0, false, 0.0}; // flat: no curvature
        out.surface_point = hit.point;
        out.triangle_id   = hit.triangle;
        out.feature       = hit.feature;
        out.feature_index = hit.index;
        out.bary          = hit.bary;
        if (hit.triangle >= 0)
            out.dvc_dx = skinned_velocity_gradient(collider.skinned_id, hit.triangle, hit.feature, hit.index);
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
Contacts detect_contacts(const Object& obj, const Positions& x, Real time, Real penetration_threshold = 0.0,
                         const std::vector<bool>* pre_claimed = nullptr)
{
    Contacts contacts;
    std::vector<bool> claimed = pre_claimed ? *pre_claimed : std::vector<bool>(obj.num_particles(), false);

    for (int ci = 0; ci < (int)colliders.size(); ++ci)
    {
        const Collider& collider = colliders[ci];
        if (collider.type == ColliderType::None) continue;

        const ColliderPose pose = collider_pose_at(collider, time);
        AABB               box  = collider_aabb(collider, pose); // valid only for finite shapes (Sphere/Capsule/Mesh/Skinned)
        const Vec3 grow = Vec3::Constant(std::max(0.0, -penetration_threshold)); // a margin grows the box too
        box.min -= grow; box.max += grow;

        for (Index i = 0; i < obj.num_particles(); ++i)
        {
            if (claimed[i]) continue;

            const Vec3 pos = x.segment<3>(3*i);

            // Cheap reject before the real distance test; skipped for Cylinder/Plane (no finite bound).
            if (box.valid && !aabb_contains(box, pos)) continue;

            Contact c;
            if (contact_geometry(collider, ci, pose, static_cast<ParticleId>(i), pos, c) < -penetration_threshold)
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
int merge_detected_contacts(Contacts& contacts, const Object& obj, const Positions& x_test, const Positions& x_geom, Real time)
{
    std::vector<bool> claimed(obj.num_particles(), false);
    for (const Contact& c : contacts) claimed[c.particle] = true;

    const Contacts fresh = detect_contacts(obj, x_test, time, 0.0, &claimed);
    for (const Contact& hit : fresh)
    {
        const Collider& collider = colliders[hit.collider_id];
        Contact c;
        contact_geometry(collider, hit.collider_id, collider_pose_at(collider, time),
                         hit.particle, x_geom.segment<3>(3 * hit.particle), c);
        contacts.push_back(c);
    }
    return (int)fresh.size();
}

// d = f_i - m_i * v_c : the free-force target shifted by the obstacle's own translation
// velocity (Ly et al., "Moving obstacle", Eq. 10: d_j := R_j^T f_i + M_i u_f,j, u_f,j = -v_c).
// Speculative contact: v_c = 0 and gap_speed = phi(x-)/h, i.e. v+.n >= -phi(x-)/h.
Vec3 update_contact_force(const Vec3& f_i, Real m_i, const Vec3& v_c, const Vec3& n, Real gap_speed = 0.0)
{
    const Real d_n = (f_i - m_i * v_c).dot(n) + m_i * gap_speed;
    if (d_n >= 0.0) return Vec3::Zero();
    return -d_n * n;
}

// ----------------
//       LOSS
// ----------------

struct Loss
{
    Real                  total;
    std::vector<RealVecX> dloss_dx;

    Loss(const Tape& guess, const Tape& target, int sample_every, int target_first = 0) // target_first: target frame that guess frame 0 is compared with (a chunk)
    {
        ASSERT((int)guess.positions.size() + target_first <= (int)target.positions.size(),
               "tape size mismatch: " << guess.positions.size() << " (+ offset " << target_first << ") vs " << target.positions.size());
        ASSERT(sample_every > 0, "sample_every must be positive");

        const int   n_frames = (int)guess.positions.size();
        const int   n_steps  = n_frames - 1;
        const Index dofs     = 3 * guess.positions[0].rows();

        total = 0.0;
        dloss_dx.resize(n_frames, RealVecX::Zero(dofs));

        for (int t = 0; t < n_frames; ++t)
        {
            const bool sampled = (t == n_steps) || ((n_steps - t) % sample_every == 0);
            if (!sampled) continue;

            const RealVecX xg = Eigen::Map<const RealVecX>(guess.positions[t].data(),  dofs);
            const RealVecX xt = Eigen::Map<const RealVecX>(target.positions[target_first + t].data(), dofs);

            const RealVecX diff = xg - xt;
            total       += diff.squaredNorm() / Real(dofs);
            dloss_dx[t]  = (2.0 / Real(dofs)) * diff;
        }
    }
};

// ----------------
//      CLOTH
// ----------------

// PinMode, HangingMode, ClothFlags, and cloth()'s declaration live in diffpd_types.h.

Object cloth(
    Index       width,
    Index       height,
    Real        size_x,
    Real        size_y,
    Stiffness   stiffness,
    Vec3        origin,
    PinMode     pin_mode,
    HangingMode hanging_mode,
    uint8_t     flags,
    Real        m_tot)
{
    ASSERT(flags & ClothFlags::STRETCH, "cloth must have stretch constraints enabled");

    const Index N  = width * height;
    const Real  sx = size_x / Real(width  - 1);
    const Real  sz = size_y / Real(height - 1);

    auto grid = [height](Index i, Index j) { return i * height + j; };

    // HORIZONTAL: (i,j) -> (x, 0, z), flat. VERTICAL: (i,j) -> (x, -z, 0), j=0 row hangs from the top.
    std::vector<Vec3> pos(N);
    for (Index i = 0; i < width; ++i)
        for (Index j = 0; j < height; ++j)
            pos[grid(i, j)] = origin + (hanging_mode == HangingMode::VERTICAL
                ? Vec3(i * sx, -j * sz, 0.0)
                : Vec3(i * sx, 0.0, j * sz));

    // --- mark pinned vertices ---------------------------------------------
    std::vector<bool> pinned(N, false);
    switch (pin_mode) {
        case PinMode::NONE:
            break;
        case PinMode::CORNERS:
            pinned[grid(0, 0)]         = true;
            pinned[grid(width - 1, 0)] = true;
            break;
        case PinMode::ROW:
            for (Index i = 0; i < width; ++i)
                pinned[grid(i, 0)] = true;
            break;
    }

    // --- compact free vertices into DOF indices (V_free, DOF_map) ---------
    // dof[v] = particle id in the stacked state, or -1 if pinned.
    std::vector<ParticleId> dof(N, -1);
    Index n = 0; // |V_free|
    for (Index v = 0; v < N; ++v)
        if (!pinned[v]) dof[v] = ParticleId(n++);

    Object obj;

    // --- state vectors (stacked 3n, free particles only) ------------------
    obj.x.resize(3 * n);
    for (Index v = 0; v < N; ++v)
        if (dof[v] >= 0)
            obj.x.segment<3>(3 * dof[v]) = pos[v];

    obj.v      = Velocities::Zero(3 * n);
    obj.prev_x = obj.x;

    // --- mass: m_tot distributed uniformly over n free particles ----------
    obj.mass = MassDiag::Constant(3 * n, m_tot / Real(n));

    // ---- bake export mesh (after dof[] and pos are built) ------------------
    SimMesh& mesh = obj.mesh;
    mesh.width  = width;
    mesh.height = height;

    for (Index i = 0; i < width; ++i)
        for (Index j = 0; j < height; ++j)
        {
            const Index v = grid(i, j);
            SimMesh::Vertex mv;
            if (dof[v] >= 0)
            {
                mv.dof = dof[v];
            }
            else
            {
                mv.dof = ParticleId(-Index(mesh.pinned_rest.size()) - 1);
                mesh.pinned_rest.push_back(pos[v]);
            }
            mesh.vertices.push_back(mv);
        }

    // structural edges only (both endpoints, pinned or not)
    auto mesh_edge = [&](Index a, Index b)
    {
        mesh.edges.emplace_back(a, b);
    };

    for (Index i = 0; i < width - 1; ++i)
        for (Index j = 0; j < height; ++j)
            mesh_edge(grid(i, j), grid(i + 1, j));
    for (Index i = 0; i < width; ++i)
        for (Index j = 0; j < height - 1; ++j)
            mesh_edge(grid(i, j), grid(i, j + 1));

    // --- constraint emission ----------------------------------------------
    // mesh.vertices[v].dof, not dof[v]: for a pinned vertex dof[v] is just the shared -1 sentinel.
    auto emit = [&](Index a, Index b, SpringKind kind)
    {
        obj.energies.springs.add_edge(mesh.vertices[a].dof, mesh.vertices[b].dof, pos[a], pos[b], kind, stiffness[kind]);
    };

    if (flags & ClothFlags::STRETCH)
    {
        // structural: along width axis (i)
        for (Index i = 0; i < width - 1; ++i)
            for (Index j = 0; j < height; ++j)
                emit(grid(i, j), grid(i + 1, j), SpringKind::Stretch);

        // structural: along height axis (j)
        for (Index i = 0; i < width; ++i)
            for (Index j = 0; j < height - 1; ++j)
                emit(grid(i, j), grid(i, j + 1), SpringKind::Stretch);
    }

    if (flags & ClothFlags::SHEAR)
    {
        for (Index i = 0; i < width - 1; ++i)
            for (Index j = 0; j < height - 1; ++j)
            {
                emit(grid(i, j),     grid(i + 1, j + 1), SpringKind::Shear);
                emit(grid(i + 1, j), grid(i, j + 1),     SpringKind::Shear);
            }
    }

    if (flags & ClothFlags::BENDING)
    {
        for (Index i = 0; i < width; ++i)
            for (Index j = 0; j < height - 2; ++j)
                emit(grid(i, j), grid(i, j + 2), SpringKind::Bend);

        for (Index i = 0; i < width - 2; ++i)
            for (Index j = 0; j < height; ++j)
                emit(grid(i, j), grid(i + 2, j), SpringKind::Bend);
    }

    return obj;
}

// ----------------
//      SKIRT
// ----------------

// A conical-frustum ("skirt") cloth: `num_rings` rings of `particles_per_ring` vertices, stacked
// along -Y from a pinned `radius_top` ring at `origin` to `radius_bottom` at `height` below
// (interpolated). Same (i,j) grid as cloth(), but i wraps (i=W-1 connects to i=0) while j stays
// open, so it reads as a skirt, not a closed cylinder.
Object skirt(
    Index   particles_per_ring,
    Index     num_rings,
    Stiffness stiffness,
    Vec3      origin,
    Real      radius_top,
    Real      radius_bottom,
    Real      height,
    uint8_t   flags,
    Real      m_tot)
{
    ASSERT(flags & ClothFlags::STRETCH, "skirt must have stretch constraints enabled");
    ASSERT(particles_per_ring >= 3, "skirt needs at least 3 particles per ring");
    ASSERT(num_rings >= 2, "skirt needs at least 2 rings (top + bottom)");

    const Index W = particles_per_ring; // circumferential axis, wraps
    const Index H = num_rings;          // vertical axis, open (no cap)

    auto grid = [H](Index i, Index j) { return i * H + j; };
    auto wrap = [W](Index i) { return (i + W) % W; };

    constexpr Real kTwoPi = 6.283185307179586476925286766559;

    // j=0 is the pinned top ring (radius_top); j=H-1 is radius_bottom, hanging -Y by `height`.
    std::vector<Vec3> pos(W * H);
    for (Index i = 0; i < W; ++i)
    {
        const Real angle = kTwoPi * Real(i) / Real(W);
        for (Index j = 0; j < H; ++j)
        {
            const Real t = Real(j) / Real(H - 1);
            const Real r = radius_top + t * (radius_bottom - radius_top);
            pos[grid(i, j)] = origin + Vec3(r * std::cos(angle), -t * height, r * std::sin(angle));
        }
    }

    // --- mark pinned vertices (top ring only) ------------------------------
    std::vector<bool> pinned(W * H, false);
    for (Index i = 0; i < W; ++i)
        pinned[grid(i, 0)] = true;

    // --- compact free vertices into DOF indices (V_free, DOF_map) ---------
    std::vector<ParticleId> dof(W * H, -1);
    Index n = 0; // |V_free|
    for (Index v = 0; v < W * H; ++v)
        if (!pinned[v]) dof[v] = ParticleId(n++);

    Object obj;

    // --- state vectors (stacked 3n, free particles only) ------------------
    obj.x.resize(3 * n);
    for (Index v = 0; v < W * H; ++v)
        if (dof[v] >= 0)
            obj.x.segment<3>(3 * dof[v]) = pos[v];

    obj.v      = Velocities::Zero(3 * n);
    obj.prev_x = obj.x;

    // --- mass: m_tot distributed uniformly over n free particles ----------
    obj.mass = MassDiag::Constant(3 * n, m_tot / Real(n));

    // ---- bake export mesh (after dof[] and pos are built) ------------------
    SimMesh& mesh = obj.mesh;
    mesh.width  = W;
    mesh.height = H;
    mesh.wrap_i = true; // the ring direction (i) is a closed loop, unlike cloth()'s open sheet

    for (Index i = 0; i < W; ++i)
        for (Index j = 0; j < H; ++j)
        {
            const Index v = grid(i, j);
            SimMesh::Vertex mv;
            if (dof[v] >= 0)
            {
                mv.dof = dof[v];
            }
            else
            {
                mv.dof = ParticleId(-Index(mesh.pinned_rest.size()) - 1);
                mesh.pinned_rest.push_back(pos[v]);
            }
            mesh.vertices.push_back(mv);
        }

    // structural edges only (both endpoints, pinned or not); circumferential edges wrap around the
    // ring, vertical edges don't (open top/bottom).
    auto mesh_edge = [&](Index a, Index b)
    {
        mesh.edges.emplace_back(a, b);
    };

    for (Index i = 0; i < W; ++i)
        for (Index j = 0; j < H; ++j)
            mesh_edge(grid(i, j), grid(wrap(i + 1), j));
    for (Index i = 0; i < W; ++i)
        for (Index j = 0; j < H - 1; ++j)
            mesh_edge(grid(i, j), grid(i, j + 1));

    // --- constraint emission ----------------------------------------------
    // mesh.vertices[v].dof, not dof[v]: for a pinned vertex dof[v] is just the shared -1 sentinel.
    auto emit = [&](Index a, Index b, SpringKind kind)
    {
        obj.energies.springs.add_edge(mesh.vertices[a].dof, mesh.vertices[b].dof, pos[a], pos[b], kind, stiffness[kind]);
    };

    if (flags & ClothFlags::STRETCH)
    {
        // structural: circumferential (i), wraps around the ring
        for (Index i = 0; i < W; ++i)
            for (Index j = 0; j < H; ++j)
                emit(grid(i, j), grid(wrap(i + 1), j), SpringKind::Stretch);

        // structural: vertical (j), open (no top/bottom cap)
        for (Index i = 0; i < W; ++i)
            for (Index j = 0; j < H - 1; ++j)
                emit(grid(i, j), grid(i, j + 1), SpringKind::Stretch);
    }

    if (flags & ClothFlags::SHEAR)
    {
        for (Index i = 0; i < W; ++i)
            for (Index j = 0; j < H - 1; ++j)
            {
                emit(grid(i, j),           grid(wrap(i + 1), j + 1), SpringKind::Shear);
                emit(grid(wrap(i + 1), j), grid(i, j + 1),           SpringKind::Shear);
            }
    }

    if (flags & ClothFlags::BENDING)
    {
        // vertical bending: open, same as cloth
        for (Index i = 0; i < W; ++i)
            for (Index j = 0; j < H - 2; ++j)
                emit(grid(i, j), grid(i, j + 2), SpringKind::Bend);

        // circumferential bending: wraps. Skipped below W=5, where stepping by 2 duplicates a
        // stretch edge (W=3) or double-emits the same pair (W=4); W>=5 gives distinct pairs.
        if (W >= 5)
            for (Index i = 0; i < W; ++i)
                for (Index j = 0; j < H; ++j)
                    emit(grid(i, j), grid(wrap(i + 2), j), SpringKind::Bend);
    }

    return obj;
}

// ----------------
//   TRI CLOTH
// ----------------

// Same layout as skirt() but the vertices/edges/triangles come from a generated meshgen::TriSurface: pinned = the vertices
// on `pinned_curves`, one Stretch spring per triangle edge (rest length from the 3D positions), mass proportional to the
// vertex's lumped area.
Object tri_cloth(
    const meshgen::TriSurface& s,
    const std::vector<int>& pinned_curves,
    Stiffness stiffness,
    Vec3      origin,
    Real      m_tot,
    TriEnergy energy,
    bool      flat_bend_rest,
    Real      seam_k)
{
    const Index nv = (Index)s.pos.size();
    std::vector<Vec3> pos(nv);
    std::vector<ParticleId> dof(nv, -1);
    Index n = 0; // |V_free|
    Real free_area = 0.0;
    for (Index v = 0; v < nv; ++v)
    {
        pos[v] = origin + s.pos[v];
        if (std::find(pinned_curves.begin(), pinned_curves.end(), s.curve[v]) == pinned_curves.end())
            { dof[v] = ParticleId(n++); free_area += s.vertex_area[v]; }
    }
    ASSERT(n > 0, "tri_cloth: the pinned curves pin all " << nv << " vertices");

    Object obj;
    obj.x.resize(3 * n);
    obj.mass.resize(3 * n);
    for (Index v = 0; v < nv; ++v)
        if (dof[v] >= 0)
        {
            obj.x.segment<3>(3 * dof[v]) = pos[v];
            obj.mass.segment<3>(3 * dof[v]).setConstant(m_tot * s.vertex_area[v] / free_area);
        }
    obj.v      = Velocities::Zero(3 * n);
    obj.prev_x = obj.x;

    SimMesh& mesh = obj.mesh;
    for (Index v = 0; v < nv; ++v)
    {
        SimMesh::Vertex mv;
        if (dof[v] >= 0) mv.dof = dof[v];
        else
        {
            mv.dof = ParticleId(-Index(mesh.pinned_rest.size()) - 1);
            mesh.pinned_rest.push_back(pos[v]);
        }
        mesh.vertices.push_back(mv);
    }
    mesh.edges     = s.edges;
    mesh.triangles = s.tris;
    mesh.seams     = s.seams;
    obj.energies.seams.k = seam_k;
    for (const auto& [a, b] : s.seams) obj.energies.seams.add_link(mesh.vertices[a].dof, mesh.vertices[b].dof);

    if (energy == TriEnergy::ARAP)
    {
        for (const auto& t : s.tris)
        {
            const ParticleId d[3] = { mesh.vertices[t[0]].dof, mesh.vertices[t[1]].dof, mesh.vertices[t[2]].dof };
            const Vec3       p[3] = { pos[t[0]], pos[t[1]], pos[t[2]] };
            obj.energies.arap.add_triangle(d, p, SpringKind::Stretch, stiffness.stretch);
        }

        // Hinge per interior edge: the two triangles sharing it, tips = their third vertices.
        std::map<std::pair<Index, Index>, std::vector<Index>> tips;
        for (const auto& t : s.tris)
            for (int a = 0; a < 3; ++a)
            {
                const Index u = t[a], v = t[(a + 1) % 3];
                tips[{ std::min(u, v), std::max(u, v) }].push_back(t[(a + 2) % 3]);
            }
        for (const auto& [edge, tip] : tips)
        {
            if (tip.size() != 2) continue; // boundary edge
            const Index vi[4] = { edge.first, edge.second, tip[0], tip[1] };
            const ParticleId d[4] = { mesh.vertices[vi[0]].dof, mesh.vertices[vi[1]].dof, mesh.vertices[vi[2]].dof, mesh.vertices[vi[3]].dof };
            const Vec3       p[4] = { pos[vi[0]], pos[vi[1]], pos[vi[2]], pos[vi[3]] };
            obj.energies.bending.add_hinge(d, p, SpringKind::Bend, stiffness.bend, flat_bend_rest);
        }
        return obj;
    }

    for (const auto& e : s.edges)
        obj.energies.springs.add_edge(mesh.vertices[e.first].dof, mesh.vertices[e.second].dof,
                                      pos[e.first], pos[e.second], SpringKind::Stretch, stiffness.stretch);
    return obj;
}

// ----------------
//   OBJ OUTPUT
// ----------------

void write_obj(const std::string& path, const Object& obj)
{
    const SimMesh& mesh = obj.mesh;

    std::ofstream out(path);
    ASSERT(out.is_open(), "could not open " << path);
    out << std::fixed << std::setprecision(6);

    for (Index vi = 0; vi < Index(mesh.vertices.size()); ++vi)
    {
        const Vec3 p = mesh.position(obj, vi);
        out << "v " << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    }
    for (const auto& e : mesh.edges)
        out << "l " << e.first + 1 << ' ' << e.second + 1 << '\n';
}

void write_obj_frame(const Object& obj, int step, const std::string& prefix = "frame") 
{
    std::ostringstream name;
    name << prefix << "_" << std::setfill('0') << std::setw(6) << step << ".obj";
    write_obj((fs::path(ANIM_DIR) / name.str()).string(), obj);
}

// ----------------
//      SOLVER
// ----------------

// Adds every elastic energy's force contribution to `b` in place, scaled by `scale`.
// No allocation: `b` is the caller's buffer, sized to obj.num_dofs().
void add_elastic_forces(const Object& obj, const RealVecX& x, RealVecX& b, Real scale = 1.0)
{
    obj.energies.for_each([&](const auto& e) { e.add_forces(x, obj.mesh.pinned_rest, b, scale); });
}

// ----------------
// VELOCITY SOLVER
// ----------------

void construct_velocity_lhs(Object& obj, Real dt)
{
    const Index n3 = obj.num_dofs();
    const Real  h2 = dt * dt;

    size_t n_elastic = 0;
    obj.energies.for_each([&](const auto& e) { n_elastic += e.triplet_count(); });

    std::vector<Triplet> triplets_L, triplets_C;
    triplets_L.reserve(n3 + n_elastic);
    triplets_C.reserve(     n_elastic);

    // M (mass diagonal, not divided by h²)
    for (Index i = 0; i < n3; ++i)
        triplets_L.emplace_back(i, i, obj.mass(i));

    // h² · K  (elastic contribution)
    obj.energies.for_each([&](const auto& e)
    {
        e.add_stiffness_triplets(triplets_L, h2);
        e.add_stiffness_triplets(triplets_C, h2);
    });

    obj.L.resize(n3, n3);
    obj.L.setFromTriplets(triplets_L.begin(), triplets_L.end());

    obj.C.resize(n3, n3);
    obj.C.setFromTriplets(triplets_C.begin(), triplets_C.end());

    // Laplacian damping: D = alpha*M + beta*K with K = C/h^2, so
    //   C_d = C + h*D = (1 + beta/h)*C + h*alpha*M,   L_d = M + C_d.
    // L and C above stay undamped (they build b_tilde and its derivative); only the solve and the
    // free force f = b_tilde - C_d*v use the damped pair. alpha = beta = 0 gives C_d == C, L_d == L.
    std::vector<Triplet> triplets_M;
    triplets_M.reserve(n3);
    for (Index i = 0; i < n3; ++i)
        triplets_M.emplace_back(i, i, obj.mass(i));
    SparseMat M_sp(n3, n3);
    M_sp.setFromTriplets(triplets_M.begin(), triplets_M.end());

    obj.C_d = (1.0 + obj.damp_beta / dt) * obj.C + (dt * obj.damp_alpha) * M_sp;
    const SparseMat L_d = M_sp + obj.C_d;

    obj.solver = std::make_unique<Cholesky>();
    obj.solver->compute(L_d);
    ASSERT(obj.solver->info() == Eigen::Success, "Cholesky factorization of velocity LHS failed");
}

RealVecX construct_velocity_rhs(
    const Object&    obj,
    const RealVecX&  b_inertia,
    const RealVecX&  x_k,
    const Positions& x_minus,
    Real             h)
{
    const Real h2 = h * h;
    RealVecX   b  = b_inertia;
    add_elastic_forces(obj, x_k, b, h2);
    return (1.0 / h) * (b - obj.L * x_minus);
}

// ----------------
//    BACKWARD
// ----------------

void precompute_constraints_local_derivative(Object& obj, const Positions& x)
{
    obj.energies.for_each([&](auto& e) { e.precompute_local_derivative(x, obj.mesh.pinned_rest); });
}

// Applies the assembled local elastic Jacobian ΔA (Σ_i Gᵢᵀ Γᵢ Gᵢ, every energy) to a vector — the same
// pattern used by both the position- and velocity-space adjoint RHS.
RealVecX apply_spring_jacobian(const Object& obj, const RealVecX& v)
{
    RealVecX out = RealVecX::Zero(v.rows());
    obj.energies.for_each([&](const auto& e) { e.add_jacobian_product(v, out); });
    return out;
}

void precompute_contacts_local_derivative(
    Contacts&          contacts,
    const Object&      obj,
    const Positions&   x_plus,
    const Velocities&  v_plus,
    const Positions&   x_minus,
    const Velocities&  v_minus,
    const Vec3&        gravity,
    Real               h)
{
    RealVecX x_tilde = x_minus + h * v_minus;
    const Vec3 dg    = h * h * gravity;
    for (Index i = 0; i < obj.num_particles(); ++i)
        x_tilde.segment<3>(3*i) += dg;

    const RealVecX b_inertia = obj.mass.cwiseProduct(x_tilde);
    const RealVecX b_tilde   = construct_velocity_rhs(obj, b_inertia, x_plus, x_minus, h);
    const RealVecX f         = b_tilde - obj.C_d * v_plus; // damped free force (C_d == C when undamped)

    for (Contact& c : contacts)
    {
        const Vec3 f_i = f.segment<3>(3 * c.particle);
        const Real m_i = obj.mass(3 * c.particle);
        const Real d_n = (f_i - m_i * c.v_c).dot(c.normal) + m_i * c.gap / h; // same as update_contact_force
        c.active = (d_n < 0.0);
        c.d_n    = d_n;
    }
}

struct ContactSplit
{
    RealVecX z_perp; // (I-P) z : normal component removed at active contacts
    RealVecX Pz;     // P z     : only the normal component, at active contacts
};

// Splits a vector into its active-contact-normal and perpendicular parts:
// v = (I-P) v + P v, where P = block-diag(n nᵀ) over active contacts.
ContactSplit split_by_contact(const Contacts& contacts, const RealVecX& v)
{
    ContactSplit s{ v, RealVecX::Zero(v.rows()) };
    for (const Contact& c : contacts)
    {
        if (!c.active) continue;
        const Vec3 vi = v.segment<3>(3 * c.particle);
        const Vec3 pn = c.normal * c.normal.dot(vi);
        s.Pz.segment<3>(3 * c.particle)     = pn;
        s.z_perp.segment<3>(3 * c.particle) = vi - pn;
    }
    return s;
}

RealVecX construct_backward_contact_rhs(
    const Object&   obj,
    const Contacts& contacts,
    const RealVecX& z,
    const RealVecX& dloss_dv,
    const RealVecX& dloss_dv_t,
    Real            h)
{

    const Real h2 = h * h;
    const ContactSplit split = split_by_contact(contacts, z);

    RealVecX b = h2 * apply_spring_jacobian(obj, split.z_perp);
    if (!contacts.empty())
        b += obj.C_d * split.Pz; // damped: the adjoint's P-coupling uses C_d (C_d == C when undamped)

    return b + dloss_dv + dloss_dv_t;
}

RealVecX compute_adjoint_vector_contact(
    Object&           obj,
    Contacts&         contacts,
    const Positions&  x_plus,
    const Velocities& v_plus,
    const Positions&  x_minus,
    const Velocities& v_minus,
    const RealVecX&   dloss_dv,
    const RealVecX&   dloss_dv_t,
    const Vec3&       gravity,
    Real              h,
    int               n_iters_adjoint,
    bool              verbose,
    Real&             residual_out)
{
    precompute_constraints_local_derivative(obj, x_plus);
    precompute_contacts_local_derivative(contacts, obj, x_plus, v_plus, x_minus, v_minus, gravity, h);

    RealVecX z = RealVecX::Zero(obj.num_dofs());
    Anderson aa(anderson_m, z.size()); // linear map, active set fixed for the step: AA acts like GMRES
    Real rel_residual = 0.0;
    for (int k = 0; k < n_iters_adjoint; ++k)
    {
        const RealVecX b_back = construct_backward_contact_rhs(obj, contacts, z, dloss_dv, dloss_dv_t, h);
        RealVecX       z_new  = obj.solver->solve(b_back);
        rel_residual = (z_new - z).norm() / std::max(z_new.norm(), Real(1e-12));
        if (k + 1 < n_iters_adjoint) aa.apply(z, z_new); // last iteration plain, as in the forward
        z = z_new;
    }
    residual_out = rel_residual;
    if (verbose && n_iters_adjoint > 0 && rel_residual > kConvergenceTol)
    {
        std::cout << "  iter " << n_iters_adjoint << " rel_delta = " << rel_residual << "\n";
    }

    return z;
}

// dphi/dk per SpringKind for one step: each spring's term lands in its own kind's slot, so the
// unified-stiffness gradient is just the sum of the three.
Stiffness compute_gradient_stiffness_contact(
    const Object&     obj,
    const Contacts&   contacts,
    const RealVecX&   z,
    const Positions&  x_minus,
    const Velocities& v_plus,
    Real              h)
{
    const RealVecX z_perp = split_by_contact(contacts, z).z_perp;
    // dC_d/dk = (h^2 + h*beta) * sum_i G_i^T G_i: the beta part is the damping force's sensitivity to k
    // (beta = 0, undamped: h^2 as before). b_tilde itself is undamped, so the p* - e^- term is unchanged.
    const Real     h2     = h * h + h * obj.damp_beta;

    Stiffness grad;
    obj.energies.for_each([&](const auto& e)
    { e.add_param_gradient(z_perp, x_minus, v_plus, obj.mesh.pinned_rest, h, h2, grad); });
    return grad;
}

// ----------------
//       PD
// ----------------

void init_pd_velocity(Object& obj, Real dt)
{
    construct_velocity_lhs(obj, dt);
}

// Called after each step/iteration of a watchable loop (pd_contact, backward_pd_contact,
// fd_check_contact_stiffness). Returning false aborts the loop early (e.g. window closed).
using StepCallback = std::function<bool(int step, int n_steps)>;

// Convergence measures (StepResidual) of the last iteration, whose velocity change is `dv` and result `v`;
// `prev_delta` is the previous iteration's ||dv|| (< 0 if there was none).
static StepResidual make_step_residual(const RealVecX& dv, const RealVecX& v, Real prev_delta, Real h, const Vec3& gravity)
{
    const Index n     = v.size() / 3;
    const Real  delta = dv.norm();
    Real vmax = 0.0;
    for (Index i = 0; i < n; ++i)
        vmax = std::max(vmax, Vec3(dv.segment<3>(3 * i)).norm());

    StepResidual r;
    r.rel  = delta / std::max(v.norm() + std::sqrt((Real)n) * h * gravity.norm(), Real(1e-12));
    r.vmax = vmax;
    r.rho  = prev_delta > 0.0 ? delta / prev_delta : std::numeric_limits<Real>::quiet_NaN();
    r.est_rel = r.rel;
    if (!std::isnan(r.rho))
    {
        const Real rho = std::min(r.rho, kMaxContraction);
        r.est_rel = r.rel * rho / (1.0 - rho);
    }
    return r;
}

void pd_contact(Object& obj, Real dt, const Vec3& gravity, int n_iters, int n_steps, int frame_substeps, Tape& tape, const std::string& prefix, bool export_obj = true, bool verbose = false, const StepCallback& on_step = nullptr, Real unresolved_penetration_threshold = 0.0, bool frozen = false, int first_step = 0)
{
    // `frozen`: every collider (animated, skinned, analytic) and the waist pins stay at their frame-0 pose with zero
    // velocity for the whole run - used to settle the cloth into a rest pose (settle_cloth). The track/FPS
    // preconditions don't apply since no pose is ever advanced.
    // Preconditions, checked once up front so the body below stays assertion-free.
    auto validate = [&]()
    {
        const bool has_skinned = std::any_of(colliders.begin(), colliders.end(),
                                              [](const Collider& c) { return c.type == ColliderType::Skinned; });
        if (collider_animations.empty() && !has_skinned) return;
        ASSERT(std::abs(dt * frame_substeps - 1.0 / kColliderAnimFPS) < 1e-9,
               "animated colliders need FPS = " << kColliderAnimFPS << " (one sim frame per track frame; "
               "use frame_substeps to refine) — got dt*frame_substeps=" << dt * frame_substeps);
        for (const ColliderAnimation& anim : collider_animations)
            ASSERT(anim.steps_per_frame == frame_substeps,
                   "collider animation '" << anim.name << "' has steps_per_frame=" << anim.steps_per_frame
                   << " but frame_substeps=" << frame_substeps);
        for (const Collider& c : colliders)
            if (c.type == ColliderType::Skinned)
                ASSERT(skinned_meshes[c.skinned_id].steps_per_frame == frame_substeps,
                       "skinned body '" << skinned_meshes[c.skinned_id].name << "' has steps_per_frame="
                       << skinned_meshes[c.skinned_id].steps_per_frame << " but frame_substeps=" << frame_substeps);
    };
    if (!frozen) validate();

    tape.clear();
    tape.first_step = first_step;
    tape.record(obj);
    if (export_obj) write_obj_frame(obj, 0, prefix);

    // Poses animated colliders (+ waist attachment) from the imported track at physics state `step`.
    // Called for state 0 up front, then once per step (end-of-step state, matching x_tilde) so contact
    // detection/velocity basis see the right pose.
    auto stamp_animated_colliders = [&](int step) { restamp_animated_colliders(colliders, collider_animations, obj, step); };

    stamp_animated_colliders(first_step);

    size_t total_added_in_iters = 0; // contacts found mid-iteration (merge_detected_contacts), whole run
    TIMER_START("loop");
    for (int step = 0; step < n_steps; ++step)
    {
        if (!frozen) stamp_animated_colliders(first_step + step + 1);

        RealVecX x_tilde = obj.x + dt * obj.v;
        const Vec3 dg = (dt * dt) * gravity;
        for (Index i = 0; i < obj.num_particles(); ++i)
            x_tilde.segment<3>(3*i) += dg;

        const Real     contact_time = frozen ? 0.0 : (first_step + step + 1) * dt;
        const RealVecX b_inertia    = obj.mass.cwiseProduct(x_tilde);
        TIMER_START("detect");
        Contacts       contacts     = detect_contacts(obj, x_tilde, contact_time, speculative_contact ? -contact_margin : 0.0);
        TIMER_END("detect");

        obj.prev_x = obj.x;
        obj.x      = x_tilde;

        // Instantaneous velocity of an animated collider is a precomputed, static property of its
        // track (see ColliderAnimation::velocity_basis) — evaluated at the same end-of-step state the
        // collider was stamped at. Evaluated at the cached closest surface point.
        auto contact_velocity = [&](const Contact& c) -> Vec3
        {
            if (frozen) return Vec3::Zero();
            const Collider& collider = colliders[c.collider_id];
            // Skinned body: the barycentric blend of the contact triangle's vertex velocities.
            if (collider.type == ColliderType::Skinned)
                return skinned_surface_velocity(collider.skinned_id, c.triangle_id, c.bary);
            if (!collider.animated)
                return collider_point_velocity(collider, c.surface_point, contact_time);
            return animated_collider_point_velocity(collider_animations[collider.anim_id], first_step + step + 1, c.surface_point);
        };

        // v_c is frozen at the surface point, stamped once per contact.
        // Speculative contact instead re-evaluates the whole contact geometry (normal, inv_r, axis, surface point) and
        // the gap phi(x-) at x-, against the collider's end-of-step pose (the pose detection used), and leaves v_c = 0:
        // the collider's motion is already in the gap. The constraint phi(x-) + h v+.n >= 0 is then the first-order
        // expansion of phi(x+) >= 0 at x-, and nothing in it depends on x~ (x~ only picks which particles are contacts).
        auto stamp_surface_velocities = [&](size_t begin, size_t end)
        {
            if (speculative_contact)
            {
                for (size_t j = begin; j < end; ++j)
                {
                    Contact& c = contacts[j];
                    const Collider& collider = colliders[c.collider_id];
                    const Vec3 pos = obj.prev_x.segment<3>(3 * c.particle);
                    Contact at_x_minus;
                    const Real gap = contact_geometry(collider, c.collider_id, collider_pose_at(collider, contact_time), c.particle, pos, at_x_minus);
                    c = at_x_minus;
                    c.gap = gap;
                    // grad phi = (x - closest point) / phi off the surface (exact for every shape); the normal on it
                    c.gap_grad = std::abs(gap) > 1e-12 ? Vec3((pos - c.surface_point) / gap) : c.normal;
                }
                return;
            }
            for (size_t j = begin; j < end; ++j)
                contacts[j].v_c = contact_velocity(contacts[j]);
        };
        stamp_surface_velocities(0, contacts.size());

        // Iterations between rechecks: Every = 1, NTimes = recheck_end / n (n checks total counting the
        // pre-solve one), Once = never (0). Rechecks stop halfway so the final contact set still gets
        // n_iters/2 iterations to converge (the backward pass assumes the fixed point). Speculative contact is
        // always Once: its margin replaces the rechecks and keeps the contact set fixed, as the backward pass assumes.
        const int recheck_end    = n_iters / 2;
        const int recheck_period = speculative_contact                                ? 0
                                 : contact_recheck_mode == ContactRecheckMode::Every  ? 1
                                 : contact_recheck_mode == ContactRecheckMode::NTimes ? std::max(1, recheck_end / std::max(2, contact_recheck_count))
                                 : 0;

        Anderson aa(frozen ? 0 : anderson_m, obj.v.size()); // never while settling
        int  contacts_added_in_iters = 0;
        Real prev_delta = -1.0; // ||dv|| of the previous iteration, for the contraction ratio
        for (int k = 0; k < n_iters; ++k)
        {
            // Re-test the current iterate and grow the contact set (merge_detected_contacts). Runs
            // before the solve so the last iteration — the one the tape records — sees the final set.
            if (k > 0 && k < recheck_end && recheck_period > 0 && k % recheck_period == 0)
            {
                TIMER_START("detect");
                const int added = merge_detected_contacts(contacts, obj, obj.x, x_tilde, contact_time);
                stamp_surface_velocities(contacts.size() - added, contacts.size());
                contacts_added_in_iters += added;
                if (added > 0) aa.reset();
                TIMER_END("detect");
            }

            TIMER_START("elastic");
            const RealVecX b_tilde = construct_velocity_rhs(obj, b_inertia, obj.x, obj.prev_x, dt);
            TIMER_END("elastic");

            TIMER_START("contact");
            const RealVecX f = b_tilde - obj.C_d * obj.v; // damped Jacobi split (C_d == C when undamped)
            RealVecX g = b_tilde;
            for (Contact& c : contacts)
            {
                const Vec3 f_i = f.segment<3>(3 * c.particle);
                const Real m_i = obj.mass(3 * c.particle);
                g.segment<3>(3 * c.particle) += update_contact_force(f_i, m_i, c.v_c, c.normal, c.gap / dt);
            }
            TIMER_END("contact");

            TIMER_START("global");
            RealVecX v_hat = obj.solver->solve(g);
            if (k >= n_iters - 2) // only the last two deltas are needed (contraction ratio)
            {
                const RealVecX dv = v_hat - obj.v;
                if (k == n_iters - 1)
                {
                    const StepResidual res = make_step_residual(dv, v_hat, prev_delta, dt, gravity);
                    tape.forward_residual.push_back(res);
                    if (verbose && res.est_rel > kConvergenceTol)
                        std::cout << "  step " << step << " iter " << k+1 << " est_error = " << res.est_rel << "\n";
                }
                prev_delta = dv.norm();
            }

            // k = 0 evaluates G at x~ (not x- + h v-), another map; the last iteration stays plain so v+ solves the global system
            if (k > 0 && k + 1 < n_iters) aa.apply(obj.v, v_hat);
            obj.v = v_hat;
            obj.x = obj.prev_x + dt * obj.v;
            TIMER_END("global");
        }

        tape.record_contacts(contacts); // after the solve: carries the converged per-contact v_c
        // Re-detect against the converged x: how many particles are still penetrating a collider
        // after this step's iterations, as opposed to `contacts` above (detected pre-solve, against
        // x_tilde). A small positive threshold treats negligible residual penetration as resolved.
        tape.unresolved_contacts.push_back((int)detect_contacts(obj, obj.x, contact_time, unresolved_penetration_threshold).size());
        tape.record(obj);
        if (export_obj && step % frame_substeps == 0) write_obj_frame(obj, (step / frame_substeps) + 1, prefix);
        bool keep_going = true;
        if (on_step)
        {
            TIMER_START("viewer_callback"); // live-window drawing: excluded from the total
            keep_going = on_step(step, n_steps);
            TIMER_END("viewer_callback");
        }
        if (!keep_going) break;
        total_added_in_iters += contacts_added_in_iters;
        if (step % 10 == 0)
            std::cout << "step " << step << "/" << n_steps
                       << "  contacts=" << contacts.size()
                       << " (added in iters=" << contacts_added_in_iters << ")"
                       << " unresolved=" << tape.unresolved_contacts.back() << "\n";
    }
    TIMER_END("loop");

    size_t total_contacts = 0;
    for (const Contacts& c : tape.contacts) total_contacts += c.size();
    size_t total_unresolved = 0;
    for (int u : tape.unresolved_contacts) total_unresolved += u;
    const Real unresolved_pct = total_contacts > 0
        ? 100.0 * (Real)total_unresolved / (Real)total_contacts
        : 0.0;
    std::cout << "[" << prefix << "] total contacts=" << total_contacts
               << " (added in iters=" << total_added_in_iters << ")"
               << " total unresolved=" << total_unresolved
               << " (" << unresolved_pct << "%)\n";
}

// ----------------
//  BACKWARD: CONTACT CORRECTIONS
// ----------------
// An active contact adds  r_i = -n n^T (f_i - m_i v_c)  to the velocity-solve RHS, where the normal n and the
// contact-point velocity v_c are both evaluated at the predicted position x~ = x- + h v- + h^2 g. So the adjoint
// needs two corrections, one per way the RHS moves with x~. Each function returns its contribution to dphi/dx~
// (sign included); the caller adds it to dphi_dx and, times h (dx~/dv- = h I), to dphi_dv. (Speculative contact:
// the geometry is evaluated at x- instead, so it goes to dphi_dx only.)
// Common notation: z_n = n . z_i is the normal part of the adjoint, the part the contact projection removes.

// A collider's angular velocity and surface radius at the current backward step.
struct RotationInfo { bool enabled; Vec3 omega_vec; Real R; };

// (1) Normal correction, n = n(x~): dn/dx~ is the collider's shape operator (curvature), inv_r (I - n n^T) for a
// sphere, also projected off the axis for a cylinder / capsule body, and 0 for planes and meshes (inv_r = 0).
//   dphi/dx~ = -inv_r P_axis [ d_n z_perp + z_n m (v+ - v_c) ]
static Vec3 normal_correction(const Contact& c, const Vec3& z_perp_i, const Vec3& v_plus_i, Real z_n, Real m_i)
{
    // Q_i d_i = m_i Q_i (v+ - v_c); the old constraint makes n.(v+ - v_c) = 0 so Q drops out, the speculative one
    // (n.v+ = -gap/h) does not (only projected there, so the default path stays bit-identical).
    Vec3 w = v_plus_i - c.v_c;
    if (speculative_contact) w -= c.normal * c.normal.dot(w);
    const Vec3 term = c.d_n * z_perp_i + z_n * m_i * w;
    return -c.inv_r * (term - c.axis * c.axis.dot(term));                 // axis = 0 unless cylinder / capsule body
}

// (2) Velocity correction, v_c = v_c(x~): for a rotating collider v_c = v + w x (q - pivot), with q the surface
// point, so only q depends on x~ (q = q(x~), J_s = dq/dx~):
//   dphi/dx~ = -m z_n J_s^T (w x n)
// with J_s = I - n n^T on a mesh face, e e^T on an edge, 0 on a vertex; (R/r)(I - n n^T) on a sphere; identity
// along the axis and (R/rho)(I - n n^T - a a^T) across it on a cylinder / capsule body. Zero if the collider
// doesn't rotate.
// Skinned body: the cached G = dv_c/dx~ (Contact::dvc_dx) plays the role of [w]_x J_s,
//   dphi/dx~ = +m z_n G^T n        (for a rigid motion G^T n = -J_s^T (w x n): the same term)
static Vec3 velocity_correction(const Contact& c, const Collider& collider, const RotationInfo& rot, Real z_n, Real m_i)
{
    if (collider.type == ColliderType::Skinned) return m_i * z_n * (c.dvc_dx.transpose() * c.normal);
    if (!rot.enabled) return Vec3::Zero();

    const Vec3 w_x_n = rot.omega_vec.cross(c.normal);
    Vec3       Jt_wn; // J_s^T (w x n)
    if (collider.type == ColliderType::Mesh)
    {
        Jt_wn = c.feature == ContactFeature::Face ? w_x_n // w x n is already perpendicular to n
              : c.feature == ContactFeature::Edge ? Vec3(c.edge_dir * c.edge_dir.dot(w_x_n))
                                                  : Vec3::Zero();
    }
    else if (c.axis.squaredNorm() > 0.0) // cylinder / capsule body
    {
        const Vec3 along  = c.axis * c.axis.dot(w_x_n);
        Jt_wn = along + (rot.R * c.inv_r) * (w_x_n - along);
    }
    else // sphere / capsule cap
    {
        Jt_wn = (rot.R * c.inv_r) * w_x_n;
    }
    return -m_i * z_n * Jt_wn;
}

struct BackwardGradContact
{
    RealVecX dphi_dv; // dphi/dv0 — gradient of loss w.r.t. initial velocity
    RealVecX dphi_dx; // dphi/dx0 — gradient of loss w.r.t. initial position
    Stiffness dphi_dk; // dphi/dk per SpringKind (sum = gradient w.r.t. a unified stiffness)
    Real     dphi_dalpha = 0.0; // dphi/d(alpha) — Laplacian damping, mass-proportional (0 unless damping on)
    Real     dphi_dbeta  = 0.0; // dphi/d(beta)  — Laplacian damping, K-proportional    (0 unless damping on)

    // Adjoint-solve convergence diagnostic (mirrors Tape::forward_residual), indexed in forward
    // chronological order even though computed in reverse.
    std::vector<Real> residual;
};

BackwardGradContact backward_pd_contact(
    Object&     obj,
    Tape&       tape,
    const Loss& loss,
    const Vec3& gravity,
    int         n_iters_adjoint,
    Real        h,
    const StepCallback& on_step = nullptr,
    bool        verbose = false)
{
    const int   n_steps = (int)tape.positions.size() - 1;
    const Index dofs    = obj.num_dofs();

    // Preconditions, checked once up front so the body below stays assertion-frees
    auto validate = [&]()
    {
        ASSERT((int)loss.dloss_dx.size() == n_steps + 1,
               "loss gradient size " << loss.dloss_dx.size()
               << " != tape size "   << tape.positions.size());
        ASSERT((int)tape.contacts.size() == n_steps,
               "contacts tape size " << tape.contacts.size()
               << " != n_steps "     << n_steps);
        for (const Collider& c : colliders)
        {
            const bool rotation_enabled = !c.animated && c.rotation_axis != RotationAxis::None && c.omega != 0.0;
            if (rotation_enabled)
                ASSERT(c.type == ColliderType::Sphere || c.type == ColliderType::Cylinder || c.type == ColliderType::Capsule,
                       "Rotation curvature correction is only implemented for Sphere, Cylinder, and Capsule colliders");
            if (c.animated)
                ASSERT(c.type == ColliderType::Sphere || c.type == ColliderType::Capsule || c.type == ColliderType::Mesh,
                       "Rotation curvature correction is only implemented for Sphere, Capsule and Mesh colliders");
        }
    };
    validate();

    std::vector<Real> residual(n_steps, 0.0);

    RealVecX dphi_dv = RealVecX::Zero(dofs); // a_t = dL/dv^+_t, seeded from later steps
    RealVecX dphi_dx = RealVecX::Zero(dofs); // b_t = dL/dx^+_t, seeded from later steps
    Stiffness dphi_dk;                       // dphi/dk per SpringKind, accumulated across all steps
    Real     dphi_dalpha = 0.0;              // dphi/d(alpha), accumulated across all steps
    Real     dphi_dbeta  = 0.0;              // dphi/d(beta),  accumulated across all steps

    // Per-collider rotation state for velocity_correction, indexed by Contact::collider_id.
    std::vector<RotationInfo> rot_info(colliders.size());
    for (int ci = 0; ci < (int)colliders.size(); ++ci)
    {
        const Collider& c = colliders[ci];
        if (c.animated) continue; // recomputed every step below — its rotation isn't constant
        const bool rotation_enabled = c.rotation_axis != RotationAxis::None && c.omega != 0.0;
        rot_info[ci] = {
            rotation_enabled,
            rotation_enabled ? Vec3(c.omega * rotation_axis_vector(c.rotation_axis)) : Vec3::Zero(),
            collider_surface_radius(c)
        };
    }

    for (int t = n_steps; t >= 1; --t)
    {
        // Restamp animated colliders + waist attachment to this step's state; shared with the
        // forward pass so collider pose and pinned positions can't drift out of sync.
        restamp_animated_colliders(colliders, collider_animations, obj, tape.first_step + t);

        // rot_info for animated colliders varies per bracket; look it up from the same precomputed
        // basis (anim, step=t) the forward pass used for this step's contact-point velocities.
        for (int ci = 0; ci < (int)colliders.size(); ++ci)
        {
            const Collider& c = colliders[ci];
            if (!c.animated || c.anim_id < 0 || c.anim_id >= (int)collider_animations.size()) continue;
            const Vec3 omega_vec = animated_collider_basis_at_step(collider_animations[c.anim_id], tape.first_step + t).omega_vec;
            rot_info[ci] = { true, omega_vec, collider_surface_radius(c) };
        }

        const Positions  x_plus_t   = Eigen::Map<const Positions>(tape.positions[t].data(),       dofs);
        const Velocities v_plus_t   = Eigen::Map<const Velocities>(tape.velocities[t].data(),     dofs);
        const Positions  x_plus_tm1 = Eigen::Map<const Positions>(tape.positions[t - 1].data(),   dofs);
        const Velocities v_plus_tm1 = Eigen::Map<const Velocities>(tape.velocities[t - 1].data(), dofs);
        Contacts         contacts_t = tape.contacts[t - 1];

        const RealVecX b_t = dphi_dx + loss.dloss_dx[t];

        Real step_residual = 0.0;
        const RealVecX z = compute_adjoint_vector_contact(
            obj, contacts_t, x_plus_t, v_plus_t, x_plus_tm1, v_plus_tm1,
            dphi_dv, h * b_t, gravity, h, n_iters_adjoint, verbose, step_residual);
        residual[t - 1] = step_residual;

        dphi_dk += compute_gradient_stiffness_contact(obj, contacts_t, z, x_plus_tm1, v_plus_t, h);

        const ContactSplit split   = split_by_contact(contacts_t, z);
        const RealVecX&     z_perp = split.z_perp; // (I-P) z

        // Damping parameters enter only through C_d (b_tilde is undamped): dC_d/dalpha = h M,
        // dC_d/dbeta = h K = C/h, so dphi/dalpha = -h z_perp^T M v+, dphi/dbeta = -(1/h) z_perp^T C v+.
        dphi_dalpha -= h * z_perp.dot(obj.mass.cwiseProduct(v_plus_t));
        dphi_dbeta  -= z_perp.dot(obj.C * v_plus_t) / h;

        dphi_dv = obj.mass.cwiseProduct(z_perp);
        dphi_dx = h * apply_spring_jacobian(obj, z_perp) - (obj.C * z_perp) / h + b_t;

        // Corrections for the contact normal and velocity depending on x~ (see above).
        for (const Contact& c : contacts_t)
        {
            if (!c.active) continue;
            const Index i   = c.particle;
            const Real  m_i = obj.mass(3 * i);
            const Real  z_n = c.normal.dot(z.segment<3>(3 * i));

            // Speculative contact: v_c = 0 (no velocity correction), the geometry is evaluated at x- (so the normal
            // correction goes to dphi/dx- only, nothing to dphi/dv-), and the gap term -m phi(x-)/h n adds
            // -(m/h) z_n grad phi to dphi/dx-.
            const Vec3 d = normal_correction(c, z_perp.segment<3>(3 * i), v_plus_t.segment<3>(3 * i), z_n, m_i)
                         + (speculative_contact ? Vec3::Zero()
                                                : velocity_correction(c, colliders[c.collider_id], rot_info[c.collider_id], z_n, m_i));

            dphi_dx.segment<3>(3 * i) += d;
            if (speculative_contact) dphi_dx.segment<3>(3 * i) -= (m_i / h) * z_n * c.gap_grad;
            else                     dphi_dv.segment<3>(3 * i) += h * d;
        }

        if (on_step && !on_step(t, n_steps)) break;
        if (t % 10 == 0) std::cout << "backward (contact) step " << t << "/" << n_steps << "\n";
    }

    dphi_dx += loss.dloss_dx[0];

    return { std::move(dphi_dv), std::move(dphi_dx), dphi_dk, dphi_dalpha, dphi_dbeta, std::move(residual) };
}

// ----------------
//    FD CHECK
// ----------------

enum class FDTarget { X0, V0 };

// FDCheckResult and FDCheckRunner are defined in diffpd_types.h (shared with the viewer).

FDCheckResult fd_check_contact_direction(
    const std::function<Object()>& build_obj,
    const Tape&     target_tape,
    Real            dt,
    const Vec3&     gravity,
    int             n_iters,
    int             n_steps,
    int             frame_substeps,
    int             sample_every,
    const RealVecX& direction,
    FDTarget        target,
    const RealVecX& analytic_grad,
    Real            eps = 1e-6)
{
    auto run_loss = [&](Real delta) -> Real
    {
        Object obj = build_obj();
        if (target == FDTarget::X0) obj.x += delta * direction;
        else                        obj.v += delta * direction;

        init_pd_velocity(obj, dt);
        Tape tape;
        pd_contact(obj, dt, gravity, n_iters, n_steps, frame_substeps, tape, "fd_check", /*export_obj=*/false);
        return Loss(tape, target_tape, sample_every).total;
    };

    const Real loss_plus  = run_loss(+eps);
    const Real loss_minus = run_loss(-eps);
    const Real fd         = (loss_plus - loss_minus) / (2.0 * eps);
    const Real analytic   = direction.dot(analytic_grad);
    const Real rel_err    = std::abs(fd - analytic)
        / std::max({ std::abs(fd), std::abs(analytic), Real(1e-12) });

    return { fd, analytic, rel_err };
}

RealVecX uniform_axis_direction(Index num_dofs, int axis)
{
    RealVecX d = RealVecX::Zero(num_dofs);
    for (Index i = axis; i < num_dofs; i += 3) d(i) = 1.0;
    return d;
}

void fd_check_contact_offsets(
    const std::function<Object()>& build_obj,
    const Tape&     target_tape,
    Real            dt,
    const Vec3&     gravity,
    int             n_iters,
    int             n_steps,
    int             frame_substeps,
    int             sample_every,
    Index           num_dofs,
    const RealVecX& dphi_dx0,
    const RealVecX& dphi_dv0,
    Real            eps = 1e-6)
{
    static const char* axis_name[3] = { "x", "y", "z" };
    std::cout << "FD check (uniform offset over all particles, eps=" << eps << "):\n";
    for (int axis = 0; axis < 3; ++axis)
    {
        const RealVecX dir = uniform_axis_direction(num_dofs, axis);

        const FDCheckResult rx = fd_check_contact_direction(
            build_obj, target_tape, dt, gravity, n_iters, n_steps, frame_substeps,
            sample_every, dir, FDTarget::X0, dphi_dx0, eps);
        const FDCheckResult rv = fd_check_contact_direction(
            build_obj, target_tape, dt, gravity, n_iters, n_steps, frame_substeps,
            sample_every, dir, FDTarget::V0, dphi_dv0, eps);

        std::cout << "  ." << axis_name[axis]
                   << "  dx0: fd=" << rx.fd << " analytic=" << rx.analytic << " rel_err=" << rx.rel_err
                   << "  |  dv0: fd=" << rv.fd << " analytic=" << rv.analytic << " rel_err=" << rv.rel_err
                   << "\n";
    }
}

FDCheckResult fd_check_contact_stiffness(
    const std::function<Object(Real)>& build_obj_k,
    const Tape&     target_tape,
    Real            k0,
    Real            dt,
    const Vec3&     gravity,
    int             n_iters,
    int             n_steps,
    int             frame_substeps,
    int             sample_every,
    Real            analytic_dphi_dk,
    Real            eps = 1e-6,
    const StepCallback& on_step = nullptr)
{
    // `pass` = 0 for +eps, 1 for -eps: on_step sees both runs as one loop of 2*n_steps steps, so
    // callers get continuous progress (step in [0, 2*n_steps)) across the pair.
    auto run_loss = [&](Real delta, int pass) -> Real
    {
        Object obj = build_obj_k(k0 + delta);
        init_pd_velocity(obj, dt);
        Tape tape;
        const StepCallback pass_on_step = on_step
            ? StepCallback([&](int step, int n) { return on_step(pass * n + step, 2 * n); })
            : nullptr;
        pd_contact(obj, dt, gravity, n_iters, n_steps, frame_substeps, tape, "fd_check", /*export_obj=*/false,
                   /*verbose=*/false, pass_on_step);
        // on_step may abort early, leaving a truncated tape; bail with a sentinel instead of
        // crashing Loss's tape-length assert.
        if ((int)tape.positions.size() != n_steps + 1) return 0.0;
        return Loss(tape, target_tape, sample_every).total;
    };

    const Real loss_plus  = run_loss(+eps, 0);
    const Real loss_minus = run_loss(-eps, 1);
    const Real fd         = (loss_plus - loss_minus) / (2.0 * eps);
    const Real rel_err    = std::abs(fd - analytic_dphi_dk)
        / std::max({ std::abs(fd), std::abs(analytic_dphi_dk), Real(1e-12) });

    return { fd, analytic_dphi_dk, rel_err };
}

// ----------------
//      MAIN
// ----------------

struct SimRunResult { Object obj; Tape tape; };

// Attaches obj's pinned vertices to the rigid hip track (anim_id >= 0) or to the pelvis of a skinned body
// (skinned_id >= 0), baking from their current positions, then poses them at state 0. No-op if neither.
static void attach_waist(Object& obj, int anim_id, int skinned_id, const std::vector<ColliderAnimation>& anims)
{
    if (anim_id >= 0)
        bake_waist_attachment(obj, anim_id, anims);
    else if (skinned_id >= 0)
        bake_waist_attachment_skinned(obj, skinned_id, find_bone(skinned_meshes[skinned_id], kWaistAttachmentBoneName));
    else
        return;
    update_waist_attachment(obj, anims, 0);
}

// Builds one cloth and forward-simulates it with a live-viewer callback — shared by the target and
// guess runs in main(), which differ only in stiffness/label/prefix/verbosity. Only the trajectory
// being computed is shown live; the other one (e.g. the target, while the guess is computing) is not
// overlaid — see viewer_interactive_playback() for the target-vs-guess side-by-side comparison after
// both runs finish.
SimRunResult run_forward_simulation(
    AppConfig& cfg, const Stiffness& stiffness, const Vec3& gravity,
    Real dt, int n_iters, int n_steps, int frame_substeps,
    int waist_attach_anim_id, int waist_attach_skinned_id, const std::vector<ColliderAnimation>& collider_animations,
    const std::string& prefix, const std::string& label, bool verbose,
    bool& aborted)
{
    SimRunResult result;
    result.obj = build_run_cloth(cfg, stiffness);
    init_pd_velocity(result.obj, dt);
    attach_waist(result.obj, waist_attach_anim_id, waist_attach_skinned_id, collider_animations);

    // Contact has no push-out: a cloth that starts inside the body stays stuck there.
    {
        int free_inside = 0, pinned_inside = 0;
        count_cloth_inside_skinned(result.obj, free_inside, pinned_inside);
        if (free_inside + pinned_inside > 0)
            WARNING(label << ": " << free_inside << " free and " << pinned_inside
                    << " pinned cloth particles start inside a skinned body (they will stay stuck); move the cloth in the setup screen");
    }

    Object& obj  = result.obj;
    Tape&   tape = result.tape;
    const StepCallback on_step = [&](int step, int n) -> bool
    {
        // Polled every physics step so a window-close is noticed within one step, not later.
        if (!viewer_poll_close()) { aborted = true; return false; }
        if (step % frame_substeps != 0) return true;
        std::ostringstream oss;
        oss << label << "   step " << step << "/" << n;
        viewer_set_scene(obj.mesh, nullptr, nullptr, &tape.positions.back(), &tape.contacts.back(),
                          colliders, (step + 1) * dt, oss.str());
        if (!viewer_render_frame()) { aborted = true; return false; }
        return true;
    };
    pd_contact(obj, dt, gravity, n_iters, n_steps, frame_substeps, tape, prefix, /*export_obj=*/false, verbose, on_step, cfg.unresolved_contact_threshold);
    return result;
}

struct WorldSetup { int waist_attach_anim_id = -1, waist_attach_skinned_id = -1; };

// Loads everything a run (or a settle) collides with from cfg into the file-scope globals: the manual colliders,
// the animation track + its velocity basis, the skinned bodies, and which of them the waist attaches to.
static WorldSetup prepare_world(const AppConfig& cfg)
{
    // the collider list is UI-managed (cfg.colliders); animated colliders are appended separately and aren't
    // editable from the config screen.
    colliders                       = cfg.colliders;
    contact_recheck_mode            = cfg.contact_recheck_mode;
    contact_recheck_count           = cfg.contact_recheck_count;
    speculative_contact             = cfg.speculative_contact;
    contact_margin                  = cfg.contact_margin;
    anderson_m                      = cfg.anderson_m;
    collider_animations.clear();
    if (cfg.animation_file.empty())
        clear_animation_assets(); // else a stale skinned body from the last preview/run still pins the waist (without colliding)
    else
        collider_animations = load_collider_animation(animation_path(cfg.animation_file), colliders);
    for (ColliderAnimation& anim : collider_animations)
    {
        precompute_velocity_basis(anim);
        anim.steps_per_frame = cfg.frame_substeps; // sim runs at kColliderAnimFPS (asserted in pd_contact)
    }
    for (SkinnedMesh& body : skinned_meshes)
        body.steps_per_frame = cfg.frame_substeps;

    // Resolve the hardcoded collider/bone names once per run: the rigid hip track, else the pelvis of a
    // skinned body. Missing warns and disables.
    WorldSetup world;
    if (cfg.waist_attach_enabled)
    {
        for (int i = 0; i < (int)collider_animations.size(); ++i)
            if (collider_animations[i].name == kWaistAttachmentColliderName) { world.waist_attach_anim_id = i; break; }
        if (world.waist_attach_anim_id < 0)
            for (int i = 0; i < (int)skinned_meshes.size(); ++i)
                if (find_bone(skinned_meshes[i], kWaistAttachmentBoneName) >= 0) { world.waist_attach_skinned_id = i; break; }
        if (world.waist_attach_anim_id < 0 && world.waist_attach_skinned_id < 0)
            WARNING("Waist Attachment enabled but no '" << kWaistAttachmentColliderName << "' collider animation or '"
                    << kWaistAttachmentBoneName << "' skinned-body bone was loaded; disabling for this run.");
    }
    return world;
}

// Waist snap of a skirt's pins onto skinned body `skinned_id` (state 0): from each pin a segment runs horizontally to the
// axis of the top circle (`center`, at the pin's height) and the pin moves to its first hit on the body, `offset` back
// toward where it was. A pin whose segment misses (including a pin already inside the body) warns and stays.
// Brute force over the triangles. Returns the new pinned positions (indexed like mesh.pinned_rest).
static std::vector<Vec3> snap_pins_to_skinned_body(const Object& obj, int skinned_id, const Vec3& center, Real offset)
{
    stamp_skinned_body(skinned_id, 0);
    const std::vector<Vec3>& V = skinned_states[skinned_id].x;
    const auto& T = skinned_meshes[skinned_id].rest.triangles;

    std::vector<Vec3> pins = obj.mesh.pinned_rest;
    int snapped = 0;
    Real max_move = 0.0;
    for (size_t k = 0; k < pins.size(); ++k)
    {
        const Vec3 p = pins[k];
        const Vec3 d = Vec3(center.x(), p.y(), center.z()) - p;
        Real t_hit = 2.0; // segment parameter in [0, 1]
        for (const auto& tri : T)  // Moller-Trumbore, double sided
        {
            const Vec3 e1 = V[tri[1]] - V[tri[0]], e2 = V[tri[2]] - V[tri[0]];
            const Vec3 h = d.cross(e2);
            const Real det = e1.dot(h);
            if (std::abs(det) < 1e-14) continue;
            const Vec3 s = p - V[tri[0]];
            const Real u = s.dot(h) / det;
            if (u < 0.0 || u > 1.0) continue;
            const Vec3 q = s.cross(e1);
            const Real v = d.dot(q) / det;
            if (v < 0.0 || u + v > 1.0) continue;
            const Real t = e2.dot(q) / det;
            if (t >= 0.0 && t <= 1.0) t_hit = std::min(t_hit, t);
        }
        if (t_hit > 1.0) { WARNING("waist snap: pin " << k << " hit no body surface, left in place"); continue; }
        pins[k] = p + t_hit * d - offset * d.normalized();
        max_move = std::max(max_move, (pins[k] - p).norm());
        ++snapped;
    }
    std::cout << "[settle] waist snap: " << snapped << "/" << pins.size() << " pins moved, max move " << max_move * 1000.0 << " mm\n";
    return pins;
}

// Moves each free particle inside a collider (poses as currently stamped, time 0) to its closest surface point plus
// `margin` along the normal, so a run starts collision free (contact has no push-out). Retries up to `max_passes`
// times, doubling the margin each pass, so a particle bouncing between two colliders eventually escapes.
// shortcut: the margin reaches ~0.8 m on the last pass, cap it if stray particles ever show up far from the cloth.
// Returns how many particles still penetrate; `moved` = how many were inside at the start. Pinned vertices are skipped.
static int project_out_of_colliders(Object& obj, int& moved, Real margin = 1e-4, int max_passes = 14)
{
    moved = 0;
    for (int pass = 0; pass < max_passes; ++pass)
    {
        const Contacts penetrating = detect_contacts(obj, obj.x, 0.0);
        if (pass == 0) moved = (int)penetrating.size();
        if (penetrating.empty()) return 0;
        for (const Contact& c : penetrating)
            obj.x.segment<3>(3 * c.particle) = c.surface_point + (margin * (1 << pass)) * c.normal;
    }
    return (int)detect_contacts(obj, obj.x, 0.0).size();
}

// "Settle cloth" button: simulates the cloth with every collider frozen at frame 0 (pd_contact `frozen`) and stores the
// result in cfg.settled as the start pose of later runs (see build_run_cloth), for target and guess alike. It is settled
// from the procedural mesh with cfg.settle_k() at a fixed dt = 1/60 s. Stops at cfg.settle_seconds or when
// the fastest particle drops below cfg.settle_rest_speed, then projects free particles out of the colliders.
// `seams_only` ("Settle seams", Pattern files with seams): no gravity, cfg.seam_settle_seconds / _iters, and the sewn links start
// at their current offset and close smoothly to 0 (smoothstep, so no velocity is left), instead of snapping. The result has
// settled.seams_only set and the next normal settle starts from it. Colliders stay frozen in both.
// Returns false if the window was closed. `headless` (--fd-config --settle): no live view.
static bool settle_cloth(AppConfig& cfg, bool headless = false, bool seams_only = false)
{
    constexpr Real kDt = 1.0 / 60.0;
    const Real     rest_speed = seams_only ? -1.0 : std::max<Real>(cfg.settle_rest_speed, 0.0); // [m/s]; the seam stage never stops early
    const int      n_iters    = std::max(seams_only ? cfg.seam_settle_iters : cfg.settle_iters, 1);
    const int      max_steps  = std::max(1, (int)((seams_only ? cfg.seam_settle_seconds : cfg.settle_seconds) / kDt));
    const Vec3     gravity    = seams_only ? Vec3::Zero() : cfg.gravity;

    const WorldSetup world = prepare_world(cfg);
    bool aborted = false;

    SettledStart result;
    auto settle_one = [&](const Stiffness& stiffness, const char* label) -> RealVecX
    {
        Object obj = build_cloth(cfg, stiffness, cfg.origin); // always from the procedural mesh
        if (cfg.cloth_type == ClothType::Skirt && !skinned_meshes.empty())  // shortcut: first skinned body only
        {
            result.pins = snap_pins_to_skinned_body(obj, 0, cfg.origin, 1e-2);
            set_pinned_positions(obj, result.pins);
        }
        init_pd_velocity(obj, kDt);
        attach_waist(obj, world.waist_attach_anim_id, world.waist_attach_skinned_id, collider_animations);
        if (seams_only) { obj.energies.seams.set_offsets(obj.x); obj.energies.seams.pull = 1.0; }
        else if (cfg.settled.valid && cfg.settled.seams_only) // continue from the sewn pose
        {
            ASSERT(cfg.settled.x.size() == obj.x.size(), "settled pose has " << cfg.settled.x.size() << " values, the cloth " << obj.x.size());
            obj.x = cfg.settled.x;
        }

        Tape tape;
        Real max_speed = 0.0;
        const StepCallback on_step = [&](int step, int n) -> bool
        {
            max_speed = tape.velocities.back().rowwise().norm().maxCoeff();
            if (seams_only) // the next step wants this fraction of the initial offset
            {
                const Real u = std::min<Real>(1.0, (step + 1.0) / n);
                obj.energies.seams.pull = 1.0 - u * u * (3.0 - 2.0 * u);
            }
            if (headless) return max_speed > rest_speed;
            if (!viewer_poll_close()) { aborted = true; return false; }
            if (step % 2 == 0)
            {
                std::ostringstream oss;
                oss << label << "   step " << step << "/" << n;
                if (seams_only) oss << "   longest link " << obj.energies.seams.max_gap(obj.x) << " m";
                else            oss << "   max speed " << max_speed << " m/s";
                viewer_set_scene(obj.mesh, nullptr, nullptr, &tape.positions.back(), &tape.contacts.back(),
                                 colliders, 0.0, oss.str());
                if (!viewer_render_frame()) { aborted = true; return false; }
            }
            return max_speed > rest_speed; // false = settled, stop
        };
        pd_contact(obj, kDt, gravity, n_iters, max_steps, 1, tape, "settle", 
                   /*export_obj=*/false, 
                   /*verbose=*/false,
                   on_step, 0.0, 
                   /*frozen=*/true);
        const int steps = (int)tape.positions.size() - 1;
        int moved = 0;
        const int still_inside = project_out_of_colliders(obj, moved); // start collision free
        std::cout << "[settle] " << label << ": " << steps << " steps, max speed " << max_speed << " m/s, projected "
                  << moved << " particles out of the colliders\n";
        if (still_inside > 0) WARNING(label << ": " << still_inside << " particles still penetrate a collider after projection");
        result.max_speed  = max_speed;
        result.steps      = steps;
        result.seams_only = seams_only;
        result.seam_gap   = obj.energies.seams.links.empty() ? -1.0 : obj.energies.seams.max_gap(obj.x);
        return obj.x;
    };

    RealVecX x = settle_one(cfg.settle_k(), seams_only ? "Settling seams" : "Settling cloth");
    if (aborted) return false;
    result.valid = true;
    result.key   = settle_key(cfg);
    result.x     = std::move(x);
    cfg.settled  = std::move(result);
    return true;
}

#include "diffpd_checks.h"

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "--regression")        return run_regression();
    if (argc > 1 && std::string(argv[1]) == "--meshgen-check")     return run_meshgen_check();
    if (argc > 1 && std::string(argv[1]) == "--arap-check")        return run_arap_check();
    if (argc > 1 && std::string(argv[1]) == "--speculative-check") return run_speculative_check();
    if (argc > 2 && std::string(argv[1]) == "--fd-config")         return run_fd_config(argv[2], argc > 3 && std::string(argv[3]) == "--settle");
    if (argc > 1 && std::string(argv[1]) == "--anderson-check")    return argc > 2 ? run_anderson_compare(argv[2]) : run_anderson_check();
    if (argc > 2 && std::string(argv[1]) == "--chunk-check")       return run_chunk_check(argv[2]);
    if (argc > 3 && std::string(argv[1]) == "--contact-diff")
        return run_contact_diff(argv[2], std::stod(argv[3]), argc > 4 && std::string(argv[4]) == "--settle");

    ANIM_DIR = ANIM_DIR_DEFAULT;
    clear_folder(ANIM_DIR);

    viewer_open();

    AppConfig cfg; // declared outside the loop so edits survive a "Back to Setup" restart

    // cfg.waist_attach_enabled defaults to true, so seed the origin here, once, from the default
    // animation's hip before the config screen is ever shown; the screen's own handlers only snap
    // on a false->true toggle or an animation change and wouldn't otherwise fire for a value that
    // starts true.
    if (cfg.waist_attach_enabled)
    {
        const AnimationPreview initial = load_animation_preview(cfg.animation_file);
        if (initial.has_waist && initial.waist_snaps_origin) cfg.origin = initial.waist_default_origin;
    }

    while (viewer_show_config_screen(cfg, [](AppConfig& c, bool seams_only) { return settle_cloth(c, false, seams_only); }))
    {

    bool aborted = false;

    // cloth parameters
    const Stiffness stiffness        = cfg.guess_k();
    const Stiffness target_stiffness = cfg.target_k();

    // world parameters: colliders, animation tracks, skinned bodies, waist attachment target
    const WorldSetup world = prepare_world(cfg);
    const int waist_attach_anim_id    = world.waist_attach_anim_id;
    const int waist_attach_skinned_id = world.waist_attach_skinned_id;

    // physics parameters
    const Vec3 gravity = cfg.gravity;

    // simulation parameters
    const int FPS            = cfg.FPS;
    const int frame_substeps = cfg.frame_substeps;
    const int secs           = cfg.secs;

    // solver parameters
    const int n_iters         = cfg.n_iters;         // forward PD global-local iterations per step
    const int n_iters_adjoint = cfg.n_iters_adjoint; // backward adjoint-vector iterations per step
    const int substeps = FPS * frame_substeps;
    const Real dt      = 1.0 / substeps;
    const int  n_steps = substeps * secs;

    // Forward-only mode: one simulation with the guess parameters, then straight to playback. No target,
    // loss, adjoint or FD check.
    if (!cfg.backward_enabled())
    {
        stage_timer.reset();
        stage_timer.enabled = true;
        SimRunResult run = run_forward_simulation(
            cfg, stiffness, gravity, dt, n_iters, n_steps, frame_substeps,
            waist_attach_anim_id, waist_attach_skinned_id, collider_animations, "forward", "Forward simulation",
            /*verbose=*/false, aborted);
        stage_timer.enabled = false;
        if (aborted) break;
        const std::vector<StageShare> forward_timing = stage_timer.shares("loop", "viewer_callback");

        const ResidualHistory residual_history{ {}, run.tape.forward_residual, {} };
        if (!viewer_interactive_playback(run.obj.mesh, run.tape, /*backward=*/nullptr, colliders, dt, 1, FPS * frame_substeps,
                                          residual_history, forward_timing, collider_animations, run.obj.waist))
            break; // window closed; "Back to Setup" loops back to the config screen
        continue;
    }

    Tape target_tape = run_forward_simulation(
        cfg, target_stiffness, gravity, dt, n_iters, n_steps, frame_substeps,
        waist_attach_anim_id, waist_attach_skinned_id, collider_animations, "target", "Target simulation",
        /*verbose=*/true, aborted).tape;

    if (aborted) break;

    const bool run_fd_check = cfg.run_fd_check;

    { // guess sim + backward pass + playback (Guess + Target + Backward mode)
        stage_timer.reset();
        stage_timer.enabled = true; // time only the guess run (the target's breakdown would be similar)
        SimRunResult guess = run_forward_simulation(
            cfg, stiffness, gravity, dt, n_iters, n_steps, frame_substeps,
            waist_attach_anim_id, waist_attach_skinned_id, collider_animations, "guess", "Guess simulation",
            /*verbose=*/false, aborted);
        stage_timer.enabled = false;
        Object& guess_obj  = guess.obj;
        Tape&   guess_tape = guess.tape;

        if (aborted) break;
        const std::vector<StageShare> forward_timing = stage_timer.shares("loop", "viewer_callback");

        Loss loss(guess_tape, target_tape, frame_substeps);
        std::cout << "loss = " << loss.total << "\n";

        const StepCallback backward_on_step = [&](int t, int n) -> bool
        {
            if (!viewer_poll_close()) { aborted = true; return false; }
            if (t % frame_substeps != 0) return true;
            std::ostringstream oss;
            oss << "Backward pass   step " << t << "/" << n;
            viewer_set_scene(guess_obj.mesh, &target_tape.positions[t], &target_tape.contacts[t - 1],
                              &guess_tape.positions[t], &guess_tape.contacts[t - 1],
                              colliders, t * dt, oss.str());
            if (!viewer_render_frame()) { aborted = true; return false; }
            return true;
        };
        const BackwardGradContact grad = backward_pd_contact(guess_obj, guess_tape, loss, gravity, n_iters_adjoint, dt, backward_on_step, /*verbose=*/true);

        if (aborted) break;

        const Vec3 dphi_dv0 = Eigen::Map<const PointsX>(
            grad.dphi_dv.data(), guess_obj.num_particles(), 3).colwise().sum().transpose();
        const Vec3 dphi_dx0 = Eigen::Map<const PointsX>(
            grad.dphi_dx.data(), guess_obj.num_particles(), 3).colwise().sum().transpose();

        std::cout << "dphi/dv0 = (" << dphi_dv0.x() << ", " << dphi_dv0.y() << ", " << dphi_dv0.z() << ")\n";
        std::cout << "dphi/dx0 = (" << dphi_dx0.x() << ", " << dphi_dx0.y() << ", " << dphi_dx0.z() << ")\n";
        // Stiffness gradients to report: the single unified dphi/dk, or one per enabled spring kind.
        std::vector<std::pair<std::string, Real>> dphi_dk_lines;
        if (!cfg.per_type_k())
        {
            dphi_dk_lines.push_back({ "dphi/dk", grad.dphi_dk.sum() });
        }
        else
        {
            dphi_dk_lines.push_back({ "dphi/dk_stretch", grad.dphi_dk.stretch });
            if (cfg.shear_on())   dphi_dk_lines.push_back({ "dphi/dk_shear", grad.dphi_dk.shear });
            if (cfg.bending_on() || cfg.use_arap()) dphi_dk_lines.push_back({ "dphi/dk_bend",  grad.dphi_dk.bend }); // ARAP: hinge
        }
        for (const auto& [name, value] : dphi_dk_lines)
            std::cout << name << " = " << value << "\n";
        if (cfg.damping_enabled)
            std::cout << "damping (alpha=" << cfg.damping_alpha << ", beta=" << cfg.damping_beta << ")"
                      << "  dphi/dalpha = " << grad.dphi_dalpha << "  dphi/dbeta = " << grad.dphi_dbeta << "\n";

        // The FD check verifies the first reported gradient: the unified k (every spring perturbed
        // together) or, per-type, k_stretch alone. In both cases the unperturbed value is stiffness.stretch.
        const std::string fd_label    = dphi_dk_lines.front().first;
        const Real        fd_analytic = dphi_dk_lines.front().second;

        // Runs a stiffness FD check against fd_analytic for each selected epsilon not already in
        // fd_cache; also handed to the playback screen as an FDCheckRunner so more epsilons can be
        // checked later without recomputing the trajectory. The cache lives for this experiment only.
        FDCheckCache fd_cache;
        auto run_fd_checks = [&](const bool (&selected)[9])
        {
            std::vector<int> todo; // indices into kFDEpsilonValues
            for (int i = 0; i < 9; ++i)
                if (selected[i] && !fd_cache[i]) todo.push_back(i);
            if (todo.empty()) return;

            auto build_guess_k = [&](Real k) -> Object
            {
                const Stiffness fd_stiffness = cfg.per_type_k()
                    ? Stiffness{ k, stiffness.shear, stiffness.bend }
                    : Stiffness::uniform(k);
                Object obj = build_run_cloth(cfg, fd_stiffness);
                attach_waist(obj, waist_attach_anim_id, waist_attach_skinned_id, collider_animations);
                return obj;
            };

            // "1m 05s" / "42s" / "1h 03m"
            auto format_duration = [](double secs) -> std::string
            {
                const long s = std::max(0L, std::lround(secs));
                char buf[32];
                if (s >= 3600)    std::snprintf(buf, sizeof(buf), "%ldh %02ldm", s / 3600, (s % 3600) / 60);
                else if (s >= 60) std::snprintf(buf, sizeof(buf), "%ldm %02lds", s / 60, s % 60);
                else              std::snprintf(buf, sizeof(buf), "%lds", s);
                return buf;
            };

            // Time estimate: every FD run is a full forward sim of the same length, so the batch is
            // todo.size() * 2 * n_steps near-identical steps; remaining = (mean time per step so far)
            // * (steps left).
            const long long total_steps = (long long)todo.size() * 2 * n_steps;
            const auto      t_start     = std::chrono::steady_clock::now();

            for (size_t j = 0; j < todo.size(); ++j)
            {
                const Real eps = kFDEpsilonValues[todo[j]];

                // FD reruns stay headless; this just pumps the window/status line so it isn't frozen.
                // `step` spans both runs of this epsilon (n = 2*n_steps): the first half is the +eps
                // run, the second the -eps run — see fd_check_contact_stiffness.
                const StepCallback fd_heartbeat = [&](int step, int n) -> bool
                {
                    if (!viewer_poll_close()) { aborted = true; return false; }
                    if (step % frame_substeps != 0) return true;
                    const int  pass_steps = n / 2; // == n_steps
                    const long long done  = (long long)j * n + step + 1;
                    const double elapsed  = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                    const double remaining = elapsed / (double)done * (double)(total_steps - done);
                    std::ostringstream oss;
                    oss << "FD check   eps = " << std::scientific << std::setprecision(0) << eps << std::defaultfloat
                        << " (" << j + 1 << "/" << todo.size() << ")   "
                        << (step < pass_steps ? "+eps" : "-eps") << " run   step "
                        << step % pass_steps + 1 << "/" << pass_steps;
                    viewer_set_status(oss.str(), "elapsed " + format_duration(elapsed)
                                                 + "   remaining ~" + format_duration(remaining));
                    if (!viewer_render_frame()) { aborted = true; return false; }
                    return true;
                };

                std::cout << "running fd check with eps=" << eps << "\n";
                const FDCheckResult r = fd_check_contact_stiffness(
                    build_guess_k, target_tape, stiffness.stretch, dt, gravity, n_iters, n_steps, frame_substeps,
                    frame_substeps, fd_analytic, eps, fd_heartbeat);
                if (aborted) break; // truncated runs give a bogus fd value: don't cache it
                std::cout << "  " << fd_label << ": eps=" << eps << " fd=" << r.fd
                          << " analytic=" << r.analytic << " rel_err=" << r.rel_err << "\n";
                fd_cache[todo[j]] = r;
            }
        };

        if (run_fd_check)
        {
            run_fd_checks(cfg.fd_eps_selected);
            if (aborted) break;
        }

        const GradientSummary grad_summary{ loss.total, dphi_dx0, dphi_dv0, dphi_dk_lines };
        const ResidualHistory residual_history{ target_tape.forward_residual, guess_tape.forward_residual, grad.residual };
        const FDCheckRunner fd_runner = run_fd_checks; // named: BackwardPlayback holds a reference
        const BackwardPlayback backward_playback{ target_tape, cfg.fd_eps_selected, fd_cache, fd_runner, grad_summary };
        if (!viewer_interactive_playback(guess_obj.mesh, guess_tape, &backward_playback, colliders, dt, 1, FPS * frame_substeps,
                                          residual_history, forward_timing, collider_animations, guess_obj.waist))
            break; // window closed; "Back to Setup" falls through and loops back to the config screen
    }

    } // while (viewer_show_config_screen(cfg))

    viewer_close();
    return 0;
}