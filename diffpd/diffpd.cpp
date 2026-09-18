#include "diffpd_types.h"
#include "diffpd_viewer.h"

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
ContactPointMode contact_point_mode = ContactPointMode::Surface; // default; overwritten in main()

// How an animated collider's contact velocity is estimated; overwritten in main().
AnimatedColliderVelocityMode animated_collider_velocity_mode = AnimatedColliderVelocityMode::MaterialPointDiff;

// Grow the contact set inside pd_contact's iteration loop (see merge_detected_contacts); overwritten in main().
bool contact_active_set_update = true;

// Parallel to `colliders`: colliders[i].anim_id, when >= 0, indexes into this track list.
std::vector<ColliderAnimation> collider_animations;

// Name of the collider (Blender empty / collider_animation.json entry) that Waist Attachment pins to.
constexpr const char* kWaistAttachmentColliderName = "collider_hip";

// Parses collider_animation.json, appending one `animated=true` Collider per "colliders_metadata"
// entry and returning the matching frame tracks. Missing/unreadable file just warns and returns
// empty rather than failing, since this is an optional layer on the config-managed collider list.
std::vector<ColliderAnimation> load_collider_animation(const std::string& path, std::vector<Collider>& out_colliders)
{
    std::vector<ColliderAnimation> anims;

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

    // name -> index into `anims`, so the per-frame loop below can find each track by name.
    std::unordered_map<std::string, int> name_to_anim_id;

    for (auto& [name, meta] : j.at("colliders_metadata").items())
    {
        ColliderAnimation anim;
        anim.name = name;

        const std::string type_str = meta.value("type", "capsule");
        anim.type = (type_str == "sphere") ? ColliderType::Sphere : ColliderType::Capsule;

        anim.radius      = meta.value("radius", 0.0);
        anim.half_length = meta.value("half_length", 0.0);
        const auto axis  = meta.value("axis_local", std::vector<Real>{0.0, 1.0, 0.0});
        anim.axis_local  = Vec3(axis[0], axis[1], axis[2]);

        Collider c;
        c.type     = anim.type;
        c.animated = true;
        c.anim_id  = (int)anims.size();

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
// passes a small positive slack so negligible residual penetration doesn't count).
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
        const AABB         box  = collider_aabb(collider, pose); // valid only for Sphere/Capsule (finite shapes)

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
Vec3 update_contact_force(const Vec3& f_i, Real m_i, const Vec3& v_c, const Vec3& n)
{
    const Real d_n = (f_i - m_i * v_c).dot(n);
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

    Loss(const Tape& guess, const Tape& target, int sample_every)
    {
        ASSERT(guess.positions.size() == target.positions.size(),
               "tape size mismatch: " << guess.positions.size() << " vs " << target.positions.size());
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
            const RealVecX xt = Eigen::Map<const RealVecX>(target.positions[t].data(), dofs);

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

constexpr Real cloth_size = 2.0;

Object cloth(
    Index       width,
    Index       height,
    Real        stiffness,
    Vec3        origin,
    PinMode     pin_mode,
    HangingMode hanging_mode,
    uint8_t     flags,
    Real        m_tot)
{
    ASSERT(flags & ClothFlags::STRETCH, "cloth must have stretch constraints enabled");

    const Index N  = width * height;
    const Real  sx = cloth_size / Real(width  - 1);
    const Real  sz = cloth_size / Real(height - 1);

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
    auto emit = [&](Index a, Index b) 
    {
        const Real l  = (pos[a] - pos[b]).norm();
        const bool pa = pinned[a], pb = pinned[b];
        if (pa && pb) return;
        if (!pa && !pb)
        {
            obj.constraints.push_back(
                Constraint::makeSpring2(stiffness, l, dof[a], dof[b]));
        }
        else
        {
            const Index  free   = pa ? b : a;
            const Index  anchor = pa ? a : b;
            const Vec3   xbar   = pos[anchor];
            // dof[anchor] is just the shared "-1 = pinned" sentinel, not a unique index — the real
            // pinned_rest index is mesh.vertices[anchor].dof, baked by the mesh loop above.
            obj.constraints.push_back(
                Constraint::makeSpring1(stiffness, l, dof[free], xbar, Index(-(mesh.vertices[anchor].dof + 1))));
        }
    };

    if (flags & ClothFlags::STRETCH) 
    {
        // structural: along width axis (i)
        for (Index i = 0; i < width - 1; ++i)
            for (Index j = 0; j < height; ++j)
                emit(grid(i, j), grid(i + 1, j));

        // structural: along height axis (j)
        for (Index i = 0; i < width; ++i)
            for (Index j = 0; j < height - 1; ++j)
                emit(grid(i, j), grid(i, j + 1));
    }

    if (flags & ClothFlags::SHEAR)
    {
        for (Index i = 0; i < width - 1; ++i)
            for (Index j = 0; j < height - 1; ++j) 
            {
                emit(grid(i, j),     grid(i + 1, j + 1));
                emit(grid(i + 1, j), grid(i, j + 1));
            }
    }

    if (flags & ClothFlags::BENDING) 
    {
        for (Index i = 0; i < width; ++i)
            for (Index j = 0; j < height - 2; ++j)
                emit(grid(i, j), grid(i, j + 2));

        for (Index i = 0; i < width - 2; ++i)
            for (Index j = 0; j < height; ++j)
                emit(grid(i, j), grid(i + 2, j));
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
    Index   num_rings,
    Real    stiffness,
    Vec3    origin,
    Real    radius_top,
    Real    radius_bottom,
    Real    height,
    uint8_t flags,
    Real    m_tot)
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
    auto emit = [&](Index a, Index b)
    {
        const Real l  = (pos[a] - pos[b]).norm();
        const bool pa = pinned[a], pb = pinned[b];
        if (pa && pb) return;
        if (!pa && !pb)
        {
            obj.constraints.push_back(
                Constraint::makeSpring2(stiffness, l, dof[a], dof[b]));
        }
        else
        {
            const Index free   = pa ? b : a;
            const Index anchor = pa ? a : b;
            const Vec3  xbar   = pos[anchor];
            // dof[anchor] is just the shared "-1 = pinned" sentinel, not a unique index — the real
            // pinned_rest index is mesh.vertices[anchor].dof, baked by the mesh loop above.
            obj.constraints.push_back(
                Constraint::makeSpring1(stiffness, l, dof[free], xbar, Index(-(mesh.vertices[anchor].dof + 1))));
        }
    };

    if (flags & ClothFlags::STRETCH)
    {
        // structural: circumferential (i), wraps around the ring
        for (Index i = 0; i < W; ++i)
            for (Index j = 0; j < H; ++j)
                emit(grid(i, j), grid(wrap(i + 1), j));

        // structural: vertical (j), open (no top/bottom cap)
        for (Index i = 0; i < W; ++i)
            for (Index j = 0; j < H - 1; ++j)
                emit(grid(i, j), grid(i, j + 1));
    }

    if (flags & ClothFlags::SHEAR)
    {
        for (Index i = 0; i < W; ++i)
            for (Index j = 0; j < H - 1; ++j)
            {
                emit(grid(i, j),           grid(wrap(i + 1), j + 1));
                emit(grid(wrap(i + 1), j), grid(i, j + 1));
            }
    }

    if (flags & ClothFlags::BENDING)
    {
        // vertical bending: open, same as cloth
        for (Index i = 0; i < W; ++i)
            for (Index j = 0; j < H - 2; ++j)
                emit(grid(i, j), grid(i, j + 2));

        // circumferential bending: wraps. Skipped below W=5, where stepping by 2 duplicates a
        // stretch edge (W=3) or double-emits the same pair (W=4); W>=5 gives distinct pairs.
        if (W >= 5)
            for (Index i = 0; i < W; ++i)
                for (Index j = 0; j < H; ++j)
                    emit(grid(i, j), grid(wrap(i + 2), j));
    }

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

void write_obj_constraints(const std::string& path, const Object& obj)
{
    std::ofstream out(path);
    ASSERT(out.is_open(), "could not open " << path);
    out << std::fixed << std::setprecision(6);

    const Index n = obj.num_particles();

    for (Index i = 0; i < n; ++i)
    {
        const Vec3 p = obj.x.segment<3>(3 * i);
        out << "v " << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    }

    for (const Constraint& c : obj.constraints)
        if (c.type == SpringType::Spring1)
            out << "v " << c.spring1.xbar[0] << ' '
                        << c.spring1.xbar[1] << ' '
                        << c.spring1.xbar[2] << '\n';

    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            out << "l " << i1 + 1 << ' ' << i2 + 1 << '\n';
        }
    }

    Index anchor_vid = n + 1;
    for (const Constraint& c : obj.constraints)
        if (c.type == SpringType::Spring1)
            out << "l " << c.spring1.i + 1 << ' ' << anchor_vid++ << '\n';
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

// Adds each constraint's elastic force contribution to `b` in place, scaled by `scale`.
// No allocation: `b` is the caller's buffer, sized to obj.num_dofs().
void add_elastic_forces(const Object& obj, const RealVecX& x, RealVecX& b, Real scale = 1.0)
{
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            const Vec3 e   = x.segment<3>(3*i1) - x.segment<3>(3*i2);
            const Vec3 p   = c.l * e / e.norm();
            b.segment<3>(3*i1) += scale * c.k * p;
            b.segment<3>(3*i2) -= scale * c.k * p;
        }
        else // SpringType::Spring1
        {
            const Vec3 xbar(c.spring1.xbar[0], c.spring1.xbar[1], c.spring1.xbar[2]);
            const Index i = c.spring1.i;
            const Vec3 e  = x.segment<3>(3*i) - xbar;
            const Vec3 p  = c.l * e / e.norm();
            b.segment<3>(3*i) += scale * c.k * (xbar + p);
        }
    }
}

// ----------------
// VELOCITY SOLVER
// ----------------

void construct_velocity_lhs(Object& obj, Real dt)
{
    const Index n3 = obj.num_dofs();
    const Real  h2 = dt * dt;

    std::vector<Triplet> triplets_L, triplets_C;
    triplets_L.reserve(n3 + 12 * Index(obj.constraints.size()));
    triplets_C.reserve(     12 * Index(obj.constraints.size()));

    // M (mass diagonal, not divided by h²)
    for (Index i = 0; i < n3; ++i)
        triplets_L.emplace_back(i, i, obj.mass(i));

    // h² · K  (elastic contribution)
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            for (int d = 0; d < 3; ++d)
            {
                triplets_L.emplace_back(3*i1+d, 3*i1+d, +h2 * c.k);
                triplets_L.emplace_back(3*i2+d, 3*i2+d, +h2 * c.k);
                triplets_L.emplace_back(3*i1+d, 3*i2+d, -h2 * c.k);
                triplets_L.emplace_back(3*i2+d, 3*i1+d, -h2 * c.k);

                triplets_C.emplace_back(3*i1+d, 3*i1+d, +h2 * c.k);
                triplets_C.emplace_back(3*i2+d, 3*i2+d, +h2 * c.k);
                triplets_C.emplace_back(3*i1+d, 3*i2+d, -h2 * c.k);
                triplets_C.emplace_back(3*i2+d, 3*i1+d, -h2 * c.k);
            }
        }
        else // Spring1
        {
            const Index i = c.spring1.i;
            for (int d = 0; d < 3; ++d)
            {
                triplets_L.emplace_back(3*i+d, 3*i+d, +h2 * c.k);
                triplets_C.emplace_back(3*i+d, 3*i+d, +h2 * c.k);
            }
        }
    }

    obj.L.resize(n3, n3);
    obj.L.setFromTriplets(triplets_L.begin(), triplets_L.end());

    obj.C.resize(n3, n3);
    obj.C.setFromTriplets(triplets_C.begin(), triplets_C.end());

    obj.solver = std::make_unique<Cholesky>();
    obj.solver->compute(obj.L);
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
    for (Constraint& c : obj.constraints)
    {
        Vec3 e;

        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            e = x.segment<3>(3*i1) - x.segment<3>(3*i2);
        }
        else // SpringType::Spring1
        {
            const Vec3 xbar(c.spring1.xbar[0], c.spring1.xbar[1], c.spring1.xbar[2]);
            const Index i = c.spring1.i;
            e = x.segment<3>(3*i) - xbar;
        }

        const Real e_norm = e.norm();

        if (e_norm < 1e-9)
        {
            c.gamma  = Mat3::Zero();
            c.p_star = Vec3::Zero();
            c.e      = Vec3::Zero();
            continue;
        }

        const Vec3 e_hat  = e / e_norm;
        const Mat3 P_perp = Mat3::Identity() - (e_hat * e_hat.transpose());

        c.e      = e;
        c.p_star = c.l * e_hat;
        c.gamma  = (c.k * c.l / e_norm) * P_perp;
    }
}

// Applies the assembled local spring Jacobian ΔA (Σ_i Gᵢᵀ Γᵢ Gᵢ) to a vector — the same
// per-constraint pattern used by both the position- and velocity-space adjoint RHS.
RealVecX apply_spring_jacobian(const Object& obj, const RealVecX& v)
{
    RealVecX out = RealVecX::Zero(v.rows());

    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;
            const Vec3 d   = c.gamma * (v.segment<3>(3*i1) - v.segment<3>(3*i2));
            out.segment<3>(3*i1) += d;
            out.segment<3>(3*i2) -= d;
        }
        else // SpringType::Spring1
        {
            const Index i = c.spring1.i;
            out.segment<3>(3*i) += c.gamma * v.segment<3>(3*i);
        }
    }

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
    const RealVecX f         = b_tilde - obj.C * v_plus;

    for (Contact& c : contacts)
    {
        const Vec3 f_i = f.segment<3>(3 * c.particle);
        const Real m_i = obj.mass(3 * c.particle);
        const Real d_n = (f_i - m_i * c.v_c).dot(c.normal);
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
        b += obj.C * split.Pz;

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

    constexpr Real kConvergenceTol = 1e-4;

    RealVecX z = RealVecX::Zero(obj.num_dofs());
    Real rel_residual = 0.0;
    for (int k = 0; k < n_iters_adjoint; ++k)
    {
        const RealVecX b_back = construct_backward_contact_rhs(obj, contacts, z, dloss_dv, dloss_dv_t, h);
        const RealVecX z_new  = obj.solver->solve(b_back);
        rel_residual = (z_new - z).norm() / std::max(z_new.norm(), Real(1e-12));
        z = z_new;
    }
    residual_out = rel_residual;
    if (verbose && n_iters_adjoint > 0 && rel_residual > kConvergenceTol)
    {
        std::cout << "  iter " << n_iters_adjoint << " rel_delta = " << rel_residual << "\n";
    }

    return z;
}

Real compute_gradient_stiffness_contact(
    const Object&     obj,
    const Contacts&   contacts,
    const RealVecX&   z,
    const Positions&  x_minus,
    const Velocities& v_plus,
    Real              h)
{
    const RealVecX z_perp = split_by_contact(contacts, z).z_perp;
    const Real     h2     = h * h;

    Real grad = 0;
    for (const Constraint& c : obj.constraints)
    {
        if (c.type == SpringType::Spring2)
        {
            const Index i1 = c.spring2.i1;
            const Index i2 = c.spring2.i2;

            const Vec3 e_minus = x_minus.segment<3>(3*i1) - x_minus.segment<3>(3*i2);
            const Vec3 dv      = v_plus.segment<3>(3*i1)  - v_plus.segment<3>(3*i2);
            const Vec3 dz      = z_perp.segment<3>(3*i1)  - z_perp.segment<3>(3*i2);

            grad += dz.dot(h * (c.p_star - e_minus) - h2 * dv);
        }
        else // Spring1
        {
            const Vec3 xbar(c.spring1.xbar[0], c.spring1.xbar[1], c.spring1.xbar[2]);
            const Index i = c.spring1.i;

            const Vec3 e_minus = x_minus.segment<3>(3*i) - xbar;
            const Vec3 vi      = v_plus.segment<3>(3*i);
            const Vec3 zi      = z_perp.segment<3>(3*i);

            grad += zi.dot(h * (c.p_star - e_minus) - h2 * vi);
        }
    }
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

void pd_contact(Object& obj, Real dt, const Vec3& gravity, int n_iters, int n_steps, int frame_substeps, Tape& tape, const std::string& prefix, bool export_obj = true, bool verbose = false, const StepCallback& on_step = nullptr, Real unresolved_penetration_threshold = 0.0)
{
    // Preconditions, checked once up front so the body below stays assertion-free.
    auto validate = [&]()
    {
        if (!collider_animations.empty())
            ASSERT(frame_substeps == 1 && std::abs(dt - 1.0 / kColliderAnimFPS) < 1e-9,
                   "animated colliders currently assume a fixed " << kColliderAnimFPS << "fps step with no "
                   "substepping (frame_substeps=1, dt=1/" << kColliderAnimFPS << ") — got frame_substeps="
                   << frame_substeps << ", dt=" << dt);
    };
    validate();

    tape.clear();
    tape.record(obj);
    if (export_obj) write_obj_frame(obj, 0, prefix);

    // Poses animated colliders (+ waist attachment) from the imported track at `frame`. Called for
    // frame 0 up front, then once per step so contact detection/velocity basis see the right pose.
    auto stamp_animated_colliders = [&](int frame) { restamp_animated_colliders(colliders, collider_animations, obj, frame); };

    stamp_animated_colliders(0);

    size_t total_added_in_iters = 0; // contacts found mid-iteration (merge_detected_contacts), whole run
    for (int step = 0; step < n_steps; ++step)
    {
        stamp_animated_colliders(step + 1);

        RealVecX x_tilde = obj.x + dt * obj.v;
        const Vec3 dg = (dt * dt) * gravity;
        for (Index i = 0; i < obj.num_particles(); ++i)
            x_tilde.segment<3>(3*i) += dg;

        const Real     contact_time = (step + 1) * dt;
        const RealVecX b_inertia    = obj.mass.cwiseProduct(x_tilde);
        Contacts       contacts     = detect_contacts(obj, x_tilde, contact_time);

        obj.prev_x = obj.x;
        obj.x      = x_tilde;

        // Instantaneous velocity of an animated collider is a precomputed, static property of its
        // track (see ColliderAnimation::velocity_basis) — just look it up for this step's frame.
        auto contact_velocity = [&](const Contact& c, const Vec3& contact_point) -> Vec3
        {
            const Collider& collider = colliders[c.collider_id];
            if (!collider.animated)
                return collider_point_velocity(collider, contact_point, contact_time);
            const auto& basis_table = collider_animations[collider.anim_id].velocity_basis;
            const int   frame       = std::clamp(step + 1, 0, (int)basis_table.size() - 1);
            return animated_collider_point_velocity_from_basis(basis_table[frame], contact_point, animated_collider_velocity_mode);
        };

        // Surface mode: v_c is frozen at the surface point, stamped once per contact (Particle mode
        // instead recomputes it every iteration below).
        auto stamp_surface_velocities = [&](size_t begin, size_t end)
        {
            if (contact_point_mode != ContactPointMode::Surface) return;
            for (size_t j = begin; j < end; ++j)
                contacts[j].v_c = contact_velocity(contacts[j], contacts[j].surface_point);
        };
        stamp_surface_velocities(0, contacts.size());

        int contacts_added_in_iters = 0;
        for (int k = 0; k < n_iters; ++k)
        {
            // Re-test the current iterate and grow the contact set (merge_detected_contacts). Runs
            // before the solve so the last iteration — the one the tape records — sees the final set.
            if (k > 0 && contact_active_set_update)
            {
                const int added = merge_detected_contacts(contacts, obj, obj.x, x_tilde, contact_time);
                stamp_surface_velocities(contacts.size() - added, contacts.size());
                contacts_added_in_iters += added;
            }

            const RealVecX b_tilde = construct_velocity_rhs(obj, b_inertia, obj.x, obj.prev_x, dt);

            const RealVecX f = b_tilde - obj.C * obj.v;
            RealVecX g = b_tilde;
            for (Contact& c : contacts)
            {
                const Vec3 f_i = f.segment<3>(3 * c.particle);
                const Real m_i = obj.mass(3 * c.particle);

                if (contact_point_mode == ContactPointMode::Particle)
                    c.v_c = contact_velocity(c, Vec3(obj.x.segment<3>(3 * c.particle)));

                g.segment<3>(3 * c.particle) += update_contact_force(f_i, m_i, c.v_c, c.normal);
            }

            const RealVecX v_hat = obj.solver->solve(g);
            if (k == n_iters - 1)
            {
                const Real rel_delta = (v_hat - obj.v).norm() / std::max(v_hat.norm(), Real(1e-12));
                tape.forward_residual.push_back(rel_delta);
                if (verbose && rel_delta > 1e-4)
                    std::cout << "  step " << step << " iter " << k+1 << " rel_delta = " << rel_delta << "\n";
            }

            obj.v = v_hat;
            obj.x = obj.prev_x + dt * obj.v;
        }

        tape.record_contacts(contacts); // after the solve: carries the converged per-contact v_c
        // Re-detect against the converged x: how many particles are still penetrating a collider
        // after this step's iterations, as opposed to `contacts` above (detected pre-solve, against
        // x_tilde). A small positive threshold treats negligible residual penetration as resolved.
        tape.unresolved_contacts.push_back((int)detect_contacts(obj, obj.x, contact_time, unresolved_penetration_threshold).size());
        tape.record(obj);
        if (export_obj && step % frame_substeps == 0) write_obj_frame(obj, (step / frame_substeps) + 1, prefix);
        if (on_step && !on_step(step, n_steps)) break;
        total_added_in_iters += contacts_added_in_iters;
        if (step % 10 == 0)
            std::cout << "step " << step << "/" << n_steps
                       << "  contacts=" << contacts.size()
                       << " (added in iters=" << contacts_added_in_iters << ")"
                       << " unresolved=" << tape.unresolved_contacts.back() << "\n";
    }

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

struct BackwardGradContact
{
    RealVecX dphi_dv; // dphi/dv0 — gradient of loss w.r.t. initial velocity
    RealVecX dphi_dx; // dphi/dx0 — gradient of loss w.r.t. initial position
    Real     dphi_dk; // dphi/dk  — gradient of loss w.r.t. uniform stiffness

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
            {
                ASSERT(c.type == ColliderType::Sphere || c.type == ColliderType::Cylinder || c.type == ColliderType::Capsule,
                       "Rotation curvature correction is only implemented for Sphere, Cylinder, and Capsule colliders");
                ASSERT(contact_point_mode == ContactPointMode::Surface,
                       "Rotation curvature correction requires ContactPointMode::Surface");
            }
            if (c.animated)
            {
                ASSERT(animated_collider_velocity_mode == AnimatedColliderVelocityMode::DecomposedRigid,
                       "Differentiating contacts against an animated collider requires "
                       "AnimatedColliderVelocityMode::DecomposedRigid");
                ASSERT(c.type == ColliderType::Sphere || c.type == ColliderType::Capsule,
                       "Rotation curvature correction is only implemented for Sphere and Capsule colliders");
                ASSERT(contact_point_mode == ContactPointMode::Surface,
                       "Rotation curvature correction requires ContactPointMode::Surface");
            }
        }
    };
    validate();

    std::vector<Real> residual(n_steps, 0.0);

    RealVecX dphi_dv = RealVecX::Zero(dofs); // a_t = dL/dv^+_t, seeded from later steps
    RealVecX dphi_dx = RealVecX::Zero(dofs); // b_t = dL/dx^+_t, seeded from later steps
    Real     dphi_dk = 0.0;                  // dphi/dk, accumulated across all steps

    // Per-collider rotation state for the curvature correction below, indexed by Contact::collider_id.
    struct ColliderRotationInfo { bool enabled; Vec3 omega_vec; Real R; };
    std::vector<ColliderRotationInfo> rot_info(colliders.size());
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
        // Restamp animated colliders + waist attachment to this step's frame; shared with the
        // forward pass so pose and Spring1 xbar can't drift out of sync.
        restamp_animated_colliders(colliders, collider_animations, obj, t);

        // rot_info for animated colliders varies per frame; look it up from the same precomputed
        // table (anim, frame=t) the forward pass used for this step's contact-point velocities.
        for (int ci = 0; ci < (int)colliders.size(); ++ci)
        {
            const Collider& c = colliders[ci];
            if (!c.animated || c.anim_id < 0 || c.anim_id >= (int)collider_animations.size()) continue;
            const auto& basis_table = collider_animations[c.anim_id].velocity_basis;
            const int   frame       = std::clamp(t, 0, (int)basis_table.size() - 1);
            rot_info[ci] = { true, basis_table[frame].omega_vec, collider_surface_radius(c) };
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

        dphi_dv = obj.mass.cwiseProduct(z_perp);
        dphi_dx = h * apply_spring_jacobian(obj, z_perp) - (obj.C * z_perp) / h + b_t;

        for (const Contact& c : contacts_t)
        {
            if (!c.active) continue;
            const Index particle  = c.particle;
            const Vec3  vi        = v_plus_t.segment<3>(3 * particle);
            const Vec3  z_i       = z.segment<3>(3 * particle);
            const Vec3  z_perp_i  = z_perp.segment<3>(3 * particle);
            const Real  z_dot_n   = c.normal.dot(z_i);
            const Real  m_i       = obj.mass(3 * particle);
            const Mat3  P         = c.normal * c.normal.transpose();
            const Mat3  Q         = Mat3::Identity() - P;
            const Vec3  term      = c.d_n * z_perp_i + z_dot_n * m_i * (vi - c.v_c); // Q_i d_i = m_i (v_i^+ - v_C)
            const Vec3  projected = term - c.axis * c.axis.dot(term); // no-op unless collider is a Cylinder
            dphi_dx.segment<3>(3 * particle) -= c.inv_r * projected;
            dphi_dv.segment<3>(3 * particle) -= h * c.inv_r * projected;

            const ColliderRotationInfo& rot = rot_info[c.collider_id];
            if (rot.enabled)
            {
                const Real R       = rot.R;
                const Vec3 omega_vec = rot.omega_vec;
                const Real r_i = 1.0 / c.inv_r; // r_i (sphere/cap) or rho_i (cylinder/capsule body)
                const Vec3 c_i = omega_vec.cross(c.normal);

                Vec3 rot_term;
                if (c.axis.squaredNorm() > 0.0)
                {
                    const Vec3 c_par  = c.axis * c.axis.dot(c_i);
                    const Vec3 c_perp = c_i - c_par;
                    rot_term = m_i * z_dot_n * (c_par + (R / r_i) * c_perp);
                }
                else
                {
                    rot_term = (m_i * R / r_i) * z_dot_n * c_i;
                }
                dphi_dx.segment<3>(3 * particle) -= rot_term;
                dphi_dv.segment<3>(3 * particle) -= h * rot_term;
            }
        }

        if (on_step && !on_step(t, n_steps)) break;
        if (t % 10 == 0) std::cout << "backward (contact) step " << t << "/" << n_steps << "\n";
    }

    dphi_dx += loss.dloss_dx[0];

    return { std::move(dphi_dv), std::move(dphi_dx), dphi_dk, std::move(residual) };
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
    auto run_loss = [&](Real delta) -> Real
    {
        Object obj = build_obj_k(k0 + delta);
        init_pd_velocity(obj, dt);
        Tape tape;
        pd_contact(obj, dt, gravity, n_iters, n_steps, frame_substeps, tape, "fd_check", /*export_obj=*/false,
                   /*verbose=*/false, on_step);
        // on_step may abort early, leaving a truncated tape; bail with a sentinel instead of
        // crashing Loss's tape-length assert.
        if ((int)tape.positions.size() != n_steps + 1) return 0.0;
        return Loss(tape, target_tape, sample_every).total;
    };

    const Real loss_plus  = run_loss(+eps);
    const Real loss_minus = run_loss(-eps);
    const Real fd         = (loss_plus - loss_minus) / (2.0 * eps);
    const Real rel_err    = std::abs(fd - analytic_dphi_dk)
        / std::max({ std::abs(fd), std::abs(analytic_dphi_dk), Real(1e-12) });

    return { fd, analytic_dphi_dk, rel_err };
}

// ----------------
//      MAIN
// ----------------

struct SimRunResult { Object obj; Tape tape; };

// Builds one cloth and forward-simulates it with a live-viewer callback — shared by the target and
// guess runs in main(), which differ only in stiffness/label/prefix/verbosity. Only the trajectory
// being computed is shown live; the other one (e.g. the target, while the guess is computing) is not
// overlaid — see viewer_interactive_playback() for the target-vs-guess side-by-side comparison after
// both runs finish.
SimRunResult run_forward_simulation(
    AppConfig& cfg, Real stiffness, const Vec3& origin, const Vec3& gravity,
    Real dt, int n_iters, int n_steps, int frame_substeps,
    int waist_attach_anim_id, const std::vector<ColliderAnimation>& collider_animations,
    const std::string& prefix, const std::string& label, bool verbose,
    bool& aborted)
{
    SimRunResult result;
    result.obj = build_cloth(cfg, stiffness, origin);
    init_pd_velocity(result.obj, dt);
    if (waist_attach_anim_id >= 0)
    {
        bake_waist_attachment(result.obj, waist_attach_anim_id, collider_animations);
        update_waist_attachment(result.obj, collider_animations, 0);
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

int main()
{
    ANIM_DIR = ANIM_DIR_DEFAULT;
    clear_folder(ANIM_DIR);

    viewer_open();

    AppConfig cfg; // declared outside the loop so edits survive a "Back to Setup" restart

    // Loaded once at startup so the config screen's preview can show animated colliders at their
    // frame-0 pose (display-only; the actual run reloads into the globals below).
    // waist_attach_default_origin: the hip's frame-0 position, so the screen can snap cfg.origin
    // there the moment Waist Attachment is checked.
    std::vector<Collider> config_preview_animated_colliders;
    Vec3                   waist_attach_default_origin = Vec3::Zero();
    {
        std::vector<Collider>          tmp_colliders;
        std::vector<ColliderAnimation> tmp_anims = load_collider_animation(COLLIDER_ANIM_PATH_DEFAULT, tmp_colliders);
        for (Collider& c : tmp_colliders)
            if (c.anim_id >= 0 && c.anim_id < (int)tmp_anims.size())
                apply_collider_frame(c, tmp_anims[c.anim_id], 0);
        config_preview_animated_colliders = tmp_colliders;

        for (const ColliderAnimation& anim : tmp_anims)
            if (anim.name == kWaistAttachmentColliderName)
            {
                waist_attach_default_origin = collider_animation_pose_at(anim, 0).position;
                break;
            }
    }

    // cfg.waist_attach_enabled defaults to true, so seed the origin here, once, before the config
    // screen is ever shown; the screen's own toggle handler (see draw_config_fields) only snaps on
    // a false->true transition and wouldn't otherwise fire for a value that starts true.
    if (cfg.waist_attach_enabled)
        cfg.origin = waist_attach_default_origin;

    while (viewer_show_config_screen(cfg, config_preview_animated_colliders, waist_attach_default_origin))
    {

    bool aborted = false;

    // cloth parameters
    const Real stiffness        = cfg.stiffness;
    const Real target_stiffness = cfg.target_stiffness;
    const Vec3 origin           = cfg.origin; // shared by both target and guess cloth

    // world parameters — the collider list is UI-managed (cfg.colliders); animated colliders are
    // appended separately and aren't editable from the config screen.
    colliders                       = cfg.colliders;
    contact_point_mode              = cfg.contact_point_mode;
    animated_collider_velocity_mode = cfg.animated_collider_velocity_mode;
    contact_active_set_update       = cfg.contact_active_set_update;
    collider_animations = load_collider_animation(COLLIDER_ANIM_PATH_DEFAULT, colliders);
    for (ColliderAnimation& anim : collider_animations)
        precompute_velocity_basis(anim, animated_collider_velocity_mode);

    // Resolve the hardcoded collider name to a track index once per run; missing warns and disables.
    int waist_attach_anim_id = -1;
    if (cfg.waist_attach_enabled)
    {
        for (int i = 0; i < (int)collider_animations.size(); ++i)
            if (collider_animations[i].name == kWaistAttachmentColliderName) { waist_attach_anim_id = i; break; }
        if (waist_attach_anim_id < 0)
            WARNING("Waist Attachment enabled but no '" << kWaistAttachmentColliderName
                    << "' collider animation was loaded; disabling for this run.");
    }

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

    Tape target_tape = run_forward_simulation(
        cfg, target_stiffness, origin, gravity, dt, n_iters, n_steps, frame_substeps,
        waist_attach_anim_id, collider_animations, "target", "Target simulation",
        /*verbose=*/true, aborted).tape;

    if (aborted) break;

    const bool run_backward = true;
    const bool run_fd_check = cfg.run_fd_check;

    if (run_backward)
    {
        SimRunResult guess = run_forward_simulation(
            cfg, stiffness, origin, gravity, dt, n_iters, n_steps, frame_substeps,
            waist_attach_anim_id, collider_animations, "guess", "Guess simulation",
            /*verbose=*/false, aborted);
        Object& guess_obj  = guess.obj;
        Tape&   guess_tape = guess.tape;

        if (aborted) break;

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
        std::cout << "dphi/dk  = " << grad.dphi_dk << "\n";

        // Runs a stiffness FD check per epsilon against grad.dphi_dk; also handed to the playback
        // screen as an FDCheckRunner so it can be re-run later without recomputing the trajectory.
        auto run_fd_checks = [&](const std::vector<Real>& epss) -> std::vector<FDCheckResult>
        {
            auto build_guess_k = [&](Real k) -> Object
            {
                Object obj = build_cloth(cfg, k, origin);
                if (waist_attach_anim_id >= 0)
                {
                    bake_waist_attachment(obj, waist_attach_anim_id, collider_animations);
                    update_waist_attachment(obj, collider_animations, 0);
                }
                return obj;
            };

            // FD reruns stay headless; this just pumps the window/status line so it isn't frozen.
            const StepCallback fd_heartbeat = [&](int step, int n) -> bool
            {
                if (!viewer_poll_close()) { aborted = true; return false; }
                if (step % frame_substeps != 0) return true;
                std::ostringstream oss;
                oss << "FD check (stiffness)   step " << step << "/" << n;
                viewer_set_status(oss.str());
                if (!viewer_render_frame()) { aborted = true; return false; }
                return true;
            };

            std::vector<FDCheckResult> results;
            for (const Real eps : epss)
            {
                std::cout << "running fd check with eps=" << eps << "\n";
                const FDCheckResult r = fd_check_contact_stiffness(
                    build_guess_k, target_tape, stiffness, dt, gravity, n_iters, n_steps, frame_substeps,
                    frame_substeps, grad.dphi_dk, eps, fd_heartbeat);
                std::cout << "  k   dk: eps=" << eps << " fd=" << r.fd
                          << " analytic=" << r.analytic << " rel_err=" << r.rel_err << "\n";
                results.push_back(r);
                if (aborted) break;
            }
            return results;
        };

        if (run_fd_check)
        {
            std::vector<Real> epss;
            for (int i = 0; i < 9; ++i)
                if (cfg.fd_eps_selected[i]) epss.push_back(kFDEpsilonValues[i]);
            run_fd_checks(epss);
            if (aborted) break;
        }

        const GradientSummary grad_summary{ loss.total, dphi_dx0, dphi_dv0, grad.dphi_dk };
        const ResidualHistory residual_history{ target_tape.forward_residual, guess_tape.forward_residual, grad.residual };
        if (!viewer_interactive_playback(guess_obj.mesh, target_tape, guess_tape, colliders, dt, 1, FPS * frame_substeps,
                                          cfg.fd_eps_selected, run_fd_checks, grad_summary, residual_history,
                                          collider_animations, guess_obj.pin_local_offset, guess_obj.waist_attach_anim_id))
            break; // window closed; "Back to Setup" falls through and loops back to the config screen
    }

    } // while (viewer_show_config_screen(cfg))

    viewer_close();
    return 0;
}