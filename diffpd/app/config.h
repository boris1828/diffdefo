#pragma once

// AppConfig: every knob of a run, edited live by the config screen and saved by config_io.h, plus the glue that turns
// a config into a cloth (tri_surface_for, build_cloth, build_run_cloth) and the settled-pose key. This is the app
// layer: the modules below (cloth/, collision/, solver/) never include it.

#include "cloth/build.h"
#include "collision/world.h"
#include "meshgen/pattern_io.h"

#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

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
    int         seam_levels = 0;   // seam levels closed (settle seams: one stage each)
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
    // disconnected pieces; no pinned vertices, no sewing. The file is in metres in the body's world (version 7), so
    // pattern_size is a plain multiplier about the world origin (1 = as drawn) and cfg.origin is not used for it (build_cloth).
    std::string pattern_file  = "tshirt.json";
    Real        pattern_size  = 1.0;
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
    bool tri_ok() const; // defined below, after tri_surface_for
    // Shear/bend springs exist on the quad grid only.
    bool shear_on()   const { return flag_shear   && !use_tri(); }
    bool bending_on() const { return flag_bending && !use_tri(); }

    // UI-managed list; starts empty, "+"/"-" grows/shrinks it (empty disables contact against
    // manually-placed colliders — animated colliders loaded from Blender are separate and unaffected).
    std::vector<Collider> colliders = {};

    // Contact model, margin and recheck (re-test the current iterate inside pd_contact's solver loop and add newly
    // penetrating particles to the step's contact set, see merge_detected_contacts in collision/detect.h).
    ContactOptions contact;

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
    Real pull_accel          = 2.0;  // "Settle cloth" (Pattern with pulls): the pattern's pulls accelerate their vertices by this [m/s^2] (force = vertex mass * pull_accel), fading 0 -> 1 -> 0 over the settle time

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
      << (cfg.cloth_type == ClothType::Pattern ? cfg.pattern_file + pattern_stamp(cfg.pattern_file) : std::string()) << ',' << cfg.pattern_size << ';'
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
            std::vector<meshgen::PatternPull> pulls;
            const std::vector<meshgen::PatternShape> shapes = load_pattern(pattern_path(cfg.pattern_file), &seams, &pulls);
            std::vector<meshgen::TriSurface> pieces;
            std::vector<meshgen::Report> reports;
            std::vector<meshgen::PatternChart> charts;
            for (const meshgen::PatternShape& shape : shapes) charts.emplace_back(shape, cfg.pattern_size, p.h / 50);
            meshgen::plan_seam_intervals(charts, seams, p.h); // both sides of a seam get the same number of vertices
            for (const meshgen::PatternChart& chart : charts)
            {
                pieces.push_back(meshgen::generate(chart, p));
                reports.push_back(meshgen::validate(pieces.back(), chart, p));
            }
            cache.surface = meshgen::merge(pieces);
            cache.surface.seams = meshgen::sew(pieces, seams, &cache.surface.seam_level);
            cache.surface.pulls = meshgen::pull_vertices(pieces, pulls);
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

inline bool AppConfig::tri_ok() const
{
    if (!use_tri()) return true;
    if (cloth_type == ClothType::Pattern) return tri_surface_for(*this).error.empty();
    if (cloth_type == ClothType::Square) return std::min(size_x, size_y) >= 2.5 * tri_edge();
    const Real r_min = std::min(radius_top, radius_bottom);
    return std::abs(radius_top - radius_bottom) >= 1e-3 && r_min >= 0.02 && skirt_height >= 0.02
           && tri_h <= 2.0 * 3.14159265358979 * r_min / 6.0;
}

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
    if (cfg.cloth_type == ClothType::Pattern) origin = Vec3::Zero(); // the file is already placed in the body's world
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
    if (c.cloth_type == ClothType::Pattern) s << c.pattern_file << pattern_stamp(c.pattern_file) << ',' << c.pattern_size << ',' << c.seam_stiffness << ';';
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
    s << (int)c.contact.recheck << ',' << c.contact.recheck_count << ',' << (int)c.contact.model << ',' << c.contact.margin << ';';
    s << c.settle_seconds << ',' << c.settle_rest_speed << ',' << c.settle_iters << ',' << c.seam_settle_seconds << ',' << c.seam_settle_iters << ',' << c.pull_accel;
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
