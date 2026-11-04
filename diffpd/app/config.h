#pragma once

// AppConfig: every knob of a run, edited live by the config screen and saved by config_io.h, plus the glue that turns
// a config into a cloth (build_cloth, build_run_cloth) and the settled-pose key. This is the app
// layer: the modules below (cloth/, collision/, solver/) never include it.

#include "cloth/build.h"
#include "collision/world_spec.h"
#include "solver/settings.h"

#include <iomanip>
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

// Every knob main() used to hardcode, now editable live in the pre-run config screen: one struct per module
// (cloth, world, solver) plus the stiffness values, the settle settings and the run / FD options.
struct AppConfig
{
    RunMode run_mode = RunMode::GuessTargetBackward;
    bool    backward_enabled() const { return run_mode == RunMode::GuessTargetBackward; }

    ClothSpec      cloth;
    WorldSpec      world;
    SolverSettings solver;

    // stiffness: constraint types (cloth.flag_*) and mode are shared by target and guess; only the values differ.
    // Unified = one k for every spring (single dphi/dk); per-type = one k per SpringKind.
    bool      per_type_stiffness        = true;
    Real      stiffness                 = 1.0;   // unified
    Real      target_stiffness          = 2.0;   // unified
    Stiffness stiffness_per_type        = { 2.0, 0.5, 0.3 };  // stretch, shear, bend
    Stiffness target_stiffness_per_type = { 4.0, 1.0, 0.6 };
    Real      arap_stiffness        = 4.0; // ARAP surface modulus (rides in the Stretch slot); not comparable to a spring k
    Real      target_arap_stiffness = 8.0;
    Real      arap_bend_stiffness        = 0.01; // hinge bending between neighbouring triangles (Bend slot), ARAP only
    Real      target_arap_bend_stiffness = 0.02;

    Stiffness guess_k()  const { return cloth.use_arap() ? Stiffness{ arap_stiffness, 0.0, arap_bend_stiffness }        : per_type_stiffness ? stiffness_per_type        : Stiffness::uniform(stiffness); }
    Stiffness target_k() const { return cloth.use_arap() ? Stiffness{ target_arap_stiffness, 0.0, target_arap_bend_stiffness } : per_type_stiffness ? target_stiffness_per_type : Stiffness::uniform(target_stiffness); }
    // Whether k is one value per slot (report dphi/dk_stretch, FD-check the Stretch slot alone) or one shared k (report the
    // sum, FD-perturb every slot together). ARAP always has its own ARAP k and Bend k, whatever per_type_stiffness says.
    bool per_type_k() const { return per_type_stiffness || cloth.use_arap(); }
    // The stiffness the cloth is settled with, for target and guess alike (the real start pose is data, not a function of
    // the guess): the target's, or the only one shown in forward-only mode.
    Stiffness settle_k() const { return backward_enabled() ? target_k() : guess_k(); }

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

// build_cloth(ClothSpec, ...) plus the solver's damping: the config's way to a cloth Object.
inline Object build_cloth(const AppConfig& cfg, const Stiffness& stiffness, Vec3 origin)
{
    Object obj = build_cloth(cfg.cloth, stiffness, origin);
    obj.damp_alpha = cfg.solver.damping_enabled ? cfg.solver.damping_alpha : 0.0;
    obj.damp_beta  = cfg.solver.damping_enabled ? cfg.solver.damping_beta  : 0.0;
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
    s << (int)c.cloth.cloth_type << ';';
    if (c.cloth.cloth_type == ClothType::Pattern) s << c.cloth.pattern_file << pattern_stamp(c.cloth.pattern_file) << ',' << c.cloth.pattern_size << ',' << c.cloth.seam_stiffness << ';';
    vec(c.cloth.origin);
    s << c.cloth.m_tot << ';' << c.cloth.shear_on() << c.cloth.bending_on() << c.per_type_stiffness << ';';
    const Stiffness k = c.settle_k();
    s << k.stretch << ',' << k.shear << ',' << k.bend << ';';
    s << c.cloth.width << ',' << c.cloth.height << ',' << c.cloth.size_x << ',' << c.cloth.size_y << ',' << (int)c.cloth.pin_mode << ',' << (int)c.cloth.hang_mode << ';';
    s << c.cloth.particles_per_ring << ',' << c.cloth.num_rings << ',' << c.cloth.radius_top << ',' << c.cloth.radius_bottom << ',' << c.cloth.skirt_height << ';';
    s << (int)c.cloth.mesh_gen << ',' << c.cloth.tri_h << ',' << c.cloth.tri_beta << ',' << c.cloth.tri_smooth_iters << ',' << c.cloth.tri_seed << ',' << (int)c.cloth.tri_energy << ',' << c.arap_stiffness << ',' << c.target_arap_stiffness << ',' << c.arap_bend_stiffness << ',' << c.target_arap_bend_stiffness << ',' << c.cloth.arap_flat_rest << ';';
    for (const Collider& k : c.world.colliders)
    {
        s << (int)k.type << ':' << k.sphere_radius << ',' << k.cylinder_radius << ',' << k.capsule_radius << ',' << k.omega << ','
          << (int)k.rotation_axis << ',' << k.animated << ':';
        vec(k.sphere_center); vec(k.cylinder_origin); vec(k.cylinder_axis); vec(k.plane_origin); vec(k.plane_normal);
        vec(k.capsule_p0); vec(k.capsule_p1); vec(k.velocity); vec(k.rotation_origin);
    }
    s << '|' << c.world.animation_file << '|' << c.world.waist_attach_enabled << ';';
    vec(c.solver.gravity);
    s << c.solver.damping_enabled << ',' << c.solver.damping_alpha << ',' << c.solver.damping_beta << ';';
    s << (int)c.world.contact.recheck << ',' << c.world.contact.recheck_count << ',' << (int)c.world.contact.model << ',' << c.world.contact.margin << ';';
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
    Object obj = build_cloth(cfg, stiffness, cfg.cloth.origin);
    if (cfg.settled.valid)
    {
        ASSERT(cfg.settled.x.size() == obj.x.size(), "settled pose has " << cfg.settled.x.size() << " values, the cloth " << obj.x.size());
        obj.x = cfg.settled.x;
        if (!cfg.settled.pins.empty()) set_pinned_positions(obj, cfg.settled.pins);
    }
    return obj;
}
