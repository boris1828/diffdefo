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
    c.rigid_pose.position = Vec3(1.0, -1.0, 1.0); // Arc centre, next to the sphere's
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

// The settle phases, in order (driver/settle.h): Seams (only a cloth with seams), then Cloth, which starts from the sewn
// pose, then Pose (only an annotated animation), which plays the frames that lead into the motion. All = every phase from the procedural mesh. Each phase keeps its own result, so it can be redone from the one before.
enum class SettlePhase { Seams, Cloth, Pose, All };

// The result of one settle phase: the cloth simulated with every collider frozen at its frame-0 pose (Pose: moving).
// The last finished phase gives the initial positions x0 of the real runs (AppConfig::start_pose, build_run_cloth).
// Valid only for the config it was computed from: `key` is settle_seam_key() / settle_key() / settle_pose_key() of that config, and the
// config screen drops the result the moment the key no longer matches.
struct SettledStart
{
    bool        valid = false;
    std::string key;
    RealVecX    x;                 // free-particle positions
    RealVecX    v;                 // their velocities (empty = at rest; only the Pose phase ends moving)
    int         first_frame = 0;   // animation frame the run starts at (Pose: the clip start)
    std::vector<Vec3> pins;        // pinned-vertex positions after the waist snap (empty = procedural ones)
    Real        max_speed = 0.0;   // fastest particle [m/s] at the end
    Real        seam_gap  = -1.0;  // longest sewn link when it stopped [m] (-1 = no seams)
    int         seam_levels = 0;   // seam levels closed (settle seams: one stage each)
    int         steps     = 0;     // steps it ran
};

// Every knob main() used to hardcode, now editable live in the pre-run config screen: one struct per module
// (cloth, world, solver) plus the stiffness values, the settle settings and the run / FD options.
struct AppConfig
{
    RunMode run_mode = RunMode::GuessTargetBackward;
    // Why the chosen contact method cannot run this config ("" = it can); the config screen shows it and disables Run.
    const char* method_error() const
    {
        if (cloth.m_tot <= 0.0) return "Total Mass must be > 0";
        if (solver.contact_method != ContactMethod::Ncp) return ""; // impulse method: both contact models, with or without friction, have a backward pass
        if (world.contact.gap()) return "NCP needs Rel. Velocity contact (Collision section)";
        if (solver.damping_enabled) return "NCP does not support damping (Damping section)";
        return "";
    }
    bool    backward_enabled() const { return run_mode == RunMode::GuessTargetBackward && solver.contact_method == ContactMethod::Impulse; } // NCP is forward only

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
    Real      arap_bend_stiffness        = 0.01; // hinge bending between neighbouring triangles (Bend slot), Tri meshes (springs or ARAP)
    Real      target_arap_bend_stiffness = 0.02;

    // Tri mesh: {ARAP k or spring k (= stiffness_per_type.stretch), 0, hinge bend k}; per_type_stiffness is ignored.
    Stiffness guess_k()  const { return cloth.use_tri() ? Stiffness{ cloth.use_arap() ? arap_stiffness : stiffness_per_type.stretch, 0.0, arap_bend_stiffness }                : per_type_stiffness ? stiffness_per_type        : Stiffness::uniform(stiffness); }
    Stiffness target_k() const { return cloth.use_tri() ? Stiffness{ cloth.use_arap() ? target_arap_stiffness : target_stiffness_per_type.stretch, 0.0, target_arap_bend_stiffness } : per_type_stiffness ? target_stiffness_per_type : Stiffness::uniform(target_stiffness); }
    // Whether k is one value per slot (report dphi/dk_stretch, FD-check the Stretch slot alone) or one shared k (report the
    // sum, FD-perturb every slot together). A Tri mesh always has its own stretch k and Bend k, whatever per_type_stiffness says.
    bool per_type_k() const { return per_type_stiffness || cloth.use_tri(); }
    // The stiffness the cloth is settled with, for target and guess alike (the real start pose is data, not a function of
    // the guess): the target's, or the only one shown in forward-only mode.
    Stiffness settle_k() const { return backward_enabled() ? target_k() : guess_k(); }

    // output
    bool record_on_run = false; // start the screen recorder automatically when Run is clicked
    bool export_anim = false;   // write <export_name>.obj (first frame) + .pc2 (animation) of the guess / forward run, for Blender
    std::string export_name = "output";

    // Settle settings (driver/settle.h), part of the keys. dt = 1/60/settle_substeps; substeps and iterations are shared by every phase.
    int  settle_substeps     = 1;
    int  settle_iters        = 20;   // PD iterations per step
    Real seam_settle_seconds = 2.0;  // "Settle seams": the links of each seam level close over this time, no gravity [s]
    Real settle_seconds      = 10.0; // "Settle cloth": duration [s]
    Real settle_jitter     = 0.0;   // [m] random z offset of every particle at the start of the settle: a planar cloth in a planar scene stays flat for ever (and is an unstable rest pose that collapses on the first z noise); not saved in .conf
    Real pull_accel          = 2.0;  // "Settle cloth" (Pattern with pulls): the pattern's pulls accelerate their vertices by this [m/s^2] (force = vertex mass * pull_accel), fading 0 -> 1 -> 0 over the settle time

    // Results of the settle phases (see SettledStart). Their validity is tied to settle_seam_key() / settle_key(): any field
    // a phase depends on must be listed there, or editing it would leave a stale pose active.
    SettledStart sewn;    // "Settle seams"
    SettledStart settled; // "Settle cloth"
    SettledStart posed;   // "Settle pose"
    const SettledStart* start_pose() const { return posed.valid ? &posed : settled.valid ? &settled : sewn.valid ? &sewn : nullptr; } // the last finished phase
    int  start_frame() const { const SettledStart* s = start_pose(); return s ? s->first_frame : 0; } // animation frame a run starts at
    int  first_step() const { return start_frame() * solver.frame_substeps; }                        // the same as a physics step of the run
    Real run_friction() const { return solver.contact_method == ContactMethod::Ncp ? solver.ncp.friction : world.contact.friction; }

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
    obj.global_solve = cfg.solver.global_solve;
    return obj;
}

// True for a cloth that must be sewn ("Settle seams") before it can be settled.
inline bool has_seams(const ClothSpec& c)
{
    return c.cloth_type == ClothType::Pattern && !tri_surface_for(c).surface.seams.empty();
}

// Everything the sewn pose depends on, serialized: cloth shape/mass/origin, stiffness, pins, colliders, animation + waist
// attachment, damping, contact model, the shared settle settings and the sew time. Deliberately NOT in it: FPS, the run's substeps,
// seconds and iteration counts, run mode, FD check and recording options (the settle has its own settings).
inline std::string settle_seam_key(const AppConfig& c)
{
    std::ostringstream s;
    s << std::setprecision(17);
    auto vec = [&](const Vec3& v) { s << v.x() << ',' << v.y() << ',' << v.z() << ';'; };
    s << (int)c.cloth.cloth_type << ';';
    if (c.cloth.cloth_type == ClothType::Pattern) s << c.cloth.pattern_file << pattern_stamp(c.cloth.pattern_file) << ',' << c.cloth.pattern_size << ',' << c.cloth.seam_stiffness << ';';
    if (c.cloth.cloth_type == ClothType::Pattern && !c.cloth.pattern_offset.isZero(0.0)) vec(c.cloth.pattern_offset); // zero adds nothing: keys saved before the field stay valid
    vec(c.cloth.origin);
    s << c.cloth.m_tot << ';' << c.cloth.shear_on() << c.cloth.bending_on() << c.per_type_stiffness << ';';
    const Stiffness k = c.settle_k();
    s << k.stretch << ',' << k.shear << ',' << k.bend << ';';
    s << c.cloth.width << ',' << c.cloth.height << ',' << c.cloth.size_x << ',' << c.cloth.size_y << ',' << (int)c.cloth.pin_mode << ',' << (int)c.cloth.hang_mode << ',' << c.cloth.rail_stride << ',' << c.cloth.rail_k << ',' << c.cloth.rail_spacer_k << ';';
    s << c.cloth.particles_per_ring << ',' << c.cloth.num_rings << ',' << c.cloth.radius_top << ',' << c.cloth.radius_bottom << ',' << c.cloth.skirt_height << ';';
    s << (int)c.cloth.mesh_gen << ',' << c.cloth.tri_h << ',' << c.cloth.tri_beta << ',' << c.cloth.tri_smooth_iters << ',' << c.cloth.tri_seed << ',' << (int)c.cloth.tri_energy << ',' << c.arap_stiffness << ',' << c.target_arap_stiffness << ',' << c.arap_bend_stiffness << ',' << c.target_arap_bend_stiffness << ',' << c.cloth.arap_flat_rest << ';';
    for (const Collider& k : c.world.colliders)
    {
        s << (int)k.type << ':' << k.sphere_radius << ',' << k.cylinder_radius << ',' << k.capsule_radius << ',' << k.omega << ','
          << (int)k.rotation_axis << ',' << k.animated << ':';
        vec(k.sphere_center); vec(k.cylinder_origin); vec(k.cylinder_axis); vec(k.plane_origin); vec(k.plane_normal);
        vec(k.capsule_p0); vec(k.capsule_p1); vec(k.velocity); vec(k.rotation_origin);
        s << k.arc_radius << ',' << k.tube_radius << ',' << k.arc_angle << ';';
        vec(k.rigid_pose.position); vec(Vec3(k.rigid_pose.rotation.vec())); s << k.rigid_pose.rotation.w() << ';';
    }
    s << '|' << c.world.animation_file << '|' << c.world.waist_attach_enabled << ';';
    s << c.solver.damping_enabled << ',' << c.solver.damping_alpha << ',' << c.solver.damping_beta << ';';
    s << (int)c.world.contact.recheck << ',' << c.world.contact.recheck_count << ',' << (int)c.world.contact.model << ',' << c.world.contact.margin << ',' << c.world.contact.thickness << ';';
    s << c.settle_substeps << ',' << c.settle_iters << ',' << c.settle_jitter << ',' << c.seam_settle_seconds;
    return s.str();
}

// The settled cloth's key: the sewn pose it starts from, plus what only "Settle cloth" uses.
inline std::string settle_key(const AppConfig& c)
{
    std::ostringstream s;
    s << std::setprecision(17) << settle_seam_key(c) << '|' << c.solver.gravity.x() << ',' << c.solver.gravity.y() << ',' << c.solver.gravity.z() << ';'
      << c.settle_seconds << ',' << c.pull_accel;
    return s.str();
}

// The posed cloth's key: the settled pose it starts from, plus the run's friction (the only settle phase that has it).
inline std::string settle_pose_key(const AppConfig& c)
{
    std::ostringstream s;
    s << std::setprecision(17) << settle_key(c) << '|' << c.run_friction();
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
    if (const SettledStart* start = cfg.start_pose())
    {
        ASSERT(start->x.size() == obj.x.size(), "settled pose has " << start->x.size() << " values, the cloth " << obj.x.size());
        obj.x = start->x;
        if (start->v.size() > 0) obj.v = start->v;
        if (!start->pins.empty()) set_pinned_positions(obj, start->pins);
    }
    return obj;
}
