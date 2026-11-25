#pragma once

// The settle phases (SettlePhase): settle_seams, then settle_cloth, relax the cloth against frozen colliders and store the poses in
// cfg.sewn / cfg.settled; settle_all runs them in order. settle_pose then moves the body to the start of the motion (cfg.posed).

#include "app/world_setup.h"
#include "diffpd_viewer.h"
#include "solver/impulse/forward.h"

#include <iostream>
#include <random>
#include <sstream>

// Waist snap of a skirt's pins onto skinned body `skinned_id` (state 0): from each pin a segment runs horizontally to the
// axis of the top circle (`center`, at the pin's height) and the pin moves to its first hit on the body, `offset` back
// toward where it was. A pin whose segment misses (including a pin already inside the body) warns and stays.
// Brute force over the triangles. Returns the new pinned positions (indexed like mesh.pinned_rest).
inline std::vector<Vec3> snap_pins_to_skinned_body(CollisionWorld& world, const Object& obj, int skinned_id, const Vec3& center, Real offset)
{
    world.stamp_skinned_body(skinned_id, 1, 0);
    const std::vector<Vec3>& V = world.skinned_states[skinned_id].x;
    const auto& T = world.skinned_meshes[skinned_id].rest.triangles;

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

// Moves each free particle inside a collider (poses as currently stamped, time 0; with gap contact inside the offset shell of
// thickness t) out to `t + margin` from the surface, so a run starts collision free (contact has no push-out).
// Every wall a particle is found inside (closest point p, normal n) is kept for that particle, and each pass the particle moves by
// the average of the corrections its walls still ask for, max(0, t + margin - n.(x - p)) n. One wall: straight out, as before.
// Two facing walls closer than 2(t + margin) (inner thighs): no position clears both, and the average settles on the midpoint
// instead of bouncing from one wall into the other. A wall found again (normal within ~25 deg) replaces the old ones, so a
// surface is not counted twice. Returns how many particles are still inside after `max_passes`; `moved` = how many were inside
// at the start. Pinned vertices are skipped.
inline int project_out_of_colliders(const CollisionWorld& world, Object& obj, int& moved, Real margin = 1e-4, int max_passes = 20)
{
    const Real t = world.contact.gap() ? world.contact.thickness : 0.0; // gap contact: out of the offset shell too, else step 1 pushes them out at t/h
    std::vector<std::vector<std::pair<Vec3, Vec3>>> walls(obj.num_particles()); // per particle: (closest point, normal)
    Contacts inside = detect_contacts(world, obj.x, 0.0, -t);
    moved = (int)inside.size();
    for (int pass = 0; pass < max_passes && !inside.empty(); ++pass)
    {
        for (const Contact& c : inside)
        {
            auto& w = walls[c.particle];
            w.erase(std::remove_if(w.begin(), w.end(), [&](const auto& pn) { return pn.second.dot(c.normal) > 0.9; }), w.end());
            w.push_back({ c.surface_point, c.normal });

            auto x = obj.x.segment<3>(3 * c.particle);
            Vec3 dx = Vec3::Zero();
            for (const auto& [p, n] : w) dx += std::max<Real>(0.0, t + margin - n.dot(x - p)) * n;
            x += dx / (Real)w.size();
        }
        inside = detect_contacts(world, obj.x, 0.0, -t);
    }
    return (int)inside.size();
}

// One phase: simulates the cloth with every collider frozen at frame 0 (pd_contact `frozen`) with cfg.settle_k(), dt = 1/60/settle_substeps
// and settle_iters iterations, then projects free particles out of the colliders. The result is the start pose of later runs (see
// build_run_cloth), for target and guess alike.
// `seams` (settle_seams): from the procedural mesh, no gravity, cfg.seam_settle_seconds per stage, one stage per seam
// level 1, 2, ...: stage L keeps the links of level <= L, the earlier ones holding at rest, and the links of level L start at their
// current offset and close smoothly to 0 (smoothstep, so no velocity is left), instead of snapping. Every stage starts at rest from the
// previous pose. Otherwise (settle_cloth): cfg.settle_seconds with gravity and cfg.settle_friction, from the sewn pose (the procedural mesh without seams).
// Returns false if the window was closed. `headless` (--fd-config --settle): no live view.
inline bool settle_phase(AppConfig& cfg, bool headless, bool seams)
{
    const int      substeps   = std::max(cfg.settle_substeps, 1);
    const Real     dt         = 1.0 / (60.0 * substeps);
    const int      n_iters    = std::max(cfg.settle_iters, 1);
    const int      max_steps  = std::max(1, (int)std::lround((seams ? cfg.seam_settle_seconds : cfg.settle_seconds) / dt));
    const Vec3     gravity    = seams ? Vec3::Zero() : cfg.solver.gravity;

    WorldSetup world = prepare_world(cfg);
    world.collision.contact.friction = seams ? 0.0 : cfg.settle_friction;
    bool aborted = false;

    SettledStart result;
    auto settle_one = [&](const Stiffness& stiffness, const char* label) -> RealVecX
    {
        Object obj = build_cloth(cfg, stiffness, cfg.cloth.origin); // always from the procedural mesh
        if (cfg.cloth.cloth_type == ClothType::Skirt && !world.collision.skinned_meshes.empty())  // shortcut: first skinned body only
        {
            result.pins = snap_pins_to_skinned_body(world.collision, obj, 0, cfg.cloth.origin, 1e-2);
            set_pinned_positions(obj, result.pins);
        }
        construct_pd_lhs(obj, dt);
        attach_waist(obj, world);
        if (cfg.settle_jitter > 0.0) { std::mt19937_64 rng(12345); for (Index i = 2; i < obj.x.size(); i += 3) obj.x[i] += cfg.settle_jitter * (2.0 * (rng() >> 11) * (1.0 / 9007199254740992.0) - 1.0); } // rest lengths stay baked from the flat mesh
        if (seams) obj.energies.bending.rest_from(obj.x, obj.mesh.pinned_rest); // keep the bent start (deformed pieces) instead of pulling it flat; the real run rebuilds with its own rest
        if (!seams && has_seams(cfg.cloth))
        {
            ASSERT(cfg.sewn.valid && cfg.sewn.x.size() == obj.x.size(), "settle_cloth: this cloth has seams, run settle_seams first");
            obj.x = cfg.sewn.x;
        }

        Real max_speed = 0.0;
        int  steps = 0;
        const int n_stages = seams ? std::max(1, obj.energies.seams.highest_level()) : 1;
        if (!seams) // the pattern's pulls (f = m a on the pulled vertices): fade 0 -> 1 -> 0 over the whole settle, set from the step callback
            for (const auto& [v, dir] : obj.mesh.pulls)
            {
                const ParticleId d = obj.mesh.vertices[v].dof;
                if (d >= 0) obj.energies.pulls.add_pull(d, obj.mass(3 * d) * cfg.pull_accel * dir);
            }
        for (int stage = 1; stage <= n_stages && !aborted; ++stage)
        {
        if (seams) // this stage's links: levels <= stage in K, the links of level `stage` pulled in from where they are
        {
            obj.energies.seams.max_level  = stage;
            obj.energies.seams.pull_level = stage;
            obj.energies.seams.pull       = 1.0;
            obj.energies.seams.set_offsets(obj.x);
            obj.v.setZero();
            construct_pd_lhs(obj, dt); // factor again with this stage's links
        }
        Tape tape;
        const StepCallback on_step = [&](int step, int n) -> bool
        {
            max_speed = tape.velocities.back().rowwise().norm().maxCoeff();
            const Real u = std::min<Real>(1.0, (step + 1.0) / n);
            if (seams) // the next step wants this fraction of the initial offset
                obj.energies.seams.pull = 1.0 - u * u * (3.0 - 2.0 * u);
            else
                obj.energies.pulls.ramp = std::sin(3.14159265358979323846 * u);
            if (headless) return true;
            if (!viewer_poll_close()) { aborted = true; return false; }
            if (step % (2 * substeps) == 0)
            {
                std::ostringstream oss;
                oss << label << "   step " << step << "/" << n;
                if (seams) oss << "   level " << stage << "/" << n_stages << "   longest link " << obj.energies.seams.max_gap(obj.x, stage) << " m";
                else            oss << "   max speed " << max_speed << " m/s";
                viewer_set_scene(obj.mesh, nullptr, nullptr, &tape.positions.back(), &tape.contacts.back(),
                                 world.collision, 0.0, oss.str());
                if (!viewer_render_frame()) { aborted = true; return false; }
            }
            return true;
        };
        pd_contact(obj, world.collision, dt, gravity, n_iters, max_steps, 1, tape, "settle", 
                   /*export_obj=*/false, 
                   /*verbose=*/false,
                   on_step, 0.0, 
                   /*frozen=*/true);
        steps += (int)tape.positions.size() - 1;
        }
        int moved = 0;
        const int still_inside = project_out_of_colliders(world.collision, obj, moved); // start collision free
        std::cout << "[settle] " << label << ": " << steps << " steps, max speed " << max_speed << " m/s, projected "
                  << moved << " particles out of the colliders\n";
        if (still_inside > 0) WARNING(label << ": " << still_inside << " particles still penetrate a collider after projection");
        result.max_speed  = max_speed;
        result.steps      = steps;
        result.seam_gap   = obj.energies.seams.links.empty() ? -1.0 : obj.energies.seams.max_gap(obj.x);
        result.seam_levels = obj.energies.seams.highest_level();
        return obj.x;
    };

    RealVecX x = settle_one(cfg.settle_k(), seams ? "Settling seams" : "Settling cloth");
    if (aborted) return false;
    result.valid = true;
    result.key   = seams ? settle_seam_key(cfg) : settle_key(cfg);
    result.x     = std::move(x);
    if (seams) cfg.settled = SettledStart{}; // it started from the previous sewn pose
    cfg.posed = SettledStart{};
    (seams ? cfg.sewn : cfg.settled) = std::move(result);
    return true;
}

inline bool settle_seams(AppConfig& cfg, bool headless = false) { return settle_phase(cfg, headless, true); }
inline bool settle_cloth(AppConfig& cfg, bool headless = false) { return settle_phase(cfg, headless, false); }

// Every phase at frame 0 in order, from the procedural mesh.
inline bool settle_all(AppConfig& cfg, bool headless = false)
{
    cfg.sewn = cfg.settled = cfg.posed = SettledStart{};
    return (!has_seams(cfg.cloth) || settle_seams(cfg, headless)) && settle_cloth(cfg, headless);
}

// "Settle pose" (an animation with SkinnedMesh::clip_start > 0, else nothing to do): from the settled cloth at rest, plays the frames
// 0 .. clip_start with the body moving, gravity and cfg.settle_friction, at the shared settle substeps / iterations. Keeps x AND v, and runs
// then start at frame clip_start (AppConfig::first_step). No push-out at the end: it would leave v inconsistent with x.
// Returns false if the window was closed. Always the impulse method.
inline bool settle_pose(AppConfig& cfg, bool headless = false)
{
    WorldSetup world = prepare_world(cfg);
    CollisionWorld& cw = world.collision;
    const int clip_start = cw.skinned_meshes.empty() ? 0 : cw.skinned_meshes[0].clip_start; // shortcut: first skinned body only
    if (clip_start == 0) return true;
    ASSERT(cfg.settled.valid, "settle_pose: run settle_cloth first");
    cfg.posed = SettledStart{};

    anderson_m = 0; // as in the other phases; prepare_world sets it again for the run
    cw.collect_backward = false;
    cw.contact.friction = cfg.settle_friction;
    const int  substeps = std::max(cfg.settle_substeps, 1);
    const Real dt       = 1.0 / (kColliderAnimFPS * substeps);
    Object obj = build_run_cloth(cfg, cfg.settle_k()); // from cfg.settled, the last finished phase
    construct_pd_lhs(obj, dt);
    attach_waist(obj, world);

    bool aborted = false;
    Tape tape;
    const StepCallback on_step = [&](int step, int n) -> bool
    {
        if (headless) return true;
        if (!viewer_poll_close()) { aborted = true; return false; }
        if (step % substeps != 0) return true;
        std::ostringstream oss;
        oss << "Settling pose   step " << step << "/" << n;
        viewer_set_scene(obj.mesh, nullptr, nullptr, &tape.positions.back(), &tape.contacts.back(), cw, tape.clock.end_state(step) * dt, oss.str());
        if (!viewer_render_frame()) { aborted = true; return false; }
        return true;
    };
    pd_contact(obj, cw, dt, cfg.solver.gravity, std::max(cfg.settle_iters, 1), clip_start * substeps, substeps, tape, "settle",
               /*export_obj=*/false, /*verbose=*/false, on_step, /*unresolved_penetration_threshold=*/-1.0);
    if (aborted) return false;

    SettledStart result = cfg.settled; // keeps the frame-0 pins: a run bakes the waist attachment from them
    result.key         = settle_key(cfg);
    result.x           = obj.x;
    result.v           = obj.v;
    result.first_frame = clip_start;
    result.steps       = (int)tape.positions.size() - 1;
    result.max_speed   = tape.velocities.back().rowwise().norm().maxCoeff();
    cfg.posed = std::move(result);
    std::cout << "[settle] Settling pose: " << cfg.posed.steps << " steps to frame " << clip_start << ", max speed " << cfg.posed.max_speed << " m/s\n";
    return true;
}

inline bool run_settle(AppConfig& cfg, SettlePhase phase)
{
    switch (phase)
    {
    case SettlePhase::Seams: return settle_seams(cfg);
    case SettlePhase::Cloth: return settle_cloth(cfg);
    case SettlePhase::Pose:  return settle_pose(cfg);
    case SettlePhase::All:   return settle_all(cfg) && settle_pose(cfg);
    }
    return true;
}
