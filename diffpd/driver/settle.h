#pragma once

// "Settle cloth" / "Settle seams": relax the cloth against frozen colliders and store the pose in cfg.settled.

#include "app/world_setup.h"
#include "diffpd_viewer.h"
#include "solver/forward.h"

#include <iostream>
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

// Moves each free particle inside a collider (poses as currently stamped, time 0) to its closest surface point plus
// `margin` along the normal, so a run starts collision free (contact has no push-out). Retries up to `max_passes`
// times, doubling the margin each pass, so a particle bouncing between two colliders eventually escapes.
// shortcut: the margin reaches ~0.8 m on the last pass, cap it if stray particles ever show up far from the cloth.
// Returns how many particles still penetrate; `moved` = how many were inside at the start. Pinned vertices are skipped.
inline int project_out_of_colliders(const CollisionWorld& world, Object& obj, int& moved, Real margin = 1e-4, int max_passes = 14)
{
    moved = 0;
    for (int pass = 0; pass < max_passes; ++pass)
    {
        const Contacts penetrating = detect_contacts(world, obj.x, 0.0);
        if (pass == 0) moved = (int)penetrating.size();
        if (penetrating.empty()) return 0;
        for (const Contact& c : penetrating)
            obj.x.segment<3>(3 * c.particle) = c.surface_point + (margin * (1 << pass)) * c.normal;
    }
    return (int)detect_contacts(world, obj.x, 0.0).size();
}

// "Settle cloth" button: simulates the cloth with every collider frozen at frame 0 (pd_contact `frozen`) and stores the
// result in cfg.settled as the start pose of later runs (see build_run_cloth), for target and guess alike. It is settled
// from the procedural mesh with cfg.settle_k() at a fixed dt = 1/60 s. Stops at cfg.settle_seconds or when
// the fastest particle drops below cfg.settle_rest_speed, then projects free particles out of the colliders.
// `seams_only` ("Settle seams", Pattern files with seams): no gravity, cfg.seam_settle_seconds / _iters per stage, one stage per seam
// level 1, 2, ...: stage L keeps the links of level <= L, the earlier ones holding at rest, and the links of level L start at their
// current offset and close smoothly to 0 (smoothstep, so no velocity is left), instead of snapping. Every stage starts at rest from the
// previous pose. The result has settled.seams_only set and the next normal settle starts from it. Colliders stay frozen in both.
// Returns false if the window was closed. `headless` (--fd-config --settle): no live view.
inline bool settle_cloth(AppConfig& cfg, bool headless = false, bool seams_only = false)
{
    constexpr Real kDt = 1.0 / 60.0;
    const Real     rest_speed = seams_only ? -1.0 : std::max<Real>(cfg.settle_rest_speed, 0.0); // [m/s]; the seam stage never stops early
    const int      n_iters    = std::max(seams_only ? cfg.seam_settle_iters : cfg.settle_iters, 1);
    const int      max_steps  = std::max(1, (int)((seams_only ? cfg.seam_settle_seconds : cfg.settle_seconds) / kDt));
    const Vec3     gravity    = seams_only ? Vec3::Zero() : cfg.solver.gravity;

    WorldSetup world = prepare_world(cfg);
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
        init_pd_velocity(obj, kDt);
        attach_waist(obj, world);
        if (seams_only) obj.energies.bending.rest_from(obj.x, obj.mesh.pinned_rest); // keep the bent start (deformed pieces) instead of pulling it flat; the real run rebuilds with its own rest
        if (!seams_only && cfg.settled.valid && cfg.settled.seams_only) // continue from the sewn pose
        {
            ASSERT(cfg.settled.x.size() == obj.x.size(), "settled pose has " << cfg.settled.x.size() << " values, the cloth " << obj.x.size());
            obj.x = cfg.settled.x;
        }

        Real max_speed = 0.0;
        int  steps = 0;
        const int n_stages = seams_only ? std::max(1, obj.energies.seams.highest_level()) : 1;
        if (!seams_only) // the pattern's pulls (f = m a on the pulled vertices): fade 0 -> 1 -> 0 over the whole settle, set from the step callback
            for (const auto& [v, dir] : obj.mesh.pulls)
            {
                const ParticleId d = obj.mesh.vertices[v].dof;
                if (d >= 0) obj.energies.pulls.add_pull(d, obj.mass(3 * d) * cfg.pull_accel * dir);
            }
        for (int stage = 1; stage <= n_stages && !aborted; ++stage)
        {
        if (seams_only) // this stage's links: levels <= stage in K, the links of level `stage` pulled in from where they are
        {
            obj.energies.seams.max_level  = stage;
            obj.energies.seams.pull_level = stage;
            obj.energies.seams.pull       = 1.0;
            obj.energies.seams.set_offsets(obj.x);
            obj.v.setZero();
            init_pd_velocity(obj, kDt); // factor again with this stage's links
        }
        Tape tape;
        const StepCallback on_step = [&](int step, int n) -> bool
        {
            max_speed = tape.velocities.back().rowwise().norm().maxCoeff();
            const Real u = std::min<Real>(1.0, (step + 1.0) / n);
            if (seams_only) // the next step wants this fraction of the initial offset
                obj.energies.seams.pull = 1.0 - u * u * (3.0 - 2.0 * u);
            else
                obj.energies.pulls.ramp = std::sin(3.14159265358979323846 * u); // shortcut: a rest-speed stop (settle_rest_speed > 0) can end it before the ramp is back at 0
            if (headless) return max_speed > rest_speed;
            if (!viewer_poll_close()) { aborted = true; return false; }
            if (step % 2 == 0)
            {
                std::ostringstream oss;
                oss << label << "   step " << step << "/" << n;
                if (seams_only) oss << "   level " << stage << "/" << n_stages << "   longest link " << obj.energies.seams.max_gap(obj.x, stage) << " m";
                else            oss << "   max speed " << max_speed << " m/s";
                viewer_set_scene(obj.mesh, nullptr, nullptr, &tape.positions.back(), &tape.contacts.back(),
                                 world.collision, 0.0, oss.str());
                if (!viewer_render_frame()) { aborted = true; return false; }
            }
            return max_speed > rest_speed; // false = settled, stop
        };
        pd_contact(obj, world.collision, kDt, gravity, n_iters, max_steps, 1, tape, "settle", 
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
        result.seams_only = seams_only;
        result.seam_gap   = obj.energies.seams.links.empty() ? -1.0 : obj.energies.seams.max_gap(obj.x);
        result.seam_levels = obj.energies.seams.highest_level();
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
