#pragma once

// The interactive pipeline: config screen -> target sim -> guess sim -> backward pass (+ FD check) -> playback.

#include "app/fd_check.h"
#include "driver/settle.h"
#include "cloth/obj_output.h"
#include "diffpd_viewer.h"
#include "solver/dispatch.h"
#include "solver/impulse/backward.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

struct SimRunResult { Object obj; Tape tape; };

// Builds one cloth and forward-simulates it with a live-viewer callback — shared by the target and
// guess runs in main(), which differ only in stiffness/label/prefix/verbosity. Only the trajectory
// being computed is shown live; the other one (e.g. the target, while the guess is computing) is not
// overlaid — see viewer_interactive_playback() for the target-vs-guess side-by-side comparison after
// both runs finish.
inline SimRunResult run_forward_simulation(
    AppConfig& cfg, const Stiffness& stiffness,
    Real dt, int n_steps,
    WorldSetup& world,
    const std::string& prefix, const std::string& label, bool verbose,
    bool& aborted)
{
    SimRunResult result;
    result.obj = build_run_cloth(cfg, stiffness);
    construct_pd_lhs(result.obj, dt);
    attach_waist(result.obj, world);

    // Contact has no push-out: a cloth that starts inside the body stays stuck there.
    {
        int free_inside = 0, pinned_inside = 0;
        count_cloth_inside_skinned(result.obj, world.collision, free_inside, pinned_inside);
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
        if (step % cfg.solver.frame_substeps != 0) return true;
        std::ostringstream oss;
        oss << label << "   step " << step << "/" << n;
        viewer_set_scene(obj.mesh, nullptr, nullptr, &tape.positions.back(), &tape.contacts.back(),
                          world.collision, (step + 1) * dt, oss.str());
        if (!viewer_render_frame()) { aborted = true; return false; }
        return true;
    };
    simulate_forward(obj, world.collision, cfg.solver, n_steps, tape, prefix, /*export_obj=*/false, verbose, on_step, cfg.world.unresolved_contact_threshold);
    return result;
}

// The whole program after the check dispatch in main(); returns main's exit code.
inline int run_app()
{
    ANIM_DIR = ANIM_DIR_DEFAULT;
    clear_folder(ANIM_DIR);

    viewer_open();

    AppConfig cfg; // declared outside the loop so edits survive a "Back to Setup" restart

    // cfg.world.waist_attach_enabled defaults to true, so seed the origin here, once, from the default
    // animation's hip before the config screen is ever shown; the screen's own handlers only snap
    // on a false->true toggle or an animation change and wouldn't otherwise fire for a value that
    // starts true.
    if (cfg.world.waist_attach_enabled)
    {
        const AnimationPreview initial = load_animation_preview(cfg.world.animation_file);
        if (initial.has_waist && initial.waist_snaps_origin) cfg.cloth.origin = initial.waist_default_origin;
    }

    while (viewer_show_config_screen(cfg, [](AppConfig& c, bool seams_only) { return settle_cloth(c, false, seams_only); }))
    {

    bool aborted = false;

    // cloth parameters
    const Stiffness stiffness        = cfg.guess_k();
    const Stiffness target_stiffness = cfg.target_k();

    // world parameters: colliders, animation tracks, skinned bodies, waist attachment target
    WorldSetup world = prepare_world(cfg);

    // physics parameters
    const Vec3 gravity = cfg.solver.gravity;

    // simulation parameters
    const int FPS            = cfg.solver.FPS;
    const int frame_substeps = cfg.solver.frame_substeps;
    const int secs           = cfg.solver.secs;

    // solver parameters
    const int n_iters         = cfg.solver.n_iters;         // forward PD global-local iterations per step
    const int n_iters_adjoint = cfg.solver.n_iters_adjoint; // backward adjoint-vector iterations per step
    const int substeps = FPS * frame_substeps;
    const Real dt      = 1.0 / substeps;
    const int  n_steps = substeps * secs;

    // Forward-only mode: one simulation with the guess parameters, then straight to playback. No target,
    // loss, adjoint or FD check.
    if (!cfg.backward_enabled())
    {
        world.collision.collect_backward = false; // nothing differentiates this run
        stage_timer.reset();
        stage_timer.enabled = true;
        SimRunResult run = run_forward_simulation(
            cfg, stiffness, dt, n_steps,
            world, "forward", "Forward simulation",
            /*verbose=*/false, aborted);
        stage_timer.enabled = false;
        if (aborted) break;
        const std::vector<StageShare> forward_timing = stage_timer.shares("loop", "viewer_callback");

        const ResidualHistory residual_history{ {}, run.tape.forward_residual, {} };
        if (!viewer_interactive_playback(run.obj.mesh, run.tape, /*backward=*/nullptr, world.collision, dt, 1, FPS * frame_substeps,
                                          residual_history, forward_timing, run.obj.waist))
            break; // window closed; "Back to Setup" loops back to the config screen
        continue;
    }

    Tape target_tape = run_forward_simulation(
        cfg, target_stiffness, dt, n_steps,
        world, "target", "Target simulation",
        /*verbose=*/true, aborted).tape;

    if (aborted) break;

    const bool run_fd_check = cfg.run_fd_check;

    { // guess sim + backward pass + playback (Guess + Target + Backward mode)
        stage_timer.reset();
        stage_timer.enabled = true; // time only the guess run (the target's breakdown would be similar)
        SimRunResult guess = run_forward_simulation(
            cfg, stiffness, dt, n_steps,
            world, "guess", "Guess simulation",
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
                              world.collision, t * dt, oss.str());
            if (!viewer_render_frame()) { aborted = true; return false; }
            return true;
        };
        const BackwardGradContact grad = backward_pd_contact(guess_obj, world.collision, guess_tape, loss, gravity, n_iters_adjoint, dt, backward_on_step, /*verbose=*/true);

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
            if (cfg.cloth.shear_on())   dphi_dk_lines.push_back({ "dphi/dk_shear", grad.dphi_dk.shear });
            if (cfg.cloth.bending_on() || cfg.cloth.use_arap()) dphi_dk_lines.push_back({ "dphi/dk_bend",  grad.dphi_dk.bend }); // ARAP: hinge
        }
        for (const auto& [name, value] : dphi_dk_lines)
            std::cout << name << " = " << value << "\n";
        if (cfg.solver.damping_enabled)
            std::cout << "damping (alpha=" << cfg.solver.damping_alpha << ", beta=" << cfg.solver.damping_beta << ")"
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
                attach_waist(obj, world);
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
                    world.collision, build_guess_k, target_tape, stiffness.stretch, dt, gravity, n_iters, n_steps, frame_substeps,
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
        if (!viewer_interactive_playback(guess_obj.mesh, guess_tape, &backward_playback, world.collision, dt, 1, FPS * frame_substeps,
                                          residual_history, forward_timing, guess_obj.waist))
            break; // window closed; "Back to Setup" falls through and loops back to the config screen
    }

    } // while (viewer_show_config_screen(cfg))

    viewer_close();
    return 0;
}
