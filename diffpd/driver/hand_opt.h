#pragma once

// Headless experiment driver: recover the path of a hand (one collider of the config, translation only) from the cloth it leaves behind, by
// gradient descent through the simulator. theta (K control points of a clamped cubic B-spline) -> positions -> animated track -> forward ->
// loss against a target cloth -> backward (collider adjoint) -> track gradient -> + path regulariser -> B^T -> Adam.
// Target and guess share the cloth (the settled start pose, built once with the hand at its start); only the hand path differs.
//   diffpd --hand-opt <config.conf> [key=value ...]   (keys: see HandOptArgs below; mode = opt | check | probe | settle)

#include "app/config_io.h"
#include "driver/run.h"
#include "optim/adam.h"
#include "optim/bspline.h"
#include "optim/path_regulariser.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>

struct HandOptSpec
{
    int               collider = 0;  // index in cfg.world.colliders of the hand (Sphere or Arc)
    int               K        = 7;
    Vec3              free_axes = Vec3(1.0, 1.0, 0.0);
    std::vector<Vec3> target_controls; // K control points of the target hand in diffpd axes; [0] is the start
};

// key=value command line options with defaults
struct HandOptArgs
{
    std::map<std::string, std::string> kv;
    Real num(const std::string& k, Real def) const { auto it = kv.find(k); return it == kv.end() ? def : std::stod(it->second); }
    std::string str(const std::string& k, const std::string& def) const { auto it = kv.find(k); return it == kv.end() ? def : it->second; }
};

namespace hand_opt_detail
{
// silences the per-step progress lines of pd_contact / backward_pd_contact
struct Quiet
{
    std::streambuf* old;
    std::ostringstream sink;
    Quiet() : old(std::cout.rdbuf(sink.rdbuf())) {}
    ~Quiet() { std::cout.rdbuf(old); }
};

inline void set_hand_position(Collider& c, const Vec3& p)
{
    if (c.type == ColliderType::Arc) c.rigid_pose.position = p;
    else if (c.type == ColliderType::Sphere) c.sphere_center = p;
    else ASSERT(false, "hand_opt: the hand must be a Sphere or an Arc collider");
}

// the settle is the slowest part of a run and identical for every run of one config: cache the pose on disk, keyed by settle_key
inline bool load_settled(const std::string& file, AppConfig& cfg)
{
    std::ifstream in(file, std::ios::binary);
    if (!in) return false;
    size_t n = 0; in.read((char*)&n, sizeof n);
    std::string key(n, ' '); in.read(key.data(), n);
    if (key != settle_key(cfg)) return false;
    size_t nx = 0, np = 0;
    in.read((char*)&nx, sizeof nx); cfg.settled.x.resize(nx); in.read((char*)cfg.settled.x.data(), nx * sizeof(Real));
    in.read((char*)&np, sizeof np); cfg.settled.pins.resize(np);
    for (Vec3& p : cfg.settled.pins) in.read((char*)p.data(), 3 * sizeof(Real));
    cfg.settled.valid = (bool)in; cfg.settled.key = key;
    return cfg.settled.valid;
}
inline void save_settled(const std::string& file, const AppConfig& cfg)
{
    const std::string tmp = file + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        const std::string key = settle_key(cfg);
        const size_t n = key.size(), nx = cfg.settled.x.size(), np = cfg.settled.pins.size();
        out.write((const char*)&n, sizeof n); out.write(key.data(), n);
        out.write((const char*)&nx, sizeof nx); out.write((const char*)cfg.settled.x.data(), nx * sizeof(Real));
        out.write((const char*)&np, sizeof np);
        for (const Vec3& p : cfg.settled.pins) out.write((const char*)p.data(), 3 * sizeof(Real));
    }
    std::filesystem::rename(tmp, file);
}
} // namespace hand_opt_detail

inline int run_hand_opt(const std::string& config, const HandOptArgs& args, HandOptSpec spec)
{
    using namespace hand_opt_detail;
    if (args.kv.count("ctrl")) // ctrl=x,y;x,y;... replaces the target control points (z stays 0)
    {
        spec.target_controls.clear();
        std::istringstream in(args.kv.at("ctrl"));
        std::string pt;
        while (std::getline(in, pt, ';')) { Real x = 0, y = 0; char comma; std::istringstream(pt) >> x >> comma >> y; spec.target_controls.push_back(Vec3(x, y, 0.0)); }
        spec.K = (int)spec.target_controls.size();
    }
    AppConfig cfg;
    ASSERT(load_config(config, cfg), "cannot load config " << config);
    // the experiment fixes these: tracks run at 60 Hz (substeps refine), no animation file / waist (they would add a body), full target/guess mode (settle uses target_k)
    cfg.run_mode = RunMode::GuessTargetBackward;
    cfg.solver.FPS = (int)kColliderAnimFPS;
    cfg.world.animation_file = ""; cfg.world.waist_attach_enabled = false;
    cfg.world.compute_unresolved = false;
    if (args.kv.count("secs"))       cfg.solver.secs = (int)args.num("secs", 3);
    if (args.kv.count("substeps"))   cfg.solver.frame_substeps = (int)args.num("substeps", 2);
    if (args.kv.count("n_iters"))    cfg.solver.n_iters = (int)args.num("n_iters", 20);
    if (args.kv.count("n_iters_adj")) cfg.solver.n_iters_adjoint = (int)args.num("n_iters_adj", 300);
    if (args.kv.count("anderson"))   cfg.solver.anderson_m = (int)args.num("anderson", 5);
    if (args.kv.count("tri_h"))      cfg.cloth.tri_h = args.num("tri_h", 0.04);
    if (args.kv.count("model"))      cfg.world.contact.model = args.num("model", 1) != 0 ? ContactModel::Gap : ContactModel::RelativeVelocity;
    cfg.world.contact.recheck = ContactRecheckMode::Once; // the backward pass assumes a fixed contact set (Gap always keeps one)
    if (args.kv.count("settle_s"))   cfg.settle_seconds = args.num("settle_s", 20);
    if (args.kv.count("settle_iters")) cfg.settle_iters = (int)args.num("settle_iters", 20);
    cfg.settle_jitter = args.num("z_jitter", 0.0); // m, symmetry breaking of the settle start (see AppConfig::settle_jitter)
    ASSERT((int)cfg.world.colliders.size() > spec.collider && (int)spec.target_controls.size() == spec.K, "hand_opt: bad collider index or control point count");

    const std::string mode = args.str("mode", "opt");
    const std::string out_dir = args.str("out", "optimization_experiment/results/hand");
    const std::string tag = args.str("tag", "run");
    std::filesystem::create_directories(out_dir);

    const Vec3 start = spec.target_controls[0];
    set_hand_position(cfg.world.colliders[spec.collider], start);
    cfg.world.colliders[spec.collider].velocity = Vec3::Zero();

    // Settle the cloth once against the hand parked at its start (static collider), then every run animates it.
    const std::string settled_file = out_dir + "/settled_" + args.str("settle_tag", "default") + ".bin";
    {
        Quiet q;
        if (!load_settled(settled_file, cfg))
        {
            // the pose is shared by every run of a tag: a run whose config differs must never silently replace it
            ASSERT(mode == "settle" || !std::filesystem::exists(settled_file), "hand_opt: " << settled_file << " was made for a different config (settle_s, settle_iters, stiffness, ...): same arguments, or another settle_tag");
            settle_cloth(cfg, /*headless=*/true);
            std::fprintf(stderr, "[hand] settle: %d steps, fastest particle at the end %.2e m/s\n", cfg.settled.steps, cfg.settled.max_speed);
            save_settled(settled_file, cfg);
        }
    }
    std::printf("[hand] settled pose ready (%d free dofs), mode %s\n", (int)cfg.settled.x.size(), mode.c_str());
    std::fflush(stdout);
    if (mode == "settle") return 0;
    if (args.kv.count("spacer_k")) cfg.cloth.rail_spacer_k = args.num("spacer_k", 0.0); // after the settle on purpose: every run starts from the same pose, whatever the spacers
    cfg.world.colliders[spec.collider].animated = true;
    cfg.world.colliders[spec.collider].anim_id  = 0;

    const Real dt      = 1.0 / (cfg.solver.FPS * cfg.solver.frame_substeps);
    const int  S       = cfg.solver.frame_substeps;
    const int  n_steps = cfg.solver.FPS * S * cfg.solver.secs;
    const int  frames  = n_steps / S + 1;
    const int  sample_every = (int)args.num("sample_every", 0) > 0 ? (int)args.num("sample_every", 0) : n_steps; // 0 = final pose only
    int        loss_until   = (int)args.num("loss_until", 0) > 0 ? std::min((int)args.num("loss_until", 0), n_steps) : n_steps; // only frames up to this step count; with final-pose sampling this is the observation time

    // loss=gather: no target, the loss is a GatherLoss (solver/loss.h) over the last `win` steps; the path at the control points (ctrl=, + perturb) is only the starting guess.
    // r0 anneals linearly from r0_start to r0 over `anneal` iterations; stop_rel > 0 stops when the loss (at the final r0) moved less than that, relatively, in 25 iterations.
    const bool gather = args.str("loss", "target") == "gather";
    const int  win = (int)args.num("win", 60), anneal = (int)args.num("anneal", 0);
    const Real r0_end = args.num("r0", 0.10), r0_start = args.num("r0_start", r0_end), lam_rest = args.num("lam_rest", 1e-2), stop_rel = args.num("stop_rel", 0.0);
    Real       r0_now = r0_start;

    BSplinePath path(spec.K, frames, start, spec.free_axes);
    {   // free=x or y: optimise one axis only and hold the other at the target's control points (a probe of which axis the loss can identify)
        const std::string free = args.str("free", "xy");
        if (free != "xy") { path.free_axes = Vec3(free == "x" ? 1.0 : 0.0, free == "y" ? 1.0 : 0.0, 0.0); path.base = spec.target_controls; }
    }
    ColliderAnimation shape;
    shape.name = "hand"; shape.type = cfg.world.colliders[spec.collider].type == ColliderType::Arc ? ColliderType::Arc : ColliderType::Sphere;
    shape.radius = cfg.world.colliders[spec.collider].sphere_radius;

    PathRegulariser reg;
    reg.speed = args.num("lam_v", 0.0); reg.accel = args.num("lam_a", 0.0); reg.bound = args.num("lam_b", 0.0);
    reg.box_min = Vec3(args.num("xmin", -1e9), args.num("ymin", -1e9), -1e9); reg.box_max = Vec3(args.num("xmax", 1e9), args.num("ymax", 1e9), 1e9);

    auto seconds_since = [](auto t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    struct Eval { Real loss = 0.0, loss_full = 0.0, reg = 0.0, open_frac = 0.0, speed_rms = 0.0, gather = 0.0, rest = 0.0; RealVecX grad; std::vector<uint64_t> contacts; int n_contacts = 0; Real fwd_s = 0, bwd_s = 0, tail10 = 0; Tape tape; };

    // forward (+ optional backward) of one hand path
    auto simulate = [&](const RealVecX& theta, const Tape* target, bool backward) -> Eval
    {
        Quiet q;
        Eval e;
        const std::vector<Vec3> p = path.positions(theta);
        AppConfig run = cfg;
        WorldSetup world = prepare_world(run);
        world.collision.animations = { make_translation_track(shape, p) };
        world.collision.collect_backward = backward;
        SimRunResult r;
        r.obj = build_run_cloth(run, run.target_k());
        construct_pd_lhs(r.obj, dt);
        auto t0 = std::chrono::steady_clock::now();
        pd_contact(r.obj, world.collision, dt, run.solver.gravity, run.solver.n_iters, n_steps, S, r.tape, "hand_opt", false, false, nullptr, -1.0);
        e.fwd_s = seconds_since(t0);
        std::vector<Real> est;
        for (const StepResidual& s : r.tape.forward_residual) est.push_back(s.est_rel);
        std::sort(est.begin(), est.end(), std::greater<Real>());
        const size_t nt = std::max<size_t>(1, est.size() / 10);
        for (size_t i = 0; i < nt && i < est.size(); ++i) e.tail10 += est[i] / nt;
        for (size_t t = 0; t < r.tape.contacts.size(); ++t)
            for (const Contact& c : r.tape.contacts[t]) { e.contacts.push_back((uint64_t)t * 1000000 + c.particle); ++e.n_contacts; }
        std::sort(e.contacts.begin(), e.contacts.end());
        if (target || gather)
        {
            GatherLoss gl;
            if (gather) gl = GatherLoss(r.tape, r0_now, win, lam_rest, dt);
            const Loss loss = gather ? Loss(gl) : [&]
            {
                if (loss_until >= n_steps) return Loss(r.tape, *target, sample_every);
                // a shorter observation window: the loss sees the frames up to loss_until only (gradient 0 after), a snapshot at loss_until when sampling the final pose
                Tape g, t;
                g.positions.assign(r.tape.positions.begin(), r.tape.positions.begin() + loss_until + 1);
                t.positions.assign((*target).positions.begin(), (*target).positions.begin() + loss_until + 1);
                Loss l(g, t, std::min(sample_every, loss_until));
                l.dloss_dx.resize(n_steps + 1, RealVecX::Zero(l.dloss_dx[0].size()));
                return l;
            }();
            e.loss = loss.total;
            e.loss_full = gather ? GatherLoss(r.tape, r0_end, win, lam_rest, dt).total : loss_until >= n_steps ? loss.total : Loss(r.tape, *target, sample_every).total;
            e.open_frac = gl.open_frac; e.speed_rms = gl.speed_rms; e.gather = gl.gather; e.rest = gl.rest;
            if (backward)
            {
                t0 = std::chrono::steady_clock::now();
                const BackwardGradContact g = backward_pd_contact(r.obj, world.collision, r.tape, loss, run.solver.gravity, run.solver.n_iters_adjoint, dt);
                e.bwd_s = seconds_since(t0);
                std::vector<Vec3> gp = track_position_gradient(world.collision.animations[0], S, g.colliders[0].d_position, g.colliders[0].d_velocity);
                for (Vec3& x : gp) x = blender_to_diffpd(x); // Blender axes -> the path's
                std::vector<Vec3> gr(frames, Vec3::Zero());
                e.reg = reg.add(p, gr);
                for (int k = 0; k < frames; ++k) gp[k] += gr[k];
                e.grad = path.pullback(gp);
            }
        }
        e.tape = std::move(r.tape);
        return e;
    };

    const RealVecX theta_star = path.theta_from_controls(spec.target_controls);
    const Eval target = simulate(theta_star, nullptr, false);
    {
        const Eval again = gather ? Eval() : simulate(theta_star, &target.tape, false);
        std::printf("[hand] %d steps, %d free dofs, %d params; target: %d contacts, fwd %.1fs, est_rel tail10 %.1e; loss at theta* (must be 0): %.3e\n",
                    n_steps, (int)cfg.settled.x.size(), path.num_params(), target.n_contacts, target.fwd_s, target.tail10, again.loss);
        std::fflush(stdout);
        ASSERT(gather || again.loss < 1e-18, "hand_opt: the simulation is not deterministic");
    }
    // cloth end state, to see what the target did (width of the gathered cloth, thickness, speed)
    {
        const PointsX& xe = target.tape.positions.back();
        const PointsX& xm = target.tape.positions[target.tape.positions.size() * 2 / 3];
        std::printf("[hand] target final cloth: x in [%.3f, %.3f] y in [%.3f, %.3f] z in [%.3f, %.3f]; speed (last step) %.3e m/s\n", xe.col(0).minCoeff(), xe.col(0).maxCoeff(),
                    xe.col(1).minCoeff(), xe.col(1).maxCoeff(), xe.col(2).minCoeff(), xe.col(2).maxCoeff(), (target.tape.velocities.back()).rowwise().norm().maxCoeff());
        (void)xm;
    }

    if (mode == "trace") // what the target hand does to the cloth over time
    {
        const std::vector<Vec3> hp = path.positions(theta_star);
        std::printf("[trace] t(s)  hand(x,y)  cloth x[min,max] y[min] z[min,max]  max speed  contacts\n");
        for (int t = 0; t <= n_steps; t += S * (int)args.num("every", 15))
        {
            const PointsX& x = target.tape.positions[t];
            std::printf("[trace] %.2f  (%.3f, %.3f)  x[%.3f, %.3f] y[%.3f] z[%.3f, %.3f]  max %.2e mean %.2e  %d\n", t * dt, hp[t / S].x(), hp[t / S].y(), x.col(0).minCoeff(), x.col(0).maxCoeff(),
                        x.col(1).minCoeff(), x.col(2).minCoeff(), x.col(2).maxCoeff(), target.tape.velocities[t].rowwise().norm().maxCoeff(), target.tape.velocities[t].rowwise().norm().mean(), t < (int)target.tape.contacts.size() ? (int)target.tape.contacts[t].size() : 0);
        }
        return 0;
    }

    std::mt19937_64 rng((unsigned)args.num("seed", 1));
    auto u01 = [&] { return (rng() >> 11) * (1.0 / 9007199254740992.0); };
    auto random_unit = [&](int n) { RealVecX d(n); for (int i = 0; i < n; ++i) d[i] = 2.0 * u01() - 1.0; return RealVecX(d / d.norm()); };

    const std::string base = out_dir + "/" + tag;
    if (mode == "check") // analytic gradient against central differences of the loss along random directions, at a perturbed theta
    {
        const Real perturb = args.num("perturb", 0.05);
        RealVecX th = theta_star;
        for (int i = 0; i < th.size(); ++i) th[i] += perturb * (u01() < 0.5 ? -1.0 : 1.0) * (0.5 + 0.5 * u01());
        const Eval base_eval = simulate(th, &target.tape, true);
        std::printf("[check] perturb %.3f m: loss %.4e, |grad| %.4e, %d contacts, fwd %.1fs bwd %.1fs, est_rel tail10 %.1e\n", perturb, base_eval.loss, base_eval.grad.norm(), base_eval.n_contacts, base_eval.fwd_s, base_eval.bwd_s, base_eval.tail10);
        for (int d = 0; d < (int)args.num("dirs", 2); ++d)
        {
            const RealVecX dir = d == 0 ? RealVecX(base_eval.grad / std::max(base_eval.grad.norm(), 1e-300)) : random_unit((int)th.size());
            for (const Real eps : { 1e-3, 1e-4, 1e-5 })
            {
                const Real fd = (simulate(th + eps * dir, &target.tape, false).loss - simulate(th - eps * dir, &target.tape, false).loss) / (2 * eps);
                const Real an = base_eval.grad.dot(dir);
                std::printf("[check] dir %d (%s) eps %.0e  analytic % .6e  fd % .6e  rel err %.2e\n", d, d == 0 ? "gradient" : "random", eps, an, fd, std::abs(an - fd) / std::max(std::abs(fd), 1e-300));
                std::fflush(stdout);
            }
        }
        return 0;
    }
    if (mode == "probe") // loss and gradient along a line through theta*: how smooth is the landscape, and does the gradient follow it
    {
        const int  n = (int)args.num("points", 21);
        const Real span = args.num("span", 0.2);
        RealVecX dir = random_unit(path.num_params());
        const std::string dir_name = args.str("dir", "random"); // x / y: every control point moved the same way along the axis (free axes are x, y in that order)
        if (dir_name == "x" || dir_name == "y")
        {
            dir = RealVecX::Zero(path.num_params());
            for (int i = (dir_name == "x" ? 0 : 1); i < path.num_params(); i += 2) dir[i] = 1.0;
            dir.normalize();
        }
        std::ofstream csv(base + "_probe.csv");
        csv << std::setprecision(17) << "# span " << span << " seed " << args.num("seed", 1) << "\ns,loss,dloss_ds_analytic,contacts,jaccard_prev,fwd_est_tail10\n";
        std::vector<uint64_t> prev;
        for (int i = 0; i < n; ++i)
        {
            const Real s = -span + 2.0 * span * i / (n - 1);
            const Eval e = simulate(theta_star + s * dir, &target.tape, true);
            std::vector<uint64_t> inter; std::set_intersection(e.contacts.begin(), e.contacts.end(), prev.begin(), prev.end(), std::back_inserter(inter));
            const Real jac = prev.empty() ? 1.0 : (Real)inter.size() / (Real)(e.contacts.size() + prev.size() - inter.size());
            csv << s << ',' << e.loss << ',' << e.grad.dot(dir) << ',' << e.n_contacts << ',' << jac << ',' << e.tail10 << '\n';
            csv.flush();
            std::printf("[probe] s % .4f  loss %.4e  dL/ds % .4e  contacts %d  jaccard %.3f\n", s, e.loss, e.grad.dot(dir), e.n_contacts, jac);
            std::fflush(stdout);
            prev = e.contacts;
        }
        return 0;
    }

    // where the target hand touches the cloth: parameters outside this window cannot be identified from the cloth, so path errors are also reported inside it
    const std::vector<Vec3> p_star = path.positions(theta_star);
    int first_contact_frame = frames, last_contact_frame = -1;
    for (int t = 0; t < (int)target.tape.contacts.size(); ++t)
        if (!target.tape.contacts[t].empty()) { first_contact_frame = std::min(first_contact_frame, t / S); last_contact_frame = std::max(last_contact_frame, t / S); }
    auto path_error = [&](const RealVecX& th, Real& all_max, Real& win_max, Real& win_rms)
    {
        const std::vector<Vec3> p = path.positions(th);
        all_max = win_max = win_rms = 0.0;
        int n = 0;
        for (int k = 0; k < frames; ++k)
        {
            const Real d = (p[k] - p_star[k]).norm();
            all_max = std::max(all_max, d);
            if (k >= first_contact_frame && k <= last_contact_frame) { win_max = std::max(win_max, d); win_rms += d * d; ++n; }
        }
        win_rms = n ? std::sqrt(win_rms / n) : 0.0;
    };
    std::printf("[hand] target touches the cloth in frames %d..%d of %d\n", first_contact_frame, last_contact_frame, frames);
    // mode opt: perturb theta*, then Adam
    const Real perturb = args.num("perturb", 0.05);
    RealVecX th = theta_star;
    for (int i = 0; i < th.size(); ++i) th[i] += perturb * (u01() < 0.5 ? -1.0 : 1.0) * (0.5 + 0.5 * u01());
    Adam adam;
    adam.lr0 = args.num("lr", 0.01); adam.lr_decay = args.num("lr_decay", 0.98);
    const int iters = (int)args.num("iters", 60);
    const Real stop_err = args.num("stop_err", 1e-3);
    std::ofstream csv(base + ".csv"), thcsv(base + "_theta.csv");
    csv << std::setprecision(17);
    csv << "# config " << config << " seed " << args.num("seed", 1) << " perturb " << perturb << " lr " << adam.lr0 << " decay " << adam.lr_decay << " sample_every " << sample_every << " secs " << cfg.solver.secs
        << " substeps " << S << " n_iters " << cfg.solver.n_iters << " anderson " << cfg.solver.anderson_m << " model " << (int)cfg.world.contact.model << " lam_v " << reg.speed << " lam_a " << reg.accel << " loss " << (gather ? "gather" : "target") << " r0 " << r0_end << " r0_start " << r0_start << " anneal " << anneal << " win " << win << " lam_rest " << lam_rest << "\n";
    csv << "iter,loss,reg,err_max,err_rms,grad_norm,contacts,jaccard_prev,fwd_est_tail10,fwd_s,bwd_s,lr,path_err_all_max,path_err_contact_max,path_err_contact_rms,loss_full,loss_until,open_frac,speed_rms,gather,rest,r0\n";
    std::vector<uint64_t> prev;
    bool recovered = false;
    std::vector<Real> loss_hist;
    const int freeze_y = path.num_free() == 2 ? (int)args.num("freeze_y", 0) : 0;
    const int smooth_m = (int)args.num("smooth_m", 0); const Real smooth_sigma = args.num("smooth_sigma", 0.01);
    const int curr_start = (int)args.num("curr_start", 0), curr_iters = (int)args.num("curr_iters", 0); // 0 = no curriculum
    for (int it = 0; it <= iters; ++it)
    {
        if (anneal > 0) r0_now = r0_end + (r0_start - r0_end) * std::max(0.0, 1.0 - (double)it / anneal);
        if (curr_iters > 0) loss_until = std::min(n_steps, curr_start + (int)((n_steps - curr_start) * std::min(1.0, (double)it / curr_iters)));
        Eval e = simulate(th, &target.tape, true);
        if (smooth_m > 0) // randomised smoothing: average the gradient over the iterate and smooth_m Gaussian-perturbed hands (sigma in m), a smoothed loss without the contact flips
        {
            RealVecX gsum = e.grad;
            for (int m = 0; m < smooth_m; ++m)
            {
                RealVecX eps(th.size());
                for (int i = 0; i < th.size(); ++i) eps[i] = smooth_sigma * std::sqrt(-2.0 * std::log(1.0 - u01())) * std::cos(6.283185307179586 * u01());
                gsum += simulate(th + eps, &target.tape, true).grad;
            }
            e.grad = gsum / Real(smooth_m + 1);
        }
        const RealVecX err = th - theta_star;
        std::vector<uint64_t> inter; std::set_intersection(e.contacts.begin(), e.contacts.end(), prev.begin(), prev.end(), std::back_inserter(inter));
        const Real jac = prev.empty() ? 1.0 : (Real)inter.size() / std::max<Real>(1.0, (Real)(e.contacts.size() + prev.size() - inter.size()));
        Real pe_all, pe_max, pe_rms; path_error(th, pe_all, pe_max, pe_rms);
        csv << it << ',' << e.loss << ',' << e.reg << ',' << err.cwiseAbs().maxCoeff() << ',' << std::sqrt(err.squaredNorm() / err.size()) << ',' << e.grad.norm() << ',' << e.n_contacts << ',' << jac << ',' << e.tail10 << ',' << e.fwd_s << ',' << e.bwd_s << ',' << adam.lr0 * std::pow(adam.lr_decay, it) << ',' << pe_all << ',' << pe_max << ',' << pe_rms << ',' << e.loss_full << ',' << loss_until << ',' << e.open_frac << ',' << e.speed_rms << ',' << e.gather << ',' << e.rest << ',' << r0_now << '\n';
        csv.flush();
        thcsv << it; for (int i = 0; i < th.size(); ++i) thcsv << ',' << th[i]; thcsv << '\n'; thcsv.flush();
        if (gather) std::printf("[gather] open %.3f  speed %.2e m/s  r0 %.3f  ", e.open_frac, e.speed_rms, r0_now);
        std::printf("[opt] it %3d  loss %.4e  reg %.2e  err max %.2e rms %.2e m  |g| %.2e  contacts %d  jaccard %.3f  tail10 %.1e  fwd %.1fs bwd %.1fs  hand path err max %.1f mm, in contact window max %.1f rms %.1f mm\n", it, e.loss, e.reg, err.cwiseAbs().maxCoeff(), std::sqrt(err.squaredNorm() / err.size()), e.grad.norm(), e.n_contacts, jac, e.tail10, e.fwd_s, e.bwd_s, 1e3 * pe_all, 1e3 * pe_max, 1e3 * pe_rms);
        std::fflush(stdout);
        prev = e.contacts;
        if (!gather && err.cwiseAbs().maxCoeff() < stop_err) { recovered = true; break; }
        loss_hist.push_back(e.loss_full);
        if (gather && stop_rel > 0.0 && it >= anneal + 25 && std::abs(loss_hist[it] - loss_hist[it - 25]) < stop_rel * loss_hist[it - 25]) { std::printf("[opt] converged at iteration %d (loss moved < %.1e relative in 25 iterations)\n", it, stop_rel); break; }
        if (it == iters) break;
        RealVecX g = e.grad;
        if (it < freeze_y) for (int i = 1; i < g.size(); i += 2) g[i] = 0.0; // stage 1: height held (theta layout: x, y per control point), released at iteration freeze_y
        adam.step(th, g);
    }
    std::printf("[opt] %s\n", gather ? "done" : recovered ? "RECOVERED" : "not recovered");
    return recovered || gather ? 0 : 1;
}

// The curtain preset: the hand starts at the bottom of the free edge, rises along it to mid height, then pulls the curtain toward the pinned end
// and stops short of it (so the gathered cloth comes to rest). Control points in diffpd axes (x along the rail, y up, z fixed).
inline HandOptSpec curtain_sweep_spec()
{
    HandOptSpec s;
    s.K = 7;
    s.target_controls = { Vec3(1.00, 0.05, 0.0), Vec3(1.00, 0.30, 0.0), Vec3(0.90, 0.45, 0.0), Vec3(0.50, 0.45, 0.0),
                          Vec3(0.22, 0.45, 0.0), Vec3(0.22, 0.45, 0.0), Vec3(0.22, 0.45, 0.0) };
    return s;
}
