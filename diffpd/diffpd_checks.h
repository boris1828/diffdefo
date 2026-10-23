#pragma once

// Headless self-checks (--regression, --meshgen-check, --arap-check) and diagnostics, split out of diffpd.cpp.
// Not a standalone header: it is #included once, just above main(), and uses that file's functions and globals.

// `diffpd --regression`: runs a few fixed configs headless (target sim, guess sim, loss, adjoint) and prints the
// results with 17 digits, so a refactor that must not change the numerics can be diffed bit for bit against a saved
// stdout. Wall times go to stderr, since they never match.
static int run_regression()
{
    auto square = []
    {
        AppConfig c;
        c.cloth_type           = ClothType::Square;
        c.mesh_gen             = MeshGen::Quad; // the golden output predates the Tri default
        c.width = c.height     = 10;
        c.size_x = c.size_y    = 2.0; // the golden output predates the 1x1 default
        c.pin_mode             = PinMode::CORNERS;
        c.colliders            = { make_sphere(Vec3(1.0, 0.3, 1.0), 0.5) };
        c.animation_file       = "";
        c.waist_attach_enabled = false;
        c.secs = 1; c.n_iters = 50; c.n_iters_adjoint = 50;
        return c;
    };
    std::vector<std::pair<std::string, AppConfig>> configs;
    configs.push_back({ "A square, per-type, sphere", square() });
    {
        AppConfig c = square();
        c.damping_enabled = true; c.damping_alpha = 0.1; c.per_type_stiffness = false;
        configs.push_back({ "B square, damped, unified", c });
    }
    {
        AppConfig c; // default skirt + skinned body + waist attachment
        c.mesh_gen = MeshGen::Quad; c.radius_top = 0.19; // the golden output predates the Tri default
        c.particles_per_ring = 16; c.num_rings = 8;
        c.secs = 3; c.n_iters = 50; c.n_iters_adjoint = 50; // 3 s: the legs reach the skirt
        configs.push_back({ "C skirt, skinned body, waist", c });
    }
    {
        AppConfig c = square();
        const Object procedural = build_cloth(c, c.target_k(), c.origin);
        c.settled.valid = true;
        c.settled.x     = procedural.x + RealVecX::Constant(procedural.x.size(), 0.01);
        for (const Vec3& p : procedural.mesh.pinned_rest) c.settled.pins.push_back(p + Vec3(0.02, 0.0, 0.0));
        configs.push_back({ "D square, settled pose with moved pins", c });
    }

    std::cout << std::setprecision(17);
    for (auto& [name, cfg] : configs)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const WorldSetup world = prepare_world(cfg);
        const Real dt      = 1.0 / (cfg.FPS * cfg.frame_substeps);
        const int  n_steps = cfg.FPS * cfg.frame_substeps * cfg.secs;
        auto simulate = [&](const Stiffness& k)
        {
            SimRunResult r;
            r.obj = build_run_cloth(cfg, k);
            init_pd_velocity(r.obj, dt);
            attach_waist(r.obj, world.waist_attach_anim_id, world.waist_attach_skinned_id, collider_animations);
            pd_contact(r.obj, dt, cfg.gravity, cfg.n_iters, n_steps, cfg.frame_substeps, r.tape, "regression",
                       /*export_obj=*/false, /*verbose=*/false, nullptr, cfg.unresolved_contact_threshold);
            return r;
        };
        SimRunResult target = simulate(cfg.target_k());
        SimRunResult guess  = simulate(cfg.guess_k());
        const Loss loss(guess.tape, target.tape, cfg.frame_substeps);
        const BackwardGradContact grad = backward_pd_contact(guess.obj, guess.tape, loss, cfg.gravity, cfg.n_iters_adjoint, dt);

        auto axis_sums = [](const RealVecX& g) -> Vec3
        { return Eigen::Map<const PointsX>(g.data(), g.size() / 3, 3).colwise().sum().transpose(); };
        Real fwd_res = 0.0, adj_res = 0.0;
        for (const StepResidual& r : guess.tape.forward_residual) fwd_res += r.est_rel;
        for (Real r : grad.residual) adj_res += r;
        size_t n_contacts = 0;
        for (const Contacts& c : guess.tape.contacts) n_contacts += c.size();
        const Vec3 dx = axis_sums(grad.dphi_dx), dv = axis_sums(grad.dphi_dv);

        std::cout << "== " << name << "\n"
                  << "loss " << loss.total << "\n"
                  << "target_x sum " << target.obj.x.sum() << " sqnorm " << target.obj.x.squaredNorm() << "\n"
                  << "guess_x sum " << guess.obj.x.sum() << " sqnorm " << guess.obj.x.squaredNorm() << "\n"
                  << "guess_v sqnorm " << guess.obj.v.squaredNorm() << "\n"
                  << "dphi_dk " << grad.dphi_dk.stretch << " " << grad.dphi_dk.shear << " " << grad.dphi_dk.bend << "\n"
                  << "dphi_dx0 " << dx.x() << " " << dx.y() << " " << dx.z() << " sqnorm " << grad.dphi_dx.squaredNorm() << "\n"
                  << "dphi_dv0 " << dv.x() << " " << dv.y() << " " << dv.z() << " sqnorm " << grad.dphi_dv.squaredNorm() << "\n"
                  << "dphi_dalpha " << grad.dphi_dalpha << " dphi_dbeta " << grad.dphi_dbeta << "\n"
                  << "residual_sums fwd " << fwd_res << " adj " << adj_res << "\n"
                  << "contacts " << n_contacts << std::endl;
        std::cerr << "[regression] " << name << ": "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
    }
    return 0;
}

// `diffpd --meshgen-check`: headless checks of the triangle-mesh generator (tri_surface.h) over a few cones, then a small
// tri-skirt sim with a dphi/dk central-difference check. Aborts (ASSERT) on any failure.
static int run_meshgen_check()
{
    struct Case { Real r_top, r_bot, H, h, beta; int smooth_iters; uint64_t seed; };
    const Case cases[] = {
        { 0.20, 0.40, 0.6, 0.07, 0.85, 5, 0 },  // widening
        { 0.40, 0.20, 0.6, 0.07, 0.85, 3, 3 },  // narrowing
        { 0.20, 0.50, 0.3, 0.05, 1.00, 10, 1 }, // dense, beta = 1
        { 0.30, 0.302, 0.6, 0.07, 0.85, 5, 2 }, // near-cylinder (thin strip, huge apex distance)
        { 0.20, 0.30, 0.5, 0.03, 0.80, 30, 0 }, // the UI default
    };
    std::printf("%-28s %5s %5s %5s %4s %4s %6s %6s %7s %s\n", "case", "V", "E", "F", "euler", "open", "minAng", "meanE/h", "area", "ok");
    for (const Case& c : cases)
    {
        meshgen::Params p;
        p.h = c.h; p.beta = c.beta; p.smooth_iters = c.smooth_iters; p.seed = c.seed;
        const meshgen::ConeFrustum cone(c.r_top, c.r_bot, c.H);
        const meshgen::TriSurface s = meshgen::generate(cone, p);
        const meshgen::Report r = meshgen::validate(s, cone, p);
        char name[64];
        std::snprintf(name, sizeof(name), "r %.3f->%.3f H %.1f h %.2f", c.r_top, c.r_bot, c.H, c.h);
        std::printf("%-28s %5lld %5lld %5lld %5d %4d %6.1f %7.2f %+6.2f%% %s\n", name, (long long)r.vertices, (long long)r.edges,
                    (long long)r.triangles, r.euler, r.open_edges, r.min_angle_deg, r.edge_mean, 100 * r.area_error,
                    r.quality_ok ? "ok" : "QUALITY");
        ASSERT(r.structure_ok, name << ": not a clean surface");
        ASSERT(c.smooth_iters < 3 || r.quality_ok, name << ": mesh quality below the thresholds");

        const meshgen::TriSurface again = meshgen::generate(cone, p);
        ASSERT(again.pos.size() == s.pos.size() && again.tris == s.tris, name << ": generate() is not deterministic");
        for (size_t i = 0; i < s.pos.size(); ++i) ASSERT(again.pos[i] == s.pos[i], name << ": generate() is not deterministic");
        meshgen::Params q = p;
        q.seed += 1;
        const meshgen::TriSurface other = meshgen::generate(cone, q);
        ASSERT(other.pos.size() != s.pos.size() || other.tris != s.tris, name << ": a different seed gave the same mesh");
    }

    {   // a chart with no seam (the Chart defaults): a flat rectangle, euler 1, the two top corners owned by their own curves
        const meshgen::Rectangle rect(0.6, 0.4, /*vertical=*/false);
        meshgen::Params p;
        p.h = 0.05; p.smooth_iters = 5;
        const meshgen::TriSurface s = meshgen::generate(rect, p);
        const meshgen::Report r = meshgen::validate(s, rect, p);
        std::printf("%-28s %5lld %5lld %5lld %5d %4d %6.1f %7.2f %+6.2f%% %s\n", "flat rectangle (no seam)", (long long)r.vertices,
                    (long long)r.edges, (long long)r.triangles, r.euler, r.open_edges, r.min_angle_deg, r.edge_mean, 100 * r.area_error,
                    r.quality_ok ? "ok" : "QUALITY");
        ASSERT(r.structure_ok && r.quality_ok, "rectangle: not a clean surface");
        for (int corner : { (int)meshgen::Rectangle::CornerTL, (int)meshgen::Rectangle::CornerTR })
            ASSERT(std::count(s.curve.begin(), s.curve.end(), corner) == 1, "rectangle: a corner curve must own exactly one vertex");
    }

    AppConfig cfg;
    cfg.mesh_gen = MeshGen::Tri;
    cfg.per_type_stiffness = false;
    cfg.tri_h = 0.07; cfg.tri_smooth_iters = 5; // coarse and fast; the UI default is finer
    cfg.colliders = { make_sphere(Vec3(0.0, 0.55, 0.0), 0.25) };
    cfg.animation_file = "";
    cfg.waist_attach_enabled = false;
    cfg.contact_recheck_mode = ContactRecheckMode::Once;
    cfg.secs = 1; cfg.n_iters = 1000; cfg.n_iters_adjoint = 1000; // converged solves, as the FD check needs
    ASSERT(cfg.tri_ok(), "default tri config rejected");
    prepare_world(cfg);
    const Real dt = 1.0 / (cfg.FPS * cfg.frame_substeps);
    const int  n_steps = cfg.FPS * cfg.frame_substeps * cfg.secs;
    auto simulate = [&](Real k)
    {
        SimRunResult r;
        r.obj = build_run_cloth(cfg, Stiffness::uniform(k));
        init_pd_velocity(r.obj, dt);
        pd_contact(r.obj, dt, cfg.gravity, cfg.n_iters, n_steps, cfg.frame_substeps, r.tape, "meshgen_check",
                   /*export_obj=*/false, /*verbose=*/false, nullptr, cfg.unresolved_contact_threshold);
        return r;
    };
    SimRunResult target = simulate(cfg.target_stiffness);
    SimRunResult guess  = simulate(cfg.stiffness);
    const Loss loss(guess.tape, target.tape, cfg.frame_substeps);
    const BackwardGradContact grad = backward_pd_contact(guess.obj, guess.tape, loss, cfg.gravity, cfg.n_iters_adjoint, dt);
    size_t n_contacts = 0;
    for (const Contacts& c : guess.tape.contacts) n_contacts += c.size();
    ASSERT(std::isfinite(loss.total) && std::isfinite(grad.dphi_dk.sum()), "tri skirt sim diverged");
    ASSERT(grad.dphi_dk.shear == 0.0 && grad.dphi_dk.bend == 0.0, "tri skirt has shear/bend gradient");
    std::printf("tri skirt: %d free particles, %zu contacts, loss %.6g, dphi/dk %.8g\n",
                (int)guess.obj.num_particles(), n_contacts, loss.total, grad.dphi_dk.sum());
    for (Real eps : { 1e-4, 1e-6 })
    {
        const FDCheckResult fd = fd_check_contact_stiffness(
            [&](Real k) { return build_run_cloth(cfg, Stiffness::uniform(k)); }, target.tape, cfg.stiffness, dt, cfg.gravity,
            cfg.n_iters, n_steps, cfg.frame_substeps, cfg.frame_substeps, grad.dphi_dk.sum(), eps);
        std::printf("  FD eps %.0e: fd %.8g analytic %.8g rel_err %.2e\n", eps, fd.fd, fd.analytic, fd.rel_err);
        std::fflush(stdout);
        ASSERT(eps > 1e-5 || fd.rel_err < 1e-4, "tri skirt dphi/dk does not match central differences");
    }
    return 0;
}

// ARAP failure-mode diagnostics (not called by default: add `arap_diagnostics(run.obj, run.tape);` after the forward-only run in main) over a finished run, printed to stdout (remove once the backward pass is validated).
// Per frame and triangle: sigma2 (collapse), |s1-s2|/(s1+s2) (isotropic: SVD frame not unique), a jump of the principal axis
// V(:,0) between consecutive frames at a near-isotropic triangle (SVD frame flip); per hinge: fold (dihedral > 90 deg from flat).
// Plus the rest triangles' min area / min quality. shortcut: pinned positions are the end-of-run ones (a waist attachment moves
// them per step), so triangles touching a pin are slightly off there.
[[maybe_unused]] static void arap_diagnostics(const Object& obj, const Tape& tape)
{
    const auto& tris = obj.energies.arap.tris;
    if (tris.empty() || tape.positions.empty()) return;
    const auto& pinned = obj.mesh.pinned_rest;
    using Mat32 = ArapEnergy::Mat32;
    using Vec2  = Eigen::Matrix<Real, 2, 1>;

    Real min_area = 1e300, min_quality = 1e300;
    for (const auto& t : tris)
    {
        const Eigen::Matrix<Real, 2, 2> Dm = Eigen::Matrix<Real, 2, 2>(t.ghat.bottomRows<2>()).inverse(); // ghat rows 1,2 = Dm^-1
        const Real area = 0.5 * std::abs(Dm.determinant());
        const Real l2   = Dm(0, 0) * Dm(0, 0) + Dm.col(1).squaredNorm() + (Dm.col(1) - Vec2(Dm(0, 0), 0.0)).squaredNorm();
        min_area    = std::min(min_area, area);
        min_quality = std::min(min_quality, 4.0 * std::sqrt(3.0) * area / l2); // 1 = equilateral
    }

    auto vertex = [&](const PointsX& P, ParticleId d) -> Vec3
    { return d >= 0 ? Vec3(P.row(d).transpose()) : pinned[-(d + 1)]; };

    std::vector<Vec2> axis_prev(tris.size(), Vec2::Zero());
    std::vector<Real> gap_prev(tris.size(), 1.0);
    Real min_s2 = 1e300, min_gap = 1e300, min_det = 1e300;
    size_t n_s2 = 0, n_iso = 0, n_jump = 0, n_det = 0, n_fold = 0, n_samples = 0, worst_fold_frame = 0;
    size_t n_pin_stretched = 0;
    Real max_s1 = 0.0, max_s1_pin = 0.0;
    Real worst_fold = 1.0; // min over hinges and frames of the cos of the dihedral deviation (flat = 1)

    for (size_t f = 0; f < tape.positions.size(); ++f)
    {
        const PointsX& P = tape.positions[f];
        for (size_t i = 0; i < tris.size(); ++i)
        {
            const auto& t = tris[i];
            Mat3 X;
            for (int a = 0; a < 3; ++a) X.col(a) = vertex(P, t.dof[a]);
            const Mat32 S = X * t.ghat;
            Eigen::JacobiSVD<Mat32> svd(S, Eigen::ComputeThinV);
            const Real s1 = svd.singularValues()(0), s2 = svd.singularValues()(1);
            const Real gap = std::abs(s1 - s2) / std::max(s1 + s2, 1e-300);
            const Vec2 axis = svd.matrixV().col(0);
            if (t.has_pinned) { n_pin_stretched += s1 > 3.0; max_s1_pin = std::max(max_s1_pin, s1); continue; } // pins are end-of-run: unreliable
            ++n_samples;
            max_s1 = std::max(max_s1, s1);
            min_s2  = std::min(min_s2, s2);
            min_gap = std::min(min_gap, gap);
            n_s2    += s2 < 0.05;                // strongly compressed along one direction
            n_iso   += gap < 1e-2;               // frame ill-defined
            if (f > 0 && (gap < 1e-2 || gap_prev[i] < 1e-2) && std::abs(axis.dot(axis_prev[i])) < std::cos(0.5)) ++n_jump;
            axis_prev[i] = axis;
            gap_prev[i]  = gap;
            // 2D-area ratio s1*s2; "inverted" in 3D has no sign, so the fold test below is the real one
            min_det = std::min(min_det, s1 * s2);
            n_det  += s1 * s2 < 0.1;
        }
        for (const auto& h : obj.energies.bending.elems)
        {
            if (h.has_pinned) continue;
            const Vec3 p0 = vertex(P, h.dof[0]), p1 = vertex(P, h.dof[1]), p2 = vertex(P, h.dof[2]), p3 = vertex(P, h.dof[3]);
            const Vec3 e0 = p1 - p0, n0 = e0.cross(p2 - p0), n1 = -e0.cross(p3 - p0); // both on the same side when flat
            const Real denom = n0.norm() * n1.norm();
            if (denom < 1e-30) continue;
            const Real c = n0.dot(n1) / denom;
            if (c < worst_fold) { worst_fold = c; worst_fold_frame = f; }
            n_fold += c < 0.0;
        }
    }

    std::printf("ARAP diagnostics (%zu triangles, %zu frames, %zu hinges)\n", tris.size(), tape.positions.size(), obj.energies.bending.elems.size());
    std::printf("  rest: min area %.3e, min quality %.3f (1 = equilateral)\n", min_area, min_quality);
    std::printf("  (free triangles/hinges only) max sigma1 %.3f; pin-touching triangles: max sigma1 %.3f, %zu samples > 3\n", max_s1, max_s1_pin, n_pin_stretched);
    std::printf("  sigma2 -> 0:        min %.3e, %zu/%zu samples < 0.05\n", min_s2, n_s2, n_samples);
    std::printf("  area ratio s1*s2:   min %.3e, %zu samples < 0.1\n", min_det, n_det);
    std::printf("  sigma1 ~ sigma2:    min gap %.3e, %zu samples < 1e-2 (expected near rest), %zu axis jumps > 0.5 rad at those\n", min_gap, n_iso, n_jump);
    std::printf("  folded hinges:      %zu samples with dihedral > 90 deg, worst cos %.3f at frame %zu\n", n_fold, worst_fold, worst_fold_frame);
    std::fflush(stdout);
}

// `diffpd --arap-check`: ARAP force balance. At a rigid motion of the rest state R = S, so K x = b must hold exactly
// (free dofs, pinned vertices moved along). Aborts (ASSERT) on failure.
static int run_arap_check()
{
    meshgen::Params p;
    p.h = 0.07; p.smooth_iters = 5;
    const meshgen::TriSurface s = meshgen::generate(meshgen::ConeFrustum(0.2, 0.4, 0.6), p);
    Object obj = tri_cloth(s, { 0 }, Stiffness::uniform(1.0), Vec3::Zero(), 1.0);
    const SimMesh& mesh = obj.mesh;
    ArapEnergy arap;
    for (const auto& t : mesh.triangles)
    {
        const ParticleId dof[3] = { mesh.vertices[t[0]].dof, mesh.vertices[t[1]].dof, mesh.vertices[t[2]].dof };
        Vec3 pos[3];
        for (int a = 0; a < 3; ++a)
            pos[a] = dof[a] >= 0 ? Vec3(obj.x.segment<3>(3 * dof[a])) : mesh.pinned_rest[-(dof[a] + 1)];
        arap.add_triangle(dof, pos, SpringKind::Stretch, 2.0);
    }
    ASSERT(!arap.tris.empty(), "arap check: no triangles");

    const Index n3 = obj.num_dofs();
    std::vector<Triplet> trip;
    trip.reserve(arap.triplet_count());
    arap.add_stiffness_triplets(trip, 1.0);
    SparseMat K(n3, n3);
    K.setFromTriplets(trip.begin(), trip.end());

    const Eigen::Quaternion<Real> q(Eigen::AngleAxis<Real>(0.7, Vec3(1.0, 2.0, -0.5).normalized()));
    const Vec3 shift(0.3, -0.2, 0.9);
    for (int moved = 0; moved < 2; ++moved)
    {
        RealVecX x = obj.x;
        std::vector<Vec3> pins = mesh.pinned_rest;
        if (moved)
        {
            for (Index i = 0; i < n3 / 3; ++i) x.segment<3>(3 * i) = q * Vec3(x.segment<3>(3 * i)) + shift;
            for (Vec3& v : pins) v = q * v + shift;
        }
        RealVecX b = RealVecX::Zero(n3);
        arap.add_forces(x, pins, b, 1.0);
        const Real rel = (K * x - b).norm() / b.norm();
        std::printf("arap force balance (%s): %zu tris, rel residual %.2e\n", moved ? "rotated" : "rest", arap.tris.size(), rel);
        ASSERT(rel < 1e-10, "ARAP force balance violated");
    }

    // ARAP + hinge bending from tri_cloth, curved rest (l0 > 0): force balance under a rigid motion, pins moved too
    {
        Object o = tri_cloth(s, { 0 }, Stiffness{ 1.0, 0.0, 0.5 }, Vec3::Zero(), 1.0, TriEnergy::ARAP, /*flat_bend_rest=*/false);
        ASSERT(!o.energies.bending.elems.empty() && o.energies.springs.constraints.empty(), "ARAP skirt: no hinges or has springs");
        std::vector<Triplet> tr;
        tr.reserve(o.energies.arap.triplet_count() + o.energies.bending.triplet_count());
        o.energies.for_each([&](const auto& e) { e.add_stiffness_triplets(tr, 1.0); });
        SparseMat Kt(o.num_dofs(), o.num_dofs());
        Kt.setFromTriplets(tr.begin(), tr.end());
        RealVecX x = o.x;
        std::vector<Vec3> pins = o.mesh.pinned_rest;
        for (Index i = 0; i < o.num_particles(); ++i) x.segment<3>(3 * i) = q * Vec3(x.segment<3>(3 * i)) + shift;
        for (Vec3& v : pins) v = q * v + shift;
        RealVecX bt = RealVecX::Zero(o.num_dofs());
        o.energies.for_each([&](const auto& e) { e.add_forces(x, pins, bt, 1.0); });
        const Real rel = (Kt * x - bt).norm() / bt.norm();
        Real l0max = 0.0;
        for (const auto& e : o.energies.bending.elems) l0max = std::max(l0max, e.l0);
        std::printf("arap+hinge force balance (rotated): %zu hinges, max l0 %.3f, rel residual %.2e\n", o.energies.bending.elems.size(), l0max, rel);
        ASSERT(l0max > 0.0 && rel < 1e-10, "ARAP + hinge force balance violated");
        Object f = tri_cloth(s, { 0 }, Stiffness{ 1.0, 0.0, 0.5 }, Vec3::Zero(), 1.0, TriEnergy::ARAP, true);
        for (const auto& e : f.energies.bending.elems) ASSERT(e.l0 == 0.0, "flat_bend_rest left l0 != 0");
    }

    // forward sim of the Tri skirt and the Tri square (row pinned) with ARAP over a sphere: must stay finite and not fly off
    for (const ClothType type : { ClothType::Skirt, ClothType::Square })
    {
        AppConfig cfg;
        cfg.cloth_type = type;
        cfg.mesh_gen = MeshGen::Tri; cfg.tri_energy = TriEnergy::ARAP;
        cfg.tri_h = 0.07; cfg.tri_smooth_iters = 5;
        cfg.width = cfg.height = 12;
        cfg.colliders = { make_sphere(type == ClothType::Skirt ? Vec3(0.0, 0.55, 0.0) : Vec3(0.5, 0.6, 0.5), 0.25) };
        cfg.animation_file = ""; cfg.waist_attach_enabled = false;
        cfg.secs = 1; cfg.n_iters = 100;
        prepare_world(cfg);
        const Real dt = 1.0 / (cfg.FPS * cfg.frame_substeps);
        SimRunResult r;
        r.obj = build_run_cloth(cfg, cfg.guess_k());
        ASSERT(r.obj.energies.springs.constraints.empty() && !r.obj.energies.arap.tris.empty() && !r.obj.energies.bending.elems.empty(), "ARAP cloth has springs, no triangles or no hinges");
        init_pd_velocity(r.obj, dt);
        pd_contact(r.obj, dt, cfg.gravity, cfg.n_iters, cfg.FPS * cfg.frame_substeps * cfg.secs, cfg.frame_substeps, r.tape, "arap_check", false, false);
        const Real max_abs = r.obj.x.cwiseAbs().maxCoeff();
        std::printf("arap %s forward: %d free particles, max |x| %.3f\n", type == ClothType::Skirt ? "skirt" : "square", (int)r.obj.num_particles(), max_abs);
        ASSERT(std::isfinite(max_abs) && max_abs < 5.0, "ARAP cloth sim diverged");
    }
    return 0;
}
