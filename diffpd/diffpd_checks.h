#pragma once

// Headless self-checks (--regression, --meshgen-check, --arap-check) and diagnostics, split out of diffpd.cpp.
// Not a standalone header: it is #included once, just above main(), and uses that file's functions and globals.

#include "config_io.h" // --fd-config
#include "meshgen/pattern_io.h"
#include <numeric>

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
        c.contact_recheck_mode = ContactRecheckMode::NTimes; // the golden output predates the Once default
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
        c.contact_recheck_mode = ContactRecheckMode::NTimes; // the golden output predates the Once default
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

// Pattern contours (meshgen::PatternChart, pattern_io.h): unit checks of flattening / contains / bad input, then the generator on
// small test polygons and on both pieces of patterns/tshirt_pattern.json. Aborts (ASSERT) on any failure; part of --meshgen-check.
static void run_pattern_check()
{
    using meshgen::PatternChart;
    using meshgen::PatternSeg;
    using meshgen::PatternShape;
    using meshgen::Vec2;
    auto polygon = [](std::initializer_list<std::pair<double, double>> pts) // closed polygon of lines, pixel coordinates
    {
        PatternShape s{ Vec2(pts.begin()->first, pts.begin()->second), true, {} };
        for (auto it = pts.begin() + 1; it != pts.end(); ++it) s.segs.push_back({ false, Vec2::Zero(), Vec2::Zero(), Vec2(it->first, it->second) });
        s.segs.push_back({ false, Vec2::Zero(), Vec2::Zero(), s.start });
        return s;
    };
    auto throws = [](auto&& f) { try { f(); } catch (const std::runtime_error&) { return true; } return false; };

    // flattening: a circle of 4 cubic Beziers, area and samples on the circle (Bezier circle error ~3e-4 R)
    const double R = 100, k = 0.5522847498 * R;
    const PatternShape circle{ Vec2(R, 0), true, { { true, Vec2(R, k), Vec2(k, R), Vec2(0, R) }, { true, Vec2(-k, R), Vec2(-R, k), Vec2(-R, 0) },
                                                    { true, Vec2(-R, -k), Vec2(-k, -R), Vec2(0, -R) }, { true, Vec2(k, -R), Vec2(R, -k), Vec2(R, 0) } } };
    {
        const PatternChart c(circle, 0.001, 1e-5);
        ASSERT(std::abs(c.area() / (3.14159265358979 * 0.1 * 0.1) - 1.0) < 1e-3, "pattern: circle area off");
        for (const meshgen::BoundaryCurve& cv : c.sample_boundary(0.02))
            for (const Vec2& p : cv.p) ASSERT(std::abs(p.norm() - 0.1) < 1e-4, "pattern: a boundary sample is off the circle");
        ASSERT(c.contains(Vec2(0, 0)) && c.contains(Vec2(0.099, 0)) && !c.contains(Vec2(0.101, 0)) && !c.contains(Vec2(0.2, 0)), "pattern: circle contains()");
    }
    {   // Bezier whose controls sit on its end points (as in the example file) is a straight edge
        PatternShape sq = polygon({ { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } });
        sq.segs[0] = { true, Vec2(0, 0), Vec2(1, 0), Vec2(1, 0) };
        ASSERT(std::abs(PatternChart(sq, 1.0, 1e-3).area() - 1.0) < 1e-12, "pattern: degenerate Bezier is not a line");
    }
    // contains(): concave L (notch at the top right) and U (two collinear, non-adjacent top edges: must not count as a self crossing)
    const PatternShape L = polygon({ { 0, 0 }, { 2, 0 }, { 2, 1 }, { 1, 1 }, { 1, 2 }, { 0, 2 } });
    const PatternShape U = polygon({ { 0, 0 }, { 3, 0 }, { 3, 2 }, { 2, 2 }, { 2, 1 }, { 1, 1 }, { 1, 2 }, { 0, 2 } });
    {
        const PatternChart c(L, 1.0, 1e-3);   // centred on (1, 1), y up: pixel (x, y) -> (x - 1, 1 - y)
        auto in = [&](double x, double y) { return c.contains(Vec2(x - 1, 1 - y)); };
        ASSERT(in(0.5, 0.5) && in(1.5, 0.5) && in(0.5, 1.5) && !in(1.5, 1.5) && !in(2.5, 0.5) && !in(-0.1, 1.0), "pattern: L contains()");
        ASSERT(std::abs(c.area() - 3.0) < 1e-12, "pattern: L area");
        const PatternChart u(U, 1.0, 1e-3);
        ASSERT(std::abs(u.area() - 5.0) < 1e-12 && u.contains(Vec2(0, 0.5)) && !u.contains(Vec2(0, -0.5)) && u.contains(Vec2(-1.0, 0.0)), "pattern: U area / contains()");
    }
    // bad input throws instead of aborting
    PatternShape open = L; open.closed = false;
    PatternShape not_back = L; not_back.segs.back().to = Vec2(0.5, 0.5);
    ASSERT(throws([&] { PatternChart(open, 1.0, 1e-3); }), "pattern: an open shape must be rejected");
    ASSERT(throws([&] { PatternChart(not_back, 1.0, 1e-3); }), "pattern: a closed shape not ending at its start must be rejected");
    ASSERT(throws([&] { PatternChart(polygon({ { 0, 0 }, { 2, 2 }, { 2, 0 }, { 0, 2 } }), 1.0, 1e-3); }), "pattern: a bow-tie must be rejected");
    ASSERT(throws([&] { load_pattern("no_such_pattern_file.json"); }), "pattern: a missing file must throw");

    // generator: boundary vertex lists in order, uniform spacing inside each segment, clean surface, deterministic
    std::printf("%-30s %5s %5s %5s %4s %4s %6s %6s %7s %s\n", "pattern case", "V", "E", "F", "euler", "open", "minAng", "meanE/h", "area", "ok");
    auto mesh = [&](const char* name, const PatternShape& shape, Real scale, Real h)
    {
        const PatternChart chart(shape, scale, h / 50);
        meshgen::Params p;
        p.h = h; p.smooth_iters = 10;
        const meshgen::TriSurface s = meshgen::generate(chart, p);
        const meshgen::Report r = meshgen::validate(s, chart, p);
        std::printf("%-30s %5lld %5lld %5lld %5d %4d %6.1f %7.2f %+6.2f%% %s\n", name, (long long)r.vertices, (long long)r.edges, (long long)r.triangles,
                    r.euler, r.open_edges, r.min_angle_deg, r.edge_mean, 100 * r.area_error, r.quality_ok ? "ok" : "QUALITY");
        ASSERT(r.structure_ok && r.euler == 1, name << ": not a clean disk");
        ASSERT(std::abs(r.area_error) < 0.02, name << ": mesh area differs from the contour area");
        ASSERT(r.min_angle_deg > 15.0, name << ": a triangle has an angle below 15 degrees");
        const std::vector<meshgen::BoundaryCurve> curves = chart.sample_boundary(h);
        ASSERT(s.boundary.size() == curves.size(), name << ": boundary lists missing");
        for (size_t c = 0; c < curves.size(); ++c)
        {
            const std::vector<Index>& b = s.boundary[c];
            ASSERT(b.size() == curves[c].p.size(), name << ": a boundary curve lost a vertex");
            for (size_t i = 0; i + 1 < b.size(); ++i)
            {
                ASSERT((s.uv[b[i]] - curves[c].p[i]).norm() < 1e-9, name << ": boundary vertices are not the samples, in order");
                const Real d = (s.pos[b[i + 1]] - s.pos[b[i]]).norm();
                ASSERT(b.size() == 2 || (d > 0.55 * h && d < 1.65 * h), name << ": boundary spacing " << d << " is not ~h");
            }
        }
        const meshgen::TriSurface again = meshgen::generate(chart, p);
        ASSERT(again.tris == s.tris && again.uv.size() == s.uv.size(), name << ": generate() is not deterministic");
        meshgen::Params q = p;
        q.seed += 1;
        ASSERT(meshgen::generate(chart, q).tris != s.tris, name << ": a different seed gave the same mesh");
    };
    mesh("L (concave)", L, 0.3, 0.05);
    mesh("U (thin arms)", U, 0.2, 0.04);
    mesh("circle (4 Beziers)", circle, 0.004, 0.05);

    {   // 3D placement: a 2 x 2 px square at pivot (1, 1), rotated +90 deg about y (quaternion x y z w), through the loader and the mesher
        auto load_text = [&](const char* name, const char* text)
        {
            const std::filesystem::path f = std::filesystem::temp_directory_path() / name;
            std::ofstream(f) << text;
            return load_pattern(f.string());
        };
        const char* sq = R"("closed":true,"start":[0,0],"segments":[{"type":"line","to":[2,0]},{"type":"line","to":[2,2]},{"type":"line","to":[0,2]},{"type":"line","to":[0,0]}])";
        const std::string v4 = std::string(R"({"version":4,"shapes":[{)") + sq + R"(,"transform":{"pivot":[1,1],"position":[10,-20,5],"rotation":[0,0.70710678118,0,0.70710678118]}}],"seams":[]})";
        const std::vector<PatternShape> placed4 = load_text("diffpd_pattern_v4.json", v4.c_str());
        const PatternChart c(placed4[0], 1.0, 1e-3);   // own centre: the pivot is the origin
        ASSERT((c.to3d(Vec2(1, 0)) - Vec3(0, 0, -1)).norm() < 1e-9 && (c.to3d(Vec2(0, 1)) - Vec3(0, 1, 0)).norm() < 1e-9, "pattern: +90 deg about y must send local +x to -z");
        ASSERT((c.outward(Vec3::Zero()) - Vec3::UnitX()).norm() < 1e-9, "pattern: the front side must follow the rotation");
        const PatternChart shifted(placed4[0], 0.5, 1e-3, meshgen::pattern_center(placed4) + Vec3(2, 0, 0)); // centre 2 px left of the pivot, half scale
        ASSERT((shifted.to3d(Vec2(0, 0)) - Vec3(-1, 0, 0)).norm() < 1e-9, "pattern: position must be relative to the centre, in scale units");
        meshgen::Params p;
        p.h = 0.25; p.smooth_iters = 5;
        const meshgen::TriSurface s = meshgen::generate(c, p);
        ASSERT(meshgen::validate(s, c, p).structure_ok, "pattern: a rotated piece must mesh as a clean disk");
        for (const Vec3& v : s.pos) ASSERT(std::abs(v.x()) < 1e-9, "pattern: the rotated piece left its plane");
        for (const auto& t : s.tris)
            ASSERT((s.pos[t[1]] - s.pos[t[0]]).cross(s.pos[t[2]] - s.pos[t[0]]).x() > 0, "pattern: triangles must face the rotated +z");
        const std::string v2 = std::string(R"({"version":2,"shapes":[{)") + sq + "}]}";
        const PatternChart flat(load_text("diffpd_pattern_v2.json", v2.c_str())[0], 1.0, 1e-3);   // no transform: flat, centred on its box
        ASSERT((flat.to3d(Vec2(1, 1)) - Vec3(1, 1, 0)).norm() < 1e-9 && (flat.outward(Vec3::Zero()) - Vec3::UnitZ()).norm() < 1e-9, "pattern: a version 2 piece must stay flat");
    }

    {   // sew(): chains of boundary vertices are paired by index (flipped for `reversed`); the two chains must have the same length
        using Links = std::vector<std::pair<Index, Index>>;
        auto piece = [](int n_curves, std::vector<int> ids, std::vector<std::vector<Index>> lists, size_t n_vertices)
        {
            meshgen::TriSurface s;
            s.n_curves = n_curves; s.boundary_id = std::move(ids); s.boundary = std::move(lists); s.pos.assign(n_vertices, Vec3::Zero());
            return s;
        };
        auto strip = [&](int n) // one edge of n vertices
        {
            std::vector<Index> v;
            for (int i = 0; i < n; ++i) v.push_back(i);
            return piece(1, { 0 }, { v }, n);
        };
        const std::vector<meshgen::TriSurface> two = { strip(4), strip(4) }; // vertex ids 0..3 and 4..7
        ASSERT(meshgen::sew(two, { { 0, 0, 1, 0, false } }) == (Links{ { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 } }), "pattern: sew() same direction");
        ASSERT(meshgen::sew(two, { { 0, 0, 1, 0, true } }) == (Links{ { 0, 7 }, { 1, 6 }, { 2, 5 }, { 3, 4 } }), "pattern: sew() reversed");
        // a side of 3 edges (neighbouring edges share their corner vertex), and one of 2 edges wrapping from the last edge to edge 0
        const meshgen::TriSurface three = piece(3, { 0, 1, 2 }, { { 0, 1, 2 }, { 2, 3, 4 }, { 4, 5, 6 } }, 7);
        ASSERT(meshgen::sew({ three, strip(7) }, { { 0, 0, 1, 0, false, 3, 1 } }) == (Links{ { 0, 7 }, { 1, 8 }, { 2, 9 }, { 3, 10 }, { 4, 11 }, { 5, 12 }, { 6, 13 } }),
               "pattern: sew() side of 3 edges");
        ASSERT(meshgen::sew({ three, strip(6) }, { { 0, 2, 1, 0, false, 2, 1 } }) == (Links{ { 0, 10 }, { 1, 11 }, { 2, 12 }, { 4, 7 }, { 5, 8 }, { 6, 9 } }),
               "pattern: sew() side wrapping from the last edge to edge 0");
        const meshgen::TriSurface gap = piece(3, { 0, 2 }, { { 0, 1, 2 }, { 2, 3, 4 } }, 5); // edge 1 had zero length: no boundary list
        ASSERT(meshgen::sew({ gap, strip(5) }, { { 0, 0, 1, 0, false, 3, 1 } }) == (Links{ { 0, 5 }, { 1, 6 }, { 2, 7 }, { 3, 8 }, { 4, 9 } }), "pattern: sew() skips a zero-length edge");
        ASSERT(throws([&] { meshgen::sew({ strip(4), strip(5) }, { { 0, 0, 1, 0, false } }); }), "pattern: sides with different vertex counts must be rejected");
        ASSERT(throws([&] { meshgen::sew(two, { { 0, 3, 1, 0, false } }); }) && throws([&] { meshgen::sew(two, { { 0, 0, 2, 0, false } }); }), "pattern: a seam to a missing edge / shape must be rejected");
    }

    {   // version 4 seam fields: count and level default to 1, bad seams are rejected with a message
        auto load_seams = [&](const std::string& seam_list, bool open0) // two squares (4 edges each; shape 0 open with 2 edges when open0) + the given seams
        {
            const std::string square = R"({"closed":true,"start":[0,0],"segments":[{"type":"line","to":[2,0]},{"type":"line","to":[2,2]},{"type":"line","to":[0,2]},{"type":"line","to":[0,0]}]})";
            const std::string open = R"({"closed":false,"start":[0,0],"segments":[{"type":"line","to":[2,0]},{"type":"line","to":[2,2]}]})";
            const std::filesystem::path f = std::filesystem::temp_directory_path() / "diffpd_pattern_seams.json";
            std::ofstream(f) << R"({"version":4,"shapes":[)" << (open0 ? open : square) << "," << square << R"(],"seams":[)" << seam_list << "]}";
            std::vector<meshgen::PatternSeam> seams;
            load_pattern(f.string(), &seams);
            return seams;
        };
        const auto d = load_seams(R"({"a":{"shape":0,"edge":1},"b":{"shape":1,"edge":3},"reversed":true})", false);
        ASSERT(d.size() == 1 && d[0].edge_a == 1 && d[0].edge_b == 3 && d[0].reversed && d[0].count_a == 1 && d[0].count_b == 1 && d[0].level == 1, "pattern: seam defaults");
        const auto w = load_seams(R"({"a":{"shape":0,"edge":3,"count":2},"b":{"shape":1,"edge":0,"count":3},"reversed":false,"level":3})", false); // wraps 3 -> 0
        ASSERT(w.size() == 1 && w[0].count_a == 2 && w[0].count_b == 3 && w[0].level == 3, "pattern: seam count / level");
        ASSERT(load_seams(R"({"a":{"shape":0,"edge":0},"b":{"shape":0,"edge":2},"reversed":false})", false).size() == 1, "pattern: a seam inside one shape");
        ASSERT(load_seams(R"({"a":{"shape":0,"edge":0,"count":2},"b":{"shape":1,"edge":0},"reversed":false})", true).size() == 1, "pattern: a side fitting an open shape");
        for (const char* bad : { R"({"a":{"shape":2,"edge":0},"b":{"shape":1,"edge":0},"reversed":false})",                       // no such shape
                                 R"({"a":{"shape":0,"edge":4},"b":{"shape":1,"edge":0},"reversed":false})",                       // no such edge
                                 R"({"a":{"shape":0,"edge":0,"count":0},"b":{"shape":1,"edge":0},"reversed":false})",            // empty side
                                 R"({"a":{"shape":0,"edge":0,"count":5},"b":{"shape":1,"edge":0},"reversed":false})",            // more edges than the shape has
                                 R"({"a":{"shape":0,"edge":0},"b":{"shape":1,"edge":0},"reversed":false,"level":0})",            // level out of range
                                 R"({"a":{"shape":0,"edge":0},"b":{"shape":1,"edge":0},"reversed":false,"level":5})",
                                 R"({"a":{"shape":0,"edge":0,"count":2},"b":{"shape":0,"edge":1},"reversed":false})" })           // the sides share edge 1
            ASSERT(throws([&] { load_seams(bad, false); }), "pattern: a bad seam must be rejected: " << bad);
        ASSERT(throws([&] { load_seams(R"({"a":{"shape":0,"edge":1,"count":2},"b":{"shape":1,"edge":0},"reversed":false})", true); }), "pattern: a side running past the end of an open shape");
    }

    {   // equal vertex counts on both sides of a seam (plan_seam_intervals)
        const PatternShape A = polygon({ { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } });                                // edge 0: one segment, 1.0 long
        const PatternShape B = polygon({ { 0, 0 }, { 0.5, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } });                    // edges 0 + 1: two segments, 0.5 + 0.5
        auto charts = [&](std::initializer_list<PatternShape> shapes)
        {
            std::vector<PatternChart> v;
            for (const PatternShape& s : shapes) v.emplace_back(s, 1.0, 1e-3);
            return v;
        };
        auto points = [](const PatternChart& c, Real h, int id) // samples on input segment `id`
        {
            for (const meshgen::BoundaryCurve& cv : c.sample_boundary(h)) if (cv.id == id) return cv.p.size();
            return size_t(0);
        };
        auto cs = charts({ A, B });
        ASSERT(points(cs[1], 0.3, 0) + points(cs[1], 0.3, 1) - 1 == 5, "pattern: without seams the two halves get round(L/h) each");
        meshgen::plan_seam_intervals(cs, { { 0, 0, 1, 0, false, 1, 2 } }, 0.3);                                     // N = round(1 / 0.3) = 3
        ASSERT(points(cs[0], 0.3, 0) == 4 && points(cs[1], 0.3, 0) + points(cs[1], 0.3, 1) - 1 == 4, "pattern: both sides of the seam must have N + 1 = 4 vertices");
        ASSERT(points(cs[1], 0.3, 0) == 3 && points(cs[1], 0.3, 1) == 2, "pattern: N = 3 splits 2 + 1 between two equal halves (ties to the first)");
        ASSERT(points(cs[1], 0.3, 2) == points(charts({ A, B })[1], 0.3, 2), "pattern: a segment in no seam keeps round(L/h)");
        auto bad_cs = charts({ A, polygon({ { 0, 0 }, { 2, 0 }, { 2, 1 }, { 0, 1 } }) });
        ASSERT(throws([&] { meshgen::plan_seam_intervals(bad_cs, { { 0, 0, 1, 0, false } }, 0.3); }), "pattern: sides of very different length must be rejected");
        auto coarse = charts({ A, B });
        ASSERT(throws([&] { meshgen::plan_seam_intervals(coarse, { { 0, 0, 1, 0, false, 1, 2 } }, 1.0); }), "pattern: a side with more edges than intervals must be rejected");
        auto conflict = charts({ A, B, polygon({ { 0, 0 }, { 1.1, 0 }, { 1.1, 1 }, { 0, 1 } }) });
        ASSERT(throws([&] { meshgen::plan_seam_intervals(conflict, { { 0, 0, 1, 0, false, 1, 2 }, { 0, 0, 2, 0, false } }, 0.29); }),
               "pattern: a segment in two seams needing different counts must be rejected");

        // meshed: a side of two edges sewn to a side of one edge gives one link per vertex, paired within half an interval along the seam
        auto sew_cs = charts({ A, B });
        const std::vector<meshgen::PatternSeam> two_one = { { 0, 0, 1, 0, false, 1, 2 } };
        meshgen::plan_seam_intervals(sew_cs, two_one, 0.3);
        meshgen::Params sp;
        sp.h = 0.3; sp.smooth_iters = 3;
        std::vector<meshgen::TriSurface> sewn;
        for (const PatternChart& c : sew_cs) sewn.push_back(meshgen::generate(c, sp));
        const auto sewn_links = meshgen::sew(sewn, two_one);
        ASSERT(sewn_links.size() == 4, "pattern: 2 edges sewn to 1 edge must give N + 1 = 4 links, got " << sewn_links.size());
        for (const auto& [a, b] : sewn_links)
        {
            const Vec3 pa = sewn[0].pos[a], pb = sewn[1].pos[b - (Index)sewn[0].pos.size()];
            ASSERT(std::abs(pa.y() - pb.y()) < 1e-9 && std::abs(pa.x() - pb.x()) <= 0.5 / 3 + 1e-6, "pattern: a sewn pair is off along the seam by " << std::abs(pa.x() - pb.x()));
        }

        // curved sides: a quarter circle (one cubic, radius 2 m) sewn to the same arc running the other way, and to the same arc split in two cubics.
        // Both pieces share one frame, so the length of a link is how far apart its two vertices are along the seam.
        const double q = 0.5522847498; // cubic quarter circle
        const Vec2 o = Vec2::Zero(), P0(1, 0), P1(1, q), P2(q, 1), P3(0, 1);
        const PatternShape arc{ P0, true, { { true, P1, P2, P3 }, { false, o, o, Vec2(0, 0) }, { false, o, o, P0 } } };
        const PatternShape arc_back{ P3, true, { { true, P2, P1, P0 }, { false, o, o, Vec2(0, 0) }, { false, o, o, P3 } } };
        const Vec2 q1 = 0.5 * (P0 + P1), q2 = 0.25 * (P0 + 2.0 * P1 + P2), mid = 0.125 * (P0 + 3.0 * P1 + 3.0 * P2 + P3), r1 = 0.25 * (P1 + 2.0 * P2 + P3), r2 = 0.5 * (P2 + P3);
        const PatternShape arc_split{ P0, true, { { true, q1, q2, mid }, { true, r1, r2, P3 }, { false, o, o, Vec2(0, 0) }, { false, o, o, P0 } } };
        auto curved = [&](const PatternShape& s0, const PatternShape& s1, const meshgen::PatternSeam& seam) // { number of links, longest link }
        {
            std::vector<PatternChart> v;
            v.emplace_back(s0, 2.0, 1e-3);
            v.emplace_back(s1, 2.0, 1e-3);
            meshgen::plan_seam_intervals(v, { seam }, 0.28);
            meshgen::Params cp;
            cp.h = 0.28; cp.smooth_iters = 3;
            std::vector<meshgen::TriSurface> m;
            for (const PatternChart& c : v) m.push_back(meshgen::generate(c, cp));
            const auto ls = meshgen::sew(m, { seam });
            Real longest = 0;
            for (const auto& [a, b] : ls) longest = std::max(longest, (m[0].pos[a] - m[1].pos[b - (Index)m[0].pos.size()]).norm());
            return std::make_pair(ls.size(), longest);
        };
        const auto same = curved(arc, arc_back, { 0, 0, 1, 0, true });    // same curve, opposite direction: the pairs must coincide
        ASSERT(same.first == 12 && same.second < 2e-3, "pattern: an arc sewn to itself reversed: " << same.first << " links, longest " << same.second);
        const auto split = curved(arc, arc_split, { 0, 0, 1, 0, false, 1, 2 }); // N = 11 is odd: the corner of the split arc cannot sit on a vertex of the other side
        ASSERT(split.first == 12 && split.second <= 0.5 * (3.1416 / 11) * 1.02 + 2e-3, "pattern: an arc sewn to the same arc split in two: " << split.first << " links, longest " << split.second);
    }

    {   // corners shared by two seams, and a seam inside one piece (a tube)
        auto edges = [](std::vector<int> ids, std::vector<std::vector<Index>> lists, size_t n_vertices) // a square piece with 4 edges, only some sampled
        {
            meshgen::TriSurface s;
            s.n_curves = 4; s.boundary_id = std::move(ids); s.boundary = std::move(lists); s.pos.assign(n_vertices, Vec3::Zero());
            return s;
        };
        using Links = std::vector<std::pair<Index, Index>>;
        // P edges 1 (ends at corner 4) and 2 (starts at corner 4); Q edges 3 (ends at corner 4) and 0 (starts at corner 4); Q's vertex ids start at 9
        const meshgen::TriSurface P = edges({ 1, 2 }, { { 0, 1, 2, 3, 4 }, { 4, 5, 6, 7, 8 } }, 9), Q = edges({ 3, 0 }, { { 0, 1, 2, 3, 4 }, { 4, 5, 6, 7, 8 } }, 9);
        const Links consistent = meshgen::sew({ P, Q }, { { 0, 1, 1, 3, false }, { 0, 2, 1, 0, false } }); // end meets end, start meets start: both give corner <-> corner
        ASSERT(consistent == (Links{ { 0, 9 }, { 1, 10 }, { 2, 11 }, { 3, 12 }, { 4, 13 }, { 5, 14 }, { 6, 15 }, { 7, 16 }, { 8, 17 } }),
               "pattern: the corner shared by two seams must be linked once (9 links, got " << consistent.size() << ")");
        const Links conflicting = meshgen::sew({ P, Q }, { { 0, 1, 1, 3, true }, { 0, 2, 1, 0, false } }); // the corner of P is asked to meet two vertices of Q: both links kept
        ASSERT(conflicting.size() == 10 && std::count_if(conflicting.begin(), conflicting.end(), [](const auto& l) { return l.first == 4; }) == 2,
               "pattern: a vertex asked to meet two vertices keeps both links, got " << conflicting.size());

        // levels: every link carries the level of its seam, the corner shared by a level 2 and a level 1 seam keeps level 1
        std::vector<int> lv;
        const Links leveled = meshgen::sew({ P, Q }, { { 0, 1, 1, 3, false, 1, 1, 2 }, { 0, 2, 1, 0, false, 1, 1, 1 } }, &lv);
        ASSERT(leveled == consistent && lv == (std::vector<int>{ 2, 2, 2, 2, 1, 1, 1, 1, 1 }), "pattern: sew() levels (the shared corner keeps the lower one)");

        // SeamEnergy: only the links up to max_level are in K and in the force; only the links of pull_level are pulled
        SeamEnergy se;
        se.k = 5.0;
        se.add_link(0, 1, 1); se.add_link(2, 3, 1); se.add_link(4, 5, 2); se.add_link(6, 7, 3); se.add_link(0, 0, 1);
        ASSERT(se.links.size() == 4 && se.level.size() == 4 && se.highest_level() == 3, "pattern: SeamEnergy links / levels");
        std::vector<Triplet> tr;
        for (int m = 1; m <= 4; ++m)
        {
            se.max_level = m;
            tr.clear();
            se.add_stiffness_triplets(tr, 1.0);
            ASSERT(tr.size() == 12 * (size_t)(m == 1 ? 2 : m == 2 ? 3 : 4), "pattern: SeamEnergy stiffness triplets up to level " << m << ": " << tr.size());
        }
        se.max_level = 4; se.offset.assign(4, Vec3(1, 0, 0)); se.pull = 0.5; se.pull_level = 2;
        RealVecX fb = RealVecX::Zero(24);
        se.add_forces(RealVecX::Zero(24), {}, fb, 2.0);
        ASSERT(std::abs(fb(12) - 5.0) < 1e-12 && std::abs(fb(15) + 5.0) < 1e-12 && fb.squaredNorm() < 50.0 + 1e-9, "pattern: SeamEnergy pulls only the links of pull_level");
        se.max_level = 1;
        fb.setZero();
        se.add_forces(RealVecX::Zero(24), {}, fb, 2.0);
        ASSERT(fb.squaredNorm() == 0.0, "pattern: a pulled level above max_level has no force");

        // a tube: the right edge of one rectangular piece sewn to its own left edge
        const PatternShape sheet = polygon({ { 0, 0 }, { 1.2, 0 }, { 1.2, 0.6 }, { 0, 0.6 } });                    // edges: 0 bottom, 1 right, 2 top, 3 left
        const std::vector<meshgen::PatternSeam> tube_seams = { { 0, 1, 0, 3, true } };
        std::vector<PatternChart> tc;
        tc.emplace_back(sheet, 1.0, 1e-3);
        meshgen::plan_seam_intervals(tc, tube_seams, 0.1);
        meshgen::Params tp;
        tp.h = 0.1; tp.smooth_iters = 3;
        const std::vector<meshgen::TriSurface> tube = { meshgen::generate(tc[0], tp) };
        meshgen::TriSurface tube_mesh = meshgen::merge(tube);
        tube_mesh.seams = meshgen::sew(tube, tube_seams);
        ASSERT(tube_mesh.seams.size() == 7, "pattern: a tube seam of 6 intervals must give 7 links, got " << tube_mesh.seams.size());
        for (const auto& [a, b] : tube_mesh.seams)
            ASSERT(a != b && std::abs(std::abs(tube_mesh.pos[a].x() - tube_mesh.pos[b].x()) - 1.2) < 1e-9, "pattern: a tube link must join the left and the right edge");
        AppConfig t;
        t.colliders.clear(); t.animation_file = ""; t.waist_attach_enabled = false;
        prepare_world(t);
        Object tube_obj = tri_cloth(tube_mesh, {}, Stiffness::uniform(2.0), Vec3::Zero(), 0.5, TriEnergy::Springs, true, 200.0);
        const Real tube_gap0 = tube_obj.energies.seams.max_gap(tube_obj.x);
        init_pd_velocity(tube_obj, 1.0 / 60.0);
        Tape tube_tape;
        pd_contact(tube_obj, 1.0 / 60.0, t.gravity, 100, 60, 1, tube_tape, "tube_check", false, false);
        std::printf("tube: %zu links, longest %.4f m at the start, %.4f m after 60 steps\n", tube_mesh.seams.size(), tube_gap0, tube_obj.energies.seams.max_gap(tube_obj.x));
        ASSERT(tube_obj.x.allFinite() && tube_obj.energies.seams.max_gap(tube_obj.x) < 0.2 * tube_gap0, "pattern: the tube seam did not close");
    }

    const std::filesystem::path file = std::filesystem::path(__FILE__).parent_path() / "patterns" / "tshirt_pattern.json";
    if (!std::filesystem::exists(file)) { std::printf("(skipped: %s not found)\n", file.string().c_str()); return; }
    std::vector<meshgen::PatternSeam> shirt_seams;
    const std::vector<PatternShape> shirt = load_pattern(file.string(), &shirt_seams);
    ASSERT(shirt_seams.size() == 4 && std::all_of(shirt_seams.begin(), shirt_seams.end(), [](const meshgen::PatternSeam& s) { return s.count_a == 1 && s.count_b == 1 && s.level == 1; }),
           "pattern: the t-shirt has 4 single-edge level 1 seams");
    ASSERT(shirt.size() == 2, "pattern: the t-shirt file has two pieces");
    for (size_t i = 0; i < shirt.size(); ++i)
        for (Real h : { 0.02, 0.03, 0.05 })
        {
            char name[48];
            std::snprintf(name, sizeof(name), "t-shirt piece %zu h %.2f", i, h);
            mesh(name, shirt[i], 0.001, h);
        }

    // wired in (ClothType::Pattern): both pieces as one pin-free object placed in 3D as in the file, bad input is an error not a crash, and it simulates
    AppConfig cfg;
    cfg.cloth_type = ClothType::Pattern;
    cfg.tri_h = 0.05; cfg.tri_smooth_iters = 5;
    cfg.animation_file = ""; cfg.waist_attach_enabled = false;
    cfg.colliders = { make_plane(Vec3(0.0, 0.3, 0.0), Vec3::UnitY()) };
    cfg.secs = 1; cfg.n_iters = 100;
    ASSERT(cfg.tri_ok(), "pattern: the default pattern config is rejected: " << tri_surface_for(cfg).error);
    const TriCache& cache = tri_surface_for(cfg);
    ASSERT(cache.report.structure_ok && cache.report.euler == 2, "pattern: two pieces must give two clean disks");
    Object obj = build_cloth(cfg, Stiffness::uniform(2.0), cfg.origin);
    ASSERT(obj.mesh.pinned_rest.empty() && obj.num_particles() == (Index)cache.surface.pos.size(), "pattern: no vertex may be pinned");
    ASSERT(cache.surface.seams.size() >= 4 && obj.mesh.seams == cache.surface.seams && obj.energies.seams.links.size() == obj.mesh.seams.size(),
           "pattern: the t-shirt's 4 seams must give sewing links");
    auto seam_gap = [](const Object& o) // longest link, m
    {
        Real g = 0;
        for (const auto& [a, b] : o.mesh.seams) g = std::max(g, (o.mesh.position(o, a) - o.mesh.position(o, b)).norm());
        return g;
    };
    const Real gap0 = seam_gap(obj);
    ASSERT(gap0 > 0.25, "pattern: the pieces should start apart (gap " << gap0 << ")");
    Vec3 lo = Vec3::Constant(1e9), hi = Vec3::Constant(-1e9);
    for (const Vec3& v : cache.surface.pos) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
    ASSERT(std::abs((hi.x() - lo.x()) - 0.46) < 0.02 && std::abs((hi.z() - lo.z()) - 0.303) < 0.01, "pattern: the pieces lost their placement from the file (width "
           << hi.x() - lo.x() << ", depth " << hi.z() - lo.z() << ")");
    AppConfig bad = cfg;
    bad.pattern_scale = 0.0;
    ASSERT(!bad.tri_ok() && !tri_surface_for(bad).error.empty(), "pattern: a zero scale must be an error");
    bad = cfg;
    bad.pattern_file = "no_such_pattern.json";
    ASSERT(!bad.tri_ok(), "pattern: a missing file must be an error");
    prepare_world(cfg);
    const Real dt = 1.0 / (cfg.FPS * cfg.frame_substeps);
    const int  n_steps = cfg.FPS * cfg.frame_substeps * cfg.secs;
    Object sim = build_run_cloth(cfg, Stiffness::uniform(2.0));
    init_pd_velocity(sim, dt);
    Tape tape;
    pd_contact(sim, dt, cfg.gravity, cfg.n_iters, n_steps, cfg.frame_substeps, tape, "pattern_check", /*export_obj=*/false, /*verbose=*/false,
               nullptr, cfg.unresolved_contact_threshold);
    size_t n_contacts = 0;
    for (const Contacts& c : tape.contacts) n_contacts += c.size();
    ASSERT(sim.x.allFinite() && sim.x.cwiseAbs().maxCoeff() < 10.0, "pattern: the free-falling pieces diverged");
    std::printf("pattern seams: %zu links, longest %.4f m at the start, %.4f m after %d steps\n", obj.mesh.seams.size(), gap0, seam_gap(sim), n_steps);
    ASSERT(seam_gap(sim) < 0.2 * gap0, "pattern: the seams did not pull the pieces together");
    {   // "Settle seams": no gravity, the links close smoothly from the 0.3 m start instead of snapping; "Settle cloth" then continues from that pose
        AppConfig s = cfg;
        s.seam_settle_seconds = 1.0; s.seam_settle_iters = 30; s.settle_seconds = 1.0;
        settle_cloth(s, /*headless=*/true, /*seams_only=*/true);
        ASSERT(s.settled.valid && s.settled.seams_only && s.settled.x.allFinite(), "pattern: settle seams produced no pose");
        std::printf("settle seams: longest link %.5f m (from %.4f), final speed %.4f m/s\n", s.settled.seam_gap, gap0, s.settled.max_speed); std::fflush(stdout);
        ASSERT(s.settled.seam_gap < 0.005, "pattern: the seams did not close gently");
        ASSERT(seam_gap(build_run_cloth(s, Stiffness::uniform(2.0))) < 0.005, "pattern: a real run must start from the sewn pose");
        settle_cloth(s, /*headless=*/true);
        ASSERT(s.settled.valid && !s.settled.seams_only && s.settled.x.allFinite(), "pattern: settle cloth after the seams failed");
        std::printf("settle cloth from the sewn pose: longest link %.5f m, final speed %.4f m/s\n", seam_gap(build_run_cloth(s, Stiffness::uniform(2.0))), s.settled.max_speed);
        ASSERT(seam_gap(build_run_cloth(s, Stiffness::uniform(2.0))) < 0.02, "pattern: the seams opened again while settling the cloth");
    }
    {   // dphi/dk with the seams in K: they have no parameter of their own, so the stretch k gradient must still match finite differences
        AppConfig f = cfg;
        f.colliders.clear(); // contact flips between k +- eps would hide the seam's own effect (they break the FD check with or without seams)
        prepare_world(f);
        f.per_type_stiffness = false;
        f.contact_recheck_mode = ContactRecheckMode::Once;
        f.n_iters = 1000; f.n_iters_adjoint = 1000;
        auto simulate = [&](Real k)
        {
            SimRunResult r;
            r.obj = build_run_cloth(f, Stiffness::uniform(k));
            init_pd_velocity(r.obj, dt);
            pd_contact(r.obj, dt, f.gravity, f.n_iters, n_steps, f.frame_substeps, r.tape, "pattern_check", false, false, nullptr, f.unresolved_contact_threshold);
            return r;
        };
        SimRunResult target = simulate(f.target_stiffness), guess = simulate(f.stiffness);
        const Loss loss(guess.tape, target.tape, f.frame_substeps);
        const BackwardGradContact grad = backward_pd_contact(guess.obj, guess.tape, loss, f.gravity, f.n_iters_adjoint, dt);
        for (Real eps : { 1e-4, 1e-6 })
        {
            const FDCheckResult fd = fd_check_contact_stiffness([&](Real k) { return build_run_cloth(f, Stiffness::uniform(k)); }, target.tape, f.stiffness, dt,
                f.gravity, f.n_iters, n_steps, f.frame_substeps, f.frame_substeps, grad.dphi_dk.sum(), eps);
            std::printf("  sewn shirt FD eps %.0e: fd %.8g analytic %.8g rel_err %.2e\n", eps, fd.fd, fd.analytic, fd.rel_err);
            ASSERT(eps > 1e-5 || fd.rel_err < 1e-3, "pattern: dphi/dk of the sewn shirt does not match central differences");
        }
    }
    std::printf("pattern cloth: %d free particles, no pins, %zu ground contacts over %d steps, ok\n", (int)sim.num_particles(), n_contacts, n_steps);

    // every pattern file shipped in patterns/ must load, mesh and sew with the default settings (last: tri_surface_for's cache is overwritten)
    for (const std::string& name : list_pattern_files())
    {
        AppConfig c = cfg;
        c.pattern_file = name;
        const TriCache& pc = tri_surface_for(c);
        std::printf("pattern file %-42s %zu vertices, %zu seam links%s%s\n", name.c_str(), pc.surface.pos.size(), pc.surface.seams.size(), pc.error.empty() ? "" : "  ERROR: ", pc.error.c_str());
        ASSERT(pc.error.empty() && pc.report.structure_ok, "pattern file " << name << " does not load: " << pc.error);
    }

    // Settle seams level by level on the sleeves file (8 seams of level 1, 2 of level 2): both levels end closed, nothing blows up
    const std::vector<std::string> files = list_pattern_files();
    if (std::find(files.begin(), files.end(), "tshirt_sleeves_pattern.json") != files.end())
    {
        AppConfig c = cfg;
        c.pattern_file = "tshirt_sleeves_pattern.json";
        c.colliders.clear();
        c.seam_settle_seconds = 1.0; c.seam_settle_iters = 30;
        const Object before = build_run_cloth(c, Stiffness::uniform(2.0));
        ASSERT(before.energies.seams.highest_level() == 2, "pattern: the sleeves file should have seams of levels 1 and 2");
        const Real gap1_before = before.energies.seams.max_gap(before.x, 1), gap2_before = before.energies.seams.max_gap(before.x, 2);
        settle_cloth(c, /*headless=*/true, /*seams_only=*/true);
        ASSERT(c.settled.valid && c.settled.seams_only && c.settled.x.allFinite(), "pattern: settle seams on the sleeves file failed");
        const Object after = build_run_cloth(c, Stiffness::uniform(2.0));
        const Real gap1 = after.energies.seams.max_gap(after.x, 1), gap2 = after.energies.seams.max_gap(after.x, 2);
        std::printf("settle seams on the sleeves file: longest link level 1 %.4f -> %.5f m, all levels %.4f -> %.5f m\n", gap1_before, gap1, gap2_before, gap2);
        ASSERT(gap1 < 0.005 && gap2 < 0.005, "pattern: the sleeves seams did not close (level 1 " << gap1 << ", all " << gap2 << ")");
    }
}

// `diffpd --meshgen-check`: headless checks of the triangle-mesh generator (tri_surface.h) over a few cones, then a small
// tri-skirt sim with a dphi/dk central-difference check. Aborts (ASSERT) on any failure.
static int run_meshgen_check()
{
    run_pattern_check();
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

// `diffpd --speculative-check`: gradient check of the experimental speculative (gap-aware) contact against the
// current model, on a 10x10 quad cloth pinned at two corners: static sphere, tilted plane, sphere moving up, capsule
// rotating about Y. Converged solves with Once (fixed contact set) so FD differentiates the same map. Prints
// dphi/dk and the uniform-offset dphi/dx0, dphi/dv0 against central differences. Report only, no asserts.
static int run_speculative_check()
{
    const Vec3 center(1.0, 0.3, 1.0);
    Collider rotating = make_capsule(Vec3(0.4, 0.3, 1.0), Vec3(1.6, 0.3, 1.0), 0.3);
    rotating.rotation_axis = RotationAxis::Y; rotating.rotation_origin = center; rotating.omega = 1.0;
    const std::pair<const char*, Collider> cases[] = {
        { "static sphere",    make_sphere(center, 0.5) },
        { "tilted plane",     make_plane(Vec3(1.0, 0.2, 1.0), Vec3(0.3, 1.0, 0.0)) },
        { "moving sphere",    make_sphere(center, 0.5, Vec3(0.0, 0.3, 0.0)) },
        { "rotating capsule", rotating },
    };
    const std::pair<bool, Real> modes[] = { { false, 0.0 }, { true, 0.0 }, { true, 0.05 } }; // { speculative, margin }
    for (const auto& [name, collider] : cases)
        for (const auto& [spec, margin] : modes)
        {
            AppConfig cfg;
            cfg.cloth_type = ClothType::Square; cfg.mesh_gen = MeshGen::Quad;
            cfg.width = cfg.height = 10; cfg.size_x = cfg.size_y = 2.0;
            cfg.pin_mode = PinMode::CORNERS;
            cfg.per_type_stiffness = false;
            cfg.colliders = { collider };
            cfg.animation_file = ""; cfg.waist_attach_enabled = false;
            cfg.contact_recheck_mode = ContactRecheckMode::Once;
            cfg.speculative_contact = spec;
            cfg.contact_margin = margin;
            cfg.secs = 1; cfg.n_iters = 1000; cfg.n_iters_adjoint = 1000;
            prepare_world(cfg);
            const Real dt = 1.0 / (cfg.FPS * cfg.frame_substeps);
            const int  n_steps = cfg.FPS * cfg.frame_substeps * cfg.secs;
            auto build = [&](Real k) { return build_run_cloth(cfg, Stiffness::uniform(k)); };
            auto simulate = [&](Real k)
            {
                SimRunResult r;
                r.obj = build(k);
                init_pd_velocity(r.obj, dt);
                pd_contact(r.obj, dt, cfg.gravity, cfg.n_iters, n_steps, cfg.frame_substeps, r.tape, "speculative_check",
                           /*export_obj=*/false, /*verbose=*/false);
                return r;
            };
            SimRunResult target = simulate(cfg.target_stiffness);
            SimRunResult guess  = simulate(cfg.stiffness);
            const Loss loss(guess.tape, target.tape, cfg.frame_substeps);
            const BackwardGradContact grad = backward_pd_contact(guess.obj, guess.tape, loss, cfg.gravity, cfg.n_iters_adjoint, dt);
            size_t n_contacts = 0;
            for (const Contacts& c : guess.tape.contacts) n_contacts += c.size();
            Real worst_est = 0.0;
            for (const StepResidual& r : guess.tape.forward_residual) worst_est = std::max(worst_est, r.est_rel);
            std::printf("== %s, speculative %s, margin %.2f: %zu contacts, loss %.8g, worst fwd est_rel %.1e\n",
                        name, spec ? "ON" : "off", margin, n_contacts, loss.total, worst_est);
            for (Real eps : { 1e-4, 1e-6 })
            {
                const FDCheckResult fd = fd_check_contact_stiffness(build, target.tape, cfg.stiffness, dt, cfg.gravity,
                    cfg.n_iters, n_steps, cfg.frame_substeps, cfg.frame_substeps, grad.dphi_dk.sum(), eps);
                std::printf("  dphi/dk eps %.0e: fd %.8g analytic %.8g rel_err %.2e\n", eps, fd.fd, fd.analytic, fd.rel_err);
            }
            std::fflush(stdout);
            fd_check_contact_offsets([&] { return build(cfg.stiffness); }, target.tape, dt, cfg.gravity, cfg.n_iters, n_steps,
                                     cfg.frame_substeps, cfg.frame_substeps, guess.obj.num_dofs(), grad.dphi_dx, grad.dphi_dv);
            std::cout << std::flush;
        }
    return 0;
}

// `diffpd --fd-config <file.conf>`: the UI's Guess + Target + Backward pipeline headless on a saved config, then the
// stiffness FD check (dphi/dk, or dphi/dk_stretch per-type) for every epsilon selected in the file (fd_eps0..8). Starts
// from the procedural cloth (a config file never holds a settled pose), or with `--settle` from the pose the "Settle
// cloth" button would compute. Prints timings and the forward convergence.
static int run_fd_config(const std::string& path, bool settle)
{
    AppConfig cfg;
    ASSERT(load_config(path, cfg), "cannot load config " << path);
    ASSERT(cfg.backward_enabled(), "config is forward-only (run_mode 1)");
    if (settle) settle_cloth(cfg, /*headless=*/true);
    const WorldSetup world = prepare_world(cfg);
    const Real dt      = 1.0 / (cfg.FPS * cfg.frame_substeps);
    const int  n_steps = cfg.FPS * cfg.frame_substeps * cfg.secs;
    auto seconds_since = [](auto t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    auto build = [&](const Stiffness& k)
    {
        Object obj = build_run_cloth(cfg, k);
        attach_waist(obj, world.waist_attach_anim_id, world.waist_attach_skinned_id, collider_animations);
        return obj;
    };
    auto simulate = [&](const Stiffness& k)
    {
        SimRunResult r;
        r.obj = build(k);
        init_pd_velocity(r.obj, dt);
        pd_contact(r.obj, dt, cfg.gravity, cfg.n_iters, n_steps, cfg.frame_substeps, r.tape, "fd_config",
                   /*export_obj=*/false, /*verbose=*/false, nullptr, cfg.unresolved_contact_threshold);
        return r;
    };
    {
        Object probe = build(cfg.guess_k());
        int free_inside = 0, pinned_inside = 0;
        count_cloth_inside_skinned(probe, free_inside, pinned_inside);
        std::printf("[fd-config] %s: %d free particles, %d steps x %d iters (adjoint %d), speculative %d margin %g, settled %d, start inside body: %d free %d pinned\n",
                    path.c_str(), (int)probe.num_particles(), n_steps, cfg.n_iters, cfg.n_iters_adjoint,
                    (int)cfg.speculative_contact, cfg.contact_margin, (int)cfg.settled.valid, free_inside, pinned_inside);
    }
    auto t0 = std::chrono::steady_clock::now();
    const SimRunResult target = simulate(cfg.target_k());
    const double t_forward = seconds_since(t0);
    SimRunResult guess = simulate(cfg.guess_k());
    const Loss loss(guess.tape, target.tape, cfg.frame_substeps);
    t0 = std::chrono::steady_clock::now();
    const BackwardGradContact grad = backward_pd_contact(guess.obj, guess.tape, loss, cfg.gravity, cfg.n_iters_adjoint, dt);
    const double t_backward = seconds_since(t0);

    std::vector<Real> est;
    for (const StepResidual& r : guess.tape.forward_residual) est.push_back(r.est_rel);
    std::sort(est.begin(), est.end(), std::greater<Real>());
    const size_t n_tail = std::max<size_t>(1, est.size() / 10);
    const Real tail10 = std::accumulate(est.begin(), est.begin() + n_tail, 0.0) / n_tail;
    size_t n_contacts = 0;
    for (const Contacts& c : guess.tape.contacts) n_contacts += c.size();
    const Real analytic = cfg.per_type_k() ? grad.dphi_dk.stretch : grad.dphi_dk.sum();
    std::printf("[fd-config] loss %.10g, %s %.10g (stretch %.6g shear %.6g bend %.6g), contacts %zu, fwd est_rel tail10 %.1e max %.1e, "
                "forward %.1f s, backward %.1f s\n", loss.total, cfg.per_type_k() ? "dphi/dk_stretch" : "dphi/dk", analytic,
                grad.dphi_dk.stretch, grad.dphi_dk.shear, grad.dphi_dk.bend, n_contacts, tail10, est.front(), t_forward, t_backward);
    std::fflush(stdout);

    const Stiffness k0 = cfg.guess_k();
    auto build_k = [&](Real k) { return build(cfg.per_type_k() ? Stiffness{ k, k0.shear, k0.bend } : Stiffness::uniform(k)); };
    for (int i = 0; i < 9; ++i)
    {
        if (!cfg.fd_eps_selected[i]) continue;
        t0 = std::chrono::steady_clock::now();
        const FDCheckResult r = fd_check_contact_stiffness(build_k, target.tape, k0.stretch, dt, cfg.gravity, cfg.n_iters, n_steps,
                                                           cfg.frame_substeps, cfg.frame_substeps, analytic, kFDEpsilonValues[i]);
        std::printf("[fd-config] FD eps %.0e: fd %.10g analytic %.10g rel_err %.2e (%.1f s)\n",
                    kFDEpsilonValues[i], r.fd, r.analytic, r.rel_err, seconds_since(t0));
        std::fflush(stdout);
    }
    return 0;
}

// `diffpd --contact-diff <file.conf> <eps> [--settle]`: runs the guess at k_stretch (or unified k) +- eps and prints,
// every 10 steps and at every step where they disagree, the trajectory sensitivity |x+ - x-| / 2eps (max over particles,
// and the particle), how many particles differ in contact membership / collider / mesh feature, and push-out stats
// (contacts starting inside, the deepest gap). Localizes where a converged, eps-consistent FD mismatch comes from.
static int run_contact_diff(const std::string& path, Real eps, bool settle)
{
    AppConfig cfg;
    ASSERT(load_config(path, cfg), "cannot load config " << path);
    if (settle) settle_cloth(cfg, /*headless=*/true);
    const WorldSetup world = prepare_world(cfg);
    const Real dt      = 1.0 / (cfg.FPS * cfg.frame_substeps);
    const int  n_steps = cfg.FPS * cfg.frame_substeps * cfg.secs;
    const Stiffness k0 = cfg.guess_k();
    auto simulate = [&](Real k)
    {
        SimRunResult r;
        r.obj = build_run_cloth(cfg, cfg.per_type_k() ? Stiffness{ k, k0.shear, k0.bend } : Stiffness::uniform(k));
        attach_waist(r.obj, world.waist_attach_anim_id, world.waist_attach_skinned_id, collider_animations);
        init_pd_velocity(r.obj, dt);
        pd_contact(r.obj, dt, cfg.gravity, cfg.n_iters, n_steps, cfg.frame_substeps, r.tape, "contact_diff", false, false);
        return r;
    };
    const SimRunResult p = simulate(k0.stretch + eps), m = simulate(k0.stretch - eps);
    std::printf("step   time  max|dx|/2eps  particle  contacts+  contacts-  set_diff  feat_diff  inside  deepest_gap\n");
    for (int t = 1; t <= n_steps; ++t)
    {
        const PointsX d = (p.tape.positions[t] - m.tape.positions[t]) / (2.0 * eps);
        Index worst = 0;
        const Real sens = d.rowwise().norm().maxCoeff(&worst);
        std::map<ParticleId, const Contact*> cm;
        for (const Contact& c : m.tape.contacts[t - 1]) cm[c.particle] = &c;
        int set_diff = 0, feat_diff = 0, inside = 0;
        Real deepest = 0.0;
        for (const Contact& c : p.tape.contacts[t - 1])
        {
            inside += c.gap < 0.0;
            deepest = std::min(deepest, c.gap);
            const auto it = cm.find(c.particle);
            if (it == cm.end() || it->second->collider_id != c.collider_id) { ++set_diff; continue; }
            feat_diff += it->second->triangle_id != c.triangle_id || it->second->feature != c.feature;
            cm.erase(it);
        }
        set_diff += (int)cm.size();
        if (t % 10 == 0 || set_diff || feat_diff)
            std::printf("%5d  %5.3f  %12.4g  %8d  %9zu  %9zu  %8d  %9d  %6d  %11.3e\n", t, t * dt, sens, (int)worst,
                        p.tape.contacts[t - 1].size(), m.tape.contacts[t - 1].size(), set_diff, feat_diff, inside, deepest);
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

// `diffpd --anderson-check`: Anderson acceleration (anderson.h) on (1) a linear contraction v <- A v + b (eigenvalues in (0, 0.99], like PD) with spectral
// radius 0.99, where it must be >= 100x closer to the fixed point than the plain iteration (3e-4 vs 0.3, as a numpy reference), and (2) the gap-aware 10x10
// cloth on a sphere, where 150 accelerated iterations must land on the pose of 1000 plain ones (same fixed point).
static int run_anderson_check()
{
    {
        const Index n = 60;
        RealVecX a(n), b(n), v_plain = RealVecX::Zero(n), v_aa = RealVecX::Zero(n);
        for (Index i = 0; i < n; ++i) { a(i) = 0.99 * (i + 1) / n; b(i) = 1.0 + 0.01 * i; }
        const RealVecX v_star = b.cwiseQuotient(RealVecX::Ones(n) - a);
        Anderson aa(5, n);
        for (int k = 0; k < 100; ++k)
        {
            v_plain = a.cwiseProduct(v_plain) + b;
            RealVecX g = a.cwiseProduct(v_aa) + b;
            aa.apply(v_aa, g);
            v_aa = g;
        }
        const Real err_plain = (v_plain - v_star).norm() / v_star.norm(), err_aa = (v_aa - v_star).norm() / v_star.norm();
        std::printf("linear contraction, 100 iterations: rel error plain %.1e, Anderson %.1e\n", err_plain, err_aa);
        std::fflush(stdout);
        ASSERT(err_aa < 1e-3 && err_plain > 1e-1, "Anderson does not accelerate a linear contraction");
    }
    AppConfig cfg;
    cfg.cloth_type = ClothType::Square; cfg.mesh_gen = MeshGen::Quad;
    cfg.width = cfg.height = 10; cfg.size_x = cfg.size_y = 2.0;
    cfg.pin_mode = PinMode::CORNERS;
    cfg.colliders = { make_sphere(Vec3(1.0, 0.3, 1.0), 0.5) };
    cfg.animation_file = ""; cfg.waist_attach_enabled = false;
    cfg.speculative_contact = true; cfg.contact_margin = 0.01;
    cfg.secs = 1;
    const Real dt = 1.0 / (cfg.FPS * cfg.frame_substeps);
    const int  n_steps = cfg.FPS * cfg.frame_substeps * cfg.secs;
    auto simulate = [&](int m, int n_iters)
    {
        cfg.anderson_m = m;
        prepare_world(cfg);
        SimRunResult r;
        r.obj = build_run_cloth(cfg, cfg.guess_k());
        init_pd_velocity(r.obj, dt);
        pd_contact(r.obj, dt, cfg.gravity, n_iters, n_steps, cfg.frame_substeps, r.tape, "anderson_check", /*export_obj=*/false);
        return r;
    };
    const SimRunResult plain = simulate(0, 1000), accel = simulate(5, 150), short_plain = simulate(0, 150);
    const Real gap_aa = (plain.obj.x - accel.obj.x).cwiseAbs().maxCoeff(), gap_plain = (plain.obj.x - short_plain.obj.x).cwiseAbs().maxCoeff();
    std::printf("cloth on sphere, final pose vs 1000 plain iterations: 150 Anderson %.1e m, 150 plain %.1e m\n", gap_aa, gap_plain);
    ASSERT(gap_aa < 1e-7, "Anderson did not converge to the plain fixed point");
    return 0;
}

// `diffpd --anderson-check <file.conf>`: settles the config's cloth, then runs its guess forward with Anderson off and on
// (m = 5), both at the config's n_iters, and prints the largest position gap between the two trajectories (over all
// frames and at the last one) plus each run's time and residual tail. Converged runs must match to solver tolerance.
static int run_anderson_compare(const std::string& path)
{
    AppConfig cfg;
    ASSERT(load_config(path, cfg), "cannot load config " << path);
    settle_cloth(cfg, /*headless=*/true);
    const Real dt      = 1.0 / (cfg.FPS * cfg.frame_substeps);
    const int  n_steps = cfg.FPS * cfg.frame_substeps * cfg.secs;
    auto simulate = [&](int m)
    {
        cfg.anderson_m = m;
        const WorldSetup world = prepare_world(cfg);
        SimRunResult r;
        r.obj = build_run_cloth(cfg, cfg.guess_k());
        attach_waist(r.obj, world.waist_attach_anim_id, world.waist_attach_skinned_id, collider_animations);
        init_pd_velocity(r.obj, dt);
        const auto t0 = std::chrono::steady_clock::now();
        pd_contact(r.obj, dt, cfg.gravity, cfg.n_iters, n_steps, cfg.frame_substeps, r.tape, "anderson_check", /*export_obj=*/false);
        std::vector<Real> rel;
        for (const StepResidual& s : r.tape.forward_residual) rel.push_back(s.rel);
        std::sort(rel.begin(), rel.end(), std::greater<Real>());
        const size_t n_tail = std::max<size_t>(1, rel.size() / 10);
        std::printf("[anderson-check] m = %d: %d free particles, %d steps x %d iters, %.1f s, rel tail10 %.1e max %.1e\n", m,
                    (int)r.obj.num_particles(), n_steps, cfg.n_iters,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                    std::accumulate(rel.begin(), rel.begin() + n_tail, 0.0) / n_tail, rel.front());
        std::fflush(stdout);
        return r;
    };
    const SimRunResult plain = simulate(0), accel = simulate(5);
    Real gap_max = 0.0; int gap_frame = 0;
    for (size_t t = 0; t < plain.tape.positions.size(); ++t)
    {
        const Real gap = (plain.tape.positions[t] - accel.tape.positions[t]).cwiseAbs().maxCoeff();
        if (gap > gap_max) { gap_max = gap; gap_frame = (int)t; }
    }
    std::printf("[anderson-check] max |x_plain - x_AA|: %.2e m over the trajectory (step %d), %.2e m at the last step\n",
                gap_max, gap_frame, (plain.tape.positions.back() - accel.tape.positions.back()).cwiseAbs().maxCoeff());
    return 0;
}

// `diffpd --chunk-check <file.conf>`: runs the config's guess for its whole horizon, then re-runs the second half as a
// chunk (first_step = n_steps / 2) started from the full run's exact state at that step. The chunk must reproduce the
// full trajectory, which proves animated colliders, skinned bodies and the waist are posed from the global step.
static int run_chunk_check(const std::string& path)
{
    AppConfig cfg;
    ASSERT(load_config(path, cfg), "cannot load config " << path);
    settle_cloth(cfg, /*headless=*/true);
    const Real dt      = 1.0 / (cfg.FPS * cfg.frame_substeps);
    const int  n_steps = cfg.FPS * cfg.frame_substeps * cfg.secs;
    const int  first   = n_steps / 2;
    const WorldSetup world = prepare_world(cfg);
    auto simulate = [&](int steps, int first_step, const Tape* start)
    {
        SimRunResult r;
        r.obj = build_run_cloth(cfg, cfg.guess_k());
        attach_waist(r.obj, world.waist_attach_anim_id, world.waist_attach_skinned_id, collider_animations);
        init_pd_velocity(r.obj, dt);
        if (start)
        {
            r.obj.x = Eigen::Map<const RealVecX>(start->positions[first_step].data(),  r.obj.num_dofs());
            r.obj.v = Eigen::Map<const RealVecX>(start->velocities[first_step].data(), r.obj.num_dofs());
        }
        pd_contact(r.obj, dt, cfg.gravity, cfg.n_iters, steps, cfg.frame_substeps, r.tape, "chunk_check", /*export_obj=*/false,
                   false, nullptr, cfg.unresolved_contact_threshold, false, first_step);
        return r;
    };
    const SimRunResult full  = simulate(n_steps, 0, nullptr);
    const SimRunResult chunk = simulate(n_steps - first, first, &full.tape);
    Real gap = 0.0;
    for (size_t t = 0; t < chunk.tape.positions.size(); ++t)
        gap = std::max(gap, (chunk.tape.positions[t] - full.tape.positions[first + t]).cwiseAbs().maxCoeff());
    const Real loss = Loss(chunk.tape, full.tape, 1, first).total;
    std::printf("[chunk-check] %d-step chunk from step %d: max position gap %.2e m, loss vs full run %.2e\n", n_steps - first, first, gap, loss);
    ASSERT(gap < 1e-9, "chunk does not reproduce the full run");
    return 0;
}
