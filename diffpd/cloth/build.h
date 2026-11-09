#pragma once

// Which cloth to build (pin / hanging / mesh-generation enums, pattern file helpers) and the three factories that build
// it. Bodies of the factories are in quad.cpp, skirt.cpp and tri_cloth.cpp. Depends on cloth/object.h and meshgen/.

#include "cloth/object.h"
#include "meshgen/pattern_io.h"
#include "meshgen/tri_surface.h"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// ----------------
//  CLOTH CONFIG
// ----------------

// How the cloth is anchored: NONE = free fall, CORNERS = two top corners, ROW = entire first row, RAIL = top-left corner pinned (the bar's
// fixed end) and rings sliding on a bar along the top edge (ClothSpec::rail_stride / rail_k, RailEnergy); square cloth only. Saved as an int: append only.
enum class PinMode { NONE, CORNERS, ROW, RAIL };

// HORIZONTAL: flat in XZ, falls and swings under gravity. VERTICAL: in XY, j=0 row at top,
// hanging straight down from the start.
enum class HangingMode { HORIZONTAL, VERTICAL };

namespace ClothFlags
{
    constexpr uint8_t STRETCH = 1 << 0;
    constexpr uint8_t SHEAR   = 1 << 1;
    constexpr uint8_t BENDING = 1 << 2;
    constexpr uint8_t ALL     = STRETCH | SHEAR | BENDING;
}

// Which factory builds the sim object: an open sheet (cloth()), a closed conical skirt (skirt()) or the pieces of a pattern file
// (Pattern: always a Tri mesh, see tri_surface_for()). AppConfig carries all shapes' params simultaneously; see build_cloth().
enum class ClothType { Square, Skirt, Pattern };

// Pattern contour files (clothcreate format, see pattern_io.h) live in PATTERNS_DIR_DEFAULT (diffpd/patterns).
#ifndef PATTERNS_DIR_DEFAULT
#define PATTERNS_DIR_DEFAULT "patterns"
#endif
inline std::string pattern_path(const std::string& file) { return (std::filesystem::path(PATTERNS_DIR_DEFAULT) / file).string(); }
inline std::vector<std::string> list_pattern_files() // *.json names, sorted
{
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(PATTERNS_DIR_DEFAULT, ec))
        if (e.is_regular_file() && e.path().extension() == ".json") out.push_back(e.path().filename().string());
    std::sort(out.begin(), out.end());
    return out;
}
// Size + modification time of a pattern file: editing the file invalidates the generated mesh and the settled pose.
inline std::string pattern_stamp(const std::string& file)
{
    std::error_code ec;
    const auto size = std::filesystem::file_size(pattern_path(file), ec);
    const auto time = std::filesystem::last_write_time(pattern_path(file), ec).time_since_epoch().count();
    return ec ? std::string("missing") : std::to_string(size) + "@" + std::to_string(time);
}

// How the mesh is made: the (i,j) quad grid (stretch + shear + bend springs) or a generated triangle mesh
// (stretch springs on the triangle edges only, or ARAP). Orthogonal to ClothType: both shapes have both.
enum class MeshGen { Quad, Tri };

// Elastic energy of a Tri mesh: one Stretch spring per edge, or one ARAP element per triangle (forward only).
enum class TriEnergy { Springs, ARAP };


// ----------------
//   CLOTH SPEC
// ----------------

// Everything that decides which cloth is built (not its stiffness values, which the caller passes). Square uses
// width/height/pin_mode/hang_mode; Skirt uses particles_per_ring/num_rings/radius_top/radius_bottom/skirt_height;
// Pattern uses pattern_file/pattern_size/seam_stiffness.
struct ClothSpec
{
    ClothType cloth_type = ClothType::Skirt;

    Vec3 origin = Vec3(0.0, 0.95, 0.0); // shared by both target and guess cloth
    Real m_tot  = 0.04;

    // Constraint types of the quad grid (STRETCH is always on, not stored). Ignored on a Tri mesh, see shear_on().
    bool flag_shear   = true;  // ClothFlags::SHEAR
    bool flag_bending = true;  // ClothFlags::BENDING

    // Square
    int         width     = 15;
    int         height    = 15;
    Real        size_x    = 1.0; // metres along width / height
    Real        size_y    = 1.0;
    PinMode     pin_mode  = PinMode::ROW;
    HangingMode hang_mode = HangingMode::HORIZONTAL;
    int         rail_stride = 4;    // PinMode::RAIL: every rail_stride-th particle of the top edge (and its last one) is a ring
    Real        rail_k      = 1e4;  // PinMode::RAIL: ring stiffness across the bar, high: a load f moves a ring by f/k
    Real        rail_spacer_k = 2.0; // PinMode::RAIL: spring between consecutive rings (and the pinned end), rest = their initial distance, a soft minimum spacing; 0 = off

    // Skirt
    int  particles_per_ring = 40;
    int  num_rings          = 20;
    Real radius_top         = 0.20;
    Real radius_bottom      = 0.3;
    Real skirt_height       = 0.5;

    // Pattern: every shape of the file, laid out as in the file (side by side, flat, vertical XY plane), one object of
    // disconnected pieces; no pinned vertices. The file is in metres in the body's world (version 7), so
    // pattern_size is a plain multiplier about the world origin (1 = as drawn) and `origin` is not used for it (build_cloth).
    std::string pattern_file   = "tshirt.json";
    Real        pattern_size   = 1.0;
    Real        seam_stiffness = 200.0; // k of the rest-length-0 springs sewing the pattern's seams (not optimised, see energy_seam.h)

    // Mesh generation (MeshGen::Tri uses tri_*; see meshgen::Params). Values are clamped when used, see tri_surface_for().
    MeshGen   mesh_gen         = MeshGen::Tri;
    Real      tri_h            = 0.03;
    Real      tri_beta         = 0.80;
    int       tri_smooth_iters = 30;
    int       tri_seed         = 0;
    TriEnergy tri_energy       = TriEnergy::Springs;
    bool      arap_flat_rest   = true;  // hinge rest l0 = 0 everywhere (flat sheet) instead of the generated mesh's curvature

    bool use_tri() const { return mesh_gen == MeshGen::Tri || cloth_type == ClothType::Pattern; }
    bool use_arap() const { return use_tri() && tri_energy == TriEnergy::ARAP; }
    Real tri_edge() const { return std::max(tri_h, 0.02); } // edge length h of the generated mesh, floored
    // The cone chart rejects cylinders (|dr| < 1e-3), degenerate sizes and an h that leaves fewer than 6 samples on a ring;
    // the rectangle chart needs at least 3 segments per edge.
    bool tri_ok() const; // defined below, after tri_surface_for
    // Shear/bend springs exist on the quad grid only.
    bool shear_on()   const { return flag_shear   && !use_tri(); }
    bool bending_on() const { return flag_bending && !use_tri(); }
};

// Declared here (bodies in cloth/*.cpp) so diffpd_viewer.cpp can rebuild a live preview without
// including diffpd.cpp, keeping it the only raylib-aware file.
Object cloth(
    Index       width,
    Index       height,
    Real        size_x, // metres spanned by `width` particles (x) and `height` particles (z or -y)
    Real        size_y,
    Stiffness   stiffness    = Stiffness::uniform(DEFAULT_STIFFNESS),
    Vec3        origin       = Vec3::Zero(),
    PinMode     pin_mode     = PinMode::CORNERS,
    HangingMode hanging_mode = HangingMode::HORIZONTAL,
    uint8_t     flags        = ClothFlags::ALL,
    Real        m_tot        = 1.0);

// A conical-frustum ("skirt") cloth: `num_rings` rings of `particles_per_ring` vertices, stacked
// along -Y from a pinned `radius_top` ring to `radius_bottom` at `height` below (interpolated).
// The ring direction wraps into a closed loop; height stays open (no caps), unlike cloth().
Object skirt(
    Index     particles_per_ring,
    Index     num_rings,
    Stiffness stiffness     = Stiffness::uniform(DEFAULT_STIFFNESS),
    Vec3      origin        = Vec3::Zero(),
    Real      radius_top    = 0.15,
    Real      radius_bottom = 0.3,
    Real      height        = 0.7,
    uint8_t   flags         = ClothFlags::ALL,
    Real      m_tot         = 0.5);

// A cloth built on a generated triangle mesh (see tri_surface.h): one Stretch spring per triangle edge, area-weighted
// masses, the vertices on `pinned_curves` (chart curve ids, may be empty) pinned. `s` is relative to `origin`.
Object tri_cloth(
    const meshgen::TriSurface& s,
    const std::vector<int>& pinned_curves,
    Stiffness stiffness,
    Vec3      origin,
    Real      m_tot,
    TriEnergy energy = TriEnergy::Springs,
    bool      flat_bend_rest = true, // ARAP only: hinge rest curvature l0 = 0 (flat sheet) or from the generated mesh
    Real      seam_k = 0);           // stiffness of the s.seams links (rest length 0)

// The generated mesh for cfg's skirt, cached per thread on its parameters (generation is far costlier than a grid, and the
// target, guess, FD check, settle and the per-frame config preview all rebuild the cloth). Out-of-range values are clamped.
struct TriCache { std::string key; meshgen::TriSurface surface; meshgen::Report report; std::string error; }; // error: Pattern only, empty = ok

inline const TriCache& tri_surface_for(const ClothSpec& cfg)
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

inline bool ClothSpec::tri_ok() const
{
    if (!use_tri()) return true;
    if (cloth_type == ClothType::Pattern) return tri_surface_for(*this).error.empty();
    if (cloth_type == ClothType::Square) return std::min(size_x, size_y) >= 2.5 * tri_edge();
    const Real r_min = std::min(radius_top, radius_bottom);
    return std::abs(radius_top - radius_bottom) >= 1e-3 && r_min >= 0.02 && skirt_height >= 0.02
           && tri_h <= 2.0 * 3.14159265358979 * r_min / 6.0;
}

// Chart curves of tri_surface_for(cfg) that are pinned: the skirt's top ring, the square's row / two corners / nothing, nothing for a pattern.
inline std::vector<int> tri_pinned_curves(const ClothSpec& cfg)
{
    using R = meshgen::Rectangle;
    if (cfg.cloth_type == ClothType::Pattern) return {};
    if (cfg.cloth_type == ClothType::Skirt) return { 0 };
    if (cfg.pin_mode == PinMode::ROW)       return { R::Top, R::CornerTL, R::CornerTR };
    if (cfg.pin_mode == PinMode::CORNERS)   return { R::CornerTL, R::CornerTR };
    if (cfg.pin_mode == PinMode::RAIL)      return { R::CornerTL };
    return {};
}

// Vertex ids of the square's top edge, in order along it (quad: grid(i, 0); tri: the Rectangle::Top boundary).
inline std::vector<Index> square_top_edge(const ClothSpec& cfg)
{
    std::vector<Index> edge;
    if (!cfg.use_tri())
    {
        for (Index i = 0; i < cfg.width; ++i) edge.push_back(i * cfg.height);
        return edge;
    }
    const meshgen::TriSurface& s = tri_surface_for(cfg).surface;
    for (size_t b = 0; b < s.boundary.size(); ++b)
        if (s.boundary_id[b] == meshgen::Rectangle::Top) return s.boundary[b];
    return edge;
}

// Rings on a bar along `edge` (vertex ids in order): every `stride`-th vertex and the last, bar = first to last position. A pinned vertex is skipped
// (add_rail), so the pinned end of the edge stays fixed. shortcut: orientation-agnostic, a Tri list that runs from the free end counts strides from there.
inline void attach_rail(Object& obj, const std::vector<Index>& edge, int stride, Real k, Real spacer_k)
{
    ASSERT(stride >= 1 && edge.size() >= 2, "attach_rail: stride " << stride << ", edge of " << edge.size() << " vertices");
    obj.energies.rails.k = k;
    const Vec3 bar = obj.mesh.position(obj, edge.back()) - obj.mesh.position(obj, edge.front());
    obj.energies.spacers.parametric = false;
    Index prev = -1; // previous selected vertex (the pinned end counts: its spacer is a free-pinned spring)
    for (size_t i = 0; i < edge.size(); ++i)
    {
        if (i % stride != 0 && i + 1 != edge.size()) continue;
        const Index v = edge[i];
        obj.energies.rails.add_rail(obj.mesh.vertices[v].dof, obj.mesh.position(obj, v), bar);
        if (spacer_k > 0 && prev >= 0)
            obj.energies.spacers.add_edge(obj.mesh.vertices[prev].dof, obj.mesh.vertices[v].dof, obj.mesh.position(obj, prev), obj.mesh.position(obj, v), SpringKind::Stretch, spacer_k);
        prev = v;
    }
}

// Dispatches to cloth(), skirt() or tri_cloth() per cfg.cloth_type. `stiffness`/`origin` are passed explicitly
// (not read from cfg) since callers need arbitrary target/guess/FD-perturbed values.
inline Object build_cloth(const ClothSpec& cfg, const Stiffness& stiffness, Vec3 origin)
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
    if (cfg.cloth_type == ClothType::Square && cfg.pin_mode == PinMode::RAIL) attach_rail(obj, square_top_edge(cfg), cfg.rail_stride, cfg.rail_k, cfg.rail_spacer_k);
    return obj;
}
