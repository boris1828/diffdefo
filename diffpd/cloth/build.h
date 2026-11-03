#pragma once

// Which cloth to build (pin / hanging / mesh-generation enums, pattern file helpers) and the three factories that build
// it. Bodies of the factories are in diffpd.cpp (they move into cloth/ in step C). Depends on cloth/object.h and meshgen/.

#include "cloth/object.h"
#include "meshgen/tri_surface.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

// ----------------
//  CLOTH CONFIG
// ----------------

// How the cloth is anchored: NONE = free fall, CORNERS = two top corners, ROW = entire first row.
enum class PinMode { NONE, CORNERS, ROW };

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


// Declared here (body in diffpd.cpp) so diffpd_viewer.cpp can rebuild a live preview without
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
