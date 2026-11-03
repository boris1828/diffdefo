#pragma once

// The cloth as the simulator sees it: the mesh (SimMesh), the pin bindings and the Object that holds state, mass,
// energies and the factorised system. Depends on core/ and energy/ only.

#include "core/diffpd_core.h"
#include "energy/elastic_energies.h"

#include <array>
#include <memory>
#include <vector>

// ----------------
//      MESH
// ----------------

struct Object; // SimMesh::position() needs Object; defined below, method body after Object.

// Named SimMesh, not Mesh, to avoid colliding with raylib's global `struct Mesh`.
struct SimMesh
{
    struct Vertex
    {
        ParticleId dof;
    };

    std::vector<Vertex> vertices;              // output order == index order
    std::vector<Vec3>   pinned_rest;           // fixed positions for pinned verts
    std::vector<std::pair<Index,Index>> edges; // indices into `vertices`
    std::vector<std::pair<Index,Index>> seams; // vertex pairs sewn together (Pattern; drawn by the viewer)
    std::vector<int>                    seam_levels; // per seam: the level it is closed in (1..4)
    std::vector<std::pair<Index,Vec3>>  pulls;       // pulled vertex -> world unit direction (Pattern; used by "Settle seams" only)

    // Grid shape: vertices[] laid out row-major as i*height+j. Set by cloth()/skirt() so
    // renderers can reconstruct quad adjacency without re-deriving it from edges[].
    Index width  = 0;
    Index height = 0;

    // True if the i axis wraps into a closed ring (set by skirt(), false for cloth()) — tells
    // grid renderers to close the seam between the last and first column.
    bool wrap_i = false;

    // Triangle list into `vertices` (empty for grid meshes; renderers use it when non-empty). Set by tri_cloth(),
    // where width = height = 0.
    std::vector<std::array<Index, 3>> triangles;

    Vec3 position(const Object& obj, Index vi) const; // defined after Object
};

// One pinned vertex riding a skinned body: up to 4 (bone, weight) pairs, like a skinned vertex.
// q[k] = S_bone[k](0)^-1 * p0 is the vertex's frame-0 world position p0 expressed through bone k's own
// frame-0 skinning matrix S, so the vertex sits at sum_k weight[k] * S_bone[k](t) * q[k] at step t
// (== S(t) S(0)^-1 p0 for a single bone with weight 1). Unused slots have weight 0.
struct PinBinding
{
    static constexpr int kMaxInfluences = 4;

    std::array<int, kMaxInfluences>  bone   = { 0, 0, 0, 0 };
    std::array<Real, kMaxInfluences> weight = { 1.0, 0.0, 0.0, 0.0 };
    std::array<Vec3, kMaxInfluences> q      = { Vec3::Zero(), Vec3::Zero(), Vec3::Zero(), Vec3::Zero() };
};

// How the cloth's pinned vertices follow a moving body ("waist attachment"): either a rigid animated
// collider track (anim_id >= 0; each pin keeps a fixed offset in the track's frame-0 pose) or a skinned
// body (skinned_id >= 0; one PinBinding per pinned vertex). Neither set = pins stay at their rest position.
// See bake_waist_attachment* / update_waist_attachment*.
struct WaistAttachment
{
    int                     anim_id = -1;   // rigid: index into a std::vector<ColliderAnimation>
    std::vector<Vec3>       local_offset;   // rigid: per pinned vertex, offset in the track's frame-0 pose
    int                     skinned_id = -1; // skinned: index into skinned_meshes
    std::vector<PinBinding> binding;         // skinned: per pinned vertex

    bool active() const { return anim_id >= 0 || skinned_id >= 0; }
};

struct Object
{
    Positions  x;        // current positions       (3n)
    Velocities v;        // current velocities      (3n)
    Positions  prev_x;   // x at start of step

    MassDiag        mass;     // diagonal of M,          (3n)
    ElasticEnergies energies;

    SparseMat L;                       // L = M/h^2 + sum_i k_i G_i^T G_i   (SPD, constant)
    SparseMat C;                       // h^2 * K — elastic block; only set by construct_velocity_lhs
    // Laplacian damping (Rayleigh with the PD Laplacian K), D = alpha*M + beta*K, set by build_cloth().
    // alpha = beta = 0 is "off": C_d == C and the factor below is of L itself. L and C stay UNDAMPED
    // (they build b_tilde and its derivative); C_d = C + h*D is only for the free force f and the
    // adjoint's P-coupling, and `solver` factors L_d = M + C_d.
    Real      damp_alpha = 0.0;        // 1/s, mass-proportional
    Real      damp_beta  = 0.0;        // s,   Laplacian-proportional
    SparseMat C_d;                     // C + h*D; only set by construct_velocity_lhs
    std::unique_ptr<Cholesky> solver;  // factor of L_d (== L when undamped); heap-allocated so Object stays moveable

    SimMesh mesh;

    // Pinned vertices follow a moving body instead of a fixed rest position, if active().
    WaistAttachment waist;

    Object() = default;

    Object(const Object&)            = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&&)                 = default;
    Object& operator=(Object&&)      = default;

    Index num_particles() const { return x.rows() / 3; }
    Index num_dofs()      const { return x.rows();     }
};

inline Vec3 SimMesh::position(const Object& obj, Index vi) const
{
    const Vertex& vert = vertices[vi];
    if (vert.dof >= 0)
    {
        ASSERT(3 * vert.dof + 2 < obj.x.rows(), "mesh dof out of range for obj.x");
        return obj.x.segment<3>(3 * vert.dof);
    }
    return pinned_rest[ -(vert.dof + 1) ];
}
