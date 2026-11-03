// skirt(): a conical frustum of particles, closed around the ring (declared in cloth/build.h).

#include "cloth/build.h"

#include <cmath>

// A conical-frustum ("skirt") cloth: `num_rings` rings of `particles_per_ring` vertices, stacked
// along -Y from a pinned `radius_top` ring at `origin` to `radius_bottom` at `height` below
// (interpolated). Same (i,j) grid as cloth(), but i wraps (i=W-1 connects to i=0) while j stays
// open, so it reads as a skirt, not a closed cylinder.
Object skirt(
    Index   particles_per_ring,
    Index     num_rings,
    Stiffness stiffness,
    Vec3      origin,
    Real      radius_top,
    Real      radius_bottom,
    Real      height,
    uint8_t   flags,
    Real      m_tot)
{
    ASSERT(flags & ClothFlags::STRETCH, "skirt must have stretch constraints enabled");
    ASSERT(particles_per_ring >= 3, "skirt needs at least 3 particles per ring");
    ASSERT(num_rings >= 2, "skirt needs at least 2 rings (top + bottom)");

    const Index W = particles_per_ring; // circumferential axis, wraps
    const Index H = num_rings;          // vertical axis, open (no cap)

    auto grid = [H](Index i, Index j) { return i * H + j; };
    auto wrap = [W](Index i) { return (i + W) % W; };

    constexpr Real kTwoPi = 6.283185307179586476925286766559;

    // j=0 is the pinned top ring (radius_top); j=H-1 is radius_bottom, hanging -Y by `height`.
    std::vector<Vec3> pos(W * H);
    for (Index i = 0; i < W; ++i)
    {
        const Real angle = kTwoPi * Real(i) / Real(W);
        for (Index j = 0; j < H; ++j)
        {
            const Real t = Real(j) / Real(H - 1);
            const Real r = radius_top + t * (radius_bottom - radius_top);
            pos[grid(i, j)] = origin + Vec3(r * std::cos(angle), -t * height, r * std::sin(angle));
        }
    }

    // --- mark pinned vertices (top ring only) ------------------------------
    std::vector<bool> pinned(W * H, false);
    for (Index i = 0; i < W; ++i)
        pinned[grid(i, 0)] = true;

    // --- compact free vertices into DOF indices (V_free, DOF_map) ---------
    std::vector<ParticleId> dof(W * H, -1);
    Index n = 0; // |V_free|
    for (Index v = 0; v < W * H; ++v)
        if (!pinned[v]) dof[v] = ParticleId(n++);

    Object obj;

    // --- state vectors (stacked 3n, free particles only) ------------------
    obj.x.resize(3 * n);
    for (Index v = 0; v < W * H; ++v)
        if (dof[v] >= 0)
            obj.x.segment<3>(3 * dof[v]) = pos[v];

    obj.v      = Velocities::Zero(3 * n);
    obj.prev_x = obj.x;

    // --- mass: m_tot distributed uniformly over n free particles ----------
    obj.mass = MassDiag::Constant(3 * n, m_tot / Real(n));

    // ---- bake export mesh (after dof[] and pos are built) ------------------
    SimMesh& mesh = obj.mesh;
    mesh.width  = W;
    mesh.height = H;
    mesh.wrap_i = true; // the ring direction (i) is a closed loop, unlike cloth()'s open sheet

    for (Index i = 0; i < W; ++i)
        for (Index j = 0; j < H; ++j)
        {
            const Index v = grid(i, j);
            SimMesh::Vertex mv;
            if (dof[v] >= 0)
            {
                mv.dof = dof[v];
            }
            else
            {
                mv.dof = ParticleId(-Index(mesh.pinned_rest.size()) - 1);
                mesh.pinned_rest.push_back(pos[v]);
            }
            mesh.vertices.push_back(mv);
        }

    // structural edges only (both endpoints, pinned or not); circumferential edges wrap around the
    // ring, vertical edges don't (open top/bottom).
    auto mesh_edge = [&](Index a, Index b)
    {
        mesh.edges.emplace_back(a, b);
    };

    for (Index i = 0; i < W; ++i)
        for (Index j = 0; j < H; ++j)
            mesh_edge(grid(i, j), grid(wrap(i + 1), j));
    for (Index i = 0; i < W; ++i)
        for (Index j = 0; j < H - 1; ++j)
            mesh_edge(grid(i, j), grid(i, j + 1));

    // --- constraint emission ----------------------------------------------
    // mesh.vertices[v].dof, not dof[v]: for a pinned vertex dof[v] is just the shared -1 sentinel.
    auto emit = [&](Index a, Index b, SpringKind kind)
    {
        obj.energies.springs.add_edge(mesh.vertices[a].dof, mesh.vertices[b].dof, pos[a], pos[b], kind, stiffness[kind]);
    };

    if (flags & ClothFlags::STRETCH)
    {
        // structural: circumferential (i), wraps around the ring
        for (Index i = 0; i < W; ++i)
            for (Index j = 0; j < H; ++j)
                emit(grid(i, j), grid(wrap(i + 1), j), SpringKind::Stretch);

        // structural: vertical (j), open (no top/bottom cap)
        for (Index i = 0; i < W; ++i)
            for (Index j = 0; j < H - 1; ++j)
                emit(grid(i, j), grid(i, j + 1), SpringKind::Stretch);
    }

    if (flags & ClothFlags::SHEAR)
    {
        for (Index i = 0; i < W; ++i)
            for (Index j = 0; j < H - 1; ++j)
            {
                emit(grid(i, j),           grid(wrap(i + 1), j + 1), SpringKind::Shear);
                emit(grid(wrap(i + 1), j), grid(i, j + 1),           SpringKind::Shear);
            }
    }

    if (flags & ClothFlags::BENDING)
    {
        // vertical bending: open, same as cloth
        for (Index i = 0; i < W; ++i)
            for (Index j = 0; j < H - 2; ++j)
                emit(grid(i, j), grid(i, j + 2), SpringKind::Bend);

        // circumferential bending: wraps. Skipped below W=5, where stepping by 2 duplicates a
        // stretch edge (W=3) or double-emits the same pair (W=4); W>=5 gives distinct pairs.
        if (W >= 5)
            for (Index i = 0; i < W; ++i)
                for (Index j = 0; j < H; ++j)
                    emit(grid(i, j), grid(wrap(i + 2), j), SpringKind::Bend);
    }

    return obj;
}
