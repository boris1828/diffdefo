// cloth(): a rectangular grid of particles with stretch / shear / bend springs (declared in cloth/build.h).

#include "cloth/build.h"

Object cloth(
    Index       width,
    Index       height,
    Real        size_x,
    Real        size_y,
    Stiffness   stiffness,
    Vec3        origin,
    PinMode     pin_mode,
    HangingMode hanging_mode,
    uint8_t     flags,
    Real        m_tot)
{
    ASSERT(flags & ClothFlags::STRETCH, "cloth must have stretch constraints enabled");

    const Index N  = width * height;
    const Real  sx = size_x / Real(width  - 1);
    const Real  sz = size_y / Real(height - 1);

    auto grid = [height](Index i, Index j) { return i * height + j; };

    // HORIZONTAL: (i,j) -> (x, 0, z), flat. VERTICAL: (i,j) -> (x, -z, 0), j=0 row hangs from the top.
    std::vector<Vec3> pos(N);
    for (Index i = 0; i < width; ++i)
        for (Index j = 0; j < height; ++j)
            pos[grid(i, j)] = origin + (hanging_mode == HangingMode::VERTICAL
                ? Vec3(i * sx, -j * sz, 0.0)
                : Vec3(i * sx, 0.0, j * sz));

    // --- mark pinned vertices ---------------------------------------------
    std::vector<bool> pinned(N, false);
    switch (pin_mode) {
        case PinMode::NONE:
            break;
        case PinMode::CORNERS:
            pinned[grid(0, 0)]         = true;
            pinned[grid(width - 1, 0)] = true;
            break;
        case PinMode::RAIL: // the bar's fixed end; the rings are added by attach_rail
            pinned[grid(0, 0)] = true;
            break;
        case PinMode::ROW:
            for (Index i = 0; i < width; ++i)
                pinned[grid(i, 0)] = true;
            break;
    }

    // --- compact free vertices into DOF indices (V_free, DOF_map) ---------
    // dof[v] = particle id in the stacked state, or -1 if pinned.
    std::vector<ParticleId> dof(N, -1);
    Index n = 0; // |V_free|
    for (Index v = 0; v < N; ++v)
        if (!pinned[v]) dof[v] = ParticleId(n++);

    Object obj;

    // --- state vectors (stacked 3n, free particles only) ------------------
    obj.x.resize(3 * n);
    for (Index v = 0; v < N; ++v)
        if (dof[v] >= 0)
            obj.x.segment<3>(3 * dof[v]) = pos[v];

    obj.v      = Velocities::Zero(3 * n);
    obj.prev_x = obj.x;

    // --- mass: m_tot distributed uniformly over n free particles ----------
    obj.mass = MassDiag::Constant(3 * n, m_tot / Real(n));

    // ---- bake export mesh (after dof[] and pos are built) ------------------
    SimMesh& mesh = obj.mesh;
    mesh.width  = width;
    mesh.height = height;

    for (Index i = 0; i < width; ++i)
        for (Index j = 0; j < height; ++j)
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

    // structural edges only (both endpoints, pinned or not)
    auto mesh_edge = [&](Index a, Index b)
    {
        mesh.edges.emplace_back(a, b);
    };

    for (Index i = 0; i < width - 1; ++i)
        for (Index j = 0; j < height; ++j)
            mesh_edge(grid(i, j), grid(i + 1, j));
    for (Index i = 0; i < width; ++i)
        for (Index j = 0; j < height - 1; ++j)
            mesh_edge(grid(i, j), grid(i, j + 1));

    // --- constraint emission ----------------------------------------------
    // mesh.vertices[v].dof, not dof[v]: for a pinned vertex dof[v] is just the shared -1 sentinel.
    auto emit = [&](Index a, Index b, SpringKind kind)
    {
        obj.energies.springs.add_edge(mesh.vertices[a].dof, mesh.vertices[b].dof, pos[a], pos[b], kind, stiffness[kind]);
    };

    if (flags & ClothFlags::STRETCH)
    {
        // structural: along width axis (i)
        for (Index i = 0; i < width - 1; ++i)
            for (Index j = 0; j < height; ++j)
                emit(grid(i, j), grid(i + 1, j), SpringKind::Stretch);

        // structural: along height axis (j)
        for (Index i = 0; i < width; ++i)
            for (Index j = 0; j < height - 1; ++j)
                emit(grid(i, j), grid(i, j + 1), SpringKind::Stretch);
    }

    if (flags & ClothFlags::SHEAR)
    {
        for (Index i = 0; i < width - 1; ++i)
            for (Index j = 0; j < height - 1; ++j)
            {
                emit(grid(i, j),     grid(i + 1, j + 1), SpringKind::Shear);
                emit(grid(i + 1, j), grid(i, j + 1),     SpringKind::Shear);
            }
    }

    if (flags & ClothFlags::BENDING)
    {
        for (Index i = 0; i < width; ++i)
            for (Index j = 0; j < height - 2; ++j)
                emit(grid(i, j), grid(i, j + 2), SpringKind::Bend);

        for (Index i = 0; i < width - 2; ++i)
            for (Index j = 0; j < height; ++j)
                emit(grid(i, j), grid(i + 2, j), SpringKind::Bend);
    }

    return obj;
}
