// tri_cloth(): a cloth on a generated triangle mesh (declared in cloth/build.h).

#include "cloth/build.h"

#include <algorithm>
#include <map>

// Same layout as skirt() but the vertices/edges/triangles come from a generated meshgen::TriSurface: pinned = the vertices
// on `pinned_curves`, one Stretch spring per triangle edge (rest length from `rest_pos`, the flat shape; `pos` is only the
// starting pose, bent for a deformed pattern piece), mass proportional to the vertex's lumped area.
Object tri_cloth(
    const meshgen::TriSurface& s,
    const std::vector<int>& pinned_curves,
    Stiffness stiffness,
    Vec3      origin,
    Real      m_tot,
    TriEnergy energy,
    bool      flat_bend_rest,
    Real      seam_k)
{
    const Index nv = (Index)s.pos.size();
    std::vector<Vec3> pos(nv), rest(nv); // starting pose, rest shape
    std::vector<ParticleId> dof(nv, -1);
    Index n = 0; // |V_free|
    Real free_area = 0.0;
    for (Index v = 0; v < nv; ++v)
    {
        pos[v] = origin + s.pos[v];
        rest[v] = origin + s.rest_pos[v];
        if (std::find(pinned_curves.begin(), pinned_curves.end(), s.curve[v]) == pinned_curves.end())
            { dof[v] = ParticleId(n++); free_area += s.vertex_area[v]; }
    }
    ASSERT(n > 0, "tri_cloth: the pinned curves pin all " << nv << " vertices");

    Object obj;
    obj.x.resize(3 * n);
    obj.mass.resize(3 * n);
    for (Index v = 0; v < nv; ++v)
        if (dof[v] >= 0)
        {
            obj.x.segment<3>(3 * dof[v]) = pos[v];
            obj.mass.segment<3>(3 * dof[v]).setConstant(m_tot * s.vertex_area[v] / free_area);
        }
    obj.v      = Velocities::Zero(3 * n);
    obj.prev_x = obj.x;

    SimMesh& mesh = obj.mesh;
    for (Index v = 0; v < nv; ++v)
    {
        SimMesh::Vertex mv;
        if (dof[v] >= 0) mv.dof = dof[v];
        else
        {
            mv.dof = ParticleId(-Index(mesh.pinned_rest.size()) - 1);
            mesh.pinned_rest.push_back(pos[v]);
        }
        mesh.vertices.push_back(mv);
    }
    mesh.edges     = s.edges;
    mesh.triangles = s.tris;
    mesh.seams     = s.seams;
    mesh.seam_levels = s.seam_level;
    mesh.pulls       = s.pulls; // not added to obj.energies.pulls: only settle_cloth(seams_only) applies them
    obj.energies.seams.k = seam_k;
    for (size_t i = 0; i < s.seams.size(); ++i)
        obj.energies.seams.add_link(mesh.vertices[s.seams[i].first].dof, mesh.vertices[s.seams[i].second].dof, i < s.seam_level.size() ? s.seam_level[i] : 1);

    if (energy == TriEnergy::ARAP)
    {
        for (const auto& t : s.tris)
        {
            const ParticleId d[3] = { mesh.vertices[t[0]].dof, mesh.vertices[t[1]].dof, mesh.vertices[t[2]].dof };
            const Vec3       p[3] = { rest[t[0]], rest[t[1]], rest[t[2]] };
            obj.energies.arap.add_triangle(d, p, SpringKind::Stretch, stiffness.stretch);
        }

        // Hinge per interior edge: the two triangles sharing it, tips = their third vertices.
        std::map<std::pair<Index, Index>, std::vector<Index>> tips;
        for (const auto& t : s.tris)
            for (int a = 0; a < 3; ++a)
            {
                const Index u = t[a], v = t[(a + 1) % 3];
                tips[{ std::min(u, v), std::max(u, v) }].push_back(t[(a + 2) % 3]);
            }
        for (const auto& [edge, tip] : tips)
        {
            if (tip.size() != 2) continue; // boundary edge
            const Index vi[4] = { edge.first, edge.second, tip[0], tip[1] };
            const ParticleId d[4] = { mesh.vertices[vi[0]].dof, mesh.vertices[vi[1]].dof, mesh.vertices[vi[2]].dof, mesh.vertices[vi[3]].dof };
            const Vec3       p[4] = { rest[vi[0]], rest[vi[1]], rest[vi[2]], rest[vi[3]] };
            obj.energies.bending.add_hinge(d, p, SpringKind::Bend, stiffness.bend, flat_bend_rest);
        }
        return obj;
    }

    for (const auto& e : s.edges)
        obj.energies.springs.add_edge(mesh.vertices[e.first].dof, mesh.vertices[e.second].dof,
                                      rest[e.first], rest[e.second], SpringKind::Stretch, stiffness.stretch);
    return obj;
}
