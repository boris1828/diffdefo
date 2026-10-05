#pragma once

#include "diffpd_types.h"

#include <filesystem>
#include <fstream>
#include <map>

// Loads a Blender-exported (Z-up) .obj into `mesh` in diffpd's Y-up rest frame, with all the
// per-face/edge/vertex data precomputed. Returns false (after a WARNING) if the file is missing
// or isn't a closed, consistently wound manifold.
inline bool load_trimesh_obj(const std::string& path, TriMesh& mesh)
{
    std::ifstream in(path);
    if (!in.is_open()) { WARNING("load_trimesh_obj: could not open " << path); return false; }

    TriMesh m;
    m.name = std::filesystem::path(path).stem().string();

    // Parse `v` and `f` only. Vertices at the same position are welded: OBJ exporters split them
    // per UV/normal, which would break edge adjacency.
    std::map<std::array<long long, 3>, int> welded;
    std::vector<int>                        obj_to_welded; // OBJ vertex index -> index into m.vertices
    std::string                             line;
    while (std::getline(in, line))
    {
        std::istringstream ss(line);
        std::string        tag;
        ss >> tag;
        if (tag == "v")
        {
            Vec3 p;
            ss >> p.x() >> p.y() >> p.z();
            p = blender_to_diffpd(p);
            const std::array<long long, 3> key = { std::llround(p.x() * 1e6), std::llround(p.y() * 1e6), std::llround(p.z() * 1e6) };
            const auto [it, inserted] = welded.try_emplace(key, (int)m.vertices.size());
            if (inserted) m.vertices.push_back(p);
            obj_to_welded.push_back(it->second);
        }
        else if (tag == "f")
        {
            std::vector<int> face;
            std::string      token; // "v", "v/vt" or "v//vn": stoi reads just the leading vertex index
            while (ss >> token)
            {
                const int i = std::stoi(token) - 1;
                if (i < 0 || i >= (int)obj_to_welded.size()) { WARNING("load_trimesh_obj: " << path << ": bad vertex index in '" << line << "'"); return false; }
                face.push_back(obj_to_welded[i]);
            }
            for (size_t k = 1; k + 1 < face.size(); ++k) // triangle fan handles quads
                m.triangles.push_back({ face[0], face[k], face[k + 1] });
        }
    }

    // Drop degenerate triangles (repeated vertex after welding, or ~zero area).
    const auto cross_of = [&](const std::array<int, 3>& t)
    { return (m.vertices[t[1]] - m.vertices[t[0]]).cross(m.vertices[t[2]] - m.vertices[t[0]]); };
    m.triangles.erase(std::remove_if(m.triangles.begin(), m.triangles.end(),
                          [&](const std::array<int, 3>& t) { return cross_of(t).norm() < 1e-12; }),
                      m.triangles.end());
    if (m.triangles.empty()) { WARNING("load_trimesh_obj: " << path << ": no usable triangles"); return false; }

    // Make the winding outward: a negative signed volume means the mesh is inside-out.
    Real volume = 0.0;
    for (const auto& t : m.triangles) volume += m.vertices[t[0]].dot(cross_of(t)) / 6.0;
    if (volume < 0.0)
        for (auto& t : m.triangles) std::swap(t[1], t[2]);

    // Closed + consistently wound <=> every directed edge a->b is used once and has a partner b->a.
    std::map<std::pair<int, int>, int> edge_owner; // directed edge -> triangle that has it
    for (int t = 0; t < (int)m.triangles.size(); ++t)
        for (int e = 0; e < 3; ++e)
            if (!edge_owner.try_emplace({ m.triangles[t][e], m.triangles[t][(e + 1) % 3] }, t).second)
            { WARNING("load_trimesh_obj: " << path << ": edge used twice in one direction (non-manifold or inconsistent winding)"); return false; }
    for (const auto& [edge, owner] : edge_owner)
        if (!edge_owner.count({ edge.second, edge.first }))
        { WARNING("load_trimesh_obj: " << path << ": mesh is not closed (open boundary edge)"); return false; }

    // Face normals/areas; vertex pseudo-normals = corner-angle-weighted sum of adjacent face normals.
    const size_t n_tris = m.triangles.size();
    m.face_normal.resize(n_tris);
    m.face_area.resize(n_tris);
    m.vertex_pseudo_normal.assign(m.vertices.size(), Vec3::Zero());
    for (size_t t = 0; t < n_tris; ++t)
    {
        const Vec3 c       = cross_of(m.triangles[t]);
        m.face_area[t]     = 0.5 * c.norm();
        m.face_normal[t]   = c.normalized();
        for (int k = 0; k < 3; ++k)
        {
            const Vec3 p     = m.vertices[m.triangles[t][k]];
            const Vec3 u     = (m.vertices[m.triangles[t][(k + 1) % 3]] - p).normalized();
            const Vec3 w     = (m.vertices[m.triangles[t][(k + 2) % 3]] - p).normalized();
            const Real angle = std::acos(std::clamp(u.dot(w), -1.0, 1.0));
            m.vertex_pseudo_normal[m.triangles[t][k]] += angle * m.face_normal[t];
        }
    }
    for (Vec3& n : m.vertex_pseudo_normal) n.normalize();

    // Edge pseudo-normal = normalized sum of the two faces sharing the edge (found via the reverse edge).
    m.edge_pseudo_normal.resize(n_tris);
    for (int t = 0; t < (int)n_tris; ++t)
        for (int e = 0; e < 3; ++e)
        {
            const int neighbor = edge_owner.at({ m.triangles[t][(e + 1) % 3], m.triangles[t][e] });
            m.edge_pseudo_normal[t][e] = (m.face_normal[t] + m.face_normal[neighbor]).normalized();
        }

    m.aabb_min = m.aabb_max = m.vertices[0];
    for (const Vec3& v : m.vertices) { m.aabb_min = m.aabb_min.cwiseMin(v); m.aabb_max = m.aabb_max.cwiseMax(v); }

    mesh = std::move(m);
    return true;
}
