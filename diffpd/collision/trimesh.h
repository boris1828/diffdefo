#pragma once

#include "collision/bvh.h"
#include "collision/collider.h" // blender_to_diffpd

#include <filesystem>
#include <fstream>
#include <limits>
#include <map>

// Per-triangle / per-edge / per-vertex normals of a closed mesh. Recomputed from the current vertices
// (compute_tri_normals) whenever the mesh deforms; the pseudo-normals give a robust inside/outside sign.
struct TriNormals
{
    std::vector<Vec3>                face_normal;          // unit, per triangle
    std::vector<Vec3>                vertex_pseudo_normal; // corner-angle-weighted sum of adjacent face normals
    std::vector<std::array<Vec3, 3>> edge_pseudo_normal;   // [t][e]: edge from triangle vertex e to (e+1)%3
};

// Rigid triangle mesh in its rest (local) frame, built once at load and never modified.
// Triangles are welded and outward-wound; pseudo-normals give a robust inside/outside sign.
struct TriMesh : TriNormals
{
    std::string                    name;
    std::vector<Vec3>              vertices;
    std::vector<std::array<int, 3>> triangles;
    std::vector<Real>              face_area;
    std::vector<std::array<int, 3>> edge_neighbor;        // [t][e]: triangle across edge e (vertex e -> (e+1)%3)
    Vec3                           aabb_min = Vec3::Zero();
    Vec3                           aabb_max = Vec3::Zero();
    Bvh                            bvh;                   // over `vertices`; built by build_trimesh
};

// Face normals, corner-angle-weighted vertex pseudo-normals and edge pseudo-normals of the closed mesh
// (V, T) at its current vertex positions. `edge_neighbor[t][e]` is the triangle sharing edge e of triangle
// t (TriMesh::edge_neighbor, topology only). Cheap enough to rerun every step for a deforming mesh.
inline void compute_tri_normals(const std::vector<Vec3>& V, const std::vector<std::array<int, 3>>& T,
                                const std::vector<std::array<int, 3>>& edge_neighbor, TriNormals& out)
{
    const size_t n_tris = T.size();
    out.face_normal.resize(n_tris);
    out.vertex_pseudo_normal.assign(V.size(), Vec3::Zero());
    for (size_t t = 0; t < n_tris; ++t)
    {
        const Vec3 c       = (V[T[t][1]] - V[T[t][0]]).cross(V[T[t][2]] - V[T[t][0]]);
        out.face_normal[t] = c.normalized();
        for (int k = 0; k < 3; ++k)
        {
            const Vec3 p     = V[T[t][k]];
            const Vec3 u     = (V[T[t][(k + 1) % 3]] - p).normalized();
            const Vec3 w     = (V[T[t][(k + 2) % 3]] - p).normalized();
            const Real angle = std::acos(std::clamp(u.dot(w), -1.0, 1.0));
            out.vertex_pseudo_normal[T[t][k]] += angle * out.face_normal[t];
        }
    }
    for (Vec3& n : out.vertex_pseudo_normal) n.normalize();

    // Edge pseudo-normal = normalized sum of the two faces sharing the edge.
    out.edge_pseudo_normal.resize(n_tris);
    for (size_t t = 0; t < n_tris; ++t)
        for (int e = 0; e < 3; ++e)
            out.edge_pseudo_normal[t][e] = (out.face_normal[t] + out.face_normal[edge_neighbor[t][e]]).normalized();
}

// Validates and finishes a triangle mesh given in diffpd's Y-up frame: drops degenerate triangles,
// flips an inside-out winding, requires the mesh to be closed + consistently wound, then precomputes
// the per-face/edge/vertex data. Vertices are never dropped, welded or reordered (only triangles may
// be dropped), so per-vertex data kept by the caller (e.g. skinning weights) stays aligned. `label`
// names the source in warnings and becomes the mesh name. Returns false (after a WARNING) if the
// input isn't a closed, consistently wound manifold.
inline bool build_trimesh(const std::string& label, std::vector<Vec3> vertices,
                          std::vector<std::array<int, 3>> triangles, TriMesh& mesh)
{
    TriMesh m;
    m.name      = label;
    m.vertices  = std::move(vertices);
    m.triangles = std::move(triangles);

    if (m.vertices.empty()) { WARNING("build_trimesh: " << label << ": no vertices"); return false; }
    for (const auto& t : m.triangles)
        for (int k = 0; k < 3; ++k)
            if (t[k] < 0 || t[k] >= (int)m.vertices.size())
            { WARNING("build_trimesh: " << label << ": triangle vertex index " << t[k] << " out of range"); return false; }

    // Drop degenerate triangles (repeated vertex after welding, or ~zero area).
    const auto cross_of = [&](const std::array<int, 3>& t)
    { return (m.vertices[t[1]] - m.vertices[t[0]]).cross(m.vertices[t[2]] - m.vertices[t[0]]); };
    m.triangles.erase(std::remove_if(m.triangles.begin(), m.triangles.end(),
                          [&](const std::array<int, 3>& t) { return cross_of(t).norm() < 1e-12; }),
                      m.triangles.end());
    if (m.triangles.empty()) { WARNING("build_trimesh: " << label << ": no usable triangles"); return false; }

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
            { WARNING("build_trimesh: " << label << ": edge used twice in one direction (non-manifold or inconsistent winding)"); return false; }
    for (const auto& [edge, owner] : edge_owner)
        if (!edge_owner.count({ edge.second, edge.first }))
        { WARNING("build_trimesh: " << label << ": mesh is not closed (open boundary edge)"); return false; }

    // Face areas, the triangle across each edge (found via the reverse edge), then all the normals.
    const size_t n_tris = m.triangles.size();
    m.face_area.resize(n_tris);
    m.edge_neighbor.resize(n_tris);
    for (int t = 0; t < (int)n_tris; ++t)
    {
        m.face_area[t] = 0.5 * cross_of(m.triangles[t]).norm();
        for (int e = 0; e < 3; ++e)
            m.edge_neighbor[t][e] = edge_owner.at({ m.triangles[t][(e + 1) % 3], m.triangles[t][e] });
    }
    compute_tri_normals(m.vertices, m.triangles, m.edge_neighbor, m);

    m.aabb_min = m.aabb_max = m.vertices[0];
    for (const Vec3& v : m.vertices) { m.aabb_min = m.aabb_min.cwiseMin(v); m.aabb_max = m.aabb_max.cwiseMax(v); }

    bvh_build(m.bvh, m.vertices, m.triangles);

    mesh = std::move(m);
    return true;
}

// Parses Blender-frame (Z-up) .obj text from `in` into `mesh` in diffpd's Y-up rest frame, with all
// the per-face/edge/vertex data precomputed (see build_trimesh). `label` names the source in
// warnings and becomes the mesh name. Returns false (after a WARNING) if the text isn't a closed,
// consistently wound manifold.
inline bool load_trimesh_obj(std::istream& in, const std::string& label, TriMesh& mesh)
{
    const std::string& path = label;

    std::vector<Vec3>               vertices;
    std::vector<std::array<int, 3>> triangles;

    // Parse `v` and `f` only. Vertices at the same position are welded: OBJ exporters split them
    // per UV/normal, which would break edge adjacency.
    std::map<std::array<long long, 3>, int> welded;
    std::vector<int>                        obj_to_welded; // OBJ vertex index -> index into `vertices`
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
            const auto [it, inserted] = welded.try_emplace(key, (int)vertices.size());
            if (inserted) vertices.push_back(p);
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
                triangles.push_back({ face[0], face[k], face[k + 1] });
        }
    }

    return build_trimesh(label, std::move(vertices), std::move(triangles), mesh);
}

// Same, reading from a file (mesh name = file stem).
inline bool load_trimesh_obj(const std::string& path, TriMesh& mesh)
{
    std::ifstream in(path);
    if (!in.is_open()) { WARNING("load_trimesh_obj: could not open " << path); return false; }
    return load_trimesh_obj(in, std::filesystem::path(path).stem().string(), mesh);
}

// Result of a signed closest-point query on a closed mesh.
struct TriMeshHit
{
    Vec3           point;        // closest point on the mesh
    Vec3           normal;       // pseudo-normal of the feature `point` lies on
    Vec3           bary = Vec3::Zero(); // barycentric coordinates of `point` in `triangle`
    Real           dist;         // signed distance, < 0 inside
    int            triangle = -1; // -1 = nothing within the query's max distance
    ContactFeature feature  = ContactFeature::Face;
    int            index    = 0; // edge/vertex slot within `triangle`
};

// Turns a geometric ClosestHit into a signed TriMeshHit: the pseudo-normal of the closest feature
// (face / edge / vertex) gives the normal and decides inside vs. outside.
inline TriMeshHit resolve_closest_hit(const std::vector<std::array<int, 3>>& T, const TriNormals& normals,
                                      const Vec3& p, const ClosestHit& h)
{
    TriMeshHit hit;
    hit.dist = std::numeric_limits<Real>::infinity();
    if (h.triangle < 0) return hit;

    hit.point    = h.point;
    hit.bary     = h.bary;
    hit.triangle = h.triangle;
    hit.feature  = h.feature;
    hit.index    = h.index;

    const auto& tri = T[hit.triangle];
    switch (hit.feature)
    {
        case ContactFeature::Face:   hit.normal = normals.face_normal[hit.triangle];                   break;
        case ContactFeature::Edge:   hit.normal = normals.edge_pseudo_normal[hit.triangle][hit.index];  break;
        case ContactFeature::Vertex: hit.normal = normals.vertex_pseudo_normal[tri[hit.index]];         break;
    }
    // An edge/vertex inside a flat region (e.g. a quad's diagonal) behaves like the face itself.
    if (hit.feature != ContactFeature::Face && hit.normal.dot(normals.face_normal[hit.triangle]) > 1.0 - 1e-9)
        hit.feature = ContactFeature::Face;

    hit.dist = std::sqrt(h.dist2);
    if ((p - hit.point).dot(hit.normal) < 0) hit.dist = -hit.dist;
    return hit;
}

// Signed distance from p to the closed mesh (V, T) with current `normals` and `bvh` over V; everything
// in one frame. Only surface points closer than sqrt(max_dist2) count: if there are none, the hit has
// triangle == -1 and dist == +inf.
inline TriMeshHit mesh_closest(const std::vector<Vec3>& V, const std::vector<std::array<int, 3>>& T,
                               const TriNormals& normals, const Bvh& bvh, const Vec3& p,
                               Real max_dist2 = std::numeric_limits<Real>::infinity())
{
    return resolve_closest_hit(T, normals, p, bvh_closest(bvh, V, T, p, max_dist2));
}

// Rigid mesh in its local frame, p in the same frame.
inline TriMeshHit trimesh_closest(const TriMesh& m, const Vec3& p,
                                  Real max_dist2 = std::numeric_limits<Real>::infinity())
{
    return mesh_closest(m.vertices, m.triangles, m, m.bvh, p, max_dist2);
}

// Reference implementation: tests every triangle (the BVH query must agree with it).
inline TriMeshHit mesh_closest_brute(const std::vector<Vec3>& V, const std::vector<std::array<int, 3>>& T,
                                     const TriNormals& normals, const Vec3& p)
{
    ClosestHit best;
    for (int t = 0; t < (int)T.size(); ++t)
    {
        ContactFeature feature;
        int            index;
        Vec3           bary;
        const Vec3     q  = closest_point_on_triangle(p, V[T[t][0]], V[T[t][1]], V[T[t][2]], feature, index, bary);
        const Real     d2 = (p - q).squaredNorm();
        if (d2 < best.dist2) { best.dist2 = d2; best.point = q; best.bary = bary; best.triangle = t; best.feature = feature; best.index = index; }
    }
    return resolve_closest_hit(T, normals, p, best);
}

inline TriMeshHit trimesh_closest_brute(const TriMesh& m, const Vec3& p)
{
    return mesh_closest_brute(m.vertices, m.triangles, m, p);
}
