#pragma once

// Frame-by-frame .obj export of a cloth Object: where it goes (ANIM_DIR) and the writers.

#include "cloth/object.h"
#include "collision/collider.h" // diffpd_to_blender

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

// Output folder of the frame .obj files; set once by main().
inline std::string ANIM_DIR;

inline void clear_folder(const std::string& folder)
{
    namespace fs = std::filesystem;

    if (!fs::exists(folder))
    {
        fs::create_directories(folder);
        return;
    }

    ASSERT(fs::is_directory(folder), "clear_folder: not a directory: " << folder);

    for (const auto& entry : fs::directory_iterator(folder))
        if (entry.is_regular_file() && entry.path().extension() == ".obj")
            fs::remove(entry.path());
}

inline void write_obj(const std::string& path, const Object& obj)
{
    const SimMesh& mesh = obj.mesh;

    std::ofstream out(path);
    ASSERT(out.is_open(), "could not open " << path);
    out << std::fixed << std::setprecision(6);

    for (Index vi = 0; vi < Index(mesh.vertices.size()); ++vi)
    {
        const Vec3 p = mesh.position(obj, vi);
        out << "v " << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    }
    for (const auto& e : mesh.edges)
        out << "l " << e.first + 1 << ' ' << e.second + 1 << '\n';
}

// ---- Blender export: one static .obj with faces + one .pc2 point cache --------------------------------
// In Blender: import the .obj, add a "Mesh Cache" modifier (Format PC2, file = the .pc2). Same vertex order,
// so the cache plays on the imported mesh. Coordinates are converted from diffpd (Y up) to Blender (Z up).

// Static mesh with faces (grid -> quads, tri mesh -> triangles) at the object's current pose.
inline void write_obj_faces(const std::string& path, const Object& obj, bool to_blender = true)
{
    const SimMesh& m = obj.mesh;
    std::ofstream out(path);
    ASSERT(out.is_open(), "could not open " << path);
    out << std::fixed << std::setprecision(6);
    for (Index vi = 0; vi < Index(m.vertices.size()); ++vi)
    {
        const Vec3 q = m.position(obj, vi);
        const Vec3 p = to_blender ? diffpd_to_blender(q) : q;
        out << "v " << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    }
    if (!m.triangles.empty())
        for (const auto& t : m.triangles) out << "f " << t[0] + 1 << ' ' << t[1] + 1 << ' ' << t[2] + 1 << '\n';
    else
    {
        const Index cells_i = m.wrap_i ? m.width : m.width - 1;
        for (Index i = 0; i < cells_i; ++i)
            for (Index j = 0; j + 1 < m.height; ++j)
            {
                const Index i1 = (i + 1) % m.width;
                out << "f " << i * m.height + j + 1 << ' ' << i1 * m.height + j + 1 << ' '
                    << i1 * m.height + j + 2 << ' ' << i * m.height + j + 2 << '\n';
            }
    }
}

// Collects every output frame of a run (all vertices, pinned included, so a moving waist is right).
struct PointCache
{
    int n_points = 0;
    std::vector<float> data; // frame-major, xyz per vertex

    void add_frame(const Object& obj)
    {
        n_points = int(obj.mesh.vertices.size());
        for (Index vi = 0; vi < n_points; ++vi)
        {
            const Vec3 p = diffpd_to_blender(obj.mesh.position(obj, vi));
            data.push_back(float(p.x())); data.push_back(float(p.y())); data.push_back(float(p.z()));
        }
    }

    void write(const std::string& path) const
    {
        if (n_points == 0) return;
        std::ofstream out(path, std::ios::binary);
        ASSERT(out.is_open(), "could not open " << path);
        const int32_t version = 1, n = n_points, samples = int32_t(data.size() / (3 * size_t(n_points)));
        const float start = 0.f, rate = 1.f; // one sample per frame; set the Blender scene fps to the sim FPS
        out.write("POINTCACHE2\0", 12);
        out.write((const char*)&version, 4); out.write((const char*)&n, 4);
        out.write((const char*)&start, 4);   out.write((const char*)&rate, 4);
        out.write((const char*)&samples, 4);
        out.write((const char*)data.data(), std::streamsize(data.size() * sizeof(float)));
    }
};

inline void write_obj_frame(const Object& obj, int step, const std::string& prefix = "frame") 
{
    std::ostringstream name;
    name << prefix << "_" << std::setfill('0') << std::setw(6) << step << ".obj";
    write_obj((std::filesystem::path(ANIM_DIR) / name.str()).string(), obj);
}
