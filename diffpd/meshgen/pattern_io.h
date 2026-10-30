#pragma once

// Loader for the pattern contour JSON (clothcreate/FORMAT.md, versions 1-4; 1 = one shape at the top level, 3+ = per-shape 3D `transform`,
// 4 adds `seams`, returned through the optional out-parameter), GL-free.
// Returns plain data (meshgen::PatternShape, pixel coordinates); PatternChart validates the geometry. Throws std::runtime_error
// with the file name on a missing file, bad JSON, an unknown version or segment type, or a malformed point.

#include "meshgen/tri_surface.h"

#include <nlohmann/json.hpp>

#include <fstream>

// `seams` (optional) receives the version 4 sewing list (empty for older files).
inline std::vector<meshgen::PatternShape> load_pattern(const std::string& path, std::vector<meshgen::PatternSeam>* seams = nullptr)
{
    using nlohmann::json;
    auto point = [](const json& a) { return meshgen::Vec2(a.at(0).get<double>(), a.at(1).get<double>()); };
    auto shape = [&](const json& s)
    {
        meshgen::PatternShape out{ point(s.at("start")), s.at("closed").get<bool>(), {} };
        for (const json& g : s.at("segments"))
        {
            const std::string type = g.at("type").get<std::string>();
            if (type == "line")       out.segs.push_back({ false, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero(), point(g.at("to")) });
            else if (type == "curve") out.segs.push_back({ true, point(g.at("c1")), point(g.at("c2")), point(g.at("to")) });
            else throw std::runtime_error("unknown segment type '" + type + "'");
        }
        if (s.contains("transform")) // pivot [x, y], position [x, y, z], rotation quaternion [x, y, z, w]
        {
            const json& t = s.at("transform");
            const json& r = t.at("rotation");
            Eigen::Quaterniond q(r.at(3).get<double>(), r.at(0).get<double>(), r.at(1).get<double>(), r.at(2).get<double>()); // Eigen: w first
            if (q.norm() < 1e-9) throw std::runtime_error("zero rotation quaternion");
            out.placed = true;
            out.pivot = point(t.at("pivot"));
            out.position = Vec3(t.at("position").at(0).get<double>(), t.at("position").at(1).get<double>(), t.at("position").at(2).get<double>());
            out.rot = q.normalized().toRotationMatrix();
        }
        return out;
    };

    // The segments of one side (edge, edge + 1, ... wrapping on a closed shape); throws if the side does not exist.
    auto side_segments = [](const std::vector<meshgen::PatternShape>& shapes, int shape, int edge, int count)
    {
        if (shape < 0 || shape >= (int)shapes.size()) throw std::runtime_error("seam: there is no shape " + std::to_string(shape));
        const meshgen::PatternShape& s = shapes[shape];
        const int n = (int)s.segs.size();
        if (edge < 0 || edge >= n) throw std::runtime_error("seam: shape " + std::to_string(shape) + " has no edge " + std::to_string(edge));
        if (count < 1 || count > n) throw std::runtime_error("seam: a side of shape " + std::to_string(shape) + " cannot have " + std::to_string(count) + " edges");
        if (!s.closed && edge + count > n) throw std::runtime_error("seam: a side runs past the end of the open shape " + std::to_string(shape));
        std::set<int> out;
        for (int i = 0; i < count; ++i) out.insert((edge + i) % n);
        return out;
    };
    auto check_seam = [&](const meshgen::PatternSeam& s, const std::vector<meshgen::PatternShape>& shapes)
    {
        const std::set<int> a = side_segments(shapes, s.shape_a, s.edge_a, s.count_a), b = side_segments(shapes, s.shape_b, s.edge_b, s.count_b);
        if (s.level < 1 || s.level > 4) throw std::runtime_error("seam: level " + std::to_string(s.level) + " is not in 1..4");
        if (s.shape_a == s.shape_b)
            for (int e : a) if (b.count(e)) throw std::runtime_error("seam: the two sides share edge " + std::to_string(e) + " of shape " + std::to_string(s.shape_a));
    };

    std::ifstream in(path);
    if (!in) throw std::runtime_error("pattern: cannot open " + path);
    try
    {
        const json j = json::parse(in);
        const int version = j.value("version", 1);
        std::vector<meshgen::PatternShape> shapes;
        if (version >= 2 && version <= 4) for (const json& s : j.at("shapes")) shapes.push_back(shape(s));
        else if (version == 1) shapes.push_back(shape(j));
        else throw std::runtime_error("unsupported version " + std::to_string(version));
        if (shapes.empty()) throw std::runtime_error("no shapes");
        if (seams && version == 4 && j.contains("seams"))
            for (const json& m : j.at("seams"))
            {
                const json &a = m.at("a"), &b = m.at("b");
                const meshgen::PatternSeam s{ a.at("shape").get<int>(), a.at("edge").get<int>(), b.at("shape").get<int>(), b.at("edge").get<int>(),
                                              m.at("reversed").get<bool>(), a.value("count", 1), b.value("count", 1), m.value("level", 1) };
                check_seam(s, shapes);
                seams->push_back(s);
            }
        return shapes;
    }
    catch (const std::exception& e) // nlohmann::json errors derive from std::exception
    {
        throw std::runtime_error("pattern: " + path + ": " + e.what());
    }
}
