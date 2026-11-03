#pragma once

// Loader for the pattern contour JSON (clothcreate/FORMAT.md, version 7 only: metres, in the body's world; seams and pulls are returned
// through the optional out-parameters), GL-free.
// Returns plain data (meshgen::PatternShape); PatternChart validates the geometry. Throws std::runtime_error
// with the file name on a missing file, bad JSON, any version but 7, a missing transform / scale or an unknown segment type, or a malformed point.

#include "meshgen/tri_surface.h"

#include <nlohmann/json.hpp>

#include <fstream>

// `seams` (optional) receives the sewing list, `pulls` (optional) the pull list (empty when the file has none).
inline std::vector<meshgen::PatternShape> load_pattern(const std::string& path, std::vector<meshgen::PatternSeam>* seams = nullptr,
                                                       std::vector<meshgen::PatternPull>* pulls = nullptr)
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
        {   // pivot [x, y], position [x, y, z], rotation quaternion [x, y, z, w], uniform scale: all required
            const json& t = s.at("transform");
            const json& r = t.at("rotation");
            Eigen::Quaterniond q(r.at(3).get<double>(), r.at(0).get<double>(), r.at(1).get<double>(), r.at(2).get<double>()); // Eigen: w first
            if (q.norm() < 1e-9) throw std::runtime_error("zero rotation quaternion");
            out.pivot = point(t.at("pivot"));
            out.position = Vec3(t.at("position").at(0).get<double>(), t.at("position").at(1).get<double>(), t.at("position").at(2).get<double>());
            out.rot = q.normalized().toRotationMatrix();
            out.sc = t.at("scale").get<double>();
            if (!(out.sc > 0)) throw std::runtime_error("transform: scale must be positive");
        }
        if (s.contains("deformation"))
        {
            const json& d = s.at("deformation");
            const std::string type = d.at("type").get<std::string>();
            if (type == "cylinder")
            {
                const std::string dir = d.at("direction").get<std::string>();
                if (dir != "x" && dir != "y") throw std::runtime_error("deformation: direction '" + dir + "' is not \"x\" or \"y\"");
                const double strength = d.at("strength").get<double>();
                if (!(std::abs(strength) <= 1.0)) throw std::runtime_error("deformation: strength " + std::to_string(strength) + " is not in [-1, 1]");
                out.deform = { meshgen::PatternDeform::Kind::Cylinder, dir == "y" ? 1 : 0, strength };
            }
            else
            {
                std::fprintf(stderr, "pattern: unknown deformation type '%s', the piece stays flat\n", type.c_str());
                out.deform.kind = meshgen::PatternDeform::Kind::Unknown;
            }
        }
        return out;
    };

    // The segments of one side (edge, edge + 1, ... wrapping on a closed shape); throws if the side does not exist. `what` = "seam" / "pull" in the messages.
    auto side_segments = [](const std::vector<meshgen::PatternShape>& shapes, int shape, int edge, int count, const char* what = "seam")
    {
        const std::string w = std::string(what) + ": ";
        if (shape < 0 || shape >= (int)shapes.size()) throw std::runtime_error(w + "there is no shape " + std::to_string(shape));
        const meshgen::PatternShape& s = shapes[shape];
        const int n = (int)s.segs.size();
        if (edge < 0 || edge >= n) throw std::runtime_error(w + "shape " + std::to_string(shape) + " has no edge " + std::to_string(edge));
        if (count < 1 || count > n) throw std::runtime_error(w + "a side of shape " + std::to_string(shape) + " cannot have " + std::to_string(count) + " edges");
        if (!s.closed && edge + count > n) throw std::runtime_error(w + "a side runs past the end of the open shape " + std::to_string(shape));
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
        if (version != 7) throw std::runtime_error("unsupported version " + std::to_string(version) + " (only 7: re-export the file from the cloth tool)");
        std::vector<meshgen::PatternShape> shapes;
        for (const json& s : j.at("shapes")) shapes.push_back(shape(s));
        if (shapes.empty()) throw std::runtime_error("no shapes");
        if (pulls && j.contains("pulls"))
            for (const json& m : j.at("pulls"))
            {
                const json& d = m.at("direction");
                int nonzero = 0;
                bool unit = d.is_array() && d.size() == 3;
                Vec3 dir = Vec3::Zero();
                for (int i = 0; unit && i < 3; ++i)
                {
                    dir[i] = d.at(i).get<double>();
                    if (dir[i] != 0.0) { ++nonzero; unit = std::abs(dir[i]) == 1.0; }
                }
                if (!unit || nonzero != 1) throw std::runtime_error("pull: direction must be one of the six world unit axes, e.g. [0, -1, 0]");
                const meshgen::PatternPull p{ m.at("shape").get<int>(), m.at("edge").get<int>(), m.value("count", 1), dir };
                side_segments(shapes, p.shape, p.edge, p.count, "pull");
                pulls->push_back(p);
            }
        if (seams && j.contains("seams"))
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
