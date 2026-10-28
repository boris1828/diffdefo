#pragma once

// Loader for the pattern contour JSON (clothcreate/FORMAT.md, version 2; version 1 = one shape at the top level), GL-free.
// Returns plain data (meshgen::PatternShape, pixel coordinates); PatternChart validates the geometry. Throws std::runtime_error
// with the file name on a missing file, bad JSON, an unknown version or segment type, or a malformed point.

#include "tri_surface.h"

#include <nlohmann/json.hpp>

#include <fstream>

inline std::vector<meshgen::PatternShape> load_pattern(const std::string& path)
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
        return out;
    };

    std::ifstream in(path);
    if (!in) throw std::runtime_error("pattern: cannot open " + path);
    try
    {
        const json j = json::parse(in);
        const int version = j.value("version", 1);
        std::vector<meshgen::PatternShape> shapes;
        if (version == 2) for (const json& s : j.at("shapes")) shapes.push_back(shape(s));
        else if (version == 1) shapes.push_back(shape(j));
        else throw std::runtime_error("unsupported version " + std::to_string(version));
        if (shapes.empty()) throw std::runtime_error("no shapes");
        return shapes;
    }
    catch (const std::exception& e) // nlohmann::json errors derive from std::exception
    {
        throw std::runtime_error("pattern: " + path + ": " + e.what());
    }
}
