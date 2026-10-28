#pragma once
// Save / load an AppConfig as a plain "key value..." text file (one key per line, `collider` repeated per collider).
// Frozen to the AppConfig fields of the day: a field missing from a file keeps its current value, an unknown key is
// ignored, and a new AppConfig field must be added to config_fields() (and collider_fields()) to be saved.
#include "diffpd_types.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <vector>
#include <type_traits>

// One value, written (ostream: ' ' + value) or read (istream) by the same call, so save and load cannot drift apart.
template <class S, class T>
void cfg_xfer(S& s, T& v)
{
    constexpr bool writing = std::is_base_of_v<std::ostream, S>;
    if constexpr (std::is_enum_v<T>)
    {
        int i = (int)v;
        cfg_xfer(s, i);
        v = (T)i;
    }
    else if constexpr (std::is_same_v<T, Vec3>)
        for (int a = 0; a < 3; ++a) cfg_xfer(s, v[a]);
    else if constexpr (std::is_same_v<T, Stiffness>)
    {
        cfg_xfer(s, v.stretch); cfg_xfer(s, v.shear); cfg_xfer(s, v.bend);
    }
    else if constexpr (writing)
        s << ' ' << v;
    else
        s >> v;
}

template <class S>
void collider_fields(S& s, Collider& c)
{
    cfg_xfer(s, c.type);
    cfg_xfer(s, c.sphere_center);   cfg_xfer(s, c.sphere_radius);
    cfg_xfer(s, c.cylinder_origin); cfg_xfer(s, c.cylinder_axis); cfg_xfer(s, c.cylinder_radius);
    cfg_xfer(s, c.plane_origin);    cfg_xfer(s, c.plane_normal);
    cfg_xfer(s, c.capsule_p0);      cfg_xfer(s, c.capsule_p1);    cfg_xfer(s, c.capsule_radius);
    cfg_xfer(s, c.velocity);
    cfg_xfer(s, c.rotation_axis);   cfg_xfer(s, c.rotation_origin); cfg_xfer(s, c.omega);
}

// F(key, member) is called for every saved AppConfig field.
template <class F>
void config_fields(AppConfig& c, F&& f)
{
    f("run_mode", c.run_mode);
    f("cloth_type", c.cloth_type);
    f("origin", c.origin);
    f("m_tot", c.m_tot);
    f("flag_shear", c.flag_shear);
    f("flag_bending", c.flag_bending);
    f("per_type_stiffness", c.per_type_stiffness);
    f("stiffness", c.stiffness);
    f("target_stiffness", c.target_stiffness);
    f("stiffness_per_type", c.stiffness_per_type);
    f("target_stiffness_per_type", c.target_stiffness_per_type);
    f("width", c.width);
    f("height", c.height);
    f("size_x", c.size_x);
    f("size_y", c.size_y);
    f("pin_mode", c.pin_mode);
    f("hang_mode", c.hang_mode);
    f("particles_per_ring", c.particles_per_ring);
    f("num_rings", c.num_rings);
    f("radius_top", c.radius_top);
    f("radius_bottom", c.radius_bottom);
    f("skirt_height", c.skirt_height);
    f("pattern_scale", c.pattern_scale);
    f("mesh_gen", c.mesh_gen);
    f("tri_h", c.tri_h);
    f("tri_beta", c.tri_beta);
    f("tri_smooth_iters", c.tri_smooth_iters);
    f("tri_seed", c.tri_seed);
    f("tri_energy", c.tri_energy);
    f("arap_stiffness", c.arap_stiffness);
    f("target_arap_stiffness", c.target_arap_stiffness);
    f("arap_bend_stiffness", c.arap_bend_stiffness);
    f("target_arap_bend_stiffness", c.target_arap_bend_stiffness);
    f("arap_flat_rest", c.arap_flat_rest);
    f("contact_recheck_mode", c.contact_recheck_mode);
    f("contact_recheck_count", c.contact_recheck_count);
    f("speculative_contact", c.speculative_contact);
    f("contact_margin", c.contact_margin);
    f("unresolved_contact_threshold", c.unresolved_contact_threshold);
    f("waist_attach_enabled", c.waist_attach_enabled);
    f("gravity", c.gravity);
    f("damping_enabled", c.damping_enabled);
    f("damping_alpha", c.damping_alpha);
    f("damping_beta", c.damping_beta);
    f("FPS", c.FPS);
    f("frame_substeps", c.frame_substeps);
    f("secs", c.secs);
    f("n_iters", c.n_iters);
    f("n_iters_adjoint", c.n_iters_adjoint);
    f("anderson_m", c.anderson_m);
    f("record_on_run", c.record_on_run);
    f("settle_seconds", c.settle_seconds);
    f("settle_rest_speed", c.settle_rest_speed);
    f("settle_iters", c.settle_iters);
    f("run_fd_check", c.run_fd_check);
    for (int i = 0; i < 9; ++i) f(("fd_eps" + std::to_string(i)).c_str(), c.fd_eps_selected[i]);
}

inline bool save_config(const std::filesystem::path& path, AppConfig cfg)
{
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream out(path);
    if (!out) return false;
    out << std::setprecision(17);
    config_fields(cfg, [&](const char* key, auto& v) { out << key; cfg_xfer(out, v); out << '\n'; });
    out << "animation_file " << cfg.animation_file << '\n'; // the strings: whole rest of the line
    out << "pattern_file " << cfg.pattern_file << '\n';
    for (Collider& c : cfg.colliders) { out << "collider"; collider_fields(out, c); out << '\n'; }
    return bool(out);
}

// Fields absent from the file keep their current value in `cfg`. Returns false if the file can't be opened.
inline bool load_config(const std::filesystem::path& path, AppConfig& cfg)
{
    std::ifstream in(path);
    if (!in) return false;
    std::map<std::string, std::string> kv;
    std::vector<std::string> collider_lines;
    for (std::string line; std::getline(in, line);)
    {
        const size_t sp = line.find(' ');
        const std::string key = line.substr(0, sp), rest = sp == std::string::npos ? "" : line.substr(sp + 1);
        if (key == "collider") collider_lines.push_back(rest);
        else kv[key] = rest;
    }
    config_fields(cfg, [&](const char* key, auto& v) {
        const auto it = kv.find(key);
        if (it == kv.end()) return;
        std::istringstream is(it->second);
        cfg_xfer(is, v);
    });
    if (kv.count("animation_file")) cfg.animation_file = kv["animation_file"];
    if (kv.count("pattern_file")) cfg.pattern_file = kv["pattern_file"];
    cfg.colliders.clear();
    for (const std::string& l : collider_lines)
    {
        Collider c = default_config_collider();
        std::istringstream is(l);
        collider_fields(is, c);
        cfg.colliders.push_back(c);
    }
    cfg.settled = SettledStart{}; // a loaded config never inherits the old rest pose
    return true;
}
