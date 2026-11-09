#pragma once
// Save / load an AppConfig as a plain "key value..." text file (one key per line, `collider` repeated per collider).
// Frozen to the AppConfig fields of the day: a field missing from a file keeps its current value, an unknown key is
// ignored, and a new AppConfig field must be added to config_fields() (and collider_fields()) to be saved.
#include "app/config.h"

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
    // appended later: a line from an older file ends before these and keeps the defaults (a read at the end of the stream leaves the value alone)
    cfg_xfer(s, c.arc_radius);      cfg_xfer(s, c.tube_radius);      cfg_xfer(s, c.arc_angle);
    cfg_xfer(s, c.rigid_pose.position);
    for (int a = 0; a < 4; ++a) cfg_xfer(s, c.rigid_pose.rotation.coeffs()[a]);
}

// F(key, member) is called for every saved AppConfig field.
template <class F>
void config_fields(AppConfig& c, F&& f)
{
    f("run_mode", c.run_mode);
    f("cloth_type", c.cloth.cloth_type);
    f("origin", c.cloth.origin);
    f("m_tot", c.cloth.m_tot);
    f("flag_shear", c.cloth.flag_shear);
    f("flag_bending", c.cloth.flag_bending);
    f("per_type_stiffness", c.per_type_stiffness);
    f("stiffness", c.stiffness);
    f("target_stiffness", c.target_stiffness);
    f("stiffness_per_type", c.stiffness_per_type);
    f("target_stiffness_per_type", c.target_stiffness_per_type);
    f("width", c.cloth.width);
    f("height", c.cloth.height);
    f("size_x", c.cloth.size_x);
    f("size_y", c.cloth.size_y);
    f("pin_mode", c.cloth.pin_mode);
    f("rail_stride", c.cloth.rail_stride);
    f("rail_k", c.cloth.rail_k);
    f("rail_spacer_k", c.cloth.rail_spacer_k);
    f("hang_mode", c.cloth.hang_mode);
    f("particles_per_ring", c.cloth.particles_per_ring);
    f("num_rings", c.cloth.num_rings);
    f("radius_top", c.cloth.radius_top);
    f("radius_bottom", c.cloth.radius_bottom);
    f("skirt_height", c.cloth.skirt_height);
    f("pattern_size", c.cloth.pattern_size); // was pattern_scale (m/px) before version 7 files: an old config line is ignored
    f("seam_stiffness", c.cloth.seam_stiffness);
    f("mesh_gen", c.cloth.mesh_gen);
    f("tri_h", c.cloth.tri_h);
    f("tri_beta", c.cloth.tri_beta);
    f("tri_smooth_iters", c.cloth.tri_smooth_iters);
    f("tri_seed", c.cloth.tri_seed);
    f("tri_energy", c.cloth.tri_energy);
    f("arap_stiffness", c.arap_stiffness);
    f("target_arap_stiffness", c.target_arap_stiffness);
    f("arap_bend_stiffness", c.arap_bend_stiffness);
    f("target_arap_bend_stiffness", c.target_arap_bend_stiffness);
    f("arap_flat_rest", c.cloth.arap_flat_rest);
    f("contact_model", c.world.contact.model);
    f("contact_margin", c.world.contact.margin);
    f("contact_recheck_mode", c.world.contact.recheck);
    f("contact_recheck_count", c.world.contact.recheck_count);
    f("contact_friction", c.world.contact.friction);
    f("compute_unresolved", c.world.compute_unresolved);
    f("unresolved_contact_threshold", c.world.unresolved_contact_threshold);
    f("waist_attach_enabled", c.world.waist_attach_enabled);
    f("gravity", c.solver.gravity);
    f("damping_enabled", c.solver.damping_enabled);
    f("damping_alpha", c.solver.damping_alpha);
    f("damping_beta", c.solver.damping_beta);
    f("FPS", c.solver.FPS);
    f("frame_substeps", c.solver.frame_substeps);
    f("secs", c.solver.secs);
    f("n_iters", c.solver.n_iters);
    f("n_iters_adjoint", c.solver.n_iters_adjoint);
    f("anderson_m", c.solver.anderson_m);
    f("global_solve", c.solver.global_solve);
    f("contact_method", c.solver.contact_method);
    f("ncp_friction", c.solver.ncp.friction);
    f("ncp_margin", c.solver.ncp.margin);
    f("ncp_iters", c.solver.ncp.n_iters);
    f("ncp_tol", c.solver.ncp.tol);
    f("ncp_schur", c.solver.ncp.schur);
    f("record_on_run", c.record_on_run);
    f("settle_seconds", c.settle_seconds);
    f("settle_rest_speed", c.settle_rest_speed);
    f("settle_iters", c.settle_iters);
    f("seam_settle_seconds", c.seam_settle_seconds);
    f("seam_settle_iters", c.seam_settle_iters);
    f("pull_accel", c.pull_accel);
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
    out << "animation_file " << cfg.world.animation_file << '\n'; // the strings: whole rest of the line
    out << "pattern_file " << cfg.cloth.pattern_file << '\n';
    for (Collider& c : cfg.world.colliders) { out << "collider"; collider_fields(out, c); out << '\n'; }
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
    if (kv.count("animation_file")) cfg.world.animation_file = kv["animation_file"];
    if (kv.count("pattern_file")) cfg.cloth.pattern_file = kv["pattern_file"];
    cfg.world.colliders.clear();
    for (const std::string& l : collider_lines)
    {
        Collider c = default_config_collider();
        std::istringstream is(l);
        collider_fields(is, c);
        cfg.world.colliders.push_back(c);
    }
    cfg.settled = SettledStart{}; // a loaded config never inherits the old rest pose
    return true;
}
