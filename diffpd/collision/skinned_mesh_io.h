#pragma once

#include "collision/skinned_mesh.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// Basis change of blender_to_diffpd() as a matrix: (x, y, z) -> (x, z, -y), a proper rotation.
inline const Mat3& blender_to_diffpd_matrix()
{
    static const Mat3 C = (Mat3() << 1, 0, 0,
                                     0, 0, 1,
                                     0, -1, 0).finished();
    return C;
}

// Reads a "diffpd_skinned_body" .json (written by animation/export_skinned_body.py) into `out`,
// converting everything from Blender's Z-up world frame to diffpd's Y-up frame: vertices become
// C v, and each skinning matrix [R | t] becomes [C R C^T | C t] (since C S x = (C S C^T)(C x)).
// `label` names the source in warnings. Returns false (after a WARNING, never throwing) if the file
// isn't a valid body: wrong format/units/fps, inconsistent array sizes, bad bone indices, weights
// that aren't a convex combination, or a mesh that isn't closed. Weight rows are renormalized to
// sum to exactly 1 (the file stores 7 decimals). Skinning matrices that aren't rotations only warn,
// since scale is legal for LBS but would break the contact velocity derived from them later.
inline bool load_skinned_body(const nlohmann::json& j, const std::string& label, SkinnedMesh& out)
{
    const auto fail = [&](const std::string& why)
    {
        WARNING("load_skinned_body: " << label << ": " << why);
        return false;
    };

    try
    {
        if (!j.is_object() || j.value("format", std::string()) != "diffpd_skinned_body") return fail("not a diffpd_skinned_body file");
        if (j.value("version", 0) != 1)                                                  return fail("unsupported version (expected 1)");
        if (j.value("coordinates", std::string()) != "blender_world_z_up")               return fail("unsupported coordinates (expected blender_world_z_up)");
        if (j.value("units", std::string()) != "m")                                      return fail("unsupported units (expected m)");
        if (std::abs(j.at("fps").get<Real>() - kColliderAnimFPS) > 1e-6)
            return fail("fps is " + std::to_string(j.at("fps").get<Real>()) + ", the sim needs " + std::to_string((int)kColliderAnimFPS));

        SkinnedMesh sb;
        sb.name = j.contains("source") ? j["source"].value("mesh", label) : label;

        // ---- skeleton ----
        const auto& skin_json = j.at("skin");
        sb.bone_names  = skin_json.at("bones").get<std::vector<std::string>>();
        sb.bone_parent = skin_json.at("parents").get<std::vector<int>>();
        const int n_bones = sb.n_bones();
        if (n_bones < 1)                          return fail("no bones");
        if ((int)sb.bone_parent.size() != n_bones) return fail("parents/bones size mismatch");
        for (int b = 0; b < n_bones; ++b)
            if (sb.bone_parent[b] < -1 || sb.bone_parent[b] >= n_bones || sb.bone_parent[b] == b)
                return fail("bad parent index for bone '" + sb.bone_names[b] + "'");
        if (skin_json.at("max_influences").get<int>() != SkinnedMesh::kMaxInfluences)
            return fail("max_influences must be " + std::to_string(SkinnedMesh::kMaxInfluences));

        // ---- mesh (vertices keep their order: weights are per vertex) ----
        const Mat3& C = blender_to_diffpd_matrix();
        std::vector<Vec3>               vertices;
        std::vector<std::array<int, 3>> triangles;
        for (const auto& v : j.at("mesh").at("vertices"))
        {
            if (v.size() != 3) return fail("a vertex does not have 3 coordinates");
            vertices.push_back(C * Vec3(v[0].get<Real>(), v[1].get<Real>(), v[2].get<Real>()));
        }
        for (const auto& t : j.at("mesh").at("triangles"))
        {
            if (t.size() != 3) return fail("a triangle does not have 3 indices");
            triangles.push_back({ t[0].get<int>(), t[1].get<int>(), t[2].get<int>() });
        }
        const size_t n_verts = vertices.size();

        // ---- weights ----
        const auto& bone_index_json = skin_json.at("bone_index");
        const auto& weight_json     = skin_json.at("weight");
        if (bone_index_json.size() != n_verts || weight_json.size() != n_verts) return fail("bone_index/weight size != vertex count");
        sb.bone_index.resize(n_verts);
        sb.weight.resize(n_verts);
        for (size_t i = 0; i < n_verts; ++i)
        {
            if (bone_index_json[i].size() != SkinnedMesh::kMaxInfluences || weight_json[i].size() != SkinnedMesh::kMaxInfluences)
                return fail("vertex " + std::to_string(i) + ": expected " + std::to_string(SkinnedMesh::kMaxInfluences) + " influence slots");
            Real sum = 0.0;
            for (int k = 0; k < SkinnedMesh::kMaxInfluences; ++k)
            {
                const int  b = bone_index_json[i][k].get<int>();
                const Real w = weight_json[i][k].get<Real>();
                if (b < 0 || b >= n_bones) return fail("vertex " + std::to_string(i) + ": bone index " + std::to_string(b) + " out of range");
                if (w < -1e-9)             return fail("vertex " + std::to_string(i) + ": negative weight");
                sb.bone_index[i][k] = b;
                sb.weight[i][k]     = std::max(w, 0.0);
                sum += sb.weight[i][k];
            }
            if (std::abs(sum - 1.0) > 1e-4) return fail("vertex " + std::to_string(i) + ": weights sum to " + std::to_string(sum) + ", not 1");
            for (Real& w : sb.weight[i]) w /= sum;
        }

        // ---- per-frame skinning matrices: 12 numbers per bone, row-major 3x4 [R | t] ----
        const auto& frames = j.at("frames");
        sb.n_frames        = j.at("n_frames").get<int>();
        if (sb.n_frames < 1 || (int)frames.size() != sb.n_frames) return fail("n_frames does not match the number of frame rows");
        sb.skin.resize((size_t)sb.n_frames * n_bones);
        Real max_non_rotation = 0.0;
        for (int f = 0; f < sb.n_frames; ++f)
        {
            const auto& row = frames[f];
            if ((int)row.size() != 12 * n_bones) return fail("frame " + std::to_string(f) + ": expected " + std::to_string(12 * n_bones) + " numbers");
            for (int b = 0; b < n_bones; ++b)
            {
                Mat3 R;
                Vec3 t;
                for (int r = 0; r < 3; ++r)
                {
                    for (int c = 0; c < 3; ++c) R(r, c) = row[b * 12 + r * 4 + c].get<Real>();
                    t(r) = row[b * 12 + r * 4 + 3].get<Real>();
                }
                AffineTransform& T = sb.skin[(size_t)f * n_bones + b];
                T.R = C * R * C.transpose();
                T.t = C * t;
                max_non_rotation = std::max(max_non_rotation, (T.R.transpose() * T.R - Mat3::Identity()).cwiseAbs().maxCoeff());
            }
        }
        if (max_non_rotation > 1e-3)
            WARNING("load_skinned_body: " << label << ": skinning matrices are not pure rotations (max |R^T R - I| = "
                    << max_non_rotation << "); scale/shear will distort the contact velocity");

        if (!build_trimesh(sb.name, std::move(vertices), std::move(triangles), sb.rest)) return false; // already warned

        std::cout << "[skinned] " << sb.name << ": " << sb.rest.vertices.size() << " vertices, " << sb.rest.triangles.size()
                  << " triangles, " << n_bones << " bones, " << sb.n_frames << " frames\n";
        out = std::move(sb);
        return true;
    }
    catch (const std::exception& e)
    {
        return fail(std::string("malformed file: ") + e.what());
    }
}
