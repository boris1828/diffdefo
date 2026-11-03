#include "collision/animation_io.h"
#include "collision/skinned_mesh_io.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace fs = std::filesystem;

// The animation-owned globals (mesh registry, skinned bodies) outlive a load, so "Animation: None" must reset them too.
static void clear_animation_assets(CollisionWorld& world)
{
    world.trimeshes.clear();
    world.skinned_meshes.clear();
    world.skinned_states.clear();
}

// Parses an animation .json, appending one `animated=true` Collider per supported "colliders_metadata"
// entry and returning the matching frame tracks. Missing/unreadable file just warns and returns
// empty rather than failing, since this is an optional layer on the config-managed collider list.
// Entries with an unsupported "type" (anything but sphere/capsule/mesh), or a mesh that fails to
// load, are skipped with a warning. A mesh entry's "mesh_file" is relative to the .json's folder.
void load_collider_animation(const std::string& path, CollisionWorld& world)
{
    std::vector<Collider>&          out_colliders = world.colliders;
    std::vector<ColliderAnimation>& anims         = world.animations;
    anims.clear();
    clear_animation_assets(world);

    std::ifstream in(path);
    if (!in.is_open())
    {
        WARNING("load_collider_animation: could not open " << path << " (skipping animated colliders)");
        return;
    }

    nlohmann::json j;
    try { in >> j; }
    catch (const std::exception& e)
    {
        WARNING("load_collider_animation: failed to parse " << path << ": " << e.what());
        return;
    }

    // A skinned-body file (animation/export_skinned_body.py) shares this folder and the Animation
    // dropdown but has no "colliders_metadata": it becomes one Skinned collider and no rigid tracks.
    if (j.is_object() && j.value("format", std::string()) == "diffpd_skinned_body")
    {
        SkinnedMesh body;
        if (load_skinned_body(j, fs::path(path).stem().string(), body))
        {
            Collider c;
            c.type       = ColliderType::Skinned;
            c.skinned_id = (int)world.skinned_meshes.size();
            world.skinned_meshes.push_back(std::move(body));
            world.skinned_states.emplace_back();
            world.stamp_skinned_body(c.skinned_id, 1, 0); // so previews/viewers have a valid pose before any run
            out_colliders.push_back(c);
        }
        return;
    }
    if (!j.is_object() || !j.contains("colliders_metadata") || !j.contains("frames"))
    {
        WARNING("load_collider_animation: " << path << ": no \"colliders_metadata\"/\"frames\" (skipping)");
        return;
    }

    // name -> index into `anims`, so the per-frame loop below can find each track by name.
    std::unordered_map<std::string, int> name_to_anim_id;

    for (auto& [name, meta] : j.at("colliders_metadata").items())
    {
        const std::string type_str = meta.value("type", "capsule");
        if (type_str != "sphere" && type_str != "capsule" && type_str != "mesh")
        {
            WARNING("load_collider_animation: " << path << ": skipping '" << name
                    << "' (unsupported collider type '" << type_str << "')");
            continue;
        }

        ColliderAnimation anim;
        anim.name = name;
        anim.type = (type_str == "sphere") ? ColliderType::Sphere
                  : (type_str == "mesh")   ? ColliderType::Mesh
                                           : ColliderType::Capsule;

        if (anim.type == ColliderType::Mesh)
        {
            TriMesh mesh;
            if (meta.contains("mesh")) // inline OBJ text under the top-level "meshes" block
            {
                const std::string key = meta["mesh"].get<std::string>();
                if (!j.contains("meshes") || !j["meshes"].contains(key))
                {
                    WARNING("load_collider_animation: " << path << ": '" << name << "' references missing mesh '" << key << "'");
                    continue;
                }
                std::istringstream obj_text(j["meshes"][key].get<std::string>());
                if (!load_trimesh_obj(obj_text, key, mesh)) continue;
            }
            else
            {
                const fs::path mesh_path = fs::path(path).parent_path() / meta.value("mesh_file", "");
                if (!load_trimesh_obj(mesh_path.string(), mesh)) continue; // load_trimesh_obj already warned
            }
            std::cout << "[mesh] " << name << ": " << mesh.vertices.size() << " vertices, "
                      << mesh.triangles.size() << " triangles\n";
            anim.mesh_id = (int)world.trimeshes.size();
            world.trimeshes.push_back(std::move(mesh));
        }

        anim.radius      = meta.value("radius", 0.0);
        anim.half_length = meta.value("half_length", 0.0);
        const auto axis  = meta.value("axis_local", std::vector<Real>{0.0, 1.0, 0.0});
        anim.axis_local  = Vec3(axis[0], axis[1], axis[2]);

        Collider c;
        c.type     = anim.type;
        c.animated = true;
        c.anim_id  = (int)anims.size();
        c.mesh_id  = anim.mesh_id;

        name_to_anim_id[name] = (int)anims.size();
        anims.push_back(anim);
        out_colliders.push_back(c);
    }

    for (const auto& frame_entry : j.at("frames"))
    {
        for (auto& [name, pose] : frame_entry.at("colliders").items())
        {
            auto it = name_to_anim_id.find(name);
            if (it == name_to_anim_id.end()) continue; // no metadata entry for this name; ignore

            const auto pos = pose.at("position").get<std::vector<Real>>();
            const auto rot = pose.at("rotation").get<std::vector<Real>>(); // [x, y, z, w]

            ColliderFrame f;
            f.position = Vec3(pos[0], pos[1], pos[2]);
            f.rotation = Eigen::Quaternion<Real>(rot[3], rot[0], rot[1], rot[2]); // ctor is (w, x, y, z)

            anims[it->second].frames.push_back(f);
        }
    }

    return;
}

std::string animation_path(const std::string& filename)
{
    return (fs::path(COLLIDER_ANIM_DIR_DEFAULT) / filename).string();
}

std::vector<std::string> list_animation_files()
{
    std::vector<std::string> files;
    std::error_code          ec;
    for (const auto& entry : fs::directory_iterator(COLLIDER_ANIM_DIR_DEFAULT, ec))
        if (entry.is_regular_file() && entry.path().extension() == ".json")
            files.push_back(entry.path().filename().string());
    std::sort(files.begin(), files.end());
    return files;
}

AnimationPreview load_animation_preview(const std::string& filename)
{
    AnimationPreview preview;
    if (filename.empty()) return preview;

    load_collider_animation(animation_path(filename), preview.world);
    const std::vector<ColliderAnimation>& anims = preview.world.animations;
    for (Collider& c : preview.world.colliders)
        if (c.anim_id >= 0 && c.anim_id < (int)anims.size() && !anims[c.anim_id].frames.empty())
            apply_collider_at_step(c, anims[c.anim_id], 1, 0);

    for (const ColliderAnimation& anim : anims)
        if (anim.name == kWaistAttachmentColliderName && !anim.frames.empty())
        {
            preview.has_waist            = true;
            preview.waist_snaps_origin   = true;
            preview.waist_default_origin = collider_animation_pose_at(anim, 0).position;
            break;
        }

    // A skinned body is attached through its pelvis bone, and the cloth is placed by hand (no origin snap).
    if (!preview.world.skinned_meshes.empty())
    {
        preview.has_skinned = true;
        if (!preview.has_waist && find_bone(preview.world.skinned_meshes[0], kWaistAttachmentBoneName) >= 0)
            preview.has_waist = true;
    }
    return preview;
}
