"""Bake the skinned body (armature + LBS mesh) of the open .blend into skinned_body.json.

In Blender: select the skinned mesh (or its armature), N-panel tab "Skinned Body" -> Export Skinned Body.
The panel appears once this file is run (Text editor -> Open -> Run Script), or on every file load through
a Register text block (see the stub in the .blend). Headless (the file's only skinned mesh):
    blender -b SMPL_colliders.blend --python export_skinned_body.py

Everything is written in Blender's world frame (Z-up, metres); the C++ loader applies
blender_to_diffpd(), like for the collider tracks. The animation is resampled to FPS_OUT, so the scene
frame rate (e.g. 30) does not matter: Blender is evaluated at whole scene frames only, and the frames in
between are slerp/lerp of the neighbouring skinning matrices (evaluating Blender at sub-frames
interpolates quaternion components separately and glitches the pose where keyframe signs differ).

Skinning matrices are world-space: x_world(t) = sum_b w_ib * S_b(t) * x_world_rest_i, with
S_b = M_arm * pose_bone.matrix * bone.matrix_local^-1 * M_arm^-1 (3x4, row-major).

Pose correctives (shape keys named "Pose*") are not linear blend skinning, so they are muted for
the whole export; the shape and expression keys stay on and are baked into the rest mesh.
"""

import json
import math
import os

import bpy
import numpy as np
from mathutils import Matrix

FPS_OUT = 60
MAX_INFLUENCES = 4
# What to do with weights on vertex groups whose bone was deleted:
#   "merge_to_ancestor": add them to the nearest surviving ancestor (SMPL-X hierarchy), like
#                        Blender's "merge bones"; physically sensible.
#   "renormalize":       drop them and renormalize over the surviving bones; this is what
#                        Blender's own Armature modifier does, so the exported mesh then
#                        matches Blender's viewport exactly.
ORPHAN_MODE = "merge_to_ancestor"
MUTE_KEY_PREFIXES = ("Pose",)
N_REFERENCE_FRAMES = 5
OUT_NAME = "skinned_body.json"


def smplx_parents():
    p = {
        "pelvis": None,
        "left_hip": "pelvis", "right_hip": "pelvis", "spine1": "pelvis",
        "left_knee": "left_hip", "right_knee": "right_hip", "spine2": "spine1",
        "left_ankle": "left_knee", "right_ankle": "right_knee", "spine3": "spine2",
        "left_foot": "left_ankle", "right_foot": "right_ankle", "neck": "spine3",
        "left_collar": "spine3", "right_collar": "spine3", "head": "neck",
        "left_shoulder": "left_collar", "right_shoulder": "right_collar",
        "left_elbow": "left_shoulder", "right_elbow": "right_shoulder",
        "left_wrist": "left_elbow", "right_wrist": "right_elbow",
        "jaw": "head", "left_eye_smplhf": "head", "right_eye_smplhf": "head",
    }
    for side in ("left", "right"):
        for finger in ("index", "middle", "pinky", "ring", "thumb"):
            p[f"{side}_{finger}1"] = f"{side}_wrist"
            p[f"{side}_{finger}2"] = f"{side}_{finger}1"
            p[f"{side}_{finger}3"] = f"{side}_{finger}2"
    return p


SMPLX_PARENT = smplx_parents()


def skin_armature(ob):
    """The armature object deforming mesh `ob`, or None."""
    for m in ob.modifiers:
        if m.type == 'ARMATURE' and m.object is not None:
            return m.object
    return None


def resolve_target(obj):
    """(armature, mesh) for the selected skinned mesh or armature; with no selection, the only skinned mesh in the file."""
    if obj is not None and obj.type == 'MESH':
        arm = skin_armature(obj)
        if arm is None:
            raise RuntimeError(f"'{obj.name}' has no Armature modifier")
        return arm, obj
    if obj is not None and obj.type == 'ARMATURE':
        arm = obj
    elif obj is None:
        arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
        if len(arms) != 1:
            raise RuntimeError(f"nothing selected and {len(arms)} armatures in the file; select the skinned mesh")
        arm = arms[0]
    else:
        raise RuntimeError(f"'{obj.name}' is a {obj.type}; select a skinned mesh or its armature")
    meshes = [o for o in bpy.data.objects if o.type == 'MESH' and skin_armature(o) == arm]
    if len(meshes) != 1:
        raise RuntimeError(f"expected one mesh skinned by '{arm.name}', found {[o.name for o in meshes]}; select the mesh")
    return arm, meshes[0]


def resolve_bone(name, bone_index):
    """The bone `name` itself if it exists, else its nearest surviving ancestor, else None."""
    while name is not None and name not in bone_index:
        name = SMPLX_PARENT.get(name)
    return name


def build_weights(ob, bone_names):
    """Dense (V, B) weights: the Blender-equivalent ones and the ones for ORPHAN_MODE, both row-normalized."""
    bone_index = {n: i for i, n in enumerate(bone_names)}
    gname = {g.index: g.name for g in ob.vertex_groups}
    V, B = len(ob.data.vertices), len(bone_names)
    w_blender = np.zeros((V, B))
    w_merged = np.zeros((V, B))
    orphan_total = {}
    lost = {}
    for v in ob.data.vertices:
        for g in v.groups:
            name, w = gname[g.group], g.weight
            if name in bone_index:
                w_blender[v.index, bone_index[name]] += w
            else:
                orphan_total[name] = orphan_total.get(name, 0.0) + w
            target = resolve_bone(name, bone_index)
            if target is None:
                lost[name] = lost.get(name, 0.0) + w
            else:
                w_merged[v.index, bone_index[target]] += w
    for w in (w_blender, w_merged):
        s = w.sum(axis=1, keepdims=True)
        if (s <= 0).any():
            raise RuntimeError("a vertex has no weight on any surviving bone")
        w /= s
    merged = {n: resolve_bone(n, bone_index) for n in orphan_total}
    return w_blender, w_merged, merged, lost


def cap_influences(w_dense, k):
    """Keep the k largest weights per vertex (renormalized); returns (bone_index (V,k), weight (V,k), dense, max_dropped)."""
    order =np.argsort(-w_dense, axis=1)[:, :k]
    top = np.take_along_axis(w_dense, order, axis=1)
    dropped = 1.0 - top.sum(axis=1)
    top = top / top.sum(axis=1, keepdims=True)
    dense = np.zeros_like(w_dense)
    np.put_along_axis(dense, order, top, axis=1)
    return order, top, dense, float(dropped.max())


def evaluated_vertices(ob):
    """World-space vertices of the object as Blender currently deforms it."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    ev.to_mesh_clear()
    m = np.array(ob.matrix_world)
    return co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]


def lbs(x_rest, s, w):
    """x_rest (V,3), s (B,3,4), w (V,B) -> (V,3)."""
    y = np.einsum('bij,vj->bvi', s[:, :, :3], x_rest) + s[:, None, :, 3]
    return np.einsum('vb,bvi->vi', w, y)


def check_closed(tris, n_verts):
    directed = {}
    for a, b, c in tris:
        for e in ((a, b), (b, c), (c, a)):
            directed[e] = directed.get(e, 0) + 1
    dup = sum(1 for n in directed.values() if n > 1)
    unmatched = sum(1 for (a, b) in directed if (b, a) not in directed)
    edges = len({tuple(sorted(e)) for e in directed})
    return {"duplicate_directed_edges": dup, "unmatched_edges": unmatched,
            "euler": n_verts - edges + len(tris)}


def signed_volume(x, tris):
    a, b, c = x[tris[:, 0]], x[tris[:, 1]], x[tris[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)


def blend_rigid(s0, s1, alpha):
    """Interpolates two (B, 3, 4) sets of rigid skinning matrices: slerp of the rotations (shortest arc), lerp of the translations."""
    out = np.empty_like(s0)
    for i in range(len(s0)):
        q0 = Matrix(s0[i, :, :3].tolist()).to_quaternion()
        q1 = Matrix(s1[i, :, :3].tolist()).to_quaternion()
        if q0.dot(q1) < 0.0:
            q1.negate()
        out[i, :, :3] = np.array(q0.slerp(q1, alpha).to_matrix())
        out[i, :, 3] = (1.0 - alpha) * s0[i, :, 3] + alpha * s1[i, :, 3]
    return out


def dumps_rows(obj, level=0):
    """JSON with one key per line and one row (vertex, triangle, frame, ...) per line, rows kept inline."""
    pad, end = "  " * (level + 1), "  " * level
    if isinstance(obj, dict) and obj:
        body = [f"{pad}{json.dumps(k)}: {dumps_rows(v, level + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(body) + "\n" + end + "}"
    if isinstance(obj, list) and any(isinstance(x, (list, dict)) for x in obj):
        return "[\n" + ",\n".join(pad + dumps_rows(x, level + 1) for x in obj) + "\n" + end + "]"
    return json.dumps(obj, separators=(",", ":"))


def export_skinned_body(arm, ob, orphan_mode=ORPHAN_MODE, out_name=OUT_NAME):
    scene = bpy.context.scene
    bones = list(arm.data.bones)
    names = [b.name for b in bones]
    name_to_i = {n: i for i, n in enumerate(names)}

    saved_frame = scene.frame_current
    saved_pose_position = arm.data.pose_position
    key_blocks = ob.data.shape_keys.key_blocks if ob.data.shape_keys else []
    saved_mute = {kb.name: kb.mute for kb in key_blocks}

    try:
        # ---- weights ----------------------------------------------------------------
        w_blender, w_merged, merged_map, lost = build_weights(ob, names)
        w_pick = w_merged if orphan_mode == "merge_to_ancestor" else w_blender
        bone_idx, bone_w, w_export, dropped = cap_influences(w_pick, MAX_INFLUENCES)

        # ---- rest mesh (shape + expression keys on, pose correctives off, armature in rest) ----
        for kb in key_blocks:
            if kb.name.startswith(MUTE_KEY_PREFIXES):
                kb.mute = True
        arm.data.pose_position = 'REST'
        bpy.context.view_layer.update()
        x_rest = evaluated_vertices(ob)
        arm.data.pose_position = saved_pose_position
        bpy.context.view_layer.update()

        me = ob.data
        me.calc_loop_triangles()
        tris = np.array([t.vertices[:] for t in me.loop_triangles], dtype=int)
        m_world = np.array(ob.matrix_world)
        if np.linalg.det(m_world[:3, :3]) < 0:
            tris = tris[:, ::-1]
        closed = check_closed(tris.tolist(), len(x_rest))
        volume = signed_volume(x_rest, tris)
        if volume < 0:
            tris = tris[:, ::-1]
            volume = -volume

        # ---- animation: sample at FPS_OUT -------------------------------------------
        m_arm = np.array(arm.matrix_world)
        m_arm_inv = np.linalg.inv(m_arm)
        rest_local_inv = [np.linalg.inv(np.array(b.matrix_local)) for b in bones]
        rest_world = np.array([(m_arm @ np.array(b.matrix_local))[:3, :] for b in bones])

        fps_scene = scene.render.fps / scene.render.fps_base
        f0, f1 = scene.frame_start, scene.frame_end
        n_out = int(round((f1 - f0) / fps_scene * FPS_OUT)) + 1

        # Blender is only evaluated at whole scene frames (exact keyframe values). Evaluating at
        # sub-frames would interpolate each quaternion component on its own, which flips the pose for
        # a frame whenever two keyframes have opposite quaternion signs; in-between output frames are
        # slerped here instead.
        cache = {}  # whole scene frame -> (skinning matrices (B, 3, 4), Blender-deformed world vertices)

        def key_frame(fi):
            if fi not in cache:
                scene.frame_set(fi)
                s = np.zeros((len(bones), 3, 4))
                for i, b in enumerate(bones):
                    a = m_arm @ np.array(arm.pose.bones[b.name].matrix) @ rest_local_inv[i] @ m_arm_inv
                    s[i] = a[:3, :]
                cache[fi] = (s, evaluated_vertices(ob))
            return cache[fi]

        skin = np.zeros((n_out, len(bones), 3, 4))
        exact = []          # output frames that land on a whole scene frame (Blender can check these)
        err_blender = 0.0   # LBS from the exported matrices + Blender-equivalent weights vs Blender
        err_export = 0.0    # LBS from the exported matrices + exported weights vs Blender
        worst_frame = 0
        for k in range(n_out):
            f = min(f0 + k / FPS_OUT * fps_scene, float(f1))
            fi = int(math.floor(f + 1e-9))
            alpha = f - fi
            s0, x_blender = key_frame(fi)
            if alpha < 1e-6:
                skin[k] = s0
                exact.append(k)
                e1 = np.abs(lbs(x_rest, s0, w_blender) - x_blender).max()
                e2 = np.abs(lbs(x_rest, s0, w_export) - x_blender).max()
                if e1 > err_blender:
                    err_blender, worst_frame = e1, k
                err_export = max(err_export, e2)
            else:
                skin[k] = blend_rigid(s0, key_frame(min(fi + 1, f1))[0], alpha)

        ref_frames = sorted({exact[i] for i in np.linspace(0, len(exact) - 1, N_REFERENCE_FRAMES).round().astype(int)})
        reference = {k: cache[int(round(f0 + k / FPS_OUT * fps_scene))][1] for k in ref_frames}

        # Largest vertex jump between consecutive output frames: a pose glitch shows up here as ~metres.
        prev = lbs(x_rest, skin[0], w_export)
        max_jump, max_jump_frame = 0.0, 0
        for k in range(1, n_out):
            cur = lbs(x_rest, skin[k], w_export)
            jump = float(np.linalg.norm(cur - prev, axis=1).max())
            if jump > max_jump:
                max_jump, max_jump_frame = jump, k
            prev = cur

        # how much the muted pose correctives would have moved the mesh (mid frame, info only)
        f_mid = (f0 + f1) // 2
        s_mid, _ = key_frame(f_mid)
        scene.frame_set(f_mid)
        for kb in key_blocks:
            if kb.name.startswith(MUTE_KEY_PREFIXES):
                kb.mute = saved_mute[kb.name]
        bpy.context.view_layer.update()
        corrective_shift = float(np.abs(evaluated_vertices(ob) - lbs(x_rest, s_mid, w_blender)).max())

    finally:
        for kb in key_blocks:
            if kb.name in saved_mute:
                kb.mute = saved_mute[kb.name]
        arm.data.pose_position = saved_pose_position
        scene.frame_set(saved_frame)

    # ---- write -----------------------------------------------------------------------
    out = {
        "format": "diffpd_skinned_body",
        "version": 1,
        "coordinates": "blender_world_z_up",
        "units": "m",
        "fps": FPS_OUT,
        "n_frames": n_out,
        "source": {
            "blend": os.path.basename(bpy.data.filepath),
            "armature": arm.name,
            "mesh": ob.name,
            "scene_fps": fps_scene,
            "frame_range": [f0, f1],
            "orphan_mode": orphan_mode,
            "orphan_groups": merged_map,
            "muted_key_prefixes": list(MUTE_KEY_PREFIXES),
        },
        "mesh": {
            "vertices": np.round(x_rest, 7).tolist(),
            "triangles": tris.tolist(),
        },
        "skin": {
            "bones": names,
            "parents": [name_to_i[b.parent.name] if b.parent else -1 for b in bones],
            "rest_world": np.round(rest_world.reshape(len(bones), 12), 9).tolist(),
            "max_influences": MAX_INFLUENCES,
            "bone_index": bone_idx.tolist(),
            "weight": np.round(bone_w, 7).tolist(),
        },
        "frames": np.round(skin.reshape(n_out, -1), 9).tolist(),
        "reference": {
            "note": "Blender-evaluated world vertices at output frames that land on whole scene frames (pose correctives muted, Armature modifier weights)",
            "frames": sorted(reference),
            "vertices": [np.round(reference[k], 7).tolist() for k in sorted(reference)],
        },
    }
    out_path = bpy.path.abspath("//" + out_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(dumps_rows(out) + "\n")

    print(f"Exported {n_out} frames @ {FPS_OUT} fps, {len(x_rest)} vertices, {len(tris)} triangles, "
          f"{len(bones)} bones -> {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")
    print(f"  closed mesh: {closed}, signed volume {volume:.5f} m^3")
    print(f"  orphan groups ({orphan_mode}): {len(merged_map)} merged, weight lost with no ancestor: {lost}")
    print(f"  influences capped to {MAX_INFLUENCES}: max weight dropped on a vertex {dropped:.4f}")
    print(f"  LBS vs Blender at {len(exact)} whole-scene-frame outputs, Blender-equivalent weights: max err {err_blender:.2e} m (worst frame {worst_frame})")
    print(f"  LBS vs Blender at those frames, exported weights:                       max err {err_export:.2e} m")
    print(f"  largest vertex jump between consecutive output frames: {max_jump:.4f} m (frame {max_jump_frame})")
    print(f"  pose correctives would move the mesh by up to {corrective_shift:.2e} m (muted in the export)")
    return out_path, {
        "frames": n_out, "vertices": len(x_rest), "bones": len(bones),
        "err_blender": err_blender, "err_export": err_export, "max_jump": max_jump,
    }


# =========================================================
# Export operator + N-panel (tab "Skinned Body")
# =========================================================

_KEY = "skinned_body_export_module"


class OBJECT_OT_export_skinned_body(bpy.types.Operator):
    bl_idname = "object.export_skinned_body"
    bl_label = "Export Skinned Body"
    bl_description = ("Bake the selected skinned mesh (or its armature's mesh) and its animation "
                      "to a JSON file next to the .blend")
    bl_options = {'REGISTER'}

    def execute(self, context):
        scene = context.scene
        try:
            arm, ob = resolve_target(context.active_object)
            path, info = export_skinned_body(arm, ob, scene.skinned_export_orphan_mode,
                                             scene.skinned_export_filename)
        except Exception as e:
            print("\nSkinned body export failed:", e)
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Exported {info['frames']} frames to {path} "
                              f"(LBS check {info['err_blender']:.1e} m)")
        return {'FINISHED'}


class VIEW3D_PT_skinned_body(bpy.types.Panel):
    bl_label = "Skinned Body"
    bl_idname = "VIEW3D_PT_skinned_body"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Skinned Body'

    def draw(self, context):
        layout, scene = self.layout, context.scene
        layout.prop(scene, "skinned_export_filename")
        layout.prop(scene, "skinned_export_orphan_mode")
        row = layout.row()
        row.scale_y = 1.5
        row.operator("object.export_skinned_body", icon='EXPORT')
        box = layout.box()
        try:
            arm, ob = resolve_target(context.active_object)
            box.label(text=f"Mesh: {ob.name}")
            box.label(text=f"Armature: {arm.name} ({len(arm.data.bones)} bones)")
            box.label(text=f"{scene.frame_start}-{scene.frame_end} @ {scene.render.fps / scene.render.fps_base:g} fps -> {FPS_OUT} fps")
        except Exception as e:
            box.label(text=str(e), icon='ERROR')


classes = (OBJECT_OT_export_skinned_body, VIEW3D_PT_skinned_body)


def register():
    old = bpy.app.driver_namespace.pop(_KEY, None)  # drop a previous run's classes (Run Script re-run)
    if old is not None:
        try:
            old["unregister"]()
        except Exception as e:
            print("skinned body export: unregister failed:", e)
    bpy.types.Scene.skinned_export_filename = bpy.props.StringProperty(
        name="File", description="Output JSON, written next to the .blend", default=OUT_NAME)
    bpy.types.Scene.skinned_export_orphan_mode = bpy.props.EnumProperty(
        name="Deleted bones",
        description="Where the weights of vertex groups whose bone was deleted go",
        items=[
            ('merge_to_ancestor', "Merge to parent", "Add to the nearest surviving ancestor bone"),
            ('renormalize', "Renormalize", "Drop and renormalize (what Blender's Armature modifier does)"),
        ],
        default=ORPHAN_MODE)
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.app.driver_namespace[_KEY] = globals()


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.skinned_export_orphan_mode
    del bpy.types.Scene.skinned_export_filename


if __name__ == "__main__":
    if bpy.app.background:
        export_skinned_body(*resolve_target(None))
    else:
        register()
