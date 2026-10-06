import bpy
import json
import math
import os

from mathutils import Vector


# =========================================================
# Capsule collider logic
# =========================================================

def setup_capsule_collider(obj):
    if obj.type != 'EMPTY':
        print(f"'{obj.name}' is not an Empty. Skipping.")
        return False

    if obj.empty_display_type != 'CUBE':
        print(
            f"'{obj.name}' is an Empty, but its display type is "
            f"'{obj.empty_display_type}', not 'CUBE'. Skipping."
        )
        return False

    display_size = obj.empty_display_size

    local_dimensions = (
        abs(obj.scale.x) * display_size,
        abs(obj.scale.y) * display_size,
        abs(obj.scale.z) * display_size,
    )

    major_axis = max(
        range(3),
        key=lambda i: local_dimensions[i]
    )

    major_dimension = local_dimensions[major_axis]

    minor_dimensions = [
        local_dimensions[i]
        for i in range(3)
        if i != major_axis
    ]

    diameter = sum(minor_dimensions) / 2.0
    radius = diameter

    half_length = major_dimension - radius

    axis_local = [0.0, 0.0, 0.0]
    axis_local[major_axis] = 1.0

    obj["axis_local"] = axis_local
    obj["collider_type"] = "capsule"
    obj["half_length"] = half_length
    obj["radius"] = radius

    print(f"\nCapsule collider updated: {obj.name}")
    print(f"  Local dimensions : {local_dimensions}")
    print(f"  Main axis        : {axis_local}")
    print(f"  Collider type    : capsule")
    print(f"  Radius           : {radius:.6f}")
    print(f"  Half length      : {half_length:.6f}")

    return True


# =========================================================
# Sphere collider logic
# =========================================================

def setup_sphere_collider(obj):
    if obj.type != 'EMPTY':
        print(f"'{obj.name}' is not an Empty. Skipping.")
        return False

    if obj.empty_display_type != 'SPHERE':
        print(
            f"'{obj.name}' is an Empty, but its display type is "
            f"'{obj.empty_display_type}', not 'SPHERE'. Skipping."
        )
        return False

    display_size = obj.empty_display_size

    local_dimensions = (
        abs(obj.scale.x) * display_size,
        abs(obj.scale.y) * display_size,
        abs(obj.scale.z) * display_size,
    )

    # Sphere empties are expected to be uniformly scaled; average the three
    # axes so a slightly non-uniform scale still produces a sane radius.
    radius = sum(local_dimensions) / 3.0

    obj["axis_local"] = [0.0, 1.0, 0.0]
    obj["collider_type"] = "sphere"
    obj["half_length"] = 0.0
    obj["radius"] = radius

    print(f"\nSphere collider updated: {obj.name}")
    print(f"  Local dimensions : {local_dimensions}")
    print(f"  Collider type    : sphere")
    print(f"  Radius           : {radius:.6f}")

    return True


def update_collider(obj):
    # Cube Empty -> capsule, Sphere Empty -> sphere
    if obj.type == 'EMPTY' and obj.empty_display_type == 'CUBE':
        return setup_capsule_collider(obj)
    if obj.type == 'EMPTY' and obj.empty_display_type == 'SPHERE':
        return setup_sphere_collider(obj)
    return False


# =========================================================
# Triangulated capsule (procedural OBJ text)
# =========================================================

def capsule_obj_text(name, radius, half_length, axis, segments, cap_rings):
    """Closed, outward-wound triangle capsule in the Empty's local frame, as OBJ text.

    `half_length` is the cylinder's half-length (the segment between the two hemisphere
    centres), `axis` the local capsule axis. Rings are spaced so body quads come out
    roughly as wide as they are tall.
    """
    a = Vector(axis).normalized()
    helper = Vector((1, 0, 0)) if abs(a.x) < 0.9 else Vector((0, 1, 0))
    u = a.cross(helper).normalized()
    w = a.cross(u)  # (u, w, a) is right-handed

    H, R = half_length, radius
    n_body = max(1, round(2 * H / (2 * math.pi * R / segments))) if H > 1e-9 else 0

    # (axial offset, ring radius), bottom to top
    rings = []
    for i in range(1, cap_rings + 1):
        th = 0.5 * math.pi * i / cap_rings
        rings.append((-H - R * math.cos(th), R * math.sin(th)))
    for k in range(1, n_body):
        rings.append((-H + 2 * H * k / n_body, R))
    for i in range(cap_rings, 0, -1):
        if n_body == 0 and i == cap_rings:
            continue  # H == 0: the two equators coincide, keep one
        th = 0.5 * math.pi * i / cap_rings
        rings.append((H + R * math.cos(th), R * math.sin(th)))

    verts = [-a * (H + R)]
    for axial, r in rings:
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            verts.append(a * axial + r * (math.cos(phi) * u + math.sin(phi) * w))
    verts.append(a * (H + R))

    def ring_v(ring, j):
        return 1 + ring * segments + j % segments

    top = len(verts) - 1
    tris = []
    for j in range(segments):
        tris.append((0, ring_v(0, j + 1), ring_v(0, j)))
        for ring in range(len(rings) - 1):
            p, q = ring_v(ring, j), ring_v(ring, j + 1)
            s, t = ring_v(ring + 1, j + 1), ring_v(ring + 1, j)
            tris += [(p, q, s), (p, s, t)]
        tris.append((ring_v(len(rings) - 1, j), ring_v(len(rings) - 1, j + 1), top))

    lines = [f"o {name}"]
    lines += [f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}" for v in verts]
    lines += [f"f {a_ + 1} {b_ + 1} {c_ + 1}" for a_, b_, c_ in tris]
    return "\n".join(lines) + "\n"


# =========================================================
# Export collider animation
# =========================================================

def export_collider_animation():

    scene = bpy.context.scene

    collider_names = [
        "collider_hip",
        "collider_thigh_L",
        "collider_shin_L",
        "collider_thigh_R",
        "collider_shin_R",
        "collider_knee_R",
        "collider_knee_L",
    ]

    # -----------------------------------------------------
    # Gather collider metadata
    # -----------------------------------------------------

    colliders_meta = {}
    active_names = []  # collider_names minus missing/hidden entries

    for name in collider_names:

        obj = bpy.data.objects.get(name)

        if obj is None:
            print(
                f"Warning: collider '{name}' "
                f"not found in scene"
            )
            continue

        if obj.hide_get():
            print(
                f"Skipping hidden collider '{name}'"
            )
            continue

        # Refresh radius / half_length / axis_local from the Empty's current
        # shape so the export never uses stale custom properties.
        if not update_collider(obj):
            print(
                f"Warning: '{name}' not updated; "
                f"exporting its existing custom properties"
            )

        active_names.append(name)

        colliders_meta[name] = {
            "type": obj.get(
                "collider_type",
                "sphere"
            ),

            "radius": obj.get(
                "radius",
                0.1
            ),

            "half_length": obj.get(
                "half_length",
                0.2
            ),

            "axis_local": list(
                obj.get(
                    "axis_local",
                    [0, 1, 0]
                )
            ),
        }

    # -----------------------------------------------------
    # Triangulated mode: swap capsules for procedural meshes
    # -----------------------------------------------------

    triangulated = scene.collider_export_mode == 'TRIANGULATED'
    meshes = {}

    if triangulated:

        # Validate before touching the output file.
        for name, meta in colliders_meta.items():
            if meta["type"] != "capsule":
                raise RuntimeError(
                    f"Triangulated export supports capsules only; "
                    f"'{name}' is a {meta['type']}. Hide it or switch to Perfect."
                )

        for name, meta in colliders_meta.items():
            meshes[name] = capsule_obj_text(
                name,
                meta["radius"],
                meta["half_length"],
                meta["axis_local"],
                scene.collider_mesh_segments,
                scene.collider_mesh_cap_rings,
            )
            meta["type"] = "mesh"
            meta["mesh"] = name
            print(
                f"  {name}: {meshes[name].count(chr(10) + 'f ')} triangles"
            )

    # -----------------------------------------------------
    # Bake animation
    # -----------------------------------------------------

    frames_data = []

    # Remember current frame so we can restore it later
    original_frame = scene.frame_current

    for frame_idx in range(
        scene.frame_start,
        scene.frame_end + 1
    ):

        scene.frame_set(frame_idx)

        frame_record = {
            "frame": frame_idx,
            "colliders": {}
        }

        for name in active_names:

            obj = bpy.data.objects.get(name)

            if obj:

                loc, rot, scale = (
                    obj.matrix_world.decompose()
                )

                frame_record["colliders"][name] = {
                    "position": [
                        float(loc.x),
                        float(loc.y),
                        float(loc.z)
                    ],

                    "rotation": [
                        float(rot.x),
                        float(rot.y),
                        float(rot.z),
                        float(rot.w)
                    ],
                }

        frames_data.append(frame_record)

    # Restore original frame
    scene.frame_set(original_frame)

    # -----------------------------------------------------
    # Resolve output path
    # -----------------------------------------------------

    file_name = (
        "collider_animation_tri.json"
        if triangulated
        else "collider_animation.json"
    )

    if bpy.data.filepath:

        output_path = bpy.path.abspath(
            "//" + file_name
        )

    else:

        # .blend has not been saved yet
        output_path = (
            r"C:\Users\Work\borsa_verona"
            r"\DiffXPBD\diffpd\animation"
            "\\" + file_name
        )

    # -----------------------------------------------------
    # Delete existing file
    # -----------------------------------------------------

    if os.path.exists(output_path):

        os.remove(output_path)

        print(
            f"Removed existing {output_path}"
        )

    # -----------------------------------------------------
    # Build output
    # -----------------------------------------------------

    output = {}

    if meshes:
        output["meshes"] = meshes  # OBJ text per collider, referenced by "mesh"

    output |= {
        "colliders_metadata": colliders_meta,
        "fps": scene.render.fps,
        "n_frames": len(frames_data),
        "frames": frames_data,
    }

    # -----------------------------------------------------
    # Write JSON
    # -----------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print(
        f"Exported {len(frames_data)} "
        f"frames to {output_path}"
    )

    print(
        f"  FPS: {scene.render.fps}"
    )

    print(
        f"  Colliders: "
        f"{list(colliders_meta.keys())}"
    )

    return output_path


# =========================================================
# Export operator
# =========================================================

class OBJECT_OT_export_collider_animation(
    bpy.types.Operator
):

    bl_idname = "object.export_collider_animation"
    bl_label = "Export Collider Animation"

    bl_description = (
        "Update collider properties from the Empties' shapes, then "
        "bake the animation and export it to collider_animation.json"
    )

    bl_options = {'REGISTER'}

    def execute(self, context):

        try:

            output_path = (
                export_collider_animation()
            )

            self.report(
                {'INFO'},
                f"Exported collider animation to "
                f"{output_path}"
            )

            return {'FINISHED'}

        except Exception as e:

            print(
                "\nCollider export failed:"
            )

            print(e)

            self.report(
                {'ERROR'},
                f"Export failed: {e}"
            )

            return {'CANCELLED'}


# =========================================================
# N-panel UI
# =========================================================

class VIEW3D_PT_capsule_collider(
    bpy.types.Panel
):

    bl_label = "Collider"
    bl_idname = "VIEW3D_PT_capsule_collider"

    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Collider'

    def draw(self, context):

        layout = self.layout
        obj = context.active_object

        # -------------------------------------------------
        # Export button (also refreshes collider properties)
        # -------------------------------------------------

        scene = context.scene

        layout.prop(scene, "collider_export_mode")

        if scene.collider_export_mode == 'TRIANGULATED':
            layout.prop(scene, "collider_mesh_segments")
            layout.prop(scene, "collider_mesh_cap_rings")

        row = layout.row()
        row.scale_y = 1.5

        row.operator(
            "object.export_collider_animation",
            icon='EXPORT'
        )

        # -------------------------------------------------
        # Object information
        # -------------------------------------------------

        if obj is not None:

            box = layout.box()

            box.label(
                text=f"Selected: {obj.name}"
            )

            if "collider_type" in obj:

                box.label(
                    text=(
                        f"Collider: "
                        f"{obj.get('collider_type', '-')}"
                    )
                )

                box.label(
                    text=(
                        f"Radius: "
                        f"{obj.get('radius', 0.0):.4f}"
                    )
                )

                if obj.get("collider_type") == "capsule":

                    box.label(
                        text=(
                            f"Half Length: "
                            f"{obj.get('half_length', 0.0):.4f}"
                        )
                    )


# =========================================================
# Registration
# =========================================================

classes = (

    OBJECT_OT_export_collider_animation,

    VIEW3D_PT_capsule_collider,

)


def register():

    bpy.types.Scene.collider_export_mode = bpy.props.EnumProperty(
        name="Capsules",
        description=(
            "Export capsule colliders as analytic capsules, or as "
            "procedural triangle meshes (written to *_tri.json)"
        ),
        items=[
            ('PERFECT', "Perfect", "Analytic capsules"),
            ('TRIANGULATED', "Triangulated", "Triangle-mesh capsules"),
        ],
        default='PERFECT',
    )
    bpy.types.Scene.collider_mesh_segments = bpy.props.IntProperty(
        name="Segments",
        description="Vertices around the capsule axis",
        default=12, min=3, max=128,
    )
    bpy.types.Scene.collider_mesh_cap_rings = bpy.props.IntProperty(
        name="Cap Rings",
        description="Latitude rings per hemisphere",
        default=4, min=1, max=64,
    )

    for cls in classes:

        bpy.utils.register_class(cls)


def unregister():

    for cls in reversed(classes):

        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.collider_mesh_cap_rings
    del bpy.types.Scene.collider_mesh_segments
    del bpy.types.Scene.collider_export_mode


if __name__ == "__main__":
    register()