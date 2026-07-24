#include "diffpd_viewer.h"

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

// raygui's implementation triggers harmless warnings (unused params, a hidden local, an unused
// static helper) under this project's strict warning flags; silence them locally rather than
// project-wide since the noise is entirely inside the vendored header.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100 4457 4505)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#include "style_dark.h"

#if defined(_MSC_VER)
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif

#include <cmath>
#include <algorithm>

namespace
{

// Blender-style orbit camera (3-button-mouse-emulation scheme, trackpad friendly):
// left-drag orbits around `target`, Shift+left-drag or right-drag pans, scroll wheel zooms.
struct OrbitCamera
{
    Vector3 target   = { 0.0f, -1.0f, 0.0f };
    float   distance = 4.0f;
    float   yaw      = -45.0f * DEG2RAD;
    float   pitch    =  25.0f * DEG2RAD;
};

void update_orbit_camera(OrbitCamera& orbit, Camera3D& camera)
{
    constexpr float rotate_speed = 0.01f;
    constexpr float pan_speed    = 0.0015f;
    constexpr float zoom_speed   = 0.1f;
    constexpr float min_distance = 0.1f;
    constexpr float max_pitch    = 89.0f * DEG2RAD;

    const Vector2 mouse_delta = GetMouseDelta();

    const bool shift_held = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
    const bool pan_active   = (IsMouseButtonDown(MOUSE_BUTTON_LEFT) && shift_held)
                            || IsMouseButtonDown(MOUSE_BUTTON_RIGHT);
    const bool orbit_active = IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !shift_held;

    if (pan_active)
    {
        const Vector3 forward = Vector3Normalize(Vector3Subtract(orbit.target, camera.position));
        const Vector3 right   = Vector3Normalize(Vector3CrossProduct(forward, camera.up));
        const Vector3 up      = Vector3CrossProduct(right, forward);
        const float scale     = pan_speed * orbit.distance;
        orbit.target = Vector3Add(orbit.target, Vector3Scale(right, -mouse_delta.x * scale));
        orbit.target = Vector3Add(orbit.target, Vector3Scale(up,     mouse_delta.y * scale));
    }
    else if (orbit_active)
    {
        orbit.yaw   -= mouse_delta.x * rotate_speed;
        orbit.pitch += mouse_delta.y * rotate_speed;
        orbit.pitch  = std::clamp(orbit.pitch, -max_pitch, max_pitch);
    }

    orbit.distance -= GetMouseWheelMove() * zoom_speed * orbit.distance;
    orbit.distance  = std::max(orbit.distance, min_distance);

    const Vector3 offset = {
        orbit.distance * cosf(orbit.pitch) * cosf(orbit.yaw),
        orbit.distance * sinf(orbit.pitch),
        orbit.distance * cosf(orbit.pitch) * sinf(orbit.yaw)
    };

    camera.target   = orbit.target;
    camera.position = Vector3Add(orbit.target, offset);
    camera.up       = { 0.0f, 1.0f, 0.0f };
}

Vec3 vertex_position(const SimMesh& mesh, const PointsX& frame, Index vi)
{
    const SimMesh::Vertex& vert = mesh.vertices[vi];
    if (vert.dof >= 0)
        return Vec3(frame(vert.dof, 0), frame(vert.dof, 1), frame(vert.dof, 2));
    return mesh.pinned_rest[-(vert.dof + 1)];
}

Vector3 to_raylib(const Vec3& v)
{
    return { (float)v.x(), (float)v.y(), (float)v.z() };
}

Vec3 from_raylib(const Vector3& v)
{
    return Vec3((Real)v.x, (Real)v.y, (Real)v.z);
}

// Per-vertex Lambertian shading for a constant directional light (raylib's DrawSphereEx already
// emits real per-vertex normals; this shader just lights them). Named attributes/uniforms match
// raylib's defaults (vertexNormal, matNormal, matModel) so rlgl wires them up automatically.
constexpr const char* kSphereVS = R"(
#version 330
in vec3 vertexPosition;
in vec3 vertexNormal;
in vec4 vertexColor;
uniform mat4 mvp;
uniform mat4 matNormal;
out vec3 fragNormal;
out vec4 fragColor;
void main()
{
    fragNormal = normalize((matNormal * vec4(vertexNormal, 0.0)).xyz);
    fragColor = vertexColor;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
)";

// gl_FrontFacing flips the normal for back-facing triangles, so a single-layer surface (the
// subdivided cloth mesh, unlike a closed sphere) is lit correctly from either side when backface
// culling is disabled for it — folded cloth showing its underside doesn't render black.
constexpr const char* kSphereFS = R"(
#version 330
in vec3 fragNormal;
in vec4 fragColor;
uniform vec3 lightDir;
out vec4 finalColor;
void main()
{
    vec3 normal = normalize(fragNormal);
    if (!gl_FrontFacing) normal = -normal;
    float diffuse = max(dot(normal, lightDir), 0.0);
    float intensity = 0.45 + 0.55 * diffuse;
    finalColor = vec4(fragColor.rgb * intensity, fragColor.a);
}
)";

void draw_tape_edges(const SimMesh& mesh, const PointsX& frame, Color color)
{
    for (const auto& e : mesh.edges)
    {
        const Vector3 a = to_raylib(vertex_position(mesh, frame, e.first));
        const Vector3 b = to_raylib(vertex_position(mesh, frame, e.second));
        //DrawLine3D(a, b, color);
        DrawCylinderEx(a, b, 0.005f, 0.005f, 4, color);
    }
}

// ----------------------------------------------------------------------------------------------
// Smooth-shaded cloth surface: subdivides each quad of the sim grid and gives every subdivided
// vertex a normal, so the cloth renders as a continuous curved sheet (like DrawSphereEx) instead
// of flat facets. Two passes:
//   1. Coarse per-vertex normals — average the analytic face normal of every quad touching a
//      grid vertex (like ordinary smooth-shaded quad-mesh normals).
//   2. Per-cell subdivision — bilinearly interpolate both position and normal across each coarse
//      quad (renormalizing the normal), the same idea as Phong tessellation. This reproduces the
//      simulated corners exactly and fills in a smoothly curved surface between them without any
//      extra physics.
// Rendered as a non-indexed triangle soup (mesh.indices left null) rather than an indexed mesh:
// raylib's Mesh::indices is `unsigned short`, which overflows past 65536 unique vertices — a
// plausible case at this app's larger cloth-size settings (e.g. 100x100 with 4x subdivision).
// Duplicating shared-edge vertices costs some memory/CPU but never silently wraps around.

Vec3 bilerp(const Vec3& p00, const Vec3& p10, const Vec3& p11, const Vec3& p01, Real u, Real v)
{
    return (1.0 - u) * (1.0 - v) * p00 + u * (1.0 - v) * p10 + u * v * p11 + (1.0 - u) * v * p01;
}

// GPU-side state for one cloth's smooth surface (reference or live). Vertex/normal CPU buffers
// double as the staging area written fresh every frame and pushed via UpdateMeshBuffer; the color
// buffer is constant per (width,height,subdiv) and only (re)filled when those change.
struct SurfaceMesh
{
    Mesh                       mesh{};
    std::vector<float>         vertex_buf;
    std::vector<float>         normal_buf;
    std::vector<unsigned char> color_buf;
    Index                      built_width  = -1;
    Index                      built_height = -1;
    int                        built_subdiv = -1;
    bool                       uploaded     = false;
};

// (Re)allocates GPU buffers sized for `width`x`height` at subdivision `subdiv`, only when those
// differ from what's already built (e.g. first use, or the user changed cloth dimensions and hit
// "Run" again). `color` is baked into every vertex here since it never changes frame to frame.
void ensure_surface_mesh(SurfaceMesh& sm, Index width, Index height, int subdiv, Color color)
{
    if (sm.built_width == width && sm.built_height == height && sm.built_subdiv == subdiv)
        return;

    if (sm.uploaded)
    {
        UnloadMesh(sm.mesh);
        sm.mesh = Mesh{};
        sm.uploaded = false;
    }

    const Index cells_i = width - 1, cells_j = height - 1;
    const Index vertex_count = cells_i * cells_j * (Index)subdiv * (Index)subdiv * 6;

    sm.vertex_buf.assign((size_t)vertex_count * 3, 0.0f);
    sm.normal_buf.assign((size_t)vertex_count * 3, 0.0f);
    sm.color_buf.assign((size_t)vertex_count * 4, 0);
    for (Index i = 0; i < vertex_count; ++i)
    {
        sm.color_buf[i * 4 + 0] = color.r;
        sm.color_buf[i * 4 + 1] = color.g;
        sm.color_buf[i * 4 + 2] = color.b;
        sm.color_buf[i * 4 + 3] = color.a;
    }

    sm.mesh.vertexCount   = (int)vertex_count;
    sm.mesh.triangleCount = (int)(vertex_count / 3);
    sm.mesh.vertices      = sm.vertex_buf.data();
    sm.mesh.normals       = sm.normal_buf.data();
    sm.mesh.colors        = sm.color_buf.data();
    sm.mesh.indices       = nullptr;

    UploadMesh(&sm.mesh, true); // dynamic = true: vertices/normals are rewritten every frame
    sm.uploaded = true;

    sm.built_width  = width;
    sm.built_height = height;
    sm.built_subdiv = subdiv;
}

// Regenerates sm.vertex_buf/normal_buf from the current frame's positions and uploads them.
void draw_tape_surface(const SimMesh& mesh, const PointsX& frame, Color color, int subdiv,
                       SurfaceMesh& sm, Material& material)
{
    const Index W = mesh.width, H = mesh.height;
    if (W < 2 || H < 2) return; // no quads to build a surface from

    ensure_surface_mesh(sm, W, H, subdiv, color);

    auto grid_index = [H](Index i, Index j) { return i * H + j; };
    auto pos_at      = [&](Index i, Index j) { return vertex_position(mesh, frame, grid_index(i, j)); };

    // Pass 1: coarse per-vertex normals, averaged from every adjacent quad's face normal.
    // Winding (p11-p00) x (p10-p00) matches the triangle winding used in pass 2 below.
    std::vector<Vec3> vert_normal(W * H, Vec3::Zero());
    for (Index qi = 0; qi < W - 1; ++qi)
    {
        for (Index qj = 0; qj < H - 1; ++qj)
        {
            const Vec3 p00 = pos_at(qi, qj), p10 = pos_at(qi + 1, qj);
            const Vec3 p11 = pos_at(qi + 1, qj + 1), p01 = pos_at(qi, qj + 1);
            const Vec3 n = (p11 - p00).cross(p10 - p00).normalized();
            vert_normal[grid_index(qi, qj)]         += n;
            vert_normal[grid_index(qi + 1, qj)]     += n;
            vert_normal[grid_index(qi + 1, qj + 1)] += n;
            vert_normal[grid_index(qi, qj + 1)]     += n;
        }
    }
    for (Vec3& n : vert_normal)
        if (n.squaredNorm() > 1e-20) n.normalize();

    // Pass 2: subdivide each coarse quad, writing directly into the mesh's staging buffers.
    Index out = 0;
    const Real inv_s = 1.0 / (Real)subdiv;
    auto emit = [&](const Vec3& p, const Vec3& n)
    {
        sm.vertex_buf[out * 3 + 0] = (float)p.x();
        sm.vertex_buf[out * 3 + 1] = (float)p.y();
        sm.vertex_buf[out * 3 + 2] = (float)p.z();
        sm.normal_buf[out * 3 + 0] = (float)n.x();
        sm.normal_buf[out * 3 + 1] = (float)n.y();
        sm.normal_buf[out * 3 + 2] = (float)n.z();
        ++out;
    };

    for (Index qi = 0; qi < W - 1; ++qi)
    {
        for (Index qj = 0; qj < H - 1; ++qj)
        {
            const Vec3 p00 = pos_at(qi, qj), p10 = pos_at(qi + 1, qj);
            const Vec3 p11 = pos_at(qi + 1, qj + 1), p01 = pos_at(qi, qj + 1);
            const Vec3 n00 = vert_normal[grid_index(qi, qj)], n10 = vert_normal[grid_index(qi + 1, qj)];
            const Vec3 n11 = vert_normal[grid_index(qi + 1, qj + 1)], n01 = vert_normal[grid_index(qi, qj + 1)];

            for (int a = 0; a < subdiv; ++a)
            {
                for (int b = 0; b < subdiv; ++b)
                {
                    const Real u0 = a * inv_s, u1 = (a + 1) * inv_s;
                    const Real v0 = b * inv_s, v1 = (b + 1) * inv_s;

                    const Vec3 s00 = bilerp(p00, p10, p11, p01, u0, v0);
                    const Vec3 s10 = bilerp(p00, p10, p11, p01, u1, v0);
                    const Vec3 s11 = bilerp(p00, p10, p11, p01, u1, v1);
                    const Vec3 s01 = bilerp(p00, p10, p11, p01, u0, v1);

                    auto interp_normal = [&](Real u, Real v)
                    {
                        Vec3 n = bilerp(n00, n10, n11, n01, u, v);
                        return n.squaredNorm() > 1e-20 ? n.normalized() : Vec3(0.0, 1.0, 0.0);
                    };
                    const Vec3 sn00 = interp_normal(u0, v0), sn10 = interp_normal(u1, v0);
                    const Vec3 sn11 = interp_normal(u1, v1), sn01 = interp_normal(u0, v1);

                    // T1 = (s00, s11, s10), T2 = (s00, s01, s11) — same winding as pass 1's
                    // (p11-p00) x (p10-p00) face normal, so front-facing matches the normal here.
                    emit(s00, sn00); emit(s11, sn11); emit(s10, sn10);
                    emit(s00, sn00); emit(s01, sn01); emit(s11, sn11);
                }
            }
        }
    }

    UpdateMeshBuffer(sm.mesh, 0, sm.vertex_buf.data(), (int)(sm.vertex_buf.size() * sizeof(float)), 0);
    UpdateMeshBuffer(sm.mesh, 2, sm.normal_buf.data(), (int)(sm.normal_buf.size() * sizeof(float)), 0);

    rlDisableBackfaceCulling(); // cloth is a single-layer sheet; the fragment shader flips
                                // fragNormal via gl_FrontFacing so both sides light correctly
    DrawMesh(sm.mesh, material, MatrixIdentity());
    rlEnableBackfaceCulling();
}

// `colliding` maps particle dof -> in-contact this frame (empty = nothing highlighted).
// A colliding particle is drawn in `collide_color`, which keeps `color`'s alpha but swaps
// the RGB to flag it (red for target, green for guess — see the color constants below).
void draw_tape_spheres(const SimMesh& mesh, const PointsX& frame, Color color, float particle_radius,
                       const std::vector<bool>& colliding, Color collide_color)
{
    for (Index vi = 0; vi < (Index)mesh.vertices.size(); ++vi)
    {
        const ParticleId dof = mesh.vertices[vi].dof;
        const bool is_colliding = dof >= 0 && dof < (ParticleId)colliding.size() && colliding[dof];
        DrawSphereEx(to_raylib(vertex_position(mesh, frame, vi)), particle_radius, 6, 6,
                     is_colliding ? collide_color : color);
    }
}

// Marks which particles have an active contact recorded at tape.contacts[tape_index].
// Contacts are detected against the frame's pre-step positions (see detect_contacts in
// diffpd.cpp), so they line up exactly with tape.positions[tape_index]. The final tape
// frame has no corresponding contacts entry (contacts.size() == positions.size() - 1),
// so it comes back with nothing marked.
std::vector<bool> colliding_mask(const Tape& tape, int tape_index, Index num_particles)
{
    std::vector<bool> mask(num_particles, false);
    if (tape_index >= 0 && tape_index < (int)tape.contacts.size())
        for (const Contact& c : tape.contacts[tape_index])
            mask[c.particle] = true;
    return mask;
}

// Same as colliding_mask, but from an already-fetched Contacts list rather than a tape index —
// used by the live-scene path (viewer_set_scene), where the caller already has the exact
// Contacts for the frame being shown (no tape/index bounds-checking needed).
std::vector<bool> mask_from_contacts(const Contacts* contacts, Index num_particles)
{
    std::vector<bool> mask(num_particles, false);
    if (contacts)
        for (const Contact& c : *contacts)
            mask[c.particle] = true;
    return mask;
}

// Draws one axis as an arrow: a thin cylinder shaft topped with a cone head.
static void draw_axis_arrow(Vector3 dir, float length, Color color)
{
    constexpr float kShaftFraction = 0.8f;
    constexpr float kShaftRadius   = 0.015f;
    constexpr float kHeadRadius    = 0.04f;

    const Vector3 shaft_end = { dir.x * length * kShaftFraction, dir.y * length * kShaftFraction,
                                 dir.z * length * kShaftFraction };
    const Vector3 tip       = { dir.x * length, dir.y * length, dir.z * length };

    DrawCylinderEx({ 0, 0, 0 }, shaft_end, kShaftRadius, kShaftRadius, 12, color);
    DrawCylinderEx(shaft_end, tip, kHeadRadius, 0.0f, 12, color);
}

void draw_axes(float length)
{
    draw_axis_arrow({ 1, 0, 0 }, length, RED);
    draw_axis_arrow({ 0, 1, 0 }, length, GREEN);
    draw_axis_arrow({ 0, 0, 1 }, length, BLUE);
}

// ----------------------------------------------------------------------------------------------
// Translate gizmo: three colored arrows (matching the RED/GREEN/BLUE X/Y/Z convention used
// elsewhere) anchored on a collider's editable point(s), click-dragged to slide it along one world
// axis. Sphere/Cylinder/Plane have one point (center/origin/origin); Capsule has two (p0, p1),
// each independently draggable. Config-screen only.
// ----------------------------------------------------------------------------------------------

constexpr Vec3  kGizmoAxisDirs[3]   = { Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1) };
const     Color kGizmoAxisColors[3] = { RED, GREEN, BLUE };

// Closest-point-between-two-lines: the axis line P(s) = origin + s*axis_dir (axis_dir unit-length)
// against the mouse ray Q(t) = ray.position + t*ray.direction, treated as an infinite line for
// numerical stability. Returns the axis parameter `s` and the perpendicular distance between the
// two closest points; `valid = false` when the ray is nearly parallel to the axis (camera looking
// straight down it), where the closest point is ill-conditioned.
struct AxisPick
{
    Real s;
    Real dist;
    bool valid;
};

AxisPick closest_axis_ray(const Vec3& origin, const Vec3& axis_dir, const Ray& ray)
{
    const Vec3 ray_pos = from_raylib(ray.position);
    const Vec3 ray_dir = from_raylib(ray.direction).normalized();
    const Vec3 r        = origin - ray_pos;
    const Real b         = axis_dir.dot(ray_dir);
    const Real c         = axis_dir.dot(r);
    const Real f         = ray_dir.dot(r);
    const Real denom     = 1.0 - b * b;

    if (std::abs(denom) < 1e-6) return { 0.0, 0.0, false };

    const Real s = (b * f - c) / denom;
    const Real t = (f - b * c) / denom;
    const Vec3 p_axis = origin + s * axis_dir;
    const Vec3 p_ray  = ray_pos + t * ray_dir;
    return { s, (p_axis - p_ray).norm(), true };
}

// Picks whichever (point, axis) the mouse ray is closest to, across up to `n` gizmo points
// (`origins`/`lengths` parallel arrays), within that point's own `length * kPickFraction` world
// units and within its visible arrow segment [0, length]. `point == -1` if none picked.
struct GizmoPick
{
    int point = -1;
    int axis  = -1;
};

GizmoPick pick_gizmo_multi(const Vec3 origins[], const float lengths[], int n, const Ray& ray)
{
    constexpr Real kPickFraction = 0.12;

    GizmoPick best;
    Real      best_dist = 1e30;
    for (int p = 0; p < n; ++p)
    {
        const Real pick_radius = (Real)lengths[p] * kPickFraction;
        for (int i = 0; i < 3; ++i)
        {
            const AxisPick pick = closest_axis_ray(origins[p], kGizmoAxisDirs[i], ray);
            if (!pick.valid || pick.s < 0.0 || pick.s > (Real)lengths[p]) continue;
            if (pick.dist < pick_radius && pick.dist < best_dist)
            {
                best_dist = pick.dist;
                best      = { p, i };
            }
        }
    }
    return best;
}

// Same shaft+cone-head shape as draw_axis_arrow, but at an explicit world origin (rather than
// always the scene origin) with radii proportional to `length` — this gizmo's length varies with
// camera distance (constant on-screen size), unlike the fixed-size world axis markers.
void draw_gizmo_arrow(Vector3 origin, Vector3 dir, float length, Color color)
{
    constexpr float kShaftFraction   = 0.8f;
    constexpr float kShaftRadiusFrac = 0.022f;
    constexpr float kHeadRadiusFrac  = 0.07f;

    const Vector3 shaft_end = Vector3Add(origin, Vector3Scale(dir, length * kShaftFraction));
    const Vector3 tip       = Vector3Add(origin, Vector3Scale(dir, length));

    DrawCylinderEx(origin, shaft_end, length * kShaftRadiusFrac, length * kShaftRadiusFrac, 12, color);
    DrawCylinderEx(shaft_end, tip, length * kHeadRadiusFrac, 0.0f, 12, color);
}

// Drawn with depth testing/writing off so the arrows always render fully on top of the collider
// (and everything else in the scene) regardless of size or zoom — the standard approach used by
// 3D editors' translate gizmos, since a gizmo anchored at a solid object's center would otherwise
// have its shaft permanently occluded by that object's own geometry.
void draw_translate_gizmo(const Vec3& origin, float length, int hover_axis, int drag_axis)
{
    rlDisableDepthTest();
    rlDisableDepthMask();

    const Vector3 origin_rl = to_raylib(origin);
    for (int i = 0; i < 3; ++i)
    {
        Color color = kGizmoAxisColors[i];
        if (drag_axis == i)       color = YELLOW;
        else if (hover_axis == i) color = ColorBrightness(color, 0.5f);
        draw_gizmo_arrow(origin_rl, to_raylib(kGizmoAxisDirs[i]), length, color);
    }

    // rlgl batches vertices and only actually issues the GL draw call at a flush; the depth
    // state in effect at *that* moment is what applies (not at rlVertex3f time). Force the flush
    // here, while depth test/write are still off, before restoring them for whatever draws next.
    rlDrawRenderBatchActive();

    rlEnableDepthMask();
    rlEnableDepthTest();
}

// Persistent (across a single viewer_show_config_screen call) drag state for the translate gizmo.
// `point` identifies which of the collider's (up to 2) editable points is being dragged.
struct GizmoDragState
{
    int  point = -1;
    int  drag_axis = -1;
    Real drag_t_start = 0.0;
    Vec3 drag_anchor_start = Vec3::Zero();
};

// Which point(s) of the current collider shape get a gizmo, and pointers to them (so dragging can
// write straight back into `c`). Returns the count (0, 1, or 2) and fills `out[0..count-1]`.
int collider_gizmo_anchors(Collider& c, Vec3* out[2])
{
    switch (c.type)
    {
        case ColliderType::Sphere:   out[0] = &c.sphere_center;   return 1;
        case ColliderType::Cylinder: out[0] = &c.cylinder_origin; return 1;
        case ColliderType::Plane:    out[0] = &c.plane_origin;    return 1;
        case ColliderType::Capsule:  out[0] = &c.capsule_p0; out[1] = &c.capsule_p1; return 2;
        case ColliderType::None:     return 0;
    }
    return 0;
}

// Small translucent panel of one-line control hints, anchored to the top-right corner.
void draw_help_box(int screen_width)
{
    static const char* kLines[] = {
        "Space: play/pause",
        "L/R arrow: step frame (Shift: x10)",
        "T / G: toggle target / guess",
        "E: toggle edges",
        "P: toggle particles",
        "S: toggle smooth surface",
        "C: highlight colliding particles",
        "Drag: orbit  |  Shift+drag / RMB: pan",
        "Scroll: zoom",
    };
    constexpr int   kFontSize   = 16;
    constexpr int   kLineHeight = 20;
    constexpr int   kPadding    = 10;
    constexpr int   kNumLines   = (int)(sizeof(kLines) / sizeof(kLines[0]));

    int max_width = 0;
    for (const char* line : kLines)
        max_width = std::max(max_width, MeasureText(line, kFontSize));

    const int box_w = max_width + 2 * kPadding;
    const int box_h = kNumLines * kLineHeight + 2 * kPadding;
    const int box_x = screen_width - box_w - 10;
    const int box_y = 10;

    DrawRectangle(box_x, box_y, box_w, box_h, { 0, 0, 0, 140 });
    DrawRectangleLines(box_x, box_y, box_w, box_h, { 255, 255, 255, 60 });

    for (int i = 0; i < kNumLines; ++i)
        DrawText(kLines[i], box_x + kPadding, box_y + kPadding + i * kLineHeight, kFontSize, RAYWHITE);
}

// DrawPlane always draws a horizontal (XZ, normal +Y) quad in model space, so an arbitrary
// plane normal is applied by rotating the world matrix from +Y onto it before drawing at
// the origin, then restoring the matrix stack.
void draw_plane_oriented(Vector3 center, Vector2 size, Vector3 normal, Color color)
{
    normal = Vector3Normalize(normal);

    Vector3 axis;
    float   angle;
    QuaternionToAxisAngle(QuaternionFromVector3ToVector3({ 0.0f, 1.0f, 0.0f }, normal), &axis, &angle);

    rlPushMatrix();
    rlTranslatef(center.x, center.y, center.z);
    if (angle != 0.0f) rlRotatef(angle * RAD2DEG, axis.x, axis.y, axis.z);
    DrawPlane({ 0.0f, 0.0f, 0.0f }, size, color);
    rlPopMatrix();
}

// Collider geometry is unbounded for Cylinder (infinite radius line) and Plane (infinite
// sheet); both are drawn with a fixed finite visual extent purely for display.
void draw_collider(const Collider& collider, float time, Color color)
{
    constexpr float kCylinderHalfLength = 10.0f;
    constexpr float kPlaneHalfExtent    = 25.0f;
    constexpr float kRadiusReduction    = 0.05f;

    const Vec3 offset_t = collider.velocity * (Real)time;

    switch (collider.type)
    {
        case ColliderType::Sphere:
        {
            const Vec3 center  = collider.sphere_center + offset_t;
            const float radius = (float)(collider.sphere_radius * (1.0 - kRadiusReduction));
            DrawSphereEx(to_raylib(center), radius, 24, 24, color);
            break;
        }
        case ColliderType::Cylinder:
        {
            const Vec3 axis    = collider.cylinder_axis.normalized();
            const Vec3 origin  = collider.cylinder_origin + offset_t;
            const Vec3 p0      = origin - axis * kCylinderHalfLength;
            const Vec3 p1      = origin + axis * kCylinderHalfLength;
            const float radius = (float)(collider.cylinder_radius * (1.0 - kRadiusReduction));
            DrawCylinderEx(to_raylib(p0), to_raylib(p1), radius, radius, 24, color);
            break;
        }
        case ColliderType::Plane:
        {
            const Vec3 origin = collider.plane_origin + offset_t;
            draw_plane_oriented(to_raylib(origin), { 2.0f * kPlaneHalfExtent, 2.0f * kPlaneHalfExtent },
                                 to_raylib(collider.plane_normal), color);
            break;
        }
        case ColliderType::Capsule:
        {
            const Vec3 p0 = collider.capsule_p0 + offset_t;
            const Vec3 p1 = collider.capsule_p1 + offset_t;
            const float radius = (float)(collider.capsule_radius * (1.0 - kRadiusReduction));
            DrawCapsule(to_raylib(p0), to_raylib(p1), radius, 24, 16, color);
            break;
        }
        case ColliderType::None:
            break; // no collider active — nothing to draw
    }
}

// Shared visual language across the live view and the interactive playback: reference/target
// trajectory in translucent orange, live/guess trajectory in white, colliding particles flagged
// in a saturated variant of the same color.
constexpr Color kBackgroundColor     = { 18, 18, 18, 255 };
constexpr Color kReferenceColor      = { 255, 165, 0, 140 }; // orange, half transparent
constexpr Color kLiveColor           = WHITE;
constexpr Color kColliderColor       = { 140, 150, 165, 255 }; // opaque, slightly bluish gray
constexpr Color kReferenceCollide    = { 255, 0, 0, kReferenceColor.a }; // red, same alpha as reference
constexpr Color kLiveCollide         = { 0, 255, 0, kLiveColor.a };      // green, same alpha as live
constexpr float kParticleRadius      = 0.01f;
constexpr float kAxisLength          = 0.25f;
constexpr int   kSurfaceSubdiv       = 4; // sub-quads per coarse cell edge for the smooth surface

// Persistent viewer state: window/shader/camera survive across every phase of a run (target sim,
// guess sim, backward pass, FD check, final playback), so the camera pose carries forward and the
// window is only opened/closed once by main().
struct ViewerState
{
    bool        open = false;
    OrbitCamera orbit;
    Camera3D    camera = { 0 };
    Shader      sphere_shader{};

    // Material wrapping sphere_shader for DrawMesh (the smooth surface). DrawMesh binds
    // material.shader explicitly regardless of any BeginShaderMode/EndShaderMode block, so the
    // surface can be drawn outside the spheres' shader-mode scope with no state interference.
    Material    surface_material{};
    SurfaceMesh reference_surface;
    SurfaceMesh live_surface;

    const SimMesh* mesh = nullptr;
    bool           has_reference = false;
    PointsX        reference_frame;
    Contacts       reference_contacts;
    bool           has_live = false;
    PointsX        live_frame;
    Contacts       live_contacts;
    Collider       collider;
    Real           collider_time = 0.0;
    std::string    status_text;
};

ViewerState g_viewer;

void draw_live_scene()
{
    BeginMode3D(g_viewer.camera);
    draw_axes(kAxisLength);

    if (g_viewer.mesh)
    {
        if (g_viewer.has_reference) draw_tape_edges(*g_viewer.mesh, g_viewer.reference_frame, kReferenceColor);
        if (g_viewer.has_live)      draw_tape_edges(*g_viewer.mesh, g_viewer.live_frame, kLiveColor);

        BeginShaderMode(g_viewer.sphere_shader);
        if (g_viewer.has_reference)
        {
            const std::vector<bool> mask = mask_from_contacts(&g_viewer.reference_contacts, g_viewer.reference_frame.rows());
            draw_tape_spheres(*g_viewer.mesh, g_viewer.reference_frame, kReferenceColor, kParticleRadius, mask, kReferenceCollide);
        }
        if (g_viewer.has_live)
        {
            const std::vector<bool> mask = mask_from_contacts(&g_viewer.live_contacts, g_viewer.live_frame.rows());
            draw_tape_spheres(*g_viewer.mesh, g_viewer.live_frame, kLiveColor, kParticleRadius, mask, kLiveCollide);
        }
        draw_collider(g_viewer.collider, (float)g_viewer.collider_time, kColliderColor);
        EndShaderMode();
    }

    EndMode3D();

    DrawText(g_viewer.status_text.c_str(), 10, 10, 20, WHITE);
    DrawFPS(10, 40);
}

// ----------------------------------------------------------------------------------------------
// Config-screen panel: a small immediate-mode row layout cursor over raygui widgets.
//
// Content height (needed by GuiScrollPanel before any row is drawn) is obtained by running the
// exact same field-drawing sequence twice per frame: once with `measuring = true` (every method
// still advances `y`, but skips the actual Gui*() call/interaction) to get the final height, then
// once for real with that height feeding GuiScrollPanel. This avoids hand-maintained "row count"
// constants silently drifting out of sync with the actual field list.
// ----------------------------------------------------------------------------------------------

// Persistent text-box state for a single scalar Real field — same rationale as Vec3TextState below.
struct FloatTextState
{
    char buf[32];
    bool edit = false;

    explicit FloatTextState(Real initial) { std::snprintf(buf, sizeof(buf), "%.3f", (double)initial); }
};

// Persistent per-field text-box state for a Vec3 (one text buffer + edit-mode flag per
// component). GuiValueBoxFloat only mutates `textValue` while `editMode` is true, so the buffer
// must be pre-seeded from the field's actual starting value once, at construction — done here
// via the constructor argument, evaluated the first time each particular static instance (see
// call sites below) is initialized.
struct Vec3TextState
{
    char buf[3][32];
    bool edit[3] = { false, false, false };

    explicit Vec3TextState(const Vec3& initial)
    {
        std::snprintf(buf[0], sizeof(buf[0]), "%.3f", (double)initial.x());
        std::snprintf(buf[1], sizeof(buf[1]), "%.3f", (double)initial.y());
        std::snprintf(buf[2], sizeof(buf[2]), "%.3f", (double)initial.z());
    }
};

struct PanelCursor
{
    Rectangle view   = { 0, 0, 0, 0 }; // visible sub-rect returned by GuiScrollPanel (screen space)
    Vector2   scroll = { 0, 0 };       // current scroll offset (raygui convention: <=0, more negative = scrolled down)
    float     y      = 0.0f;           // running content-space Y cursor
    bool      measuring = false;       // true: advance y only, draw/interact with nothing

    static constexpr float kRowH   = 24.0f;
    static constexpr float kGap    = 6.0f;
    static constexpr float kPadX   = 8.0f;
    static constexpr float kWidth  = 328.0f; // panel content width, excluding padding/scrollbar
    static constexpr float kLabelW = 130.0f; // label sub-column width for slider/spinner/checkbox rows

    Rectangle row(float height = kRowH)
    {
        const Rectangle r{ view.x + scroll.x + kPadX, view.y + scroll.y + y, kWidth, height };
        y += height + kGap;
        return r;
    }

    void section(const char* title) { const Rectangle r = row(); if (!measuring) GuiLine(r, title); }
    void label(const char* text)    { const Rectangle r = row(); if (!measuring) GuiLabel(r, text);  }

    // GuiSpinner/GuiCheckBox draw their `text` label *outside* the bounds rect passed in (to the
    // left/right), which got clipped by the scroll panel's scissor region when the control's
    // bounds spanned the full row width. Fix: never pass text to those controls — reserve a label
    // sub-column drawn via GuiLabel (confirmed to render inside its own bounds) and give the
    // control itself only the remaining width.
    //
    // Free-typed float box (not a slider) — user enters the exact value; an unparsable/empty
    // string reads back as 0.0 (TextToFloat's own behavior, needs no extra handling here). Every
    // Real (double) field does the float round-trip here, once, rather than at each call site.
    void float_box(const char* name, Real* value, char* buf, bool& edit_mode)
    {
        const Rectangle r = row();
        if (measuring) return;
        const Rectangle label_rect = { r.x, r.y, kLabelW, r.height };
        const Rectangle box_rect   = { r.x + kLabelW + kGap, r.y, r.width - kLabelW - kGap, r.height };
        GuiLabel(label_rect, name);
        // raygui only writes `buf` from keystrokes while editing, never from external changes to
        // *value (e.g. a value dragged in the 3D view) — resync it here whenever the user isn't
        // actively typing so the displayed text never goes stale.
        if (!edit_mode) std::snprintf(buf, 32, "%.3f", (double)*value);
        float v = (float)*value;
        if (GuiValueBoxFloat(box_rect, nullptr, buf, &v, edit_mode)) edit_mode = !edit_mode;
        *value = (Real)v;
    }

    // Same as float_box, but the label is "<prefix> <axis>" with the axis letter drawn in
    // axis_color (matching the RED/GREEN/BLUE of the viewport's own X/Y/Z axis arrows) instead
    // of the normal label color.
    void float_box_axis(const char* prefix, char axis, Color axis_color, Real* value, char* buf, bool& edit_mode)
    {
        const Rectangle r = row();
        if (measuring) return;
        const Rectangle label_rect = { r.x, r.y, kLabelW, r.height };
        const Rectangle box_rect   = { r.x + kLabelW + kGap, r.y, r.width - kLabelW - kGap, r.height };

        const char* prefix_sp = TextFormat("%s ", prefix);
        const char  axis_str[2] = { axis, '\0' };
        GuiLabel(label_rect, prefix_sp);
        const float axis_x = label_rect.x + (float)GuiGetTextWidth(prefix_sp);
        const Rectangle axis_rect = { axis_x, label_rect.y, label_rect.x + label_rect.width - axis_x, label_rect.height };
        const int prev_color = GuiGetStyle(LABEL, TEXT_COLOR_NORMAL);
        GuiSetStyle(LABEL, TEXT_COLOR_NORMAL, ColorToInt(axis_color));
        GuiLabel(axis_rect, axis_str);
        GuiSetStyle(LABEL, TEXT_COLOR_NORMAL, prev_color);

        // See float_box's comment: keep the text buffer synced to *value while not editing, so an
        // external write (e.g. the collider gizmo dragging this axis) doesn't leave stale text.
        if (!edit_mode) std::snprintf(buf, 32, "%.3f", (double)*value);
        float v = (float)*value;
        if (GuiValueBoxFloat(box_rect, nullptr, buf, &v, edit_mode)) edit_mode = !edit_mode;
        *value = (Real)v;
    }

    void vec3_field(const char* name, Vec3* value, Vec3TextState& state)
    {
        float_box_axis(name, 'X', RED,   &value->x(), state.buf[0], state.edit[0]);
        float_box_axis(name, 'Y', GREEN, &value->y(), state.buf[1], state.edit[1]);
        float_box_axis(name, 'Z', BLUE,  &value->z(), state.buf[2], state.edit[2]);
    }

    void int_spinner(const char* name, int* value, int lo, int hi, bool* edit_mode)
    {
        const Rectangle r = row();
        if (measuring) return;
        const Rectangle label_rect   = { r.x, r.y, kLabelW, r.height };
        const Rectangle spinner_rect = { r.x + kLabelW + kGap, r.y, r.width - kLabelW - kGap, r.height };
        GuiLabel(label_rect, name);
        if (GuiSpinner(spinner_rect, nullptr, value, lo, hi, *edit_mode)) *edit_mode = !*edit_mode;
    }

    void checkbox_field(const char* name, bool* value)
    {
        const Rectangle r = row();
        if (measuring) return;
        constexpr float kBoxSize = 18.0f;
        const Rectangle box_rect   = { r.x, r.y + (r.height - kBoxSize) * 0.5f, kBoxSize, kBoxSize };
        const Rectangle label_rect = { r.x + kBoxSize + kGap, r.y, r.width - kBoxSize - kGap, r.height };
        GuiCheckBox(box_rect, nullptr, value);
        GuiLabel(label_rect, name);
    }

    // One-off enum<->int shims — only 3 enum fields total, not worth a generic templated helper.
    void pin_mode_field(PinMode& mode)
    {
        label("Pin Mode");
        const Rectangle r = row();
        if (measuring) return;
        int active = (int)mode;
        GuiToggleGroup(r, "None;Corners;Row", &active);
        mode = (PinMode)active;
    }

    void hanging_mode_field(HangingMode& mode)
    {
        label("Hanging Mode");
        const Rectangle r = row();
        if (measuring) return;
        int active = (int)mode;
        GuiToggleGroup(r, "Horizontal;Vertical", &active);
        mode = (HangingMode)active;
    }

    void collider_type_field(ColliderType& type)
    {
        label("Collider Shape");
        const Rectangle r = row();
        if (measuring) return;
        int active = (int)type;
        GuiToggleGroup(r, "Sphere;Cylinder;Plane;Capsule;None", &active);
        type = (ColliderType)active;
    }

    // One checkbox per order of magnitude (1e-2 .. 1e-10), label drawn above each box. `selected`
    // must point at an array of (at least) 9 bools, indexed the same way as kFDEpsilonValues.
    void fd_epsilon_row_field(bool* selected)
    {
        label("FD Epsilon (order of magnitude)");
        const Rectangle r = row(32.0f);
        if (measuring) return;

        static const char* kNames[9] = { "1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9", "1e-10" };
        constexpr int   kCount   = 9;
        constexpr float kBoxSize = 16.0f;
        const float colW = r.width / (float)kCount;

        const int prev_align = GuiGetStyle(LABEL, TEXT_ALIGNMENT);
        GuiSetStyle(LABEL, TEXT_ALIGNMENT, TEXT_ALIGN_CENTER);
        for (int i = 0; i < kCount; ++i)
        {
            const Rectangle label_rect = { r.x + i * colW, r.y, colW, 14.0f };
            const Rectangle box_rect   = { r.x + i * colW + (colW - kBoxSize) * 0.5f, r.y + 16.0f, kBoxSize, kBoxSize };
            GuiLabel(label_rect, kNames[i]);
            GuiCheckBox(box_rect, nullptr, &selected[i]);
        }
        GuiSetStyle(LABEL, TEXT_ALIGNMENT, prev_align);
    }
};

// Draws only the fields relevant to whichever collider shape is currently active. Each shape's
// Vec3TextState statics are seeded from Collider's own defaults the first time that shape's case
// actually runs (which may be later than frame 1, if the user switches shape) — correct either
// way, since default_config_collider() pre-fills every shape's fields up front regardless of
// which one is initially active.
void collider_shape_fields(PanelCursor& cur, Collider& c)
{
    switch (c.type)
    {
        case ColliderType::Sphere:
        {
            static Vec3TextState center_state(c.sphere_center);
            static FloatTextState radius_state(c.sphere_radius);
            cur.vec3_field("Sphere Center", &c.sphere_center, center_state);
            cur.float_box("Sphere Radius", &c.sphere_radius, radius_state.buf, radius_state.edit);
            break;
        }
        case ColliderType::Cylinder:
        {
            static Vec3TextState origin_state(c.cylinder_origin);
            static Vec3TextState axis_state(c.cylinder_axis);
            static FloatTextState radius_state(c.cylinder_radius);
            cur.vec3_field("Cylinder Origin", &c.cylinder_origin, origin_state);
            cur.vec3_field("Cylinder Axis", &c.cylinder_axis, axis_state);
            cur.float_box("Cylinder Radius", &c.cylinder_radius, radius_state.buf, radius_state.edit);
            break;
        }
        case ColliderType::Plane:
        {
            static Vec3TextState origin_state(c.plane_origin);
            static Vec3TextState normal_state(c.plane_normal);
            cur.vec3_field("Plane Origin", &c.plane_origin, origin_state);
            cur.vec3_field("Plane Normal", &c.plane_normal, normal_state);
            break;
        }
        case ColliderType::Capsule:
        {
            static Vec3TextState p0_state(c.capsule_p0);
            static Vec3TextState p1_state(c.capsule_p1);
            static FloatTextState radius_state(c.capsule_radius);
            cur.vec3_field("Capsule P0", &c.capsule_p0, p0_state);
            cur.vec3_field("Capsule P1", &c.capsule_p1, p1_state);
            cur.float_box("Capsule Radius", &c.capsule_radius, radius_state.buf, radius_state.edit);
            break;
        }
        case ColliderType::None:
            cur.label("No collider active — contact disabled");
            break;
    }
}

// Full field list for the config screen, in display order. Run identically for the measuring
// pass and the real draw pass (see PanelCursor comment above).
void draw_config_fields(PanelCursor& cur, AppConfig& cfg)
{
    static bool edit_width = false, edit_height = false, edit_fps = false,
                edit_frame_substeps = false, edit_secs = false,
                edit_n_iters = false, edit_n_iters_adjoint = false;
    static Vec3TextState origin_state(cfg.origin);
    static Vec3TextState target_origin_state(cfg.target_origin);
    static Vec3TextState velocity_state(cfg.collider.velocity);
    static Vec3TextState gravity_state(cfg.gravity);
    static FloatTextState stiffness_state(cfg.stiffness);
    static FloatTextState target_stiffness_state(cfg.target_stiffness);
    static FloatTextState m_tot_state(cfg.m_tot);

    cur.section("Cloth");
    cur.int_spinner("Width",  &cfg.width,  2, 200, &edit_width);
    cur.int_spinner("Height", &cfg.height, 2, 200, &edit_height);
    cur.float_box("Stiffness", &cfg.stiffness, stiffness_state.buf, stiffness_state.edit);
    cur.vec3_field("Origin", &cfg.origin, origin_state);
    cur.pin_mode_field(cfg.pin_mode);
    cur.hanging_mode_field(cfg.hang_mode);
    cur.checkbox_field("Shear constraints",   &cfg.flag_shear);
    cur.checkbox_field("Bending constraints", &cfg.flag_bending);
    cur.label("Stretch constraints: always on");
    cur.float_box("Total Mass", &cfg.m_tot, m_tot_state.buf, m_tot_state.edit);

    cur.section("Target Cloth");
    cur.float_box("Target Stiffness", &cfg.target_stiffness, target_stiffness_state.buf, target_stiffness_state.edit);
    cur.vec3_field("Target Origin", &cfg.target_origin, target_origin_state);

    cur.section("Collision");
    cur.collider_type_field(cfg.collider.type);
    collider_shape_fields(cur, cfg.collider);
    cur.vec3_field("Collider Velocity", &cfg.collider.velocity, velocity_state);

    cur.section("Physics");
    cur.vec3_field("Gravity", &cfg.gravity, gravity_state);

    cur.section("Simulation / Solver");
    cur.int_spinner("FPS",             &cfg.FPS,             1, 240,  &edit_fps);
    cur.int_spinner("Frame Substeps",  &cfg.frame_substeps,  1, 64,   &edit_frame_substeps);
    cur.int_spinner("Seconds",         &cfg.secs,            1, 120,  &edit_secs);
    cur.int_spinner("Solver Iters",    &cfg.n_iters,          1, 1000, &edit_n_iters);
    cur.int_spinner("Adjoint Iters",   &cfg.n_iters_adjoint,  1, 1000, &edit_n_iters_adjoint);

    cur.section("Gradient Check");
    cur.checkbox_field("Enable FD Check (dphi/dk)", &cfg.run_fd_check);
    if (cfg.run_fd_check) cur.fd_epsilon_row_field(cfg.fd_eps_selected);

    const int substeps = cfg.FPS * cfg.frame_substeps;
    const Real dt      = substeps > 0 ? 1.0 / substeps : 0.0;
    const int  n_steps = substeps * cfg.secs;
    cur.label(TextFormat("substeps=%d  dt=%.5f  n_steps=%d", substeps, dt, n_steps));
}

float measure_content_height(AppConfig& cfg)
{
    PanelCursor cur;
    cur.measuring = true;
    draw_config_fields(cur, cfg);
    return cur.y;
}

} // namespace

void viewer_open()
{
    InitWindow(1280, 800, "diffpd viewer");
    SetTargetFPS(60);
    GuiLoadStyleDark();
    GuiSetStyle(TOGGLE, GROUP_WIDTH_FULL, 1); // one GuiToggleGroup() call divides the full row width evenly

    g_viewer.camera.fovy       = 45.0f;
    g_viewer.camera.projection = CAMERA_PERSPECTIVE;
    update_orbit_camera(g_viewer.orbit, g_viewer.camera);

    g_viewer.sphere_shader = LoadShaderFromMemory(kSphereVS, kSphereFS);
    const Vector3 light_dir = Vector3Normalize({ 0.4f, 1.0f, 0.3f }); // from above, diagonal
    SetShaderValue(g_viewer.sphere_shader, GetShaderLocation(g_viewer.sphere_shader, "lightDir"),
                   &light_dir, SHADER_UNIFORM_VEC3);

    g_viewer.surface_material         = LoadMaterialDefault();
    g_viewer.surface_material.shader  = g_viewer.sphere_shader;

    g_viewer.open = true;
}

void viewer_close()
{
    if (g_viewer.reference_surface.uploaded) UnloadMesh(g_viewer.reference_surface.mesh);
    if (g_viewer.live_surface.uploaded)      UnloadMesh(g_viewer.live_surface.mesh);

    // surface_material.shader aliases sphere_shader (see viewer_open) — clear it before
    // UnloadMaterial so it only frees the maps array, not the shader we're about to unload below.
    g_viewer.surface_material.shader = Shader{};
    UnloadMaterial(g_viewer.surface_material);

    UnloadShader(g_viewer.sphere_shader);
    CloseWindow();
    g_viewer.open = false;
}

void viewer_set_scene(const SimMesh& mesh,
                       const PointsX* reference_frame, const Contacts* reference_contacts,
                       const PointsX* live_frame, const Contacts* live_contacts,
                       const Collider& collider, Real collider_time,
                       const std::string& status_text)
{
    g_viewer.mesh = &mesh;

    g_viewer.has_reference = reference_frame != nullptr;
    if (g_viewer.has_reference) g_viewer.reference_frame = *reference_frame;
    g_viewer.reference_contacts = reference_contacts ? *reference_contacts : Contacts{};

    g_viewer.has_live = live_frame != nullptr;
    if (g_viewer.has_live) g_viewer.live_frame = *live_frame;
    g_viewer.live_contacts = live_contacts ? *live_contacts : Contacts{};

    g_viewer.collider      = collider;
    g_viewer.collider_time = collider_time;
    g_viewer.status_text   = status_text;
}

void viewer_set_status(const std::string& status_text)
{
    g_viewer.status_text = status_text;
}

bool viewer_render_frame()
{
    ASSERT(g_viewer.open, "viewer_render_frame: viewer_open() was not called");

    update_orbit_camera(g_viewer.orbit, g_viewer.camera);

    BeginDrawing();
    ClearBackground(kBackgroundColor);
    draw_live_scene();
    EndDrawing();

    return !WindowShouldClose();
}

bool viewer_show_config_screen(AppConfig& cfg)
{
    ASSERT(g_viewer.open, "viewer_show_config_screen: viewer_open() was not called");

    constexpr float kPanelWidth = 360.0f;
    constexpr float kFooterH    = 50.0f;

    const int viewport_w = std::max(1, GetScreenWidth() - (int)kPanelWidth);
    const int viewport_h = std::max(1, GetScreenHeight());
    RenderTexture2D viewport_rt = LoadRenderTexture(viewport_w, viewport_h);

    Vector2 scroll = { 0, 0 };
    bool viewport_has_mouse_capture = false; // latched at press time; see mouse-arbitration note below

    GizmoDragState gizmo_state;

    bool run_clicked = false;

    while (!WindowShouldClose() && !run_clicked)
    {
        const Vector2 mouse = GetMousePosition();
        const bool over_viewport = mouse.x >= kPanelWidth;

        // --- collider translate gizmo(s): hover pick + drag update -----------------------------
        // Done before the orbit-camera arbitration below so a press that grabs an axis arrow can
        // suppress that same press from also starting an orbit.
        Vec3* anchor_ptrs[2] = { nullptr, nullptr };
        const int n_points = collider_gizmo_anchors(cfg.collider, anchor_ptrs);
        if (gizmo_state.point >= n_points) { gizmo_state.point = -1; gizmo_state.drag_axis = -1; }

        const Ray mouse_ray = GetScreenToWorldRayEx({ mouse.x - kPanelWidth, mouse.y },
                                                     g_viewer.camera, viewport_w, viewport_h);

        constexpr float kGizmoScreenScale = 0.15f; // gizmo length as a fraction of camera distance
        constexpr float kGizmoMinLength   = 0.05f;

        Vec3  gizmo_origins[2];
        float gizmo_lengths[2];
        for (int p = 0; p < n_points; ++p)
        {
            gizmo_origins[p] = *anchor_ptrs[p];
            const float dist_to_camera = Vector3Distance(g_viewer.camera.position, to_raylib(gizmo_origins[p]));
            gizmo_lengths[p] = std::max(kGizmoMinLength, dist_to_camera * kGizmoScreenScale);
        }

        int hover_point = -1, hover_axis = -1;
        if (gizmo_state.point == -1 && over_viewport && n_points > 0)
        {
            const GizmoPick pick = pick_gizmo_multi(gizmo_origins, gizmo_lengths, n_points, mouse_ray);
            hover_point = pick.point;
            hover_axis  = pick.axis;
        }

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && hover_axis != -1)
        {
            gizmo_state.point             = hover_point;
            gizmo_state.drag_axis         = hover_axis;
            gizmo_state.drag_anchor_start = gizmo_origins[hover_point];
            gizmo_state.drag_t_start      = closest_axis_ray(gizmo_state.drag_anchor_start,
                                                              kGizmoAxisDirs[hover_axis], mouse_ray).s;
        }
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
        {
            gizmo_state.point     = -1;
            gizmo_state.drag_axis = -1;
        }

        if (gizmo_state.point != -1 && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
        {
            const Vec3&    axis_dir = kGizmoAxisDirs[gizmo_state.drag_axis];
            const AxisPick pick     = closest_axis_ray(gizmo_state.drag_anchor_start, axis_dir, mouse_ray);
            if (pick.valid)
                *anchor_ptrs[gizmo_state.point] = gizmo_state.drag_anchor_start
                                                 + axis_dir * (pick.s - gizmo_state.drag_t_start);

            // gizmo_origins/lengths were captured before this update; resync the dragged point so
            // the arrows drawn below track the mouse this frame instead of lagging by one.
            gizmo_origins[gizmo_state.point] = *anchor_ptrs[gizmo_state.point];
            const float dist_to_camera = Vector3Distance(g_viewer.camera.position, to_raylib(gizmo_origins[gizmo_state.point]));
            gizmo_lengths[gizmo_state.point] = std::max(kGizmoMinLength, dist_to_camera * kGizmoScreenScale);
        }

        // --- mouse arbitration: panel vs. viewport --------------------------------------------
        // Latched at the moment a button is *pressed*, not re-checked continuously — otherwise a
        // slider drag that carries the cursor past the panel/viewport boundary mid-drag would
        // spuriously also start orbiting the camera that same frame. A press that grabbed a gizmo
        // axis this frame (point != -1) must not also capture the viewport for orbiting.
        const bool any_button_down = IsMouseButtonDown(MOUSE_BUTTON_LEFT) || IsMouseButtonDown(MOUSE_BUTTON_RIGHT);
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsMouseButtonPressed(MOUSE_BUTTON_RIGHT))
            viewport_has_mouse_capture = over_viewport && gizmo_state.point == -1;
        else if (!any_button_down)
            viewport_has_mouse_capture = over_viewport;

        if (viewport_has_mouse_capture)
            update_orbit_camera(g_viewer.orbit, g_viewer.camera);

        // --- rebuild the initial-condition preview (cheap: grid generation, no solve) ----------
        const uint8_t flags = ClothFlags::STRETCH
                             | (cfg.flag_shear   ? ClothFlags::SHEAR   : 0)
                             | (cfg.flag_bending ? ClothFlags::BENDING : 0);
        Object target_obj = cloth(cfg.width, cfg.height, cfg.target_stiffness, cfg.target_origin,
                                   cfg.pin_mode, cfg.hang_mode, flags, cfg.m_tot);
        Object guess_obj  = cloth(cfg.width, cfg.height, cfg.stiffness, cfg.origin,
                                   cfg.pin_mode, cfg.hang_mode, flags, cfg.m_tot);
        const PointsX target_frame = Eigen::Map<const PointsX>(target_obj.x.data(), target_obj.num_particles(), 3);
        const PointsX guess_frame  = Eigen::Map<const PointsX>(guess_obj.x.data(),  guess_obj.num_particles(),  3);

        // --- render the 3D preview into its own render texture ---------------------------------
        // (BeginMode3D's projection aspect ratio is derived from the *current* render target's
        // size, not the window's — so routing through a RenderTexture2D, rather than clipping the
        // full-window draw with rlViewport(), gets the right aspect ratio for the sub-rect for free.)
        BeginTextureMode(viewport_rt);
            ClearBackground(kBackgroundColor);
            BeginMode3D(g_viewer.camera);
                draw_axes(kAxisLength);
                draw_tape_edges(target_obj.mesh, target_frame, kReferenceColor);
                draw_tape_edges(guess_obj.mesh,  guess_frame,  kLiveColor);
                BeginShaderMode(g_viewer.sphere_shader);
                    draw_tape_spheres(target_obj.mesh, target_frame, kReferenceColor, kParticleRadius, {}, kReferenceCollide);
                    draw_tape_spheres(guess_obj.mesh,  guess_frame,  kLiveColor,      kParticleRadius, {}, kLiveCollide);
                    draw_collider(cfg.collider, 0.0f, kColliderColor); // time=0: static preview
                EndShaderMode();
                for (int p = 0; p < n_points; ++p)
                {
                    const int point_hover_axis = (hover_point == p) ? hover_axis : -1;
                    const int point_drag_axis  = (gizmo_state.point == p) ? gizmo_state.drag_axis : -1;
                    draw_translate_gizmo(gizmo_origins[p], gizmo_lengths[p], point_hover_axis, point_drag_axis);
                }
            EndMode3D();
        EndTextureMode();

        // --- composite: viewport texture + scrollable panel + Run button -----------------------
        BeginDrawing();
            ClearBackground(kBackgroundColor);

            DrawTextureRec(viewport_rt.texture,
                           { 0, 0, (float)viewport_w, -(float)viewport_h }, // negative height: render textures are Y-flipped
                           { kPanelWidth, 0 }, WHITE);

            const float scroll_area_h = (float)GetScreenHeight() - kFooterH;
            const Rectangle panel_bounds = { 0, 0, kPanelWidth, scroll_area_h };
            const Rectangle content = { 0, 0, kPanelWidth - 16.0f, measure_content_height(cfg) };
            Rectangle view;
            GuiScrollPanel(panel_bounds, nullptr, content, &scroll, &view);

            BeginScissorMode((int)view.x, (int)view.y, (int)view.width, (int)view.height);
                PanelCursor cur;
                cur.view   = view;
                cur.scroll = scroll;
                draw_config_fields(cur, cfg);
            EndScissorMode();

            const Rectangle run_rect = { 8.0f, scroll_area_h + 8.0f, kPanelWidth - 16.0f, kFooterH - 16.0f };
            run_clicked = GuiButton(run_rect, "Run");
        EndDrawing();
    }

    UnloadRenderTexture(viewport_rt);
    return run_clicked;
}

bool viewer_interactive_playback(const SimMesh& mesh, const Tape& target_tape, const Tape& guess_tape,
                                  const Collider& collider, Real dt, int frame_substeps, int fps)
{
    ASSERT(g_viewer.open, "viewer_interactive_playback: viewer_open() was not called");
    ASSERT(target_tape.positions.size() == guess_tape.positions.size(),
           "viewer_interactive_playback: tape length mismatch");
    ASSERT(frame_substeps > 0, "viewer_interactive_playback: frame_substeps must be positive");
    ASSERT(fps > 0, "viewer_interactive_playback: fps must be positive");

    const int n_tape_frames = (int)target_tape.positions.size();
    ASSERT((n_tape_frames - 1) % frame_substeps == 0,
           "viewer_interactive_playback: tape length - 1 must be a multiple of frame_substeps");
    const int n_frames = (n_tape_frames - 1) / frame_substeps + 1;

    const double frame_period = 1.0 / fps;

    constexpr int    kFrameJump          = 10;   // frames skipped per Shift+arrow step
    constexpr double kScrubInitialDelay  = 0.35; // seconds an arrow must be held before auto-repeat kicks in
    constexpr double kScrubRepeatPeriod  = 0.05; // seconds between steps once auto-repeat is active (20 steps/s)

    int    current_frame     = 0;
    double accumulator       = 0.0;
    double scrub_hold_time   = 0.0; // how long the currently-held arrow key has been down
    double scrub_accumulator = 0.0; // time banked toward the next auto-repeat step
    bool   paused          = false;
    bool   show_target     = true;
    bool   show_guess      = true;
    bool   show_edges      = true;
    bool   show_particles  = true;
    bool   show_surface    = true;
    bool   show_collisions = false;
    bool   back_to_config  = false;

    while (!WindowShouldClose() && !back_to_config)
    {
        update_orbit_camera(g_viewer.orbit, g_viewer.camera);

        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_T))     show_target     = !show_target;
        if (IsKeyPressed(KEY_G))     show_guess      = !show_guess;
        if (IsKeyPressed(KEY_E))     show_edges      = !show_edges;
        if (IsKeyPressed(KEY_P))     show_particles  = !show_particles;
        if (IsKeyPressed(KEY_S))     show_surface    = !show_surface;
        if (IsKeyPressed(KEY_C))     show_collisions = !show_collisions;

        if (!paused)
        {
            accumulator += GetFrameTime();
            while (accumulator >= frame_period)
            {
                accumulator -= frame_period;
                current_frame = (current_frame + 1) % n_frames;
            }
        }
        else
        {
            const bool shift_held = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
            const int  step       = shift_held ? kFrameJump : 1;

            const bool right_down = IsKeyDown(KEY_RIGHT);
            const bool left_down  = IsKeyDown(KEY_LEFT);
            const int  held_dir   = right_down != left_down ? (right_down ? 1 : -1) : 0; // ignore both-held

            if (IsKeyPressed(KEY_RIGHT)) { current_frame = std::min(current_frame + step, n_frames - 1); scrub_hold_time = 0.0; scrub_accumulator = 0.0; }
            if (IsKeyPressed(KEY_LEFT))  { current_frame = std::max(current_frame - step, 0);             scrub_hold_time = 0.0; scrub_accumulator = 0.0; }

            if (held_dir != 0)
            {
                scrub_hold_time += GetFrameTime();
                if (scrub_hold_time >= kScrubInitialDelay)
                {
                    scrub_accumulator += GetFrameTime();
                    while (scrub_accumulator >= kScrubRepeatPeriod)
                    {
                        scrub_accumulator -= kScrubRepeatPeriod;
                        current_frame = std::clamp(current_frame + held_dir * step, 0, n_frames - 1);
                    }
                }
            }
            else
            {
                scrub_hold_time   = 0.0;
                scrub_accumulator = 0.0;
            }
        }

        const int   tape_index    = current_frame * frame_substeps;
        const float collider_time = (float)(current_frame * frame_substeps) * (float)dt;
        const double sim_time     = (double)(current_frame * frame_substeps) * (double)dt;

        BeginDrawing();
        ClearBackground(kBackgroundColor);

        BeginMode3D(g_viewer.camera);
        draw_axes(kAxisLength);

        // Smooth surface first (base layer), then edges, then particles — so the wireframe and
        // particles remain visible on top of the filled surface rather than getting depth-fought.
        if (show_target && show_surface)
            draw_tape_surface(mesh, target_tape.positions[tape_index], kReferenceColor, kSurfaceSubdiv,
                               g_viewer.reference_surface, g_viewer.surface_material);
        if (show_guess && show_surface)
            draw_tape_surface(mesh, guess_tape.positions[tape_index], kLiveColor, kSurfaceSubdiv,
                               g_viewer.live_surface, g_viewer.surface_material);

        if (show_target && show_edges) draw_tape_edges(mesh, target_tape.positions[tape_index], kReferenceColor);
        if (show_guess  && show_edges) draw_tape_edges(mesh, guess_tape.positions[tape_index],  kLiveColor);

        BeginShaderMode(g_viewer.sphere_shader);

        if (show_target && show_particles)
        {
            const std::vector<bool> mask = show_collisions
                ? colliding_mask(target_tape, tape_index, target_tape.positions[tape_index].rows())
                : std::vector<bool>{};
            draw_tape_spheres(mesh, target_tape.positions[tape_index], kReferenceColor, kParticleRadius, mask, kReferenceCollide);
        }
        if (show_guess && show_particles)
        {
            const std::vector<bool> mask = show_collisions
                ? colliding_mask(guess_tape, tape_index, guess_tape.positions[tape_index].rows())
                : std::vector<bool>{};
            draw_tape_spheres(mesh, guess_tape.positions[tape_index], kLiveColor, kParticleRadius, mask, kLiveCollide);
        }
        draw_collider(collider, collider_time, kColliderColor);
        EndShaderMode();
        EndMode3D();

        DrawText(TextFormat("Frame %d / %d   t = %.3fs%s",
                             current_frame + 1, n_frames, sim_time, paused ? "  (paused)" : ""),
                 10, 10, 20, WHITE);
        DrawFPS(10, 40);
        draw_help_box(GetScreenWidth());

        const Rectangle back_button_rect = { 10.0f, (float)GetScreenHeight() - 40.0f, 170.0f, 30.0f };
        if (GuiButton(back_button_rect, "Back to Setup")) back_to_config = true;

        EndDrawing();
    }

    return back_to_config;
}
