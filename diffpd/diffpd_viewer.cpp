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
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

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
    bool    ortho    = false; // toggled by Numpad-5, Blender-style
    bool    top_locked = false; // true right after Numpad-7: camera looks straight down with
                                 // up = +Z, bypassing yaw/pitch below (which are singular at the
                                 // pole — up would end up parallel to the view direction)
};

// Perspective FOV (degrees) used both for the perspective projection itself and, in orthographic
// mode, to size camera.fovy (there re-purposed by raylib as the ortho view's world-space height —
// see rcore.c's rlOrtho(-top*aspect, top*aspect, -top, top, ...) with top = camera.fovy/2) so that
// switching projections at the current distance doesn't jump the apparent framing.
constexpr float kPerspectiveFovY = 45.0f;

void update_orbit_camera(OrbitCamera& orbit, Camera3D& camera)
{
    constexpr float rotate_speed = 0.01f;
    constexpr float pan_speed    = 0.0015f;
    constexpr float zoom_speed   = 0.1f;
    constexpr float min_distance = 0.1f;
    constexpr float max_pitch    = 89.0f * DEG2RAD;

    if (IsKeyPressed(KEY_KP_5)) orbit.ortho = !orbit.ortho;

    // Blender-style axis-aligned view snaps. World up here is +Y (not Blender's +Z), so Front/Right
    // reuse the normal yaw/pitch orbit unchanged — their up is world +Y, same as ordinary orbiting.
    // Top can't: it needs up = +Z, which the fixed up=(0,1,0) below can't produce without becoming
    // degenerate at pitch=90°, so it's a separate locked pose instead of a yaw/pitch value.
    if (IsKeyPressed(KEY_KP_1)) { orbit.yaw = 90.0f * DEG2RAD; orbit.pitch = 0.0f; orbit.top_locked = false; orbit.ortho = true; } // Front: look -Z
    if (IsKeyPressed(KEY_KP_3)) { orbit.yaw =  0.0f * DEG2RAD; orbit.pitch = 0.0f; orbit.top_locked = false; orbit.ortho = true; } // Right: look -X
    if (IsKeyPressed(KEY_KP_7)) { orbit.top_locked = true; orbit.ortho = true; }                                                  // Top:   look -Y

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
        // Rotating away from a Numpad-7 top-lock resumes from just under the pole (the same
        // max_pitch clamp used everywhere else) rather than exactly at it, since up=(0,1,0) is
        // only valid strictly below pitch=90°. yaw keeps whatever value it last held.
        if (orbit.top_locked) { orbit.top_locked = false; orbit.pitch = max_pitch; }
        orbit.yaw   -= mouse_delta.x * rotate_speed;
        orbit.pitch += mouse_delta.y * rotate_speed;
        orbit.pitch  = std::clamp(orbit.pitch, -max_pitch, max_pitch);
    }

    orbit.distance -= GetMouseWheelMove() * zoom_speed * orbit.distance;
    orbit.distance  = std::max(orbit.distance, min_distance);

    camera.target = orbit.target;
    if (orbit.top_locked)
    {
        camera.position = Vector3Add(orbit.target, { 0.0f, orbit.distance, 0.0f });
        camera.up       = { 0.0f, 0.0f, 1.0f };
    }
    else
    {
        const Vector3 offset = {
            orbit.distance * cosf(orbit.pitch) * cosf(orbit.yaw),
            orbit.distance * sinf(orbit.pitch),
            orbit.distance * cosf(orbit.pitch) * sinf(orbit.yaw)
        };
        camera.position = Vector3Add(orbit.target, offset);
        camera.up       = { 0.0f, 1.0f, 0.0f };
    }
    camera.projection = orbit.ortho ? CAMERA_ORTHOGRAPHIC : CAMERA_PERSPECTIVE;
    camera.fovy       = orbit.ortho
        ? 2.0f * orbit.distance * tanf(kPerspectiveFovY * DEG2RAD * 0.5f) // world-space ortho height
        : kPerspectiveFovY;                                               // perspective FOV, degrees
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
    bool                       built_wrap_i = false;
    bool                       uploaded     = false;
};

// sm.mesh.vertices/normals/colors point into vertex_buf/normal_buf/color_buf (below), which
// std::vector owns — but UnloadMesh() unconditionally RL_FREE()s whatever those pointers are (it
// assumes the usual raylib mesh, whose CPU arrays it owns itself). Null them out first so that
// free is a no-op and the vectors stay the sole owner of their memory; otherwise this is a double
// free the moment the vector next reallocates or is destroyed (heap corruption, manifesting as an
// unpredictable crash sometime after this call).
void unload_surface_mesh(SurfaceMesh& sm)
{
    if (!sm.uploaded) return;
    sm.mesh.vertices = nullptr;
    sm.mesh.normals  = nullptr;
    sm.mesh.colors   = nullptr;
    UnloadMesh(sm.mesh);
    sm.mesh     = Mesh{};
    sm.uploaded = false;
}

// (Re)allocates GPU buffers sized for `width`x`height` at subdivision `subdiv`, only when those
// differ from what's already built (e.g. first use, or the user changed cloth dimensions and hit
// "Run" again). `color` is baked into every vertex here since it never changes frame to frame.
// `wrap_i` adds one extra ring of quads closing the seam between column width-1 and column 0 (see
// SimMesh::wrap_i) — a skirt has one more quad column around its circumference than an open sheet.
void ensure_surface_mesh(SurfaceMesh& sm, Index width, Index height, int subdiv, Color color, bool wrap_i)
{
    if (sm.built_width == width && sm.built_height == height && sm.built_subdiv == subdiv
        && sm.built_wrap_i == wrap_i)
        return;

    unload_surface_mesh(sm);

    const Index cells_i = wrap_i ? width : width - 1, cells_j = height - 1;
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
    sm.built_wrap_i = wrap_i;
}

// Regenerates sm.vertex_buf/normal_buf from the current frame's positions and uploads them.
void draw_tape_surface(const SimMesh& mesh, const PointsX& frame, Color color, int subdiv,
                       SurfaceMesh& sm, Material& material)
{
    const Index W = mesh.width, H = mesh.height;
    if (W < 2 || H < 2) return; // no quads to build a surface from

    const bool wrap_i = mesh.wrap_i;
    ensure_surface_mesh(sm, W, H, subdiv, color, wrap_i);

    auto grid_index = [H](Index i, Index j) { return i * H + j; };
    auto pos_at      = [&](Index i, Index j) { return vertex_position(mesh, frame, grid_index(i, j)); };
    // Neighboring column, one step around i — wraps back to 0 past the last column when wrap_i
    // (closing the seam), otherwise just i+1 (never reached when wrap_i is false: the quad loops
    // below stop one short of W in that case).
    auto next_i = [W, wrap_i](Index i) { return wrap_i ? (i + 1) % W : i + 1; };

    const Index cells_i = wrap_i ? W : W - 1;

    // Pass 1: coarse per-vertex normals, averaged from every adjacent quad's face normal.
    // Winding (p11-p00) x (p10-p00) matches the triangle winding used in pass 2 below.
    std::vector<Vec3> vert_normal(W * H, Vec3::Zero());
    for (Index qi = 0; qi < cells_i; ++qi)
    {
        const Index qi1 = next_i(qi);
        for (Index qj = 0; qj < H - 1; ++qj)
        {
            const Vec3 p00 = pos_at(qi, qj), p10 = pos_at(qi1, qj);
            const Vec3 p11 = pos_at(qi1, qj + 1), p01 = pos_at(qi, qj + 1);
            const Vec3 n = (p11 - p00).cross(p10 - p00).normalized();
            vert_normal[grid_index(qi, qj)]   += n;
            vert_normal[grid_index(qi1, qj)]   += n;
            vert_normal[grid_index(qi1, qj + 1)] += n;
            vert_normal[grid_index(qi, qj + 1)] += n;
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

    for (Index qi = 0; qi < cells_i; ++qi)
    {
        const Index qi1 = next_i(qi);
        for (Index qj = 0; qj < H - 1; ++qj)
        {
            const Vec3 p00 = pos_at(qi, qj), p10 = pos_at(qi1, qj);
            const Vec3 p11 = pos_at(qi1, qj + 1), p01 = pos_at(qi, qj + 1);
            const Vec3 n00 = vert_normal[grid_index(qi, qj)], n10 = vert_normal[grid_index(qi1, qj)];
            const Vec3 n11 = vert_normal[grid_index(qi1, qj + 1)], n01 = vert_normal[grid_index(qi, qj + 1)];

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
    constexpr float kPinnedRadiusScale = 1.4f; // slightly bigger, so pinned anchors stand out

    for (Index vi = 0; vi < (Index)mesh.vertices.size(); ++vi)
    {
        const ParticleId dof = mesh.vertices[vi].dof;
        if (dof < 0) // pinned vertex
        {
            DrawSphereEx(to_raylib(vertex_position(mesh, frame, vi)),
                         particle_radius * kPinnedRadiusScale, 6, 6, YELLOW);
            continue;
        }
        const bool is_colliding = dof < (ParticleId)colliding.size() && colliding[dof];
        DrawSphereEx(to_raylib(vertex_position(mesh, frame, vi)), particle_radius, 6, 6,
                     is_colliding ? collide_color : color);
    }
}

// Builds a particle-dof -> in-contact bool mask from an already-fetched Contacts list (null yields
// an all-false mask). Used by the live-scene path (viewer_set_scene), where the caller already has
// the exact Contacts for the frame being shown.
std::vector<bool> mask_from_contacts(const Contacts* contacts, Index num_particles)
{
    std::vector<bool> mask(num_particles, false);
    if (contacts)
        for (const Contact& c : *contacts)
            mask[c.particle] = true;
    return mask;
}

// Same, but pulls the Contacts out of tape.contacts[tape_index] (bounds-checked). Contacts are
// detected against the frame's pre-step positions (see detect_contacts in diffpd.cpp), so they line
// up exactly with tape.positions[tape_index]. The final tape frame has no corresponding contacts
// entry (contacts.size() == positions.size() - 1), so it comes back with nothing marked.
std::vector<bool> colliding_mask(const Tape& tape, int tape_index, Index num_particles)
{
    const bool in_range = tape_index >= 0 && tape_index < (int)tape.contacts.size();
    return mask_from_contacts(in_range ? &tape.contacts[tape_index] : nullptr, num_particles);
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
// axis — or, via the center sphere, freely within the plane facing the camera (kGizmoFreeAxis).
// Sphere/Cylinder/Plane have one point (center/origin/origin); Capsule has two (p0, p1), each
// independently draggable. Config-screen only.
// ----------------------------------------------------------------------------------------------

constexpr Vec3  kGizmoAxisDirs[3]   = { Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1) };
const     Color kGizmoAxisColors[3] = { RED, GREEN, BLUE };
constexpr int   kGizmoFreeAxis      = 3; // hover_axis/drag_axis value for the free-move center sphere

// Gizmo drags move continuously but snap the result to a 0.1-unit grid, so a placement like 1.72
// lands on 1.7. Typed values in the config-screen text boxes are left exact (no snapping there).
constexpr Real kGizmoSnapStep = 0.1;

Real snap_to_grid(Real value, Real step)
{
    return std::round(value / step) * step;
}

Vec3 snap_to_grid(const Vec3& v, Real step)
{
    return Vec3(snap_to_grid(v.x(), step), snap_to_grid(v.y(), step), snap_to_grid(v.z(), step));
}

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

// Perpendicular distance from `point` to the mouse ray (treated as an infinite line) — the pick
// test for the free-move center sphere, which (unlike the axis arrows) has no direction to project
// onto, just a position.
Real dist_point_to_ray(const Vec3& point, const Ray& ray)
{
    const Vec3 ray_pos = from_raylib(ray.position);
    const Vec3 ray_dir = from_raylib(ray.direction).normalized();
    const Real t = (point - ray_pos).dot(ray_dir);
    return (ray_pos + t * ray_dir - point).norm();
}

// Ray/plane intersection, plane given by a point on it and its (unit) normal. `valid = false` when
// the ray is (near-)parallel to the plane, or the hit is behind the ray origin.
struct PlaneHit
{
    Vec3 point;
    bool valid;
};

PlaneHit ray_plane_hit(const Vec3& plane_point, const Vec3& plane_normal, const Ray& ray)
{
    const Vec3 ray_pos = from_raylib(ray.position);
    const Vec3 ray_dir = from_raylib(ray.direction).normalized();
    const Real denom    = plane_normal.dot(ray_dir);
    if (std::abs(denom) < 1e-6) return { Vec3::Zero(), false };

    const Real t = plane_normal.dot(plane_point - ray_pos) / denom;
    if (t < 0.0) return { Vec3::Zero(), false };
    return { ray_pos + t * ray_dir, true };
}

// The plane the free-move handle drags within: facing the camera, through `anchor`. Recomputed
// fresh every frame from the live camera rather than cached at drag-start — cheap, and correct
// even though in practice the camera can't move mid-drag anyway (grabbing a gizmo point already
// suppresses orbit-camera mouse capture for that same press, see viewer_show_config_screen).
Vec3 view_plane_normal(const Camera3D& camera)
{
    return from_raylib(Vector3Normalize(Vector3Subtract(camera.target, camera.position)));
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

        // Free-move sphere takes priority within its own pick radius: an axis line can pass much
        // closer to the ray than the sphere's *center* does (e.g. the ray nearly grazes an arrow's
        // shaft somewhere along its length), so comparing raw distances lets an arrow win even with
        // the cursor visibly over the sphere. Being inside the sphere's radius at all is decisive —
        // this point's arrows aren't considered further — rather than just another distance to beat.
        const Real sphere_dist = dist_point_to_ray(origins[p], ray);
        if (sphere_dist < pick_radius)
        {
            if (sphere_dist < best_dist)
            {
                best_dist = sphere_dist;
                best      = { p, kGizmoFreeAxis };
            }
            continue;
        }

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
    constexpr float kFreeRadiusFrac = 0.09f;

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

    // Free-move handle: dim orange at rest, brighter on hover, full orange while dragging — same
    // resting/hover/drag idiom as the arrows above, just applied to a color that's orange throughout
    // rather than only on drag (so it visually reads as "the free-move ball" even when idle, and
    // stays visually distinct from the arrows' own yellow drag-highlight).
    Color free_color = ColorBrightness(ORANGE, -0.35f);
    if (drag_axis == kGizmoFreeAxis)       free_color = ORANGE;
    else if (hover_axis == kGizmoFreeAxis) free_color = ColorBrightness(ORANGE, 0.4f);
    DrawSphere(origin_rl, length * kFreeRadiusFrac, free_color);

    // rlgl batches vertices and only actually issues the GL draw call at a flush; the depth
    // state in effect at *that* moment is what applies (not at rlVertex3f time). Force the flush
    // here, while depth test/write are still off, before restoring them for whatever draws next.
    rlDrawRenderBatchActive();

    rlEnableDepthMask();
    rlEnableDepthTest();
}

// Thin line through rotation_origin along the world axis the collider spins about — longer than
// the translate-gizmo arrows but much thinner, so it reads as an axis indicator rather than another
// draggable handle. Colored the same as the matching gizmo arrow (kGizmoAxisColors) for consistency.
void draw_rotation_axis_indicator(const Vec3& origin, RotationAxis axis, float gizmo_length, Color color)
{
    if (axis == RotationAxis::None) return;

    constexpr float kHalfLengthFactor = 1.6f;   // longer (each way) than the gizmo arrows
    constexpr float kRadiusFrac       = 0.008f; // thinner than the arrow shafts (kShaftRadiusFrac = 0.022f)

    const Vec3   dir      = rotation_axis_vector(axis);
    const float  half_len = gizmo_length * kHalfLengthFactor;
    const Vector3 p0 = to_raylib(origin - dir * half_len);
    const Vector3 p1 = to_raylib(origin + dir * half_len);

    rlDisableDepthTest();
    rlDisableDepthMask();
    DrawCylinderEx(p0, p1, gizmo_length * kRadiusFrac, gizmo_length * kRadiusFrac, 10, color);
    rlDrawRenderBatchActive();
    rlEnableDepthMask();
    rlEnableDepthTest();
}

// Persistent (across a single viewer_show_config_screen call) drag state for the translate gizmo.
// `point` identifies which of the collider's (up to 3, including rotation_origin) editable points
// is being dragged.
struct GizmoDragState
{
    int  point = -1;
    int  drag_axis = -1;
    Real drag_t_start = 0.0;              // axis drag only
    Vec3 drag_anchor_start = Vec3::Zero();
    Vec3 drag_plane_hit_start = Vec3::Zero(); // free (kGizmoFreeAxis) drag only
};

// Which point(s) of the current collider shape get a gizmo, and pointers to them (so dragging can
// write straight back into `c`). Returns the count (0-3) and fills `out[0..count-1]`. When rotation
// is enabled, rotation_origin is appended as an extra draggable point after the shape's own.
int collider_gizmo_anchors(Collider& c, Vec3* out[3])
{
    int n = 0;
    switch (c.type)
    {
        case ColliderType::Sphere:   out[0] = &c.sphere_center;   n = 1; break;
        case ColliderType::Cylinder: out[0] = &c.cylinder_origin; n = 1; break;
        case ColliderType::Plane:    out[0] = &c.plane_origin;    n = 1; break;
        case ColliderType::Capsule:  out[0] = &c.capsule_p0; out[1] = &c.capsule_p1; n = 2; break;
        case ColliderType::None:     n = 0; break;
    }
    if (c.rotation_axis != RotationAxis::None) out[n++] = &c.rotation_origin;
    return n;
}

// Gizmo screen-size tuning: arrow length is this fraction of the camera distance (constant on-screen
// size), floored at kGizmoMinLength so it never collapses when the camera sits right on a point.
constexpr float kGizmoScreenScale = 0.15f;
constexpr float kGizmoMinLength   = 0.05f;

float gizmo_length_for(const Vec3& origin, const Camera3D& camera)
{
    const float dist = Vector3Distance(camera.position, to_raylib(origin));
    return std::max(kGizmoMinLength, dist * kGizmoScreenScale);
}

// Per-frame result of updating the collider gizmos: where to draw each editable point's arrows and
// which (if any) axis is hovered. Drag state persists separately in the caller's GizmoDragState.
struct GizmoFrame
{
    int   n_points   = 0;
    Vec3  origins[3];
    float lengths[3] = { 0.0f, 0.0f, 0.0f };
    int   hover_point = -1;
    int   hover_axis  = -1;
};

// Runs one frame of the collider-gizmo interaction: computes each editable point's arrow origin and
// on-screen-constant length, hover-picks (only when not already dragging and the mouse is over the
// 3D viewport), and processes press/drag/release — writing dragged positions straight back into
// `collider` via collider_gizmo_anchors. Returns what draw_translate_gizmo needs for this frame.
GizmoFrame update_collider_gizmos(Collider& collider, GizmoDragState& state,
                                  const Camera3D& camera, const Ray& mouse_ray, bool over_viewport)
{
    GizmoFrame frame;

    Vec3* anchor_ptrs[3] = { nullptr, nullptr, nullptr };
    frame.n_points = collider_gizmo_anchors(collider, anchor_ptrs);

    // Collider shape changed under us (now has fewer points than the one being dragged): drop drag.
    if (state.point >= frame.n_points) { state.point = -1; state.drag_axis = -1; }

    for (int p = 0; p < frame.n_points; ++p)
    {
        frame.origins[p] = *anchor_ptrs[p];
        frame.lengths[p] = gizmo_length_for(frame.origins[p], camera);
    }

    if (state.point == -1 && over_viewport && frame.n_points > 0)
    {
        const GizmoPick pick = pick_gizmo_multi(frame.origins, frame.lengths, frame.n_points, mouse_ray);
        frame.hover_point = pick.point;
        frame.hover_axis  = pick.axis;
    }

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && frame.hover_axis != -1)
    {
        state.point             = frame.hover_point;
        state.drag_axis         = frame.hover_axis;
        state.drag_anchor_start = frame.origins[frame.hover_point];
        if (frame.hover_axis == kGizmoFreeAxis)
        {
            const PlaneHit hit = ray_plane_hit(state.drag_anchor_start, view_plane_normal(camera), mouse_ray);
            state.drag_plane_hit_start = hit.valid ? hit.point : state.drag_anchor_start;
        }
        else
        {
            state.drag_t_start = closest_axis_ray(state.drag_anchor_start,
                                                    kGizmoAxisDirs[frame.hover_axis], mouse_ray).s;
        }
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
    {
        state.point     = -1;
        state.drag_axis = -1;
    }

    if (state.point != -1 && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
    {
        if (state.drag_axis == kGizmoFreeAxis)
        {
            const PlaneHit hit = ray_plane_hit(state.drag_anchor_start, view_plane_normal(camera), mouse_ray);
            if (hit.valid)
                *anchor_ptrs[state.point] = snap_to_grid(state.drag_anchor_start
                                           + (hit.point - state.drag_plane_hit_start), kGizmoSnapStep);
        }
        else
        {
            const Vec3&    axis_dir = kGizmoAxisDirs[state.drag_axis];
            const AxisPick pick     = closest_axis_ray(state.drag_anchor_start, axis_dir, mouse_ray);
            if (pick.valid)
                *anchor_ptrs[state.point] = snap_to_grid(state.drag_anchor_start
                                           + axis_dir * (pick.s - state.drag_t_start), kGizmoSnapStep);
        }

        // origins/lengths were captured before this update; resync the dragged point so the arrows
        // drawn this frame track the mouse instead of lagging by one.
        frame.origins[state.point] = *anchor_ptrs[state.point];
        frame.lengths[state.point] = gizmo_length_for(frame.origins[state.point], camera);
    }

    return frame;
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

// Small translucent panel showing the run's result (loss, dphi/dx0, dphi/dv0, dphi/dk), anchored
// top-left just below the frame counter/FPS readout. Static for the whole playback — none of these
// change frame to frame, only the display formatting. Returns the panel's height in pixels so
// callers can stack further panels directly underneath without a hand-maintained offset constant.
int draw_gradient_panel(int screen_x, int screen_y, const GradientSummary& grad)
{
    char line_loss[64], line_x[96], line_v[96], line_k[64];
    std::snprintf(line_loss, sizeof(line_loss), "loss     =  % .6g", grad.loss);
    std::snprintf(line_x, sizeof(line_x), "dphi/dx0 = (% .6g, % .6g, % .6g)",
                  grad.dphi_dx0.x(), grad.dphi_dx0.y(), grad.dphi_dx0.z());
    std::snprintf(line_v, sizeof(line_v), "dphi/dv0 = (% .6g, % .6g, % .6g)",
                  grad.dphi_dv0.x(), grad.dphi_dv0.y(), grad.dphi_dv0.z());
    std::snprintf(line_k, sizeof(line_k), "dphi/dk  =  % .6g", grad.dphi_dk);

    const char* kTitle = "Loss / Gradients";

    constexpr int kFontSize          = 16;
    constexpr int kTitleSize         = 16;
    constexpr int kLineHeight        = 20;
    constexpr int kPadding           = 10;
    constexpr int kTitleGap          = 4;  // extra space between the title and the first data line
    constexpr int kHighlightSize     = 24; // dphi/dk is the quantity actually being optimized —
    constexpr int kHighlightLineHeight = 30; // bigger + yellow so it stands out from the rest

    struct Line { const char* text; int font_size; int line_height; Color color; };
    const Line lines[4] = {
        { line_loss, kFontSize, kLineHeight, RAYWHITE },
        { line_x,    kFontSize, kLineHeight, RAYWHITE },
        { line_v,    kFontSize, kLineHeight, RAYWHITE },
        { line_k,    kHighlightSize, kHighlightLineHeight, YELLOW },
    };

    int max_width = MeasureText(kTitle, kTitleSize);
    int content_h = 0;
    for (const Line& l : lines)
    {
        max_width  = std::max(max_width, MeasureText(l.text, l.font_size));
        content_h += l.line_height;
    }

    const int box_w = max_width + 2 * kPadding;
    const int box_h = kLineHeight + kTitleGap + content_h + 2 * kPadding;

    DrawRectangle(screen_x, screen_y, box_w, box_h, { 0, 0, 0, 140 });
    DrawRectangleLines(screen_x, screen_y, box_w, box_h, { 255, 255, 255, 60 });

    int y = screen_y + kPadding;
    DrawText(kTitle, screen_x + kPadding, y, kTitleSize, YELLOW);
    y += kLineHeight + kTitleGap;
    for (const Line& l : lines)
    {
        DrawText(l.text, screen_x + kPadding, y, l.font_size, l.color);
        y += l.line_height;
    }

    return box_h;
}

// Simple line-graph panel: a title, a plain white L-axis, a colored polyline through `data`
// (linearly min/max-normalized to the plot area — no charting library, just DrawLineEx between
// consecutive points, the same "hand-rolled primitives" style as draw_help_box/draw_gradient_panel
// above), and a red vertical marker at `progress_fraction` (0..1 across the plot width) showing
// where the current playback frame sits in the recorded trajectory.
void draw_residual_graph(Rectangle bounds, const char* title, const std::vector<Real>& data,
                          Color line_color, double progress_fraction)
{
    constexpr int   kTitleSize   = 16;
    constexpr int   kAxisSize    = 12; // y-axis order-of-magnitude labels
    constexpr float kPadding     = 8.0f;
    constexpr float kTickGap     = 3.0f; // tick mark length, left of the axis line
    constexpr Real  kEpsFloor    = 1e-12; // residuals are norms (>= 0); floors log10(0) to a finite value
    constexpr int   kMaxTicks    = 4;     // caps label crowding in this small a panel

    DrawRectangle((int)bounds.x, (int)bounds.y, (int)bounds.width, (int)bounds.height, { 0, 0, 0, 140 });
    DrawRectangleLines((int)bounds.x, (int)bounds.y, (int)bounds.width, (int)bounds.height, { 255, 255, 255, 60 });
    DrawText(title, (int)(bounds.x + kPadding), (int)(bounds.y + kPadding - 2.0f), kTitleSize, YELLOW);

    // Residuals routinely span several decades over a trajectory, so the y-axis is log10-scaled
    // (order of magnitude per gridline) rather than linear — a linear scale would flatten
    // everything near zero except for the rare large spike. Reserve a left margin for the "1eN"
    // labels; "1e-10" is the widest case this app's convergence thresholds ever produce.
    const float y_label_w = (float)MeasureText("1e-10", kAxisSize) + kTickGap;

    const float plot_x = bounds.x + kPadding + y_label_w;
    const float plot_y = bounds.y + kPadding + (float)kTitleSize + 4.0f;
    const float plot_w = bounds.x + bounds.width - kPadding - plot_x;
    const float plot_h = bounds.y + bounds.height - kPadding - plot_y;
    if (plot_w <= 0.0f || plot_h <= 0.0f) return;

    // Axes: a plain L, like the reference sketch, plus the order-of-magnitude tick labels below.
    DrawLine((int)plot_x, (int)plot_y, (int)plot_x, (int)(plot_y + plot_h), RAYWHITE);
    DrawLine((int)plot_x, (int)(plot_y + plot_h), (int)(plot_x + plot_w), (int)(plot_y + plot_h), RAYWHITE);

    if (data.size() >= 2)
    {
        // log10 domain, rounded out to whole decades so tick labels land on round exponents
        // (e.g. 1e-4, 1e-6) rather than the data's exact (and visually meaningless) min/max.
        Real lo_log = data[0], hi_log = data[0];
        for (Real v : data)
        {
            const Real log_v = std::log10(std::max(v, kEpsFloor));
            lo_log = std::min(lo_log, log_v);
            hi_log = std::max(hi_log, log_v);
        }
        int lo_exp = (int)std::floor(lo_log);
        int hi_exp = (int)std::ceil(hi_log);
        if (hi_exp <= lo_exp) hi_exp = lo_exp + 1; // degenerate case: every value the same order of magnitude
        const Real exp_range = (Real)(hi_exp - lo_exp);

        auto frac_for_exp = [&](int e) { return (float)((e - lo_exp) / exp_range); };

        auto point_at = [&](size_t i) -> Vector2
        {
            const float fx = (float)i / (float)(data.size() - 1);
            const Real log_v = std::log10(std::max(data[i], kEpsFloor));
            const float fv = (float)((log_v - lo_exp) / exp_range); // 0 at lo_exp, 1 at hi_exp
            return { plot_x + fx * plot_w, plot_y + plot_h - fv * plot_h };
        };

        for (size_t i = 0; i + 1 < data.size(); ++i)
            DrawLineEx(point_at(i), point_at(i + 1), 2.0f, line_color);

        const int step = std::max(1, (int)std::ceil((double)(hi_exp - lo_exp) / kMaxTicks));
        for (int e = lo_exp; e <= hi_exp; e += step)
        {
            const float ty = plot_y + plot_h - frac_for_exp(e) * plot_h;
            char label[16];
            std::snprintf(label, sizeof(label), "1e%d", e);
            DrawLine((int)(plot_x - kTickGap), (int)ty, (int)plot_x, (int)ty, RAYWHITE);
            DrawText(label, (int)(bounds.x + kPadding), (int)(ty - kAxisSize * 0.5f), kAxisSize, GRAY);
        }
    }

    const float marker_x = plot_x + (float)std::clamp(progress_fraction, 0.0, 1.0) * plot_w;
    DrawLine((int)marker_x, (int)plot_y, (int)marker_x, (int)(plot_y + plot_h), RED);
}

// ----------------------------------------------------------------------------------------------
// Screen recorder: an always-on-screen Record button + filename box (bottom-right corner, drawn
// on every screen — live view, config screen, playback) that captures the whole window to a video.
//
// Approach: while recording, every frame is grabbed via raylib's LoadImageFromScreen() (a
// glReadPixels of the current backbuffer) and its raw RGBA bytes are appended, uncompressed, to a
// single flat binary file. An earlier version PNG-compressed and wrote out each frame individually
// (mirroring diffpd.cpp's .obj export pattern) — but PNG's zlib deflate pass is expensive enough
// (a multi-megabyte image, every single rendered frame) to visibly tank the frame rate while
// recording. Skipping compression entirely and just memcpy-speed-appending raw bytes removes that
// cost; the one-time expense of decoding+encoding the whole raw stream is deferred to Stop, where
// ffmpeg reads it back as a `rawvideo` source (`-f rawvideo -pix_fmt rgba`) and encodes the .mp4,
// then the (potentially large — width*height*4 bytes per frame) raw file is deleted. This avoids
// linking any video-encoding library: ffmpeg.exe just needs to be reachable on PATH. If it isn't,
// encoding fails but the raw file survives (nothing is lost, and the WARNING below says where).
// ----------------------------------------------------------------------------------------------

namespace fs = std::filesystem;

// Injected by CMake (mirrors ANIM_DIR_DEFAULT) as an absolute path so recordings land in the same
// place regardless of the executable's working directory; falls back to a relative path if built
// outside that CMake target.
#ifndef RECORDINGS_DIR_DEFAULT
#define RECORDINGS_DIR_DEFAULT "../recordings"
#endif
constexpr const char* kRecordingsDir = RECORDINGS_DIR_DEFAULT;

struct Recorder
{
    bool          recording   = false;
    char          name_buf[128] = "recording";
    bool          name_edit   = false;
    int           frame_index = 0;
    double        start_time  = 0.0;
    int           width       = 0;    // capture resolution, locked in at record-start
    int           height      = 0;
    bool          warned_resize = false; // only warn once per recording if the window resizes mid-capture
    std::ofstream raw_stream;          // append-only sink for raw RGBA frames
    fs::path      raw_path;
    std::string   output_name;        // sanitized name captured at record-start
};

Recorder g_recorder;

// Replaces anything that isn't alnum/-/_ with '_' so the name is always a valid filename
// component; falls back to "recording" if that leaves nothing.
std::string sanitize_recording_name(const char* raw)
{
    std::string s(raw);
    for (char& c : s)
        if (!std::isalnum((unsigned char)c) && c != '-' && c != '_') c = '_';
    while (!s.empty() && s.front() == '_') s.erase(s.begin());
    while (!s.empty() && s.back()  == '_') s.pop_back();
    return s.empty() ? "recording" : s;
}

void start_recording(Recorder& rec)
{
    rec.output_name = sanitize_recording_name(rec.name_buf);

    std::error_code ec;
    fs::create_directories(kRecordingsDir, ec);

    rec.raw_path = fs::path(kRecordingsDir) / (rec.output_name + "_raw_tmp.rgba");
    rec.raw_stream.open(rec.raw_path, std::ios::binary | std::ios::trunc);

    // Must match what LoadImageFromScreen() actually returns (GetRenderWidth/Height, i.e. the
    // framebuffer size) rather than GetScreenWidth/Height, which differ under DPI scaling — using
    // the wrong pair here would make every captured frame look "resized" and get dropped.
    rec.width         = GetRenderWidth();
    rec.height        = GetRenderHeight();
    rec.frame_index   = 0;
    rec.start_time    = GetTime();
    rec.warned_resize = false;
    rec.recording     = true;
}

void capture_recording_frame(Recorder& rec)
{
    // LoadImageFromScreen() (via rlReadScreenPixels) calls glReadPixels directly — it does NOT
    // flush raylib's batched 2D draw queue first. EndDrawing() itself calls
    // rlDrawRenderBatchActive() before swapping buffers, but this capture runs earlier in the
    // frame (right before EndDrawing), so anything drawn since the last implicit flush (e.g.
    // EndMode3D's own flush) — the help box, gradient panel, residual graphs, buttons, the
    // recorder overlay itself — is still sitting unrasterized and would be invisible to the
    // readback without forcing this flush first.
    rlDrawRenderBatchActive();
    Image img = LoadImageFromScreen();

    // rawvideo has no per-frame header, so every frame must match the resolution ffmpeg is told
    // to expect — skip (rather than corrupt the whole stream) if the window was resized mid-capture.
    if (img.width != rec.width || img.height != rec.height)
    {
        if (!rec.warned_resize)
        {
            WARNING("capture_recording_frame: window resized mid-recording ("
                     << rec.width << "x" << rec.height << " -> " << img.width << "x" << img.height
                     << ") — dropping frames until it matches again");
            rec.warned_resize = true;
        }
        UnloadImage(img);
        return;
    }

    const int data_size = GetPixelDataSize(img.width, img.height, img.format);
    rec.raw_stream.write(reinterpret_cast<const char*>(img.data), data_size);
    UnloadImage(img);
    ++rec.frame_index;
}

// Picks a destination path that doesn't clobber an existing file: name.mp4, name_1.mp4, name_2.mp4, ...
fs::path unique_output_path(const std::string& stem)
{
    fs::path candidate = fs::path(kRecordingsDir) / (stem + ".mp4");
    for (int suffix = 1; fs::exists(candidate); ++suffix)
        candidate = fs::path(kRecordingsDir) / (stem + "_" + std::to_string(suffix) + ".mp4");
    return candidate;
}

void stop_recording(Recorder& rec)
{
    rec.recording = false;
    rec.raw_stream.close();

    std::error_code ec;
    if (rec.frame_index == 0) { fs::remove(rec.raw_path, ec); return; }

    // Encode at the actual observed capture rate rather than assuming SetTargetFPS's cap, so
    // playback speed matches real elapsed time even if the app dipped below 60fps while recording.
    const double elapsed = GetTime() - rec.start_time;
    const int    fps     = (elapsed > 0.0) ? std::clamp((int)std::lround(rec.frame_index / elapsed), 1, 240) : 60;

    const fs::path out_path = unique_output_path(rec.output_name);

    // PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 (LoadImageFromScreen's format) is 4 bytes/pixel, byte order
    // R,G,B,A — matches ffmpeg's "rgba" pix_fmt exactly, so no channel swizzling is needed here.
    //
    // -crf 16 -preset slow: source is a lossless raw capture, so the default CRF 23 (medium
    // preset) would throw away a lot of that fidelity for no benefit here — this is a one-time,
    // non-realtime encode, so trading encode time for quality is free. CRF 16 is close to visually
    // lossless for x264; drop to ~12-14 for still higher quality (much bigger files), or push CRF
    // up (~20-23) if file size matters more than sharpness.
    char cmd[2048];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -f rawvideo -pix_fmt rgba -s %dx%d -framerate %d -i \"%s\" "
                  "-c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p \"%s\" >NUL 2>NUL",
                  rec.width, rec.height, fps, rec.raw_path.string().c_str(), out_path.string().c_str());

    const int result = std::system(cmd);

    if (result == 0)
    {
        fs::remove(rec.raw_path, ec);
    }
    else
    {
        WARNING("stop_recording: ffmpeg encode failed (is ffmpeg.exe on PATH?) — raw frames kept at "
                 << rec.raw_path.string());
    }
}

// Fixed bottom-right panel: filename box + Record/Stop button. Drawn (and interacted with) every
// frame on every screen so it's always available, regardless of what else is on screen.
void draw_recorder_overlay(int screen_width, int screen_height)
{
    constexpr float kBoxW = 200.0f, kBoxH = 30.0f, kBtnW = 90.0f, kPad = 8.0f;
    const float panel_w = kBoxW + kBtnW + 3.0f * kPad;
    const float panel_h = kBoxH + 2.0f * kPad;
    const float x = (float)screen_width  - panel_w - 10.0f;
    const float y = (float)screen_height - panel_h - 10.0f;

    DrawRectangle((int)x, (int)y, (int)panel_w, (int)panel_h, { 0, 0, 0, 140 });
    DrawRectangleLines((int)x, (int)y, (int)panel_w, (int)panel_h, { 255, 255, 255, 60 });

    const Rectangle box_rect = { x + kPad, y + kPad, kBoxW, kBoxH };
    const Rectangle btn_rect = { x + 2.0f * kPad + kBoxW, y + kPad, kBtnW, kBoxH };

    if (g_recorder.recording) GuiDisable(); // can't rename mid-recording
    if (GuiTextBox(box_rect, g_recorder.name_buf, (int)sizeof(g_recorder.name_buf), g_recorder.name_edit))
        g_recorder.name_edit = !g_recorder.name_edit;
    if (g_recorder.recording) GuiEnable();

    // Snapshot once: GuiButton's click handler below can flip g_recorder.recording mid-function
    // (Stop -> false), so re-reading g_recorder.recording *after* the click to decide what to
    // restore would see the post-click value. Styling both branches unconditionally (rather than
    // only the "recording" one, as an earlier version did) and always restoring afterward sidesteps
    // that class of bug entirely — apply and restore no longer need to agree on which state fired.
    const bool was_recording = g_recorder.recording;

    // Idle: red "record" button (the universal record-button color). Recording: white square,
    // conventional "stop" iconography — the button itself carries the state, no text needed.
    const Color btn_color  = was_recording ? RAYWHITE : RED;
    const Color text_color = was_recording ? BLACK    : RAYWHITE;

    const int prev_base_normal    = GuiGetStyle(BUTTON, BASE_COLOR_NORMAL);
    const int prev_base_focused   = GuiGetStyle(BUTTON, BASE_COLOR_FOCUSED);
    const int prev_base_pressed   = GuiGetStyle(BUTTON, BASE_COLOR_PRESSED);
    const int prev_border_normal  = GuiGetStyle(BUTTON, BORDER_COLOR_NORMAL);
    const int prev_border_focused = GuiGetStyle(BUTTON, BORDER_COLOR_FOCUSED);
    const int prev_border_pressed = GuiGetStyle(BUTTON, BORDER_COLOR_PRESSED);
    const int prev_text_normal    = GuiGetStyle(BUTTON, TEXT_COLOR_NORMAL);
    const int prev_text_focused   = GuiGetStyle(BUTTON, TEXT_COLOR_FOCUSED);
    const int prev_text_pressed   = GuiGetStyle(BUTTON, TEXT_COLOR_PRESSED);
    const int prev_text_size      = GuiGetStyle(DEFAULT, TEXT_SIZE);

    const Color focused_color = ColorBrightness(btn_color, was_recording ? -0.15f : 0.2f);
    const Color pressed_color = ColorBrightness(btn_color, was_recording ? -0.3f  : -0.2f);
    GuiSetStyle(BUTTON, BASE_COLOR_NORMAL,    ColorToInt(btn_color));
    GuiSetStyle(BUTTON, BASE_COLOR_FOCUSED,   ColorToInt(focused_color));
    GuiSetStyle(BUTTON, BASE_COLOR_PRESSED,   ColorToInt(pressed_color));
    GuiSetStyle(BUTTON, BORDER_COLOR_NORMAL,  ColorToInt(btn_color));
    GuiSetStyle(BUTTON, BORDER_COLOR_FOCUSED, ColorToInt(focused_color));
    GuiSetStyle(BUTTON, BORDER_COLOR_PRESSED, ColorToInt(pressed_color));
    GuiSetStyle(BUTTON, TEXT_COLOR_NORMAL,    ColorToInt(text_color));
    GuiSetStyle(BUTTON, TEXT_COLOR_FOCUSED,   ColorToInt(text_color));
    GuiSetStyle(BUTTON, TEXT_COLOR_PRESSED,   ColorToInt(text_color));
    GuiSetStyle(DEFAULT, TEXT_SIZE, 20);

    // Plain ASCII labels: raylib's default font only covers the basic ASCII range, so the Unicode
    // "●"/"■" glyphs an earlier version used here rendered as "?" (missing-glyph fallback).
    if (GuiButton(btn_rect, was_recording ? "STOP" : "REC"))
    {
        if (was_recording) stop_recording(g_recorder);
        else                start_recording(g_recorder);
    }

    GuiSetStyle(BUTTON, BASE_COLOR_NORMAL,    prev_base_normal);
    GuiSetStyle(BUTTON, BASE_COLOR_FOCUSED,   prev_base_focused);
    GuiSetStyle(BUTTON, BASE_COLOR_PRESSED,   prev_base_pressed);
    GuiSetStyle(BUTTON, BORDER_COLOR_NORMAL,  prev_border_normal);
    GuiSetStyle(BUTTON, BORDER_COLOR_FOCUSED, prev_border_focused);
    GuiSetStyle(BUTTON, BORDER_COLOR_PRESSED, prev_border_pressed);
    GuiSetStyle(BUTTON, TEXT_COLOR_NORMAL,    prev_text_normal);
    GuiSetStyle(BUTTON, TEXT_COLOR_FOCUSED,   prev_text_focused);
    GuiSetStyle(BUTTON, TEXT_COLOR_PRESSED,   prev_text_pressed);
    GuiSetStyle(DEFAULT, TEXT_SIZE, prev_text_size);
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

// Same tube+end-cap shape as DrawCylinderEx(p0, p1, radius, radius, sides, color), but with correct
// per-vertex normals (radial on the tube, axis-aligned on the two end caps). raylib's own
// DrawCylinderEx (and DrawCapsule, below) never call rlNormal3f, so under our shaded material every
// vertex just inherits whatever normal happened to be left over from a previous draw call, so the
// whole shape renders as one flat, unlit-looking tone instead of a properly lit curved surface
// (visible on the Cylinder/Capsule colliders) — unlike DrawSphereEx, which does emit correct
// normals, hence the Sphere collider already shades correctly. Vertex order/winding mirrors
// DrawCylinderEx exactly so backface culling still sees the same front faces.
void draw_cylinder_shaded(Vector3 p0, Vector3 p1, float radius, int sides, Color color)
{
    const Vector3 direction = Vector3Subtract(p1, p0);
    const Vector3 axis      = Vector3Normalize(direction);
    const Vector3 b1        = Vector3Normalize(Vector3Perpendicular(direction));
    const Vector3 b2        = Vector3Normalize(Vector3CrossProduct(b1, direction));
    const float   step      = (2.0f * PI) / sides;

    rlBegin(RL_TRIANGLES);
    rlColor4ub(color.r, color.g, color.b, color.a);
    for (int i = 0; i < sides; ++i)
    {
        const Vector3 n1 = Vector3Add(Vector3Scale(b1, sinf(step * i)),       Vector3Scale(b2, cosf(step * i)));
        const Vector3 n2 = Vector3Add(Vector3Scale(b1, sinf(step * (i + 1))), Vector3Scale(b2, cosf(step * (i + 1))));
        const Vector3 w1 = Vector3Add(p0, Vector3Scale(n1, radius));
        const Vector3 w2 = Vector3Add(p0, Vector3Scale(n2, radius));
        const Vector3 w3 = Vector3Add(p1, Vector3Scale(n1, radius));
        const Vector3 w4 = Vector3Add(p1, Vector3Scale(n2, radius));

        // start cap fan
        rlNormal3f(-axis.x, -axis.y, -axis.z);
        rlVertex3f(p0.x, p0.y, p0.z);
        rlVertex3f(w2.x, w2.y, w2.z);
        rlVertex3f(w1.x, w1.y, w1.z);

        // side quad (two triangles), radial normal per vertex
        rlNormal3f(n1.x, n1.y, n1.z); rlVertex3f(w1.x, w1.y, w1.z);
        rlNormal3f(n2.x, n2.y, n2.z); rlVertex3f(w2.x, w2.y, w2.z);
        rlNormal3f(n1.x, n1.y, n1.z); rlVertex3f(w3.x, w3.y, w3.z);

        rlNormal3f(n2.x, n2.y, n2.z); rlVertex3f(w2.x, w2.y, w2.z);
        rlNormal3f(n2.x, n2.y, n2.z); rlVertex3f(w4.x, w4.y, w4.z);
        rlNormal3f(n1.x, n1.y, n1.z); rlVertex3f(w3.x, w3.y, w3.z);

        // end cap fan
        rlNormal3f(axis.x, axis.y, axis.z);
        rlVertex3f(p1.x, p1.y, p1.z);
        rlVertex3f(w3.x, w3.y, w3.z);
        rlVertex3f(w4.x, w4.y, w4.z);
    }
    rlEnd();
}

// Same two-hemisphere-cap + cylindrical-body shape as DrawCapsule(p0, p1, radius, slices, rings,
// color), but with correct per-vertex normals — see draw_cylinder_shaded's comment above for why
// that's needed. Each surface point here lies at `radius` from either a cap center (hemispheres) or
// the central axis (cylindrical body) along a unit direction vector — that same direction vector,
// already computed to place the vertex, *is* its outward normal, so no separate computation is
// needed. Vertex order/winding mirrors DrawCapsule exactly so backface culling still sees the same
// front faces.
void draw_capsule_shaded(Vector3 p0, Vector3 p1, float radius, int slices, int rings, Color color)
{
    const Vector3 direction = Vector3Subtract(p1, p0);
    Vector3       b0        = (Vector3LengthSqr(direction) > 1e-12f) ? Vector3Normalize(direction) : Vector3{ 0.0f, 1.0f, 0.0f };
    const Vector3 b1        = Vector3Normalize(Vector3Perpendicular(direction));
    const Vector3 b2        = Vector3Normalize(Vector3CrossProduct(b1, direction));

    const float sliceStep = (2.0f * PI) / slices;
    const float ringStep  = (PI * 0.5f) / rings;

    // Direction (== outward normal) of the hemisphere-cap vertex at ring `i` (0 = equator, rings =
    // pole), slice `j`, relative to whichever end's own `axis` (+b0 for the end cap, -b0 for the
    // start cap — see the `axis` flip below).
    auto cap_dir = [&](const Vector3& axis, int i, int j)
    {
        const float ring_s = sinf(ringStep * i), ring_c = cosf(ringStep * i);
        const float sl_s   = sinf(sliceStep * j), sl_c  = cosf(sliceStep * j);
        return Vector3Add(Vector3Scale(axis, ring_s),
               Vector3Add(Vector3Scale(b1, sl_s * ring_c), Vector3Scale(b2, sl_c * ring_c)));
    };

    rlBegin(RL_TRIANGLES);
    rlColor4ub(color.r, color.g, color.b, color.a);

    Vector3 capCenter = p1;
    Vector3 axis      = b0;
    for (int c = 0; c < 2; ++c)
    {
        for (int i = 0; i < rings; ++i)
        {
            for (int j = 0; j < slices; ++j)
            {
                const Vector3 d1 = cap_dir(axis, i,     j), d2 = cap_dir(axis, i,     j + 1);
                const Vector3 d3 = cap_dir(axis, i + 1, j), d4 = cap_dir(axis, i + 1, j + 1);
                const Vector3 w1 = Vector3Add(capCenter, Vector3Scale(d1, radius));
                const Vector3 w2 = Vector3Add(capCenter, Vector3Scale(d2, radius));
                const Vector3 w3 = Vector3Add(capCenter, Vector3Scale(d3, radius));
                const Vector3 w4 = Vector3Add(capCenter, Vector3Scale(d4, radius));

                if (c == 0)
                {
                    rlNormal3f(d1.x, d1.y, d1.z); rlVertex3f(w1.x, w1.y, w1.z);
                    rlNormal3f(d2.x, d2.y, d2.z); rlVertex3f(w2.x, w2.y, w2.z);
                    rlNormal3f(d3.x, d3.y, d3.z); rlVertex3f(w3.x, w3.y, w3.z);

                    rlNormal3f(d2.x, d2.y, d2.z); rlVertex3f(w2.x, w2.y, w2.z);
                    rlNormal3f(d4.x, d4.y, d4.z); rlVertex3f(w4.x, w4.y, w4.z);
                    rlNormal3f(d3.x, d3.y, d3.z); rlVertex3f(w3.x, w3.y, w3.z);
                }
                else
                {
                    rlNormal3f(d1.x, d1.y, d1.z); rlVertex3f(w1.x, w1.y, w1.z);
                    rlNormal3f(d3.x, d3.y, d3.z); rlVertex3f(w3.x, w3.y, w3.z);
                    rlNormal3f(d2.x, d2.y, d2.z); rlVertex3f(w2.x, w2.y, w2.z);

                    rlNormal3f(d2.x, d2.y, d2.z); rlVertex3f(w2.x, w2.y, w2.z);
                    rlNormal3f(d3.x, d3.y, d3.z); rlVertex3f(w3.x, w3.y, w3.z);
                    rlNormal3f(d4.x, d4.y, d4.z); rlVertex3f(w4.x, w4.y, w4.z);
                }
            }
        }
        capCenter = p0;
        axis      = Vector3Scale(b0, -1.0f);
    }

    // cylindrical middle
    for (int j = 0; j < slices; ++j)
    {
        const Vector3 n1 = Vector3Add(Vector3Scale(b1, sinf(sliceStep * j)),       Vector3Scale(b2, cosf(sliceStep * j)));
        const Vector3 n2 = Vector3Add(Vector3Scale(b1, sinf(sliceStep * (j + 1))), Vector3Scale(b2, cosf(sliceStep * (j + 1))));
        const Vector3 w1 = Vector3Add(p0, Vector3Scale(n1, radius));
        const Vector3 w2 = Vector3Add(p0, Vector3Scale(n2, radius));
        const Vector3 w3 = Vector3Add(p1, Vector3Scale(n1, radius));
        const Vector3 w4 = Vector3Add(p1, Vector3Scale(n2, radius));

        rlNormal3f(n1.x, n1.y, n1.z); rlVertex3f(w1.x, w1.y, w1.z);
        rlNormal3f(n2.x, n2.y, n2.z); rlVertex3f(w2.x, w2.y, w2.z);
        rlNormal3f(n1.x, n1.y, n1.z); rlVertex3f(w3.x, w3.y, w3.z);

        rlNormal3f(n2.x, n2.y, n2.z); rlVertex3f(w2.x, w2.y, w2.z);
        rlNormal3f(n2.x, n2.y, n2.z); rlVertex3f(w4.x, w4.y, w4.z);
        rlNormal3f(n1.x, n1.y, n1.z); rlVertex3f(w3.x, w3.y, w3.z);
    }

    rlEnd();
}

// Collider geometry is unbounded for Cylinder (infinite radius line) and Plane (infinite
// sheet); both are drawn with a fixed finite visual extent purely for display.
void draw_collider(const Collider& collider, float time, Color color)
{
    constexpr float kCylinderHalfLength = 10.0f;
    constexpr float kPlaneHalfExtent    = 25.0f;
    constexpr float kRadiusReduction    = 0.05f;

    const ColliderPose pose = collider_pose_at(collider, (Real)time);

    switch (collider.type)
    {
        case ColliderType::Sphere:
        {
            const float radius = (float)(collider.sphere_radius * (1.0 - kRadiusReduction));
            DrawSphereEx(to_raylib(pose.sphere_center), radius, 24, 24, color);
            break;
        }
        case ColliderType::Cylinder:
        {
            const Vec3 axis    = pose.cylinder_axis.normalized();
            const Vec3 p0      = pose.cylinder_origin - axis * kCylinderHalfLength;
            const Vec3 p1      = pose.cylinder_origin + axis * kCylinderHalfLength;
            const float radius = (float)(collider.cylinder_radius * (1.0 - kRadiusReduction));
            draw_cylinder_shaded(to_raylib(p0), to_raylib(p1), radius, 24, color);
            break;
        }
        case ColliderType::Plane:
        {
            draw_plane_oriented(to_raylib(pose.plane_origin), { 2.0f * kPlaneHalfExtent, 2.0f * kPlaneHalfExtent },
                                 to_raylib(pose.plane_normal), color);
            break;
        }
        case ColliderType::Capsule:
        {
            const float radius = (float)(collider.capsule_radius * (1.0 - kRadiusReduction));
            draw_capsule_shaded(to_raylib(pose.capsule_p0), to_raylib(pose.capsule_p1), radius, 24, 16, color);
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
constexpr Color kColliderColor          = { 140, 150, 165, 255 }; // opaque, slightly bluish gray
constexpr Color kColliderHighlightColor = { 210, 215, 222, 255 }; // lighter gray: the selected collider on the config screen
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
    std::vector<Collider> colliders;
    Real           collider_time = 0.0;
    std::string    status_text;
};

ViewerState g_viewer;

// One cloth trajectory's render inputs for draw_scene_layers.
struct SceneLayer
{
    const SimMesh*    mesh  = nullptr;
    const PointsX*    frame = nullptr;
    Color             color;
    Color             collide_color;
    std::vector<bool> mask;              // dof -> in-contact; empty = nothing highlighted
    SurfaceMesh*      surface = nullptr; // non-null AND show_surface -> draw the smooth surface
    bool              show_surface   = false;
    bool              show_edges     = true;
    bool              show_particles = true;
};

// Draws up to two trajectory layers plus the collider, in the fixed order the viewer relies on for
// correct transparency/layering: all smooth surfaces first (opaque base fill), then all edges, then
// — inside the sphere shader — all particles followed by the collider. The caller must already be
// inside a BeginMode3D/EndMode3D block; the sphere shader is entered and exited internally.
void draw_scene_layers(const SceneLayer* layers, int n, const std::vector<Collider>& colliders, float collider_time,
                       int highlight_index = -1)
{
    for (int i = 0; i < n; ++i)
        if (layers[i].show_surface && layers[i].surface)
            draw_tape_surface(*layers[i].mesh, *layers[i].frame, layers[i].color, kSurfaceSubdiv,
                              *layers[i].surface, g_viewer.surface_material);

    for (int i = 0; i < n; ++i)
        if (layers[i].show_edges)
            draw_tape_edges(*layers[i].mesh, *layers[i].frame, layers[i].color);

    BeginShaderMode(g_viewer.sphere_shader);
    for (int i = 0; i < n; ++i)
        if (layers[i].show_particles)
            draw_tape_spheres(*layers[i].mesh, *layers[i].frame, layers[i].color, kParticleRadius,
                              layers[i].mask, layers[i].collide_color);
    for (int i = 0; i < (int)colliders.size(); ++i)
        draw_collider(colliders[i], collider_time, i == highlight_index ? kColliderHighlightColor : kColliderColor);
    EndShaderMode();
}

void draw_live_scene()
{
    BeginMode3D(g_viewer.camera);
    draw_axes(kAxisLength);

    if (g_viewer.mesh)
    {
        SceneLayer layers[2];
        int n = 0;
        if (g_viewer.has_reference)
            layers[n++] = { g_viewer.mesh, &g_viewer.reference_frame, kReferenceColor, kReferenceCollide,
                            mask_from_contacts(&g_viewer.reference_contacts, g_viewer.reference_frame.rows()) };
        if (g_viewer.has_live)
            layers[n++] = { g_viewer.mesh, &g_viewer.live_frame, kLiveColor, kLiveCollide,
                            mask_from_contacts(&g_viewer.live_contacts, g_viewer.live_frame.rows()) };

        draw_scene_layers(layers, n, g_viewer.colliders, (float)g_viewer.collider_time);
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

    explicit FloatTextState(Real initial) { std::snprintf(buf, sizeof(buf), "%.1f", (double)initial); }
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
        std::snprintf(buf[0], sizeof(buf[0]), "%.1f", (double)initial.x());
        std::snprintf(buf[1], sizeof(buf[1]), "%.1f", (double)initial.y());
        std::snprintf(buf[2], sizeof(buf[2]), "%.1f", (double)initial.z());
    }
};

// Row of 9 (label above checkbox) FD-epsilon-order-of-magnitude toggles within `r`. Shared between
// the config screen's PanelCursor::fd_epsilon_row_field (below) and the on-demand FD-check panel
// offered during playback (see viewer_interactive_playback) — same widget, two different host
// screens, so the drawing itself is factored out here rather than duplicated.
void draw_fd_epsilon_row(Rectangle r, bool* selected)
{
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

// A GuiDropdownBox that's open must be (a) drawn after every other sibling control, or later-drawn
// rows paint over its popup list, and (b) drawn outside the scroll panel's scissor clip, or a popup
// taller than the remaining visible panel height gets cut off. Both are impossible to guarantee from
// inside the normal top-to-bottom field pass, so PanelCursor::dropdown_field skips drawing entirely
// when open and instead fills this out — the caller (viewer_show_config_screen) redraws the same box
// at `rect`, unclipped, as the very last thing in the frame.
struct PendingDropdown
{
    bool        active = false;
    Rectangle   rect{};
    std::string items;
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
        if (!edit_mode) std::snprintf(buf, 32, "%.1f", (double)*value);
        float v = (float)*value;
        if (GuiValueBoxFloat(box_rect, nullptr, buf, &v, edit_mode)) edit_mode = !edit_mode;
        *value = (Real)v;
    }

    // All three components of a Vec3 on a single row: one label on the left, then three boxes
    // side by side, each bordered in its axis color (matching the gizmo/axis arrow RED/GREEN/BLUE
    // convention).
    void vec3_field_inline(const char* name, Vec3* value, Vec3TextState& state)
    {
        const Rectangle r = row();
        if (measuring) return;
        const Rectangle label_rect = { r.x, r.y, kLabelW, r.height };
        GuiLabel(label_rect, name);

        constexpr float kBoxGap = 4.0f;
        const float boxes_x = r.x + kLabelW + kGap;
        const float boxes_w = r.width - kLabelW - kGap;
        const float box_w   = (boxes_w - 2.0f * kBoxGap) / 3.0f;

        const Color axis_color[3] = { RED, GREEN, BLUE };
        Real* const comp[3]       = { &value->x(), &value->y(), &value->z() };

        for (int i = 0; i < 3; ++i)
        {
            const Rectangle box_rect = { boxes_x + i * (box_w + kBoxGap), r.y, box_w, r.height };

            const int prev_normal  = GuiGetStyle(VALUEBOX, BORDER_COLOR_NORMAL);
            const int prev_focused = GuiGetStyle(VALUEBOX, BORDER_COLOR_FOCUSED);
            const int prev_pressed = GuiGetStyle(VALUEBOX, BORDER_COLOR_PRESSED);
            const int c = ColorToInt(axis_color[i]);
            GuiSetStyle(VALUEBOX, BORDER_COLOR_NORMAL,  c);
            GuiSetStyle(VALUEBOX, BORDER_COLOR_FOCUSED, c);
            GuiSetStyle(VALUEBOX, BORDER_COLOR_PRESSED, c);

            // See float_box's comment: keep the text buffer synced to *value while not editing.
            if (!state.edit[i]) std::snprintf(state.buf[i], 32, "%.1f", (double)*comp[i]);
            float v = (float)*comp[i];
            if (GuiValueBoxFloat(box_rect, nullptr, state.buf[i], &v, state.edit[i])) state.edit[i] = !state.edit[i];
            *comp[i] = (Real)v;

            GuiSetStyle(VALUEBOX, BORDER_COLOR_NORMAL,  prev_normal);
            GuiSetStyle(VALUEBOX, BORDER_COLOR_FOCUSED, prev_focused);
            GuiSetStyle(VALUEBOX, BORDER_COLOR_PRESSED, prev_pressed);
        }
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

    // Dropdown listing `items` (a ';'-joined label string, e.g. "sphere1;capsule1;ground1") with
    // `active` as the in/out selected index. `edit_mode` is caller-owned, toggled the same way as
    // GuiValueBoxFloat/GuiSpinner elsewhere in this file (flip it whenever the widget reports a
    // click that should open/close the popup). While open, drawing is deferred to `pending` (see
    // its comment) instead of happening here in-place.
    void dropdown_field(const char* name, const std::string& items, int& active, bool& edit_mode, PendingDropdown& pending)
    {
        label(name);
        const Rectangle r = row();
        if (measuring) return;
        if (edit_mode)
        {
            pending.active = true;
            pending.rect   = r;
            pending.items  = items;
            return;
        }
        if (GuiDropdownBox(r, items.c_str(), &active, edit_mode)) edit_mode = !edit_mode;
    }

    // Two buttons side by side on one row (used for the collider list's "+"/"-" controls).
    void button_pair(const char* left_text, const char* right_text, bool& left_clicked, bool& right_clicked)
    {
        const Rectangle r = row();
        left_clicked  = false;
        right_clicked = false;
        if (measuring) return;
        const float w = (r.width - kGap) / 2.0f;
        const Rectangle left_rect  = { r.x, r.y, w, r.height };
        const Rectangle right_rect = { r.x + w + kGap, r.y, w, r.height };
        left_clicked  = GuiButton(left_rect, left_text);
        right_clicked = GuiButton(right_rect, right_text);
    }

    // One-off enum<->int shims — not worth a generic templated helper.
    void cloth_type_field(ClothType& type)
    {
        label("Cloth Type");
        const Rectangle r = row();
        if (measuring) return;
        int active = (int)type;
        GuiToggleGroup(r, "Square;Skirt", &active);
        type = (ClothType)active;
    }

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

    void contact_point_mode_field(ContactPointMode& mode)
    {
        label("Contact Velocity Point");
        const Rectangle r = row();
        if (measuring) return;
        int active = (int)mode;
        GuiToggleGroup(r, "Particle;Surface", &active);
        mode = (ContactPointMode)active;
    }

    void collider_type_field(ColliderType& type)
    {
        label("Collider Shape");
        const Rectangle r = row();
        if (measuring) return;
        // Sphere/Capsule/Plane("Ground")/None are selectable here (Cylinder still exists in the
        // physics and viewer code, just not reachable from this picker) — map the 4-way toggle
        // index to the corresponding ColliderType ordinal explicitly since they aren't contiguous.
        static constexpr ColliderType kSelectable[4] = { ColliderType::Sphere, ColliderType::Capsule,
                                                           ColliderType::Plane, ColliderType::None };
        int active = 0;
        for (int i = 0; i < 4; ++i)
            if (kSelectable[i] == type) active = i;
        GuiToggleGroup(r, "Sphere;Capsule;Ground;None", &active);
        type = kSelectable[active];
    }

    // One checkbox per order of magnitude (1e-2 .. 1e-10), label drawn above each box. `selected`
    // must point at an array of (at least) 9 bools, indexed the same way as kFDEpsilonValues.
    void fd_epsilon_row_field(bool* selected)
    {
        label("FD Epsilon (order of magnitude)");
        const Rectangle r = row(32.0f);
        if (measuring) return;
        draw_fd_epsilon_row(r, selected);
    }
};

// Builds a ';'-joined GuiDropdownBox item list naming every collider by shape with a per-shape
// running count over list order (e.g. "sphere1;capsule1;ground1;sphere2") — matches the name a
// collider would get if added one at a time via "+".
std::string collider_dropdown_items(const std::vector<Collider>& colliders)
{
    int n_sphere = 0, n_capsule = 0, n_ground = 0, n_cylinder = 0, n_none = 0;
    std::string items;
    for (size_t i = 0; i < colliders.size(); ++i)
    {
        if (i > 0) items += ';';
        switch (colliders[i].type)
        {
            case ColliderType::Sphere:   items += "sphere"   + std::to_string(++n_sphere);   break;
            case ColliderType::Capsule:  items += "capsule"  + std::to_string(++n_capsule);  break;
            case ColliderType::Plane:    items += "ground"   + std::to_string(++n_ground);   break;
            case ColliderType::Cylinder: items += "cylinder" + std::to_string(++n_cylinder); break;
            case ColliderType::None:     items += "none"     + std::to_string(++n_none);     break;
        }
    }
    return items;
}

// Draws only the fields relevant to whichever collider shape is currently active. Each shape's
// Vec3TextState statics are seeded from Collider's own defaults the first time that shape's case
// actually runs (which may be later than frame 1, if the user switches shape) — correct either
// way, since default_config_collider() pre-fills every shape's fields up front regardless of
// which one is initially active. `force_reseed`: the *selected collider* (not just its shape) just
// changed, so whichever case runs below must drop any in-progress edit-mode/text left over from
// whichever other collider last used that same shared static state, and resync its text buffer from
// this collider's own current value (the ordinary "resync while not editing" path below already
// does that correctly once edit is cleared).
void collider_shape_fields(PanelCursor& cur, Collider& c, bool force_reseed)
{
    switch (c.type)
    {
        case ColliderType::Sphere:
        {
            static Vec3TextState center_state(c.sphere_center);
            static FloatTextState radius_state(c.sphere_radius);
            if (force_reseed) { center_state.edit[0] = center_state.edit[1] = center_state.edit[2] = false; radius_state.edit = false; }
            cur.vec3_field_inline("Sphere Center", &c.sphere_center, center_state);
            cur.float_box("Sphere Radius", &c.sphere_radius, radius_state.buf, radius_state.edit);
            break;
        }
        case ColliderType::Cylinder:
        {
            static Vec3TextState origin_state(c.cylinder_origin);
            static Vec3TextState axis_state(c.cylinder_axis);
            static FloatTextState radius_state(c.cylinder_radius);
            if (force_reseed) { origin_state.edit[0] = origin_state.edit[1] = origin_state.edit[2] = false;
                                 axis_state.edit[0]   = axis_state.edit[1]   = axis_state.edit[2]   = false;
                                 radius_state.edit = false; }
            cur.vec3_field_inline("Cylinder Origin", &c.cylinder_origin, origin_state);
            cur.vec3_field_inline("Cylinder Axis", &c.cylinder_axis, axis_state);
            cur.float_box("Cylinder Radius", &c.cylinder_radius, radius_state.buf, radius_state.edit);
            break;
        }
        case ColliderType::Plane:
        {
            static Vec3TextState origin_state(c.plane_origin);
            static Vec3TextState normal_state(c.plane_normal);
            if (force_reseed) { origin_state.edit[0] = origin_state.edit[1] = origin_state.edit[2] = false;
                                 normal_state.edit[0] = normal_state.edit[1] = normal_state.edit[2] = false; }
            cur.vec3_field_inline("Plane Origin", &c.plane_origin, origin_state);
            cur.vec3_field_inline("Plane Normal", &c.plane_normal, normal_state);
            break;
        }
        case ColliderType::Capsule:
        {
            static Vec3TextState p0_state(c.capsule_p0);
            static Vec3TextState p1_state(c.capsule_p1);
            static FloatTextState radius_state(c.capsule_radius);
            if (force_reseed) { p0_state.edit[0] = p0_state.edit[1] = p0_state.edit[2] = false;
                                 p1_state.edit[0] = p1_state.edit[1] = p1_state.edit[2] = false;
                                 radius_state.edit = false; }
            cur.vec3_field_inline("Capsule P0", &c.capsule_p0, p0_state);
            cur.vec3_field_inline("Capsule P1", &c.capsule_p1, p1_state);
            cur.float_box("Capsule Radius", &c.capsule_radius, radius_state.buf, radius_state.edit);
            break;
        }
        case ColliderType::None:
            cur.label("No collider active — contact disabled");
            break;
    }
}

// Rotation is orthogonal to which shape is active (like Collider::velocity), so it's drawn once
// here rather than duplicated per shape. `enabled` and `last_axis` are static so switching the
// checkbox off and back on restores whichever axis was last selected instead of forgetting it —
// RotationAxis::None on the Collider itself is the actual "disabled" state read everywhere else
// (detect_contacts, draw_collider, collider_gizmo_anchors). `force_reseed`: see collider_shape_fields
// — the selected collider itself just changed, so `enabled`/`last_axis` (which have no automatic
// resync path, unlike the text fields below) must be re-derived from *this* collider's actual state
// instead of continuing to reflect whichever other collider last edited these shared statics.
void collider_rotation_fields(PanelCursor& cur, Collider& c, bool force_reseed)
{
    static bool         enabled   = (c.rotation_axis != RotationAxis::None);
    static RotationAxis last_axis = (c.rotation_axis != RotationAxis::None) ? c.rotation_axis : RotationAxis::X;
    if (force_reseed)
    {
        enabled   = (c.rotation_axis != RotationAxis::None);
        last_axis = (c.rotation_axis != RotationAxis::None) ? c.rotation_axis : RotationAxis::X;
    }

    cur.checkbox_field("Enable Rotation", &enabled);
    // Never write back during the measuring pass: `enabled`/`last_axis` reflect whichever collider
    // was last *really* (non-measuring) drawn, which — now that the same static state is reused
    // across every collider in the list — may not be `c` at all, e.g. right after switching the
    // dropdown to a collider not yet visited this frame's measuring pass. Writing them back
    // unconditionally would silently stamp that stale enabled/axis onto `c` (this was harmless when
    // there was only ever one collider, since `c` was always the same object).
    if (!cur.measuring) c.rotation_axis = enabled ? last_axis : RotationAxis::None;

    if (!enabled) return;

    cur.label("Rotation Axis");
    const Rectangle r = cur.row();
    if (!cur.measuring)
    {
        // Manual re-implementation of GuiToggleGroup's own layout (bounds.width/itemCount per item,
        // consecutive items offset by width + GROUP_PADDING) so each button's text can be tinted to
        // match its gizmo-arrow color (RED/GREEN/BLUE) — GuiToggleGroup itself has no per-item style.
        static const char* kNames[3] = { "X", "Y", "Z" };
        const float pad  = (float)GuiGetStyle(TOGGLE, GROUP_PADDING);
        const float colW = r.width / 3.0f;

        const int prev_normal  = GuiGetStyle(TOGGLE, TEXT_COLOR_NORMAL);
        const int prev_pressed = GuiGetStyle(TOGGLE, TEXT_COLOR_PRESSED);

        int active = (int)c.rotation_axis; // X=0, Y=1, Z=2 — matches RotationAxis's declaration order
        for (int i = 0; i < 3; ++i)
        {
            const Rectangle br = { r.x + i * (colW + pad), r.y, colW, r.height };
            GuiSetStyle(TOGGLE, TEXT_COLOR_NORMAL,  ColorToInt(kGizmoAxisColors[i]));
            GuiSetStyle(TOGGLE, TEXT_COLOR_PRESSED, ColorToInt(kGizmoAxisColors[i]));
            bool toggle = (active == i);
            GuiToggle(br, kNames[i], &toggle);
            if (toggle) active = i;
        }

        GuiSetStyle(TOGGLE, TEXT_COLOR_NORMAL,  prev_normal);
        GuiSetStyle(TOGGLE, TEXT_COLOR_PRESSED, prev_pressed);

        c.rotation_axis = (RotationAxis)active;
        last_axis       = c.rotation_axis;
    }
    // else (measuring pass): row() above already advanced the cursor by this row's height.

    static Vec3TextState  origin_state(c.rotation_origin);
    static FloatTextState omega_state(c.omega);
    if (force_reseed) { origin_state.edit[0] = origin_state.edit[1] = origin_state.edit[2] = false; omega_state.edit = false; }
    cur.vec3_field_inline("Rotation Origin",  &c.rotation_origin, origin_state);
    cur.float_box("Angular Velocity", &c.omega, omega_state.buf, omega_state.edit);
}

// Full field list for the config screen, in display order. Run identically for the measuring
// pass and the real draw pass (see PanelCursor comment above). `selected_collider` is the index
// into cfg.colliders currently shown/edited (the dropdown + "+"/"-" controls in the Collision
// section mutate it); `collider_dropdown_edit` is that dropdown's own open/closed state;
// `dropdown_pending` receives the deferred-draw request when the dropdown is open (see PendingDropdown).
void draw_config_fields(PanelCursor& cur, AppConfig& cfg, int& selected_collider, bool& collider_dropdown_edit,
                        PendingDropdown& dropdown_pending)
{
    static bool edit_width = false, edit_height = false, edit_fps = false,
                edit_frame_substeps = false, edit_secs = false,
                edit_n_iters = false, edit_n_iters_adjoint = false,
                edit_particles_per_ring = false, edit_num_rings = false;
    static Vec3TextState origin_state(cfg.origin);
    static Vec3TextState target_origin_state(cfg.target_origin);
    static Vec3TextState velocity_state(Vec3::Zero()); // reseeded below once a collider is selected
    static Vec3TextState gravity_state(cfg.gravity);
    static FloatTextState stiffness_state(cfg.stiffness);
    static FloatTextState target_stiffness_state(cfg.target_stiffness);
    static FloatTextState m_tot_state(cfg.m_tot);
    static FloatTextState radius_top_state(cfg.radius_top);
    static FloatTextState radius_bottom_state(cfg.radius_bottom);
    static FloatTextState skirt_height_state(cfg.skirt_height);

    cur.section("Cloth");
    cur.cloth_type_field(cfg.cloth_type);
    if (cfg.cloth_type == ClothType::Square)
    {
        cur.int_spinner("Width",  &cfg.width,  2, 200, &edit_width);
        cur.int_spinner("Height", &cfg.height, 2, 200, &edit_height);
        cur.pin_mode_field(cfg.pin_mode);
        cur.hanging_mode_field(cfg.hang_mode);
    }
    else // ClothType::Skirt
    {
        cur.int_spinner("Particles per Ring", &cfg.particles_per_ring, 3, 200, &edit_particles_per_ring);
        cur.int_spinner("Num Rings",          &cfg.num_rings,          2, 200, &edit_num_rings);
        cur.float_box("Radius Top",    &cfg.radius_top,    radius_top_state.buf,    radius_top_state.edit);
        cur.float_box("Radius Bottom", &cfg.radius_bottom, radius_bottom_state.buf, radius_bottom_state.edit);
        cur.float_box("Skirt Height",  &cfg.skirt_height,  skirt_height_state.buf,  skirt_height_state.edit);
        cur.label("Top ring is always pinned");
    }
    cur.float_box("Stiffness", &cfg.stiffness, stiffness_state.buf, stiffness_state.edit);
    cur.vec3_field_inline("Origin", &cfg.origin, origin_state);
    cur.checkbox_field("Shear constraints",   &cfg.flag_shear);
    cur.checkbox_field("Bending constraints", &cfg.flag_bending);
    cur.label("Stretch constraints: always on");
    cur.float_box("Total Mass", &cfg.m_tot, m_tot_state.buf, m_tot_state.edit);

    cur.section("Target Cloth");
    cur.float_box("Target Stiffness", &cfg.target_stiffness, target_stiffness_state.buf, target_stiffness_state.edit);
    cur.vec3_field_inline("Target Origin", &cfg.target_origin, target_origin_state);

    cur.section("Collision");

    // Clamp into range: the list may have shrunk (via "-") or started empty.
    if (selected_collider >= (int)cfg.colliders.size()) selected_collider = (int)cfg.colliders.size() - 1;
    if (selected_collider < 0 && !cfg.colliders.empty()) selected_collider = 0;

    if (!cfg.colliders.empty())
        cur.dropdown_field("Collider", collider_dropdown_items(cfg.colliders), selected_collider, collider_dropdown_edit, dropdown_pending);

    // The selected collider itself (not just its shape) may have changed this frame — either via
    // the dropdown above, or via last frame's "+"/"-" (see below) — in which case every field's
    // shared static text/edit state must drop whatever it was showing for the previously selected
    // collider before it's drawn against this one. Tracked only on the real (non-measuring) pass,
    // and only updated after being used, so the comparison is always against last frame's value.
    static int last_selected_shown = -2; // sentinel: forces a reseed on the very first real draw
    const bool force_reseed = !cur.measuring && selected_collider != last_selected_shown;

    if (!cfg.colliders.empty())
    {
        Collider& c = cfg.colliders[selected_collider];
        cur.collider_type_field(c.type);
        collider_shape_fields(cur, c, force_reseed);
        if (force_reseed) velocity_state.edit[0] = velocity_state.edit[1] = velocity_state.edit[2] = false;
        cur.vec3_field_inline("Collider Velocity", &c.velocity, velocity_state);
        collider_rotation_fields(cur, c, force_reseed);
    }
    else
    {
        cur.label("No colliders");
    }

    if (!cur.measuring) last_selected_shown = selected_collider;

    bool add_clicked = false, remove_clicked = false;
    cur.button_pair("+", "-", add_clicked, remove_clicked);
    if (add_clicked)
    {
        cfg.colliders.push_back(default_config_collider());
        selected_collider = (int)cfg.colliders.size() - 1;
    }
    if (remove_clicked && !cfg.colliders.empty())
    {
        cfg.colliders.erase(cfg.colliders.begin() + selected_collider);
        if (selected_collider >= (int)cfg.colliders.size()) selected_collider = (int)cfg.colliders.size() - 1;
    }

    cur.contact_point_mode_field(cfg.contact_point_mode);

    cur.section("Physics");
    cur.vec3_field_inline("Gravity", &cfg.gravity, gravity_state);

    cur.section("Simulation / Solver");
    cur.int_spinner("FPS",             &cfg.FPS,             1, 240,  &edit_fps);
    cur.int_spinner("Frame Substeps",  &cfg.frame_substeps,  1, 64,   &edit_frame_substeps);
    cur.int_spinner("Seconds",         &cfg.secs,            1, 120,  &edit_secs);
    cur.int_spinner("Solver Iters",    &cfg.n_iters,          1, 1000, &edit_n_iters);
    cur.int_spinner("Adjoint Iters",   &cfg.n_iters_adjoint,  1, 1000, &edit_n_iters_adjoint);

    cur.section("Output");
    cur.checkbox_field("Record on Run", &cfg.record_on_run);

    cur.section("Gradient Check");
    cur.checkbox_field("Enable FD Check (dphi/dk)", &cfg.run_fd_check);
    if (cfg.run_fd_check) cur.fd_epsilon_row_field(cfg.fd_eps_selected);

    const int substeps = cfg.FPS * cfg.frame_substeps;
    const Real dt      = substeps > 0 ? 1.0 / substeps : 0.0;
    const int  n_steps = substeps * cfg.secs;
    cur.label(TextFormat("substeps=%d  dt=%.5f  n_steps=%d", substeps, dt, n_steps));
}

float measure_content_height(AppConfig& cfg, int selected_collider, bool collider_dropdown_edit)
{
    PanelCursor cur;
    cur.measuring = true;
    PendingDropdown dummy_pending; // never populated during the measuring pass (see dropdown_field)
    draw_config_fields(cur, cfg, selected_collider, collider_dropdown_edit, dummy_pending);
    return cur.y;
}

} // namespace

void viewer_open()
{
    InitWindow(1280, 800, "diffpd viewer");
    SetTargetFPS(60);
    GuiLoadStyleDark();
    GuiSetStyle(TOGGLE, GROUP_WIDTH_FULL, 1); // one GuiToggleGroup() call divides the full row width evenly

    update_orbit_camera(g_viewer.orbit, g_viewer.camera); // sets fovy/projection from orbit defaults too

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
    // Don't lose an in-progress recording just because the window was closed instead of Stop
    // being clicked — encode whatever was captured so far.
    if (g_recorder.recording) stop_recording(g_recorder);

    unload_surface_mesh(g_viewer.reference_surface);
    unload_surface_mesh(g_viewer.live_surface);

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
                       const std::vector<Collider>& colliders, Real collider_time,
                       const std::string& status_text)
{
    g_viewer.mesh = &mesh;

    g_viewer.has_reference = reference_frame != nullptr;
    if (g_viewer.has_reference) g_viewer.reference_frame = *reference_frame;
    g_viewer.reference_contacts = reference_contacts ? *reference_contacts : Contacts{};

    g_viewer.has_live = live_frame != nullptr;
    if (g_viewer.has_live) g_viewer.live_frame = *live_frame;
    g_viewer.live_contacts = live_contacts ? *live_contacts : Contacts{};

    g_viewer.colliders     = colliders;
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
    draw_recorder_overlay(GetScreenWidth(), GetScreenHeight());
    if (g_recorder.recording) capture_recording_frame(g_recorder);
    EndDrawing();

    return !WindowShouldClose();
}

bool viewer_poll_close()
{
    ASSERT(g_viewer.open, "viewer_poll_close: viewer_open() was not called");
    PollInputEvents(); // same event pump EndDrawing() runs internally, without the draw/swap cost
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

    // Which collider (index into cfg.colliders) the dropdown/fields/gizmo currently target, and
    // that dropdown's own open/closed state — both function-locals, reset each time the config
    // screen is (re)entered, same scoping as gizmo_state/scroll above.
    int  selected_collider      = cfg.colliders.empty() ? -1 : 0;
    bool collider_dropdown_edit = false;

    bool run_clicked  = false;
    bool quit_clicked = false;

    while (!WindowShouldClose() && !run_clicked && !quit_clicked)
    {
        const Vector2 mouse = GetMousePosition();
        const bool over_viewport = mouse.x >= kPanelWidth;
        const bool has_selected_collider = selected_collider >= 0 && selected_collider < (int)cfg.colliders.size();

        // --- collider translate gizmo(s): hover pick + drag update -----------------------------
        // Done before the orbit-camera arbitration below so a press that grabs an axis arrow can
        // suppress that same press from also starting an orbit. Only the *selected* collider is
        // gizmo-editable; with none selected (empty list) there's nothing to drag.
        const Ray mouse_ray = GetScreenToWorldRayEx({ mouse.x - kPanelWidth, mouse.y },
                                                     g_viewer.camera, viewport_w, viewport_h);
        const GizmoFrame gizmo = has_selected_collider
            ? update_collider_gizmos(cfg.colliders[selected_collider], gizmo_state, g_viewer.camera, mouse_ray, over_viewport)
            : GizmoFrame{};

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
        // While a spinner is being edited, raygui writes the field's live int value (including a
        // transient 0 while the box is empty mid-edit) straight into cfg before the panel is
        // redrawn — cloth()/skirt() index their pin/constraint grids assuming width,height >= 1 (and
        // skirt() further ASSERTs particles_per_ring >= 3, num_rings >= 2), so a stray 0/low value
        // here would corrupt memory or abort. Build from a clamped copy of cfg (same minimums the
        // spinners/skirt() enforce) so the preview always has valid geometry, without touching
        // cfg's own fields (which would stomp on whatever the user is mid-typing).
        AppConfig preview_cfg           = cfg;
        preview_cfg.width               = std::max(cfg.width,  2);
        preview_cfg.height              = std::max(cfg.height, 2);
        preview_cfg.particles_per_ring  = std::max(cfg.particles_per_ring, 3);
        preview_cfg.num_rings           = std::max(cfg.num_rings,          2);
        Object target_obj = build_cloth(preview_cfg, cfg.target_stiffness, cfg.target_origin);
        Object guess_obj  = build_cloth(preview_cfg, cfg.stiffness,        cfg.origin);
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
                const SceneLayer preview_layers[2] = {
                    { &target_obj.mesh, &target_frame, kReferenceColor, kReferenceCollide, {} },
                    { &guess_obj.mesh,  &guess_frame,  kLiveColor,      kLiveCollide,      {} },
                };
                // time=0: static preview; the selected collider is drawn lighter-gray (see kColliderHighlightColor)
                draw_scene_layers(preview_layers, 2, cfg.colliders, 0.0f, selected_collider);
                for (int p = 0; p < gizmo.n_points; ++p)
                {
                    const int point_hover_axis = (gizmo.hover_point == p) ? gizmo.hover_axis : -1;
                    const int point_drag_axis  = (gizmo_state.point == p) ? gizmo_state.drag_axis : -1;
                    draw_translate_gizmo(gizmo.origins[p], gizmo.lengths[p], point_hover_axis, point_drag_axis);
                }
                if (has_selected_collider && cfg.colliders[selected_collider].rotation_axis != RotationAxis::None)
                {
                    const Collider& sel = cfg.colliders[selected_collider];
                    const Color axis_color = kGizmoAxisColors[(int)sel.rotation_axis];
                    const float axis_gizmo_length = gizmo_length_for(sel.rotation_origin, g_viewer.camera);
                    draw_rotation_axis_indicator(sel.rotation_origin, sel.rotation_axis,
                                                  axis_gizmo_length, axis_color);
                }
            EndMode3D();
        EndTextureMode();

        // --- composite: viewport texture + scrollable panel + Run button -----------------------
        BeginDrawing();
            ClearBackground(kBackgroundColor);

            DrawTextureRec(viewport_rt.texture,
                           { 0, 0, (float)viewport_w, -(float)viewport_h }, // negative height: render textures are Y-flipped
                           { kPanelWidth, 0 }, WHITE);

            const int selected_collider_before_frame = selected_collider;

            // While the collider dropdown's popup is open, lock every other control so a click that
            // lands on a field/button underneath the (visually on-top, deferred-drawn) popup isn't
            // also processed by that field/button this frame — raygui controls run their own input
            // handling at the moment each is drawn, in row order, so without this a click on the
            // popup would fall through to whatever row is at that same screen position.
            if (collider_dropdown_edit) GuiLock();

            const float scroll_area_h = (float)GetScreenHeight() - kFooterH;
            const Rectangle panel_bounds = { 0, 0, kPanelWidth, scroll_area_h };
            const Rectangle content = { 0, 0, kPanelWidth - 16.0f,
                                        measure_content_height(cfg, selected_collider, collider_dropdown_edit) };
            Rectangle view;
            GuiScrollPanel(panel_bounds, nullptr, content, &scroll, &view);

            PendingDropdown collider_dropdown_pending;
            BeginScissorMode((int)view.x, (int)view.y, (int)view.width, (int)view.height);
                PanelCursor cur;
                cur.view   = view;
                cur.scroll = scroll;
                draw_config_fields(cur, cfg, selected_collider, collider_dropdown_edit, collider_dropdown_pending);
            EndScissorMode();

            // The collider selector may have changed which collider is selected (dropdown click, or
            // "+"/"-" inside draw_config_fields) — reset any in-progress gizmo drag so it can never
            // get silently redirected onto a different collider's memory.
            if (selected_collider != selected_collider_before_frame) gizmo_state = GizmoDragState{};

            constexpr float kQuitButtonW = 90.0f;
            const Rectangle quit_rect = { 8.0f, scroll_area_h + 8.0f, kQuitButtonW, kFooterH - 16.0f };
            const Rectangle run_rect  = { quit_rect.x + kQuitButtonW + 8.0f, scroll_area_h + 8.0f,
                                          kPanelWidth - 16.0f - kQuitButtonW - 8.0f, kFooterH - 16.0f };
            quit_clicked = GuiButton(quit_rect, "Quit");
            run_clicked  = GuiButton(run_rect, "Run");

            // "Record on Run": same effect as clicking Record yourself first, just folded into
            // the Run click. Guarded on !g_recorder.recording so it's a no-op if a recording was
            // already started by hand (don't stomp on whatever name/state that recording has).
            if (run_clicked && cfg.record_on_run && !g_recorder.recording)
                start_recording(g_recorder);

            draw_recorder_overlay(GetScreenWidth(), GetScreenHeight());
            if (g_recorder.recording) capture_recording_frame(g_recorder);

            if (collider_dropdown_edit) GuiUnlock(); // re-enable input for the dropdown itself, drawn next

            // The collider dropdown's open popup, redrawn last (so nothing painted above overlays
            // it) and outside any scissor region (so it's never clipped by the scroll panel) — see
            // PendingDropdown / PanelCursor::dropdown_field.
            if (collider_dropdown_pending.active)
            {
                if (GuiDropdownBox(collider_dropdown_pending.rect, collider_dropdown_pending.items.c_str(),
                                   &selected_collider, collider_dropdown_edit))
                    collider_dropdown_edit = !collider_dropdown_edit;
            }
        EndDrawing();
    }

    UnloadRenderTexture(viewport_rt);
    // quit_clicked closes exactly like the window's X button (return false, no run requested) —
    // main()'s `while (viewer_show_config_screen(cfg))` already treats a false return as "quit".
    return run_clicked;
}

bool viewer_interactive_playback(const SimMesh& mesh, const Tape& target_tape, const Tape& guess_tape,
                                  const std::vector<Collider>& colliders, Real dt, int frame_substeps, int fps,
                                  const bool (&fd_eps_seed)[9], const FDCheckRunner& run_fd_check,
                                  const GradientSummary& grad, const ResidualHistory& residuals)
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
    bool   quit_clicked    = false;

    // On-demand FD-check panel: same checkboxes/epsilons as the config screen (seeded from
    // whatever was selected there), so a forgotten or different epsilon doesn't require going back
    // to Setup and recomputing target+guess+backward pass from scratch. Running is a long,
    // window-pumping call (run_fd_check), so it's never invoked from inside this loop's own
    // BeginDrawing/EndDrawing block — only after it closes, via fd_run_requested below.
    bool fd_panel_open = false;
    bool fd_selected[9];
    std::copy(std::begin(fd_eps_seed), std::end(fd_eps_seed), fd_selected);
    std::vector<Real>         fd_last_eps;
    std::vector<FDCheckResult> fd_last_results;

    while (!WindowShouldClose() && !back_to_config && !quit_clicked)
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

        // draw_scene_layers keeps the surface-first/edges/particles ordering the layering relies on.
        SceneLayer layers[2];
        int n = 0;
        if (show_target)
            layers[n++] = { &mesh, &target_tape.positions[tape_index], kReferenceColor, kReferenceCollide,
                            show_collisions ? colliding_mask(target_tape, tape_index, target_tape.positions[tape_index].rows())
                                            : std::vector<bool>{},
                            &g_viewer.reference_surface, show_surface, show_edges, show_particles };
        if (show_guess)
            layers[n++] = { &mesh, &guess_tape.positions[tape_index], kLiveColor, kLiveCollide,
                            show_collisions ? colliding_mask(guess_tape, tape_index, guess_tape.positions[tape_index].rows())
                                            : std::vector<bool>{},
                            &g_viewer.live_surface, show_surface, show_edges, show_particles };

        draw_scene_layers(layers, n, colliders, collider_time);
        EndMode3D();

        DrawText(TextFormat("Frame %d / %d   t = %.3fs%s",
                             current_frame + 1, n_frames, sim_time, paused ? "  (paused)" : ""),
                 10, 10, 20, WHITE);
        DrawFPS(10, 40);
        draw_help_box(GetScreenWidth());
        const int grad_panel_h = draw_gradient_panel(10, 70, grad);

        // Three stacked residual graphs, one per solve, sharing the same x-axis (trajectory
        // progress) and the same red marker tracking the current playback frame.
        {
            constexpr float kGraphX = 10.0f, kGraphW = 260.0f, kGraphH = 100.0f, kGraphGap = 8.0f;
            const float graph_y0 = 70.0f + (float)grad_panel_h + 10.0f;
            const double progress = (n_frames > 1) ? (double)current_frame / (double)(n_frames - 1) : 0.0;

            draw_residual_graph({ kGraphX, graph_y0,                              kGraphW, kGraphH },
                                 "Target Forward Residual", residuals.target_forward, kReferenceColor, progress);
            draw_residual_graph({ kGraphX, graph_y0 + (kGraphH + kGraphGap),       kGraphW, kGraphH },
                                 "Guess Forward Residual",  residuals.guess_forward,  kLiveColor,      progress);
            draw_residual_graph({ kGraphX, graph_y0 + 2.0f * (kGraphH + kGraphGap), kGraphW, kGraphH },
                                 "Backward Adjoint Residual", residuals.backward_adjoint, SKYBLUE,     progress);
        }

        const Rectangle back_button_rect = { 10.0f, (float)GetScreenHeight() - 40.0f, 170.0f, 30.0f };
        if (GuiButton(back_button_rect, "Back to Setup")) back_to_config = true;

        const Rectangle quit_button_rect = { 190.0f, (float)GetScreenHeight() - 40.0f, 90.0f, 30.0f };
        if (GuiButton(quit_button_rect, "Quit")) quit_clicked = true;

        // --- on-demand FD check panel: same widget as the config screen's, offered again here ---
        const Rectangle fd_toggle_rect = { 290.0f, (float)GetScreenHeight() - 40.0f, 130.0f, 30.0f };
        if (GuiButton(fd_toggle_rect, fd_panel_open ? "FD Check ^" : "FD Check v")) fd_panel_open = !fd_panel_open;

        // Detected here (inside this frame's draw), but actually run after EndDrawing() below —
        // run_fd_check pumps its own BeginDrawing/EndDrawing frames internally (see main()'s
        // fd_heartbeat), and raylib doesn't support nesting those inside this loop's own.
        bool fd_run_requested = false;
        if (fd_panel_open)
        {
            constexpr float kFdPanelW = 400.0f;
            constexpr float kLineH    = 18.0f;
            const float results_h = std::max((size_t)1, fd_last_results.size()) * kLineH;
            const float panel_h   = 20.0f + 32.0f + 8.0f + 30.0f + 8.0f + results_h + 12.0f;
            const Rectangle panel_rect = { fd_toggle_rect.x, fd_toggle_rect.y - panel_h - 8.0f, kFdPanelW, panel_h };

            DrawRectangleRec(panel_rect, Fade(BLACK, 0.55f));
            DrawRectangleLinesEx(panel_rect, 1.0f, GRAY);

            float y = panel_rect.y + 6.0f;
            DrawText("FD Check (dphi/dk)", (int)panel_rect.x + 8, (int)y, 16, WHITE);
            y += 20.0f;

            const Rectangle row_rect = { panel_rect.x + 8.0f, y, panel_rect.width - 16.0f, 32.0f };
            draw_fd_epsilon_row(row_rect, fd_selected);
            y += 32.0f + 8.0f;

            const Rectangle run_rect = { panel_rect.x + 8.0f, y, panel_rect.width - 16.0f, 30.0f };
            if (GuiButton(run_rect, "Run")) fd_run_requested = true;
            y += 30.0f + 8.0f;

            if (fd_last_results.empty())
                DrawText("(no results yet)", (int)panel_rect.x + 8, (int)y, 14, GRAY);
            for (size_t i = 0; i < fd_last_results.size(); ++i)
            {
                const FDCheckResult& r = fd_last_results[i];
                DrawText(TextFormat("eps=%.0e  fd=%.6g  analytic=%.6g  rel_err=%.4g",
                                     fd_last_eps[i], r.fd, r.analytic, r.rel_err),
                         (int)panel_rect.x + 8, (int)y, 14, RAYWHITE);
                y += kLineH;
            }
        }

        draw_recorder_overlay(GetScreenWidth(), GetScreenHeight());
        if (g_recorder.recording) capture_recording_frame(g_recorder);

        EndDrawing();

        if (fd_run_requested)
        {
            std::vector<Real> epss;
            for (int i = 0; i < 9; ++i)
                if (fd_selected[i]) epss.push_back(kFDEpsilonValues[i]);
            fd_last_eps     = epss;
            fd_last_results = run_fd_check(epss);
        }
    }

    // quit_clicked closes exactly like the window's X button (return false) — main() already
    // treats a false return from viewer_interactive_playback as "quit".
    return back_to_config;
}
