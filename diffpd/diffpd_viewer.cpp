#include "diffpd_viewer.h"

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

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

constexpr const char* kSphereFS = R"(
#version 330
in vec3 fragNormal;
in vec4 fragColor;
uniform vec3 lightDir;
out vec4 finalColor;
void main()
{
    float diffuse = max(dot(normalize(fragNormal), lightDir), 0.0);
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
        DrawLine3D(a, b, color);
    }
}

void draw_tape_spheres(const SimMesh& mesh, const PointsX& frame, Color color, float particle_radius)
{
    for (Index vi = 0; vi < (Index)mesh.vertices.size(); ++vi)
        // DrawSphere(to_raylib(vertex_position(mesh, frame, vi)), particle_radius, color);
        DrawSphereEx(to_raylib(vertex_position(mesh, frame, vi)), particle_radius, 6, 6, color);
}

void draw_axes(float length)
{
    DrawLine3D({ 0, 0, 0 }, { length, 0, 0 }, RED);
    DrawLine3D({ 0, 0, 0 }, { 0, length, 0 }, GREEN);
    DrawLine3D({ 0, 0, 0 }, { 0, 0, length }, BLUE);
}

// Small translucent panel of one-line control hints, anchored to the top-right corner.
void draw_help_box(int screen_width)
{
    static const char* kLines[] = {
        "Space: play/pause",
        "L/R arrow: step frame (Shift: x10)",
        "T / G: toggle target / guess",
        "E: edges only",
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
    }
}

} // namespace

void play_tapes(const SimMesh& mesh, const Tape& target_tape, const Tape& guess_tape,
                 const Collider& collider, Real dt, int frame_substeps, int fps)
{
    ASSERT(target_tape.positions.size() == guess_tape.positions.size(),
           "play_tapes: tape length mismatch");
    ASSERT(frame_substeps > 0, "play_tapes: frame_substeps must be positive");
    ASSERT(fps > 0, "play_tapes: fps must be positive");


    const int n_tape_frames = (int)target_tape.positions.size();
    ASSERT((n_tape_frames - 1) % frame_substeps == 0,
           "play_tapes: tape length - 1 must be a multiple of frame_substeps");
    const int n_frames = (n_tape_frames - 1) / frame_substeps + 1;

    InitWindow(1280, 800, "diffpd viewer");
    SetTargetFPS(60);

    OrbitCamera orbit;
    Camera3D    camera = { 0 };
    camera.fovy        = 45.0f;
    camera.projection  = CAMERA_PERSPECTIVE;
    update_orbit_camera(orbit, camera);

    Shader sphere_shader = LoadShaderFromMemory(kSphereVS, kSphereFS);
    const Vector3 light_dir = Vector3Normalize({ 0.4f, 1.0f, 0.3f }); // from above, diagonal
    SetShaderValue(sphere_shader, GetShaderLocation(sphere_shader, "lightDir"),
                   &light_dir, SHADER_UNIFORM_VEC3);

    const Color background      = { 18, 18, 18, 255 };
    const Color target_color    = { 255, 165, 0, 140 }; // orange, half transparent
    const Color guess_color     = WHITE;
    const Color collider_color  = { 160, 160, 160, 255 }; // opaque gray
    const float particle_radius = 0.01f;
    const float axis_length     = 1.0f;
    const double frame_period   = 1.0 / fps;

    constexpr int kFrameJump = 10; // frames skipped per Shift+arrow press

    int    current_frame = 0;
    double accumulator   = 0.0;
    bool   paused        = false;
    bool   show_target   = true;
    bool   show_guess    = true;
    bool   edges_only    = false;

    while (!WindowShouldClose())
    {
        update_orbit_camera(orbit, camera);

        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_T))     show_target = !show_target;
        if (IsKeyPressed(KEY_G))     show_guess  = !show_guess;
        if (IsKeyPressed(KEY_E))     edges_only  = !edges_only;

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
            if (IsKeyPressed(KEY_RIGHT)) current_frame = std::min(current_frame + step, n_frames - 1);
            if (IsKeyPressed(KEY_LEFT))  current_frame = std::max(current_frame - step, 0);
        }

        const int   tape_index    = current_frame * frame_substeps;
        const float collider_time = (float)(current_frame * frame_substeps) * (float)dt;
        const double sim_time     = (double)(current_frame * frame_substeps) * (double)dt;

        BeginDrawing();
        ClearBackground(background);

        BeginMode3D(camera);
        draw_axes(axis_length);
        if (show_target) draw_tape_edges(mesh, target_tape.positions[tape_index], target_color);
        if (show_guess)  draw_tape_edges(mesh, guess_tape.positions[tape_index],  guess_color);

        BeginShaderMode(sphere_shader);
        if (show_target && !edges_only)
            draw_tape_spheres(mesh, target_tape.positions[tape_index], target_color, particle_radius);
        if (show_guess && !edges_only)
            draw_tape_spheres(mesh, guess_tape.positions[tape_index], guess_color, particle_radius);
        draw_collider(collider, collider_time, collider_color);
        EndShaderMode();
        EndMode3D();

        DrawText(TextFormat("Frame %d / %d   t = %.3fs%s",
                             current_frame + 1, n_frames, sim_time, paused ? "  (paused)" : ""),
                 10, 10, 20, WHITE);
        DrawFPS(10, 40);
        draw_help_box(GetScreenWidth());

        EndDrawing();
    }

    UnloadShader(sphere_shader);
    CloseWindow();
}
