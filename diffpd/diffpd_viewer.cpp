#include "diffpd_viewer.h"

#include "raylib.h"
#include "raymath.h"

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

} // namespace

void play_tapes(const SimMesh& mesh, const Tape& target_tape, const Tape& guess_tape,
                 int frame_substeps, int fps)
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
    const float particle_radius = 0.01f;
    const float axis_length     = 1.0f;
    const double frame_period   = 1.0 / fps;

    int    current_frame = 0;
    double accumulator   = 0.0;
    bool   paused        = false;

    while (!WindowShouldClose())
    {
        update_orbit_camera(orbit, camera);

        if (IsKeyPressed(KEY_SPACE)) paused = !paused;

        if (!paused)
        {
            accumulator += GetFrameTime();
            while (accumulator >= frame_period)
            {
                accumulator -= frame_period;
                current_frame = (current_frame + 1) % n_frames;
            }
        }

        const int tape_index = current_frame * frame_substeps;

        BeginDrawing();
        ClearBackground(background);

        BeginMode3D(camera);
        draw_axes(axis_length);
        draw_tape_edges(mesh, target_tape.positions[tape_index], target_color);
        draw_tape_edges(mesh, guess_tape.positions[tape_index],  guess_color);

        BeginShaderMode(sphere_shader);
        draw_tape_spheres(mesh, target_tape.positions[tape_index], target_color, particle_radius);
        draw_tape_spheres(mesh, guess_tape.positions[tape_index],  guess_color,  particle_radius);
        EndShaderMode();
        EndMode3D();

        DrawText(TextFormat("Frame %d / %d%s", current_frame + 1, n_frames, paused ? "  (paused)" : ""),
                 10, 10, 20, WHITE);
        DrawFPS(10, 40);

        EndDrawing();
    }

    UnloadShader(sphere_shader);
    CloseWindow();
}
