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
        //DrawLine3D(a, b, color);
        DrawCylinderEx(a, b, 0.005f, 0.005f, 4, color);
    }
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
constexpr float kAxisLength          = 1.0f;

// Persistent viewer state: window/shader/camera survive across every phase of a run (target sim,
// guess sim, backward pass, FD check, final playback), so the camera pose carries forward and the
// window is only opened/closed once by main().
struct ViewerState
{
    bool        open = false;
    OrbitCamera orbit;
    Camera3D    camera = { 0 };
    Shader      sphere_shader{};

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

} // namespace

void viewer_open()
{
    InitWindow(1280, 800, "diffpd viewer");
    SetTargetFPS(60);

    g_viewer.camera.fovy       = 45.0f;
    g_viewer.camera.projection = CAMERA_PERSPECTIVE;
    update_orbit_camera(g_viewer.orbit, g_viewer.camera);

    g_viewer.sphere_shader = LoadShaderFromMemory(kSphereVS, kSphereFS);
    const Vector3 light_dir = Vector3Normalize({ 0.4f, 1.0f, 0.3f }); // from above, diagonal
    SetShaderValue(g_viewer.sphere_shader, GetShaderLocation(g_viewer.sphere_shader, "lightDir"),
                   &light_dir, SHADER_UNIFORM_VEC3);

    g_viewer.open = true;
}

void viewer_close()
{
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

void viewer_interactive_playback(const SimMesh& mesh, const Tape& target_tape, const Tape& guess_tape,
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
    bool   paused        = false;
    bool   show_target    = true;
    bool   show_guess     = true;
    bool   edges_only     = false;
    bool   show_collisions = false;

    while (!WindowShouldClose())
    {
        update_orbit_camera(g_viewer.orbit, g_viewer.camera);

        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_T))     show_target     = !show_target;
        if (IsKeyPressed(KEY_G))     show_guess      = !show_guess;
        if (IsKeyPressed(KEY_E))     edges_only      = !edges_only;
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

        if (show_target) draw_tape_edges(mesh, target_tape.positions[tape_index], kReferenceColor);
        if (show_guess)  draw_tape_edges(mesh, guess_tape.positions[tape_index],  kLiveColor);

        BeginShaderMode(g_viewer.sphere_shader);
        
        if (show_target && !edges_only)
        {
            const std::vector<bool> mask = show_collisions
                ? colliding_mask(target_tape, tape_index, target_tape.positions[tape_index].rows())
                : std::vector<bool>{};
            draw_tape_spheres(mesh, target_tape.positions[tape_index], kReferenceColor, kParticleRadius, mask, kReferenceCollide);
        }
        if (show_guess && !edges_only)
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

        EndDrawing();
    }
}
