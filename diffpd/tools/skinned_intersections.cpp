// TEMPORARY tool: plays a skinned body's imported track (the plain 60 FPS LBS frames, no substeps) in a
// loop and highlights the triangles that cross another, non-adjacent triangle (mesh self-intersections).
// Gray + very transparent = clean, red + more opaque = self-intersecting.
//
//   skinned_intersections.exe [path/to/skinned_body.json]
//   Space: play/pause    Left/Right: step one frame (pauses)    G: show/hide the gray triangles
//   F: camera follows the body (default on)    mouse: orbit / pan (shift+drag or right-drag) / zoom, as in the player
//
// Single translation unit on purpose: it #includes the simulator and the viewer sources so it can reuse the
// viewer's internals (orbit camera, shaders, axes) and the animation loader without touching either file.
// Delete this file and its CMake target when the investigation is over.

#define main diffpd_app_main
#include "../diffpd.cpp"
#undef main
#include "../diffpd_viewer.cpp"

namespace si
{
using Tris = std::vector<std::array<int, 3>>;

// Proper crossing of segment p0->p1 with triangle (a, b, c): endpoints and edges excluded.
bool seg_tri(const Vec3& p0, const Vec3& p1, const Vec3& a, const Vec3& b, const Vec3& c)
{
    const Vec3 d = p1 - p0, e1 = b - a, e2 = c - a, h = d.cross(e2);
    const Real det = e1.dot(h);
    if (std::abs(det) < 1e-18) return false;
    const Real f = 1.0 / det;
    const Vec3 s = p0 - a;
    const Real u = f * s.dot(h);
    if (u < 1e-9 || u > 1 - 1e-9) return false;
    const Vec3 q = s.cross(e1);
    const Real v = f * d.dot(q);
    if (v < 1e-9 || u + v > 1 - 1e-9) return false;
    const Real t = f * e2.dot(q);
    return t > 1e-9 && t < 1 - 1e-9;
}

bool crosses(const std::vector<Vec3>& V, const std::array<int, 3>& A, const std::array<int, 3>& B)
{
    for (int e = 0; e < 3; ++e)
        if (seg_tri(V[A[e]], V[A[(e + 1) % 3]], V[B[0]], V[B[1]], V[B[2]]) ||
            seg_tri(V[B[e]], V[B[(e + 1) % 3]], V[A[0]], V[A[1]], V[A[2]]))
            return true;
    return false;
}

// flags[t] = 1 if triangle t properly crosses some triangle it shares no vertex with. `bvh` must already
// be refit to V; its leaves only prune the pair search by box overlap.
std::vector<char> self_intersections(const std::vector<Vec3>& V, const Tris& T, const Bvh& bvh)
{
    std::vector<char> flag(T.size(), 0);
    for (int i = 0; i < (int)T.size(); ++i)
    {
        const BvhBox bi = bvh_detail::triangle_box(V, T[i]);
        int stack[64], top = 0;
        stack[top++] = 0;
        while (top > 0)
        {
            const BvhNode& n = bvh.nodes[stack[--top]];
            if ((n.box.lo.array() > bi.hi.array()).any() || (bi.lo.array() > n.box.hi.array()).any()) continue;
            if (n.count == 0) { stack[top++] = n.left; stack[top++] = n.right; continue; }
            for (int k = 0; k < n.count; ++k)
            {
                const int j = bvh.tri_ids[n.first + k];
                if (j <= i) continue; // each pair once
                bool shares = false;
                for (int a : T[i]) for (int b : T[j]) shares = shares || a == b;
                if (!shares && crosses(V, T[i], T[j])) flag[i] = flag[j] = 1;
            }
        }
    }
    return flag;
}

// Per-triangle colors in one pass so the two layers can be drawn back to front without depth writes.
void draw_layer(const std::vector<Vec3>& V, const Tris& T, const std::vector<char>& flag, bool want_flagged, Color color)
{
    rlBegin(RL_TRIANGLES);
    rlColor4ub(color.r, color.g, color.b, color.a);
    for (size_t t = 0; t < T.size(); ++t)
    {
        if ((flag[t] != 0) != want_flagged) continue;
        const auto& tri = T[t];
        const Vec3 nrm = (V[tri[1]] - V[tri[0]]).cross(V[tri[2]] - V[tri[0]]).normalized();
        rlNormal3f((float)nrm.x(), (float)nrm.y(), (float)nrm.z());
        for (int k = 0; k < 3; ++k)
        {
            const Vector3 p = to_raylib(V[tri[k]]);
            rlVertex3f(p.x, p.y, p.z);
        }
    }
    rlEnd();
}
} // namespace si

int main(int argc, char** argv)
{
    const std::string path = argc > 1 ? argv[1] : std::string(COLLIDER_ANIM_DIR_DEFAULT) + "/fem_skinned_body_stand_to_walk.json";

    std::vector<Collider> cols;
    load_collider_animation(path, cols);
    if (skinned_meshes.empty()) { std::fprintf(stderr, "could not load a skinned body from %s\n", path.c_str()); return 1; }
    const SkinnedMesh& body = skinned_meshes[0];
    const si::Tris&    T    = body.rest.triangles;
    const int          NF   = body.n_frames;

    // Poses and intersection flags per frame, computed on first view and cached.
    std::vector<std::vector<Vec3>> pose(NF);
    std::vector<std::vector<char>> flags(NF);
    Bvh bvh;
    auto ensure = [&](int f)
    {
        if (!flags[f].empty()) return;
        skin_vertices(body, f, pose[f]);
        if (bvh.nodes.empty()) bvh_build(bvh, pose[f], T); else bvh_refit(bvh, pose[f], T);
        flags[f] = si::self_intersections(pose[f], T, bvh);
    };

    if (argc > 2 && std::string(argv[2]) == "--count") // headless: per-frame counts, no window
    {
        int worst = 0, worst_frame = 0, frames_with = 0;
        for (int f = 0; f < NF; ++f)
        {
            ensure(f);
            const int n = (int)std::count(flags[f].begin(), flags[f].end(), (char)1);
            frames_with += n > 0;
            if (n > worst) { worst = n; worst_frame = f; }
            if (f % 62 == 0) std::printf("frame %3d: %d intersecting triangles\n", f, n);
        }
        std::printf("%d / %d frames have intersections; worst frame %d with %d triangles\n", frames_with, NF, worst_frame, worst);
        return 0;
    }

    viewer_open();
    SetWindowTitle("skinned body self-intersections (temporary)");

    // Frame the first pose; the follow camera below keeps it centered afterwards.
    ensure(0);
    {
        Vec3 lo = pose[0][0], hi = pose[0][0];
        for (const Vec3& v : pose[0]) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
        g_viewer.orbit.target   = to_raylib(0.5 * (lo + hi));
        g_viewer.orbit.distance = 1.6f * (float)(hi - lo).maxCoeff() / tanf(kPerspectiveFovY * DEG2RAD * 0.5f) * 0.5f;
    }

    int    frame = 0;
    bool   playing = true, show_gray = true, follow = true;
    double acc = 0.0;
    const Color gray = { 170, 170, 170, 40 }, red = { 235, 40, 40, 215 };

    while (!WindowShouldClose())
    {
        if (IsKeyPressed(KEY_SPACE)) playing = !playing;
        if (IsKeyPressed(KEY_G)) show_gray = !show_gray;
        if (IsKeyPressed(KEY_F)) follow = !follow;
        if (IsKeyPressed(KEY_RIGHT) || IsKeyPressedRepeat(KEY_RIGHT)) { playing = false; frame = (frame + 1) % NF; }
        if (IsKeyPressed(KEY_LEFT)  || IsKeyPressedRepeat(KEY_LEFT))  { playing = false; frame = (frame + NF - 1) % NF; }
        if (playing)
        {
            acc += GetFrameTime();
            const double step = 1.0 / kColliderAnimFPS;
            while (acc >= step) { acc -= step; frame = (frame + 1) % NF; }
        }
        ensure(frame);
        const std::vector<Vec3>& V = pose[frame];
        const std::vector<char>& F = flags[frame];

        if (follow)
        {
            Vec3 lo = V[0], hi = V[0];
            for (const Vec3& v : V) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
            g_viewer.orbit.target = to_raylib(0.5 * (lo + hi));
        }
        update_orbit_camera(g_viewer.orbit, g_viewer.camera);

        BeginDrawing();
        ClearBackground(kBackgroundColor);
        BeginMode3D(g_viewer.camera);
        draw_axes(kAxisLength);
        BeginShaderMode(g_viewer.sphere_shader);
        rlDisableBackfaceCulling(); // see through to the far side too
        rlDisableDepthMask();       // transparent layers: test depth against the axes, don't write it
        BeginBlendMode(BLEND_ALPHA);
        if (show_gray) si::draw_layer(V, T, F, false, gray);
        si::draw_layer(V, T, F, true, red);
        rlDrawRenderBatchActive();
        EndBlendMode();
        rlEnableDepthMask();
        rlEnableBackfaceCulling();
        EndShaderMode();
        EndMode3D();

        int n_red = 0;
        for (char c : F) n_red += c;
        DrawText(TextFormat("frame %d / %d   %s   self-intersecting triangles: %d / %d", frame, NF - 1,
                            playing ? "playing" : "paused", n_red, (int)T.size()), 10, 10, 20, WHITE);
        DrawText("Space play/pause   Left/Right step   G gray on/off   F follow camera on/off", 10, 36, 16, LIGHTGRAY);
        EndDrawing();
    }

    viewer_close();
    return 0;
}
