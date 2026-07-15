#include "raylib.h"

int main()
{
    InitWindow(800, 450, "diffpd viewer - raylib smoke test");
    SetTargetFPS(60);

    Camera3D camera = { 0 };
    camera.position   = { 4.0f, 4.0f, 4.0f };
    camera.target     = { 0.0f, 0.0f, 0.0f };
    camera.up         = { 0.0f, 1.0f, 0.0f };
    camera.fovy       = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    const int total_frames = 30;
    int frame = 0;
    while (!WindowShouldClose() && frame < total_frames)
    {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        BeginMode3D(camera);
        DrawGrid(10, 1.0f);
        DrawCube({ 0.0f, 0.0f, 0.0f }, 2.0f, 2.0f, 2.0f, RED);
        DrawCubeWires({ 0.0f, 0.0f, 0.0f }, 2.0f, 2.0f, 2.0f, MAROON);
        EndMode3D();

        DrawText("raylib smoke test OK", 10, 10, 20, DARKGRAY);
        EndDrawing();

        if (frame == total_frames - 1)
            TakeScreenshot("diffpd_viewer_smoke_test.png");

        frame++;
    }

    CloseWindow();
    return 0;
}
