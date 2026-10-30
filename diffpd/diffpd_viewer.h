#pragma once

#include "diffpd_types.h"
#include "diffpd_timer.h"

#include <string>

// Opens the viewer window (raylib InitWindow + shader/camera setup). Call once before any other
// viewer_* call; the camera/orbit state set up here persists across all later calls.
void viewer_open();

// Closes the window and releases GPU resources. Call once, after everything else is done.
void viewer_close();

// Updates what the next viewer_render_frame() call will draw. `reference_*` is the orange,
// half-transparent trajectory (the target, when known); `live_*` is the white trajectory currently
// playing out. Either side may be null. Frames, contacts and the collider list are copied internally; the
// world's meshes and skinned bodies are only referenced, so `world` must stay alive until the next
// viewer_set_scene call (or until the config screen opens, which forgets it).
void viewer_set_scene(const SimMesh& mesh,
                       const PointsX* reference_frame, const Contacts* reference_contacts,
                       const PointsX* live_frame, const Contacts* live_contacts,
                       const CollisionWorld& world, Real collider_time,
                       const std::string& status_text);

// Updates just the status line, keeping the scene from the last viewer_set_scene call. Used to
// keep the window responsive during headless phases with nothing new to show (e.g. FD reruns).
// `status_text_right` is drawn right-aligned to the window edge, so a fixed-position field (e.g. a
// time estimate) doesn't jitter as the left text changes width. viewer_set_scene clears it.
void viewer_set_status(const std::string& status_text, const std::string& status_text_right = "");

// Shows a scrollable config panel (left) plus a live 3D preview (right) of the initial condition —
// target cloth (orange) and guess cloth (white) overlaid, plus the selected collider — updating as
// fields change. Runs until "Run" is clicked (returns true, `cfg` holds the edits) or the window
// is closed (returns false). Acts as the gate before running any experiment.
//
// The "Animation" dropdown (None + every *.json in the animation folder, see list_animation_files)
// edits cfg.animation_file; the selected animation's colliders are previewed at frame 0 alongside
// cfg.colliders (display-only, not part of cfg). The folder is rescanned each time the screen opens.
// When the selected animation has a waist-attachment collider, checking "Waist Attachment" (or
// switching to such an animation with it checked) snaps cfg.origin to that collider's frame-0 position
// so the cloth spawns where it'll actually be pinned.
//
// The "Rest pose" section's "Settle cloth" button calls `on_settle(cfg, false)` ("Settle seams": `on_settle(cfg, true)`) between frames (it is expected to run a
// blocking simulation with its own live view and fill cfg.settled); returning false means the window was closed.
// The screen discards cfg.settled whenever settle_key(cfg) changes, and previews the settled pose while it is valid.
bool viewer_show_config_screen(AppConfig& cfg, const std::function<bool(AppConfig&, bool seams_only)>& on_settle = nullptr);

// Draws one frame of the current scene (axes, colliders, meshes+spheres, status text, FPS) and
// pumps camera/input. Returns false once the window is closed — callers should abort.
bool viewer_render_frame();

// Cheaply pumps OS/input events (just enough to notice a window close) without drawing anything.
// Call every physics step inside long blocking solves, where viewer_render_frame() only fires once
// per `frame_substeps` steps — otherwise a close request is only noticed at that coarser cadence,
// showing up as "Not Responding" during a slow solve. Same return convention as viewer_render_frame().
bool viewer_poll_close();

// Everything the playback screen needs only because a backward pass was run: the target trajectory
// and the gradient/FD/adjoint-residual displays. Passed to viewer_interactive_playback as a pointer;
// null = forward-only run, in which all of it (target layer, T key, gradient panel, FD results/panel
// and button, target + adjoint residual graphs) is hidden.
struct BackwardPlayback
{
    const Tape&            target_tape;
    const bool           (&fd_eps_seed)[9]; // seeds the on-screen FD-check panel's checkboxes (typically
                                            // cfg.fd_eps_selected), so a forgotten epsilon doesn't need a trip back to Setup
    const FDCheckCache&    fd_results;      // the experiment's FD cache (filled by `run_fd_check`, including any
                                            // up-front check), shown in a results panel under the help box
    const FDCheckRunner&   run_fd_check;    // the closure main() uses for the up-front check (see FDCheckRunner); it
                                            // skips already-cached epsilons and may pump the window internally, so
                                            // it's only ever called between frames
    const GradientSummary& grad;            // the backward pass's result (dphi/dx0, dphi/dv0, dphi/dk), shown in a
                                            // fixed panel throughout playback
};

// Runs the full interactive playback loop (pause/scrub/T-G-E-C toggles, orbit/pan/zoom, a "Back to
// Setup" button) over the completed guess tape (plus the target tape when `backward` is given).
// Assumes viewer_open() was already called. `dt` lets
// the viewer reconstruct each collider's pose per displayed frame, mirroring detect_contacts's own
// time convention. Returns true if "Back to Setup" was clicked (caller re-enters the config
// screen), false if the window was closed (caller should quit).
//
// `backward` — see BackwardPlayback; null for a forward-only run (`residuals` then only needs
// guess_forward). `residuals` are the per-step forward/backward convergence residuals, drawn as
// line graphs with a marker on the current frame.
// `forward_timing` is the guess forward pass's per-stage time breakdown (StageTimer::shares), drawn as a
// small panel in the bottom-right corner (empty = no panel).
// `world` supplies the colliders and lets `animated` ones (and skinned bodies) be re-posed as the timeline
// scrubs/plays: without it they'd sit frozen at their final simulated pose (the tape's StepClock maps
// the tape index to track time, so substepped runs play back at the right speed). Playback restamps the
// shared skinned-body state, hence non-const. `waist` is the guess object's waist attachment (rigid hip
// offsets or skinned-body pin bindings), needed for the same reason to keep the waistband following the
// timeline. The default disables it.
bool viewer_interactive_playback(const SimMesh& mesh, const Tape& guess_tape, const BackwardPlayback* backward,
                                  CollisionWorld& world, Real dt, int frame_substeps, int fps,
                                  const ResidualHistory& residuals,
                                  const std::vector<StageShare>& forward_timing,
                                  const WaistAttachment& waist = {});
