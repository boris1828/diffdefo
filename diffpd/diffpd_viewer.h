#pragma once

#include "diffpd_types.h"

#include <string>

// Opens the viewer window (raylib InitWindow + shader/camera setup). Persistent across the whole
// program: call once before any other viewer_* call. The camera/orbit state set up here persists
// across every subsequent viewer_set_scene/viewer_render_frame/viewer_interactive_playback call.
void viewer_open();

// Closes the window and releases GPU resources. Call once, after everything else (including
// viewer_interactive_playback) is done.
void viewer_close();

// Updates what the next viewer_render_frame() call will draw. `reference_*` is the orange,
// half-transparent trajectory (the target, when already known/computed); `live_*` is the white
// trajectory currently playing out (the target itself during the target sim, the guess during the
// guess sim and backward pass). Either side may be null to omit it. Frame/contact data is copied
// into internal state, so callers don't need to keep the arguments alive past this call.
void viewer_set_scene(const SimMesh& mesh,
                       const PointsX* reference_frame, const Contacts* reference_contacts,
                       const PointsX* live_frame, const Contacts* live_contacts,
                       const Collider& collider, Real collider_time,
                       const std::string& status_text);

// Updates just the status line, keeping whatever scene was last set via viewer_set_scene. Used to
// keep the window responsive (pumping input/redrawing) during headless phases that have nothing
// new to show, e.g. the FD-check reruns.
void viewer_set_status(const std::string& status_text);

// Shows a scrollable config panel (left) with every editable experiment parameter, and a live 3D
// preview (right) of the initial condition — target cloth (orange) and guess cloth (white)
// overlaid, plus the selected collider — that updates as fields change, with the orbit camera
// active over the viewport region. Runs until "Run" is clicked (returns true, `cfg` holds the
// edited values) or the window is closed (returns false). Assumes viewer_open() was already
// called. Meant as the initial gate before running any experiment.
bool viewer_show_config_screen(AppConfig& cfg);

// Draws one frame of the current scene (axes, collider, reference/live meshes+spheres, status
// text, FPS) and pumps camera/input. Returns false once the window has been closed by the user —
// callers should treat that as "abort everything".
bool viewer_render_frame();

// Cheaply pumps OS/input events (just enough to notice the window being closed) without drawing
// anything. Meant to be called every physics step inside long blocking solves (pd_contact's
// n_steps loop, FD-check reruns, etc.), where the full viewer_render_frame() only fires once every
// `frame_substeps` steps — without this, a close request (the window's X button, or any future
// on-screen Quit control) would only be noticed at that coarser cadence, and for a slow solve
// (many steps/iterations) that shows up as the window going "Not Responding" for a stretch before
// it finally closes. Returns false once the window has been closed, same convention as
// viewer_render_frame().
bool viewer_poll_close();

// Runs the full interactive playback loop (pause/scrub/T-G-E-C toggles, orbit/pan/zoom help box,
// a "Back to Setup" button) over the two completed tapes. Assumes viewer_open() was already
// called; does not itself open or close the window. `collider` and `dt` let the viewer reconstruct
// the collider's position at each displayed frame (time = frame * frame_substeps * dt), mirroring
// detect_contacts's own time-shift for a moving collider. Returns true if "Back to Setup" was
// clicked (caller should return to viewer_show_config_screen() and re-run), false if the window
// was closed instead (caller should quit).
//
// `fd_eps_seed` seeds the on-screen FD-check panel's checkboxes (typically cfg.fd_eps_selected, so
// re-running here starts from whatever was last selected on the config screen) — offered here too
// so a forgotten or different epsilon doesn't require going back to Setup and recomputing the whole
// trajectory. `run_fd_check` is the same closure main() uses for the up-front check (see
// FDCheckRunner in diffpd_types.h); it may pump the window internally (long-running), so it's only
// ever invoked between frames, never from inside this function's own BeginDrawing/EndDrawing block.
bool viewer_interactive_playback(const SimMesh& mesh, const Tape& target_tape, const Tape& guess_tape,
                                  const Collider& collider, Real dt, int frame_substeps, int fps,
                                  const bool (&fd_eps_seed)[9], const FDCheckRunner& run_fd_check);
