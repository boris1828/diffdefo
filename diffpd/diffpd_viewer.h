#pragma once

#include "diffpd_types.h"

// Opens a window and loops playback of target_tape and guess_tape overlaid, both
// reconstructed from the given SimMesh (target and guess are assumed to share topology).
// Only every frame_substeps-th tape entry is shown (matching the recorded output frames),
// played back at fps. `collider` and `dt` let the viewer reconstruct the collider's
// position at each displayed frame (time = frame * frame_substeps * dt), mirroring
// detect_contacts's own time-shift for a moving collider. Blocks until the window is closed.
void play_tapes(const SimMesh& mesh, const Tape& target_tape, const Tape& guess_tape,
                 const Collider& collider, Real dt, int frame_substeps, int fps);
