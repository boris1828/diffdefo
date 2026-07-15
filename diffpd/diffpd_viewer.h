#pragma once

#include "diffpd_types.h"

// Opens a window and loops playback of target_tape and guess_tape overlaid, both
// reconstructed from the given SimMesh (target and guess are assumed to share topology).
// Blocks until the window is closed. Currently a no-op placeholder.
void play_tapes(const SimMesh& mesh, const Tape& target_tape, const Tape& guess_tape);
