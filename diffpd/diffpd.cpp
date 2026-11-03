#include <map>
#include "app/config.h"
#include "app/fd_types.h"
#include "app/fd_check.h"
#include "app/run.h"
#include "app/settle.h"
#include "app/world_setup.h"
#include "cloth/obj_output.h"
#include "cloth/waist.h"
#include "collision/animation_io.h"
#include "collision/detect.h"
#include "solver/backward.h"
#include "solver/forward.h"
#include "solver/loss.h"
#include "solver/velocity_solver.h"
#include "solver/tape.h"
#include "diffpd_viewer.h"
#include "core/timer.h"
#include "anderson.h"
#include "collision/skinned_mesh_io.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <nlohmann/json.hpp>

#include <vector>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <limits>
#include <functional>
#include <cmath>

namespace fs = std::filesystem;

#include "diffpd_checks.h"

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "--regression")        return run_regression();
    if (argc > 1 && std::string(argv[1]) == "--meshgen-check")     return run_meshgen_check();
    if (argc > 1 && std::string(argv[1]) == "--arap-check")        return run_arap_check();
    if (argc > 1 && std::string(argv[1]) == "--shape-check")       return run_shape_check();
    if (argc > 1 && std::string(argv[1]) == "--speculative-check") return run_speculative_check();
    if (argc > 2 && std::string(argv[1]) == "--fd-config")         return run_fd_config(argv[2], argc > 3 && std::string(argv[3]) == "--settle");
    if (argc > 1 && std::string(argv[1]) == "--anderson-check")    return argc > 2 ? run_anderson_compare(argv[2]) : run_anderson_check();
    if (argc > 2 && std::string(argv[1]) == "--chunk-check")       return run_chunk_check(argv[2]);
    if (argc > 3 && std::string(argv[1]) == "--contact-diff")
        return run_contact_diff(argv[2], std::stod(argv[3]), argc > 4 && std::string(argv[4]) == "--settle");

    return run_app();
}
