#include <map>
#include "app/config.h"
#include "app/fd_types.h"
#include "app/fd_check.h"
#include "driver/run.h"
#include "driver/settle.h"
#include "app/world_setup.h"
#include "cloth/obj_output.h"
#include "cloth/waist.h"
#include "collision/animation_io.h"
#include "collision/detect.h"
#include "solver/impulse/backward.h"
#include "solver/impulse/forward.h"
#include "solver/loss.h"
#include "solver/pd_system.h"
#include "solver/tape.h"
#include "diffpd_viewer.h"
#include "core/timer.h"
#include "solver/impulse/anderson.h"
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
#include <random>

namespace fs = std::filesystem;

#include "diffpd_checks.h"
#include "diffpd_ncp_check.h"
#include "diffpd_friction_check.h"

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "--regression")        return run_regression();
    if (argc > 1 && std::string(argv[1]) == "--meshgen-check")     return run_meshgen_check();
    if (argc > 1 && std::string(argv[1]) == "--arap-check")        return run_arap_check();
    if (argc > 1 && std::string(argv[1]) == "--shape-check")       return run_shape_check();
    if (argc > 1 && std::string(argv[1]) == "--global-solve-check") return run_global_solve_check();
    if (argc > 1 && std::string(argv[1]) == "--rail-check")        return run_rail_check();
    if (argc > 1 && std::string(argv[1]) == "--collider-grad-check") return run_collider_grad_check();
    if (argc > 1 && std::string(argv[1]) == "--ncp-check")         return run_ncp_check();
    if (argc > 1 && std::string(argv[1]) == "--friction-check")    return run_friction_check();
    if (argc > 1 && std::string(argv[1]) == "--speculative-check") return run_speculative_check();
    if (argc > 2 && std::string(argv[1]) == "--fd-config")         return run_fd_config(argv[2], argc > 3 && std::string(argv[3]) == "--settle");
    if (argc > 1 && std::string(argv[1]) == "--anderson-check")    return argc > 2 ? run_anderson_compare(argv[2]) : run_anderson_check();
    if (argc > 2 && std::string(argv[1]) == "--chunk-check")       return run_chunk_check(argv[2]);
    if (argc > 3 && std::string(argv[1]) == "--contact-diff")
        return run_contact_diff(argv[2], std::stod(argv[3]), argc > 4 && std::string(argv[4]) == "--settle");

    return run_app();
}
