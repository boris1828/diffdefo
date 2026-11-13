cmake_minimum_required(VERSION 3.25)
project(diffpd LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to a Release build on single-config generators (e.g. Ninja/Makefiles) if
# the user didn't pick one. Multi-config generators (Visual Studio) ignore this and
# select the config via --config at build time instead.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Put executables in build/bin so Windows doesn't scatter them per-config
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Eigen (header-only, vendored in external/eigen) ---
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE
    ${CMAKE_SOURCE_DIR}/../external/eigen
)

# --- nlohmann/json (header-only, vendored in external/nlohmann_json) ---
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
    ${CMAKE_SOURCE_DIR}/../external/nlohmann_json
)

# --- raylib (vendored in external/raylib, built from source) ---
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/../external/raylib ${CMAKE_BINARY_DIR}/raylib_build)

# --- Modules: one CMake target per folder ------------------------------------------------------------------
# Order (each may use only the ones before it): core < energy, meshgen, collision < cloth < solver < app < viewer < driver.
# Every module sees the project root as its include path, so CMake alone cannot stop a wrong #include: tools/check_includes.py does.

# core: the include root, Eigen, the output folders, the warning / optimisation flags every module is built with.
add_library(diffpd_core INTERFACE)
target_include_directories(diffpd_core INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(diffpd_core INTERFACE eigen)
target_compile_definitions(diffpd_core INTERFACE
    ANIM_DIR_DEFAULT="${CMAKE_SOURCE_DIR}/../animation"
    RECORDINGS_DIR_DEFAULT="${CMAKE_SOURCE_DIR}/../recordings"
    COLLIDER_ANIM_DIR_DEFAULT="${CMAKE_SOURCE_DIR}/animation"
    CONFIGS_DIR_DEFAULT="${CMAKE_SOURCE_DIR}/configs"
    PATTERNS_DIR_DEFAULT="${CMAKE_SOURCE_DIR}/patterns"
)
if(MSVC)
    target_compile_options(diffpd_core INTERFACE /W4 /permissive-)
    # Eigen triggers a lot of harmless MSVC warnings; silence the noisiest
    target_compile_definitions(diffpd_core INTERFACE _CRT_SECURE_NO_WARNINGS)
    # AVX2+FMA vectorization for Eigen's dense Vec3/Mat3 math and sparse mat-vec
    # ops (MSVC defaults to SSE2 only). Release-only: unlike -O3/-DNDEBUG (which
    # MSVC's /O2 Release config already implies), /arch:AVX2 changes the set of
    # instructions emitted, so it's kept out of Debug/RelWithDebInfo builds.
    # target_compile_options(diffpd_core INTERFACE $<$<CONFIG:Release>:/arch:AVX2>)
else()
    target_compile_options(diffpd_core INTERFACE -Wall -Wextra -Wpedantic)
    # Same rationale as the MSVC branch above: -O3 -DNDEBUG duplicate what
    # CMAKE_CXX_FLAGS_RELEASE already provides but are made explicit here for
    # robustness; -march=native additionally enables AVX2/FMA on this machine.
    target_compile_options(diffpd_core INTERFACE $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>)
endif()

# energy: header-only.
add_library(diffpd_energy INTERFACE)
target_link_libraries(diffpd_energy INTERFACE diffpd_core)

# meshgen: header-only (pattern_io.h reads json, tri_surface.h uses the vendored delaunator).
add_library(diffpd_meshgen INTERFACE)
target_include_directories(diffpd_meshgen INTERFACE ${CMAKE_SOURCE_DIR}/../external/delaunator)
target_link_libraries(diffpd_meshgen INTERFACE diffpd_core nlohmann_json)

# collision: header-only shapes and detection, plus the animation loader (animation_io.cpp).
add_library(diffpd_collision STATIC collision/animation_io.cpp)
target_link_libraries(diffpd_collision PUBLIC diffpd_core nlohmann_json)

# cloth: the object and the three factories.
add_library(diffpd_cloth STATIC cloth/quad.cpp cloth/skirt.cpp cloth/tri_cloth.cpp)
target_link_libraries(diffpd_cloth PUBLIC diffpd_core diffpd_energy diffpd_meshgen diffpd_collision)

# solver: header-only for now (inline functions); step D2 of the refactor plan makes it a library.
add_library(diffpd_solver INTERFACE)
target_link_libraries(diffpd_solver INTERFACE diffpd_cloth diffpd_collision)

# app: the config and the run helpers, header-only for now.
add_library(diffpd_app INTERFACE)
target_link_libraries(diffpd_app INTERFACE diffpd_solver diffpd_cloth diffpd_collision diffpd_meshgen)

# viewer: the only module that sees raylib / raygui.
add_library(diffpd_viewer STATIC diffpd_viewer.cpp)
target_include_directories(diffpd_viewer PRIVATE
    ${CMAKE_SOURCE_DIR}/../external/raygui/src
    ${CMAKE_SOURCE_DIR}/../external/raygui/styles/dark
)
target_link_libraries(diffpd_viewer PUBLIC diffpd_app raylib)

# Eigen and json dominate the compile time: parse them once per library that compiles them.
foreach(lib diffpd_collision diffpd_cloth diffpd_viewer)
    target_precompile_headers(${lib} PRIVATE <Eigen/Dense> <Eigen/Sparse> <Eigen/SparseCholesky> <nlohmann/json.hpp>)
endforeach()

# driver: the program. diffpd.cpp is main() and the check dispatch; the run pipeline is in driver/.
add_executable(diffpd diffpd.cpp)
target_precompile_headers(diffpd PRIVATE <Eigen/Dense> <Eigen/Sparse> <Eigen/SparseCholesky> <nlohmann/json.hpp>)
target_link_libraries(diffpd PRIVATE diffpd_viewer diffpd_app)
if(MSVC)
    target_compile_options(diffpd PRIVATE /bigobj) # diffpd.cpp includes every check header: past MSVC's default section limit (C1128)
endif()

# --- Throwaway headless experiments (optimization_experiment/): stiffness_opt ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/optimization_experiment/CMakeLists.txt") # git-ignored: absent in a fresh clone
    add_subdirectory(optimization_experiment EXCLUDE_FROM_ALL) # not in the default build: cd build/optimization_experiment, then cmake --build . --config Release --target stiffness_opt (the Visual Studio generator cannot build it from build/)
endif()
