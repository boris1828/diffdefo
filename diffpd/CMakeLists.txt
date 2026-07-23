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

# --- raylib (vendored in external/raylib, built from source) ---
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/../external/raylib ${CMAKE_BINARY_DIR}/raylib_build)

# --- Main executable (simulation + backward pass + viewer, all in one binary) ---
add_executable(diffpd
    diffpd.cpp
    diffpd_viewer.cpp
)

target_include_directories(diffpd PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../external/raygui/src
    ${CMAKE_SOURCE_DIR}/../external/raygui/styles/dark
)

target_link_libraries(diffpd PRIVATE
    eigen
    raylib
)

target_compile_definitions(diffpd PRIVATE
    ANIM_DIR_DEFAULT="${CMAKE_SOURCE_DIR}/../animation"
)

# --- Warnings & optimization flags ---
if(MSVC)
    target_compile_options(diffpd PRIVATE /W4 /permissive-)
    # Eigen triggers a lot of harmless MSVC warnings; silence the noisiest
    target_compile_definitions(diffpd PRIVATE _CRT_SECURE_NO_WARNINGS)
    # AVX2+FMA vectorization for Eigen's dense Vec3/Mat3 math and sparse mat-vec
    # ops (MSVC defaults to SSE2 only). Release-only: unlike -O3/-DNDEBUG (which
    # MSVC's /O2 Release config already implies), /arch:AVX2 changes the set of
    # instructions emitted, so it's kept out of Debug/RelWithDebInfo builds.
    # target_compile_options(diffpd PRIVATE $<$<CONFIG:Release>:/arch:AVX2>)
else()
    target_compile_options(diffpd PRIVATE -Wall -Wextra -Wpedantic)
    # Same rationale as the MSVC branch above: -O3 -DNDEBUG duplicate what
    # CMAKE_CXX_FLAGS_RELEASE already provides but are made explicit here for
    # robustness; -march=native additionally enables AVX2/FMA on this machine.
    target_compile_options(diffpd PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native>)
endif()
