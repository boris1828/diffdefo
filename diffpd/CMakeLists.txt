cmake_minimum_required(VERSION 3.25)
project(diffpd LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# --- Main executable ---
add_executable(diffpd
    diffpd.cpp
)

target_include_directories(diffpd PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(diffpd PRIVATE
    eigen
    # polyscope
)

target_compile_definitions(diffpd PRIVATE
    ANIM_DIR_DEFAULT="${CMAKE_SOURCE_DIR}/../animation"
)

# --- Warnings & optimization flags ---
if(MSVC)
    target_compile_options(diffpd PRIVATE /W4 /permissive-)
    # Eigen triggers a lot of harmless MSVC warnings; silence the noisiest
    target_compile_definitions(diffpd PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(diffpd PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- Viewer executable (raylib smoke test for now; real viewer comes later) ---
add_executable(diffpd_viewer
    diffpd_viewer.cpp
)

target_link_libraries(diffpd_viewer PRIVATE
    raylib
)
