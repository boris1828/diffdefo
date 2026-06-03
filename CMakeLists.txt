cmake_minimum_required(VERSION 3.16)
project(xpbd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Put executables in build/bin so Windows doesn't scatter them per-config
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Eigen (header-only, vendored in external/eigen) ---
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE
    ${CMAKE_SOURCE_DIR}/external/eigen
)

# --- Polyscope ---
# add_subdirectory(external/polyscope)
# get_target_property(POLYSCOPE_INCLUDES polyscope INTERFACE_INCLUDE_DIRECTORIES)
# set_target_properties(polyscope PROPERTIES
#     INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${POLYSCOPE_INCLUDES}")

# --- Main executable ---
add_executable(xpbd
    src/main.cpp
)

target_include_directories(xpbd PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(xpbd PRIVATE
    eigen
    # polyscope
)

# --- Warnings & optimization flags ---
if(MSVC)
    target_compile_options(xpbd PRIVATE /W4 /permissive-)
    # Eigen triggers a lot of harmless MSVC warnings; silence the noisiest
    target_compile_definitions(xpbd PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(xpbd PRIVATE -Wall -Wextra -Wpedantic)
endif()