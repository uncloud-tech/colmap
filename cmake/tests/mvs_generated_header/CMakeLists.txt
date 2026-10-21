# SPDX-License-Identifier: BSD-3-Clause

# Dependency-light regression for generated-header usage requirements. Run:
# cmake -S cmake/tests/mvs_generated_header -B <build> -G Ninja
# cmake --build <build>
# This is a compile/link guard only; inherited CTest entries use stub bodies,
# not the native test implementations, and should not be run as product tests.
cmake_minimum_required(VERSION 3.20)
project(MvsGeneratedHeaderTest LANGUAGES CXX)

enable_testing()
get_filename_component(COLMAP_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
# The production generator resolves its frozen closure relative to this root.
set(PROJECT_SOURCE_DIR "${COLMAP_SOURCE_DIR}")
include("${COLMAP_SOURCE_DIR}/cmake/CMakeHelper.cmake")
set(MVS_ENABLED ON)
set(TESTS_ENABLED ON)
set(CUDA_ENABLED OFF)
set(HIP_ENABLED OFF)
set(GUI_ENABLED OFF)
set(CGAL_ENABLED OFF)
set(OPENMP_FOUND OFF)
set(OWNED_PATCH_MATCH_DRIVER_ENABLED OFF CACHE BOOL "" FORCE)
set(OWNED_PATCH_MATCH_PAIR_DRIVER_ENABLED OFF CACHE BOOL "" FORCE)

foreach(dependency IN ITEMS controllers estimators geometry optim scene sfm util
                            math sensor image poisson_recon retrieval gtest_main)
    add_library(colmap_${dependency} INTERFACE)
endforeach()
add_library(OpenImageIO::OpenImageIO INTERFACE IMPORTED)
add_library(Eigen3::Eigen INTERFACE IMPORTED)

# Exercise the real targets, links, include roots and generation dependencies.
add_subdirectory("${COLMAP_SOURCE_DIR}/src/colmap/mvs" mvs)
add_subdirectory("${COLMAP_SOURCE_DIR}/src/colmap/exe" exe)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/consumer.cc"
    "#include \"colmap/mvs/generated/resident_allocation_fingerprint.h\"\nint main() { return 0; }\n")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/plain_consumer.cc"
    "int main() { return 0; }\n")
# Replace only translation units, avoiding the native dependency toolchain.
# Independent header-only tests deliberately do not consume MVS usage requirements.
foreach(directory IN ITEMS mvs exe)
    get_property(targets DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${directory}"
                 PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_target_property(type ${target} TYPE)
        if(NOT type STREQUAL "UTILITY")
            get_target_property(links ${target} LINK_LIBRARIES)
            if(target STREQUAL "colmap_mvs" OR target STREQUAL "colmap_main"
                    OR "colmap_mvs" IN_LIST links)
                set_property(TARGET ${target} PROPERTY SOURCES
                    "${CMAKE_CURRENT_BINARY_DIR}/consumer.cc")
            else()
                set_property(TARGET ${target} PROPERTY SOURCES
                    "${CMAKE_CURRENT_BINARY_DIR}/plain_consumer.cc")
            endif()
        endif()
    endforeach()
endforeach()
