# SPDX-License-Identifier: BSD-3-Clause
# Stdlib-only frontend: deliberately independent of pycolmap's wheel toolchain.
set(_reconstruction_python "${CMAKE_INSTALL_DATAROOTDIR}/colmap/python")
file(RELATIVE_PATH RECONSTRUCTION_MODULE_PATH
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
    "${CMAKE_INSTALL_PREFIX}/${_reconstruction_python}/launch.py")
configure_file("${PROJECT_SOURCE_DIR}/python/reconstruct.in"
    "${CMAKE_CURRENT_BINARY_DIR}/colmap-reconstruct" @ONLY)
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/colmap-reconstruct"
    DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES "${PROJECT_SOURCE_DIR}/python/reconstruction_launch.py"
    DESTINATION "${_reconstruction_python}" RENAME launch.py)
install(FILES
    "${PROJECT_SOURCE_DIR}/python/colmap_reconstruction/__init__.py"
    "${PROJECT_SOURCE_DIR}/python/colmap_reconstruction/cli.py"
    "${PROJECT_SOURCE_DIR}/python/colmap_reconstruction/process.py"
    "${PROJECT_SOURCE_DIR}/python/colmap_reconstruction/resource_policy.py"
    DESTINATION "${_reconstruction_python}/colmap_reconstruction")
install(FILES "${PROJECT_SOURCE_DIR}/python/colmap_reconstruction/NOTICE"
    "${PROJECT_SOURCE_DIR}/LICENSE"
    DESTINATION "${_reconstruction_python}/colmap_reconstruction")
install(FILES "${PROJECT_SOURCE_DIR}/doc/reconstruction-cli.md"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/colmap")

if(TESTS_ENABLED AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME reconstruction/frontend_test
        COMMAND "${CMAKE_COMMAND}" -E env
            "PYTHONPATH=${PROJECT_SOURCE_DIR}/python:${PROJECT_SOURCE_DIR}/python/tests"
            "${Python3_EXECUTABLE}" -m unittest
            colmap_reconstruction.reconstruction_test test_reconstruction)
endif()
