# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.10)
project(reconstruction_install NONE)
set(CMAKE_INSTALL_BINDIR bin)
set(CMAKE_INSTALL_DATAROOTDIR share)
set(PROJECT_SOURCE_DIR "${PRODUCT_ROOT}")
include("${PRODUCT_ROOT}/python/install_reconstruction.cmake")
