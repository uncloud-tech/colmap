# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "exe")

if(IS_MSVC)
    add_compile_options("/bigobj")
endif()

set(OPTIONAL_LIBS)
if(CUDA_ENABLED)
    list(APPEND OPTIONAL_LIBS colmap_util_cuda)
    if(MVS_ENABLED)
        list(APPEND OPTIONAL_LIBS colmap_mvs_cuda)
    endif()
endif()
if(HIP_ENABLED)
    list(APPEND OPTIONAL_LIBS colmap_util_cuda)
    if(MVS_ENABLED)
        list(APPEND OPTIONAL_LIBS colmap_mvs_cuda)
    endif()
endif()
if(GUI_ENABLED)
    list(APPEND OPTIONAL_LIBS
        colmap_ui
    )
endif()

set(COLMAP_EXE_SRCS
    feature.h feature.cc
    image.h image.cc
    sfm.h sfm.cc
    model.h model.cc
)
set(COLMAP_EXE_PUBLIC_LIBS
    colmap_controllers
    colmap_estimators
    colmap_geometry
    colmap_optim
    colmap_scene
    colmap_sfm
    colmap_util
)
if(MVS_ENABLED)
    list(APPEND COLMAP_EXE_SRCS mvs.h mvs.cc)
    list(APPEND COLMAP_EXE_PUBLIC_LIBS colmap_mvs)
endif()

COLMAP_ADD_LIBRARY(
    NAME colmap_exe
    SRCS
        ${COLMAP_EXE_SRCS}
    PUBLIC_LINK_LIBS
        ${COLMAP_EXE_PUBLIC_LIBS}
    PRIVATE_LINK_LIBS
        ${COLMAP_BOOST_HEADER_LIBS}
        colmap_sfm
)

COLMAP_ADD_TEST(
    NAME image_test
    SRCS image_test.cc
    LINK_LIBS colmap_exe
)

COLMAP_ADD_TEST(
    NAME model_test
    SRCS model_test.cc
    LINK_LIBS colmap_exe ${COLMAP_BOOST_HEADER_LIBS}
)

COLMAP_ADD_TEST(
    NAME sfm_test
    SRCS sfm_test.cc
    LINK_LIBS colmap_exe
)

set(COLMAP_MAIN_SRCS
    colmap.cc
    database.cc
    feature.cc
    gui.cc
    image.cc
    model.cc
    sfm.cc
    vocab_tree.cc
)
if(MVS_ENABLED)
    list(APPEND COLMAP_MAIN_SRCS mvs.cc)
endif()

COLMAP_ADD_EXECUTABLE(
    NAME colmap_main
    SRCS
        ${COLMAP_MAIN_SRCS}
    LINK_LIBS
        colmap_controllers
        colmap_retrieval
        colmap_scene
        colmap_sfm
        colmap_util
        ${OPTIONAL_LIBS}
        ${COLMAP_BOOST_HEADER_LIBS}
)
if(HIP_ENABLED)
    target_link_libraries(colmap_main hip::host hip::hiprand roc::rocrand)
endif()
set_target_properties(colmap_main PROPERTIES OUTPUT_NAME colmap)

option(OWNED_PATCH_MATCH_DRIVER_ENABLED
       "Build the experimental CUDA-only single-reference owned driver" OFF)
if(OWNED_PATCH_MATCH_DRIVER_ENABLED)
    if(NOT CUDA_ENABLED OR NOT MVS_ENABLED OR HIP_ENABLED)
        message(FATAL_ERROR "Owned reference driver requires CUDA MVS, not HIP")
    endif()
    COLMAP_ADD_EXECUTABLE(
        NAME colmap_owned_reference
        SRCS owned_reference.cc
        LINK_LIBS colmap_exe colmap_mvs_cuda ${COLMAP_BOOST_HEADER_LIBS}
    )
endif()

option(OWNED_PATCH_MATCH_PAIR_DRIVER_ENABLED
       "Build the experimental finite TWO-reference owned driver" OFF)
if(OWNED_PATCH_MATCH_PAIR_DRIVER_ENABLED)
    if(NOT CUDA_ENABLED OR NOT MVS_ENABLED OR HIP_ENABLED OR
       NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "Owned pair driver requires Linux CUDA MVS, not HIP")
    endif()
    COLMAP_ADD_EXECUTABLE(
        NAME colmap_owned_pair
        SRCS owned_pair.cc
        LINK_LIBS colmap_exe colmap_mvs_cuda ${COLMAP_BOOST_HEADER_LIBS}
    )
    # CPU-only callbacks; including the driver still requires its link deps.
    COLMAP_ADD_TEST(
        NAME owned_pair_test
        SRCS owned_pair_test.cc
        LINK_LIBS colmap_exe colmap_mvs_cuda ${COLMAP_BOOST_HEADER_LIBS}
    )
    if(TESTS_ENABLED)
        set_tests_properties(exe/owned_pair_test PROPERTIES TIMEOUT 30)
    endif()
endif()
