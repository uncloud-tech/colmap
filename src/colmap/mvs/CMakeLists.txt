# SPDX-License-Identifier: BSD-3-Clause

set(FOLDER_NAME "mvs")

# Compiled allocation-closure fingerprint: generated at build time from
# the frozen explicit ordered closure by scripts/resident/generate_
# resident_allocation_fingerprint.py. The generated header is excluded
# from the closure (no self-reference); the build fails closed on missing
# closure entries and never invents a digest.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(RESIDENT_ALLOCATION_GENERATOR
    "${PROJECT_SOURCE_DIR}/scripts/resident/generate_resident_allocation_fingerprint.py")
# Emitted under <build>/colmap/mvs/generated/ with the include root at the
# build dir, so consumers include
# "colmap/mvs/generated/resident_allocation_fingerprint.h".
set(RESIDENT_ALLOCATION_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/colmap/mvs/generated/resident_allocation_fingerprint.h")
set(RESIDENT_ALLOCATION_DEPFILE
    "${CMAKE_CURRENT_BINARY_DIR}/resident_allocation_fingerprint.d")
add_custom_command(
    OUTPUT "${RESIDENT_ALLOCATION_HEADER}"
    COMMAND ${Python3_EXECUTABLE} "${RESIDENT_ALLOCATION_GENERATOR}"
            --repo-root "${PROJECT_SOURCE_DIR}"
            --output "${RESIDENT_ALLOCATION_HEADER}"
            --depfile "${RESIDENT_ALLOCATION_DEPFILE}"
    DEPENDS "${RESIDENT_ALLOCATION_GENERATOR}"
    DEPFILE "${RESIDENT_ALLOCATION_DEPFILE}"
    COMMENT "Generating the compiled resident allocation fingerprint"
    VERBATIM)
add_custom_target(resident_allocation_fingerprint
    DEPENDS "${RESIDENT_ALLOCATION_HEADER}")

# Recompute PHOTO from committed HEAD on every build; refuse dirty closure
# sources rather than compiling a stale provenance header. No pinned digest.
set(PHOTO_PHASE_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/colmap/mvs/generated/photo_phase_fingerprint.h")
add_custom_target(photo_phase_fingerprint
    COMMAND ${Python3_EXECUTABLE}
            "${PROJECT_SOURCE_DIR}/scripts/resident/generate_photo_phase_fingerprint.py"
            --repo-root "${PROJECT_SOURCE_DIR}" --output "${PHOTO_PHASE_HEADER}"
    BYPRODUCTS "${PHOTO_PHASE_HEADER}"
    VERBATIM)

COLMAP_ADD_LIBRARY(
    NAME colmap_mvs
    SRCS
        owned_prng_seed.h
        advancing_front_meshing.h advancing_front_meshing.cc
        consistency_graph.h consistency_graph.cc
        delaunay_meshing.h delaunay_meshing.cc
        depth_map.h depth_map.cc
        fusion.h fusion.cc
        fusion_dependencies.h
        fusion_mapped_mat.h fusion_mapped_mat.cc
        image.h image.cc
        mat.h mat.cc
        mesh_simplification.h mesh_simplification.cc
        model.h model.cc
        poisson_meshing.h poisson_meshing.cc
        normal_map.h normal_map.cc
        owned_input_bank.h owned_input_bank.cc
        owned_cohort.h
        optimized_photo_delivery.h
        optimized_photo_options.h
        optimized_photo_paths.h
        owned_rotation_memory.h
        owned_resident_minima.h
        resident_forecast.h resident_forecast.cc resident_forecast_minima.cc
        resident_wire_lexical.h
        resident_cohort_wire.h resident_cohort_wire.cc
        resident_cohort_wire_minima.cc
        resident_cohort_runner.h
        resident_cohorts.h
        resident_runner.h
        native_run_deadline.h
        resident_contract.h resident_contract.cc
        patch_match_options.h patch_match_options.cc
        patch_match_execution.h
        texture_mapping.h texture_mapping.cc
        workspace.h workspace.cc
    PUBLIC_LINK_LIBS
        colmap_util
        colmap_scene
    PRIVATE_LINK_LIBS
        colmap_math
        colmap_sensor
        colmap_image
        colmap_poisson_recon
        OpenImageIO::OpenImageIO
        Eigen3::Eigen
)
if(CGAL_ENABLED)
    target_link_libraries(colmap_mvs PRIVATE CGAL)
endif()
if(OPENMP_FOUND)
    target_link_libraries(colmap_mvs PRIVATE OpenMP::OpenMP_CXX)
endif()
# Target commands AFTER the library target exists (Sol 3cdc fix 1).
target_include_directories(colmap_mvs PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
add_dependencies(colmap_mvs resident_allocation_fingerprint photo_phase_fingerprint)

# Public MVS headers include these build-generated fingerprints. The library
# dependencies above generate them for the normal build/install target, even
# when CUDA is disabled; ship the same bytes at their public include paths.
install(FILES
    "${PHOTO_PHASE_HEADER}"
    "${RESIDENT_ALLOCATION_HEADER}"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/colmap/mvs/generated")

COLMAP_ADD_TEST(
    NAME owned_prng_seed_test
    SRCS owned_prng_seed_test.cc
    LINK_LIBS colmap_mvs
)

COLMAP_ADD_TEST(
    NAME consistency_graph_test
    SRCS consistency_graph_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME depth_map_test
    SRCS depth_map_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME fusion_mapped_mat_test
    SRCS fusion_mapped_mat_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME fusion_test
    SRCS fusion_test.cc
    LINK_LIBS colmap_mvs colmap_controllers
)
if(TESTS_ENABLED AND TARGET colmap_main)
    add_dependencies(colmap_mvs_fusion_test colmap_main)
    set_tests_properties(mvs/fusion_test PROPERTIES
        ENVIRONMENT "COLMAP_BINARY=$<TARGET_FILE:colmap_main>")
endif()
COLMAP_ADD_TEST(
    NAME image_test
    SRCS image_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME mat_test
    SRCS mat_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME mesh_simplification_test
    SRCS mesh_simplification_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME poisson_meshing_test
    SRCS poisson_meshing_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME delaunay_meshing_test
    SRCS delaunay_meshing_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME advancing_front_meshing_test
    SRCS advancing_front_meshing_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME model_test
    SRCS model_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME normal_map_test
    SRCS normal_map_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME texture_mapping_test
    SRCS texture_mapping_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME workspace_test
    SRCS workspace_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME resident_contract_test
    SRCS resident_contract_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME resident_contract_capture_test
    SRCS resident_contract_capture_test.cc
    LINK_LIBS colmap_mvs
)

# Native resident-forecast wire/gate tests. The pure parser + gate slice is
# self-contained (no estimator); minima tests require the linked bank
# estimator (never stubbed). The focused stdlib-only parser/gate binary is
# the local dev-box slice; the estimator recompute runs at the real build.
COLMAP_ADD_TEST(
    NAME resident_forecast_test
    SRCS resident_forecast_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME resident_forecast_native_test
    SRCS resident_forecast_native_test.cc
    LINK_LIBS colmap_mvs
)
COLMAP_ADD_TEST(
    NAME resident_runner_lifecycle_test
    SRCS resident_runner_lifecycle_test.cc
    LINK_LIBS colmap_mvs
)
# CPU structural/arithmetic cohort boundary tests (typed validated plan; no
# payloads, no CUDA, no wire). The real planner + boundary validator run with
# a standalone test estimator; native execution surfaces stay dev1 coverage.
COLMAP_ADD_TEST(
    NAME resident_cohorts_test
    SRCS resident_cohorts_test.cc
    LINK_LIBS colmap_mvs
)
# Native per-cohort minima/charge gate tests (estimator-dependent; the
# gate itself is compiled into colmap_mvs and never stubbed).
COLMAP_ADD_TEST(
    NAME resident_cohort_wire_minima_test
    SRCS resident_cohort_wire_minima_test.cc
    LINK_LIBS colmap_mvs
)
# CPU output-path-discipline + cohort runner lifecycle tests (real
# coordinator/pipeline/observer interfaces; no CUDA, no payloads).
COLMAP_ADD_TEST(
    NAME resident_cohort_runner_lifecycle_test
    SRCS resident_cohort_runner_lifecycle_test.cc
    LINK_LIBS colmap_mvs
)
# Native resident-cohort-forecast v1 wire parser/gate tests. Cross-language
# fixtures: the canonical Python emitter bytes (photogram 9fb4957 spec) are
# committed verbatim under testdata/resident_cohort with their shas
# asserted; the contract binding reuses the shared contract fixtures.
COLMAP_ADD_TEST(
    NAME resident_cohort_wire_test
    SRCS resident_cohort_wire_test.cc
    LINK_LIBS colmap_mvs
)
if(TESTS_ENABLED)
    target_include_directories(colmap_mvs_resident_cohort_wire_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(colmap_mvs_resident_cohort_wire_test PRIVATE
        RESIDENT_COHORT_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata/resident_cohort"
        RESIDENT_COHORT_CONTRACT_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata/resident_forecast")
endif()
if(TESTS_ENABLED)
    target_include_directories(colmap_mvs_resident_forecast_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(colmap_mvs_resident_forecast_test PRIVATE
        RESIDENT_FORECAST_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata/resident_forecast")
    target_include_directories(colmap_mvs_resident_forecast_native_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(CUDA_ENABLED)
    find_package(Threads REQUIRED)
    COLMAP_ADD_LIBRARY(
        NAME colmap_mvs_cuda
        SRCS
            gpu_mat_prng.h gpu_mat_prng.cu
            gpu_mat_ref_image.h gpu_mat_ref_image.cu
            patch_match.h patch_match.cc
            patch_match_cuda.h patch_match_cuda.cu
            patch_match_cuda_owned.h patch_match_cuda_owned.cu
            owned_prng.h owned_prng_seed.h owned_prng_boundary.h
            patch_match_calibration.h
            owned_input_bank.cu
            owned_graph.h owned_graph.cu
            owned_graph_topology.h
            owned_graph_coordinator.h
            owned_async_pool.h
            owned_cuda_execution.h
            owned_rotation_storage.h
            owned_selection_scratch.h
            owned_tile.h
            owned_source_cache.h
            owned_texture_row_batch.h
            owned_pair_coordinator.h
            owned_pair_validation.h
            owned_pair_input.h
            owned_pair_image.h
            owned_pair_output.h
            owned_pair_events.h
            owned_pair_run.h
            owned_host_lookahead.h
            owned_normal_lookahead.h
        PUBLIC_LINK_LIBS
            colmap_mvs
            colmap_util_cuda
            CUDA::cudart
            CUDA::curand
            Threads::Threads
    )

    COLMAP_ADD_TEST(
        NAME owned_host_lookahead_test
        SRCS owned_host_lookahead_test.cc
        LINK_LIBS colmap_mvs_cuda
    )
    COLMAP_ADD_TEST(
        NAME owned_normal_lookahead_test
        SRCS owned_normal_lookahead_test.cc
        LINK_LIBS colmap_mvs_cuda
    )
    COLMAP_ADD_TEST(
        NAME owned_host_snapshot_test
        SRCS owned_host_snapshot_test.cc
        LINK_LIBS colmap_mvs_cuda
    )

    COLMAP_ADD_TEST(
        NAME owned_input_bank_test
        SRCS owned_input_bank_test.cc
        LINK_LIBS colmap_mvs_cuda
    )

    COLMAP_ADD_TEST(
        NAME owned_input_bank_cuda_test
        SRCS owned_input_bank_test.cu
        LINK_LIBS colmap_mvs_cuda
    )

    # Work around NVCC miscompilation on Blackwell GPUs (sm_100+).
    # The PatchMatch kernels are miscompiled at -O2/-O3 when generating native
    # SASS for sm_100+. Instead, we emit sm_90 PTX which Blackwell GPUs
    # JIT-compile at first launch, bypassing the buggy codegen path.
    # See: https://github.com/colmap/colmap/issues/3514
    #      https://github.com/ggml-org/llama.cpp/issues/18331
    set(_COLMAP_BLACKWELL_TARGET FALSE)

    # Classify architecture keywords. "all"/"all-major" expand based on the
    # CUDA toolkit version, while "native" expands based on installed GPUs.
    set(_HAS_ALL_KEYWORD FALSE)
    set(_HAS_NATIVE_KEYWORD FALSE)
    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(_arch STREQUAL "all" OR _arch STREQUAL "all-major")
            set(_HAS_ALL_KEYWORD TRUE)
        elseif(_arch STREQUAL "native")
            set(_HAS_NATIVE_KEYWORD TRUE)
        endif()
    endforeach()

    # Query installed GPUs only for "native", since it targets present hardware.
    if(_HAS_NATIVE_KEYWORD)
        execute_process(
            COMMAND nvidia-smi
                --query-gpu=compute_cap
                --format=csv,noheader
            OUTPUT_VARIABLE _GPU_COMPUTE_CAPS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _NVIDIA_SMI_RESULT
        )
        if(_NVIDIA_SMI_RESULT EQUAL 0 AND _GPU_COMPUTE_CAPS)
            string(REPLACE "\n" ";" _GPU_COMPUTE_CAPS
                   "${_GPU_COMPUTE_CAPS}")
        endif()
    endif()

    # Detect Blackwell from explicit numeric architectures.
    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        string(REGEX REPLACE "[-].*" "" _arch_num "${_arch}")
        if(_arch_num MATCHES "^[0-9]+$" AND _arch_num GREATER_EQUAL 100)
            set(_COLMAP_BLACKWELL_TARGET TRUE)
            break()
        endif()
    endforeach()

    # Detect Blackwell from "all"/"all-major": these expand based on the
    # toolkit, not installed GPUs. CUDA 12.8 added sm_100 support.
    if(NOT _COLMAP_BLACKWELL_TARGET AND _HAS_ALL_KEYWORD)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
            set(_COLMAP_BLACKWELL_TARGET TRUE)
        endif()
    endif()

    # Detect Blackwell from "native" via nvidia-smi (installed GPU check).
    if(NOT _COLMAP_BLACKWELL_TARGET AND _HAS_NATIVE_KEYWORD)
        if(_NVIDIA_SMI_RESULT EQUAL 0 AND _GPU_COMPUTE_CAPS)
            foreach(_cap IN LISTS _GPU_COMPUTE_CAPS)
                string(STRIP "${_cap}" _cap)
                string(REGEX MATCH "^([0-9]+)" _major "${_cap}")
                if(_major AND _major GREATER_EQUAL 10)
                    set(_COLMAP_BLACKWELL_TARGET TRUE)
                    break()
                endif()
            endforeach()
        endif()
    endif()

    if(_COLMAP_BLACKWELL_TARGET)
        message(WARNING
            "Blackwell GPU (sm_100+) detected. Replacing sm_100+ architectures "
            "with 90-virtual (PTX) for colmap_mvs_cuda to work around an NVCC "
            "compiler bug. Blackwell GPUs will JIT-compile from PTX at first "
            "launch (see https://github.com/colmap/colmap/issues/3514).")

        # Build filtered architecture list: keep archs < 100, drop >= 100.
        set(_COLMAP_MVS_CUDA_ARCHS)
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "[-].*" "" _arch_num "${_arch}")
            if(_arch_num MATCHES "^[0-9]+$")
                if(_arch_num LESS 100)
                    list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_arch}")
                endif()
            elseif(_arch STREQUAL "all")
                # "all"/"all-major" keywords require CMake 3.24+, which
                # provides CMAKE_CUDA_ARCHITECTURES_ALL[_MAJOR] variables.
                foreach(_exp IN LISTS CMAKE_CUDA_ARCHITECTURES_ALL)
                    string(REGEX REPLACE "[-].*" "" _exp_num "${_exp}")
                    if(_exp_num MATCHES "^[0-9]+$" AND _exp_num LESS 100)
                        list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_exp}")
                    endif()
                endforeach()
            elseif(_arch STREQUAL "all-major")
                foreach(_exp IN LISTS CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR)
                    string(REGEX REPLACE "[-].*" "" _exp_num "${_exp}")
                    if(_exp_num MATCHES "^[0-9]+$" AND _exp_num LESS 100)
                        list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_exp}")
                    endif()
                endforeach()
            elseif(_arch STREQUAL "native")
                # Replace with explicit non-Blackwell GPU architectures.
                if(_NVIDIA_SMI_RESULT EQUAL 0 AND _GPU_COMPUTE_CAPS)
                    foreach(_cap IN LISTS _GPU_COMPUTE_CAPS)
                        string(STRIP "${_cap}" _cap)
                        string(REGEX MATCH
                               "^([0-9]+)\.([0-9]+)" _match "${_cap}")
                        if(_match)
                            math(EXPR _sm
                                 "${CMAKE_MATCH_1} * 10 + ${CMAKE_MATCH_2}")
                            if(_sm LESS 100)
                                list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_sm}")
                            endif()
                        endif()
                    endforeach()
                endif()
            else()
                list(APPEND _COLMAP_MVS_CUDA_ARCHS "${_arch}")
            endif()
        endforeach()

        # Add PTX for the highest pre-Blackwell arch so Blackwell GPUs
        # can JIT-compile at runtime.
        list(APPEND _COLMAP_MVS_CUDA_ARCHS "90-virtual")
        list(REMOVE_DUPLICATES _COLMAP_MVS_CUDA_ARCHS)

        set_target_properties(colmap_mvs_cuda PROPERTIES
            CUDA_ARCHITECTURES "${_COLMAP_MVS_CUDA_ARCHS}"
        )
    endif()

    COLMAP_ADD_TEST(
        NAME owned_distinct_source_cache_test
        SRCS owned_distinct_source_cache_test.cc
        LINK_LIBS colmap_mvs_cuda
    )
    COLMAP_ADD_TEST(
        NAME owned_distinct_source_cache_trace_test
        SRCS owned_distinct_source_cache_trace_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
    if(TESTS_ENABLED)
        get_target_property(_owned_cache_test_archs colmap_mvs_cuda CUDA_ARCHITECTURES)
        set_target_properties(colmap_mvs_owned_distinct_source_cache_trace_test
            PROPERTIES CUDA_ARCHITECTURES "${_owned_cache_test_archs}")
    endif()

    COLMAP_ADD_TEST(
        NAME gpu_mat_test
        SRCS gpu_mat_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
    COLMAP_ADD_TEST(
        NAME owned_prng_test
        SRCS owned_prng_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
endif()

# ROCm HIP support for MVS (patch_match_stereo on AMD GPUs).
#
# enable_language(HIP) and find_package(hip/hiprand/rocrand) are handled
# centrally in cmake/FindDependencies.cmake. Here we just mark the three
# .cu translation units as LANGUAGE HIP so CMake hands them to the HIP
# toolchain (clang++ -x hip) instead of the CUDA one. The source itself
# is unmodified; symbol remapping happens in src/colmap/util/cuda_to_hip.h.
if(HIP_ENABLED)
    COLMAP_ADD_LIBRARY(
        NAME colmap_mvs_cuda
        SRCS
            gpu_mat_prng.h gpu_mat_prng.cu
            gpu_mat_ref_image.h gpu_mat_ref_image.cu
            patch_match.h patch_match.cc
            patch_match_cuda.h patch_match_cuda.cu
        PUBLIC_LINK_LIBS
            colmap_mvs
            colmap_util_cuda
            hip::host
            hip::hiprand
            roc::rocrand
    )
    set_source_files_properties(
        gpu_mat_prng.cu
        gpu_mat_ref_image.cu
        patch_match_cuda.cu
        PROPERTIES LANGUAGE HIP
    )

    COLMAP_ADD_TEST(
        NAME gpu_mat_test
        SRCS gpu_mat_test.cu
        LINK_LIBS colmap_mvs_cuda
    )
    set_source_files_properties(gpu_mat_test.cu PROPERTIES LANGUAGE HIP)
endif()

# Dependency-free optimized-dispatch contract executables also run without a
# configured COLMAP tree using the commands in doc/optimized_dispatch.md.
if(TESTS_ENABLED)
    add_test(NAME "mvs/photo_phase_fingerprint_test"
        COMMAND ${Python3_EXECUTABLE} -I
            "${PROJECT_SOURCE_DIR}/scripts/resident/test_photo_phase_fingerprint.py")
    find_package(Threads REQUIRED)
    foreach(contract_test optimized_dispatch_test optimized_photo_forecast_test
            optimized_photo_binding_test optimized_photo_dispatch_test
            optimized_photo_delivery_test
            optimized_photo_options_test optimized_photo_paths_test
            optimized_photo_prepared_test
            optimized_photo_concurrency_test fusion_dependencies_test)
        add_executable(colmap_mvs_${contract_test} ${contract_test}.cc)
        target_include_directories(colmap_mvs_${contract_test} PRIVATE
            "${PROJECT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}")
        add_dependencies(colmap_mvs_${contract_test} photo_phase_fingerprint)
        target_link_libraries(colmap_mvs_${contract_test} PRIVATE Threads::Threads)
        if(MSVC)
            target_compile_options(colmap_mvs_${contract_test} PRIVATE /UNDEBUG)
        else()
            target_compile_options(colmap_mvs_${contract_test} PRIVATE -UNDEBUG)
        endif()
        add_test(NAME "mvs/${contract_test}" COMMAND colmap_mvs_${contract_test})
    endforeach()
endif()

# Uses real coordinator and owner threads; POSIX subprocess death diagnostics.
if(TESTS_ENABLED AND UNIX)
    add_executable(colmap_mvs_native_run_deadline_test native_run_deadline_test.cc)
    target_include_directories(colmap_mvs_native_run_deadline_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(colmap_mvs_native_run_deadline_test PRIVATE Threads::Threads)
    target_compile_options(colmap_mvs_native_run_deadline_test PRIVATE -UNDEBUG)
    add_test(NAME "mvs/native_run_deadline_test" COMMAND colmap_mvs_native_run_deadline_test)
    set_tests_properties(mvs/native_run_deadline_test PROPERTIES TIMEOUT 15)
endif()

# Dependency-free topology/argument packing regression, including CUDA-off builds.
if(TESTS_ENABLED)
    add_executable(owned_graph_cpu_test owned_graph_test.cc)
    target_include_directories(owned_graph_cpu_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(owned_graph_cpu_test PRIVATE cxx_std_17)
    # Assertions are this small standalone fixture's test mechanism.
    target_compile_options(owned_graph_cpu_test PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-UNDEBUG>
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>)
    add_test(NAME mvs/owned_graph_cpu_test COMMAND owned_graph_cpu_test)

    add_executable(owned_graph_coordinator_cpu_test owned_graph_coordinator_test.cc)
    target_include_directories(owned_graph_coordinator_cpu_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(owned_graph_coordinator_cpu_test PRIVATE cxx_std_17)
    target_link_libraries(owned_graph_coordinator_cpu_test PRIVATE Threads::Threads)
    target_compile_options(owned_graph_coordinator_cpu_test PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-UNDEBUG>
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>)
    add_test(NAME mvs/owned_graph_coordinator_cpu_test COMMAND owned_graph_coordinator_cpu_test)
    set_tests_properties(mvs/owned_graph_coordinator_cpu_test PROPERTIES TIMEOUT 15)
endif()
