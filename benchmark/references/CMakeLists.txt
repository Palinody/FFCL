# Reference-library benchmark suite. Deliberately separate from the project's own CMakeLists: each
# library has its own dependency and build story, and a missing one must not break the others.
#
# One executable per library, for two reasons:
#   * pdbscan and pargeo hdbscan each vendor a different parlay scheduler; linking both into one
#     binary makes the scheduler singleton resolve to the wrong layout (observed: segfault).
#   * linking mlpack/armadillo into a shared binary measurably slows the nanoflann hot loop
#     (4-D kNN 0.68 -> 0.86 ms), so every library is measured in a clean binary.
#
# Each library can be switched off with -DREFBENCH_<LIB>=OFF to isolate a build problem.
cmake_minimum_required(VERSION 3.20)
project(ffcl_reference_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

find_package(OpenMP REQUIRED)

option(REFBENCH_NANOFLANN "build the nanoflann adapter" ON)
option(REFBENCH_FLANN "build the FLANN adapter" ON)
option(REFBENCH_PICO_TREE "build the pico_tree adapter" ON)
option(REFBENCH_MLPACK "build the mlpack adapter" ON)
option(REFBENCH_ARBORX "build the ArborX adapter" ON)

set(REFS "${CMAKE_CURRENT_SOURCE_DIR}/../bin/../../benchmark_references")
get_filename_component(REFS "${REFS}" ABSOLUTE)

# ---- nanoflann (header only) ----------------------------------------------------------------
if(REFBENCH_NANOFLANN AND EXISTS "${REFS}/nanoflann/include/nanoflann.hpp")
    add_executable(refbench_nanoflann refbench_nanoflann.cpp)
    target_include_directories(refbench_nanoflann PRIVATE "${REFS}/nanoflann/include")
    target_link_libraries(refbench_nanoflann PRIVATE OpenMP::OpenMP_CXX)
    message("refbench: nanoflann")
endif()

# ---- pico_tree (header only) ----------------------------------------------------------------
if(REFBENCH_PICO_TREE AND EXISTS "${REFS}/pico_tree/src/pico_tree/pico_tree/kd_tree.hpp")
    add_executable(refbench_pico_tree refbench_pico_tree.cpp)
    target_include_directories(refbench_pico_tree PRIVATE "${REFS}/pico_tree/src/pico_tree")
    target_link_libraries(refbench_pico_tree PRIVATE OpenMP::OpenMP_CXX)
    message("refbench: pico_tree")
endif()

# ---- FLANN (system) -------------------------------------------------------------------------
find_path(FLANN_INCLUDE_DIR flann/flann.hpp)
find_library(FLANN_LIBRARY flann_cpp)
if(REFBENCH_FLANN AND FLANN_INCLUDE_DIR AND FLANN_LIBRARY)
    add_executable(refbench_flann refbench_flann.cpp)
    target_link_libraries(refbench_flann PRIVATE ${FLANN_LIBRARY} lz4 OpenMP::OpenMP_CXX)
    message("refbench: flann")
endif()

# ---- mlpack (headers from the clone, armadillo + ensmallen from the system) -----------------
find_library(ARMADILLO_LIBRARY armadillo)
if(REFBENCH_MLPACK AND EXISTS "${REFS}/mlpack/src/mlpack/core.hpp" AND ARMADILLO_LIBRARY)
    add_executable(refbench_mlpack refbench_mlpack.cpp)
    target_include_directories(refbench_mlpack PRIVATE "${REFS}/mlpack/src")
    target_link_libraries(refbench_mlpack PRIVATE ${ARMADILLO_LIBRARY} blas lapack OpenMP::OpenMP_CXX)
    message("refbench: mlpack")
endif()

# ---- ArborX (installed prefix, Kokkos backend) ----------------------------------------------
if(REFBENCH_ARBORX AND EXISTS "${REFS}/install/include/ArborX/ArborX.hpp")
    # Kokkos needs its own CMake config for the compile definitions, not just the static libraries
    set(Kokkos_DIR "${REFS}/install/lib/cmake/Kokkos" CACHE PATH "" FORCE)
    find_package(Kokkos REQUIRED)
    set(ArborX_DIR "${REFS}/install/share/cmake/ArborX" CACHE PATH "" FORCE)
    find_package(ArborX REQUIRED)
    add_executable(refbench_arborx refbench_arborx.cpp)
    # ArborX headers include each other by bare name, so every subdirectory must be on the path.
    target_include_directories(refbench_arborx PRIVATE "${REFS}/install/include/ArborX" "${REFS}/install/include")
    target_link_libraries(refbench_arborx PRIVATE ArborX::ArborX OpenMP::OpenMP_CXX)
    message("refbench: ArborX")
endif()

# ---- pdbscan (C API, bundled homegrown scheduler) -------------------------------------------
if(EXISTS "${REFS}/dbscan/include/dbscan/capi.h")
    add_executable(refbench_pdbscan refbench_pdbscan.cpp "${REFS}/dbscan/src/capi.cpp")
    target_include_directories(refbench_pdbscan PRIVATE "${REFS}/dbscan/include")
    target_compile_options(refbench_pdbscan PRIVATE -pthread)
    target_link_libraries(refbench_pdbscan PRIVATE pthread)
    message("refbench: pdbscan")
endif()

# ---- pargeo hdbscan (headers + two sources, bundled parlay) ----------------------------------
if(EXISTS "${REFS}/hdbscan/include/hdbscan/hdbscan.h")
    add_executable(refbench_pargeo
        refbench_pargeo.cpp
        "${REFS}/hdbscan/src/hdbscan.cpp"
        "${REFS}/hdbscan/src/dendrogram.cpp")
    target_include_directories(refbench_pargeo PRIVATE "${REFS}/hdbscan/include")
    target_compile_options(refbench_pargeo PRIVATE -DHOMEGROWN -pthread)
    target_link_libraries(refbench_pargeo PRIVATE pthread)
    message("refbench: pargeo hdbscan")
endif()
