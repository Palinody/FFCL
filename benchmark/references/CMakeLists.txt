# Reference-library benchmark suite. Deliberately separate from the project's own CMakeLists: each
# library has its own dependency and build story, and a missing one must not break the others.
cmake_minimum_required(VERSION 3.20)
project(ffcl_reference_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

find_package(OpenMP REQUIRED)

set(REFS "${CMAKE_CURRENT_SOURCE_DIR}/../bin/../../benchmark_references")
get_filename_component(REFS "${REFS}" ABSOLUTE)

# ---- nanoflann (header only) ----------------------------------------------------------------
if(EXISTS "${REFS}/nanoflann/include/nanoflann.hpp")
    add_compile_definitions(HAVE_NANOFLANN)
    include_directories("${REFS}/nanoflann/include")
    message("refbench: nanoflann")
endif()

# ---- pico_tree (header only) ----------------------------------------------------------------
if(EXISTS "${REFS}/pico_tree/src/pico_tree/pico_tree/kd_tree.hpp")
    add_compile_definitions(HAVE_PICO_TREE)
    include_directories("${REFS}/pico_tree/src/pico_tree")
    message("refbench: pico_tree")
endif()

# ---- FLANN (system) -------------------------------------------------------------------------
find_path(FLANN_INCLUDE_DIR flann/flann.hpp)
find_library(FLANN_LIBRARY flann_cpp)
if(FLANN_INCLUDE_DIR AND FLANN_LIBRARY)
    add_compile_definitions(HAVE_FLANN)
    message("refbench: flann")
endif()

# ---- mlpack (headers from the clone, armadillo + ensmallen from the system) -----------------
if(EXISTS "${REFS}/mlpack/src/mlpack/core.hpp")
    add_compile_definitions(HAVE_MLPACK)
    include_directories("${REFS}/mlpack/src")
    message("refbench: mlpack")
endif()

# ---- ArborX (installed prefix, Kokkos backend) ----------------------------------------------
set(ARBORX_AVAILABLE OFF)
if(EXISTS "${REFS}/install/include/ArborX/ArborX.hpp")
    add_compile_definitions(HAVE_ARBORX)
    set(ARBORX_AVAILABLE ON)
    # Kokkos needs its own CMake config for the compile definitions, not just the static libraries
    set(Kokkos_DIR "${REFS}/install/lib/cmake/Kokkos" CACHE PATH "" FORCE)
    find_package(Kokkos REQUIRED)
    set(ArborX_DIR "${REFS}/install/share/cmake/ArborX" CACHE PATH "" FORCE)
    find_package(ArborX REQUIRED)
    include_directories("${REFS}/install/include/ArborX" "${REFS}/install/include")
    message("refbench: ArborX")
endif()

add_executable(refbench refbench.cpp)
target_link_libraries(refbench PRIVATE OpenMP::OpenMP_CXX)
if(FLANN_LIBRARY)
    target_link_libraries(refbench PRIVATE ${FLANN_LIBRARY} lz4)
endif()
if(ARBORX_AVAILABLE)
    target_link_libraries(refbench PRIVATE ArborX::ArborX)
endif()
