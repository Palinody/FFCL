cmake_minimum_required(VERSION 3.0.2)

project(ffcl)

# =============================================================================
#  FFCL
#
#  The tree is split into three independent parts:
#
#    include/  src/   the library itself (header-only, src/ has one translation unit)
#    test/            fast functional unit tests (should stay well under a minute)
#    benchmark/       dataset-driven measurements against reference libraries
#
#  Build the library only:      cmake .. && make
#  Build and run the tests:     cmake .. -DMODE=gtest && make && ctest
#  Build and run the benchmark: cmake .. -DMODE=benchmark && make && ./benchmark/ffcl_benchmark
#
#  Options:
#    -DVERBOSE=true          print per-stage timings
#    -DTHREADS_ENABLED=true  parallelise the independent loops with OpenMP
# =============================================================================

# --- build type -------------------------------------------------------------
# The optimization flags below are only added for an explicit build type. When
# CMAKE_BUILD_TYPE is left empty (the common `cmake ..` invocation) no `-O` flag
# at all is passed to the compiler, which makes every benchmark run at -O0 and
# therefore 10-20x slower than it should be. Default to Release in that case.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Release, Debug, ...)" FORCE)
    message("No CMAKE_BUILD_TYPE specified: defaulting to Release")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Building in Debug mode")
    add_compile_options(-std=c++17 -g -pg -O0)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("Building in Release mode")
    add_compile_options(
        -std=c++17
        -Ofast
        -march=native
        -Wall
        -Wextra
        -Werror
        -Wpedantic
        -Wfloat-equal
        -Wshadow
        -Wfatal-errors
        -fopenmp
    )
endif()

find_package(OpenMP REQUIRED)

# Optional reference libraries for the benchmark. Each one guards its own
# comparisons, so the benchmark still builds when they are absent.
#
#   PCL    -> KD-tree / FLANN index comparison for the neighbour searches
#   FLANN  -> standalone KD-tree comparison
#   MLPACK -> dual-tree Boruvka MST comparison
#   HDBSCAN-> not a C++ dependency: compared against through the Python scripts
find_package(PCL QUIET)
find_package(Armadillo QUIET)
find_package(MLPACK QUIET)

# CMake ships no FindFLANN module, so locate the headers and the library directly.
find_path(FLANN_INCLUDE_DIR flann/flann.hpp)
find_library(FLANN_LIBRARY NAMES flann_cpp flann)
if(FLANN_INCLUDE_DIR AND FLANN_LIBRARY)
    set(FLANN_FOUND TRUE)
endif()

option(VERBOSE "Print the per-stage timings (tree build, MST, full pipeline)" OFF)
option(THREADS_ENABLED "Parallelise the independent loops with OpenMP" OFF)

if(VERBOSE)
    add_definitions(-DVERBOSE=true)
else()
    add_definitions(-DVERBOSE=false)
endif()

if(THREADS_ENABLED)
    add_definitions(-DTHREADS_ENABLED=true)
else()
    add_definitions(-DTHREADS_ENABLED=false)
endif()

include_directories(include)

# =============================================================================
#  Code
# =============================================================================

set(HEADER_FILES
    include/ffcl/kmeans/KMeans.hpp
    include/ffcl/kmeans/Lloyd.hpp
    include/ffcl/kmeans/Hamerly.hpp
    include/ffcl/kmeans/KMeansUtils.hpp
    include/ffcl/kmeans/KMeansPlusPlus.hpp

    include/ffcl/kmedoids/FasterMSC.hpp
    include/ffcl/kmedoids/FasterPAM.hpp
    include/ffcl/kmedoids/KMedoids.hpp
    include/ffcl/kmedoids/PAMUtils.hpp
    include/ffcl/kmedoids/PAMBuild.hpp

    include/ffcl/dbscan/DBSCAN.hpp
    include/ffcl/hdbscan/HDBSCAN.hpp

    include/ffcl/search/Search.hpp
    include/ffcl/search/TreeTraverser.hpp
    include/ffcl/search/ClosestPairOfSamples.hpp
    include/ffcl/search/buffer/StaticBuffer.hpp
    include/ffcl/search/buffer/Unsorted.hpp
    include/ffcl/search/buffer/WithMemory.hpp
    include/ffcl/search/buffer/WithUnionFind.hpp
    include/ffcl/search/buffer/EdgesBuffer.hpp

    include/ffcl/datastruct/tree/kdtree/KDTree.hpp
    include/ffcl/datastruct/tree/kdtree/KDNodeView.hpp
    include/ffcl/datastruct/tree/kdtree/KDTreeAlgorithms.hpp
    include/ffcl/datastruct/tree/kdtree/policy/AxisSelectionPolicy.hpp
    include/ffcl/datastruct/tree/kdtree/policy/SplittingRulePolicy.hpp

    include/ffcl/datastruct/tree/single_linkage_cluster_tree/SingleLinkageClusterTree.hpp
    include/ffcl/datastruct/tree/single_linkage_cluster_tree/SingleLinkageClusterNode.hpp
    include/ffcl/datastruct/tree/condensed_cluster_tree/CondensedClusterTree.hpp
    include/ffcl/datastruct/tree/condensed_cluster_tree/CondensedClusterNode.hpp

    include/ffcl/datastruct/graph/spanning_tree/BoruvkasAlgorithm.hpp
    include/ffcl/datastruct/graph/spanning_tree/MSTBuilder.hpp
    include/ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp
    include/ffcl/datastruct/graph/spanning_tree/CoreDistances.hpp

    include/ffcl/datastruct/bounds/AABB.hpp
    include/ffcl/datastruct/bounds/AABBWithCentroid.hpp
    include/ffcl/datastruct/bounds/Ball.hpp
    include/ffcl/datastruct/bounds/UnboundedBall.hpp
    include/ffcl/datastruct/bounds/StaticBound.hpp
    include/ffcl/datastruct/bounds/StaticBoundWithCentroid.hpp
    include/ffcl/datastruct/bounds/distances/MinDistance.hpp
    include/ffcl/datastruct/bounds/segment/LowerBoundAndUpperBound.hpp
    include/ffcl/datastruct/bounds/segment/LowerBoundAndLength.hpp
    include/ffcl/datastruct/bounds/segment/StaticSegment.hpp
    include/ffcl/datastruct/bounds/segment/distances/MinDistance.hpp

    include/ffcl/datastruct/matrix/PairwiseDistanceMatrix.hpp
    include/ffcl/datastruct/matrix/PairwiseDistanceMatrixDynamic.hpp
    include/ffcl/datastruct/vector/FeaturesVector.hpp
    include/ffcl/datastruct/vector/FeaturesMask.hpp
    include/ffcl/datastruct/UnionFind.hpp
    include/ffcl/datastruct/Interval.hpp

    include/ffcl/common/Utils.hpp
    include/ffcl/common/Timer.hpp
    include/ffcl/common/algorithms/Sorting.hpp
    include/ffcl/common/math/statistics/Statistics.hpp
    include/ffcl/common/math/heuristics/Distances.hpp
    include/ffcl/common/math/heuristics/SilhouetteMethod.hpp
    include/ffcl/common/math/linear_algebra/Transpose.hpp
    include/ffcl/common/math/linear_algebra/Translation.hpp
    include/ffcl/common/math/random/Distributions.hpp
    include/ffcl/common/math/random/Sampling.hpp
    include/ffcl/common/math/random/VosesAliasMethod.hpp
)

set(SOURCE_FILES
    # FFCL is header-only; this single translation unit gives the shared library target a linker
    # language and a real artifact.
    src/ffcl.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

# =============================================================================
#  Test
# =============================================================================
# Fast functional unit tests. Everything here must stay cheap: the dataset-driven
# accuracy and performance measurements live in benchmark/.

if(MODE STREQUAL "gtest")
    include(FetchContent)

    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    include_directories(test/include test/include/base_fixtures)

    # One target per test translation unit, named after the file. The directory layout under
    # test/src mirrors include/ffcl so a header and its test are always in the same relative place.
    set(FFCL_TEST_SOURCES
        test/src/common/UtilsTest.cpp
        test/src/common/algorithms/SortingTest.cpp
        test/src/common/math/linear_algebra/TransposeTest.cpp
        test/src/common/math/linear_algebra/TranslationTest.cpp
        test/src/common/math/random/DistributionsTest.cpp
        test/src/common/math/random/SamplingTest.cpp
        test/src/common/math/random/VosesAliasMethodTest.cpp
        test/src/common/math/statistics/StatisticsTest.cpp
        test/src/common/math/heuristics/DistancesTest.cpp
        test/src/common/math/heuristics/SilhouetteMethodTest.cpp

        test/src/datastruct/UnionFindTest.cpp
        test/src/datastruct/IntervalTest.cpp
        test/src/datastruct/vector/FeaturesVectorTest.cpp
        test/src/datastruct/vector/FeaturesMaskTest.cpp
        test/src/datastruct/matrix/PairwiseDistanceMatrixTest.cpp
        test/src/datastruct/bounds/distance/BoundsDistanceTest.cpp

        test/src/datastruct/tree/kdtree/KDTreeTest.cpp
        test/src/datastruct/tree/kdtree/KDTreeAlgorithmsTest.cpp
        test/src/datastruct/tree/kdtree/policy/AxisSelectionPolicyTest.cpp
        test/src/datastruct/tree/kdtree/policy/SplittingRulePolicyTest.cpp
        test/src/datastruct/tree/kdtree/policy/MidpointRangeTest.cpp
        test/src/datastruct/tree/single_linkage_cluster_tree/SingleLinkageClusterTreeTest.cpp
        test/src/datastruct/tree/condensed_cluster_tree/CondensedClusterTreeTest.cpp
        test/src/datastruct/graph/spanning_tree/BoruvkasAlgorithmTest.cpp

        test/src/search/SearchTest.cpp
        test/src/search/buffer/UnsortedTest.cpp
        test/src/search/buffer/WithMemoryTest.cpp
        test/src/search/buffer/WithUnionFindTest.cpp
        test/src/search/buffer/EdgesBufferTest.cpp
        test/src/search/ClosestPairOfSamplesTest.cpp

        test/src/dbscan/DBSCANTest.cpp
        test/src/hdbscan/HDBSCANTest.cpp
        test/src/kmeans/KMeansTest.cpp
        test/src/kmedoids/KMedoidsTest.cpp
    )

    foreach(test_source ${FFCL_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)

        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} PUBLIC GTest::gtest_main GTest::gmock_main OpenMP::OpenMP_CXX)
        add_test(${test_name} ${test_name})
    endforeach()

    set_tests_properties(
        SortingTest DistributionsTest SamplingTest VosesAliasMethodTest
        PROPERTIES LABELS "utilities"
    )
    set_tests_properties(
        UtilsTest StatisticsTest DistancesTest SilhouetteMethodTest TransposeTest TranslationTest
        PROPERTIES LABELS "utilities"
    )
    set_tests_properties(
        UnionFindTest IntervalTest FeaturesVectorTest FeaturesMaskTest PairwiseDistanceMatrixTest
        BoundsDistanceTest
        PROPERTIES LABELS "datastruct"
    )
    set_tests_properties(
        KDTreeTest KDTreeAlgorithmsTest AxisSelectionPolicyTest SplittingRulePolicyTest MidpointRangeTest
        PROPERTIES LABELS "indexing"
    )
    set_tests_properties(
        SingleLinkageClusterTreeTest CondensedClusterTreeTest BoruvkasAlgorithmTest
        PROPERTIES LABELS "datastruct"
    )
    set_tests_properties(
        SearchTest UnsortedTest WithMemoryTest WithUnionFindTest EdgesBufferTest ClosestPairOfSamplesTest
        PROPERTIES LABELS "search"
    )
    set_tests_properties(
        DBSCANTest HDBSCANTest KMeansTest KMedoidsTest
        PROPERTIES LABELS "clustering"
    )
endif()

# =============================================================================
#  Benchmark
# =============================================================================
# Dataset-driven measurements of every algorithm against the reference libraries that are available
# on this machine. Nothing here is registered with CTest: these runs take minutes.

if(MODE STREQUAL "benchmark")
    include_directories(benchmark/include)

    add_executable(ffcl_benchmark
        benchmark/src/main.cpp

        benchmark/src/kdtree/KDTreeBenchmark.cpp
        benchmark/src/dbscan/DBSCANBenchmark.cpp
        benchmark/src/hdbscan/HDBSCANBenchmark.cpp
        benchmark/src/kmeans/KMeansBenchmark.cpp
        benchmark/src/kmedoids/KMedoidsBenchmark.cpp
        benchmark/src/graph/MinimumSpanningTreeBenchmark.cpp

        benchmark/src/reference/PCLBenchmark.cpp
        benchmark/src/reference/FLANNBenchmark.cpp
        benchmark/src/reference/MLPACKBenchmark.cpp
    )

    target_link_libraries(ffcl_benchmark PUBLIC ${PROJECT_NAME} OpenMP::OpenMP_CXX)

    # Single-dataset driver, used by benchmark/scripts/CompareHDBSCAN.py and for ad-hoc timings.
    add_executable(BoruvkasRunner benchmark/src/tools/BoruvkasRunner.cpp)
    target_link_libraries(BoruvkasRunner PUBLIC ${PROJECT_NAME} OpenMP::OpenMP_CXX)

    if(PCL_FOUND)
        message("Benchmark: PCL found, enabling the PCL comparisons")
        include_directories(${PCL_INCLUDE_DIRS})
        target_link_libraries(ffcl_benchmark PUBLIC ${PCL_LIBRARIES})
        target_compile_definitions(ffcl_benchmark PUBLIC FFCL_BENCH_HAS_PCL)
    else()
        message("Benchmark: PCL not found, its comparisons will be reported as unavailable")
    endif()

    if(FLANN_FOUND)
        message("Benchmark: FLANN found (${FLANN_LIBRARY}), enabling the FLANN comparisons")
        include_directories(${FLANN_INCLUDE_DIR})
        # libflann_cpp pulls its serialization symbols from LZ4
        find_library(LZ4_LIBRARY NAMES lz4)
        target_link_libraries(ffcl_benchmark PUBLIC ${FLANN_LIBRARY} ${LZ4_LIBRARY})
        target_compile_definitions(ffcl_benchmark PUBLIC FFCL_BENCH_HAS_FLANN)
    else()
        message("Benchmark: FLANN not found, its comparisons will be reported as unavailable")
    endif()

    if(MLPACK_FOUND)
        message("Benchmark: MLPACK found, enabling the MLPACK comparisons")
        include_directories(${MLPACK_INCLUDE_DIRS})
        target_link_libraries(ffcl_benchmark PUBLIC ${MLPACK_LIBRARIES})
        target_compile_definitions(ffcl_benchmark PUBLIC FFCL_BENCH_HAS_MLPACK)
    else()
        message("Benchmark: MLPACK not found, its comparisons will be reported as unavailable")
    endif()
endif()
