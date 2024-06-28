cmake_minimum_required(VERSION 3.0.2)

project(ffcl)

# add_compile_options(-std=c++17 -g -pg) # <--- for debug mode
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
find_package(OpenMP REQUIRED)

# find_package(TBB REQUIRED)
include_directories(
    include
)

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

    include/ffcl/search/buffer/StaticBuffer.hpp
    include/ffcl/search/buffer/Unsorted.hpp
    include/ffcl/search/buffer/WithMemory.hpp
    include/ffcl/search/buffer/WithUnionFind.hpp

    include/ffcl/search/Search.hpp

    include/ffcl/datastruct/tree/kdtree/policy/AxisSelectionPolicy.hpp
    include/ffcl/datastruct/tree/kdtree/policy/SplittingRulePolicy.hpp
    include/ffcl/datastruct/tree/kdtree/KDNodeView.hpp
    include/ffcl/datastruct/tree/kdtree/KDTreeAlgorithms.hpp
    include/ffcl/datastruct/tree/kdtree/KDTree.hpp

    include/ffcl/datastruct/tree/single_linkage_cluster_tree/SingleLinkageClusterTree.hpp

    include/ffcl/datastruct/graph/spanning_tree/BoruvkasAlgorithm.hpp

    include/ffcl/datastruct/matrix/PairwiseDistanceMatrix.hpp
    include/ffcl/datastruct/matrix/PairwiseDistanceMatrixDynamic.hpp

    include/ffcl/datastruct/vector/FeaturesMask.hpp
    include/ffcl/datastruct/vector/FeaturesVector.hpp
    include/ffcl/datastruct/Interval.hpp

    include/ffcl/common/math/random/Distributions.hpp
    include/ffcl/common/math/random/VosesAliasMethod.hpp

    include/ffcl/common/math/heuristics/Distances.hpp
    include/ffcl/common/math/heuristics/SilhouetteMethod.hpp

    include/ffcl/common/Utils.hpp
    include/ffcl/common/Timer.hpp
)

set(SOURCE_FILES
)

add_definitions(-DTHREADS_ENABLED=false)
add_definitions(-DVERBOSE=false)

# disable all assertions
add_definitions(-DNDEBUG)

execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} --version
    OUTPUT_VARIABLE GPP_VERSION_OUTPUT
)

string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" GPP_VERSION "${GPP_VERSION_OUTPUT}")

message(" Using g++ Version: ${GPP_VERSION} ")

# set(MODE "benchmarks")
if(MODE STREQUAL "benchmark")
    # Find PCL package
    find_package(PCL REQUIRED)

    # Include PCL headers
    include_directories(${PCL_INCLUDE_DIRS})

    # Link against PCL libraries
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})

    add_executable(
        ${PROJECT_NAME}
        benchmark/main.cpp

        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
    target_link_libraries(
        ${PROJECT_NAME}

        # PUBLIC
        # TBB::tbb
        ${PCL_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
elseif(MODE STREQUAL "gtest")
    # -------- GOOGLE TEST ----------
    if(CMAKE_VERSION VERSION_LESS 3.14)
        # Download and unpack googletest at configure time
        configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
        execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)

        if(result)
            message(FATAL_ERROR "CMake step for googletest failed: ${result}")
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)

        if(result)
            message(FATAL_ERROR "Build step for googletest failed: ${result}")
        endif()

        # Prevent overriding the parent project's compiler/linker
        # settings on Windows
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        # Add googletest directly to our build. This defines
        # the gtest and gtest_main targets.
        add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
            ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
            EXCLUDE_FROM_ALL)

        # The gtest/gtest_main targets carry header search path
        # dependencies automatically when using CMake 2.8.11 or
        # later. Otherwise we have to add them here ourselves.
        if(CMAKE_VERSION VERSION_LESS 2.8.11)
            include_directories("${gtest_SOURCE_DIR}/include")
        endif()
    else()
        include(FetchContent)

        cmake_policy(SET CMP0135 NEW)

        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )

        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    # -------------------------------
    enable_testing()

    # add_compile_options(-std=c++17 -g -pg) # <--- for debug modes
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
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")

    include(CTest)

    include_directories(test/include/base_fixtures)

    add_executable(
        SortingTest

        # ---
        test/src/common/algorithms/SortingTest.cpp
    )
    target_link_libraries(
        SortingTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        SortingTest
        SortingTest
    )

    add_executable(
        UnsortedTest

        # ---
        test/src/search/buffer/UnsortedTest.cpp
    )
    target_link_libraries(
        UnsortedTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        UnsortedTest
        UnsortedTest
    )

    add_executable(
        WithMemoryTest

        # ---
        test/src/search/buffer/WithMemoryTest.cpp
    )
    target_link_libraries(
        WithMemoryTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        WithMemoryTest
        WithMemoryTest
    )

    add_executable(
        WithUnionFindTest

        # ---
        test/src/search/buffer/WithUnionFindTest.cpp
    )
    target_link_libraries(
        WithUnionFindTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        WithUnionFindTest
        WithUnionFindTest
    )

    add_executable(
        SearchTest

        # ---
        test/src/search/SearchTest.cpp
    )
    target_link_libraries(
        SearchTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        SearchTest
        SearchTest
    )

    add_executable(
        StatisticsTest

        # ---
        test/src/common/math/statistics/StatisticsTest.cpp
    )
    target_link_libraries(
        StatisticsTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        StatisticsTest
        StatisticsTest
    )

    add_executable(
        TransposeTest

        # ---
        test/src/common/math/linear_algebra/TransposeTest.cpp
    )
    target_link_libraries(
        TransposeTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        TransposeTest
        TransposeTest
    )

    add_executable(
        DistributionsTest

        # ---
        test/src/common/math/random/DistributionsTest.cpp
    )
    target_link_libraries(
        DistributionsTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        DistributionsTest
        DistributionsTest
    )

    add_executable(
        UtilsTest

        # ---
        test/src/common/UtilsTest.cpp
    )
    target_link_libraries(
        UtilsTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        UtilsTest
        UtilsTest
    )

    add_executable(
        KMedoidsTest

        # ---
        test/src/kmedoids/KMedoidsTest.cpp
    )
    target_link_libraries(
        KMedoidsTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        KMedoidsTest
        KMedoidsTest
    )

    add_executable(
        KMeansTest

        # ---
        test/src/kmeans/KMeansTest.cpp
    )
    target_link_libraries(
        KMeansTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        KMeansTest
        KMeansTest
    )

    add_executable(
        DBSCANTest

        # ---
        test/src/dbscan/DBSCANTest.cpp
    )
    target_link_libraries(
        DBSCANTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        DBSCANTest
        DBSCANTest
    )

    # ---
    add_executable(
        BoundsDistanceTest

        # ---
        test/src/datastruct/bounds/distance/BoundsDistanceTest.cpp
    )
    target_link_libraries(
        BoundsDistanceTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        BoundsDistanceTest
        BoundsDistanceTest
    )

    # ---
    add_executable(
        CondensedClusterTreeTest

        # ---
        test/src/datastruct/tree/condensed_cluster_tree/CondensedClusterTreeTest.cpp
    )
    target_link_libraries(
        CondensedClusterTreeTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        CondensedClusterTreeTest
        CondensedClusterTreeTest
    )

    add_executable(
        HDBSCANTest

        # ---
        test/src/hdbscan/HDBSCANTest.cpp
    )
    target_link_libraries(
        HDBSCANTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        HDBSCANTest
        HDBSCANTest
    )

    add_executable(
        KDTreeAlgorithmsTest

        # ---
        test/src/datastruct/tree/kdtree/KDTreeAlgorithmsTest.cpp
    )
    target_link_libraries(
        KDTreeAlgorithmsTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        KDTreeAlgorithmsTest
        KDTreeAlgorithmsTest
    )

    add_executable(
        SplittingRulePolicyTest

        # ---
        test/src/datastruct/tree/kdtree/policy/SplittingRulePolicyTest.cpp
    )
    target_link_libraries(
        SplittingRulePolicyTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        SplittingRulePolicyTest
        SplittingRulePolicyTest
    )

    add_executable(
        AxisSelectionPolicyTest

        # ---
        test/src/datastruct/tree/kdtree/policy/AxisSelectionPolicyTest.cpp
    )
    target_link_libraries(
        AxisSelectionPolicyTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        AxisSelectionPolicyTest
        AxisSelectionPolicyTest
    )

    add_executable(
        KDTreeTest

        # ---
        test/src/datastruct/tree/kdtree/KDTreeTest.cpp
    )
    target_link_libraries(
        KDTreeTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        KDTreeTest
        KDTreeTest
    )

    add_executable(
        BoruvkasAlgorithmTest

        # ---
        test/src/datastruct/graph/spanning_tree/BoruvkasAlgorithmTest.cpp
    )
    target_link_libraries(
        BoruvkasAlgorithmTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        BoruvkasAlgorithmTest
        BoruvkasAlgorithmTest
    )

    add_executable(
        SingleLinkageClusterTreeTest

        # ---
        test/src/datastruct/tree/single_linkage_cluster_tree/SingleLinkageClusterTreeTest.cpp
    )
    target_link_libraries(
        SingleLinkageClusterTreeTest

        # PUBLIC
        # TBB::tbb
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        SingleLinkageClusterTreeTest
        SingleLinkageClusterTreeTest
    )

    set_tests_properties(SortingTest DistributionsTest PROPERTIES
        LABELS "DistributionTests"
    )

    set_tests_properties(UtilsTest KDTreeAlgorithmsTest PROPERTIES
        LABELS "UtilityTests"
    )

    set_tests_properties(KDTreeTest PROPERTIES
        LABELS "IndexingTests"
    )

    set_tests_properties(DBSCANTest KMedoidsTest KMeansTest PROPERTIES
        LABELS "ClusteringTests"
    )

else()
    add_library(
        ${PROJECT_NAME}
        SHARED
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
    target_link_libraries(
        ${PROJECT_NAME}

        # PUBLIC
        # TBB::tbb
        OpenMP::OpenMP_CXX
    )
endif()