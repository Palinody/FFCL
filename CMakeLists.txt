cmake_minimum_required(VERSION 3.0.2)

project(ffcl)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-std=c++17 -Ofast -march=native -Wall -Wextra -Werror -Wfloat-equal -Wfatal-errors -fopenmp)

# add_compile_options(-std=c++17 -g -pg)
find_package(OpenMP REQUIRED)

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

    include/ffcl/math/random/Distributions.hpp
    include/ffcl/math/random/VosesAliasMethod.hpp

    include/ffcl/containers/kdtree/KDTree.hpp
    include/ffcl/containers/kdtree/KDNodeView.hpp
    include/ffcl/containers/kdtree/KDTreeUtils.hpp
    include/ffcl/containers/LowerTriangleMatrix.hpp

    include/ffcl/heuristics/Heuristics.hpp
    include/ffcl/heuristics/SilhouetteMethod.hpp

    include/ffcl/common/Utils.hpp
    include/ffcl/common/Timer.hpp
)

set(SOURCE_FILES
)

include_directories(
    include
)

add_definitions(-DTHREADS_ENABLED=true)
add_definitions(-DVERBOSE=false)

# disable all assertions
add_definitions(-DNDEBUG)

# set(MODE "run-benchmarks")
if(MODE STREQUAL "benchmark")
    add_executable(
        ${PROJECT_NAME}
        src/main.cpp
        ${HEADER_FILES}
        ${SOURCE_FILES}
    )
    target_link_libraries(
        ${PROJECT_NAME}
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
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
        )

        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    # -------------------------------
    enable_testing()

    add_compile_options(-std=c++17 -Ofast -march=native -Wall -Wextra -Werror -Wfloat-equal -Wfatal-errors -fopenmp)

    include(CTest)

    add_executable(
        DistributionsTest

        # ---
        test/DistributionsTest.cpp
    )
    target_link_libraries(
        DistributionsTest
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
        test/UtilsTest.cpp
    )
    target_link_libraries(
        UtilsTest
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
        test/KMedoidsTest.cpp
    )
    target_link_libraries(
        KMedoidsTest
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
        test/KMeansTest.cpp
    )
    target_link_libraries(
        KMeansTest
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        KMeansTest
        KMeansTest
    )

    add_executable(
        KDTreeTest

        # ---
        test/KDTreeTest.cpp
    )
    target_link_libraries(
        KDTreeTest
        GTest::gtest_main
        GTest::gmock_main
        OpenMP::OpenMP_CXX
    )
    add_test(
        KDTreeTest
        KDTreeTest
    )

    add_test(
        NAME AllTests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -R "DistributionsTest|UtilsTest|KMedoidsTest|KMeansTest|KDTreeTest"
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
        OpenMP::OpenMP_CXX
    )
endif()
