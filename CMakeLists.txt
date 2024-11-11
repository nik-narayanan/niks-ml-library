cmake_minimum_required(VERSION 3.25)
project(NML)

enable_testing()

set(CMAKE_CXX_STANDARD 20)

add_definitions(-DPROJECT_DIR="${CMAKE_SOURCE_DIR}")

add_library(nml INTERFACE)
target_include_directories(nml INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/library)

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

FetchContent_MakeAvailable(googletest)

function(add_gtest  TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} gtest_main)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endfunction()

function(add_test_simd TARGET_NAME SOURCE_FILE)
    add_gtest(${TARGET_NAME} ${SOURCE_FILE})

    if (MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /arch:AVX)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -msse -msse2 -mavx -march=native)
    endif()
endfunction()

add_executable(iris_logistic_sample samples/iris_logistic.cpp)

add_gtest(pca_tests tests/pca_tests.cpp)
add_gtest(svd_tests tests/svd_tests.cpp)
#add_test_simd(file_tests tests/file_tests.cpp)
add_gtest(hash_tests tests/hash_tests.cpp)
add_gtest(tree_tests tests/tree_tests.cpp)
add_gtest(algorithm_tests tests/algorithm_tests.cpp)
add_gtest(allocator_tests tests/allocator_tests.cpp)
add_gtest(data_frame_tests tests/data_frame_tests.cpp)
add_gtest(eigen_solver_tests tests/eigen_solver_tests.cpp)
add_gtest(matrix_basic_tests tests/matrix_basic_tests.cpp)
add_gtest(qr_algorithm_tests tests/qr_algorithm_tests.cpp)
add_gtest(data_structures_tests tests/data_structures_tests.cpp)
add_gtest(lu_decomposition_tests tests/lu_decomposition_tests.cpp)
add_gtest(regular_expression_tests tests/regular_expression_tests.cpp)
add_gtest(logistic_regression_tests tests/logistic_regression_tests.cpp)
add_gtest(distribution_functions_tests tests/distribution_functions_tests.cpp)
add_gtest(ordinary_least_squares_tests tests/ordinary_least_squares_tests.cpp)
add_gtest(gauss_jordan_elimination_tests tests/gauss_jordan_elimination_tests.cpp)
add_gtest(approximate_nearest_neighbor_tests tests/approximate_nearest_neighbor_tests.cpp)
add_gtest(randomized_low_rank_approximation_tests tests/randomized_low_rank_approximation_tests.cpp)