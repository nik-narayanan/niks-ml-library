cmake_minimum_required(VERSION 3.25)
project(NML)

set(CMAKE_CXX_STANDARD 20)

add_definitions(-DPROJECT_DIR="${CMAKE_SOURCE_DIR}")

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

FetchContent_MakeAvailable(googletest)

function(add_test TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} gtest_main)
endfunction()

add_executable(iris_logistic_sample library/nml/samples/iris_logistic.cpp)

add_test(pca_tests library/nml/tests/pca_tests.cpp)
add_test(svd_tests library/nml/tests/svd_tests.cpp)
add_test(file_tests library/nml/tests/file_tests.cpp)
add_test(hash_tests library/nml/tests/hash_tests.cpp)
add_test(tree_tests library/nml/tests/tree_tests.cpp)
add_test(algorithm_tests library/nml/tests/algorithm_tests.cpp)
add_test(data_frame_tests library/nml/tests/data_frame_tests.cpp)
add_test(eigen_solver_tests library/nml/tests/eigen_solver_tests.cpp)
add_test(matrix_basic_tests library/nml/tests/matrix_basic_tests.cpp)
add_test(qr_algorithm_tests library/nml/tests/qr_algorithm_tests.cpp)
add_test(data_structures_tests library/nml/tests/data_structures_tests.cpp)
add_test(lu_decomposition_tests library/nml/tests/lu_decomposition_tests.cpp)
add_test(regular_expression_tests library/nml/tests/regular_expression_tests.cpp)
add_test(logistic_regression_tests library/nml/tests/logistic_regression_tests.cpp)
add_test(distribution_functions_tests library/nml/tests/distribution_functions_tests.cpp)
add_test(ordinary_least_squares_tests library/nml/tests/ordinary_least_squares_tests.cpp)
add_test(gauss_jordan_elimination_tests library/nml/tests/gauss_jordan_elimination_tests.cpp)
add_test(approximate_nearest_neighbor_tests library/nml/tests/approximate_nearest_neighbor_tests.cpp)
add_test(randomized_low_rank_approximation_tests library/nml/tests/randomized_low_rank_approximation_tests.cpp)
