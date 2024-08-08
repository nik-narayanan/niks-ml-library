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

add_executable(iris_logistic_sample library/nml/samples/iris_logistic.cpp)

add_executable(pca_tests library/nml/tests/pca_tests.cpp)
add_executable(svd_tests library/nml/tests/svd_tests.cpp)
add_executable(file_tests library/nml/tests/file_tests.cpp)
add_executable(hash_tests library/nml/tests/hash_tests.cpp)
add_executable(tree_tests library/nml/tests/tree_tests.cpp)
add_executable(algorithm_tests library/nml/tests/algorithm_tests.cpp)
add_executable(data_frame_tests library/nml/tests/data_frame_tests.cpp)
add_executable(eigen_solver_tests library/nml/tests/eigen_solver_tests.cpp)
add_executable(matrix_basic_tests library/nml/tests/matrix_basic_tests.cpp)
add_executable(qr_algorithm_tests library/nml/tests/qr_algorithm_tests.cpp)
add_executable(data_structures_tests library/nml/tests/data_structures_tests.cpp)
add_executable(lu_decomposition_tests library/nml/tests/lu_decomposition_tests.cpp)
add_executable(regular_expression_tests library/nml/tests/regular_expression_tests.cpp)
add_executable(logistic_regression_tests library/nml/tests/logistic_regression_tests.cpp)
add_executable(distribution_functions_tests library/nml/tests/distribution_functions_tests.cpp)
add_executable(ordinary_least_squares_tests library/nml/tests/ordinary_least_squares_tests.cpp)
add_executable(gauss_jordan_elimination_tests library/nml/tests/gauss_jordan_elimination_tests.cpp)
add_executable(approximate_nearest_neighbor_tests library/nml/tests/approximate_nearest_neighbor_tests.cpp)
add_executable(randomized_low_rank_approximation_tests library/nml/tests/randomized_low_rank_approximation_tests.cpp)

target_link_libraries(pca_tests gtest_main)
target_link_libraries(svd_tests gtest_main)
target_link_libraries(file_tests gtest_main)
target_link_libraries(hash_tests gtest_main)
target_link_libraries(tree_tests gtest_main)
target_link_libraries(algorithm_tests gtest_main)
target_link_libraries(data_frame_tests gtest_main)
target_link_libraries(eigen_solver_tests gtest_main)
target_link_libraries(matrix_basic_tests gtest_main)
target_link_libraries(qr_algorithm_tests gtest_main)
target_link_libraries(data_structures_tests gtest_main)
target_link_libraries(lu_decomposition_tests gtest_main)
target_link_libraries(regular_expression_tests gtest_main)
target_link_libraries(logistic_regression_tests gtest_main)
target_link_libraries(distribution_functions_tests gtest_main)
target_link_libraries(ordinary_least_squares_tests gtest_main)
target_link_libraries(gauss_jordan_elimination_tests gtest_main)
target_link_libraries(approximate_nearest_neighbor_tests gtest_main)
target_link_libraries(randomized_low_rank_approximation_tests gtest_main)