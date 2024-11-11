//
// Created by nik on 3/25/2024.
//

#include <gtest/gtest.h>

#include "../library/nml/primitives/file.h"
#include "../library/nml/algorithms/_algorithms.h"
#include "../library/nml/primitives/memory_owner.h"

using namespace nml;

#define PRINT_RESULTS false
#define RUN_BENCHMARKS true

const float zero_threshold = 1e-06;


//TEST(algorithm_tests, gaussian_elimination)
//{
//    MatrixOwner test_matrix = MatrixOwner(4, 5, {
//            1, 0, 2, -1, 1,
//            3, 0, 0, 5, 1,
//            2, 1, 4, -3, 1,
//            1, 0, 5, 0, 1,
//    });
//
//    MatrixSpan test_span = test_matrix.to_span();
//
//    gauss_jordan_elimination(test_span, zero_threshold);
//
//    ASSERT_TRUE(std::abs(test_span.get_unsafe(0, 4) -  0.666667) < zero_threshold * 10);
//    ASSERT_TRUE(std::abs(test_span.get_unsafe(1, 4) - -1.200000) < zero_threshold * 10);
//    ASSERT_TRUE(std::abs(test_span.get_unsafe(2, 4) -  0.066667) < zero_threshold * 10);
//    ASSERT_TRUE(std::abs(test_span.get_unsafe(3, 4) - -0.200000) < zero_threshold * 10);
//}

//TEST(algorithm_tests, cholesky_decomposition)
//{
//    MatrixOwner test_matrix = MatrixOwner(4, 5, {
//            1.00671141, -0.11835884, 0.87760447, 0.82343066, 1,
//            -0.11835884,  1.00671141, -0.43131554, -0.36858315, 1,
//            0.87760447, -0.43131554,  1.00671141,  0.96932762, 1,
//            0.82343066, -0.36858315,  0.96932762,  1.00671141, 1,
//    });
//
//    MatrixOwner expected_matrix = MatrixOwner(4, 4, {
//            1.003350,   0.000000,   0.000000,   0.000000,
//           -0.117964,   0.996391,   0.000000,   0.000000,
//            0.874674,  -0.329324,   0.364968,   0.000000,
//            0.820681,  -0.272757,   0.442979,   0.250133,
//    });
//
//    MatrixOwner working_matrix = MatrixOwner(4, 4);
//
//    MatrixSpan
//        test_span = test_matrix.to_span(),
//        working_span = working_matrix.to_span(),
//        expected_span = expected_matrix.to_span()
//    ;
//
//    cholesky_decomposition(test_span, working_span);
//
//    ASSERT_TRUE(working_span.equals(expected_span, 1e-05));
//}
//
//TEST(algorithm_tests, eigen)
//{
//    MatrixOwner test_matrix = MatrixOwner(4, 4, {
//        1.00671141, -0.11835884, 0.87760447, 0.82343066,
//        -0.11835884,  1.00671141, -0.43131554, -0.36858315,
//        0.87760447, -0.43131554,  1.00671141,  0.96932762,
//        0.82343066, -0.36858315,  0.96932762,  1.00671141
//    });
//
//    MatrixSpan test_span = test_matrix.to_span();
//
//    unsigned memory_length = test_span.row_ct * test_span.column_ct * 3;
//    auto memory = static_cast<float*>(std::malloc(memory_length * sizeof(float)));
//    auto span_memory = VectorSpan(memory, memory_length);
//
//    {
//        auto eigen_result = compute_eigen(test_span, span_memory, linear_algebra::QRDecompositionType::GIVENS, zero_threshold);
//        ASSERT_TRUE(eigen_result.is_ok());
////        auto eigen = eigen_result.ok();
////        eigen.eigenvalues.print();
////        eigen.eigenvectors.print();
//    }
//
//    {
//        auto eigen_result = compute_eigen(test_span, span_memory, linear_algebra::QRDecompositionType::HOUSEHOLDER, zero_threshold);
//        ASSERT_TRUE(eigen_result.is_ok());
////        auto eigen = eigen_result.ok();
////        eigen.eigenvalues.print();
////        eigen.eigenvectors.print();
//    }
//
//    std::free(memory);
//}
//
//TEST(algorithm_tests, pca)
//{
//    MatrixOwner test_matrix = get_new_dataset_matrix(Dataset::IRIS_VAL),
//                expected_matrix = get_new_dataset_matrix(Dataset::IRIS_PCA);
//
//    MatrixSpan test_span = test_matrix.to_span(), expected_span = expected_matrix.to_span();
//
//    unsigned working_memory_length = pca_required_memory_length(test_span),
//             result_memory_length = pca_result_memory_length(test_span, test_span.column_ct);
//
//    auto memory = static_cast<float*>(std::malloc((result_memory_length + working_memory_length) * sizeof(float)));
//
//    VectorSpan result_memory = VectorSpan(memory, result_memory_length),
//         working_memory = VectorSpan(memory + result_memory_length, working_memory_length);
//
//    {
//        auto pca_result = pca(test_span, result_memory, working_memory, test_span.column_ct, zero_threshold);
//        ASSERT_TRUE(pca_result.is_ok());
//        auto pca = pca_result.ok();
//        ASSERT_TRUE(expected_span.equals(pca.projection, 1e-04));
//    }
//
//    std::free(memory);
//}
