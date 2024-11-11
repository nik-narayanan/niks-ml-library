//
// Created by nik on 3/27/2024.
//

#include <gtest/gtest.h>

#include "../library/nml/primitives/file.h"
#include "../library/nml/algorithms/eigen_solver.h"
#include "../library/nml/primitives/memory_owner.h"

using namespace nml;

const float zero_threshold = 1e-06;

TEST(eigen_solver_tests, eigen)
{
    MatrixOwner test_matrix = MatrixOwner(4, 4, {
         1.00671141, -0.11835884,  0.87760447,  0.82343066,
        -0.11835884,  1.00671141, -0.43131554, -0.36858315,
         0.87760447, -0.43131554,  1.00671141,  0.96932762,
         0.82343066, -0.36858315,  0.96932762,  1.00671141
    });

    MatrixOwner expected_matrix = MatrixOwner(4, 4, {
         0.521067, 0.3774190,  0.719563, -0.261292,
        -0.269344, 0.923296 , -0.244384,  0.123512,
         0.580413, 0.0244887, -0.142120,  0.801451,
         0.564857, 0.0669382, -0.634277, -0.523592
    });

    MatrixSpan test_span = test_matrix.to_span();
    MatrixSpan expected_span = expected_matrix.to_span();

    auto eigen_request = Eigen::Request(test_span);

    auto required_memory = Eigen::required_memory(eigen_request);
    auto memory = MemoryOwner(required_memory.total_bytes());
    auto request_memory = memory.to_request_memory(required_memory);

    {
        eigen_request.type = QRDecomposition::Type::GIVENS;
        auto eigen_result = Eigen::compute(eigen_request, request_memory);
        ASSERT_TRUE(eigen_result.is_ok());
        auto eigen = eigen_result.ok();
        eigen.eigenvalues.print();
        ASSERT_TRUE(expected_span.equals(eigen.eigenvectors, 1e-3));
    }

    {
        eigen_request.type = QRDecomposition::Type::HOUSEHOLDER;
        auto eigen_result = Eigen::compute(eigen_request, request_memory);
        ASSERT_TRUE(eigen_result.is_ok());
        auto eigen = eigen_result.ok();
        ASSERT_TRUE(expected_span.equals(eigen.eigenvectors, 1e-3));
    }
}