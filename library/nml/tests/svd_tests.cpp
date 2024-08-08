//
// Created by nik on 5/6/2024.
//

#include <gtest/gtest.h>
#include "datasets/test_data.h"

#include "../primitives/file.h"
#include "../algorithms/svd.h"
#include "../primitives/memory_owner.h"

using namespace nml;

TEST(svd_tests, get_low_dimensional_projection)
{
    MatrixOwner test_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_VAL));

//    MatrixOwner expected_left_singular_vectors = MatrixOwner(150, 4);

    MatrixOwner expected_singular_values = MatrixOwner(1, 2, {
        25.0999611, 6.01314612
    });

    MatrixOwner expected_right_singular_vectors = MatrixOwner(2, 4, {
        0.36138659, -0.08452251,  0.85667061,  0.35828920,
        0.65658877,  0.73016143, -0.17337266, -0.07548102
    });

    MatrixSpan test_span = test_matrix.to_span();

    test_span.center();

    auto request = SVD::Request(test_span, 2);

    auto required_memory = SVD::required_memory(request);

    auto memory = MemoryOwner(required_memory.total_bytes());

    auto svd_result = SVD::compute(request, memory.to_request_memory(required_memory));

    ASSERT_TRUE(svd_result.is_ok());

    auto svd = svd_result.ok();

//    svd.singular_values.print();
//    svd.left_singular_vectors.print();
//    svd.right_singular_vectors.print();

    ASSERT_TRUE(svd.singular_values.equals(expected_singular_values.to_span()[0], 1e-5));
//    ASSERT_TRUE(svd.left_singular_vectors.equals(expected_left_singular_vectors.to_span(), 1e-2));
    ASSERT_TRUE(svd.right_singular_vectors.equals(expected_right_singular_vectors.to_span(), 1e-5));
}