//
// Created by nik on 5/6/2024.
//

#include <gtest/gtest.h>
#include "datasets/test_data.h"

#include "../primitives/file.h"
#include "../primitives/memory_owner.h"
#include "../algorithms/randomized_low_rank_approximation.h"

using namespace nml;

TEST(randomized_low_rank_approximation_tests, get_low_dimensional_projection)
{
    MatrixOwner test_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_VAL));

    MatrixOwner expected_matrix = MatrixOwner(4, 4, {
        9.07078857, -2.12151241, 21.50239848,  8.99304474,
        3.94992547,  4.38875769, -1.04274778, -0.45551515,
        1.98508528, -2.04669369, -0.26194056, -1.85877252,
        0.58877268, -0.59663162, -0.90350648,  1.42567537,
    });

    MatrixSpan test_span = test_matrix.to_span();

    test_span.center();

    auto request = RandomizedLowRankApproximation::Request(test_span);

    auto required_memory = RandomizedLowRankApproximation::required_memory(request);

    auto memory = MemoryOwner(required_memory.total_bytes());

    auto projection_result = RandomizedLowRankApproximation::compute(request, memory.to_request_memory(required_memory));

    ASSERT_TRUE(projection_result.is_ok());

    auto projection = projection_result.ok();

    if (false) projection.print();

    ASSERT_TRUE(projection.equals(expected_matrix.to_span(), 1e-2));
}