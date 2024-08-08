//
// Created by nik on 5/8/2024.
//

#include <gtest/gtest.h>

#include "../primitives/file.h"
#include "../primitives/memory_owner.h"
#include "../algorithms/ordinary_least_squares.h"

using namespace nml;

const float zero_threshold = 1e-06;

TEST(ordinary_least_squares_tests, random_ols)
{
    auto test_matrix = MatrixOwner(100'000, 4);

    auto test_span = test_matrix.to_span();

    test_span.fill_random_gaussian();

    for (unsigned row = 0; row < test_span.row_ct; ++row)
    {
        auto test_row = test_span[row];

        test_row[0] += 4 + 5 * test_row[1] + 7 * (test_row[2]) + 8 * (test_row[3]);
    }

    auto request = OLS::Request(test_span);

    auto required_memory = OLS::required_memory(request);

    auto memory = MemoryOwner(required_memory.total_bytes());

    auto ols_result = OLS::compute(request, memory.to_request_memory(required_memory));

    ASSERT_TRUE(ols_result.is_ok());

    auto ols = ols_result.ok();

    ols.summary(test_span).print();
}