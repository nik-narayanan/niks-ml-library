//
// Created by nik on 5/26/2024.
//

#include <gtest/gtest.h>
#include "datasets/test_data.h"

#include "../library/nml/primitives/file.h"
#include "../library/nml/primitives/memory_owner.h"
#include "../library/nml/primitives/matrix_owner.h"
#include "../library/nml/algorithms/logistic_regression.h"

using namespace nml;

const float zero_threshold = 1e-06;

TEST(logistic_regression_tests, random_mle)
{
    unsigned sample_size = 100'000;

    float label = 1.0f;

    auto test_matrix = MatrixOwner(sample_size, 3);

    auto test_span = test_matrix.to_span();

    test_span.fill_random_gaussian();

    auto random = Random();

    for (unsigned row = 0; row < test_span.row_ct; ++row)
    {
        auto test_row = test_span[row];

        float linear_combination = 1 + 6 * test_row[1] - 9 * test_row[2];

        float probability = 1.0f / (1.0f + std::exp(-linear_combination));

        test_row[0] = random.value(0, 1) < probability ? label : 0.0f;
    }

    auto request = LogisticRegression::Request(test_span);

    request.label = label;

    auto required_memory = LogisticRegression::required_memory(request);

    auto memory = MemoryOwner(required_memory.total_bytes());

    auto result = LogisticRegression::compute(request, memory.to_request_memory(required_memory));

    ASSERT_TRUE(result.is_ok());

    auto regression = result.ok();

    regression.summary(test_span).print();
}

TEST(logistic_regression_tests, iris_single)
{
    float label = 2.0f;

    auto test_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_LAB));

    auto test_span = test_matrix.to_span();

    auto request = LogisticRegression::Request(test_span);

    request.label = label;

    auto required_memory = LogisticRegression::required_memory(request);

    auto memory = MemoryOwner(required_memory.total_bytes());

    auto result = LogisticRegression::compute(request, memory.to_request_memory(required_memory));

    ASSERT_TRUE(result.is_ok());

    auto regression = result.ok();

    regression.summary(test_span).print();
}

TEST(logistic_regression_tests, iris_multi)
{
    auto test_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_LAB));

    auto test_span = test_matrix.to_span();

    auto labels = test_span.distinct(0);

    auto request = LogisticRegressionMulti::Request(test_span, VectorSpan(labels.to_memory()));

    RequiredMemory required_memory = LogisticRegressionMulti::required_memory(request);

    auto memory = MemoryOwner(required_memory.total_bytes());

    auto result = LogisticRegressionMulti::compute(request, memory.to_request_memory(required_memory));

    ASSERT_TRUE(result.is_ok());

    auto regression = result.ok();

    regression.summary(test_span).print();
}