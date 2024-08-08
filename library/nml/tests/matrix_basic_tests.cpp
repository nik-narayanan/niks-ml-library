//
// Created by nik on 3/24/2024.
//


#include <gtest/gtest.h>
#include "datasets/test_data.h"

#include "../primitives/matrix_span.h"
#include "../primitives/matrix_owner.h"
#include "../primitives/file.h"
#include "../primitives/span.h"

using namespace nml;

const bool print_results = false;

const float zero_threshold = 1e-06;

TEST(matrix_transform_tests, parse_float)
{
    const char* string = ".2";

    auto fl = matrix_owner_internal::parse_float(Span<const char>(string, strlen(string)));

    ASSERT_EQ(0.2f, static_cast<float>(fl));
}

TEST(matrix_transform_tests, get_set)
{
    float test_value = 99;

    auto test_matrix = MatrixOwner(4, 4, {
        1,  1,  1,  1,
        1,  1,  1,  1,
        1,  1,  1,  1,
        1,  1,  1,  1,
    });

    {
        auto test_matrix_view = test_matrix.to_span_unsafe(2, 2, 0, 0);
        test_matrix_view.set_unsafe(1, 1, test_value);
        float value = test_matrix.to_span().get_unsafe(1, 1);
        ASSERT_EQ(value, test_value);
    }

    {
        MatrixSpan test_matrix_span = test_matrix.to_span();
        test_matrix_span.fill(1);
        auto test_matrix_view = test_matrix_span.to_subspan_unsafe(2, 2, 0, 0);
        test_matrix_view.set_unsafe(1, 1, test_value);
        float value = test_matrix.to_span().get_unsafe(1, 1);
        ASSERT_EQ(value, test_value);
    }

    {
        MatrixSpan test_matrix_span = test_matrix.to_span();
        test_matrix_span.fill(1);
        auto test_matrix_view = test_matrix.to_span_unsafe(2, 2, 1, 1);
        test_matrix_view.set_unsafe(1, 1, test_value);
        float value = test_matrix.to_span().get_unsafe(2, 2);
        ASSERT_EQ(value, test_value);
    }
}

TEST(matrix_transform_tests, copy)
{
    auto mini_matrix = MatrixOwner(2, 2, {
        99, 99,
        99, 99
    });

    auto left_matrix = MatrixOwner(4, 4, {
        17, 12,  4, 17,
        1, 16, 19,  1,
        7,  8, 10,  4,
        22,  4,  2,  8
    });

    auto right_matrix = MatrixOwner(4, 4, {
        0.638124,   0.447214,  -0.718646,   1.577864,
        -1.306635,   1.341641,   1.550762,  -1.079591,
        -0.577350,  -0.447214,   0.189117,  -0.581318,
        1.245861,  -1.341641,  -1.021234,   0.083045
    });

    MatrixSpan
        left_span = left_matrix.to_span(),
        right_span = right_matrix.to_span(),
        mini_span = mini_matrix.to_span(),
        mini_test_span = left_matrix.to_span_unsafe(2, 2, 2, 2)
    ;

    left_span.copy_into_unsafe(right_matrix.get_pointer());

    ASSERT_TRUE(left_span.equals(right_span, zero_threshold));

    mini_test_span.copy_from_unsafe(mini_span);

    ASSERT_TRUE(mini_test_span.equals(mini_span, zero_threshold));

    if (print_results) right_span.print();
}

TEST(matrix_transform_tests, fill)
{
    auto test_matrix = MatrixOwner(3, 3);

    auto span = test_matrix.to_span();

    float fill_value = 1.0f;

    span.fill(fill_value);

    auto memory = test_matrix.get_pointer();

    for (int i = 0; i < span.element_count(); ++i)
    {
        ASSERT_EQ(memory[i], fill_value);
    }
}

TEST(matrix_transform_tests, multiply)
{
    auto left = MatrixOwner(2, 3, {
        1, 2, 3,
        4, 5, 6
    });

    auto right = MatrixOwner(3, 2, {
        7, 8,
        9, 10,
        11, 12
    });

    auto output = MatrixOwner(2, 2);

    MatrixSpan left_span = left.to_span(), right_span = right.to_span(), output_span = output.to_span();

    auto multiply_result = left_span.multiply(right_span, output_span);

    ASSERT_TRUE(multiply_result.is_ok());

    ASSERT_EQ(output_span.get_unsafe(0, 0), 58);
    ASSERT_EQ(output_span.get_unsafe(0, 1), 64);
    ASSERT_EQ(output_span.get_unsafe(1, 0), 139);
    ASSERT_EQ(output_span.get_unsafe(1, 1), 154);

    if (print_results) output_span.print();
}


TEST(matrix_transform_tests, multiply_2)
{
    auto left = MatrixOwner(3, 3, {
            0, 1, 0,
            -1, 0, 0,
            0, 0, 1
    });

    auto right = MatrixOwner(3, 3, {
            0, -1, 1,
            4, 2, 0,
            3, 4, 0
    });

    auto output = MatrixOwner(3, 3);

    MatrixSpan left_span = left.to_span(), right_span = right.to_span(), output_span = output.to_span();

    auto multiply_result = left_span.multiply(right_span, output_span);

    ASSERT_TRUE(multiply_result.is_ok());
//
//    ASSERT_EQ(output_span.get_unsafe(0, 0), 58);
//    ASSERT_EQ(output_span.get_unsafe(0, 1), 64);
//    ASSERT_EQ(output_span.get_unsafe(1, 0), 139);
//    ASSERT_EQ(output_span.get_unsafe(1, 1), 154);

    if (print_results) output_span.print();
}

TEST(matrix_transform_tests, standardize_basic)
{
    auto test_matrix = MatrixOwner(4, 4, {
            17, 12,  4, 17,
            1, 16, 19,  1,
            7,  8, 10,  4,
            22,  4,  2,  8
    });

    auto expected_matrix = MatrixOwner(4, 4, {
            0.638124,   0.447214,  -0.718646,   1.577864,
            -1.306635,   1.341641,   1.550762,  -1.079591,
            -0.577350,  -0.447214,   0.189117,  -0.581318,
            1.245861,  -1.341641,  -1.021234,   0.083045
    });

    MatrixSpan test_span = test_matrix.to_span(), expected_span = expected_matrix.to_span();

    test_span.standardize(zero_threshold);

    ASSERT_TRUE(test_span.equals(expected_span, zero_threshold));
}

TEST(matrix_transform_tests, standardize_iris)
{
    MatrixOwner test_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_VAL)),
            expected_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_STD));

    MatrixSpan test_span = test_matrix.to_span(), expected_span = expected_matrix.to_span();

    test_span.standardize(zero_threshold);

    ASSERT_TRUE(test_span.equals(expected_span, zero_threshold));
}

TEST(matrix_transform_tests, covariance_iris)
{
    MatrixOwner test_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_STD)),
            expected_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_COV)),
            output_matrix = MatrixOwner(4, 4);

    MatrixSpan test_span = test_matrix.to_span(), expected_span = expected_matrix.to_span(), output = output_matrix.to_span();

    test_span.covariance(output);

    ASSERT_TRUE(output.equals(expected_span, zero_threshold));
}

TEST(matrix_transform_tests, fill_random_uniform)
{
    MatrixOwner test_matrix = MatrixOwner(500, 500);

    MatrixSpan test_span = test_matrix.to_span();

    for (unsigned seed = 0; seed < 5'000'000; seed += ++seed * 113)
    {
        test_span.fill_random_uniform(-1, 1, seed);

        float sum = 0, min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();

        for (int i = 0; i < test_span.row_ct; ++i)
        {
            for (int j = 0; j < test_span.column_ct; ++j)
            {
                float value = test_span.get_unsafe(i, j);
                sum += value;
                min = std::min(min, value);
                max = std::max(max, value);
            }
        }

        ASSERT_TRUE(std::fabs(sum / test_span.element_count()) < 0.005);
        ASSERT_TRUE(max <= 1.0f);
        ASSERT_TRUE(min >= -1.0f);
    }
}

TEST(matrix_transform_tests, fill_random_gaussian)
{
    MatrixOwner test_matrix = MatrixOwner(500, 500);

    MatrixSpan test_span = test_matrix.to_span();

    for (unsigned seed = 0; seed < 5'000'000; seed += ++seed * 113)
    {
        test_span.fill_random_gaussian(1, seed);

        float sum = 0;

        for (int i = 0; i < test_span.row_ct; ++i)
        {
            for (int j = 0; j < test_span.column_ct; ++j)
            {
                float value = test_span.get_unsafe(i, j);
                sum += value;
            }
        }

        float mean = sum / test_span.element_count(), ssd = 0;

        for (int i = 0; i < test_span.row_ct; ++i)
        {
            for (int j = 0; j < test_span.column_ct; ++j)
            {
                float value = test_span.get_unsafe(i, j);
                ssd += (value - mean) * (value - mean);
            }
        }

        float sd = std::sqrt(ssd / (test_span.element_count() - 1));

        ASSERT_TRUE(std::fabs(sum / test_span.element_count()) < 0.005);
        ASSERT_TRUE(std::fabs(sd - 1) < 0.005);
    }
}

TEST(matrix_transform_tests, transpose_multiply)
{
    unsigned dimensions = 50, addition = 50;

    MatrixOwner test_1_matrix = MatrixOwner(dimensions + addition, dimensions);
    MatrixOwner test_2_matrix = MatrixOwner(dimensions + addition, dimensions + addition);
    MatrixOwner reconstruct_1_ = MatrixOwner(dimensions, dimensions + addition);

    MatrixSpan test_1_span = test_1_matrix.to_span(),
                test_2_span = test_2_matrix.to_span(),
                reconstruct_1 = reconstruct_1_.to_span();

    test_1_span.fill_random_uniform(0, 100, 42);
    test_2_span.set_identity();

    test_1_span.transpose_multiply(test_2_span, reconstruct_1);

    for (unsigned row = 0; row < test_1_span.row_ct; ++row)
    {
        for (unsigned column = 0; column < test_1_span.column_ct; ++column)
        {
            float original = test_1_span.get_unsafe(row, column);
            float transpose = reconstruct_1.get_unsafe(column, row);
            ASSERT_EQ(original, transpose);
        }
    }
}