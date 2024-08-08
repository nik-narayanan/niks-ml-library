//
// Created by nik on 5/14/2024.
//
#include <gtest/gtest.h>
#include "datasets/test_data.h"

#include "../primitives/file.h"
#include "../primitives/memory_owner.h"
#include "../primitives/matrix_owner.h"
#include "../algorithms/distribution_functions.h"

using namespace nml;

double round_power_of_ten(float num)
{
    if (num == 0) return 0;

    int exponent = static_cast<int>(std::round(std::log10(std::abs(num))));

    double power = std::pow(10, exponent);

    return (num < 0) ? -power : power;
}

TEST(distribution_functions_tests, beta_cdf)
{
    auto beta_cdf_tests = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::CDF_BETA));

    for (unsigned i = 0; i < beta_cdf_tests.row_ct; ++i)
    {
        auto test = beta_cdf_tests[i];

        auto p_value = Distribution::beta_cdf(test[0], test[1], test[2]);

        if (static_cast<float>(p_value) != test[3])
        {
            std::cout << "failed test: " << i << "\n";
            test.print();
        }

        ASSERT_EQ(static_cast<float>(p_value), test[3]);
    }
}

TEST(distribution_functions_tests, normal_cdf)
{
    auto normal_cdf_tests = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::CDF_NORM));

    for (unsigned i = 0; i < normal_cdf_tests.row_ct; ++i)
    {
        auto test = normal_cdf_tests[i];

        auto p_value = Distribution::normal_cdf(
            round_power_of_ten(test[0]),
            round_power_of_ten(test[1]),
            round_power_of_ten(test[2])
        );

        if (static_cast<float>(p_value) != test[3])
        {
            std::cout << "failed test: " << i << "\n";
            test.print();
        }

        ASSERT_EQ(static_cast<float>(p_value), test[3]);
    }
}