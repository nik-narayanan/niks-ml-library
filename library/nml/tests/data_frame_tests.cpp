//
// Created by nik on 6/1/2024.
//

#include <gtest/gtest.h>
#include "datasets/test_data.h"

#include "../primitives/data_frame.h"
#include "../primitives/memory_owner.h"

using namespace nml;

const bool print_results = false;

const float zero_threshold = 1e-06;

TEST(data_frame_tests, basic_operations)
{
    auto df = DataFrame();

    auto ints = ResizableList<int64_t>();

    for (int i = 0; i < 10; ++i)
    {
        ints.add(i);
    }

    df.add_column(ints.to_span()).set_column_name("one");
    df.add_column(ints.to_span()).set_column_name("two");

    df.add_row({1, 2});

    df[0][1] = -9223372036854775807LL;
    df[1][1] = 9223372036854775807LL;

    for (int64_t i = 2; i < 10; ++i)
    {
        df[i][1] += -i * 10000000000000000;
    }

    auto memory = MemoryOwner(2 * sizeof(DataFrame::Value));

    df.add_column(DataFrame::Type::INT, "int column");

    for (int i = 0; i < 10; ++i)
    {
        df.add_row({i, i + 1, i + 3});
    }

    df.add_column(df.get_column_unsafe(1));
    auto float_col = df.add_column(DataFrame::Type::FLOAT, "floats", 0);

    float_col[0] = 1.1;

    auto copy = df.add_column(df.get_column_unsafe(0));

    copy.set_column_name("copy column");

    float_col = -11111111111111.1111111111111f;
    float_col += 1.1f;

    df[0][0] = 1 * 10000000000000000.0f;
    df[df.row_count() - 1][df.column_count() - 1] = 2.1f;

    auto str_col = df.add_column(DataFrame::Value("test string 1"), "dynamic");
    df.print(5);
    str_col = 100.001;
    df.print(5);
    str_col = "string 100.123456789012345";
    df.print(5);
    str_col = 1;
    df.print(5);
    str_col = DataFrame::Value::parse_datetime("1992-02-18");
    df.print(5);
}

TEST(data_frame_tests, parse_numbers)
{
    auto value = DataFrame::Value::parse_value(".2");

    ASSERT_EQ(value.type, DataFrame::Type::FLOAT);
    ASSERT_EQ(value.value_float, 0.2);


}

TEST(data_frame_tests, read_floats_from_file)
{
    auto df = DataFrame::from_delimited(TestData::test_file_path(Dataset::IRIS_VAL), false);

    for (uint32_t col = 0; col < df.column_count(); ++col)
    {
        auto column = df.get_column_unsafe(col);

        ASSERT_EQ(column.get_type(), DataFrame::Type::FLOAT);

        for (uint32_t i = 0; i < df.row_count(); ++i)
        {
            ASSERT_TRUE(!column[i].is_null());
        }
    }

    ASSERT_EQ(df.column_count(), 4);
    ASSERT_EQ(df.row_count(), 150);
}

TEST(data_frame_tests, read_from_file)
{
    auto df = DataFrame::from_delimited(TestData::test_file_path(Dataset::IRIS_RAW));

    for (uint32_t col = 0; col < df.column_count(); ++col)
    {
        auto column = df.get_column_unsafe(col);

        if (col < 4)
        {
            ASSERT_EQ(column.get_type(), DataFrame::Type::FLOAT);
        }
        else
        {
            ASSERT_EQ(column.get_type(), DataFrame::Type::STRING);
        }

        for (uint32_t i = 0; i < df.row_count(); ++i)
        {
            ASSERT_TRUE(!column[i].is_null());
        }
    }

    ASSERT_EQ(df.column_count(), 5);
    ASSERT_EQ(df.row_count(), 150);
}