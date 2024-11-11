//
// Created by nik on 6/24/2024.
//


#include <gtest/gtest.h>

#include "../library/nml/primitives/file.h"

using namespace nml;

TEST(file_tests, count_csv)
{
    uint64_t time = 0, trials = 5;

    for (uint32_t i = 0; i < trials; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();

        auto parser = DataReader::delimited_iterator(R"(D:\Coding\Python\data\count.csv)");

        uint64_t cell_ct = 0;

        while (parser.has_next())
        {
            auto next = parser.next();

            if (next.type == DataReader::DelimitedTokenType::VALUE)
            {
                cell_ct += 1;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();

        ASSERT_GT(cell_ct, 10);

        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    std::cout << "Average Time (ms): " << time / 1e+6 / trials << "\n";
}
