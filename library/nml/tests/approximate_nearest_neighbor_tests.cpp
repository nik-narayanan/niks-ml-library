//
// Created by nik on 4/8/2024.
//

#include <gtest/gtest.h>
#include "datasets/test_data.h"

#include "../primitives/file.h"
#include "../primitives/memory_owner.h"
#include "../primitives/matrix_owner.h"
#include "../algorithms/approximate_nearest_neighbor.h"

using namespace nml;

#define RUN_BENCHMARKS false

TEST(algorithm_tests, ann)
{
    unsigned thread_count = 1;

    MatrixOwner test_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_VAL));
    MatrixOwner expected_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_ANN));

    MatrixSpan test_matrix_span = test_matrix.to_span(), expected_matrix_span = expected_matrix.to_span();

    auto build_request = ApproximateNearestNeighborIndex::BuildRequest(test_matrix_span, 10, thread_count, 42);

    auto required_build_memory = ApproximateNearestNeighborIndex::build_required_memory(build_request);

    auto memory = MemoryOwner(required_build_memory.total_bytes());

    auto build_request_memory = RequestMemory
    {
        .result_memory = memory.to_memory_span(0, required_build_memory.result_required_bytes),
        .working_memory = memory.to_memory_span(required_build_memory.result_required_bytes),
    };

    auto ann_index_result = ApproximateNearestNeighborIndex::build(build_request, build_request_memory);

    auto ann_index = ann_index_result.ok();

    auto bulk_find_request = ApproximateNearestNeighborIndex::BulkFindRequest(12, 0, thread_count);

    auto bulk_find_required_memory = ann_index.bulk_find_required_memory(bulk_find_request);

    auto bulk_find_memory = MemoryOwner(bulk_find_required_memory.total_bytes());

    auto find_request_memory = RequestMemory::from_required_unsafe(bulk_find_required_memory, bulk_find_memory.to_memory_span());

    auto bulk_nearest_neighbors_result = ann_index.bulk_find_nearest_neighbors(bulk_find_request, find_request_memory);

    ASSERT_TRUE(bulk_nearest_neighbors_result.is_ok());

    auto nearest_neighbors = bulk_nearest_neighbors_result.ok();

    for (int row = 0; row < test_matrix.row_ct; ++row)
    {
        auto result_row = nearest_neighbors[row];
        auto expected_row = expected_matrix_span[row];

        for (int index = 0; index < expected_matrix_span.column_ct; ++index)
        {
            if (result_row[index].score != expected_row[index])
            {
                std::cout << "Row: " << row << ", Column: " << index << std::endl;

                for (int i = 0; i < result_row.length; ++i)
                {
                    if (i > 0) std::cout << ", ";
                    std::cout << result_row[i];
                }

                std::cout << std::endl;

                for (int i = 0; i < expected_row.length; ++i)
                {
                    if (i > 0) std::cout << ", ";
                    std::cout << expected_row[i];
                }

                std::cout << std::endl;
            }

            ASSERT_EQ(result_row[index].score,  expected_row[index]);
        }
    }
}

#if RUN_BENCHMARKS
TEST(algorithm_tests, ann_build_benchmark)
{
    MatrixOwner test_matrix = get_new_dataset_matrix(Dataset::IRIS_VAL);

    MatrixSpan test_span = test_matrix.to_span();

    const auto build_request = ApproximateNearestNeighborIndex::BuildRequest
    {
        .data = test_span,
        .tree_count = 3,
        .thread_count = 1,
        .seed = 42
    };

    auto required_build_memory = ApproximateNearestNeighborIndex::build_required_memory(build_request);

    auto memory = MemoryOwner(required_build_memory.total());

    auto build_request_memory = RequestMemory
    {
        .result_memory = memory.to_memory_span(0, required_build_memory.result_required_bytes),
        .working_memory = memory.to_memory_span(required_build_memory.result_required_bytes),
    };

    size_t time = 0, trials = 5000;

    for (int trial = 0; trial < trials; ++trial)
    {
        auto start = std::chrono::high_resolution_clock::now();

        auto ann_index_result = ApproximateNearestNeighborIndex::build(build_request, build_request_memory);

        auto end = std::chrono::high_resolution_clock::now();

        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    std::cout << "Average Build Time (ns): " << time / trials << "\n";
}

TEST(algorithm_tests, ann_find_benchmark)
{
    MatrixOwner test_matrix = get_new_dataset_matrix(Dataset::IRIS_VAL);

    MatrixSpan test_span = test_matrix.to_span();

    const auto build_request = ApproximateNearestNeighborIndex::BuildRequest
    {
            .data = test_span,
            .tree_count = 3,
            .thread_count = 1,
            .seed = 42
    };

    auto required_build_memory = ApproximateNearestNeighborIndex::build_required_memory(build_request);

    auto memory = MemoryOwner(required_build_memory.total());

    auto build_request_memory = RequestMemory
    {
        .result_memory = memory.to_memory_span(0, required_build_memory.result_required_bytes),
        .working_memory = memory.to_memory_span(required_build_memory.result_required_bytes),
    };

    auto ann_index_result = ApproximateNearestNeighborIndex::build(build_request, build_request_memory);

    auto ann_index = ann_index_result.ok();

    auto find_request = ApproximateNearestNeighborIndex::FindRequest
    {
        .index = 0,
        .neighbor_count = 12
    };

    auto find_required_memory = ann_index.find_required_memory(find_request);

    auto find_memory = MemoryOwner(find_required_memory.total());

    auto find_request_memory = RequestMemory
    {
        .result_memory = find_memory.to_memory_span(0, find_required_memory.result_required_bytes),
        .working_memory = find_memory.to_memory_span(find_required_memory.result_required_bytes),
    };

    size_t time = 0, trials = 5000;

    for (int trial = 0; trial < trials; ++trial)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (int row = 0; row < test_matrix.row_ct; ++row)
        {
            find_request.index = row;

            auto nearest_neighbors_result = ann_index.find_nearest_neighbors(find_request, find_request_memory);
        }

        auto end = std::chrono::high_resolution_clock::now();

        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    std::cout << "Average Find Time (ns): " << time / (trials * test_matrix.row_ct)  << "\n";
}
#endif //RUN_BENCHMARKS