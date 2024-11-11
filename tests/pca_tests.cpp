//
// Created by nik on 4/14/2024.
//

#include <gtest/gtest.h>
#include "datasets/test_data.h"

#include "../library/nml/algorithms/pca.h"
#include "../library/nml/primitives/file.h"
#include "../library/nml/primitives/memory_owner.h"
#include "../library/nml/primitives/matrix_owner.h"

using namespace nml;

TEST(pca_tests, pca)
{
    MatrixOwner test_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_VAL)),
                expected_matrix = MatrixOwner::from_delimited(TestData::test_file_path(Dataset::IRIS_PCA));

    MatrixSpan test_span = test_matrix.to_span(), expected_span = expected_matrix.to_span();

    auto pca_request = PCA::Request(test_span, test_span.column_ct);
    auto required_memory = PCA::required_memory(pca_request);

    auto memory = MemoryOwner(required_memory.total_bytes());
    auto request_memory = memory.to_request_memory(required_memory);

    {
//        pca_request.preprocess = PCA::PreProcessOption::CENTER;
        auto pca_result = PCA::compute(pca_request, request_memory);
        ASSERT_TRUE(pca_result.is_ok());
        auto pca = pca_result.ok();
        ASSERT_TRUE(expected_span.equals(pca.projection, 1.5e-04));
//        pca.projection.print();
    }
}