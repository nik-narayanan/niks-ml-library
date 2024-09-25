//
// Created by nik on 7/1/2024.
//

#include "../tests/datasets/test_data.h"
#include "../primitives/data_frame.h"
#include "../algorithms/logistic_regression.h"

using namespace nml;

int main()
{
    auto df = DataFrame::from_delimited(TestData::test_file_path(Dataset::IRIS_RAW));

    df.print(5);

    auto dependent = df.get_column_by_name_unsafe("variety");

    auto distinct = ResizableList<int64_t>();

    df.factorize_column_with_distinct(dependent, "variety_int", &distinct, 0);

    df.print(5);

    df.remove_column("variety");

    MatrixOwner matrix = df.to_matrix();

    auto labels = VectorSpan(distinct.to_memory(), distinct.count);

    for (uint32_t i = 0; i < distinct.count; ++i)
    {
        labels[i] = static_cast<float>(distinct[i]);
    }

    auto request = LogisticRegressionMulti::Request(matrix.to_span(), labels);

    auto required_memory = LogisticRegressionMulti::required_memory(request);

    auto memory = MemoryOwner(required_memory.total_bytes());

    auto result = LogisticRegressionMulti::compute(request, memory.to_request_memory(required_memory));

    if (result.is_err())
    {
        std::cout << "Failed to converge";
        return (int)result.err();
    }

    auto regression = result.ok();

    regression.summary(matrix.to_span()).print(); // TODO

    return 0;
}