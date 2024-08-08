//
// Created by nik on 6/29/2024.
//

#ifndef NML_TEST_DATA_H
#define NML_TEST_DATA_H

#include <string>
#include <stdexcept>

enum class Dataset : char
{
    IRIS_VAL,
    IRIS_PCA,
    IRIS_ANN,
    IRIS_STD,
    IRIS_COV,
    IRIS_LAB,
    IRIS_RAW,
    CDF_BETA,
    CDF_NORM,
};

struct TestData
{
    static std::string test_file_path(Dataset dataset);
};

std::string TestData::test_file_path(Dataset dataset)
{
    const std::string dataset_root = std::string(PROJECT_DIR) + R"(/library/nml/tests/datasets/)";

    switch (dataset)
    {
        case Dataset::IRIS_VAL: return dataset_root + R"(iris/iris_data.csv)";
        case Dataset::IRIS_PCA: return dataset_root + R"(iris/iris_pca.csv)";
        case Dataset::IRIS_ANN: return dataset_root + R"(iris/iris_ann.csv)";
        case Dataset::IRIS_STD: return dataset_root + R"(iris/iris_std.csv)";
        case Dataset::IRIS_COV: return dataset_root + R"(iris/iris_cov.csv)";
        case Dataset::IRIS_RAW: return dataset_root + R"(iris/iris_raw.csv)";
        case Dataset::IRIS_LAB: return dataset_root + R"(iris/iris_labeled.csv)";
        case Dataset::CDF_BETA: return dataset_root + R"(cdfs/beta_distribution_cdf.csv)";
        case Dataset::CDF_NORM: return dataset_root + R"(cdfs/normal_distribution_cdf.csv)";
        default: break;
    }

    throw std::invalid_argument("Unknown Dataset");
}

#endif //NML_TEST_DATA_H
