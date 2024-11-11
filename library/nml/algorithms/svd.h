//
// Created by nik on 4/20/2024.
//

#ifndef NML_SVD_H
#define NML_SVD_H

#include "../primitives/span.h"
#include "../primitives/vector_span.h"
#include "../primitives/matrix_span.h"

#include "eigen_solver.h"
#include "qr_algorithm.h"
#include "lu_decomposition.h"
#include "randomized_low_rank_approximation.h"

const float TALL_SKINNY_RATIO = 10.0f;

namespace nml
{
    struct SVD
    {
        MatrixSpan left_singular_vectors;
        VectorSpan singular_values;
        MatrixSpan right_singular_vectors;

        enum class PreProcessOption : char
        {
            NONE, CENTER, STANDARDIZE
        };

        struct Request
        {
            const MatrixSpan& data;
            const unsigned n_components;

            float zero_threshold = 1e-5;
            PreProcessOption preprocess = PreProcessOption::NONE;

            explicit Request(const MatrixSpan& data, unsigned n_components = 0) noexcept
                : data(data), n_components(n_components == 0 ? data.column_ct : n_components)
            { }
        };

        static RequiredMemory required_memory(const Request &request) noexcept;
        static NMLResult<SVD> compute(const Request &request, RequestMemory memory) noexcept;
    };
}

namespace nml::svd_internal
{
    MatrixSpan get_pre_processed_matrix(const SVD::Request& request, MemorySpan& memory, unsigned& memory_offset) noexcept
    {
        if (request.preprocess == SVD::PreProcessOption::NONE)
        {
            return request.data.to_subspan_unsafe(request.data.row_ct, request.data.column_ct);
        }

        auto pre_processed_matrix_pointer = memory.get_pointer<float>(memory_offset);

        request.data.copy_into_unsafe(pre_processed_matrix_pointer);

        auto pre_processed_matrix = MatrixSpan(pre_processed_matrix_pointer, request.data.row_ct, request.data.column_ct);

        memory_offset += pre_processed_matrix.bytes();

        switch (request.preprocess)
        {
            case SVD::PreProcessOption::CENTER: pre_processed_matrix.center(); break;
            case SVD::PreProcessOption::STANDARDIZE: pre_processed_matrix.standardize(request.zero_threshold); break;
            default: break;
        }

        return pre_processed_matrix;
    }

    bool is_tall_and_skinny(const SVD::Request& request)
    {
        return static_cast<float>(request.data.column_ct) * TALL_SKINNY_RATIO < static_cast<float>(request.data.row_ct);
    }

    MatrixSpan get_working_matrix(const SVD::Request& request, MatrixSpan& pre_processed_matrix, MemorySpan& memory, unsigned& memory_offset) noexcept
    {
        if (!is_tall_and_skinny(request))
        {
            return pre_processed_matrix;
        }

        auto lr_request = RandomizedLowRankApproximation::Request(pre_processed_matrix);

        auto required_memory = RandomizedLowRankApproximation::required_memory(lr_request);

        auto low_rank_memory = RequestMemory::from_required_unsafe(required_memory, memory.offset(memory_offset));

        auto projection_result = RandomizedLowRankApproximation::compute(lr_request, low_rank_memory);

        auto working_matrix = projection_result.ok();

        memory_offset += working_matrix.bytes();

        return working_matrix;
    }

    MatrixSpan get_sum_of_squares_matrix(const MatrixSpan& working_matrix, MemorySpan& memory, unsigned& memory_offset) noexcept
    {
        auto sum_of_squares_matrix = MatrixSpan(memory.offset(memory_offset),
                                                working_matrix.column_ct, working_matrix.column_ct);

        memory_offset += sum_of_squares_matrix.bytes();
        working_matrix.transpose_multiply(working_matrix, sum_of_squares_matrix);

        return sum_of_squares_matrix;
    }

    Eigen get_eigen(const MatrixSpan& sum_of_squares_matrix, MemorySpan& memory, unsigned& memory_offset) noexcept
    {
        auto eigen_request = Eigen::Request(sum_of_squares_matrix);

        auto eigen_memory = RequestMemory::from_required_unsafe(
            Eigen::required_memory(eigen_request),
            memory.offset(memory_offset)
        );

        return Eigen::compute(eigen_request, eigen_memory).ok();
    }

    SVD initialize_svd(const SVD::Request& request, MemorySpan& memory, unsigned& memory_offset) noexcept
    {
        auto left_singular_vectors = MatrixSpan(memory.offset(memory_offset), request.data.row_ct, request.n_components);

        memory_offset += left_singular_vectors.bytes();

        auto singular_values = VectorSpan(memory.offset(memory_offset), request.n_components);

        memory_offset += singular_values.bytes();

        auto right_singular_vectors = MatrixSpan(memory.offset(memory_offset), request.n_components, request.data.column_ct);

        memory_offset += right_singular_vectors.bytes();

        return SVD
        {
            .left_singular_vectors = left_singular_vectors,
            .singular_values = singular_values,
            .right_singular_vectors = right_singular_vectors,
        };
    }
}

namespace nml
{
    using namespace nml::svd_internal;

    RequiredMemory SVD::required_memory(const Request& request) noexcept
    {
        unsigned singular_values_bytes = VectorSpan::required_bytes(request.n_components);
        unsigned left_singular_vectors_bytes = MatrixSpan::required_bytes(request.data.row_ct, request.n_components);
        unsigned right_singular_vectors_bytes = MatrixSpan::required_bytes(request.n_components, request.data.column_ct);

        unsigned preprocess_bytes = request.preprocess == PreProcessOption::NONE ? 0u : request.data.bytes();

        auto mock_lr_request = RandomizedLowRankApproximation::Request(request.data);
        unsigned working_matrix_bytes = !is_tall_and_skinny(request) ? 0u : MatrixSpan::required_bytes(request.data.column_ct, request.data.column_ct);
        unsigned working_matrix_working_bytes = RandomizedLowRankApproximation::required_memory(mock_lr_request).total_bytes();

        unsigned covariance_matrix_bytes = MatrixSpan::required_bytes(request.data.column_ct, request.data.column_ct);

        MatrixSpan mock_covariance_matrix = MatrixSpan(nullptr, request.data.column_ct, request.data.column_ct);
        unsigned eigen_bytes = Eigen::required_memory(Eigen::Request(mock_covariance_matrix)).total_bytes();

        return RequiredMemory
        {
            .result_required_bytes = singular_values_bytes + left_singular_vectors_bytes + right_singular_vectors_bytes,
            .working_required_bytes = preprocess_bytes + working_matrix_bytes + std::max(working_matrix_working_bytes, covariance_matrix_bytes + eigen_bytes)
        };
    }

    NMLResult<SVD> SVD::compute(const Request& request, RequestMemory memory) noexcept
    {
        if (!memory.is_sufficient(required_memory(request)))
        {
            return NMLResult<SVD>::err(NMLErrorCode::INSUFFICIENT_MEMORY);
        }

        unsigned working_memory_offset = 0, result_memory_offset = 0;

        MatrixSpan pre_processed_matrix = get_pre_processed_matrix(request, memory.working_memory, working_memory_offset);

        MatrixSpan working_matrix = get_working_matrix(request, pre_processed_matrix, memory.working_memory, working_memory_offset);

        MatrixSpan sum_of_squares_matrix = get_sum_of_squares_matrix(working_matrix, memory.working_memory,
                                                                         working_memory_offset);

        Eigen eigen = get_eigen(sum_of_squares_matrix, memory.working_memory, working_memory_offset);

        SVD result = initialize_svd(request, memory.result_memory, result_memory_offset);

        for (unsigned i = 0; i < request.n_components; ++i)
        {
            result.singular_values[i] = std::sqrt(eigen.eigenvalues[i]);
        }

        for (unsigned row = 0; row < request.n_components; ++row)
        {
            for (unsigned column = 0; column < eigen.eigenvectors.column_ct; ++column)
            {
                result.right_singular_vectors.set_unsafe(row, column, eigen.eigenvectors.get_unsafe(column, row));
            }
        }

        pre_processed_matrix.multiply(
            eigen.eigenvectors.to_subspan_unsafe(eigen.eigenvectors.row_ct, request.n_components),
            result.left_singular_vectors
        );

        for (unsigned row = 0; row < pre_processed_matrix.row_ct; ++row)
        {
            for (unsigned column = 0; column < request.n_components; ++column)
            {
                result.left_singular_vectors.get_unsafe_ref(row, column) /= result.singular_values[column];
            }
        }

        return NMLResult<SVD>::ok(result);
    }
}

#endif //NML_SVD_H