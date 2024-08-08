//
// Created by nik on 3/29/2024.
//

#ifndef NML_PCA_H
#define NML_PCA_H

#include "eigen_solver.h"
#include "../primitives/vector_span.h"
#include "../primitives/matrix_span.h"

namespace nml
{
    struct PCA
    {
        MatrixSpan projection;
        VectorSpan explained_variance;

        enum class PreProcessOption : char
        {
            NONE, CENTER, STANDARDIZE
        };

        struct Request
        {
            const MatrixSpan& matrix;
            const unsigned n_components;

            float zero_threshold = 1e-5;
            float error_threshold = 1e-3;
            unsigned max_iterations = 1000;
            QRDecomposition::Type type = QRDecomposition::Type::GIVENS;
            PreProcessOption preprocess = PreProcessOption::STANDARDIZE;

            explicit Request(const MatrixSpan& matrix, unsigned n_components) noexcept
                : matrix(matrix), n_components(n_components)
            { }
        };

        static RequiredMemory required_memory(const Request& request) noexcept;
        static NMLResult<PCA> compute(const Request& request, RequestMemory& memory) noexcept;

    private:
        static inline Eigen::Request map_to_eigen_request(const Request& request, const MatrixSpan& covariance_matrix) noexcept;

        static inline MatrixSpan get_working_matrix(const Request& request, RequestMemory& memory, unsigned& working_memory_offset) noexcept;
        static inline MatrixSpan get_covariance_matrix(const MatrixSpan& working_matrix, RequestMemory& memory, unsigned& working_memory_offset) noexcept;
        static inline Eigen get_eigen(const Request& request, const MatrixSpan& covariance_matrix, RequestMemory& memory, unsigned& working_memory_offset) noexcept;
        static inline MatrixSpan get_pca_projection(const Request& request, const MatrixSpan& working_matrix, const Eigen& eigen, RequestMemory& memory, unsigned& result_memory_offset) noexcept;
        static inline VectorSpan get_explained_variance(const Eigen& eigen, RequestMemory& memory, unsigned& result_memory_offset) noexcept;
    };

    RequiredMemory PCA::required_memory(const Request& request) noexcept
    {
        unsigned element_size = sizeof(float);

        auto working_matrix = request.preprocess == PreProcessOption::NONE ? 0 : request.matrix.bytes();
        auto mock_covariance_matrix = request.matrix.to_subspan_unsafe(request.matrix.column_ct, request.matrix.column_ct);

        auto eigen_request = map_to_eigen_request(request, mock_covariance_matrix);
        auto eigen_requirement = Eigen::required_memory(eigen_request);

        return RequiredMemory
        {
            .result_required_bytes = (request.matrix.row_ct * request.n_components + request.matrix.column_ct) * element_size,
            .working_required_bytes = working_matrix + mock_covariance_matrix.bytes() + eigen_requirement.total_bytes()
        };
    }

    NMLResult<PCA> PCA::compute(const Request& request, RequestMemory& memory) noexcept
    {
        if (!memory.is_sufficient(required_memory(request)))
        {
            return NMLResult<PCA>::err(NMLErrorCode::INSUFFICIENT_MEMORY);
        }

        unsigned working_memory_offset = 0, result_memory_offset = 0;

        MatrixSpan working_matrix = get_working_matrix(request, memory, working_memory_offset);

        MatrixSpan covariance_matrix = get_covariance_matrix(working_matrix, memory, working_memory_offset);

        Eigen eigen = get_eigen(request, covariance_matrix, memory, working_memory_offset);

        MatrixSpan projection = get_pca_projection(request, working_matrix, eigen, memory, result_memory_offset);

        VectorSpan explained_variance = get_explained_variance(eigen, memory, result_memory_offset);

        return NMLResult<PCA>::ok({
            .projection = projection,
            .explained_variance = explained_variance
        });
    }

    MatrixSpan PCA::get_working_matrix(const Request& request, RequestMemory& memory, unsigned& working_memory_offset) noexcept
    {
        if (request.preprocess == PreProcessOption::NONE)
        {
            return request.matrix.to_subspan_unsafe(0, 0);
        }

        auto working_matrix_pointer = memory.working_memory.get_pointer<float>(working_memory_offset);

        request.matrix.copy_into_unsafe(working_matrix_pointer);

        auto working_matrix = MatrixSpan(working_matrix_pointer, request.matrix.row_ct, request.matrix.column_ct);

        working_memory_offset += working_matrix.bytes();

        switch (request.preprocess)
        {
            case PreProcessOption::CENTER: working_matrix.center(); break;
            case PreProcessOption::STANDARDIZE: working_matrix.standardize(request.zero_threshold); break;
            default: break;
        }

        return working_matrix;
    }

    Eigen::Request PCA::map_to_eigen_request(const Request& request, const MatrixSpan& covariance_matrix) noexcept
    {
        auto eigen_request = Eigen::Request(covariance_matrix);

        eigen_request.type = request.type;
        eigen_request.max_iterations = request.max_iterations;
        eigen_request.error_threshold = request.error_threshold;
        eigen_request.zero_threshold = request.zero_threshold;

        return eigen_request;
    }

    MatrixSpan PCA::get_covariance_matrix(const MatrixSpan& working_matrix, RequestMemory& memory, unsigned& working_memory_offset) noexcept
    {
        auto covariance_matrix = MatrixSpan(memory.working_memory.get_pointer<float>(working_memory_offset),
                                            working_matrix.column_ct, working_matrix.column_ct);

        working_memory_offset += covariance_matrix.bytes();
        working_matrix.covariance(covariance_matrix);

        return covariance_matrix;
    }

    Eigen PCA::get_eigen(const Request& request, const MatrixSpan& covariance_matrix, RequestMemory& memory, unsigned& working_memory_offset) noexcept
    {
        auto eigen_request = map_to_eigen_request(request, covariance_matrix);

        auto eigen_memory = RequestMemory::from_required_unsafe(
            Eigen::required_memory(eigen_request),
            memory.working_memory,
            working_memory_offset
        );

        return Eigen::compute(eigen_request, eigen_memory).ok();
    }

    MatrixSpan PCA::get_pca_projection(const Request& request, const MatrixSpan& working_matrix, const Eigen& eigen, RequestMemory& memory, unsigned& result_memory_offset) noexcept
    {
        auto projection = MatrixSpan(memory.result_memory.get_pointer<float>(result_memory_offset), request.matrix.row_ct, request.n_components);

        result_memory_offset += projection.bytes();

        working_matrix.multiply(eigen.eigenvectors, projection);

        return projection;
    }

    VectorSpan PCA::get_explained_variance(const Eigen& eigen, RequestMemory& memory, unsigned& result_memory_offset) noexcept
    {
        auto explained_variance = VectorSpan(memory.result_memory.get_pointer<float>(result_memory_offset), eigen.eigenvalues.length);

        float eigen_total = 0;

        for (unsigned index = 0; index < eigen.eigenvalues.length; ++index)
        {
            eigen_total += eigen.eigenvalues[index];
        }

        for (unsigned index = 0; index < eigen.eigenvalues.length; ++index)
        {
            explained_variance[index] = eigen.eigenvalues[index] / eigen_total;
        }

        return explained_variance;
    }
}

#endif //NML_PCA_H