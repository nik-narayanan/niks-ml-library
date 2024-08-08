//
// Created by nik on 3/27/2024.
//

#ifndef NML_EIGEN_SOLVER_H
#define NML_EIGEN_SOLVER_H

#include "qr_algorithm.h"
#include "gauss_jordan_elimination.h"

#include "../primitives/matrix_span.h"

namespace nml
{
    struct Eigen
    {
        VectorSpan eigenvalues;
        MatrixSpan eigenvectors;

        struct Request
        {
            const MatrixSpan& matrix;

            float zero_threshold = 1e-5;
            float error_threshold = 1e-3;
            unsigned max_iterations = 1000;
            QRDecomposition::Type type = QRDecomposition::Type::HOUSEHOLDER;

            explicit Request(const MatrixSpan& matrix) noexcept: matrix(matrix) { }
        };

        static RequiredMemory required_memory(const Request &request) noexcept;
        static NMLResult<Eigen> compute(const Request &request, RequestMemory &memory) noexcept;

        void print();

    private:
        static inline QRDecomposition::Request map_to_qr_request(const Request &request) noexcept;
    };

    RequiredMemory Eigen::required_memory(const Request &request) noexcept
    {
        auto qr_request = map_to_qr_request(request);
        auto qr_required = QRDecomposition::qr_algorithm_required_memory(qr_request);

        return RequiredMemory
        {
            .result_required_bytes =  qr_required.result_required_bytes + request.matrix.bytes(),
            .working_required_bytes = qr_required.working_required_bytes
        };
    }

    NMLResult<Eigen> Eigen::compute(const Request &request, RequestMemory &memory) noexcept
    {
        if (!request.matrix.is_square())
        {
            return NMLResult<Eigen>::err(NMLErrorCode::NOT_SQUARE);
        }

        if (!memory.is_sufficient(required_memory(request)))
        {
            return NMLResult<Eigen>::err(NMLErrorCode::INSUFFICIENT_MEMORY);
        }

        auto qr_request = map_to_qr_request(request);
        auto qr_result = QRDecomposition::qr_algorithm(qr_request, memory);

        if (qr_result.is_err())
        {
            return NMLResult<Eigen>::err(qr_result.err());
        }

        unsigned result_memory_offset = 0;

        auto eigenvalues = qr_result.ok();

        result_memory_offset += eigenvalues.bytes();

        auto eigenvectors = MatrixSpan(
            memory.result_memory.offset(result_memory_offset),
            request.matrix.row_ct, request.matrix.column_ct
        );

        auto working_matrix = MatrixSpan(
            memory.working_memory.offset(0),
            request.matrix.row_ct, request.matrix.row_ct + 1
        );

        auto pivots = Span<unsigned>(memory.working_memory.offset(working_matrix.bytes()), working_matrix.row_ct);

        for (unsigned eigenvalue_index = 0; eigenvalue_index < eigenvalues.length; ++eigenvalue_index)
        {
            float eigenvalue = eigenvalues[eigenvalue_index];

            for (unsigned row = 0; row < request.matrix.row_ct; ++row)
            {
                for (unsigned column = 0; column < request.matrix.row_ct; ++column)
                {
                    float copy_value = request.matrix.get_unsafe(row, column);
                    if (row == column) copy_value -= eigenvalue;
                    working_matrix.set_unsafe(row, column, copy_value);
                }

                working_matrix.set_unsafe(row, working_matrix.column_ct - 1, 1);
            }

            gauss_jordan_elimination(working_matrix, pivots, request.zero_threshold);

            auto vector_buffer = eigenvectors[eigenvalue_index];

            float max_abs_value = 0;

            for (unsigned row = 0; row < request.matrix.row_ct; ++row)
            {
                float value = working_matrix.get_unsafe(pivots[row], working_matrix.column_ct - 1);

                if (std::fabs(max_abs_value) < std::fabs(value))
                {
                    max_abs_value = value;
                }

                vector_buffer[row] = value;
            }

            vector_buffer.normalize(request.zero_threshold, std::fabs(max_abs_value) > max_abs_value);
        }

        eigenvectors.transpose_inplace();

        Eigen result =
        {
            .eigenvalues = eigenvalues,
            .eigenvectors = eigenvectors
        };

        return NMLResult<Eigen>::ok(result);
    }

    QRDecomposition::Request Eigen::map_to_qr_request(const Request &request) noexcept
    {
        return QRDecomposition::Request(request.matrix, request.type, request.max_iterations, request.error_threshold,
                                        request.zero_threshold);
    }

    void Eigen::print()
    {
        std::cout << "Eigenvalues: \n";
        eigenvalues.print();

        std::cout << "Eigenvectors: \n";
        eigenvectors.print();
    }
}

#endif //NML_EIGEN_SOLVER_H