//
// Created by nik on 3/24/2024.
//

#ifndef NML_QR_ALGORITHM_H
#define NML_QR_ALGORITHM_H

#include <cmath>
#include "../primitives/matrix_span.h"
#include "../primitives/matrix_owner.h"
#include "../primitives/memory_span.h"

namespace nml
{
    struct QRDecomposition
    {
        MatrixSpan q;
        MatrixSpan r;

        enum struct Type
        {
            GIVENS, HOUSEHOLDER
        };

        struct Request
        {
            const MatrixSpan& data;

            Type type;
            float zero_threshold;
            float error_threshold;
            unsigned max_iterations;

            explicit Request(const MatrixSpan& data, Type type, unsigned max_iterations = 1000, float error_threshold = 1e-3, float zero_threshold = 1e-5) noexcept
                : data(data), type(type), zero_threshold(zero_threshold), error_threshold(error_threshold), max_iterations(max_iterations)
            {}
        };

        static RequiredMemory qr_algorithm_required_memory(const Request& request) noexcept;
        static NMLResult<VectorSpan> qr_algorithm(const Request& request, RequestMemory& memory) noexcept;

        static inline NMLResult<bool> qr_full_initialize(const MatrixSpan& matrix, MatrixSpan& q, MatrixSpan& r) noexcept;
        static void qr_full_givens(MatrixSpan& q, MatrixSpan& r, float zero_threshold = 1e-5) noexcept;
        static void qr_full_householder(MatrixSpan& q, MatrixSpan& r, VectorSpan& householder_vector, float zero_threshold = 1e-5) noexcept;

        static NMLResult<QRDecomposition> qr_partial_householder(MatrixSpan& q, MatrixSpan& r, VectorSpan& householder_vector, VectorSpan& tau, bool zero_r = true, float zero_threshold = 1e-5) noexcept;
    };

    RequiredMemory QRDecomposition::qr_algorithm_required_memory(const Request &request) noexcept
    {
        unsigned row_size = request.data.bytes() / request.data.column_ct;

        return RequiredMemory
        {
            .result_required_bytes = row_size,
            .working_required_bytes = request.data.bytes() * 3
                                      + (request.type == Type::HOUSEHOLDER ? row_size : 0)
        };
    }

    NMLResult<VectorSpan> QRDecomposition::qr_algorithm(const QRDecomposition::Request& request, RequestMemory& memory) noexcept
    {
        if (!request.data.is_symmetric(request.zero_threshold))
        {
            return NMLResult<VectorSpan>::err(NMLErrorCode::NOT_SYMMETRIC);
        }

        if (!memory.is_sufficient(qr_algorithm_required_memory(request)))
        {
            return NMLResult<VectorSpan>::err(NMLErrorCode::OUT_OF_BOUNDS);
        }

        unsigned working_memory_offset = 0;

        MatrixSpan r = MatrixSpan(memory.working_memory.get_pointer<float>(working_memory_offset), request.data.row_ct, request.data.column_ct);

        working_memory_offset += r.bytes();

        MatrixSpan q = MatrixSpan(memory.working_memory.get_pointer<float>(working_memory_offset), request.data.row_ct, request.data.column_ct);

        working_memory_offset += q.bytes();

        MatrixSpan eigen_matrix = MatrixSpan(memory.working_memory.get_pointer<float>(working_memory_offset), request.data.row_ct, request.data.column_ct);

        working_memory_offset += eigen_matrix.bytes();

        q.set_identity();
        r.copy_from_unsafe(request.data);

        bool converged = false;

        if (request.type == QRDecomposition::Type::GIVENS)
        {
            for (unsigned iteration = 0; iteration < request.max_iterations; ++iteration)
            {
                qr_full_givens(q, r, request.zero_threshold);

                r.multiply(q, eigen_matrix);

                if (eigen_matrix.is_row_echelon(request.error_threshold))
                {
                    converged = true; break;
                }

                q.set_identity();
                r.copy_from_unsafe(eigen_matrix);
            }
        }
        else
        {
            VectorSpan householder_vector = VectorSpan(memory.working_memory.get_pointer<float>(working_memory_offset), r.row_ct);

            for (unsigned iteration = 0; iteration < request.max_iterations; ++iteration)
            {
                qr_full_householder(q, r, householder_vector, request.zero_threshold);

                r.multiply(q, eigen_matrix);

                if (eigen_matrix.is_row_echelon(request.error_threshold))
                {
                    converged = true; break;
                }

                q.set_identity();
                r.copy_from_unsafe(eigen_matrix);
            }
        }

        if (!converged)
        {
            return NMLResult<VectorSpan>::err(NMLErrorCode::UNABLE_TO_CONVERGE);
        }

        VectorSpan eigenvalues = VectorSpan(memory.result_memory.get_pointer<float>(0), r.row_ct);

        for (unsigned row = 0; row < eigen_matrix.row_ct; ++row)
        {
            eigenvalues[row] = eigen_matrix.get_unsafe(row, row);
        }

        eigenvalues.sort_descending();

        return NMLResult<VectorSpan>::ok(eigenvalues);
    }

    NMLResult<bool> QRDecomposition::qr_full_initialize(const MatrixSpan& matrix, MatrixSpan& q, MatrixSpan& r) noexcept
    {
        if (!matrix.has_same_dimensions_as(q) || !matrix.has_same_dimensions_as(r))
        {
            return NMLResult<bool>(NMLErrorCode::OUT_OF_BOUNDS);
        }

        q.set_identity();
        r.copy_from_unsafe(matrix);

        return NMLResult<bool>(true);
    }

    void QRDecomposition::qr_full_givens(MatrixSpan& q, MatrixSpan& r, float zero_threshold) noexcept
    {
        for (unsigned col = 0; col < r.column_ct; ++col)
        {
            for (unsigned row = col + 1; row < r.row_ct; ++row)
            {
                float rotation_cos, rotation_sin,
                        &diagonal_element = r.get_unsafe_ref(col, col),
                        &off_diagonal_element = r.get_unsafe_ref(row, col);

                if (std::abs(off_diagonal_element) < zero_threshold)
                {
                    rotation_cos = 1;
                    rotation_sin = 0;
                }
                else
                {
                    float distance = std::hypot(diagonal_element, off_diagonal_element);
                    rotation_cos = diagonal_element / distance;
                    rotation_sin = -off_diagonal_element / distance;
                }

                if (std::abs(rotation_sin) < zero_threshold) continue;

                for (unsigned k = 0; k < q.row_ct; ++k)
                {
                    float q_cos = rotation_cos * q.get_unsafe(k, col) - rotation_sin * q.get_unsafe(k, row);
                    float q_sin = rotation_sin * q.get_unsafe(k, col) + rotation_cos * q.get_unsafe(k, row);
                    q.set_unsafe(k, col, q_cos);
                    q.set_unsafe(k, row, q_sin);
                }

                diagonal_element = diagonal_element * rotation_cos - off_diagonal_element * rotation_sin;
                off_diagonal_element = 0;

                for (unsigned k = col + 1; k < r.column_ct; ++k)
                {
                    float r_cos = rotation_cos * r.get_unsafe(col, k) - rotation_sin * r.get_unsafe(row, k);
                    float r_sin = rotation_sin * r.get_unsafe(col, k) + rotation_cos * r.get_unsafe(row, k);
                    r.set_unsafe(row, k, r_sin);
                    r.set_unsafe(col, k, r_cos);
                }
            }
        }
    }

    void QRDecomposition::qr_full_householder(MatrixSpan& q, MatrixSpan& r, VectorSpan& householder_vector, float zero_threshold) noexcept
    {
        unsigned diagonals = std::min(r.row_ct, r.column_ct);

        for (unsigned diagonal = 0; diagonal < diagonals; ++diagonal)
        {
            if (diagonal == r.row_ct - 1 && diagonal == r.column_ct - 1)
            {
                continue;
            }

            float norm = 0;

            for (unsigned row = diagonal; row < r.row_ct; ++row)
            {
                norm += r.get_unsafe(row, diagonal) * r.get_unsafe(row, diagonal);
            }

            if (norm < zero_threshold) continue;

            norm = std::sqrt(norm);

            for (unsigned row = 0; row < diagonal; ++row) householder_vector[row] = 0;

            for (unsigned row = diagonal; row < r.row_ct; ++row)
            {
                householder_vector[row] = r.get_unsafe(row, diagonal);
            }

            householder_vector[diagonal] += (r.get_unsafe(diagonal, diagonal) >= 0) ? norm : -norm;

            float u_norm = 0;

            for (unsigned row = diagonal; row < r.row_ct; ++row)
            {
                u_norm += householder_vector[row] * householder_vector[row];
            }

            u_norm = std::sqrt(u_norm);

            for (unsigned row = diagonal; row < r.row_ct; ++row)
            {
                householder_vector[row] /= u_norm;
            }

            for (unsigned column = diagonal; column < r.column_ct; ++column)
            {
                float dot_product = 0;

                for (unsigned row = diagonal; row < r.row_ct; ++row)
                {
                    dot_product += householder_vector[row] * r.get_unsafe(row, column);
                }

                dot_product *= 2;

                for (unsigned row = diagonal; row < r.row_ct; ++row)
                {
                    r.get_unsafe_ref(row, column) -= householder_vector[row] * dot_product;
                }
            }

            for (unsigned row = 0; row < q.row_ct; ++row)
            {
                float dot_product = 0;

                for (unsigned column = diagonal; column < q.column_ct; ++column)
                {
                    dot_product += householder_vector[column] * q.get_unsafe(row, column);
                }

                dot_product *= 2;

                for (unsigned column = diagonal; column < q.column_ct; ++column)
                {
                    q.get_unsafe_ref(row, column) -= householder_vector[column] * dot_product;
                }
            }
        }
    }

    NMLResult<QRDecomposition> QRDecomposition::qr_partial_householder(MatrixSpan& q, MatrixSpan& r, VectorSpan& householder_vector, VectorSpan& tau, bool zero_r, float zero_threshold) noexcept
    {
        bool q_has_same_rows_as_r = r.row_ct == q.row_ct;
        bool is_tall_and_skinny = r.row_ct >= r.column_ct;
        bool tau_is_long_enough = tau.length >= q.column_ct && tau.length > 0;

        if (!q_has_same_rows_as_r || !tau_is_long_enough || !is_tall_and_skinny)
        {
            return NMLResult<QRDecomposition>::err(NMLErrorCode::INVALID_REQUEST);
        }

        unsigned diagonals = std::min(r.row_ct, r.column_ct);

        for (unsigned diagonal = 0; diagonal < diagonals; ++diagonal)
        {
            if (diagonal == r.row_ct - 1 && diagonal == r.column_ct - 1)
            {
                tau[diagonal] = 0; continue;
            }

            float norm = 0;

            for (unsigned row = diagonal; row < r.row_ct; ++row)
            {
                norm += r.get_unsafe(row, diagonal) * r.get_unsafe(row, diagonal);
            }

            if (norm < zero_threshold) continue;

            norm = std::sqrt(norm);

            for (unsigned row = 0; row < diagonal; ++row) householder_vector[row] = 0;

            for (unsigned row = diagonal; row < r.row_ct; ++row)
            {
                householder_vector[row] = r.get_unsafe(row, diagonal);
            }

            float diagonal_sign_flip = householder_vector[diagonal] >= 0 ? -1 : 1;

            float scale = householder_vector[diagonal] - diagonal_sign_flip * norm;

            tau[diagonal] = -diagonal_sign_flip * scale / norm;

            for (unsigned row = diagonal; row < r.row_ct; ++row)
            {
                householder_vector[row] /= scale;
            }

            householder_vector[diagonal] = 1;

            r.get_unsafe_ref(diagonal, diagonal) = diagonal_sign_flip * norm;

            for (unsigned row = diagonal + 1; row < r.row_ct; ++row)
            {
                r.get_unsafe_ref(row, diagonal) = householder_vector[row];
            }

            for (unsigned column = diagonal + 1; column < r.column_ct; ++column)
            {
                float dot_product = 0;

                for (unsigned row = diagonal; row < r.row_ct; ++row)
                {
                    dot_product += householder_vector[row] * r.get_unsafe(row, column);
                }

                dot_product *= tau[diagonal];

                for (unsigned row = diagonal; row < r.row_ct; ++row)
                {
                    r.get_unsafe_ref(row, column) -= dot_product * householder_vector[row];
                }
            }
        }

        for (unsigned diagonal = q.column_ct - 1; diagonal < q.column_ct; --diagonal)
        {
            householder_vector[diagonal] = 1;

            for (unsigned row = diagonal + 1; row < q.row_ct; ++row)
            {
                householder_vector[row] = r.get_unsafe(row, diagonal);
            }

            for (unsigned column = diagonal; column < q.column_ct; ++column)
            {
                float dot_product = 0;

                if (column == diagonal)
                {
                    dot_product = tau[diagonal];
                }
                else
                {
                    for (unsigned row = diagonal; row < q.row_ct; ++row)
                    {
                        dot_product += householder_vector[row] * q.get_unsafe(row, column);
                    }

                    dot_product *= tau[diagonal];
                }

                for (unsigned row = diagonal; row < q.row_ct; ++row)
                {
                    q.get_unsafe_ref(row, column) -= dot_product * householder_vector[row];
                }
            }
        }

        if (zero_r)
        {
            for (unsigned row = 0; row < r.column_ct; ++row)
            {
                unsigned max_column = std::min(row, r.column_ct);

                for (unsigned column = 0; column < max_column; ++column)
                {
                    r.set_unsafe(row, column, 0);
                }
            }
        }

        return NMLResult<QRDecomposition>::ok({
            .q = q,
            .r = r.to_subspan_unsafe(r.column_ct,  r.column_ct)
        });
    }

//    void householder_qr_blocked(MatrixSpan& r, VectorSpan& householder_vector, VectorSpan& tau, unsigned start_row, unsigned start_col, unsigned block_size, float zero_threshold) noexcept;
//
//    void blocked_qr(MatrixSpan& r, VectorSpan& householder_vector, VectorSpan& tau, unsigned block_size = 4)
//    {
//        for (unsigned i = 0; i < r.column_ct; i += block_size)
//        {
//            unsigned bs = std::min(block_size, r.column_ct - i);
//            householder_qr_blocked(r, householder_vector, tau, i, i, bs, 1e-6);
//        }
//    }

    void householder_qr_blocked(MatrixSpan& r, VectorSpan& householder_vector, VectorSpan& tau, float zero_threshold) noexcept
    {
        unsigned diagonals = std::min(r.row_ct, r.column_ct);

        for (unsigned diagonal = 0; diagonal < diagonals; ++diagonal)
        {
            float norm = 0;

            for (unsigned row = diagonal; row < r.row_ct; ++row)
            {
                norm += r.get_unsafe(row, diagonal) * r.get_unsafe(row, diagonal);
            }

            if (norm < zero_threshold) continue;

            norm = std::sqrt(norm);

            for (unsigned row = 0; row < diagonal; ++row) householder_vector[row] = 0;

            for (unsigned row = diagonal; row < r.row_ct; ++row)
            {
                householder_vector[row] = r.get_unsafe(row, diagonal);
            }

            householder_vector[diagonal] += (r.get_unsafe(diagonal, diagonal) > 0) ? norm : -norm;

            float u_norm = 0;

            for (unsigned row = diagonal; row < r.row_ct; ++row)
            {
                u_norm += householder_vector[row] * householder_vector[row];
            }

            if (u_norm < zero_threshold) continue;

            u_norm = std::sqrt(u_norm);

            tau[diagonal] = 2.0f / u_norm;

            for (unsigned row = diagonal; row < r.row_ct; ++row)
            {
                householder_vector[row] /= u_norm;
            }

            for (unsigned column = diagonal; column < r.column_ct; ++column)
            {
                float dot_product = 0;

                for (unsigned row = diagonal; row < r.row_ct; ++row)
                {
                    dot_product += householder_vector[row - diagonal] * r.get_unsafe(row, column);
                }

                dot_product *= tau[diagonal];

                for (unsigned row = diagonal; row < r.row_ct; ++row)
                {
                    r.get_unsafe_ref(row, column) -= householder_vector[row - diagonal] * dot_product;
                }
            }
        }
    }
}

#endif //NML_QR_ALGORITHM_H