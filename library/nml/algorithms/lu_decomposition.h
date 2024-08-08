//
// Created by nik on 3/28/2024.
//

#ifndef NML_LU_DECOMPOSITION_H
#define NML_LU_DECOMPOSITION_H

#include <cmath>
#include "../primitives/span.h"
#include "../primitives/matrix_span.h"

namespace nml
{
    struct LUDecomposition
    {
        MatrixSpan l;
        MatrixSpan u;

        struct Request
        {
            const MatrixSpan& data;
            const float zero_threshold;

            explicit Request(const MatrixSpan& data, float zero_threshold = 1e-6) noexcept
                : data(data), zero_threshold(zero_threshold)
            {}
        };

        static RequiredMemory required_memory(const Request& request) noexcept;
        static NMLResult<LUDecomposition> decompose(const Request& request, RequestMemory memory) noexcept;

        static void decompose_unsafe(const MatrixSpan& data, MatrixSpan& l, MatrixSpan& u, Span<unsigned>& pivots, float zero_threshold = 1e-6) noexcept;
    };

    RequiredMemory LUDecomposition::required_memory(const Request& request) noexcept
    {
        unsigned diagonals = std::min(request.data.row_ct, request.data.column_ct);

        unsigned float_size = sizeof(float), unsigned_size = sizeof(unsigned);

        return RequiredMemory
        {
            .result_required_bytes = float_size * request.data.row_ct * diagonals + float_size * diagonals * request.data.column_ct,
            .working_required_bytes = unsigned_size * request.data.row_ct
        };
    }

    NMLResult<LUDecomposition> LUDecomposition::decompose(const Request& request, RequestMemory memory) noexcept
    {
        if (!memory.is_sufficient(required_memory(request)))
        {
            return NMLResult<LUDecomposition>::err(NMLErrorCode::INSUFFICIENT_MEMORY);
        }

        unsigned diagonals = std::min(request.data.row_ct, request.data.column_ct);

        auto l = MatrixSpan(memory.result_memory, request.data.row_ct, diagonals);
        auto u = MatrixSpan(memory.result_memory.offset(l.bytes()), diagonals, request.data.column_ct);

        auto pivots = Span<unsigned>(memory.working_memory, request.data.row_ct);

        decompose_unsafe(request.data, l, u, pivots, request.zero_threshold);

        return NMLResult<LUDecomposition>({
           .l = l,
           .u = u
       });
    }

    void LUDecomposition::decompose_unsafe(const MatrixSpan& data, MatrixSpan& l, MatrixSpan& u, Span<unsigned>& pivots, float zero_threshold) noexcept
    {
        l.fill(0); u.fill(0);

        unsigned diagonals = std::min(data.row_ct, data.column_ct);

        for (unsigned pivot = 0; pivot < pivots.length; ++pivot)
        {
            pivots[pivot] = pivot;
        }

        for (unsigned diagonal = 0; diagonal < diagonals; ++diagonal)
        {
            float max_value = 0;
            unsigned max_row_index = diagonal;

            for (unsigned row = diagonal; row < data.row_ct; ++row)
            {
                float abs_value = std::fabs(data.get_unsafe(pivots[row], diagonal));

                if (abs_value > max_value)
                {
                    max_value = abs_value;
                    max_row_index = row;
                }
            }

            if (max_row_index != diagonal)
            {
                std::swap(pivots[diagonal], pivots[max_row_index]);
            }

            for (unsigned row = diagonal; row < data.column_ct; ++row)
            {
                float sum = 0;

                for (unsigned column = 0; column < diagonal; ++column)
                {
                    sum += l.get_unsafe(pivots[diagonal], column) * u.get_unsafe(column, row);
                }

                u.get_unsafe_ref(diagonal, row) = data.get_unsafe(pivots[diagonal], row) - sum;
            }

            for (unsigned row = diagonal; row < data.row_ct; ++row)
            {
                if (diagonal == row)
                {
                    l.set_unsafe(pivots[diagonal], diagonal, 1);
                }
                else
                {
                    float sum = 0;

                    for (unsigned column = 0; column < diagonal; ++column)
                    {
                        sum += l.get_unsafe(pivots[row], column) * u.get_unsafe(column, diagonal);
                    }

                    float numerator = data.get_unsafe(pivots[row], diagonal) - sum;
                    float denominator = u.get_unsafe(diagonal, diagonal);

                    if (std::fabs(denominator) < zero_threshold)
                    {
                        denominator = zero_threshold * (denominator < 0 ? -1.0f : 1.0f);
                    }

                    l.get_unsafe_ref(pivots[row], diagonal) = numerator / denominator;
                }
            }
        }
    }
}

#endif //NML_LU_DECOMPOSITION_H