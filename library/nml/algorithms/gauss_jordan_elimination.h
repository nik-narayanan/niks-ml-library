//
// Created by nik on 3/24/2024.
//

#ifndef NML_GAUSS_JORDAN_ELIMINATION_H
#define NML_GAUSS_JORDAN_ELIMINATION_H

#include <cmath>
#include "../primitives/matrix_span.h"
#include "../primitives/span.h"

namespace nml
{
    static inline void gauss_jordan_elimination(MatrixSpan& matrix, const float zero_threshold = 1e-6) noexcept
    {
        for (unsigned diagonal = 0; diagonal < matrix.row_ct; ++diagonal)
        {
            unsigned max_row_index = diagonal;

            float max_value = matrix.get_unsafe(diagonal, diagonal);

            for (unsigned row = diagonal + 1; row < matrix.row_ct; ++row)
            {
                float value = matrix.get_unsafe(row, diagonal);

                if (std::abs(value) > std::abs(max_value))
                {
                    max_value = value;
                    max_row_index = row;
                }
            }

            if (max_row_index != diagonal)
            {
                matrix.swap_rows(diagonal, max_row_index);
            }

            float& diagonal_value = matrix.get_unsafe_ref(diagonal, diagonal);

            if (std::abs(diagonal_value) < zero_threshold)
            {
                diagonal_value = zero_threshold * (diagonal_value < 0 ? -1.0f : 1.0f);
            }

            for (unsigned column = diagonal + 1; column < matrix.column_ct; ++column)
            {
                matrix.get_unsafe_ref(diagonal, column) /= diagonal_value;
            }

            diagonal_value = 1;

            for (unsigned row = 0; row < matrix.row_ct; ++row)
            {
                if (row == diagonal) continue;

                float factor = matrix.get_unsafe(row, diagonal);

                for (unsigned column = diagonal; column < matrix.column_ct; ++column)
                {
                    float adjustment = factor * matrix.get_unsafe(diagonal, column);
                    float& value = matrix.get_unsafe_ref(row, column);
                    value -= adjustment;
                }
            }
        }
    }

    static inline void gauss_jordan_elimination(MatrixSpan& matrix, Span<unsigned>& pivots, const float zero_threshold = 1e-6) noexcept
    {
        if (pivots.length < matrix.row_ct) return;

        for (unsigned pivot = 0; pivot < matrix.row_ct; ++pivot)
        {
            pivots[pivot] = pivot;
        }

        for (unsigned diagonal = 0; diagonal < matrix.row_ct; ++diagonal)
        {
            unsigned max_row_index = diagonal;

            float max_value = std::fabs(matrix.get_unsafe(pivots[diagonal], diagonal));

            for (unsigned row = diagonal + 1; row < matrix.row_ct; ++row)
            {
                float value = std::fabs(matrix.get_unsafe(pivots[row], diagonal));

                if (value > max_value)
                {
                    max_value = value;
                    max_row_index = row;
                }
            }

            if (max_row_index != diagonal)
            {
                std::swap(pivots[diagonal], pivots[max_row_index]);
            }

            float& diagonal_value = matrix.get_unsafe_ref(pivots[diagonal], diagonal);

            if (std::abs(diagonal_value) < zero_threshold)
            {
                diagonal_value = zero_threshold * (diagonal_value < 0 ? -1.0f : 1.0f);
            }

            for (unsigned column = diagonal + 1; column < matrix.column_ct; ++column)
            {
                matrix.get_unsafe_ref(pivots[diagonal], column) /= diagonal_value;
            }

            diagonal_value = 1;

            for (unsigned row = 0; row < matrix.row_ct; ++row)
            {
                if (row == diagonal) continue;

                float factor = matrix.get_unsafe(pivots[row], diagonal);

                for (unsigned column = diagonal; column < matrix.column_ct; ++column)
                {
                    float adjustment = factor * matrix.get_unsafe(pivots[diagonal], column);
                    float& value = matrix.get_unsafe_ref(pivots[row], column);
                    value -= adjustment;
                }
            }
        }
    }
}

#endif //NML_GAUSS_JORDAN_ELIMINATION_H
