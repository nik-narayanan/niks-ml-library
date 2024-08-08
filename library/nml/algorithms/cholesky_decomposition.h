//
// Created by nik on 3/30/2024.
//

#ifndef NML_CHOLESKY_DECOMPOSITION_H
#define NML_CHOLESKY_DECOMPOSITION_H

#include "../primitives/matrix_span.h"

namespace nml
{
    static inline void cholesky_decomposition(const MatrixSpan& matrix, MatrixSpan& decomposition) noexcept
    {
        decomposition.fill(0);

        for (unsigned diagonal = 0; diagonal < matrix.row_ct; diagonal++)
        {
            for (unsigned row = 0; row <= diagonal; row++)
            {
                float sum = 0.0;

                if (row == diagonal)
                {
                    for (int column = 0; column < row; column++)
                    {
                        float value = decomposition.get_unsafe(row, column);
                        sum += value * value;
                    }

                    float value = std::sqrt(matrix.get_unsafe(row, row) - sum);
                    decomposition.set_unsafe(diagonal, diagonal, value);
                }
                else
                {
                    for (int k = 0; k < row; k++)
                    {
                        float left = decomposition.get_unsafe(diagonal, k);
                        float right = decomposition.get_unsafe(row, k);
                        sum += left * right;
                    }

                    float diagonal_value = decomposition.get_unsafe(row, row);
                    float value = (matrix.get_unsafe(diagonal, row) - sum);
                    decomposition.set_unsafe(diagonal, row, value / diagonal_value);
                }
            }
        }
    }


    //static inline void inplace_cholesky_decomposition(MatrixSpan& data) noexcept
    //{
    //    for (unsigned diagonal = 0; diagonal < data.row_ct; diagonal++)
    //    {
    //        for (unsigned row = 0; row <= diagonal; row++)
    //        {
    //            float sum = 0.0;

    //            if (row == diagonal)
    //            {
    //                for (int column = 0; column < row; column++)
    //                {
    //                    float value = data.get_unsafe(row, column);
    //                    sum += value * value;
    //                }

    //                float value = std::sqrt(data.get_unsafe(row, row) - sum);
    //                data.set_unsafe(diagonal, diagonal, value);
    //            }
    //            else
    //            {
    //                for (int k = 0; k < row; k++)
    //                {
    //                    float left = data.get_unsafe(diagonal, k);
    //                    float right = data.get_unsafe(row, k);
    //                    sum += left * right;
    //                }

    //                float diagonal_value = data.get_unsafe(row, row);
    //                float value = (data.get_unsafe(diagonal, row) - sum);

    //                if (diagonal_value < 1e-05)
    //                {
    //                    auto a = 1;
    //                }

    //                data.set_unsafe(diagonal, row, value / diagonal_value);
    //            }
    //        }
    //    }

    //    if (data.column_ct != data.row_ct + 1) return;

    //    float forward[data.row_ct], backward[data.row_ct];

    //    for (unsigned row = 0; row < data.row_ct; ++row)
    //    {
    //        forward[row] = 0; backward[row] = 0;
    //    }

    //    for (unsigned row = 0; row < data.row_ct; row++)
    //    {
    //        float sum = 0.0;

    //        for (unsigned column = 0; column < row; column++)
    //        {
    //            sum += data.get_unsafe(row, column) * forward[column];
    //        }

    //        forward[row] = (data.get_unsafe(row, data.column_ct - 1) - sum) / data.get_unsafe(row, row);
    //    }

    //    for (int column = data.row_ct - 1; column >= 0; --column)
    //    {
    //        float sum = 0.0;

    //        for (unsigned row = column + 1; row < data.row_ct; row++)
    //        {
    //            sum += data.get_unsafe(row, column) * backward[row];
    //        }

    //        backward[column] = (forward[column] - sum) / data.get_unsafe(column, column);
    //    }

    //    for (unsigned row = 0; row < data.row_ct; row++)
    //    {
    //        data.set_unsafe(row, data.column_ct - 1, backward[row]);
    //    }
    //}
}

#endif //NML_CHOLESKY_DECOMPOSITION_H
