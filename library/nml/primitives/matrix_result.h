//
// Created by nik on 3/24/2024.
//

#ifndef NML_MATRIX_RESULT_H
#define NML_MATRIX_RESULT_H

#include "result.h"

namespace nml
{
    enum class NMLErrorCode : char
    {
        NOT_FOUND,
        NOT_SQUARE,
        NOT_SYMMETRIC,
        OUT_OF_BOUNDS,
        ROUNDING_ERROR,
        INVALID_REQUEST,
        UNABLE_TO_CONVERGE,
        MAXIMUM_ITERATIONS,
        INSUFFICIENT_MEMORY,
    };

    template<typename TData>
    using NMLResult = Result<TData, NMLErrorCode>;
}
#endif //NML_MATRIX_RESULT_H
