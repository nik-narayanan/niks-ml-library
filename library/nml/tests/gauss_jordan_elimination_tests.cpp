//
// Created by nik on 5/24/2024.
//

#include <gtest/gtest.h>

#include "../primitives/file.h"
#include "../algorithms/eigen_solver.h"
#include "../primitives/memory_owner.h"

using namespace nml;

const float zero_threshold = 1e-06;

TEST(gauss_jordan_elimination_tests, swap)
{
    MatrixOwner test_matrix = MatrixOwner(4, 5, {
            -1.931374,  -0.118359,   0.877604,   0.823431,   1.000000,
            -0.118359,  -1.931374,  -0.431316,  -0.368583,   1.000000,
            0.877604,  -0.431316,  -1.931374,   0.969328,   1.000000,
            0.823431,  -0.368583,   0.969328,  -1.931374,   1.000000
    });

    MatrixSpan test_span = test_matrix.to_span();

    test_span.print();

    gauss_jordan_elimination(test_span);

    test_span.print();
}

TEST(gauss_jordan_elimination_tests, pivot_index)
{
    MatrixOwner test_matrix = MatrixOwner(4, 5, {
            -1.931374,  -0.118359,   0.877604,   0.823431,   1.000000,
            -0.118359,  -1.931374,  -0.431316,  -0.368583,   1.000000,
            0.877604,  -0.431316,  -1.931374,   0.969328,   1.000000,
            0.823431,  -0.368583,   0.969328,  -1.931374,   1.000000
    });

    MatrixSpan test_span = test_matrix.to_span();

    auto pm = reinterpret_cast<uint32_t*>(alloca(test_span.row_ct * sizeof(uint32_t)));

    auto p = Span<uint32_t>(pm, test_span.row_ct);

    std::cout << "\npivots\n";

    test_span.print();

    gauss_jordan_elimination(test_span, p);

    test_span.print();
}