//
// Created by nik on 3/24/2024.
//

#include <gtest/gtest.h>

#include "../library/nml/primitives/file.h"
#include "../library/nml/algorithms/qr_algorithm.h"
#include "../library/nml/primitives/memory_owner.h"

using namespace nml;

const float zero_threshold = 1e-06;

TEST(qr_algorithm_tests, qr_decomposition_small)
{
    unsigned dimension = 4;

    MatrixOwner
        q = MatrixOwner(dimension, dimension),
        r = MatrixOwner(dimension, dimension),
        reconstruct = MatrixOwner(dimension, dimension),
        test_matrix = MatrixOwner(dimension, dimension, {
            2,    1,  7,   -8,
            1,    3,  3, -1.5,
            7,    3,  3,    3,
            -8, -1.5,  3,   51
        })
    ;

    MatrixSpan
        q_span = q.to_span(),
        r_span = r.to_span(),
        test_span = test_matrix.to_span(),
        reconstruct_span = reconstruct.to_span()
    ;

    {
        auto init_result = QRDecomposition::qr_full_initialize(test_span, q_span, r_span);
        ASSERT_TRUE(init_result.is_ok());
        QRDecomposition::qr_full_givens(q_span, r_span, zero_threshold);
        q_span.multiply(r_span, reconstruct_span);
        ASSERT_TRUE(reconstruct_span.equals(test_span, 1e-04));
    }

    {
        auto init_result = QRDecomposition::qr_full_initialize(test_span, q_span, r_span);
        ASSERT_TRUE(init_result.is_ok());
        auto householder_vector = reconstruct_span.to_vector_subspan_unsafe(dimension,0,0);
        QRDecomposition::qr_full_householder(q_span, r_span, householder_vector, zero_threshold);
        q_span.multiply(r_span, reconstruct_span);
        ASSERT_TRUE(reconstruct_span.equals(test_span, 1e-04));
    }

    {
        auto init_result = QRDecomposition::qr_full_initialize(test_span, q_span, r_span);
        ASSERT_TRUE(init_result.is_ok());

        auto tao_vector = reconstruct_span.to_vector_subspan_unsafe(dimension,0,0);
        auto householder_vector = reconstruct_span.to_vector_subspan_unsafe(dimension,1,0);

        QRDecomposition::qr_partial_householder(q_span, r_span, householder_vector, tao_vector, true, zero_threshold);
        q_span.multiply(r_span, reconstruct_span);
        ASSERT_TRUE(reconstruct_span.equals(test_span, 1e-04));
    }
}

TEST(qr_algorithm_tests, qr_decomposition_large)
{
    unsigned dimension = 60;

    MatrixOwner
        q = MatrixOwner(dimension, dimension),
        r = MatrixOwner(dimension, dimension),
        reconstruct = MatrixOwner(dimension, dimension),
        test_matrix = MatrixOwner(dimension, dimension, { 2 ,7 ,-1 ,-2 ,6 ,0 ,-5 ,0 ,-7 ,-6 ,3 ,1 ,8 ,-8 ,9 ,-8 ,9 ,-7 ,-2 ,-2 ,-4 ,6 ,-8 ,3 ,-1 ,-2 ,-6 ,1 ,-8 ,9 ,7 ,-3 ,2 ,8 ,-5 ,-3 ,1 ,-8 ,9 ,4 ,-6 ,2 ,5 ,-1 ,-5 ,1 ,6 ,6 ,-8 ,-6 ,-1 ,-8 ,9 ,-2 ,7 ,-3 ,-6 ,-9 ,7 ,-6,8 ,-6 ,4 ,-3 ,-1 ,7 ,-6 ,-9 ,6 ,9 ,8 ,-9 ,7 ,-5 ,2 ,7 ,6 ,2 ,2 ,6 ,-3 ,-3 ,-8 ,6 ,-6 ,-8 ,6 ,8 ,1 ,-2 ,-4 ,-8 ,6 ,-8 ,3 ,-8 ,3 ,7 ,-9 ,-1 ,8 ,6 ,-7 ,-9 ,1 ,7 ,7 ,2 ,9 ,-4 ,7 ,-3 ,9 ,3 ,-4 ,2 ,3 ,-7 ,-4 ,0,2 ,5 ,5 ,1 ,7 ,6 ,9 ,7 ,6 ,-5 ,6 ,0 ,5 ,7 ,3 ,9 ,4 ,2 ,-9 ,-9 ,0 ,6 ,-3 ,3 ,9 ,5 ,-9 ,-4 ,-7 ,0 ,4 ,7 ,-1 ,-5 ,-5 ,0 ,-6 ,-7 ,0 ,6 ,-3 ,-9 ,-1 ,5 ,5 ,8 ,7 ,4 ,8 ,0 ,-1 ,-7 ,3 ,0 ,-3 ,-8 ,-1 ,-8 ,5 ,1,3 ,-1 ,7 ,8 ,3 ,7 ,-3 ,-9 ,2 ,-6 ,6 ,-8 ,-1 ,2 ,2 ,3 ,-2 ,6 ,-4 ,-7 ,-3 ,-2 ,0 ,-2 ,-9 ,-5 ,-2 ,-9 ,-9 ,-6 ,-8 ,8 ,8 ,6 ,2 ,-3 ,-5 ,-2 ,0 ,5 ,8 ,-1 ,-6 ,-7 ,8 ,-4 ,-1 ,6 ,-9 ,8 ,6 ,-8 ,-5 ,-1 ,0 ,-2 ,9 ,-7 ,-1 ,6,8 ,4 ,-3 ,4 ,1 ,-8 ,-8 ,3 ,-9 ,9 ,-5 ,4 ,-1 ,4 ,0 ,-7 ,3 ,7 ,7 ,1 ,4 ,-5 ,-7 ,-1 ,8 ,-3 ,-4 ,-6 ,0 ,3 ,8 ,-3 ,-2 ,3 ,-5 ,6 ,2 ,-5 ,-4 ,6 ,0 ,-7 ,2 ,9 ,8 ,-4 ,9 ,0 ,-7 ,0 ,4 ,2 ,-5 ,1 ,9 ,-3 ,6 ,1 ,0 ,7,-3 ,0 ,3 ,5 ,8 ,7 ,3 ,-2 ,-1 ,6 ,-4 ,-4 ,4 ,-8 ,-9 ,-4 ,5 ,-9 ,-1 ,5 ,0 ,1 ,3 ,2 ,2 ,-5 ,6 ,6 ,-4 ,3 ,-9 ,-6 ,-1 ,8 ,-9 ,8 ,-7 ,3 ,0 ,4 ,1 ,-2 ,-4 ,7 ,-7 ,0 ,2 ,2 ,-4 ,9 ,3 ,5 ,-5 ,6 ,-9 ,5 ,-4 ,-5 ,-1 ,9,-2 ,-5 ,6 ,6 ,2 ,0 ,-1 ,7 ,-4 ,-7 ,-9 ,-8 ,3 ,6 ,8 ,-8 ,0 ,-9 ,-9 ,-5 ,-3 ,9 ,-8 ,-3 ,1 ,-4 ,-1 ,5 ,4 ,-3 ,3 ,-9 ,0 ,-4 ,-7 ,-5 ,-5 ,-4 ,3 ,-6 ,-5 ,3 ,1 ,-3 ,-7 ,8 ,2 ,5 ,2 ,-2 ,-2 ,4 ,1 ,-2 ,3 ,8 ,-4 ,5 ,0 ,1,-2 ,-2 ,-7 ,4 ,5 ,8 ,8 ,-4 ,-2 ,-5 ,5 ,-4 ,-4 ,-9 ,-9 ,8 ,8 ,7 ,-5 ,-2 ,-7 ,5 ,-9 ,9 ,-7 ,8 ,-7 ,4 ,2 ,1 ,-7 ,-8 ,7 ,-5 ,1 ,-8 ,8 ,-8 ,4 ,-4 ,-9 ,7 ,3 ,-8 ,-7 ,8 ,2 ,-1 ,5 ,-1 ,-4 ,2 ,-7 ,-2 ,8 ,-3 ,9 ,7 ,-4 ,7,-7 ,2 ,0 ,9 ,6 ,9 ,-1 ,4 ,4 ,8 ,6 ,7 ,-4 ,8 ,7 ,7 ,2 ,5 ,-6 ,4 ,5 ,-9 ,8 ,-6 ,3 ,-4 ,-1 ,8 ,8 ,4 ,4 ,8 ,8 ,3 ,2 ,6 ,3 ,-8 ,-5 ,-9 ,-3 ,-5 ,6 ,4 ,-1 ,-7 ,3 ,-9 ,3 ,-9 ,-2 ,2 ,-7 ,1 ,5 ,3 ,0 ,5 ,-1 ,-6,1 ,4 ,-4 ,2 ,-1 ,-7 ,-3 ,8 ,9 ,-6 ,6 ,2 ,2 ,3 ,-4 ,-2 ,-7 ,7 ,-4 ,-5 ,9 ,3 ,-1 ,-5 ,-5 ,-6 ,3 ,-4 ,4 ,-3 ,-1 ,9 ,0 ,3 ,8 ,5 ,0 ,6 ,-6 ,4 ,9 ,-1 ,9 ,6 ,-1 ,-1 ,8 ,6 ,0 ,-2 ,6 ,6 ,-3 ,-8 ,-4 ,-9 ,1 ,2 ,-7 ,1,1 ,2 ,9 ,2 ,0 ,-4 ,-8 ,9 ,5 ,-2 ,4 ,2 ,-6 ,6 ,6 ,-6 ,8 ,2 ,-3 ,4 ,-3 ,-4 ,-6 ,3 ,-2 ,9 ,4 ,-3 ,2 ,6 ,-8 ,-8 ,-2 ,8 ,-1 ,7 ,-5 ,7 ,0 ,-3 ,2 ,-8 ,-7 ,-3 ,-7 ,-6 ,-8 ,-7 ,4 ,-4 ,9 ,5 ,-6 ,-3 ,-9 ,-8 ,1 ,9 ,1 ,-3,-9 ,6 ,8 ,-5 ,7 ,-4 ,8 ,-8 ,7 ,-9 ,-5 ,1 ,8 ,-7 ,9 ,-2 ,4 ,5 ,-3 ,-4 ,-6 ,0 ,-5 ,1 ,6 ,0 ,-8 ,5 ,5 ,4 ,-7 ,7 ,5 ,-9 ,-4 ,-1 ,2 ,0 ,8 ,-6 ,-7 ,-4 ,9 ,-2 ,4 ,-8 ,-1 ,1 ,0 ,-9 ,2 ,-2 ,-3 ,0 ,-1 ,-9 ,-1 ,7 ,-4 ,1,-1 ,1 ,-8 ,3 ,-9 ,8 ,-7 ,2 ,-6 ,-5 ,-3 ,-1 ,-9 ,-4 ,0 ,-5 ,5 ,-8 ,3 ,-3 ,-7 ,9 ,9 ,3 ,1 ,8 ,-6 ,-2 ,-9 ,-3 ,4 ,9 ,9 ,-7 ,-5 ,-6 ,-1 ,0 ,-9 ,-6 ,6 ,8 ,4 ,-6 ,7 ,-5 ,-5 ,-4 ,-8 ,1 ,6 ,5 ,-1 ,0 ,-4 ,1 ,9 ,1 ,-7 ,5,-2 ,-6 ,4 ,8 ,-7 ,5 ,-1 ,-8 ,4 ,-4 ,1 ,7 ,-4 ,7 ,-1 ,4 ,-7 ,-1 ,4 ,-9 ,-6 ,-5 ,5 ,0 ,-3 ,5 ,4 ,6 ,2 ,-2 ,-1 ,-2 ,4 ,-1 ,-8 ,-2 ,-4 ,8 ,-4 ,-6 ,4 ,6 ,2 ,-9 ,-1 ,5 ,6 ,-4 ,-1 ,-8 ,7 ,-6 ,3 ,4 ,8 ,-1 ,0 ,-8 ,3 ,4,-4 ,-9 ,8 ,-3 ,1 ,-5 ,-2 ,-6 ,4 ,-4 ,-7 ,-4 ,-8 ,8 ,4 ,1 ,-3 ,-2 ,7 ,8 ,-1 ,-1 ,3 ,-1 ,0 ,8 ,-9 ,9 ,4 ,-2 ,7 ,7 ,-6 ,2 ,1 ,-2 ,-9 ,2 ,-6 ,-9 ,-8 ,-8 ,-4 ,-3 ,3 ,9 ,-9 ,8 ,5 ,8 ,-4 ,6 ,-1 ,5 ,-7 ,-2 ,-8 ,4 ,5 ,-8,3 ,-7 ,-6 ,9 ,4 ,-4 ,-5 ,7 ,-1 ,8 ,7 ,3 ,8 ,-4 ,-1 ,-2 ,9 ,-7 ,-2 ,9 ,-4 ,2 ,-7 ,6 ,6 ,-1 ,2 ,7 ,-5 ,-1 ,-9 ,9 ,9 ,8 ,5 ,6 ,-5 ,-7 ,4 ,-1 ,-2 ,-1 ,6 ,4 ,-8 ,-4 ,-6 ,3 ,6 ,8 ,3 ,-1 ,-5 ,9 ,8 ,4 ,-9 ,-9 ,6 ,8,-3 ,-3 ,4 ,5 ,-2 ,2 ,5 ,9 ,4 ,6 ,0 ,4 ,0 ,7 ,8 ,7 ,-1 ,-9 ,-2 ,-1 ,-9 ,-6 ,8 ,0 ,-4 ,1 ,0 ,5 ,-9 ,-3 ,-9 ,8 ,0 ,-7 ,3 ,4 ,-7 ,7 ,6 ,-2 ,-8 ,-1 ,-2 ,3 ,3 ,3 ,0 ,1 ,-8 ,-7 ,8 ,-7 ,-1 ,-9 ,1 ,4 ,1 ,-3 ,-2 ,-9,-3 ,0 ,-9 ,5 ,8 ,9 ,9 ,-6 ,-4 ,8 ,-9 ,-7 ,-7 ,-7 ,6 ,-3 ,7 ,0 ,2 ,-8 ,-5 ,-3 ,-8 ,2 ,-8 ,9 ,-6 ,-9 ,4 ,-6 ,6 ,-7 ,8 ,-7 ,-8 ,9 ,0 ,-2 ,6 ,1 ,-5 ,-4 ,0 ,9 ,1 ,-8 ,5 ,8 ,-4 ,0 ,7 ,5 ,0 ,4 ,6 ,8 ,6 ,-3 ,-6 ,-3,-7 ,9 ,3 ,5 ,-2 ,-2 ,8 ,0 ,-8 ,9 ,7 ,-2 ,1 ,-2 ,1 ,2 ,-6 ,8 ,-1 ,5 ,7 ,0 ,-8 ,4 ,5 ,-4 ,1 ,-1 ,-7 ,-6 ,7 ,-2 ,-7 ,-3 ,5 ,-4 ,-2 ,-8 ,3 ,7 ,-8 ,9 ,-6 ,-7 ,-5 ,0 ,-7 ,5 ,-8 ,7 ,2 ,5 ,-3 ,7 ,-9 ,8 ,6 ,-8 ,-8 ,5,6 ,4 ,1 ,6 ,-3 ,-8 ,0 ,-8 ,6 ,-9 ,-3 ,-9 ,-8 ,7 ,0 ,8 ,8 ,-6 ,5 ,-6 ,-6 ,-7 ,-7 ,-8 ,5 ,-3 ,-4 ,4 ,2 ,-3 ,-3 ,3 ,-9 ,1 ,-7 ,6 ,2 ,0 ,-5 ,8 ,-7 ,-9 ,-6 ,-8 ,0 ,-9 ,8 ,-4 ,4 ,-2 ,-8 ,9 ,-7 ,-5 ,-6 ,7 ,-3 ,-6 ,-9 ,8,-2 ,-3 ,8 ,1 ,-9 ,-4 ,-5 ,-7 ,8 ,-2 ,-7 ,-8 ,-9 ,-4 ,-4 ,0 ,-2 ,7 ,-6 ,4 ,-2 ,0 ,0 ,-7 ,5 ,-2 ,4 ,-6 ,-3 ,-2 ,5 ,-1 ,1 ,-2 ,-9 ,-1 ,5 ,-8 ,-2 ,8 ,1 ,8 ,-1 ,-3 ,-6 ,2 ,-7 ,-7 ,0 ,-6 ,5 ,4 ,-1 ,4 ,4 ,5 ,-5 ,5 ,-4 ,-2,6 ,0 ,3 ,-9 ,5 ,-1 ,-6 ,6 ,8 ,9 ,4 ,0 ,4 ,2 ,7 ,3 ,5 ,6 ,4 ,-3 ,8 ,2 ,-6 ,-5 ,-6 ,6 ,6 ,0 ,-3 ,-3 ,3 ,9 ,2 ,-7 ,8 ,6 ,-1 ,9 ,-9 ,6 ,-2 ,8 ,9 ,4 ,5 ,9 ,-2 ,-8 ,1 ,-9 ,-4 ,-5 ,-7 ,-7 ,-4 ,6 ,1 ,-6 ,-8 ,2,1 ,2 ,-9 ,8 ,7 ,-2 ,6 ,9 ,-5 ,-1 ,-9 ,5 ,1 ,1 ,8 ,9 ,8 ,1 ,-8 ,2 ,8 ,-2 ,8 ,-7 ,-4 ,-7 ,-7 ,2 ,-3 ,-6 ,3 ,0 ,-3 ,1 ,-8 ,-6 ,-9 ,4 ,0 ,-4 ,-3 ,6 ,5 ,-5 ,-4 ,-8 ,0 ,-8 ,9 ,4 ,1 ,-6 ,3 ,5 ,7 ,-1 ,-3 ,8 ,0 ,8,2 ,8 ,-2 ,3 ,4 ,1 ,-4 ,8 ,5 ,4 ,6 ,-6 ,1 ,-6 ,-6 ,-9 ,1 ,-6 ,3 ,-6 ,4 ,9 ,2 ,7 ,-7 ,-4 ,4 ,-6 ,-6 ,9 ,-9 ,-3 ,-6 ,6 ,4 ,-1 ,-3 ,-9 ,-4 ,-1 ,-4 ,-8 ,-4 ,-2 ,7 ,0 ,4 ,8 ,6 ,5 ,-5 ,6 ,6 ,5 ,5 ,7 ,-9 ,2 ,-9 ,8,6 ,3 ,-9 ,4 ,-3 ,-4 ,-2 ,1 ,8 ,-1 ,-2 ,2 ,5 ,4 ,1 ,-1 ,-9 ,5 ,-3 ,9 ,5 ,3 ,1 ,4 ,-4 ,-4 ,4 ,-6 ,-8 ,2 ,9 ,9 ,4 ,3 ,1 ,-3 ,-4 ,4 ,2 ,1 ,7 ,8 ,-5 ,4 ,9 ,8 ,1 ,-6 ,-2 ,-7 ,-8 ,-5 ,-1 ,5 ,-4 ,7 ,0 ,-3 ,2 ,-3,6 ,-7 ,-1 ,1 ,-2 ,2 ,-7 ,-3 ,6 ,4 ,-8 ,9 ,-5 ,0 ,-3 ,9 ,-5 ,-6 ,-3 ,-4 ,-5 ,-6 ,9 ,-4 ,-2 ,-5 ,-8 ,2 ,4 ,1 ,-4 ,-6 ,1 ,-3 ,-2 ,-4 ,0 ,5 ,7 ,-3 ,-6 ,2 ,1 ,3 ,7 ,-8 ,0 ,-4 ,7 ,-2 ,2 ,-6 ,-7 ,-5 ,-6 ,0 ,6 ,-1 ,-4 ,9,5 ,-2 ,1 ,8 ,-2 ,0 ,-6 ,-1 ,-4 ,2 ,-6 ,-2 ,-3 ,1 ,-9 ,3 ,-8 ,9 ,0 ,5 ,-3 ,4 ,4 ,-6 ,-9 ,-4 ,7 ,0 ,6 ,-8 ,-9 ,4 ,-4 ,1 ,2 ,-2 ,-8 ,-2 ,-3 ,1 ,3 ,2 ,4 ,-7 ,4 ,4 ,8 ,6 ,8 ,-2 ,1 ,-3 ,-6 ,8 ,2 ,-7 ,9 ,-5 ,8 ,6,-3 ,-1 ,0 ,0 ,-5 ,5 ,-4 ,-8 ,-8 ,-9 ,5 ,-4 ,5 ,-4 ,9 ,-4 ,-7 ,7 ,6 ,1 ,2 ,6 ,6 ,4 ,-2 ,0 ,0 ,2 ,3 ,0 ,1 ,3 ,-2 ,-4 ,-8 ,1 ,-4 ,7 ,-4 ,-8 ,-8 ,-5 ,-5 ,-8 ,-4 ,3 ,-3 ,-6 ,9 ,-1 ,6 ,-3 ,-1 ,-8 ,2 ,2 ,0 ,7 ,9 ,-4,1 ,-9 ,-7 ,-3 ,2 ,5 ,2 ,7 ,4 ,4 ,7 ,0 ,6 ,7 ,9 ,0 ,6 ,-2 ,8 ,4 ,6 ,2 ,-3 ,-9 ,-2 ,-4 ,1 ,6 ,7 ,-4 ,5 ,7 ,-7 ,-3 ,3 ,3 ,-8 ,7 ,-1 ,4 ,-7 ,5 ,-5 ,-1 ,-3 ,-9 ,5 ,-1 ,9 ,-1 ,-1 ,0 ,-7 ,-3 ,-7 ,8 ,-8 ,-4 ,0 ,-9,9 ,0 ,8 ,4 ,8 ,8 ,7 ,3 ,-4 ,-5 ,6 ,-1 ,8 ,7 ,2 ,-1 ,2 ,-7 ,4 ,1 ,-2 ,9 ,-8 ,1 ,1 ,1 ,-3 ,1 ,-3 ,1 ,7 ,3 ,-5 ,3 ,-7 ,-5 ,6 ,1 ,6 ,5 ,1 ,6 ,7 ,-7 ,5 ,-3 ,-6 ,3 ,-6 ,4 ,4 ,1 ,2 ,-6 ,1 ,3 ,-9 ,4 ,6 ,6,7 ,-4 ,-5 ,-2 ,6 ,3 ,-3 ,-2 ,-6 ,0 ,5 ,-4 ,-2 ,-6 ,4 ,6 ,-5 ,-6 ,8 ,-5 ,-3 ,-7 ,-8 ,-9 ,0 ,-1 ,-3 ,2 ,8 ,9 ,7 ,-5 ,-3 ,4 ,-9 ,-8 ,4 ,6 ,-3 ,1 ,0 ,9 ,-9 ,6 ,-1 ,-8 ,-5 ,1 ,8 ,5 ,-2 ,-1 ,8 ,0 ,-2 ,3 ,7 ,-9 ,-2 ,7,7 ,-4 ,2 ,-9 ,-9 ,-6 ,-3 ,2 ,-7 ,-8 ,-5 ,3 ,-6 ,-7 ,-7 ,5 ,-6 ,0 ,-7 ,-6 ,6 ,-4 ,-1 ,-5 ,-4 ,-9 ,-8 ,3 ,7 ,-6 ,-8 ,0 ,-4 ,-5 ,5 ,-8 ,9 ,7 ,0 ,5 ,4 ,5 ,8 ,5 ,3 ,-4 ,-2 ,-4 ,9 ,-1 ,4 ,6 ,-1 ,-9 ,6 ,8 ,4 ,-2 ,-2 ,-3,7 ,7 ,7 ,-9 ,-1 ,7 ,4 ,3 ,7 ,-6 ,6 ,-4 ,-3 ,-3 ,-9 ,2 ,4 ,5 ,9 ,-7 ,0 ,-2 ,2 ,2 ,1 ,-8 ,-7 ,0 ,-3 ,-6 ,-4 ,-5 ,-4 ,-1 ,-3 ,-4 ,0 ,-2 ,2 ,4 ,6 ,9 ,9 ,-2 ,-9 ,-3 ,-9 ,5 ,-7 ,-4 ,-1 ,-8 ,-9 ,-3 ,-9 ,-9 ,0 ,2 ,-7 ,-3,2 ,7 ,8 ,3 ,8 ,-3 ,8 ,-6 ,-9 ,4 ,-9 ,8 ,0 ,8 ,-8 ,1 ,3 ,5 ,0 ,4 ,9 ,-1 ,5 ,3 ,5 ,-7 ,2 ,-7 ,-9 ,7 ,5 ,8 ,-4 ,-1 ,-9 ,7 ,-3 ,5 ,-4 ,-2 ,3 ,8 ,5 ,-5 ,-6 ,6 ,1 ,-5 ,9 ,6 ,1 ,8 ,-4 ,-2 ,-7 ,9 ,-8 ,-7 ,8 ,3,1 ,9 ,3 ,9 ,-5 ,5 ,-9 ,-5 ,5 ,-9 ,9 ,-6 ,-9 ,7 ,1 ,-4 ,2 ,8 ,6 ,-4 ,-5 ,-1 ,-6 ,5 ,-9 ,5 ,3 ,9 ,-4 ,-6 ,5 ,7 ,-1 ,4 ,8 ,3 ,-8 ,-3 ,1 ,3 ,-1 ,-2 ,-2 ,3 ,9 ,9 ,9 ,8 ,6 ,-5 ,9 ,3 ,3 ,-1 ,-3 ,0 ,5 ,7 ,8 ,-6,3 ,-7 ,-8 ,-5 ,-4 ,5 ,5 ,-1 ,9 ,5 ,-8 ,8 ,2 ,8 ,-5 ,-3 ,-6 ,8 ,-6 ,-6 ,2 ,5 ,9 ,-3 ,2 ,3 ,-5 ,2 ,6 ,1 ,-7 ,-5 ,-5 ,-1 ,-8 ,8 ,-8 ,-9 ,-3 ,2 ,-3 ,-5 ,1 ,3 ,6 ,2 ,9 ,-8 ,-8 ,3 ,7 ,5 ,2 ,-4 ,-9 ,0 ,6 ,9 ,-9 ,-9,5 ,7 ,-2 ,5 ,5 ,-7 ,2 ,-6 ,-1 ,7 ,7 ,5 ,4 ,1 ,0 ,-5 ,-2 ,9 ,0 ,5 ,-7 ,5 ,-8 ,3 ,0 ,4 ,9 ,2 ,-3 ,-9 ,2 ,-6 ,-3 ,4 ,2 ,9 ,1 ,5 ,4 ,-9 ,-5 ,3 ,-1 ,2 ,-4 ,9 ,-6 ,1 ,8 ,-4 ,-4 ,0 ,-8 ,4 ,5 ,4 ,-6 ,-9 ,7 ,2,-9 ,-8 ,-5 ,4 ,-8 ,-8 ,8 ,6 ,6 ,5 ,2 ,-3 ,2 ,7 ,2 ,2 ,5 ,5 ,-6 ,-8 ,3 ,5 ,-6 ,-5 ,6 ,6 ,2 ,-1 ,2 ,1 ,-6 ,3 ,-9 ,6 ,-9 ,9 ,7 ,-2 ,0 ,-2 ,8 ,-2 ,6 ,-1 ,-6 ,6 ,1 ,2 ,-6 ,4 ,-8 ,-7 ,-3 ,9 ,-3 ,-2 ,1 ,7 ,2 ,4,2 ,0 ,7 ,4 ,6 ,-9 ,1 ,5 ,8 ,-4 ,1 ,-8 ,6 ,-1 ,0 ,6 ,-8 ,-5 ,7 ,9 ,2 ,-5 ,0 ,-5 ,-4 ,4 ,8 ,5 ,3 ,-9 ,-5 ,-2 ,3 ,-5 ,8 ,0 ,6 ,4 ,-7 ,2 ,-3 ,-4 ,-6 ,9 ,1 ,-6 ,9 ,4 ,1 ,-9 ,7 ,9 ,5 ,-8 ,-2 ,-1 ,3 ,-9 ,-5 ,-2,0 ,4 ,2 ,9 ,-4 ,3 ,-2 ,9 ,-1 ,4 ,3 ,6 ,1 ,-3 ,4 ,-7 ,9 ,2 ,1 ,-5 ,-9 ,2 ,-5 ,-3 ,-3 ,-1 ,-3 ,-5 ,-6 ,8 ,-7 ,1 ,-9 ,-7 ,-9 ,0 ,9 ,-9 ,-2 ,1 ,5 ,-5 ,0 ,8 ,-8 ,6 ,-4 ,-1 ,5 ,0 ,1 ,-5 ,-9 ,-8 ,-7 ,6 ,6 ,1 ,7 ,-9,-2 ,6 ,0 ,2 ,2 ,-4 ,1 ,2 ,8 ,-8 ,-1 ,-1 ,-5 ,-2 ,5 ,9 ,-7 ,7 ,-8 ,-8 ,6 ,-4 ,-7 ,-9 ,5 ,-8 ,-9 ,-2 ,7 ,-4 ,2 ,9 ,-8 ,3 ,6 ,5 ,-4 ,-2 ,-4 ,-8 ,8 ,7 ,9 ,-7 ,8 ,-3 ,4 ,8 ,-6 ,-9 ,-9 ,5 ,4 ,9 ,-7 ,-3 ,5 ,1 ,-7 ,-5,-9 ,2 ,6 ,3 ,6 ,-9 ,-8 ,-2 ,0 ,3 ,-7 ,-9 ,-6 ,1 ,-6 ,2 ,9 ,-1 ,-7 ,2 ,4 ,6 ,9 ,-9 ,1 ,2 ,1 ,4 ,-6 ,3 ,-2 ,-8 ,-9 ,-6 ,-6 ,-2 ,-5 ,3 ,-1 ,7 ,-9 ,4 ,-6 ,0 ,-8 ,-6 ,7 ,1 ,7 ,-4 ,-4 ,7 ,9 ,-2 ,-1 ,-9 ,9 ,7 ,7 ,-5,-3 ,-4 ,-9 ,4 ,-7 ,-9 ,5 ,7 ,-2 ,-8 ,-5 ,-3 ,9 ,-8 ,-7 ,-4 ,-8 ,5 ,4 ,2 ,-1 ,-6 ,-9 ,-4 ,-7 ,0 ,-3 ,9 ,-3 ,-4 ,-5 ,-9 ,-8 ,1 ,-5 ,0 ,-9 ,-8 ,0 ,9 ,2 ,-6 ,-5 ,6 ,4 ,-2 ,-1 ,-6 ,9 ,4 ,2 ,-6 ,4 ,-1 ,-6 ,2 ,9 ,-5 ,-7 ,-3,4 ,-1 ,-9 ,5 ,7 ,3 ,-4 ,-4 ,6 ,9 ,-1 ,-4 ,-1 ,-7 ,3 ,2 ,4 ,-8 ,-2 ,6 ,-5 ,2 ,-6 ,8 ,-2 ,-1 ,-8 ,2 ,3 ,-7 ,-7 ,4 ,-5 ,-6 ,6 ,-9 ,-8 ,2 ,-7 ,-3 ,-8 ,-3 ,5 ,-2 ,8 ,-6 ,-4 ,-3 ,-7 ,-9 ,-7 ,-6 ,-2 ,8 ,-3 ,6 ,-8 ,3 ,-3 ,2,6 ,5 ,4 ,5 ,4 ,-2 ,-8 ,7 ,-5 ,0 ,4 ,-2 ,9 ,-4 ,9 ,-2 ,-3 ,-2 ,-3 ,5 ,1 ,8 ,0 ,-5 ,-7 ,3 ,-3 ,7 ,-5 ,-4 ,-8 ,7 ,2 ,1 ,-2 ,-1 ,-6 ,2 ,-2 ,5 ,-4 ,5 ,1 ,-3 ,-8 ,8 ,-4 ,2 ,-9 ,3 ,-2 ,8 ,-7 ,-9 ,-7 ,-4 ,2 ,8 ,1 ,8,1 ,5 ,-2 ,-7 ,4 ,4 ,0 ,6 ,-3 ,-3 ,1 ,3 ,-9 ,7 ,-6 ,7 ,2 ,1 ,6 ,-4 ,4 ,-2 ,-3 ,-5 ,4 ,-1 ,4 ,4 ,-2 ,-5 ,-3 ,-4 ,-8 ,4 ,-9 ,8 ,1 ,-4 ,-2 ,-6 ,-3 ,1 ,0 ,-3 ,6 ,-3 ,-5 ,-1 ,3 ,-3 ,5 ,-2 ,-1 ,1 ,6 ,4 ,2 ,-5 ,-6 ,-7,-9 ,-4 ,0 ,-1 ,4 ,3 ,-2 ,-5 ,-6 ,-7 ,-7 ,2 ,6 ,-5 ,-4 ,-3 ,-4 ,-5 ,1 ,2 ,-2 ,8 ,2 ,4 ,-2 ,3 ,9 ,5 ,-1 ,3 ,-5 ,9 ,-5 ,-7 ,-8 ,0 ,2 ,-6 ,-4 ,1 ,6 ,-7 ,8 ,1 ,2 ,6 ,1 ,-8 ,2 ,-6 ,-9 ,-3 ,6 ,2 ,-4 ,2 ,-8 ,7 ,7 ,-5,5 ,2 ,0 ,-5 ,-9 ,8 ,-9 ,9 ,0 ,1 ,-7 ,-6 ,-9 ,1 ,6 ,5 ,-1 ,3 ,3 ,8 ,9 ,-2 ,5 ,6 ,-1 ,-3 ,-9 ,1 ,3 ,-3 ,-5 ,-5 ,1 ,3 ,6 ,8 ,-8 ,4 ,2 ,-7 ,5 ,-9 ,-3 ,-9 ,-2 ,-8 ,-5 ,3 ,9 ,6 ,7 ,-6 ,9 ,5 ,-9 ,8 ,7 ,-4 ,-3 ,0,4 ,4 ,3 ,9 ,4 ,1 ,-7 ,1 ,9 ,0 ,1 ,-1 ,-9 ,-2 ,2 ,6 ,0 ,-2 ,5 ,6 ,2 ,6 ,0 ,-1 ,7 ,-6 ,-2 ,-8 ,-9 ,-6 ,-3 ,-7 ,-8 ,7 ,-6 ,-5 ,8 ,-5 ,-1 ,-5 ,9 ,-4 ,-6 ,3 ,-3 ,4 ,6 ,4 ,1 ,-2 ,0 ,1 ,7 ,-3 ,-3 ,-3 ,6 ,2 ,-6 ,1,3 ,-3 ,-5 ,-6 ,0 ,4 ,9 ,-3 ,0 ,1 ,5 ,-3 ,-2 ,1 ,-5 ,2 ,-5 ,2 ,2 ,1 ,-7 ,2 ,-3 ,-1 ,0 ,0 ,-8 ,-7 ,-8 ,3 ,5 ,-9 ,6 ,7 ,6 ,-9 ,-4 ,-5 ,-9 ,2 ,0 ,-4 ,-2 ,-3 ,-3 ,4 ,9 ,9 ,-5 ,-1 ,-5 ,-6 ,-1 ,-7 ,0 ,4 ,4 ,-6 ,3 ,2,0 ,-3 ,7 ,5 ,8 ,1 ,-9 ,-6 ,8 ,6 ,0 ,9 ,0 ,1 ,7 ,0 ,4 ,4 ,3 ,-4 ,2 ,0 ,9 ,2 ,6 ,-1 ,-7 ,-8 ,1 ,5 ,-9 ,0 ,8 ,9 ,2 ,-4 ,-3 ,2 ,-9 ,6 ,2 ,4 ,-5 ,-5 ,-8 ,-1 ,6 ,1 ,1 ,6 ,8 ,-8 ,1 ,-2 ,9 ,-4 ,-6 ,4 ,-5 ,5,-4 ,-8 ,7 ,-4 ,-9 ,-8 ,-4 ,-4 ,-8 ,-7 ,1 ,5 ,8 ,3 ,7 ,8 ,7 ,8 ,5 ,-3 ,-9 ,-2 ,-7 ,-8 ,0 ,8 ,0 ,-5 ,-4 ,1 ,-9 ,-4 ,1 ,4 ,-6 ,-3 ,-6 ,-8 ,7 ,8 ,-6 ,-2 ,0 ,-1 ,8 ,-4 ,7 ,8 ,3 ,1 ,-6 ,-1 ,-2 ,-8 ,7 ,-6 ,-7 ,4 ,1 ,-6,7 ,4 ,-6 ,0 ,-2 ,-6 ,3 ,-2 ,0 ,6 ,2 ,8 ,6 ,9 ,-7 ,-3 ,3 ,-6 ,-4 ,-9 ,2 ,-4 ,0 ,-4 ,-6 ,1 ,-4 ,-8 ,3 ,0 ,-9 ,-2 ,7 ,9 ,-6 ,-9 ,-4 ,7 ,-3 ,-1 ,-5 ,1 ,2 ,-8 ,6 ,1 ,-9 ,-1 ,-2 ,-6 ,-5 ,4 ,4 ,0 ,2 ,-5 ,3 ,-7 ,-5 ,5,0 ,-6 ,4 ,6 ,-8 ,7 ,9 ,-2 ,5 ,0 ,9 ,-6 ,-9 ,4 ,5 ,8 ,-2 ,6 ,2 ,-6 ,0 ,0 ,4 ,1 ,4 ,5 ,6 ,-6 ,9 ,-9 ,-4 ,-2 ,-5 ,8 ,-9 ,-8 ,4 ,-4 ,9 ,2 ,-1 ,1 ,4 ,4 ,0 ,8 ,5 ,8 ,1 ,-3 ,5 ,0 ,6 ,6 ,2 ,-6 ,-9 ,-5 ,4 ,-9,-5 ,9 ,-8 ,4 ,4 ,9 ,-5 ,3 ,3 ,7 ,-9 ,1 ,1 ,3 ,-1 ,-5 ,-1 ,1 ,-8 ,-9 ,3 ,6 ,-4 ,0 ,8 ,0 ,-5 ,7 ,-5 ,6 ,2 ,-4 ,-1 ,8 ,-6 ,6 ,-2 ,3 ,4 ,-9 ,-9 ,6 ,-2 ,-8 ,-6 ,8 ,-7 ,0 ,6 ,-8 ,0 ,1 ,1 ,3 ,-1 ,0 ,-3 ,-6 ,8 ,9,-5 ,-9 ,-7 ,0 ,-8 ,-5 ,-8 ,8 ,3 ,-1 ,-1 ,5 ,4 ,-4 ,-7 ,-2 ,7 ,5 ,0 ,-9 ,-3 ,5 ,3 ,-5 ,-8 ,-1 ,7 ,4 ,0 ,3 ,-9 ,0 ,4 ,-6 ,7 ,6 ,8 ,-7 ,7 ,-1 ,9 ,-7 ,-6 ,-8 ,8 ,0 ,-4 ,-9 ,-6 ,1 ,-4 ,7 ,-4 ,4 ,2 ,-6 ,4 ,-7 ,-1 ,-7,5 ,-8 ,-5 ,-3 ,-2 ,-2 ,-3 ,-9 ,6 ,-9 ,-8 ,-9 ,-3 ,-3 ,7 ,0 ,6 ,-9 ,1 ,9 ,6 ,3 ,4 ,1 ,9 ,-5 ,0 ,-4 ,-4 ,8 ,-6 ,2 ,-9 ,1 ,-3 ,-4 ,-9 ,5 ,2 ,9 ,8 ,4 ,3 ,6 ,6 ,8 ,4 ,-3 ,-2 ,-8 ,6 ,-4 ,3 ,6 ,-1 ,-1 ,-2 ,1 ,1 ,-2,5 ,-7 ,9 ,5 ,4 ,-3 ,4 ,5 ,4 ,9 ,1 ,1 ,4 ,3 ,1 ,-4 ,-9 ,9 ,-5 ,2 ,-9 ,-9 ,8 ,8 ,8 ,-4 ,6 ,-3 ,-1 ,-3 ,5 ,3 ,5 ,1 ,7 ,9 ,8 ,1 ,7 ,1 ,-3 ,4 ,-1 ,7 ,-3 ,9 ,1 ,-4 ,1 ,6 ,-1 ,-1 ,-6 ,-2 ,7 ,7 ,7 ,-2 ,5 ,1,3 ,-7 ,-1 ,-6 ,-4 ,9 ,4 ,6 ,-7 ,2 ,-9 ,7 ,8 ,-4 ,-5 ,-9 ,9 ,-5 ,-3 ,-2 ,-3 ,5 ,-9 ,2 ,6 ,-5 ,9 ,5 ,4 ,-4 ,-2 ,4 ,5 ,4 ,-9 ,-2 ,9 ,-5 ,5 ,8 ,8 ,-9 ,6 ,3 ,-7 ,-9 ,9 ,-3 ,-1 ,-8 ,-2 ,0 ,4 ,-1 ,7 ,-7 ,-5 ,-7 ,-8 ,2,0 ,7 ,8 ,3 ,-5 ,1 ,5 ,-3 ,5 ,5 ,-3 ,1 ,9 ,-1 ,-3 ,-4 ,8 ,-4 ,-5 ,-1 ,-1 ,9 ,-9 ,-6 ,-3 ,5 ,-4 ,-4 ,8 ,7 ,5 ,5 ,6 ,7 ,6 ,6 ,0 ,-7 ,5 ,-7 ,4 ,9 ,6 ,4 ,1 ,-4 ,3 ,-8 ,-7 ,-2 ,4 ,9 ,-9 ,5 ,7 ,-4 ,-9 ,4 ,-7 ,-2 })
    ;

    MatrixSpan
        q_span = q.to_span(),
        r_span = r.to_span(),
        test_span = test_matrix.to_span(),
        reconstruct_span = reconstruct.to_span()
    ;

    {
        auto init_result = QRDecomposition::qr_full_initialize(test_span, q_span, r_span);
        ASSERT_TRUE(init_result.is_ok());
        QRDecomposition::qr_full_givens(q_span, r_span, zero_threshold);
        q_span.multiply(r_span, reconstruct_span);
        ASSERT_TRUE(reconstruct_span.equals(test_span, 1e-04));
    }

    {
        auto init_result = QRDecomposition::qr_full_initialize(test_span, q_span, r_span);
        ASSERT_TRUE(init_result.is_ok());
        auto householder_vector = reconstruct_span.to_vector_subspan_unsafe(dimension,0,0);
        QRDecomposition::qr_full_householder(q_span, r_span, householder_vector, zero_threshold);
        q_span.multiply(r_span, reconstruct_span);
        ASSERT_TRUE(reconstruct_span.equals(test_span, 1e-04));
    }

    {
        auto init_result = QRDecomposition::qr_full_initialize(test_span, q_span, r_span);
        ASSERT_TRUE(init_result.is_ok());

        auto tao_vector = reconstruct_span.to_vector_subspan_unsafe(dimension,0,0);
        auto householder_vector = reconstruct_span.to_vector_subspan_unsafe(dimension,1,0);

        QRDecomposition::qr_partial_householder(q_span, r_span, householder_vector, tao_vector, true, zero_threshold);
        q_span.multiply(r_span, reconstruct_span);
        ASSERT_TRUE(reconstruct_span.equals(test_span, 1e-04));
    }
}


TEST(qr_algorithm_tests, qr_algorithm)
{
    MatrixOwner test_matrix = MatrixOwner(5, 4, {
         2,    1,  7,   -8,
         1,    3,  3, -1.5,
         7,    3,  3,    3,
        -8, -1.5,  3,   51,
        52.3917, 10.3737, 2.11125, -5.87669
    });

    MatrixSpan test_span = test_matrix.to_span_unsafe(4, 4, 0, 0);
    VectorSpan expected_vector = test_matrix.to_vector_span_unsafe(4, 4, 0);

    auto qr_request = QRDecomposition::Request(test_span, QRDecomposition::Type::HOUSEHOLDER);

    auto required_memory = QRDecomposition::qr_algorithm_required_memory(qr_request);
    auto memory = MemoryOwner(required_memory.total_bytes());
    auto request_memory = memory.to_request_memory(required_memory);

    {
        qr_request.type = QRDecomposition::Type::GIVENS;
        auto qr_result = QRDecomposition::qr_algorithm(qr_request, request_memory);
        ASSERT_TRUE(qr_result.is_ok());
        auto qr = qr_result.ok();
        ASSERT_TRUE(qr.equals(expected_vector, 1e-03));
    }

    {
        qr_request.type = QRDecomposition::Type::HOUSEHOLDER;
        auto qr_result = QRDecomposition::qr_algorithm(qr_request, request_memory);
        ASSERT_TRUE(qr_result.is_ok());
        auto qr = qr_result.ok();
        ASSERT_TRUE(qr.equals(expected_vector, 1e-03));
    }
}


TEST(qr_algorithm_tests, qr_parial_non_square)
{
    unsigned rows = 5, columns = 2;

    MatrixOwner
        q = MatrixOwner(rows, columns),
        r = MatrixOwner(rows, columns),
        reconstruct = MatrixOwner(rows, columns),
        test_matrix = MatrixOwner(rows, columns, {
            1, 2,
            2, 1,
            2, 3,
            3, 4,
            4, 5,
        }),
        expected_q_matrix = MatrixOwner(rows, columns, {
            -0.17149859,  0.43309279,
            -0.34299717, -0.83287076,
            -0.34299717,  0.29983347,
            -0.51449576,  0.16657415,
            -0.68599434,  0.03331483,
        })
    ;

    MatrixSpan
        q_span = q.to_span(),
        r_span = r.to_span(),
        test_span = test_matrix.to_span(),
        expected_q_span = expected_q_matrix.to_span()
    ;

    {
        q_span.set_identity();
        r_span.copy_from_unsafe(test_span);

        float* hhv = reinterpret_cast<float*>(alloca(sizeof(float) * q_span.row_ct));
        float* tv = reinterpret_cast<float*>(alloca(sizeof(float) * q_span.column_ct));

        auto householder_vector = VectorSpan(hhv, q_span.row_ct);
        auto tau = VectorSpan(tv, q_span.column_ct);

        QRDecomposition::qr_partial_householder(q_span, r_span, householder_vector, tau, false, zero_threshold);
        ASSERT_TRUE(expected_q_span.equals(q_span, 1e-05));
    }
}

TEST(qr_algorithm_tests, qr_decomposition_non_square)
{
    unsigned max_rows = 200, row_increment = 5, max_cols = 10;

    for (unsigned row_ct = 5; row_ct <= max_rows; row_ct += row_increment)
    {
        for (unsigned column_ct = 1; column_ct <= std::min(max_cols, row_ct); ++column_ct)
        {
            MatrixOwner
                q_full = MatrixOwner(row_ct, row_ct),
                q_partial = MatrixOwner(row_ct, column_ct),
                r = MatrixOwner(row_ct, column_ct),
                test_matrix = MatrixOwner(row_ct, column_ct)
            ;

            MatrixSpan
                q_full_span = q_full.to_span(),
                q_partial_span = q_partial.to_span(),
                r_span = r.to_span(),
                test_span = test_matrix.to_span()
            ;

            test_span.fill_random_uniform(-3, 3);

            {
                q_full_span.set_identity();
                q_partial_span.set_identity();
                r_span.copy_from_unsafe(test_span);

                float* hhv = reinterpret_cast<float*>(alloca(sizeof(float) * q_partial_span.row_ct));
                float* tv = reinterpret_cast<float*>(alloca(sizeof(float) * q_partial_span.column_ct));

                auto householder_vector = VectorSpan(hhv, q_partial_span.row_ct);
                auto tau = VectorSpan(tv, q_partial_span.column_ct);

                QRDecomposition::qr_full_householder(q_full_span, r_span, householder_vector, zero_threshold);

                r_span.copy_from_unsafe(test_span);

                QRDecomposition::qr_partial_householder(q_partial_span, r_span, householder_vector, tau, false, zero_threshold);

                auto q_full_span_truncated = q_full_span.to_subspan_unsafe(q_partial_span.row_ct, q_partial_span.column_ct);

                if (!q_partial_span.equals(q_full_span_truncated, 1e-05))
                {
                    std::cout << "Full: \n";
                    q_full_span.print();

                    std::cout << "Full Truncated: \n";
                    q_full_span_truncated.print();

                    std::cout << "Partial: \n";
                    q_partial_span.print();
                }

                ASSERT_TRUE(q_partial_span.equals(q_full_span_truncated, 1e-05));
            }
        }
    }
}