//
// Created by nik on 5/6/2024.
//

#ifndef NML_RANDOMIZED_LOW_RANK_APPROXIMATION_H
#define NML_RANDOMIZED_LOW_RANK_APPROXIMATION_H

#include "../algorithms/qr_algorithm.h"
#include "../algorithms/lu_decomposition.h"

namespace nml
{
    struct RandomizedLowRankApproximation
    {
        struct Request
        {
            const MatrixSpan& data;
            const unsigned iterations;
            const unsigned over_samples;

            explicit Request(const MatrixSpan& data, unsigned iterations = 5, unsigned over_samples = 10) noexcept
                : data(data), iterations(iterations), over_samples(over_samples)
            { }
        };

        static RequiredMemory required_memory(const Request& request) noexcept;
        static NMLResult <MatrixSpan> compute(const Request& request, RequestMemory memory) noexcept;
    };
}

namespace nml::randomized_low_rank_approximation_internal
{
    struct LowRankApproximationState
    {
        MatrixSpan l;
        MatrixSpan u;
        VectorSpan tau;
        Span<unsigned> lu_pivots;
        MatrixSpan random_matrix;
        MatrixSpan random_projection;
        VectorSpan householder_vector;
        MatrixSpan random_projection_window;
    };

    static LowRankApproximationState get_low_dimensional_projection_state(const RandomizedLowRankApproximation::Request& request, MemorySpan& memory, unsigned& memory_offset) noexcept
    {
        MatrixSpan random_matrix = MatrixSpan(memory.offset(0), request.data.column_ct, request.data.column_ct + request.over_samples);

        random_matrix.fill_random_uniform(-1, 1);

        memory_offset += random_matrix.bytes();

        MatrixSpan random_projection = MatrixSpan(memory.offset(memory_offset), request.data.row_ct, request.data.column_ct);

        memory_offset += random_projection.bytes();

        MatrixSpan random_projection_window = MatrixSpan(memory.offset(memory_offset), request.data.column_ct, request.data.column_ct);

        memory_offset += random_projection_window.bytes();

        MatrixSpan l_1 = MatrixSpan(memory.offset(memory_offset), request.data.row_ct, request.data.column_ct);

        memory_offset += l_1.bytes();

        MatrixSpan u_1 = MatrixSpan(memory.offset(memory_offset), request.data.column_ct, request.data.column_ct);

        memory_offset += u_1.bytes();

        Span<unsigned> pivots = Span<unsigned>(memory.offset(memory_offset), request.data.row_ct);

        memory_offset += pivots.bytes();

        VectorSpan tau = VectorSpan(memory.offset(memory_offset), request.data.column_ct);

        memory_offset += tau.bytes();

        VectorSpan householder_vector = VectorSpan(memory.offset(memory_offset), request.data.row_ct);

        memory_offset += householder_vector.bytes();

        return LowRankApproximationState
        {
            .l = l_1,
            .u = u_1,
            .tau = tau,
            .lu_pivots = pivots,
            .random_matrix = random_matrix,
            .random_projection = random_projection,
            .householder_vector = householder_vector,
            .random_projection_window = random_projection_window
        };
    }
}

namespace nml
{
    using namespace nml::randomized_low_rank_approximation_internal;

    RequiredMemory RandomizedLowRankApproximation::required_memory(const Request& request) noexcept
    {
        const unsigned result_projection_matrix_bytes = MatrixSpan::required_bytes(request.data.column_ct, request.data.column_ct);

        const unsigned working_matrix_bytes = MatrixSpan::required_bytes(request.data.column_ct, request.data.column_ct + request.over_samples);
        const unsigned random_projection_bytes = MatrixSpan::required_bytes(request.data.row_ct, request.data.column_ct);
        const unsigned random_projection_window_bytes = MatrixSpan::required_bytes(request.data.column_ct, request.data.column_ct);

        const unsigned l_bytes = MatrixSpan::required_bytes(request.data.row_ct, request.data.column_ct);
        const unsigned u_bytes = MatrixSpan::required_bytes(request.data.column_ct, request.data.column_ct);
        const unsigned lu_pivots_bytes = Span<unsigned>::required_bytes(request.data.row_ct);

        const unsigned tau_bytes = VectorSpan::required_bytes(request.data.column_ct);
        const unsigned householder_vector_bytes = VectorSpan::required_bytes(request.data.row_ct);

        return RequiredMemory
        {
            .result_required_bytes = result_projection_matrix_bytes,
            .working_required_bytes = working_matrix_bytes + random_projection_bytes + random_projection_window_bytes
                                      + l_bytes + u_bytes + lu_pivots_bytes
                                      + tau_bytes + householder_vector_bytes,
        };
    }

    NMLResult<MatrixSpan> RandomizedLowRankApproximation::compute(const Request& request, RequestMemory memory) noexcept
    {
        if (!memory.is_sufficient(required_memory(request)))
        {
            return NMLResult<MatrixSpan>::err(NMLErrorCode::INSUFFICIENT_MEMORY);
        }

        unsigned working_memory_offset = 0;

        auto working_matrix = MatrixSpan(memory.result_memory.offset(0), request.data.column_ct, request.data.column_ct);

        LowRankApproximationState state = get_low_dimensional_projection_state(request, memory.working_memory, working_memory_offset);

        request.data.multiply(state.random_matrix, state.random_projection);

        LUDecomposition::decompose_unsafe(state.random_projection, state.l, state.u, state.lu_pivots);

        request.data.transpose_multiply(state.l, state.random_projection_window);

        LUDecomposition::decompose_unsafe(state.random_projection_window, working_matrix, state.u, state.lu_pivots);

        for (unsigned iteration = 0; iteration < request.iterations; ++iteration)
        {
            request.data.multiply(working_matrix, state.random_projection);

            LUDecomposition::decompose_unsafe(state.random_projection, state.l, state.u, state.lu_pivots);

            request.data.transpose_multiply(state.l, state.random_projection_window);

            LUDecomposition::decompose_unsafe(state.random_projection_window, working_matrix, state.u, state.lu_pivots);
        }

        request.data.multiply(working_matrix, state.random_projection);

        state.l.set_identity();

        QRDecomposition::qr_partial_householder(state.l, state.random_projection, state.householder_vector, state.tau);

        state.l.transpose_multiply(request.data, working_matrix);

        return NMLResult<MatrixSpan>::ok(working_matrix);
    }
}

#endif //NML_RANDOMIZED_LOW_RANK_APPROXIMATION_H