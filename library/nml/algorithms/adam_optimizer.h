//
// Created by nik on 5/26/2024.
//

#ifndef NML_ADAM_OPTIMIZER_H
#define NML_ADAM_OPTIMIZER_H

#include <cmath>

#include "../primitives/span.h"
#include "../primitives/matrix_span.h"

namespace nml
{
    struct AdamOptimizer
    {
        struct Parameters
        {
            float learning_rate;
            float zero_threshold;
            float first_moment_decay_rate;
            float second_moment_decay_rate;
        };

        VectorSpan gradient;
        Parameters parameters;
        VectorSpan first_moment;
        VectorSpan second_moment;

        static unsigned required_bytes(unsigned elements) noexcept
        {
            return VectorSpan::required_bytes(elements) * 3;
        }

        static AdamOptimizer build(Parameters parameters, unsigned elements, MemorySpan memory, unsigned& memory_offset) noexcept
        {
            const unsigned starting_offset = memory_offset;

            auto gradient = VectorSpan(memory.offset(memory_offset), elements);

            memory_offset += gradient.bytes();

            auto first_moment_vector = VectorSpan(memory.offset(memory_offset), elements);

            memory_offset += first_moment_vector.bytes();

            auto second_moment_vector = VectorSpan(memory.offset(memory_offset), elements);

            memory_offset += second_moment_vector.bytes();

            memset(memory.get_pointer(starting_offset), 0, memory_offset - starting_offset);

            return AdamOptimizer
            {
                .gradient = gradient,
                .parameters = parameters,
                .first_moment = first_moment_vector,
                .second_moment = second_moment_vector,
            };
        }

        inline void update(VectorSpan& projection, unsigned iteration)
        {
            auto adjusted_iteration = static_cast<float>(iteration + 1);

            float learning_rate = parameters.learning_rate
                                  * std::sqrt(1.0f - std::pow(parameters.second_moment_decay_rate, adjusted_iteration))
                                  / (1.0f - std::pow(parameters.first_moment_decay_rate, adjusted_iteration));

            for (unsigned element = 0; element < gradient.length; ++element)
            {
                first_moment[element] += (1 - parameters.first_moment_decay_rate) * (gradient[element] - first_moment[element]);
                second_moment[element] += (1 - parameters.second_moment_decay_rate) * (gradient[element] * gradient[element] - second_moment[element]);
                projection[element] += learning_rate * first_moment[element] / (std::sqrt(second_moment[element]) + parameters.zero_threshold);
            }
        }

        static constexpr AdamOptimizer::Parameters DEFAULT =
        {
            .learning_rate = 0.01,
            .zero_threshold = 1e-6,
            .first_moment_decay_rate = 0.9,
            .second_moment_decay_rate = 0.999
        };
    };
}

#endif //NML_ADAM_OPTIMIZER_H