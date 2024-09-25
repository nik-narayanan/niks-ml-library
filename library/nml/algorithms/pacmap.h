//
// Created by nik on 3/31/2024.
//

#ifndef NML_PACMAP_H
#define NML_PACMAP_H

#include "pca.h"
#include "approximate_nearest_neighbor.h"

#include "../primitives/grid_span.h"
#include "../primitives/vector_span.h"
#include "../primitives/matrix_span.h"

#define PACMAP_DEBUG 0

#if PACMAP_DEBUG
    #define ALLOW_EXCEPT
#else
    #define ALLOW_EXCEPT noexcept
#endif

namespace nml::pacmap_internal
{
    struct AdamState;
    struct WorkingPairs;
    struct PacmapWeights;
}

namespace nml
{
    using namespace nml::pacmap_internal;

    struct Pacmap
    {
        MatrixSpan projection;

        struct Request
        {
            const MatrixSpan& data;
            const unsigned char thread_count;
            const unsigned char n_components;

            unsigned seed = 42;
            float zero_threshold = 1e-6;
            unsigned char n_neighbors = 10;
            unsigned char learning_rate = 100;
            float first_moment_decay_rate = 0.9;
            float second_moment_decay_rate = 0.999;
            unsigned char mid_near_pairs_count = 5;
            unsigned char further_pairs_count = 20;
            float mid_near_weight_initialization = 1000;
            unsigned short phase_1_iteration_count = 100;
            unsigned short phase_2_iteration_count = 100;
            unsigned short phase_3_iteration_count = 250;
            unsigned char approximate_nearest_neighbors_tree_count = 20;

            explicit Request(const MatrixSpan& data, unsigned char n_components, bool multithread = false) noexcept
                : data(data), n_components(n_components)
                , thread_count(multithread ? std::max(1u, std::thread::hardware_concurrency()) : 1)
            { }

            [[nodiscard]] StringResult<bool> is_valid() const ALLOW_EXCEPT
            {
                if (n_components < 2) return StringResult<bool>::err("n_components must be > 2");
                if (n_neighbors > 20) return StringResult<bool>::err("n_components must be < 20");
                if (learning_rate <= 0) return StringResult<bool>::err("learning_rate must be > 0");
                if (mid_near_pairs_count > 1) return StringResult<bool>::err("mid_near_ratio must be > 0 and < 1");
                if (further_pairs_count < n_neighbors) return StringResult<bool>::err("further_pairs_ratio must be > n_neighbors");
                if (n_components >= data.column_ct) return StringResult<bool>::err("n_components must be greater than the starting dimensions");

                return StringResult<bool>::ok(true);
            }
        };

        static RequiredMemory required_memory(const Request& request) ALLOW_EXCEPT;
        static StringResult<MatrixSpan> compute(const Request& request, RequestMemory& memory) ALLOW_EXCEPT;

    private:

        static RequiredMemory get_preprocessed_working_matrix_required_memory(const Request& request) ALLOW_EXCEPT;
        static MatrixSpan get_preprocessed_working_matrix(const Request& request, MemorySpan& working_memory, unsigned& working_memory_offset) ALLOW_EXCEPT;

        static RequiredMemory get_working_pairs_required_memory(const Request& request) ALLOW_EXCEPT;
        static WorkingPairs get_working_pairs(const Request& request, const MatrixSpan& working_matrix, MemorySpan& working_memory, unsigned& working_memory_offset) ALLOW_EXCEPT;

        static void update_gradient(const MatrixSpan& working_matrix, MatrixSpan& gradient, const PacmapWeights& weights, const WorkingPairs& pairs, VectorSpan& point_distances) ALLOW_EXCEPT;
        static void update_adam_state(MatrixSpan& working_matrix, AdamState& adam, unsigned iteration) ALLOW_EXCEPT;
    };
}

namespace nml::pacmap_internal
{
    struct WorkingPairs
    {
        GridSpan<unsigned> further_pairs;
        GridSpan<unsigned> mid_near_pairs;
        GridSpan<unsigned> neighbor_pairs;
    };

    struct PacmapWeights
    {
        float neighbors_weight;
        float further_points_weight;
        float mid_near_points_weight;
    };

    struct AdamState
    {
        MatrixSpan gradient;
        float learning_rate;
        float zero_threshold;
        float first_moment_decay_rate;
        float second_moment_decay_rate;
        MatrixSpan first_moment_vector;
        MatrixSpan second_moment_vector;
    };

    static inline RequiredMemory get_approximate_nearest_neighbors_required_memory(const Pacmap::Request& request) ALLOW_EXCEPT
    {
        const auto row_count = request.data.row_ct;
        const auto tree_count = request.approximate_nearest_neighbors_tree_count;

        auto ann_request = ApproximateNearestNeighborIndex::BuildRequest(request.data, tree_count, request.thread_count, request.seed);

        unsigned n_neighbors_tree = std::min(request.n_neighbors + 50u, request.data.row_ct - 1u);

        auto ann_build_required_memory = ApproximateNearestNeighborIndex::build_required_memory(ann_request);

        auto ann_bulk_find_request = ApproximateNearestNeighborIndex::BulkFindRequest(n_neighbors_tree, 0, request.thread_count);
        auto ann_bulk_find_required_memory = ApproximateNearestNeighborIndex::bulk_find_required_memory(ann_bulk_find_request, tree_count, row_count);

        return
        {
            .result_required_bytes = ann_bulk_find_required_memory.result_required_bytes,
            .working_required_bytes = ann_build_required_memory.total_bytes() + ann_bulk_find_required_memory.working_required_bytes
        };
    }

    static inline GridSpan<ScoredValue<unsigned>> get_approximate_nearest_neighbors(
            const Pacmap::Request& request,
            const MatrixSpan& working_matrix,
            MemorySpan& working_memory,
            unsigned& working_memory_offset
    ) ALLOW_EXCEPT
    {
        const auto tree_count = request.approximate_nearest_neighbors_tree_count;

        unsigned n_neighbors_tree = std::min(request.n_neighbors + 50u, working_matrix.row_ct - 1u);

        auto ann_matrix = GridSpan<ScoredValue<unsigned>>(working_memory.offset(working_memory_offset), working_matrix.row_ct, n_neighbors_tree);

        working_memory_offset += ann_matrix.bytes();

        unsigned local_working_memory_offset = working_memory_offset;

        auto ann_request = ApproximateNearestNeighborIndex::BuildRequest(working_matrix, tree_count, request.thread_count, request.seed);

        auto ann_build_required_memory = ApproximateNearestNeighborIndex::build_required_memory(ann_request);

        auto ann_build_memory = RequestMemory::from_required_unsafe(ann_build_required_memory, working_memory, working_memory_offset);

        auto ann_index_result = ApproximateNearestNeighborIndex::build(ann_request, ann_build_memory);

        if (ann_index_result.is_err() && PACMAP_DEBUG)
        {
            throw std::runtime_error("error finding ann_index");
        }

        auto ann_index = ann_index_result.ok();

        local_working_memory_offset += ann_build_required_memory.result_required_bytes;

        auto ann_bulk_find_request = ApproximateNearestNeighborIndex::BulkFindRequest(n_neighbors_tree, 0, request.thread_count);
        auto ann_bulk_find_required_memory = ann_index.bulk_find_required_memory(ann_bulk_find_request);

        RequestMemory bulk_find_request_memory =
        {
            .result_memory = ann_matrix.to_memory(),
            .working_memory = working_memory.offset(local_working_memory_offset)
        };

        local_working_memory_offset += bulk_find_request_memory.working_memory.bytes;

        auto bulk_find_nearest_neighbors_result = ann_index.bulk_find_nearest_neighbors(ann_bulk_find_request, bulk_find_request_memory);

        if (bulk_find_nearest_neighbors_result.is_err() && PACMAP_DEBUG)
        {
            throw std::runtime_error("unable complete bulk_find_nearest_neighbors_result");
        }

        return bulk_find_nearest_neighbors_result.ok();
    }

    static inline VectorSpan get_neighbor_weights(
            const Pacmap::Request& request,
            const GridSpan<ScoredValue<unsigned>>& ann,
            MemorySpan result_memory,
            unsigned& working_memory_offset
    ) ALLOW_EXCEPT
    {
        auto sigma = VectorSpan(result_memory, ann.row_ct);

        working_memory_offset += sigma.bytes();

        for (unsigned row_offset = 0; row_offset < ann.row_ct; ++row_offset)
        {
            float mean = 0;
            unsigned neighbor_offset = 3;
            auto row = ann[row_offset];

            for (; neighbor_offset < ann.row_ct && neighbor_offset < 6; ++neighbor_offset)
            {
                mean += row[neighbor_offset].score;
            }

            mean /= static_cast<float>(neighbor_offset - 3);

            sigma[row_offset] = std::max(mean, request.zero_threshold);
        }

        return sigma;
    }

    static inline GridSpan<unsigned> get_neighbor_pairs(
            const Pacmap::Request& request,
            const GridSpan<ScoredValue<unsigned>>& ann_matrix,
            RequestMemory memory
    ) ALLOW_EXCEPT
    {
        unsigned working_memory_offset = 0;

        auto neighbor_pairs = GridSpan<unsigned>(memory.result_memory, request.data.row_ct, request.n_neighbors);

        auto neighbor_weights = get_neighbor_weights(request, ann_matrix, memory.working_memory, working_memory_offset);

        auto scaled_neighbor_queue = MinMaxHeap<ScoredValue<unsigned>>(memory.working_memory.offset(working_memory_offset), request.n_neighbors);

        working_memory_offset += scaled_neighbor_queue.memory_size();

        for (unsigned row = 0; row < ann_matrix.row_ct; ++row)
        {
            auto ann_row = ann_matrix[row];

            for (int column = 0; column < ann_matrix.column_ct; ++column)
            {
                auto scored_neighbor = ann_row[column];

                float scaled_distance = scored_neighbor.score * scored_neighbor.score / neighbor_weights[row] / neighbor_weights[scored_neighbor.value];
                scaled_neighbor_queue.insert_min({scaled_distance, scored_neighbor.value });
            }

            unsigned pair_offset = 0;

            auto neighbor_pair_row = neighbor_pairs[row];

            while (!scaled_neighbor_queue.is_empty())
            {
                auto next = scaled_neighbor_queue.remove_min_unsafe();

                neighbor_pair_row[pair_offset++] = next.value;
            }
        }

        return neighbor_pairs;
    }

    static inline GridSpan<unsigned> get_mid_near_pairs_sample(const Pacmap::Request& request, const MatrixSpan& working_matrix, RequestMemory memory) ALLOW_EXCEPT
    {
        auto mid_near_pairs = GridSpan<unsigned>(memory.result_memory, request.data.row_ct, request.mid_near_pairs_count);

        // TODO for each thread
        auto random = Random((request.seed + request.mid_near_pairs_count) * mid_near_pairs.bytes());

        for (unsigned row = 0; row < working_matrix.row_ct; ++row)
        {
            auto data_row = working_matrix[row];

            for (unsigned mid_pair = 0; mid_pair < request.mid_near_pairs_count; ++mid_pair)
            {
                ScoredValue<unsigned> closest_score = {std::numeric_limits<float>::max(), 0},
                                      second_closest_score = {std::numeric_limits<float>::max(), 0};

                for (unsigned sample = 0; sample < 6; ++sample)
                {
                    unsigned sample_index = random.index(working_matrix.row_ct);
                    float sample_distance = euclidean_distance(data_row, working_matrix[sample_index]);

                    ScoredValue<unsigned> scored_sample = {sample_distance, sample_index };

                    if (scored_sample < closest_score)
                    {
                        second_closest_score = closest_score;
                        closest_score = scored_sample;
                    }
                    else if (scored_sample < second_closest_score)
                    {
                        second_closest_score = scored_sample;
                    }
                }

                mid_near_pairs.set_unsafe(row, mid_pair, second_closest_score.value);
            }
        }

        return mid_near_pairs;
    }

    static inline GridSpan<unsigned> get_further_pairs_sample(const Pacmap::Request& request, const GridSpan<unsigned>& neighbor_pairs, RequestMemory memory) ALLOW_EXCEPT
    {
        const unsigned max_iterations = 3 * (request.further_pairs_count + request.further_pairs_count);

        auto further_pairs = GridSpan<unsigned>(memory.result_memory.offset(0), neighbor_pairs.row_ct, request.further_pairs_count);

        // TODO for each thread
        auto random = Random((request.seed + request.further_pairs_count) * memory.working_memory.bytes);

        for (unsigned row = 0; row < neighbor_pairs.row_ct; ++row)
        {
            auto row_further_pairs = further_pairs[row];
            auto row_neighbor_pairs = neighbor_pairs[row];

            for (unsigned further_pair = 0; further_pair < request.further_pairs_count; ++further_pair)
            {
                auto existing_pairs = row_further_pairs.to_subspan_unsafe(0, further_pair);

                unsigned iteration = 0;

                while (iteration++ < max_iterations)
                {
                    unsigned random_index = random.index(neighbor_pairs.row_ct);

                    if (existing_pairs.scan_contains(random_index)) continue;

                    if (row_neighbor_pairs.scan_contains(random_index)) continue;

                    row_further_pairs[further_pair] = random_index; break;
                }

                row_further_pairs[further_pair] = random.index(neighbor_pairs.row_ct);
            }
        }

        return further_pairs;
    }

    static PacmapWeights get_iteration_weights(const Pacmap::Request request, unsigned iteration) ALLOW_EXCEPT
    {
        if (iteration < request.phase_1_iteration_count)
        {
            const auto iteration_float = static_cast<float>(iteration),
                    phase_1_iteration_float = static_cast<float>(request.phase_1_iteration_count);

            return PacmapWeights
            {
                .neighbors_weight = 2.0f,
                .further_points_weight = 1.0f,
                .mid_near_points_weight = (1 - iteration_float / phase_1_iteration_float) * request.mid_near_weight_initialization
                                          + iteration_float / phase_1_iteration_float * 3.0f,
            };
        }

        if (iteration < request.phase_1_iteration_count + request.phase_2_iteration_count) return PacmapWeights
        {
            .neighbors_weight = 3.0f,
            .further_points_weight = 1.0f,
            .mid_near_points_weight = 3.0f,
        };

        return PacmapWeights
        {
            .neighbors_weight = 1.0f,
            .further_points_weight = 1.0f,
            .mid_near_points_weight = 0.0f,
        };
    }

    static inline unsigned get_initial_adam_state_required_memory(const Pacmap::Request request) ALLOW_EXCEPT
    {
        unsigned element_size = sizeof(float);

        return 3 * request.data.row_ct * request.data.column_ct * element_size;
    }

    static inline AdamState get_initial_adam_state(const Pacmap::Request request, MemorySpan& working_memory, unsigned& working_memory_offset) ALLOW_EXCEPT
    {
        const unsigned starting_offset = working_memory_offset;

        auto gradient = MatrixSpan(working_memory.offset(working_memory_offset), request.data.row_ct, request.data.column_ct);

        working_memory_offset += gradient.bytes();

        auto first_moment_vector = MatrixSpan(working_memory.offset(working_memory_offset), request.data.row_ct, request.data.column_ct);

        working_memory_offset += first_moment_vector.bytes();

        auto second_moment_vector = MatrixSpan(working_memory.offset(working_memory_offset), request.data.row_ct, request.data.column_ct);

        working_memory_offset += second_moment_vector.bytes();

        auto iterations_memory = working_memory.to_subspan_unsafe(starting_offset, working_memory_offset - starting_offset);

        memset(iterations_memory.get_pointer(0), 0, iterations_memory.bytes);

        return AdamState
        {
            .gradient = gradient,
            .learning_rate = static_cast<float>(request.learning_rate),
            .zero_threshold = request.zero_threshold,
            .first_moment_decay_rate = request.first_moment_decay_rate,
            .second_moment_decay_rate = request.second_moment_decay_rate,
            .first_moment_vector = first_moment_vector,
            .second_moment_vector = second_moment_vector
        };
    }

    PCA::Request get_pca_request(const Pacmap::Request& request) ALLOW_EXCEPT
    {
        auto pca_request = PCA::Request(request.data, request.n_components);

        pca_request.preprocess = PCA::PreProcessOption::CENTER;

        return pca_request;
    }

    RequiredMemory get_pca_projection_required_memory(const Pacmap::Request& request) ALLOW_EXCEPT
    {
        auto pca_request = get_pca_request(request);

        return PCA::required_memory(pca_request);
    }

    MatrixSpan get_pca_projection(const Pacmap::Request& request, MemorySpan& working_memory, unsigned& working_memory_offset) ALLOW_EXCEPT
    {
        auto pca_request = get_pca_request(request);

        auto pca_required_memory = PCA::required_memory(pca_request);

        auto pca_memory = RequestMemory::from_required_unsafe(pca_required_memory, working_memory, working_memory_offset);

        auto pca_result = PCA::compute(pca_request, pca_memory);

        if (pca_result.is_err() && PACMAP_DEBUG) throw std::runtime_error("unable to compute pca_result");

        auto pca = pca_result.ok();

        working_memory_offset += pca.projection.bytes();

        float* pca_ptr = pca.projection.get_pointer();
        unsigned pca_element_count = pca.projection.element_count();

        return pca.projection;
    }
}

namespace nml
{
    RequiredMemory Pacmap::required_memory(const Request& request) ALLOW_EXCEPT
    {
        RequiredMemory required_memory =
        {
            .result_required_bytes = 0,
            .working_required_bytes = 0
        };

        RequiredMemory working_matrix = get_preprocessed_working_matrix_required_memory(request);

        required_memory.result_required_bytes = working_matrix.result_required_bytes;
        required_memory.working_required_bytes += working_matrix.working_required_bytes;

        RequiredMemory working_pairs = get_working_pairs_required_memory(request);

        working_matrix.working_required_bytes = std::max(working_matrix.working_required_bytes, working_pairs.total_bytes());

        uint64_t adam_state = get_initial_adam_state_required_memory(request);
        uint64_t point_distances = request.data.column_ct * sizeof(float);
        uint64_t iteration_memory = working_pairs.result_required_bytes + adam_state + point_distances;

        working_matrix.working_required_bytes = std::max(working_matrix.working_required_bytes, iteration_memory);

        return required_memory;
    }

    StringResult<MatrixSpan> Pacmap::compute(const Request& request, RequestMemory& memory) ALLOW_EXCEPT
    {
        auto request_check = request.is_valid();

        if (request_check.is_err())
        {
            return StringResult<MatrixSpan>::err(request_check.err());
        }

        if (!memory.is_sufficient(required_memory(request)))
        {
            return StringResult<MatrixSpan>::err("insufficient memory to compute pacmap");
        }

        unsigned working_memory_offset = 0, result_memory_offset = 0;

        MatrixSpan working_matrix = get_preprocessed_working_matrix(request, memory.working_memory, working_memory_offset);

        WorkingPairs working_pairs = get_working_pairs(request, working_matrix, memory.working_memory, working_memory_offset);

        AdamState adam_state = get_initial_adam_state(request, memory.working_memory, working_memory_offset);

        unsigned total_iterations = request.phase_1_iteration_count + request.phase_2_iteration_count + request.phase_3_iteration_count;

        auto update_gradient_memory = VectorSpan(memory.working_memory.offset(working_memory_offset), working_matrix.column_ct);

        for (unsigned iteration = 0; iteration < total_iterations; ++iteration)
        {
            PacmapWeights weights = get_iteration_weights(request, iteration);

#if PACMAP_DEBUG
            std::cout << "Iteration: " << iteration << '\n';
#endif
            update_gradient(working_matrix, adam_state.gradient, weights, working_pairs, update_gradient_memory);

            update_adam_state(working_matrix, adam_state, iteration);
        }

        return StringResult<MatrixSpan>::ok(working_matrix);
    }

    RequiredMemory Pacmap::get_preprocessed_working_matrix_required_memory(const Request& request) ALLOW_EXCEPT
    {
        //TODO truncated svd

        return get_pca_projection_required_memory(request);
    }

    MatrixSpan Pacmap::get_preprocessed_working_matrix(const Request& request, MemorySpan& working_memory, unsigned& working_memory_offset) ALLOW_EXCEPT
    {
        //TODO truncated svd

        auto pca_projection = get_pca_projection(request, working_memory, working_memory_offset);

        return pca_projection;
    }

    RequiredMemory Pacmap::get_working_pairs_required_memory(const Pacmap::Request& request) ALLOW_EXCEPT
    {
        const auto approximate_nearest_neighbors_required_memory = get_approximate_nearest_neighbors_required_memory(request);

        const unsigned neighbor_pairs_working = MinMaxHeap<ScoredValue<unsigned>>::required_bytes(request.n_neighbors) + VectorSpan::required_bytes(request.data.row_ct);
        const unsigned neighbor_pairs_result = GridSpan<unsigned>::required_bytes(request.data.row_ct, request.n_neighbors);

        const unsigned mid_near_pairs_working = 0;
        const unsigned mid_near_pairs_result = GridSpan<unsigned>::required_bytes(request.data.row_ct, request.mid_near_pairs_count);

        const unsigned further_pairs_working = 0;
        const unsigned further_pairs_result = GridSpan<unsigned>::required_bytes(request.data.row_ct, request.further_pairs_count);

        return RequiredMemory
        {
            .result_required_bytes = neighbor_pairs_result + mid_near_pairs_result + further_pairs_result,
            .working_required_bytes = std::max(
                    neighbor_pairs_working + mid_near_pairs_working + further_pairs_working + approximate_nearest_neighbors_required_memory.result_required_bytes,
                    approximate_nearest_neighbors_required_memory.total_bytes()
            )
        };
    }

    WorkingPairs Pacmap::get_working_pairs(const Request& request, const MatrixSpan& working_matrix, MemorySpan& working_memory, unsigned& working_memory_offset) ALLOW_EXCEPT
    {
        unsigned local_working_memory_offset = 0, result_memory_offset = 0;

        auto required_memory = get_working_pairs_required_memory(request);

        auto local_memory = RequestMemory::from_required_unsafe(required_memory, working_memory, working_memory_offset);

        working_memory_offset += local_memory.result_memory.bytes;

        auto ann_matrix = get_approximate_nearest_neighbors(request, working_matrix, local_memory.working_memory, local_working_memory_offset);

        auto neighbor_pairs = get_neighbor_pairs(request, ann_matrix, {
            local_memory.result_memory.offset(result_memory_offset),
            local_memory.working_memory.offset(local_working_memory_offset)
        });

        result_memory_offset += neighbor_pairs.bytes();

        auto mid_near_pairs = get_mid_near_pairs_sample(request, working_matrix, {
            local_memory.result_memory.offset(result_memory_offset),
            local_memory.working_memory.offset(local_working_memory_offset)
        });

        result_memory_offset += mid_near_pairs.bytes();

        auto further_pairs = get_further_pairs_sample(request, neighbor_pairs, {
            local_memory.result_memory.offset(result_memory_offset),
            local_memory.working_memory.offset(local_working_memory_offset)
        });

        result_memory_offset += further_pairs.bytes();

        return
        {
            .further_pairs = further_pairs,
            .mid_near_pairs = mid_near_pairs,
            .neighbor_pairs = neighbor_pairs,
        };
    }

    void Pacmap::update_adam_state(MatrixSpan& working_matrix, AdamState& adam, unsigned iteration) ALLOW_EXCEPT
    {
        auto adjusted_iteration = static_cast<float>(iteration + 1);

        float learning_rate = adam.learning_rate
                            * std::sqrt(1.0f - std::pow(adam.second_moment_decay_rate, adjusted_iteration))
                            / (1.0f - std::pow(adam.first_moment_decay_rate, adjusted_iteration));

        float* gradient = adam.gradient.get_pointer();
        float* projection = working_matrix.get_pointer();
        float* first_moment = adam.first_moment_vector.get_pointer();
        float* second_moment = adam.second_moment_vector.get_pointer();

        unsigned total_elements = working_matrix.element_count();

        for (unsigned element = 0; element < total_elements; ++element) // TODO vector_cuda
        {
            first_moment[element] += (1 - adam.first_moment_decay_rate) * (gradient[element] - first_moment[element]);
            second_moment[element] += (1 - adam.second_moment_decay_rate) * (gradient[element] * gradient[element] - second_moment[element]);
            projection[element] -= learning_rate * first_moment[element] / (std::sqrt(second_moment[element]) + adam.zero_threshold);
        }
    }

    // TODO vector_cuda
    void Pacmap::update_gradient(const MatrixSpan& working_matrix, MatrixSpan& gradient, const PacmapWeights& weights, const WorkingPairs& pairs, VectorSpan& point_distances) ALLOW_EXCEPT
    {
        gradient.fill(0);

#if PACMAP_DEBUG
        float neighbor_pairs_loss = 0, mid_near_pairs_loss = 0, further_pairs_loss = 0;
#endif //PACMAP_DEBUG

        for (unsigned row = 0; row < pairs.neighbor_pairs.row_ct; ++row)
        {
            auto neighbor_pairs_row = pairs.neighbor_pairs[row];

            for (unsigned pair = 0; pair < pairs.neighbor_pairs.column_ct; ++pair)
            {
                float sum_of_squared_distances = 1.0f;
                auto neighbor_pair_row = neighbor_pairs_row[pair];

                for (unsigned column = 0; column < working_matrix.column_ct; ++column)
                {
                    float projection_value = working_matrix.get_unsafe(row, column);
                    float neighbor_value = working_matrix.get_unsafe(neighbor_pair_row, column);

                    point_distances[column] = projection_value - neighbor_value;
                    sum_of_squared_distances += point_distances[column] * point_distances[column];
                }

#if PACMAP_DEBUG
                neighbor_pairs_loss += weights.neighbors_weight * (d_ij / (10.0f + d_ij));
#endif //PACMAP_DEBUG

                float adjusted_neighbor_weight = weights.neighbors_weight * (20.0f / ((10.0f + sum_of_squared_distances) * (10.0f + sum_of_squared_distances)));

                for (unsigned column = 0; column < working_matrix.column_ct; ++column)
                {
                    gradient.get_unsafe_ref(row, column) += adjusted_neighbor_weight * point_distances[column];
                    gradient.get_unsafe_ref(neighbor_pair_row, column) -= adjusted_neighbor_weight * point_distances[column];
                }
            }
        }

        for (unsigned row = 0; row < pairs.mid_near_pairs.row_ct; ++row)
        {
            auto mid_near_pairs_row = pairs.mid_near_pairs[row];

            for (unsigned pair = 0; pair < pairs.mid_near_pairs.column_ct; ++pair)
            {
                float sum_of_squared_distances = 1.0f;
                auto mid_near_pair_row = mid_near_pairs_row[pair];

                for (unsigned column = 0; column < working_matrix.column_ct; ++column)
                {
                    float projection_point = working_matrix.get_unsafe(row, column);
                    float mid_near_point = working_matrix.get_unsafe(mid_near_pair_row, column);

                    point_distances[column] = projection_point - mid_near_point;
                    sum_of_squared_distances += point_distances[column] * point_distances[column];
                }

#if PACMAP_DEBUG
                mid_near_pairs_loss += weights.mid_near_points_weight * (d_ij / (10000.0f + d_ij));
#endif //PACMAP_DEBUG

                float adjusted_mid_near_weight = weights.mid_near_points_weight
                        * (20000.0f / ((10000.0f + sum_of_squared_distances) * (10000.0f + sum_of_squared_distances)));

                for (unsigned column = 0; column < working_matrix.column_ct; ++column)
                {
                    gradient.get_unsafe_ref(row, column) += adjusted_mid_near_weight * point_distances[column];
                    gradient.get_unsafe_ref(mid_near_pair_row, column) -= adjusted_mid_near_weight * point_distances[column];
                }
            }
        }

        for (unsigned row = 0; row < working_matrix.row_ct; ++row)
        {
            auto further_pairs_row = pairs.further_pairs[row];

            for (unsigned pair = 0; pair < working_matrix.column_ct; ++pair)
            {
                float sum_of_squared_distances = 1.0f;
                auto further_pair_row = further_pairs_row[pair];

                for (unsigned column = 0; column < working_matrix.column_ct; ++column)
                {
                    float projection_point = working_matrix.get_unsafe(row, column);
                    float further_point = working_matrix.get_unsafe(further_pair_row, column);

                    point_distances[column] = projection_point - further_point;
                    sum_of_squared_distances += point_distances[column] * point_distances[column];
                }

#if PACMAP_DEBUG
                further_pairs_loss += weights.further_points_weight * (1.0f / (1.0f + d_ij));
#endif //PACMAP_DEBUG

                float adjusted_further_points_weight = weights.further_points_weight
                        * (2.0f / ((1.0f + sum_of_squared_distances) * (1.0f + sum_of_squared_distances)));

                for (unsigned column = 0; column < working_matrix.column_ct; ++column)
                {
                    gradient.get_unsafe_ref(row, column) += adjusted_further_points_weight * point_distances[column];
                    gradient.get_unsafe_ref(further_pair_row, column) -= adjusted_further_points_weight * point_distances[column];
                }
            }
        }

#if PACMAP_DEBUG
        float total_loss = neighbor_pairs_loss + mid_near_pairs_loss + further_pairs_loss;

        std::cout    "    Neighbor Loss: " << neighbor_pairs_loss << '\n'
                  << "    Mid Near Loss: " << mid_near_pairs_loss << '\n'
                  << "    Further Loss: " << further_pairs_loss << '\n'
                  << "    Total Loss: " << total_loss << '\n';
#endif //PACMAP_DEBUG
    }


}

#endif //NML_PACMAP_H