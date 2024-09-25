//
// Created by nik on 5/26/2024.
//

#ifndef NML_LOGISTIC_REGRESSION_H
#define NML_LOGISTIC_REGRESSION_H

#include <cmath>

#include "similarity_distance.h"
#include "distribution_functions.h"
#include "gauss_jordan_elimination.h"

#include "../primitives/span.h"
#include "../primitives/heap.h"
#include "../primitives/hash.h"
#include "../primitives/matrix_span.h"
#include "../primitives/memory_owner.h"
#include "../primitives/scored_value.h"
#include "../primitives/regression_summary.h"

namespace nml
{
    struct LogisticRegression
    {
        struct Summary;

        float label;
        float constant;
        VectorSpan variables;
        bool complete_separation;

        struct Request
        {
            const MatrixSpan& data;

            float label;
            unsigned iterations;
            float zero_threshold;

            explicit Request(const MatrixSpan& data, float label = 1.0f, unsigned iterations = 50, float zero_threshold = 1e-6)
                : data(data), label(label), iterations(iterations), zero_threshold(zero_threshold)
            { }
        };

        [[nodiscard]] Summary summary(const MatrixSpan& data) const noexcept;
        [[nodiscard]] float predict(const VectorSpan& observation) const noexcept;

        static RequiredMemory required_memory(const Request& request) noexcept;
        static NMLResult<LogisticRegression> compute(const Request& request, RequestMemory memory) noexcept;
    };

    struct LogisticRegressionMulti
    {
        struct Summary;

        VectorSpan labels;
        VectorSpan constants;
        Span<VectorSpan> variables;

        struct Request
        {
            const MatrixSpan& data;
            const VectorSpan labels;

            uint16_t max_iteration_ct = 100;
            uint16_t retained_iteration_ct = 8;
            float convergence_threshold = 1e-4;
            float rounding_tolerance = 1.0e-10;

            float line_search_min_step_size = 1e-20;
            float line_search_max_step_size = 1e+20;
            uint16_t line_search_max_iteration_ct = 40;
            float line_search_gradient_tolerance = 0.2;
            float line_search_accuracy_tolerance = 1e-2;

            explicit Request(const MatrixSpan& data, const VectorSpan labels)
                : data(data), labels(labels)
            { }
        };

        [[nodiscard]] Summary summary(const MatrixSpan& data) const noexcept;
        void predict(const VectorSpan& observation, Heap<ScoredValue<unsigned>>& probabilities) const noexcept;

        static RequiredMemory required_memory(const Request& request) noexcept;
        static NMLResult<LogisticRegressionMulti> compute(const Request& request, RequestMemory memory) noexcept;
    };
}

/*******************************************************************************************************************
 SINGLE CLASS
 ******************************************************************************************************************/
namespace nml
{
    float LogisticRegression::predict(const VectorSpan& observation) const noexcept
    {
        float linear_combination = constant;

        for (unsigned i = 0; i < observation.length; ++i)
        {
            linear_combination += observation[i] * variables[i];
        }

        return 1.0f / (1.0f + std::exp(-linear_combination));
    }

    RequiredMemory LogisticRegression::required_memory(const Request& request) noexcept
    {
        unsigned gradients_bytes = VectorSpan::required_bytes(request.data.column_ct);

        unsigned coefficients_bytes = VectorSpan::required_bytes(request.data.column_ct);

        unsigned hessian_bytes = MatrixSpan::required_bytes(request.data.column_ct, request.data.column_ct + 1);

        return RequiredMemory
        {
            .result_required_bytes = VectorSpan::required_bytes(request.data.column_ct - 1),
            .working_required_bytes = gradients_bytes + coefficients_bytes + hessian_bytes
        };
    }

    NMLResult<LogisticRegression> LogisticRegression::compute(const Request& request, RequestMemory memory) noexcept
    {
        unsigned working_memory_offset = 0;

        const auto row_ct_float =  static_cast<float>(request.data.row_ct);

        auto working_matrix = request.data.to_subspan_unsafe(
            request.data.row_ct, request.data.column_ct - 1, 0, 1
        );

        auto y = request.data.to_subspan_unsafe(request.data.row_ct, 1);

        auto coefficients = VectorSpan(
            memory.working_memory.offset(working_memory_offset), request.data.column_ct
        );

        working_memory_offset += coefficients.bytes();
        coefficients.fill(0);

        auto pivots = Span<unsigned>(
            memory.working_memory.offset(working_memory_offset), request.data.column_ct
        );

        working_memory_offset += pivots.bytes();

        auto hessian_solver = MatrixSpan(
            memory.working_memory.offset(working_memory_offset), request.data.column_ct, request.data.column_ct + 1
        );
        working_memory_offset += hessian_solver.bytes();

        auto hessian_matrix = hessian_solver.to_subspan_unsafe(
            request.data.column_ct, request.data.column_ct
        );

        auto solver_matrix = hessian_solver.to_subspan_unsafe(
            request.data.column_ct, 1, 0, request.data.column_ct
        );

        bool converged = false; unsigned predicted_ct = 0;

        for (uint64_t iteration = 0; iteration < request.iterations && !converged; ++iteration)
        {
            predicted_ct = 0, hessian_solver.fill(0);

            for (uint64_t row_offset = 0; row_offset < request.data.row_ct; ++row_offset)
            {
                float linear_combination = coefficients[0];

                auto row = working_matrix[row_offset];

                for (uint64_t column = 1; column < coefficients.length; ++column)
                {
                    linear_combination += row[column - 1] * coefficients[column];
                }

                float prediction = 1.0f / (1.0f + std::exp(-linear_combination));

                float expected = y.get_unsafe(row_offset, 0) == request.label ? 1.0f : 0;

                float row_error = expected - prediction;

                predicted_ct += std::fabs(row_error) < request.zero_threshold;

                for (uint64_t hessian_row = 0; hessian_row < hessian_matrix.row_ct; ++hessian_row)
                {
                    auto row_value = hessian_row == 0 ? 1.0f : row[hessian_row - 1];

                    solver_matrix.get_unsafe_ref(hessian_row, 0) += row_error * row_value;

                    auto hessian_matrix_row = hessian_matrix[hessian_row];

                    hessian_matrix_row[0] -= prediction * (1 - prediction) * row_value * 1;

                    for (uint64_t hessian_column = 1; hessian_column < hessian_matrix.column_ct; ++hessian_column)
                    {
                        hessian_matrix_row[hessian_column] -= prediction * (1 - prediction) * row_value * row[hessian_column - 1];
                    }
                }
            }

            if (predicted_ct == request.data.row_ct)
            {
                break;
            }

            for (unsigned hessian_row = 0; hessian_row < hessian_matrix.row_ct; ++hessian_row)
            {
                hessian_matrix.get_unsafe_ref(hessian_row, hessian_row) += request.zero_threshold;
            }

            gauss_jordan_elimination(hessian_solver, pivots);

            converged = true;

            for (unsigned c = 0; c < coefficients.length; ++c)
            {
                auto coefficient_adjustment = solver_matrix.get_unsafe(pivots[c], 0);

                if (std::fabs(coefficient_adjustment) > request.zero_threshold)
                {
                    converged = false;
                }

                coefficients[c] -= coefficient_adjustment;
            }
        }

        auto independent_coefficients = VectorSpan(memory.result_memory, coefficients.length - 1);

        independent_coefficients.copy_from_unsafe(coefficients.to_subspan_unsafe(1));

        return NMLResult<LogisticRegression>::ok({
            .label = request.label,
            .constant = coefficients[0],
            .variables = independent_coefficients,
            .complete_separation = predicted_ct == request.data.row_ct
        });
    }

    struct LogisticRegression::Summary
    {
        float label;
        bool complete_separation;

        unsigned observation_count;
        unsigned model_degrees_of_freedom;
        unsigned residuals_degrees_of_freedom;

        unsigned true_positives;
        unsigned true_negatives;
        unsigned false_positives;
        unsigned false_negatives;

        float recall;
        float accuracy;
        float precision;

        float log_loss;
        float pseudo_r_squared;

        struct ConfidenceInterval
        {
            float upper_boundary;
            float lower_boundary;
        };

        struct Coefficient
        {
            float p_value;
            float odds_ratio;
            float coefficient;
            float z_statistic;
            float standard_error;
            std::string variable_name;

            ConfidenceInterval confidence_interval_95;
        };

        Coefficient constant;
        std::vector<Coefficient> independent_variables;

        void print();
    };

    LogisticRegression::Summary LogisticRegression::summary(const MatrixSpan& data) const noexcept
    {
        auto row_ct_float = static_cast<float>(data.row_ct);

        float log_loss = 0, log_loss_null = 0;

        unsigned true_positives = 0;
        unsigned true_negatives = 0;
        unsigned false_positives = 0;
        unsigned false_negatives = 0;

        for (unsigned r = 0; r < data.row_ct; ++r)
        {
            auto row = data[r];

            float prediction_null = 1.0f / (1.0f + std::exp(-constant));
            float prediction = predict(row.to_subspan_unsafe(1));

            float row_label = row[0] == label ? 1 : 0;

            log_loss -= row_label * std::log(prediction + 1e-6f) + (1.0f - row_label) * std::log(1.0f - prediction + 1e-6f);
            log_loss_null -= row_label * std::log(prediction_null + 1e-6f) + (1.0f - row_label) * std::log(1.0f - prediction_null + 1e-6f);

            bool is_positive = row_label >= 0.5;
            bool predicted_positive = prediction >= 0.5;

            if (is_positive && predicted_positive)
            {
                true_positives++;
            }
            else if (!is_positive && !predicted_positive)
            {
                true_negatives++;
            }
            else if (!is_positive)
            {
                false_positives++;
            }
            else
            {
                false_negatives++;
            }
        }

        auto true_positives_float = static_cast<float>(true_positives);
        auto true_negatives_float = static_cast<float>(true_negatives);
        auto false_positives_float = static_cast<float>(false_positives);
        auto false_negatives_float = static_cast<float>(false_negatives);

        float accuracy = (true_positives_float + true_negatives_float) / row_ct_float;
        float precision = true_positives_float / (true_positives_float + false_positives_float);
        float recall = true_positives_float / (true_positives_float + false_negatives_float);

        log_loss /= row_ct_float;
        log_loss_null /= row_ct_float;
        float pseudo_r_squared = 1 - log_loss / log_loss_null;

        auto independent_variables = std::vector<LogisticRegression::Summary::Coefficient>();

        unsigned offset = 0;
        auto memory = MemoryOwner(MatrixSpan::required_bytes(data.column_ct, data.column_ct * 2) + Span<unsigned>::required_bytes(data.column_ct));
        auto memory_span = memory.to_memory_span();
        auto fisher_weights_inverse = MatrixSpan(memory_span, data.column_ct, data.column_ct * 2);

        auto y = data.to_subspan_unsafe(data.row_ct, 1);

        auto fisher_weights = fisher_weights_inverse.to_subspan_unsafe(data.column_ct, data.column_ct);
        auto fisher_weights_inverse_right = fisher_weights_inverse.to_subspan_unsafe(data.column_ct, data.column_ct, 0, data.column_ct);

        auto pivots = Span<unsigned>(memory_span.offset(fisher_weights_inverse.bytes()), data.column_ct);

        fisher_weights_inverse.fill(0);

        for (unsigned i = 0; i < data.row_ct; ++i)
        {
            auto row = data[i];

            float predicted = predict(row.to_subspan_unsafe(1));
            float fisher_weight = predicted * (1 - predicted);

            for (unsigned j = 0; j < data.column_ct; ++j)
            {
                float data_t = j == 0 ? 1.0f : row[j];
                auto fisher_row = fisher_weights[j];

                for (unsigned k = 0; k < data.column_ct; ++k)
                {
                    float data_m = k == 0 ? 1.0f : row[k];
                    fisher_row[k] += fisher_weight * data_t * data_m;
                }
            }
        }

        fisher_weights_inverse_right.set_identity();

        gauss_jordan_elimination(fisher_weights_inverse);

        auto z_critical_95 = static_cast<float>(Distribution::normal_pdf(0.975));

        for (int i = 0; i < variables.length; ++i)
        {
            float standard_error = std::sqrt(std::fabs(fisher_weights_inverse_right[i + 1][i + 1]));
            float z_statistic = variables[i] / standard_error;
            auto p_value = static_cast<float>(2 * Distribution::normal_cdf(std::fabs(z_statistic), 0, 1, false));

            independent_variables.push_back({
                .p_value = p_value,
                .odds_ratio = std::exp(variables[i]),
                .coefficient = variables[i],
                .z_statistic = z_statistic,
                .standard_error = standard_error,
                .variable_name = "Variable " + std::to_string(i + 1),
                .confidence_interval_95 =
                {
                    .upper_boundary = variables[i] + z_critical_95 * standard_error,
                    .lower_boundary = variables[i] - z_critical_95 * standard_error
                }
            });
        }

        float standard_error = std::sqrt(std::fabs(fisher_weights_inverse_right[0][0]));
        float z_statistic = constant / standard_error;
        auto p_value = static_cast<float>(2 * Distribution::normal_cdf(std::fabs(z_statistic), 0, 1, false));

        return LogisticRegression::Summary
        {
            .label = label,
            .complete_separation = complete_separation,
            .observation_count = data.row_ct,
            .model_degrees_of_freedom = static_cast<uint32_t>(variables.length),
            .residuals_degrees_of_freedom = static_cast<uint32_t>(data.row_ct - variables.length - 1),
            .true_positives = true_positives,
            .true_negatives = true_negatives,
            .false_positives = false_positives,
            .false_negatives = false_negatives,
            .recall = recall,
            .accuracy = accuracy,
            .precision = precision,
            .log_loss = log_loss,
            .pseudo_r_squared = pseudo_r_squared,
            .constant =
            {
                .p_value = p_value,
                .odds_ratio = std::exp(constant),
                .coefficient = constant,
                .z_statistic = z_statistic,
                .standard_error = standard_error,
                .confidence_interval_95 =
                {
                    .upper_boundary = constant + z_critical_95 * standard_error,
                    .lower_boundary = constant - z_critical_95 * standard_error
                }
            },
            .independent_variables = independent_variables
        };
    }

    void LogisticRegression::Summary::print()
    {
        auto table = SummaryTable("Logistic Regression Summary");

        table.row_items.push_back({.name = "Observations", .is_float = false, .value_unsigned = observation_count });
        table.row_items.push_back({.name = "False Positives", .is_float = false, .value_unsigned = false_positives });
        table.row_items.push_back({.name = "Model Degrees Of Freedom", .is_float = false, .value_unsigned = model_degrees_of_freedom });
        table.row_items.push_back({.name = "False Negatives", .is_float = false, .value_unsigned = false_negatives });
        table.row_items.push_back({.name = "Residuals Degrees Of Freedom", .is_float = false, .value_unsigned = residuals_degrees_of_freedom });
        table.row_items.push_back({.name = "True Positives", .is_float = false, .value_unsigned = true_positives });
        table.row_items.push_back({.name = "Recall", .is_float = true, .value_float = recall });
        table.row_items.push_back({.name = "True Negatives", .is_float = false, .value_unsigned = true_negatives });
        table.row_items.push_back({.name = "Accuracy", .is_float = true, .value_float = accuracy });
        table.row_items.push_back({.name = "Log-Loss", .is_float = true, .value_float = log_loss });
        table.row_items.push_back({.name = "Precision", .is_float = true, .value_float = precision });
        table.row_items.push_back({.name = "Pseudo R-Squared", .is_float = true, .value_float = pseudo_r_squared });

        table.groups.resize(1);

        table.groups[0].title = "y = " + std::to_string(static_cast<int>(label));

        table.groups[0].column_headers.emplace_back("Coefficient");
        table.groups[0].column_headers.emplace_back("Odds Ratio");
        table.groups[0].column_headers.emplace_back("z Value");
        table.groups[0].column_headers.emplace_back("Pr > |z|");
        table.groups[0].column_headers.emplace_back("[0.025");
        table.groups[0].column_headers.emplace_back("0.975]");

        table.groups[0].variable_names.emplace_back("Constant");

        auto constant_values = std::vector<float>();

        constant_values.emplace_back(constant.coefficient);
        constant_values.emplace_back(constant.odds_ratio);
        constant_values.emplace_back(constant.z_statistic);
        constant_values.emplace_back(constant.p_value);
        constant_values.emplace_back(constant.confidence_interval_95.lower_boundary);
        constant_values.emplace_back(constant.confidence_interval_95.upper_boundary);

        table.groups[0].column_values.push_back(constant_values);

        for (int i = 0; i < independent_variables.size(); ++i)
        {
            table.groups[0].variable_names.emplace_back(independent_variables[i].variable_name);

            auto variable_values = std::vector<float>();

            variable_values.emplace_back(independent_variables[i].coefficient);
            variable_values.emplace_back(independent_variables[i].odds_ratio);
            variable_values.emplace_back(independent_variables[i].z_statistic);
            variable_values.emplace_back(independent_variables[i].p_value);
            variable_values.emplace_back(independent_variables[i].confidence_interval_95.lower_boundary);
            variable_values.emplace_back(independent_variables[i].confidence_interval_95.upper_boundary);

            table.groups[0].column_values.push_back(variable_values);
        }

        if (complete_separation)
        {
            table.warnings.emplace_back("WARNING: Complete Separation - Encountered complete separation. "
                "The Maximum Likelihood Estimator does not exist and the resulting parameters may not be interpreted.");
        }

        table.print();
    }
}

/*******************************************************************************************************************
 MULTI CLASS
 ******************************************************************************************************************/
namespace nml::logistic_regression_multi_internal
{
    typedef LogisticRegressionMulti::Request MultiRequest;
    typedef NMLResult<LogisticRegressionMulti> MultiResult;

    constexpr static bool test = false;

    struct LBFGSIteration
    {
        float learning_rate;
        float scaling_factor;
        VectorSpan solution_vector_differences;
        VectorSpan gradient_vector_differences;

        static inline uint64_t required_bytes(uint64_t length) noexcept
        {
            return 2 * VectorSpan::required_bytes(length);
        }
    };

    struct LBFGSWorkingMemory
    {
        VectorSpan labels;
        VectorSpan gradient;
        VectorSpan workspace;
        VectorSpan parameters;
        VectorSpan search_direction;
        VectorSpan gradient_previous;
        VectorSpan parameters_previous;
        Span<LBFGSIteration> retained_iterations;

        static inline uint64_t required_bytes(const MultiRequest& request) noexcept
        {
            uint64_t feature_ct = (request.labels.length - 1) * request.data.column_ct;

            return (6 + 2 * request.retained_iteration_ct) * VectorSpan::required_bytes(feature_ct)
                    + Span<LBFGSIteration>::required_bytes(request.retained_iteration_ct);
        }
    };

    struct LineSearchState
    {
        struct StageState
        {
            float loss;
            float step_size;
            float loss_midpoint;
            float gradient_derivative;
            float gradient_derivative_midpoint;
        };

        float& loss;
        float& step_size;
        int error_code = 0;
        bool is_stage_one = true;
        bool is_bracketed = false;
        float width, width_previous;
        StageState stage_one, stage_two;
        float step_size_min, step_size_max;
        float loss_initial, loss_midpoint, loss_threshold;
        float gradient_derivative, gradient_derivative_initial;
        float gradient_derivative_midpoint, gradient_derivative_threshold;

        explicit LineSearchState(const MultiRequest& request, float& loss, float& step_size, float gradient_derivative) noexcept
            : loss(loss), step_size(step_size), gradient_derivative_initial(gradient_derivative)
        {
            stage_one.step_size = stage_two.step_size = 0;
            stage_one.loss = stage_two.loss = loss_initial = loss;
            stage_one.gradient_derivative = stage_two.gradient_derivative = gradient_derivative_initial;
            gradient_derivative_threshold = request.line_search_accuracy_tolerance * gradient_derivative_initial;
            width = request.line_search_max_step_size - request.line_search_min_step_size; width_previous = 2.0f * width;
        }

        struct Moment
        {
            float& loss;
            float& loss_stage_one;
            float& loss_stage_two;
            float& gradient_derivative;
            float& gradient_derivative_stage_one;
            float& gradient_derivative_stage_two;
        };

        inline Moment get_moment_current() noexcept
        {
            return Moment
            {
                .loss = loss,
                .loss_stage_one = stage_one.loss,
                .loss_stage_two = stage_two.loss,
                .gradient_derivative = gradient_derivative,
                .gradient_derivative_stage_one = stage_one.gradient_derivative,
                .gradient_derivative_stage_two = stage_two.gradient_derivative,
            };
        }

        inline Moment get_moment_midpoint() noexcept
        {
            return Moment
            {
                .loss = loss_midpoint,
                .loss_stage_one = stage_one.loss_midpoint,
                .loss_stage_two = stage_two.loss_midpoint,
                .gradient_derivative = gradient_derivative_midpoint,
                .gradient_derivative_stage_one = stage_one.gradient_derivative_midpoint,
                .gradient_derivative_stage_two = stage_two.gradient_derivative_midpoint,
            };
        }
    };

    static inline LBFGSWorkingMemory construct_LBFGS_working_memory(
        const MultiRequest& request
        , RequestMemory memory
        , uint64_t& working_memory_offset
        , uint64_t& result_memory_offset
    ) noexcept
    {
        auto labels = VectorSpan(memory.working_memory.offset(result_memory_offset), request.labels.length);

        result_memory_offset += labels.bytes();

        uint64_t feature_ct = (request.labels.length) * request.data.column_ct;

        auto parameters = VectorSpan(memory.working_memory.offset(working_memory_offset), feature_ct);

//        parameters.fill_random_gaussian(0.25);

        working_memory_offset += parameters.bytes();

        auto parameters_previous = VectorSpan(memory.working_memory.offset(working_memory_offset), feature_ct);

        working_memory_offset += parameters_previous.bytes();

        auto gradient = VectorSpan(memory.working_memory.offset(working_memory_offset),feature_ct);

        working_memory_offset += gradient.bytes();

        auto gradient_previous = VectorSpan(memory.working_memory.offset(working_memory_offset), feature_ct);

        working_memory_offset += gradient_previous.bytes();

        auto search_direction = VectorSpan(memory.working_memory.offset(working_memory_offset),feature_ct);

        working_memory_offset += search_direction.bytes();

        auto workspace = VectorSpan(memory.working_memory.offset(working_memory_offset),feature_ct);

        working_memory_offset += workspace.bytes();

        auto retained_iterations = Span<LBFGSIteration>(
            memory.working_memory.offset(working_memory_offset), request.retained_iteration_ct
        );

        working_memory_offset += retained_iterations.bytes();

        for (uint32_t i = 0; i < retained_iterations.length; ++i)
        {
            retained_iterations[i].learning_rate = 0;
            retained_iterations[i].scaling_factor = 0;

            retained_iterations[i].solution_vector_differences = VectorSpan(
                memory.working_memory.offset(working_memory_offset),feature_ct
            );

            working_memory_offset += retained_iterations[i].solution_vector_differences.bytes();

            retained_iterations[i].gradient_vector_differences = VectorSpan(
                memory.working_memory.offset(working_memory_offset),feature_ct
            );

            working_memory_offset += retained_iterations[i].gradient_vector_differences.bytes();
        }

        return LBFGSWorkingMemory
        {
            .labels = labels,
            .gradient = gradient,
            .workspace = workspace,
            .parameters = parameters,
            .search_direction = search_direction,
            .gradient_previous = gradient_previous,
            .parameters_previous = parameters_previous,
            .retained_iterations = retained_iterations,
        };
    }

    static inline float evaluate_loss(const MultiRequest& request, LBFGSWorkingMemory& wm) noexcept
    {
        wm.gradient.zero();

        float loss = 0, row_ct = static_cast<float>(request.data.row_ct);

        for (uint32_t row_offset = 0; row_offset < request.data.row_ct; ++row_offset)
        {
            VectorSpan row = request.data[row_offset];

            float max_score = -std::numeric_limits<float>::infinity(), sum = 0;

            for (uint32_t class_offset = 0; class_offset < request.labels.length; ++class_offset)
            {
                uint32_t parameter_offset = class_offset * row.length;

                wm.workspace[class_offset] = wm.parameters[parameter_offset];

                for (uint32_t feature = 1; feature < row.length; ++feature)
                {
                    wm.workspace[class_offset] += wm.parameters[parameter_offset + feature] * row[feature];
                }

                max_score = std::max(wm.workspace[class_offset], max_score);
            }

            for (uint32_t class_offset = 0; class_offset < request.labels.length; ++class_offset)
            {
                wm.workspace[class_offset] = std::exp(wm.workspace[class_offset] - max_score);

                sum += wm.workspace[class_offset];
            }

            for (uint32_t class_offset = 0; class_offset < request.labels.length; ++class_offset)
            {
                float probability = wm.workspace[class_offset] / sum;

                bool is_current_class = row[0] == request.labels[class_offset];

                float error = (probability - (is_current_class ? 1.0f : 0.0f)) / row_ct;

                if (is_current_class)
                {
                    loss -= std::log(probability);
                }

                uint32_t parameter_offset = class_offset * row.length;

                wm.gradient[parameter_offset] += error;

                for (uint32_t feature = 1; feature < row.length; ++feature)
                {
                    wm.gradient[parameter_offset + feature] += error * row[feature];
                }
            }
        }

        return loss;
    }

    static inline float evaluate_loss_test(const MultiRequest& request, LBFGSWorkingMemory& wm) noexcept
    {
        int i;
        float fx = 0.0;

        for (i = 0; i < wm.parameters.length; i += 2)
        {
            float t1 = 1.0 - wm.parameters[i];
            float t2 = 10.0 * (wm.parameters[i + 1] - wm.parameters[i] * wm.parameters[i]);
            wm.gradient[i + 1] = 20.0 * t2;
            wm.gradient[i] = -2.0 * (wm.parameters[i] * wm.gradient[i+1] + t1);
            fx += t1 * t1 + t2 * t2;
        }

        return fx;
    }

    static inline float cubic_minimizer(float u, float fu, float du, float v, float fv, float dv) noexcept
    {
        float d = v - u;
        float theta = (fu - fv) * 3 / d + du + dv;
        float p = std::fabs(theta);
        float q = std::fabs(du);
        float r = std::fabs(dv);
        float s = std::max(std::max(p, q), r);
        float a = theta / s;
        float gamma = s * std::sqrt(a * a - (du / s) * (dv / s));
        if (v < u) gamma = -gamma;
        p = gamma - du + theta;
        q = gamma - du + gamma + dv;
        r = p / q;

        return u + r * d;
    }

    static inline float cubic_minimizer_bound(float u, float fu, float du, float v, float fv, float dv, float min_boundary, float max_boundary) noexcept
    {
        float d = v - u;
        float theta = (fu - fv) * 3 / d + du + dv;
        float p = std::fabs(theta);
        float q = std::fabs(du);
        float r = std::fabs(dv);
        float s = std::max(std::max(p, q), r);
        float a = theta / s;
        float gamma = s * std::sqrt(std::max(0.0f, a * a - (du / s) * (dv / s)));
        if (u < v) gamma = -gamma;
        p = gamma - dv + theta;
        q = gamma - dv + gamma + du;
        float ratio = p / q;

        if (ratio < 0.0 && gamma != 0.0)
        {
            return v - ratio * d;
        }
        else if (d > 0)
        {
            return max_boundary;
        }
        else
        {
            return min_boundary;
        }
    }

    static inline float quad_minimizer(float u, float fu, float du, float v, float fv) noexcept
    {
        float a = v - u;

        return u + du / ((fu - fv) / a + du) / 2 * a;
    }

    static inline float quad_minimizer_derivatives(float u, float du, float v, float dv) noexcept
    {
        float a = u - v;

        return v + (dv / (dv - du)) * a;
    }

    static inline int update_trial_interval(LineSearchState& state, LineSearchState::Moment moment) noexcept
    {
        bool bound;
        bool gradient_sign = 0 > moment.gradient_derivative
            * (moment.gradient_derivative_stage_one / std::fabs(moment.gradient_derivative_stage_one));

        float minimizer_cubic, minimizer_quadratic, step_size_trial;

        if (state.is_bracketed)
        {
            if (state.step_size <= std::min(state.stage_one.step_size, state.stage_two.step_size)
                || state.step_size >= std::max(state.stage_one.step_size, state.stage_two.step_size))
            {
                return -(int32_t)NMLErrorCode::INVALID_REQUEST;
            }

            if (0 <= moment.gradient_derivative_stage_one * (state.step_size - state.stage_one.step_size))
            {
                return -(int32_t)NMLErrorCode::INVALID_REQUEST;
            }

            if (state.step_size_max < state.step_size_min) // TODO remove
            {
                return -(int32_t)NMLErrorCode::INVALID_REQUEST;
            }
        }

        if (moment.loss_stage_one < moment.loss)
        {
            state.is_bracketed = true, bound = true;

            minimizer_cubic = cubic_minimizer(
                state.stage_one.step_size
                , moment.loss_stage_one
                , moment.gradient_derivative_stage_one
                , state.step_size
                , moment.loss
                , moment.gradient_derivative
            );

            minimizer_quadratic = quad_minimizer(
                state.stage_one.step_size
                , moment.loss_stage_one
                , moment.gradient_derivative_stage_one
                , state.step_size
                , moment.loss
            );

            float cubic_magnitude = std::fabs(state.stage_one.step_size - minimizer_cubic);
            float quadratic_magnitude = std::fabs(state.stage_one.step_size - minimizer_quadratic);

            step_size_trial = cubic_magnitude < quadratic_magnitude ?
                              minimizer_cubic : minimizer_cubic + 0.5f * (minimizer_quadratic - minimizer_cubic);
        }
        else if (gradient_sign)
        {
            state.is_bracketed = true, bound = false;

            minimizer_cubic = cubic_minimizer(
                state.stage_one.step_size
                , moment.loss_stage_one
                , moment.gradient_derivative_stage_one
                , state.step_size
                , moment.loss
                , moment.gradient_derivative
            );

            minimizer_quadratic = quad_minimizer_derivatives(
                state.stage_one.step_size
                , moment.gradient_derivative_stage_one
                , state.step_size
                , moment.gradient_derivative
            );

            float cubic_magnitude = std::fabs(state.step_size - minimizer_cubic);
            float quadratic_magnitude = std::fabs(state.step_size - minimizer_quadratic);

            step_size_trial = cubic_magnitude > quadratic_magnitude ? minimizer_cubic : minimizer_quadratic;
        }
        else if (std::fabs(moment.gradient_derivative) < std::fabs(moment.gradient_derivative_stage_one))
        {
            bound = true;

            minimizer_cubic = cubic_minimizer_bound(
                state.stage_one.step_size
                , moment.loss_stage_one
                , moment.gradient_derivative_stage_one
                , state.step_size
                , moment.loss
                , moment.gradient_derivative
                , state.step_size_min
                , state.step_size_max
            );

            minimizer_quadratic = quad_minimizer_derivatives(
                state.stage_one.step_size
                , moment.gradient_derivative_stage_one
                , state.step_size
                , moment.gradient_derivative
            );

            float cubic_magnitude = std::fabs(state.step_size - minimizer_cubic);
            float quadratic_magnitude = std::fabs(state.step_size - minimizer_quadratic);

            if (state.is_bracketed)
            {
                step_size_trial = cubic_magnitude < quadratic_magnitude ? minimizer_cubic : minimizer_quadratic;
            }
            else
            {
                step_size_trial = cubic_magnitude > quadratic_magnitude ? minimizer_cubic : minimizer_quadratic;
            }
        }
        else
        {
            bound = false;

            if (state.is_bracketed)
            {
                step_size_trial = cubic_minimizer(
                    state.step_size
                    , moment.loss
                    , moment.gradient_derivative
                    , state.stage_two.step_size
                    , moment.loss_stage_two
                    , moment.gradient_derivative_stage_two
                );
            }
            else if (state.stage_one.step_size < state.step_size)
            {
                step_size_trial = state.step_size_max;
            }
            else
            {
                step_size_trial = state.step_size_min;
            }
        }

        if (moment.loss_stage_one < moment.loss)
        {
            state.stage_two.step_size = state.step_size;
            moment.loss_stage_two = moment.loss;
            moment.gradient_derivative_stage_two = moment.gradient_derivative;
        }
        else
        {
            if (gradient_sign)
            {
                state.stage_two.step_size = state.stage_one.step_size;
                moment.loss_stage_two = moment.loss_stage_one;
                moment.gradient_derivative_stage_two = moment.gradient_derivative_stage_one;
            }

            state.stage_one.step_size = state.step_size;
            moment.loss_stage_one = moment.loss;
            moment.gradient_derivative_stage_one = moment.gradient_derivative;
        }

        if (state.step_size_max < step_size_trial) step_size_trial = state.step_size_max;
        if (step_size_trial < state.step_size_min) step_size_trial = state.step_size_min;

        if (state.is_bracketed && bound)
        {
            minimizer_quadratic = state.stage_one.step_size + 0.66f * (state.stage_two.step_size - state.stage_one.step_size);

            step_size_trial = state.stage_one.step_size < state.stage_two.step_size ?
                    std::min(step_size_trial, minimizer_quadratic) : std::max(step_size_trial, minimizer_quadratic);
        }

        state.step_size = step_size_trial;

        return 0;
    }

    static inline int32_t line_search(const LogisticRegressionMulti::Request& request, LBFGSWorkingMemory& wm, float& loss, float& step) noexcept
    {
        uint16_t iteration = 0;

        if (step <= 0)
        {
            return -(int32_t)NMLErrorCode::INVALID_REQUEST;
        }

        const float gradient_derivative = wm.gradient.dot_product_unsafe(wm.search_direction);

        if (gradient_derivative > 0)
        {
            return -(int32_t)NMLErrorCode::INVALID_REQUEST;
        }

        LineSearchState state = LineSearchState(request, loss, step, gradient_derivative);

        while (true)
        {
            if (state.is_bracketed)
            {
                state.step_size_min = std::min(state.stage_one.step_size, state.stage_two.step_size);
                state.step_size_max = std::max(state.stage_one.step_size, state.stage_two.step_size);
            }
            else
            {
                state.step_size_min = state.stage_one.step_size;
                state.step_size_max = step + 4.0f * (step - state.stage_one.step_size);
            }

            step = std::max(step, request.line_search_min_step_size);
            step = std::min(step, request.line_search_max_step_size);

            if ((state.is_bracketed && (
                    (step <= state.step_size_min || state.step_size_max <= step)
                    || request.line_search_max_iteration_ct <= iteration + 1 || state.error_code != 0
                ))
                || (state.is_bracketed && (state.step_size_max - state.step_size_min <= request.rounding_tolerance * state.step_size_max)))
            {
                step = state.stage_one.step_size;
            }

            for (uint64_t i = 0; i < wm.parameters.length; ++i)
            {
                wm.parameters[i] = step * wm.search_direction[i] + wm.parameters_previous[i];
            }

            state.loss = test ? evaluate_loss_test(request, wm) : evaluate_loss(request, wm);

            state.gradient_derivative = wm.gradient.dot_product_unsafe(wm.search_direction);

            state.loss_threshold = state.loss_initial + step * state.gradient_derivative_threshold;

            iteration += 1;

            if (state.is_bracketed && ((step <= state.step_size_min || state.step_size_max <= step) || state.error_code != 0))
            {
                return -(int32_t)NMLErrorCode::ROUNDING_ERROR;
            }

            if (step == request.line_search_max_step_size
                && state.loss <= state.loss_threshold
                && state.gradient_derivative <= state.gradient_derivative_threshold)
            {
                return -(int32_t)NMLErrorCode::UNABLE_TO_CONVERGE;
            }

            if (step == request.line_search_min_step_size && (state.loss_threshold < step
                || state.gradient_derivative_threshold <= state.gradient_derivative))
            {
                return -(int32_t)NMLErrorCode::UNABLE_TO_CONVERGE;
            }

            if (state.is_bracketed && (state.step_size_max - state.step_size_min) <= request.rounding_tolerance * state.step_size_max)
            {
                return -(int32_t)NMLErrorCode::UNABLE_TO_CONVERGE;
            }

            if (iteration >= request.line_search_max_iteration_ct)
            {
                return -(int32_t)NMLErrorCode::MAXIMUM_ITERATIONS;
            }

            if (state.loss <= state.loss_threshold
                && std::fabs(state.gradient_derivative) <= request.line_search_gradient_tolerance * (-state.gradient_derivative_initial))
            {
                return iteration;
            }

            if (state.is_stage_one
                && state.loss <= state.loss_threshold
                && state.gradient_derivative >= std::min(request.line_search_accuracy_tolerance, request.line_search_gradient_tolerance) * state.gradient_derivative_initial)
            {
                state.is_stage_one = false;
            }
            
            if (state.is_stage_one && state.loss_threshold < state.loss && state.loss <= state.stage_one.loss)
            {
                state.loss_midpoint = state.loss - step * state.gradient_derivative_threshold;
                state.stage_one.loss_midpoint = state.stage_one.loss - state.stage_one.step_size * state.gradient_derivative_threshold;
                state.stage_two.loss_midpoint = state.stage_two.loss - state.stage_two.step_size * state.gradient_derivative_threshold;

                state.gradient_derivative_midpoint = state.gradient_derivative - state.gradient_derivative_threshold;
                state.stage_one.gradient_derivative_midpoint = state.stage_one.gradient_derivative - state.gradient_derivative_threshold;
                state.stage_two.gradient_derivative_midpoint = state.stage_two.gradient_derivative - state.gradient_derivative_threshold;

                state.error_code = update_trial_interval(state, state.get_moment_midpoint());

                state.stage_one.loss = state.stage_one.loss + state.stage_one.step_size * state.gradient_derivative_threshold;
                state.stage_two.loss = state.stage_two.loss + state.stage_two.step_size * state.gradient_derivative_threshold;
                state.stage_one.gradient_derivative = state.stage_one.gradient_derivative_midpoint + state.gradient_derivative_threshold;
                state.stage_two.gradient_derivative = state.stage_two.gradient_derivative_midpoint + state.gradient_derivative_threshold;
            }
            else
            {
                state.error_code = update_trial_interval(state, state.get_moment_current());
            }

            if (state.is_bracketed)
            {
                if (0.66 * state.width_previous <= std::fabs(state.stage_two.step_size - state.stage_one.step_size))
                {
                    step = state.stage_one.step_size + 0.5f * (state.stage_two.step_size - state.stage_one.step_size);
                }

                state.width_previous = state.width;
                state.width = std::fabs(state.stage_two.step_size - state.stage_one.step_size);
            }
        }
    }
}

namespace nml
{
    void LogisticRegressionMulti::predict(const VectorSpan& observation, Heap<ScoredValue<unsigned>>& probabilities) const noexcept
    {
        probabilities.reset();

        float max_score = -std::numeric_limits<float>::infinity(), sum = 0;

        float* scores = static_cast<float*>(alloca(labels.length * sizeof(float)));

        for (uint32_t class_offset = 0; class_offset < labels.length; ++class_offset)
        {
            scores[class_offset] = constants[class_offset];

            for (uint32_t i = 0; i < observation.length; ++i)
            {
                scores[class_offset] += observation[i] * variables[class_offset][i];
            }

            max_score = std::max(scores[class_offset], max_score);
        }

        for (uint32_t class_offset = 0; class_offset < labels.length; ++class_offset)
        {
            scores[class_offset] = std::exp(scores[class_offset] - max_score);

            sum += scores[class_offset];
        }

        for (uint32_t class_offset = 0; class_offset < labels.length; ++class_offset)
        {
            float probability = scores[class_offset] / sum;

            probabilities.push({ .score = probability, .value = class_offset });
        }
    }

    RequiredMemory LogisticRegressionMulti::required_memory(const Request& request) noexcept
    {
        using namespace nml::logistic_regression_multi_internal;

        uint64_t feature_ct = (request.labels.length) * request.data.column_ct;

        uint64_t vector_size = VectorSpan::required_bytes(feature_ct);

        uint64_t retained_iteration_size = Span<LBFGSIteration>::required_bytes(request.retained_iteration_ct);

        uint64_t labels_bytes = VectorSpan::required_bytes(request.labels.length);

        uint64_t constants_bytes = VectorSpan::required_bytes(request.labels.length);

        uint64_t variables_span_bytes = request.labels.length * sizeof(VectorSpan);

        uint64_t variables_bytes = request.labels.length * VectorSpan::required_bytes(request.data.column_ct);

        return
        {
            .result_required_bytes = labels_bytes + constants_bytes + variables_bytes + variables_span_bytes,
            .working_required_bytes = (6 + 2 * request.retained_iteration_ct) * vector_size + retained_iteration_size,
        };
    }

    NMLResult<LogisticRegressionMulti> LogisticRegressionMulti::compute(const Request& request, RequestMemory memory) noexcept
    {
        using namespace nml::logistic_regression_multi_internal;

        if (!memory.is_sufficient(required_memory(request)))
        {
            return MultiResult::err(NMLErrorCode::INSUFFICIENT_MEMORY);
        }

        memory.working_memory.zero();

        uint64_t result_memory_offset = 0, working_memory_offset = 0;

        LBFGSWorkingMemory wm = construct_LBFGS_working_memory(
            request, memory, working_memory_offset, result_memory_offset
        );

        float loss = test ? evaluate_loss_test(request, wm) : evaluate_loss(request, wm);

        uint32_t iteration = 1, end = 0, max_look_back = 0;

        for (uint64_t i = 0; i < wm.gradient.length; ++i)
        {
            wm.search_direction[i] = -wm.gradient[i];
        }

        float step = 1.0f / l2_norm(wm.search_direction);
        float gradients_normalized = l2_norm(wm.gradient);
        float parameters_normalized = l2_norm(wm.parameters);

        if (parameters_normalized < 1.0) parameters_normalized = 1.0;

        while (loss > 0)
        {
            wm.gradient_previous.copy_from_unsafe(wm.gradient);
            wm.parameters_previous.copy_from_unsafe(wm.parameters);

            int32_t line_search_result = line_search(request, wm, loss, step);

            if (line_search_result < 0)
            {
                return MultiResult::err((NMLErrorCode)(-line_search_result));
            }

            gradients_normalized = l2_norm(wm.gradient);
            parameters_normalized = l2_norm(wm.parameters);

            if (gradients_normalized / parameters_normalized <= request.convergence_threshold)
            {
                break;
            }

            if (iteration + 1 > request.max_iteration_ct)
            {
                return MultiResult::err(NMLErrorCode::MAXIMUM_ITERATIONS);
            }

            LBFGSIteration& current_iteration = wm.retained_iterations[end];

            float gradient_scale = 0;
            current_iteration.scaling_factor = 0;

            for (uint32_t i = 0; i < wm.parameters.length; ++i)
            {
                current_iteration.gradient_vector_differences[i] = wm.gradient[i] - wm.gradient_previous[i];
                current_iteration.solution_vector_differences[i] = wm.parameters[i] - wm.parameters_previous[i];

                gradient_scale += current_iteration.gradient_vector_differences[i] * current_iteration.gradient_vector_differences[i];
                current_iteration.scaling_factor += current_iteration.gradient_vector_differences[i] * current_iteration.solution_vector_differences[i];
            }

            gradient_scale = current_iteration.scaling_factor / gradient_scale;

            max_look_back = (request.retained_iteration_ct <= iteration) ? request.retained_iteration_ct : iteration;

            iteration += 1;

            end = (end + 1) % request.retained_iteration_ct;

            for (uint64_t i = 0; i < wm.parameters.length; ++i)
            {
                wm.search_direction[i] = -wm.gradient[i];
            }

            uint32_t previous = end;

            for (uint32_t look_back = 0; look_back < max_look_back; ++look_back)
            {
                previous = (previous + request.retained_iteration_ct - 1) % request.retained_iteration_ct;

                LBFGSIteration& previous_iteration = wm.retained_iterations[previous];

                previous_iteration.learning_rate = previous_iteration.solution_vector_differences.dot_product_unsafe(wm.search_direction);

                previous_iteration.learning_rate /= previous_iteration.scaling_factor;

                for (uint32_t i = 0; i < wm.parameters.length; ++i)
                {
                    wm.search_direction[i] -= previous_iteration.gradient_vector_differences[i] * previous_iteration.learning_rate;
                }
            }

            for (uint32_t i = 0; i < wm.parameters.length; ++i)
            {
                wm.search_direction[i] *= gradient_scale;
            }

            for (uint32_t look_back = 0; look_back < max_look_back; ++look_back)
            {
                LBFGSIteration& previous_iteration = wm.retained_iterations[previous];

                float beta = 0;

                for (uint32_t i = 0; i < wm.parameters.length; ++i)
                {
                    beta += previous_iteration.gradient_vector_differences[i] * wm.search_direction[i];
                }

                beta = previous_iteration.learning_rate - (beta / previous_iteration.scaling_factor);

                for (uint32_t i = 0; i < wm.parameters.length; ++i)
                {
                    wm.search_direction[i] += previous_iteration.solution_vector_differences[i] * beta;
                }

                previous = (previous + 1) % request.retained_iteration_ct;
            }

            step = 1.0;
        }

        auto labels = VectorSpan(memory.result_memory.offset(result_memory_offset), request.labels.length);

        result_memory_offset += labels.bytes();

        auto constants = VectorSpan(memory.result_memory.offset(result_memory_offset), request.labels.length);

        result_memory_offset += constants.bytes();

        auto variables = Span<VectorSpan>(memory.result_memory.offset(result_memory_offset), request.labels.length);

        result_memory_offset += variables.bytes();

        for (uint32_t i = 0; i < variables.length; ++i)
        {
            variables[i] = VectorSpan(memory.result_memory.offset(result_memory_offset), request.data.column_ct - 1);

            result_memory_offset += variables[i].bytes();
        }

        labels.copy_from_unsafe(request.labels);

        uint32_t parameter_ct = 0;

        for (uint32_t label = 0; label < labels.length; ++label)
        {
            constants[label] = wm.parameters[parameter_ct++];

            VectorSpan variable = variables[label];

            for (uint32_t i = 0; i < variable.length; ++i)
            {
                variable[i] = wm.parameters[parameter_ct++];
            }
        }

        return NMLResult<LogisticRegressionMulti>::ok({
            .labels = labels,
            .constants = constants,
            .variables = variables
        });
    }

    struct LogisticRegressionMulti::Summary
    {
        unsigned observation_count;
        unsigned model_degrees_of_freedom;
        unsigned residuals_degrees_of_freedom;

        float recall;
        float accuracy;
        float precision;

        float log_loss;
        float pseudo_r_squared;

        struct ConfidenceInterval
        {
            float upper_boundary;
            float lower_boundary;
        };

        struct Coefficient
        {
            float p_value;
            float odds_ratio;
            float coefficient;
            float z_statistic;
            float standard_error;
            std::string variable_name;

            ConfidenceInterval confidence_interval_95;
        };

        struct Classifier
        {
            float label;
            Coefficient constant;
            std::vector<Coefficient> independent_variables;
        };

        std::vector<Classifier> classifiers;

        void print();
    };

    LogisticRegressionMulti::Summary LogisticRegressionMulti::summary(const MatrixSpan& data) const noexcept
    {
        // TODO
        auto row_ct_float = static_cast<float>(data.row_ct);

        auto model_degrees_of_freedom = static_cast<uint32_t>(variables[0].length * (labels.length - 1));
        auto residuals_degrees_of_freedom = static_cast<uint32_t>(data.row_ct - model_degrees_of_freedom - (labels.length - 1));

        float log_loss = 0, log_loss_null = 0;

        unsigned correct = 0;

        uint64_t memory_size = 4 * Span<uint32_t>::required_bytes(labels.length)
            + Heap<ScoredValue<uint32_t>>::required_bytes(labels.length);

        MemorySpan heap_memory = MemorySpan(alloca(memory_size), memory_size);

        uint32_t offset = 0;

        auto probabilities = Heap<ScoredValue<uint32_t>>(heap_memory);

        offset += probabilities.bytes();

        auto frequencies = HashMap<float, uint32_t>();

        for (uint32_t i = 0; i < labels.length; ++i)
        {
            frequencies.insert(labels[i], 0);
        }

        for (unsigned r = 0; r < data.row_ct; ++r)
        {
            auto row = data[r];

            predict(row.to_subspan_unsafe(1), probabilities);

            correct += labels[probabilities.peek().value] == row[0];

            while (!probabilities.is_empty())
            {
                auto next = probabilities.pop();

                if (labels[next.value] == row[0])
                {
                    log_loss -= std::log(next.score);
                    break;
                }
            }

            frequencies.get_value_unsafe(row[0])++;
        }

        for (uint32_t i = 0; i < labels.length; ++i)
        {
            uint32_t frequency = frequencies.get_value_unsafe(labels[i]);

            log_loss_null -= frequency * std::log(frequency / row_ct_float);
        }

        auto correct_float = static_cast<float>(correct);

        float accuracy = correct_float / row_ct_float;

//        log_loss /= row_ct_float;
//        log_loss_null /= row_ct_float;
        float pseudo_r_squared = 1 - log_loss / log_loss_null;

        auto classifiers = std::vector<Summary::Classifier>();

        for (int class_offset = 0; class_offset < labels.length; ++class_offset)
        {
            auto independent_variables = std::vector<Summary::Coefficient>();

            offset = 0;
            auto memory = MemoryOwner(MatrixSpan::required_bytes(data.column_ct, data.column_ct * 2) + Span<unsigned>::required_bytes(data.column_ct));
            auto memory_span = memory.to_memory_span();
            auto fisher_weights_inverse = MatrixSpan(memory_span, data.column_ct, data.column_ct * 2);

            auto y = data.to_subspan_unsafe(data.row_ct, 1);

            auto fisher_weights = fisher_weights_inverse.to_subspan_unsafe(data.column_ct, data.column_ct);
            auto fisher_weights_inverse_right = fisher_weights_inverse.to_subspan_unsafe(data.column_ct, data.column_ct, 0, data.column_ct);

            auto pivots = Span<unsigned>(memory_span.offset(fisher_weights_inverse.bytes()), data.column_ct);

            fisher_weights_inverse.fill(0);

            for (unsigned i = 0; i < data.row_ct; ++i)
            {
                auto row = data[i];

//                float predicted = predict(row.to_subspan_unsafe(1));
//                float fisher_weight = predicted * (1 - predicted);
//
//                for (unsigned j = 0; j < data.column_ct; ++j)
//                {
//                    float data_t = j == 0 ? 1.0f : row[j];
//                    auto fisher_row = fisher_weights[j];
//
//                    for (unsigned k = 0; k < data.column_ct; ++k)
//                    {
//                        float data_m = k == 0 ? 1.0f : row[k];
//                        fisher_row[k] += fisher_weight * data_t * data_m;
//                    }
//                }
            }

            fisher_weights_inverse_right.set_identity();

            gauss_jordan_elimination(fisher_weights_inverse);

            auto z_critical_95 = static_cast<float>(Distribution::normal_pdf(0.975));

            for (int i = 0; i < variables[class_offset].length; ++i)
            {
                float standard_error = std::sqrt(std::fabs(fisher_weights_inverse_right[i + 1][i + 1]));
                float z_statistic = variables[class_offset][i] / standard_error;
                auto p_value = static_cast<float>(2 * Distribution::normal_cdf(std::fabs(z_statistic), 0, 1, false));

                independent_variables.push_back({
                    .p_value = p_value,
                    .odds_ratio = 0,
                    .coefficient = variables[class_offset][i],
                    .z_statistic = z_statistic,
                    .standard_error = standard_error,
                    .variable_name = "Variable " + std::to_string(i + 1),
                    .confidence_interval_95 =
                    {
                        .upper_boundary = variables[class_offset][i] + z_critical_95 * standard_error,
                        .lower_boundary = variables[class_offset][i] - z_critical_95 * standard_error
                    }
                });
            }

            float standard_error = std::sqrt(std::fabs(fisher_weights_inverse_right[0][0]));
            float z_statistic = constants[class_offset] / standard_error;
            auto p_value = static_cast<float>(2 * Distribution::normal_cdf(std::fabs(z_statistic), 0, 1, false));

            auto classifier = Summary::Classifier
            {
                .label = labels[class_offset],
                .constant =
                {
                    .p_value = p_value,
                    .odds_ratio = 0,
                    .coefficient = constants[class_offset],
                    .z_statistic = z_statistic,
                    .standard_error = standard_error,
                    .variable_name = "Constant",
                    .confidence_interval_95 = {
                        .upper_boundary = constants[class_offset] + z_critical_95 * standard_error,
                        .lower_boundary = constants[class_offset] - z_critical_95 * standard_error
                    }
                },
                .independent_variables = independent_variables
            };

            classifiers.push_back(classifier);
        }

        return LogisticRegressionMulti::Summary
        {
            .observation_count = data.row_ct,
            .model_degrees_of_freedom = model_degrees_of_freedom,
            .residuals_degrees_of_freedom = residuals_degrees_of_freedom,
            .recall = 0,
            .accuracy = accuracy,
            .precision = 0,
            .log_loss = log_loss,
            .pseudo_r_squared = pseudo_r_squared,
            .classifiers = classifiers
        };
    }

    void LogisticRegressionMulti::Summary::print()
    {
        auto table = SummaryTable("Multinomial Logistic Regression Summary");

        table.row_items.push_back({.name = "Observations", .is_float = false, .value_unsigned = observation_count });
        table.row_items.push_back({.name = "Accuracy", .is_float = true, .value_float = accuracy });
        table.row_items.push_back({.name = "Degrees Of Freedom - Model", .is_float = false, .value_unsigned = model_degrees_of_freedom });
        table.row_items.push_back({.name = "Log-Loss", .is_float = true, .value_float = log_loss });
        table.row_items.push_back({.name = "Degrees Of Freedom - Residuals", .is_float = false, .value_unsigned = residuals_degrees_of_freedom });
        table.row_items.push_back({.name = "Pseudo R-Squared", .is_float = true, .value_float = pseudo_r_squared });
        table.row_items.push_back({.name = "Recall (Weighted-Average)", .is_float = true, .value_float = recall });
        table.row_items.push_back({.name = "Precision (Weighted-Average)", .is_float = true, .value_float = precision });

        table.groups.resize(classifiers.size());

        for (int g = 0; g < classifiers.size(); ++g)
        {
            table.groups[g].title = "y = " + std::to_string(static_cast<int>(classifiers[g].label));

            table.groups[g].column_headers.emplace_back("Coefficient");
            table.groups[g].column_headers.emplace_back("Odds Ratio");
            table.groups[g].column_headers.emplace_back("z Value");
            table.groups[g].column_headers.emplace_back("Pr > |z|");
            table.groups[g].column_headers.emplace_back("[0.025");
            table.groups[g].column_headers.emplace_back("0.975]");

            table.groups[g].variable_names.emplace_back("Constant");

            auto constant_values = std::vector<float>();

            constant_values.emplace_back(classifiers[g].constant.coefficient);
            constant_values.emplace_back(classifiers[g].constant.odds_ratio);
            constant_values.emplace_back(classifiers[g].constant.z_statistic);
            constant_values.emplace_back(classifiers[g].constant.p_value);
            constant_values.emplace_back(classifiers[g].constant.confidence_interval_95.lower_boundary);
            constant_values.emplace_back(classifiers[g].constant.confidence_interval_95.upper_boundary);

            table.groups[g].column_values.push_back(constant_values);

            for (int i = 0; i < classifiers[g].independent_variables.size(); ++i)
            {
                table.groups[g].variable_names.emplace_back(classifiers[g].independent_variables[i].variable_name);

                auto variable_values = std::vector<float>();

                variable_values.emplace_back(classifiers[g].independent_variables[i].coefficient);
                variable_values.emplace_back(classifiers[g].independent_variables[i].odds_ratio);
                variable_values.emplace_back(classifiers[g].independent_variables[i].z_statistic);
                variable_values.emplace_back(classifiers[g].independent_variables[i].p_value);
                variable_values.emplace_back(classifiers[g].independent_variables[i].confidence_interval_95.lower_boundary);
                variable_values.emplace_back(classifiers[g].independent_variables[i].confidence_interval_95.upper_boundary);

                table.groups[g].column_values.push_back(variable_values);
            }
        }

        table.print();
    }
}

#endif //NML_LOGISTIC_REGRESSION_H