//
// Created by nik on 5/8/2024.
//

#ifndef NML_ORDINARY_LEAST_SQUARES_H
#define NML_ORDINARY_LEAST_SQUARES_H

#include "qr_algorithm.h"
#include "gauss_jordan_elimination.h"
#include "distribution_functions.h"

#include "../primitives/regression_summary.h"

namespace nml
{
    struct OLS
    {
        struct Summary;

        float constant;
        VectorSpan variables;

        struct Request
        {
            const MatrixSpan &data;

            explicit Request(const MatrixSpan& data)
                : data(data)
            { }
        };

        [[nodiscard]] Summary summary(const MatrixSpan& data) const noexcept;
        [[nodiscard]] float estimate(const VectorSpan& observation) const noexcept;

        static RequiredMemory required_memory(const Request& request) noexcept;
        static NMLResult<OLS> compute(const Request& request, RequestMemory memory) noexcept;
    };
}

namespace nml::ordinary_least_squares_internal
{
    static inline MatrixSpan get_working_matrix(const OLS::Request& request, MemorySpan& memory, unsigned& memory_offset) noexcept
    {
        auto working_matrix = MatrixSpan(memory.offset(memory_offset), request.data.row_ct, request.data.column_ct);

        memory_offset += working_matrix.bytes();

        for (unsigned row = 0; row < request.data.row_ct; ++row)
        {
            auto request_row = request.data.to_vector_subspan_unsafe(request.data.column_ct - 1, row, 1);
            auto working_matrix_row = working_matrix[row];

            working_matrix_row[0] = 1;

            working_matrix_row.to_subspan_unsafe(1).copy_from_unsafe(request_row);
        }

        return working_matrix;
    }

    static inline QRDecomposition qr_decomposition(MatrixSpan& working_matrix, MemorySpan& memory, unsigned& memory_offset) noexcept
    {
        auto q = MatrixSpan(memory.offset(memory_offset), working_matrix.row_ct, working_matrix.column_ct);

        q.set_identity();

        memory_offset += q.bytes();

        auto householder_vector = VectorSpan(memory.offset(memory_offset), working_matrix.row_ct);

        memory_offset += householder_vector.bytes();

        auto tau = VectorSpan(memory.offset(memory_offset), working_matrix.column_ct);

        memory_offset += tau.bytes();

        auto qr_decomposition = QRDecomposition::qr_partial_householder(q, working_matrix, householder_vector, tau, true);

        memory_offset -= householder_vector.bytes() + tau.bytes();

        return qr_decomposition.ok();
    }

    static inline VectorSpan solve_for_coefficients(MatrixSpan& r, MatrixSpan& projected_y, MemorySpan& memory, unsigned& memory_offset) noexcept
    {
        auto beta = MatrixSpan(memory.offset(memory_offset), r.column_ct,  r.column_ct + 1);

        auto pivots = Span<unsigned>(memory.offset(memory_offset + beta.bytes()), beta.row_ct);

        beta.to_subspan_unsafe(r.column_ct, r.column_ct).copy_from_unsafe(r);

        for (unsigned row = 0; row < projected_y.row_ct; ++row)
        {
            beta.set_unsafe(row, r.column_ct, projected_y.get_unsafe(row, 0));
        }

//        gauss_jordan_elimination(beta);
        gauss_jordan_elimination(beta, pivots);

        auto beta_vector = beta.to_vector_subspan_unsafe(r.column_ct, 0, 0);

        for (unsigned i = 0; i < beta_vector.length; ++i)
        {
            beta_vector[i] = beta.get_unsafe(pivots[i], r.column_ct);
        }

        memory_offset += beta_vector.bytes();

        return beta_vector;
    }
}

namespace nml
{
    using namespace nml::ordinary_least_squares_internal;

    RequiredMemory OLS::required_memory(const Request& request) noexcept
    {
        unsigned independent_coefficients_bytes = VectorSpan::required_bytes(request.data.column_ct - 1);

        unsigned working_matrix_bytes = MatrixSpan::required_bytes(request.data.row_ct, request.data.column_ct);

        unsigned qr_bytes = working_matrix_bytes * 2 + VectorSpan::required_bytes(request.data.row_ct + request.data.column_ct);

        unsigned solve_coefficients_bytes = MatrixSpan::required_bytes(request.data.column_ct, request.data.column_ct + 1)
                                          + Span<unsigned>::required_bytes(request.data.column_ct);
        return RequiredMemory
        {
            .result_required_bytes = independent_coefficients_bytes,
            .working_required_bytes = working_matrix_bytes + qr_bytes + solve_coefficients_bytes
        };
    }

    NMLResult<OLS> OLS::compute(const Request& request, RequestMemory memory) noexcept
    {
        unsigned working_memory_offset = 0;

        auto y = request.data.to_subspan_unsafe(request.data.row_ct, 1);

        auto working_matrix = get_working_matrix(request, memory.working_memory, working_memory_offset);

        auto qr = qr_decomposition(working_matrix, memory.working_memory, working_memory_offset);

        auto projected_y = MatrixSpan(memory.working_memory.offset(working_memory_offset), working_matrix.column_ct, 1);

        working_memory_offset += projected_y.bytes();

        qr.q.transpose_multiply(y, projected_y);

        auto coefficients = solve_for_coefficients(qr.r, projected_y, memory.working_memory, working_memory_offset);

        auto independent_coefficients = VectorSpan(memory.result_memory, coefficients.length - 1);

        independent_coefficients.copy_from_unsafe(coefficients.to_subspan_unsafe(1));

        return NMLResult<OLS>::ok({
            .constant = coefficients[0],
            .variables = independent_coefficients
        });
    }

    float OLS::estimate(const VectorSpan& observation) const noexcept
    {
        float result = constant;

        for (unsigned i = 0; i < variables.length; ++i)
        {
            result += variables[i] * observation[i];
        }

        return result;
    }

    struct OLS::Summary
    {
        float f_statistic;
        float probability_of_f_statistic;

        unsigned observation_count;
        unsigned model_degrees_of_freedom;
        unsigned residuals_degrees_of_freedom;

        float log_likelihood;

        float r_squared;
        float r_squared_adjusted;

        float akaike_information_criterion;
        float bayesian_information_criterion;

        struct ConfidenceInterval
        {
            float upper_boundary;
            float lower_boundary;
        };

        struct Coefficient
        {
            float p_value;
            float coefficient;
            float t_statistic;
            float standard_error;
            std::string variable_name;

            ConfidenceInterval confidence_interval_95;
        };

        Coefficient constant;
        std::vector<Coefficient> independent_variables;

        void print()
        {
            auto table = SummaryTable("OLS Regression Summary");

            table.row_items.push_back({.name = "Observations", .is_float = false, .value_unsigned = observation_count });
            table.row_items.push_back({.name = "F-Statistic", .is_float = true, .value_float = f_statistic });
            table.row_items.push_back({.name = "Model Degrees Of Freedom", .is_float = false, .value_unsigned = model_degrees_of_freedom });
            table.row_items.push_back({.name = "Prob (F-statistic)", .is_float = true, .value_float = probability_of_f_statistic });
            table.row_items.push_back({.name = "Residuals Degrees Of Freedom", .is_float = false, .value_unsigned = residuals_degrees_of_freedom });
            table.row_items.push_back({.name = "Log-Likelihood", .is_float = true, .value_float = log_likelihood });
            table.row_items.push_back({.name = "R-Squared", .is_float = true, .value_float = r_squared });
            table.row_items.push_back({.name = "Akaike Information Criterion", .is_float = true, .value_float = akaike_information_criterion });
            table.row_items.push_back({.name = "Adjusted R-Squared", .is_float = true, .value_float = r_squared_adjusted });
            table.row_items.push_back({.name = "Bayesian Information Criterion", .is_float = true, .value_float = bayesian_information_criterion });

            table.groups.resize(1);

            table.groups[0].column_headers.emplace_back("Coefficient");
            table.groups[0].column_headers.emplace_back("Std Error");
            table.groups[0].column_headers.emplace_back("t Value");
            table.groups[0].column_headers.emplace_back("Pr > |t|");
            table.groups[0].column_headers.emplace_back("[0.025");
            table.groups[0].column_headers.emplace_back("0.975]");

            table.groups[0].variable_names.emplace_back("Constant");

            auto constant_values = std::vector<float>();

            constant_values.emplace_back(constant.coefficient);
            constant_values.emplace_back(constant.standard_error);
            constant_values.emplace_back(constant.t_statistic);
            constant_values.emplace_back(constant.p_value);
            constant_values.emplace_back(constant.confidence_interval_95.lower_boundary);
            constant_values.emplace_back(constant.confidence_interval_95.upper_boundary);

            table.groups[0].column_values.push_back(constant_values);

            for (int i = 0; i < independent_variables.size(); ++i)
            {
                table.groups[0].variable_names.emplace_back(independent_variables[i].variable_name);

                auto variable_values = std::vector<float>();

                variable_values.emplace_back(independent_variables[i].coefficient);
                variable_values.emplace_back(independent_variables[i].standard_error);
                variable_values.emplace_back(independent_variables[i].t_statistic);
                variable_values.emplace_back(independent_variables[i].p_value);
                variable_values.emplace_back(independent_variables[i].confidence_interval_95.lower_boundary);
                variable_values.emplace_back(independent_variables[i].confidence_interval_95.upper_boundary);

                table.groups[0].column_values.push_back(variable_values);
            }

            table.print();
        }
    };

    OLS::Summary OLS::summary(const MatrixSpan& data) const noexcept
    {
        auto row_ct = static_cast<float>(data.row_ct);

        float mean_y = 0, sse = 0, ssr = 0, sst = 0;

        for (unsigned row_offset = 0; row_offset < data.row_ct; ++row_offset)
        {
            mean_y += data.get_unsafe(row_offset, 0);
        }

        mean_y /= row_ct;

        for (unsigned row_offset = 0; row_offset < data.row_ct; ++row_offset)
        {
            auto row = data[row_offset];
            float est = estimate(row.to_subspan_unsafe(1));

            float residual = est - mean_y;
            ssr += residual * residual;

            float error = est - row[0];
            sse += error * error;

            float ss = row[0] - mean_y;
            sst += ss * ss;
        }

        auto df_model = static_cast<float>(variables.length);
        auto df_error = row_ct - df_model - 1;

        float msr = ssr / df_model;
        float mse = sse / df_error;

        float f_statistic = msr / mse;
        double p_f_statistic = Distribution::f_cdf(f_statistic, df_model, df_error, false);

        double t_confidence_interval = Distribution::t_cdf(0.975, df_error);

        float r_squared = 1 - (sse / sst);
        float r_squared_adjusted = 1 - ((1 - r_squared) * (row_ct - 1)) / df_error;

        float sigma_squared = sse / df_error;
        double log_likelihood = -0.5f * row_ct * (std::log(2 * M_PI) + std::log(sigma_squared) + 1);

        double num_parameters = variables.length + 1;
        double aic = 2 * num_parameters - 2 * log_likelihood;
        double bic = std::log(row_ct) * num_parameters - 2 * log_likelihood;

        auto independent_variables = std::vector<OLS::Summary::Coefficient>();

        // yuck
        unsigned a = 0;
        auto memory = MemoryOwner(data.bytes() + MatrixSpan::required_bytes(data.column_ct, data.column_ct) * 3);
        auto memory_span = memory.to_memory_span();
        auto working_matrix = get_working_matrix(OLS::Request(data), memory_span, a);

        auto xt_t = MatrixSpan(memory_span.offset(a), data.column_ct, data.column_ct);

        a += xt_t.bytes();

        working_matrix.transpose_multiply(working_matrix, xt_t);

        auto inverse = MatrixSpan(memory_span.offset(a), data.column_ct, data.column_ct * 2);

        auto left_copy = inverse.to_subspan_unsafe(data.column_ct, data.column_ct);
        auto right_identity = inverse.to_subspan_unsafe(data.column_ct, data.column_ct, 0, data.column_ct);

        left_copy.copy_from_unsafe(xt_t);
        right_identity.set_identity();

        gauss_jordan_elimination(inverse);

        for (int i = 0; i < variables.length; ++i)
        {
            float standard_error = std::sqrt(sigma_squared * right_identity.get_unsafe(i + 1, i + 1));
            float t_statistic = variables[i] / standard_error;
            auto p_t_statistic = static_cast<float>(Distribution::t_cdf(t_statistic, df_error, false));
            float ci_adjustment = static_cast<float>(t_confidence_interval) * standard_error;

            independent_variables.push_back({
                .p_value = p_t_statistic,
                .coefficient = variables[i],
                .t_statistic = t_statistic,
                .standard_error = standard_error,
                .variable_name = "Variable " + std::to_string(i + 1),
                .confidence_interval_95 =
                {
                    .upper_boundary = variables[i] + ci_adjustment,
                    .lower_boundary = variables[i] - ci_adjustment
                }
            });
        }

        float standard_error = std::sqrt(sigma_squared * right_identity.get_unsafe(0, 0));
        float t_statistic = constant / standard_error;
        double p_t_statistic = Distribution::t_cdf(t_statistic, df_error, false);
        float ci_adjustment = static_cast<float>(t_confidence_interval) * standard_error;

        return OLS::Summary
        {
            .f_statistic = f_statistic,
            .probability_of_f_statistic = static_cast<float>(p_f_statistic),
            .observation_count = data.row_ct,
            .model_degrees_of_freedom = (unsigned)variables.length,
            .residuals_degrees_of_freedom = (unsigned)(data.row_ct - variables.length - 1),
            .log_likelihood = static_cast<float>(log_likelihood),
            .r_squared = r_squared,
            .r_squared_adjusted = r_squared_adjusted,
            .akaike_information_criterion = static_cast<float>(aic),
            .bayesian_information_criterion = static_cast<float>(bic),
            .constant =
            {
                .p_value = static_cast<float>(p_t_statistic),
                .coefficient = constant,
                .t_statistic = constant / standard_error,
                .standard_error = standard_error,
                .variable_name = "Constant",
                .confidence_interval_95 =
                {
                    .upper_boundary = constant + ci_adjustment,
                    .lower_boundary = constant - ci_adjustment
                }
            },
            .independent_variables = independent_variables
        };
    }
}

#endif //NML_ORDINARY_LEAST_SQUARES_H