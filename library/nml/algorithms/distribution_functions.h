//
// Created by nik on 5/8/2024.
//

#ifndef NML_DISTRIBUTION_FUNCTIONS_H
#define NML_DISTRIBUTION_FUNCTIONS_H

#include <cmath>
#include <cfloat>

namespace nml
{
    struct Distribution
    {
        static double t_pdf(float t_statistic, float df, bool log_scale = false) noexcept;
        static double t_cdf(float t_statistic, float df, bool lower_tail = true, bool log_scale = false) noexcept;
        static double f_cdf(float f_statistic, float df_model, float df_error, bool lower_tail = true, bool log_scale = false) noexcept;
        static double beta_cdf(double upper_limit, double alpha, double beta, bool lower_tail = true, bool log_scale = false) noexcept;
        static double normal_pdf(double value, double mean = 0, double standard_deviation = 1, bool lower_tail = true, bool log_scale = false) noexcept;
        static double normal_cdf(double value, double mean = 0, double standard_deviation = 1, bool lower_tail = true, bool log_scale = false) noexcept;
    };
}

#ifndef M_PI
#define M_PI 3.14159265358979311599796346854
#endif

#ifndef M_LN2
#define M_LN2 0.69314718055994530942
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

namespace nml::cumulative_distribution_functions_internal // based off of https://github.com/wch/r-source/tree/trunk/src/nmath
{
    const unsigned max_iteration_count = 1e7;

    // Coefficients for the polynomial approximation of the correction term in Stirling's approximation
    const double sc[6]
    {
        .0833333333333333, -.00277777777760991, 7.9365066682539e-4,
        -5.9520293135187e-4, 8.37308034031215e-4, -.00165322962780713,
    };

    const double sqrt_32 = 5.656854249492380195206754896838; /* sqrt(32) */
    const double ln_two_pi = 1.837877066409345483560659472811; /* log(2 * pi) */
    const double ln_sqrt_two_pi = 0.918938533204672741780329736406; /* log(sqrt(2 * pi)) */
    const double inverse_sqrt_two_pi = 0.398942280401432677939946059934; /* 1 / sqrt(2 * pi); */

    const double positive_infinity = std::numeric_limits<double>::infinity();
    const double negative_infinity = -std::numeric_limits<double>::infinity();

    static inline bool is_finite(const double value)
    {
        return (!std::isnan(value) && (value != positive_infinity) && (value != negative_infinity));
    }

    // exp(upper_limit - 1)
    static inline double stable_e_to_the_x_minus_one(double x) noexcept
    {
        double result, magnitude = fabs(x);

        if (magnitude < DBL_EPSILON) return x;
        if (magnitude > 0.697) return exp(x) - 1;

        if (magnitude > 1e-8) result = exp(x) - 1;
        else result = (x / 2 + 1) * x;

        result -= (1 + result) * (log1p(result) - x);

        return result;
    }

    // ln(1 - exp(upper_limit))
    static inline double stable_ln_of_one_minus_e_to_the_x(double x) noexcept
    {
        return x > -M_LN2 ? log(-stable_e_to_the_x_minus_one(x)) : log1p(-exp(x));
    }

    static inline double numeric_zero(bool log_scale) noexcept
    {
        return log_scale ? negative_infinity : 0;
    }

    static inline double numeric_one(bool log_scale) noexcept
    {
        return log_scale ? 0 : 1;
    }

    static inline double max_exp_argument(bool is_negative) noexcept
    {
        const double ln_two = .69314718055995;

        int m = is_negative ? DBL_MIN_EXP - 1 : DBL_MAX_EXP;

        return m * ln_two * .99999;
    }

    // exp(value) - 1
    static inline double exp_value_minus_one(double value) noexcept
    {
        const double p1 = 9.14041914819518e-10, p2 = .0238082361044469;

        const double q1 = -.499999999085958, q2 = .107141568980644,
                     q3 = -.0119041179760821, q4 = 5.95130811860248e-4;

        if (fabs(value) <= 0.15)
        {
            double numerator = ((p2 * value + p1) * value + 1);
            double denominator = ((((q4 * value + q3) * value + q2) * value + q1) * value + 1);

            return value * (numerator / denominator);
        }
        else
        {
            double w = exp(value);

            return value > 0 ?
               w * (0.5 - 1 / w + 0.5) :
               w - 0.5 - 0.5;
        }
    }

    // ln(exp(ln_x) + exp(ln_y))
    double ln_sum_of_ln(double ln_x, double ln_y)
    {
        return std::max(ln_x, ln_y) + log1p(exp(-fabs(ln_x - ln_y)));
    }

    // ln(value + 1)
    static inline double ln_value_plus_one(double value) noexcept
    {
        if (fabs(value) > 0.375)
        {
            return log(1 + value);
        }

        const double p1 = -1.29418923021993, p2 = .405303492862024, p3 = -.0178874546012214;

        const double q1 = -1.62752256355323, q2 = .747811014037616, q3 = -.0845104217945565;

        double t = value / (value + 2);
        double t2 = t * t;
        double w = (((p3 * t2 + p2) * t2 + p1) * t2 + 1) / (((q3 * t2 + q2) * t2 + q1) * t2 + 1);

        return t * 2 * w;
    }

    static inline double error_function(double value) noexcept
    {
        const double c = .564189583547756;

        const double a[5] =
        {
            7.7105849500132e-5,-.00133733772997339,
            .0323076579225834,.0479137145607681,.128379167095513
        };

        const double b[3] =
        {
            .00301048631703895,.0538971687740286, .375795757275549
        };

        const double p[8] =
        {
            -1.36864857382717e-7,.564195517478974,
            7.21175825088309,43.1622272220567,152.98928504694,
            339.320816734344,451.918953711873,300.459261020162
        };

        const double q[8] =
        {
            1.,12.7827273196294,77.0001529352295,
            277.585444743988,638.980264465631,
            931.35409485061, 790.950925327898,300.459260956983
        };

        const double r[5] =
        {
            2.10144126479064,26.2370141675169,
            21.3688200555087,4.6580782871847,.282094791773523
        };

        const double s[4] =
        {
            94.153775055546,187.11481179959,
            99.0191814623914,18.0124575948747
        };

        double result;
        double t, value_squared, numerator, denominator;

        double ax = fabs(value);

        if (ax <= 0.5)
        {
            t = value * value;
            numerator = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1;
            denominator = ((b[0] * t + b[1]) * t + b[2]) * t + 1;

            return value * (numerator / denominator);
        }
        else if (ax <= 4)
        {
            numerator = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4]) * ax + p[5]) * ax + p[6]) * ax + p[7];
            denominator = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4]) * ax + q[5]) * ax + q[6]) * ax + q[7];
            result = 0.5 - exp(-value * value) * numerator / denominator + 0.5;

            return (value < 0) ? -result : result;
        }
        else if (ax >= 5.8)
        {
            return value < 0 ? -1 : 1;
        }
        else
        {
            value_squared = value * value;
            t = 1 / value_squared;
            numerator = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
            denominator = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1.;
            t = (c - numerator / (value_squared * denominator)) / ax;
            result = 0.5 - exp(-value_squared) * t + 0.5;

            return (value < 0) ? -result : result;
        }
    }

    // scale ? exp(value * value) * complementary_error_function(value) : complementary_error_function(value)
    static inline double scaled_complementary_error_function(bool scale, double value) noexcept
    {
        const double c = .564189583547756;

        const double a[5] =
        {
            7.7105849500132e-5,-.00133733772997339,
            .0323076579225834,.0479137145607681,.128379167095513
        };

        const double b[3] =
        {
            .00301048631703895,.0538971687740286, .375795757275549
        };

        const double p[8] =
        {
            -1.36864857382717e-7,.564195517478974,
            7.21175825088309,43.1622272220567,152.98928504694,
            339.320816734344,451.918953711873,300.459261020162
        };

        const double q[8] =
        {
            1.,12.7827273196294,77.0001529352295,
            277.585444743988,638.980264465631,
            931.35409485061, 790.950925327898,300.459260956983
        };

        const double r[5] =
        {
            2.10144126479064,26.2370141675169,
            21.3688200555087,4.6580782871847,.282094791773523
        };

        const double s[4] =
        {
            94.153775055546,187.11481179959,
            99.0191814623914,18.0124575948747
        };

        double result;
        double e, t, w, numerator, denominator;

        double ax = fabs(value);

        if (ax <= 0.5)
        {
            t = value * value;
            numerator = (((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4] + 1;
            denominator = ((b[0] * t + b[1]) * t + b[2]) * t + 1;
            result = 0.5 - value * (numerator / denominator) + 0.5;

            return scale ? exp(t) * result : result;
        }

        if (ax <= 4)
        {
            numerator = ((((((p[0] * ax + p[1]) * ax + p[2]) * ax + p[3]) * ax + p[4]) * ax + p[5]) * ax + p[6]) * ax + p[7];
            denominator = ((((((q[0] * ax + q[1]) * ax + q[2]) * ax + q[3]) * ax + q[4]) * ax + q[5]) * ax + q[6]) * ax + q[7];
            result = numerator / denominator;
        }
        else if (value <= -5.6)
        {
            return scale ? exp(value * value) * 2 : 2;
        }
        else if (scale == 0 && (value > 100. || value * value > -max_exp_argument(true)))
        {
            return 0;
        }
        else
        {
            t = 1 / (value * value);
            numerator = (((r[0] * t + r[1]) * t + r[2]) * t + r[3]) * t + r[4];
            denominator = (((s[0] * t + s[1]) * t + s[2]) * t + s[3]) * t + 1;
            result = (c - t * numerator / denominator) / ax;
        }

        if (scale)
        {
            if (value < 0)
            {
                result = exp(value * value) * 2 - result;
            }
        }
        else
        {
            w = value * value;
            t = w;
            e = w - t;
            result = (0.5 - e + 0.5) * exp(-t) * result;

            if (value < 0)
            {
                result = 2 - result;
            }
        }

        return result;
    }

    // value - ln(1 + value)
    static inline double value_minus_ln_value_plus_one(double value) noexcept
    {
        const double q1 = -1.27408923933623, q2 = .354508718369557;

        const double alpha = .0566749439387324, beta = .0456512608815524;

        const double p0 = .333333333333333, p1 = -.224696413112536, p2 = .00620886815375787;

        double h, r, t, w, w1;

        if (value < -0.39 || value > 0.57)
        {
            w = value + 0.5 + 0.5;

            return value - log(w);
        }

        if (value < -0.18)
        {
            h = value + 0.3;
            h /= 0.7;
            w1 = alpha - h * 0.3;
        }
        else if (value > 0.18)
        {
            h = value * 0.75 - 0.25;
            w1 = beta + h / 3;
        }
        else
        {
            h = value;
            w1 = 0;
        }

        r = h / (h + 2);
        t = r * r;
        w = ((p2 * t + p1) * t + p0) / ((q2 * t + q1) * t + 1);

        return t * 2 * (1 / (1 - r) - r * w) + w1;
    }

    // log_scale ? integer + double : exp(integer + double)
    static inline double exp_sum(int integer, double value, bool log_scale) noexcept
    {
        auto cast = static_cast<double>(integer);
        double sum = value + cast;

        if (log_scale)
        {
            return sum;
        }

        if (value > 0)
        {
            if (integer > 0 || sum < 0)
            {
                return exp(cast) * exp(value);
            }
        }
        else
        {
            if (integer < 0 || sum > 0)
            {
                return exp(cast) * exp(value);
            }
        }

        return exp(sum);
    }

    // https://en.wikipedia.org/wiki/Digamma_function
    // CONSTRAINT value != 0
    static inline double derivative_ln_gamma(double value) noexcept
    {
        const double pi_over_four = .785398163397448;

        const double dx0 = 1.461632144968362341262659542325721325;

        const double q2[4] =
        {
            32.2703493791143,89.2920700481861,
            54.6117738103215,7.77788548522962
        };

        const double p2[4] =
        {
            -2.12940445131011,-7.01677227766759,
            -4.48616543918019,-.648157123766197
        };

        const double q1[6] =
        {
            44.8452573429826,520.752771467162, 2210.0079924783,
            3641.27349079381,1908.310765963, 6.91091682714533e-6
        };

        const double p1[7] =
        {
            .0089538502298197,4.77762828042627, 142.441585084029,
            1186.45200713425,3633.51846806499, 4138.10161269013,1305.60269827897
        };

        double max_int = std::min(static_cast<double>(INT_MAX), 0.5 / (0.5 * DBL_EPSILON));

        if (value == 0 || fabs(value) >= max_int)
        {
            return 0;
        }

        double w, z, den, aug = 0, sgn, xmx0;

        if (value < 0.5)
        {
            if (fabs(value) <= 1e-9)
            {
                aug = -1 / value;
            }
            else
            {
                w = -value;
                sgn = pi_over_four;

                if (w <= 0.)
                {
                    w = -w;
                    sgn = -sgn;
                }

                auto nq = static_cast<int>(w);
                w -= static_cast<double>(nq);

                nq = static_cast<int>(w * 4);
                w = (w - static_cast<double>(nq) * 0.25) * 4;

                int n = nq / 2;

                if (n + n != nq)
                {
                    w = 1. - w;
                }

                z = pi_over_four * w;
                int m = n / 2;

                if (m + m != n)
                {
                    sgn = -sgn;
                }

                n = (nq + 1) / 2;
                m = n / 2;
                m += m;

                if (m == n)
                {
                    if (z == 0)
                    {
                        return 0;
                    }

                    aug = sgn * (cos(z) / sin(z) * 4.);
                }
                else
                {
                    aug = sgn * (sin(z) / cos(z) * 4.);
                }
            }

            value = 1 - value;
        }

        if (value <= 3)
        {
            den = value;
            double upper = p1[0] * value;

            for (unsigned i = 1; i <= 5; ++i)
            {
                den = (den + q1[i - 1]) * value;
                upper = (upper + p1[i]) * value;
            }

            den = (upper + p1[6]) / (den + q1[5]);
            xmx0 = value - dx0;

            return den * xmx0 + aug;
        }
        else if (value < max_int)
        {
            w = 1 / (value * value);
            den = w;

            double upper = p2[0] * w;

            for (unsigned i = 1; i <= 3; ++i)
            {
                den = (den + q2[i - 1]) * w;
                upper = (upper + p2[i]) * w;
            }

            aug = upper / (den + q2[3]) - 0.5 / value + aug;
        }

        return aug + log(value);
    }

    // 1 / gamma(value + 1)
    // CONSTRAINT -0.5 <= value <= 1.5
    static inline double inverse_gamma_value_plus_one(double value) noexcept
    {
        double t = value;
        double d = value - 0.5;

        if (d > 0) t = d - 0.5;

        if (t == 0)
        {
            return 0;
        }
        else if (t < 0)
        {
            const double r[9] =
            {
                -.422784335098468, -.771330383816272, -.244757765222226,
                .118378989872749, 9.30357293360349e-4, -.0118290993445146,
                .00223047661158249, 2.66505979058923e-4, -1.32674909766242e-4
            };

            const double s1 = .273076135303957, s2 = .0559398236957378;

            double numerator = (((((((r[8] * t + r[7]) * t + r[6]) * t + r[5]) * t + r[4]) * t + r[3]) * t + r[2]) * t + r[1]) * t + r[0];
            double denominator = (s2 * t + s1) * t + 1;

            double w = numerator / denominator;

            return d > 0 ? t * w / value : value * (w + 0.5 + 0.5);
        }
        else
        {
            const double p[7] =
            {
                .577215664901533,-.409078193005776, -.230975380857675,
                .0597275330452234,.0076696818164949, -.00514889771323592,5.89597428611429e-4
            };

            const double q[5] =
            {
                1.,.427569613095214,.158451672430138,
                .0261132021441447,.00423244297896961
            };

            double numerator = (((((p[6] * t + p[5]) * t + p[4]) * t + p[3]) * t + p[2]) * t + p[1]) * t + p[0];
            double denominator = (((q[4] * t + q[3]) * t + q[2]) * t + q[1]) * t + 1;

            double w = numerator / denominator;

            return d > 0 ? t / value * (w - 0.5 - 0.5) : value * w;
        }
    }

    // ln(gamma(right) / gamma(left + right))
    // CONSTRAINT 8 <= right
    static inline double ln_gamma_right_over_gamma_sum(double left, double right) noexcept
    {
        double c, d, h, t, u, v, w, x, s3, s5, x2, s7, s9, s11;

        if (left > right)
        {
            h = right / left;
            c = 1 / (h + 1);
            x = h / (h + 1);
            d = left + (right - 0.5);
        }
        else
        {
            h = left / right;
            c = h / (h + 1);
            x = 1 / (h + 1);
            d = right + (left - 0.5);
        }

        x2 = x * x;
        s3 = x + x2 + 1;
        s5 = x + x2 * s3 + 1;
        s7 = x + x2 * s5 + 1;
        s9 = x + x2 * s7 + 1;
        s11 = x + x2 * s9 + 1;

        t = 1 / (right * right);
        w = ((((sc[5] * s11 * t + sc[4] * s9) * t + sc[3] * s7) * t + sc[2] * s5) * t + sc[1] * s3) * t + sc[0];
        w *= c / right;

        u = d * ln_value_plus_one(left / right);
        v = left * (log(right) - 1);

        return u > v ? w - v - u : w - u - v;
    }

    // ln(gamma(value + 1))
    // CONSTRAINT -0.2 <= value <= 1.25
    static inline double ln_gamma_value_plus_one(double value) noexcept
    {
        if (value < 0.6)
        {
            const double p0 = .577215664901533, p1 = .844203922187225,
                         p2 = -.168860593646662, p3 = -.780427615533591,
                         p4 = -.402055799310489, p5 = -.0673562214325671, p6 = -.00271935708322958;

            const double q1 = 2.88743195473681, q2 = 3.12755088914843, q3 = 1.56875193295039,
                         q4 = .361951990101499, q5 = .0325038868253937, q6 = 6.67465618796164e-4;

            const double numerator = ((((((p6 * value + p5) * value + p4) * value + p3) * value + p2) * value + p1) * value + p0);
            const double denominator = ((((((q6 * value + q5) * value + q4) * value + q3) * value + q2) * value + q1) * value + 1);

            return -(value) * numerator / denominator;
        }
        else
        {
            const double r0 = .422784335098467, r1 = .848044614534529, r2 = .565221050691933,
                         r3 = .156513060486551, r4 = .017050248402265, r5 = 4.97958207639485e-4;

            const double s1 = 1.24313399877507, s2 = .548042109832463,
                         s3 = .10155218743983, s4 = .00713309612391, s5 = 1.16165475989616e-4;

            const double x = value - 0.5 - 0.5;

            const double numerator = (((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x + r0);
            const double denominator = (((((s5 * x + s4) * x + s3) * x + s2) * x + s1) * x + 1);

            return x * numerator / denominator;
        }
    }

    // CONSTRAINT 0 < value
    static inline double ln_gamma(double value) noexcept
    {
        const double d = .418938533204673;

        if (value <= 0.8)
        {
            return ln_gamma_value_plus_one(value) - log(value);
        }
        else if (value <= 2.25)
        {
            return ln_gamma_value_plus_one(value - 0.5 - 0.5);
        }
        else if (value < 10)
        {
            auto n = static_cast<unsigned>(value - 1.25);

            double factorial = 1;

            for (unsigned i = 1; i <= n; ++i)
            {
                factorial *= --value; // TODO
            }

            return ln_gamma_value_plus_one(value - 1) + log(factorial);
        }
        else
        {
            double t = 1 / (value * value);
            double w = (((((sc[5] * t + sc[4]) * t + sc[3]) * t + sc[2]) * t + sc[1]) * t + sc[0]) / value;
            return d + w + (value - 0.5) * (log(value) - 1);
        }
    }

    // CONSTRAINT 1 <= alpha, beta <= 2
    static inline double ln_gamma_sum(double alpha, double beta) noexcept
    {
        double sum = alpha + beta;

        if (sum <= 2.25)
        {
            return ln_gamma_value_plus_one(sum - 1);
        }

        if (sum <= 3.25)
        {
            sum -= 2; return ln_gamma_value_plus_one(sum) + ln_value_plus_one(sum);
        }

        return ln_gamma_value_plus_one(sum - 3) + log((sum - 2) * (sum - 1));
    }

    // CONSTRAINT 1 >= alpha
    static inline double scaled_complement_incomplete_gamma_ratio(double alpha, double upper_limit,
                                                                  double tolerance, double log_r) noexcept
    {
        const double sqrt_pi = 1.772453850905516027298167483341;

        if (alpha * upper_limit == 0.)
        {
            return upper_limit <= alpha ? exp(-log_r) : 0;
        }
        else if (alpha == 0.5)
        {
            if (upper_limit < 0.25)
            {
                double p = error_function(sqrt(upper_limit));

                return (0.5 - p + 0.5) * exp(-log_r);
            }
            else
            {
                double sx = sqrt(upper_limit);

                return scaled_complementary_error_function(true, sx) / sx * sqrt_pi;
            }
        }
        else if (upper_limit < 1.1)
        {
            double an = 3,
                   c = upper_limit,
                   sum = upper_limit / (alpha + 3),
                   tol = tolerance * 0.1 / (alpha + 1), t;

            do
            {
                an += 1;
                c *= -(upper_limit / an);
                t = c / (alpha + an);
                sum += t;
            }
            while (an < max_iteration_count && fabs(t) > tol);

            double j = alpha * upper_limit * ((sum / 6 - 0.5 / (alpha + 2)) * upper_limit + 1 / (alpha + 1)),
                   z = alpha * log(upper_limit),
                   h = inverse_gamma_value_plus_one(alpha),
                   g = h + 1;

            if ((upper_limit >= 0.25 && alpha < upper_limit / 2.59) || z > -0.13394)
            {
                double l = exp_value_minus_one(z);
                double q = ((l + 0.5 + 0.5) * j - l) * g - h;

                return q <= 0 ? 0 : q * exp(-log_r);
            }
            else
            {
                double p = exp(z) * g * (0.5 - j + 0.5);

                return (0.5 - p + 0.5) * exp(-log_r);
            }
        }
        else
        {
            double a2n_1 = 1,
                   a2n = 1,
                   b2n_1 = upper_limit,
                   b2n = upper_limit + (1 - alpha),
                   c = 1, am0, an0;

            do
            {
                a2n_1 = upper_limit * a2n + c * a2n_1;
                b2n_1 = upper_limit * b2n + c * b2n_1;
                am0 = a2n_1 / b2n_1;
                c += 1;
                double c_a = c - alpha;
                a2n = a2n_1 + c_a * a2n;
                b2n = b2n_1 + c_a * b2n;
                an0 = a2n / b2n;
            }
            while (c < max_iteration_count && fabs(an0 - am0) >= tolerance * an0);

            return an0;
        }
    }


    double log_likelihood_ratio_deviance(double observed, double expected) noexcept
    {
        if (fabs(observed - expected) < 0.1 * (observed + expected))
        {
            double normalized_difference = (observed - expected) / (observed + expected);
            double taylor_series = (observed - expected) * normalized_difference;

            if(fabs(taylor_series) < DBL_MIN)
            {
                return taylor_series;
            }

            double series_adjustment = 2 * observed * normalized_difference;

            normalized_difference *= normalized_difference;

            for (int iteration = 1; iteration < 1000; iteration++)
            {
                series_adjustment *= normalized_difference;

                double taylor_series_previous = taylor_series;

                taylor_series += series_adjustment / ((iteration << 1) + 1);

                if (taylor_series == taylor_series_previous)
                {
                    return taylor_series;
                }
            }
        }

        return(observed * log(observed / expected) + expected - observed);
    }

    double log_stirling_error(double value) noexcept
    {
        const double s0   = 0.083333333333333333333;        // 1 / 12
        const double s1   = 0.00277777777777777777778;      // 1 / 360
        const double s2   = 0.00079365079365079365079365;   // 1 / 1260
        const double s3   = 0.000595238095238095238095238;  // 1 / 1680
        const double s4   = 0.0008417508417508417508417508; // 1 / 1188
        const double s5   = 0.0019175269175269175269175262; // 691 / 360360
        const double s6   = 0.0064102564102564102564102561; // 1 / 156
        const double s7   = 0.029550653594771241830065352;  // 3617 / 122400
        const double s8   = 0.17964437236883057316493850;   // 43867 / 244188
        const double s9   = 1.3924322169059011164274315;    // 174611 / 125400
        const double s10  = 13.402864044168391994478957;    // 77683 / 5796
        const double s11  = 156.84828462600201730636509;    // 236364091 / 1506960
        const double s12  = 2193.1033333333333333333333;    // 657931 / 300
        const double s13  = 36108.771253724989357173269;    // 3392780147 / 93960
        const double s14  = 691472.26885131306710839498;    // 1723168255201 / 2492028
        const double s15  = 15238221.539407416192283370;    // 7709321041217 / 505920
        const double s16  = 382900751.39141414141414141;    // 151628697551 / 396

        const double stirling_error_stored_values[31] =
        {
            0.0,                            // place-holder
            0.1534264097200273452913848,    // 0.5
            0.0810614667953272582196702,    // 1.0
            0.0548141210519176538961390,    // 1.5
            0.0413406959554092940938221,    // 2.0
            0.03316287351993628748511048,   // 2.5
            0.02767792568499833914878929,   // 3.0
            0.02374616365629749597132920,   // 3.5
            0.02079067210376509311152277,   // 4.0
            0.01848845053267318523077934,   // 4.5
            0.01664469118982119216319487,   // 5.0
            0.01513497322191737887351255,   // 5.5
            0.01387612882307074799874573,   // 6.0
            0.01281046524292022692424986,   // 6.5
            0.01189670994589177009505572,   // 7.0
            0.01110455975820691732662991,   // 7.5
            0.010411265261972096497478567,  // 8.0
            0.009799416126158803298389475,  // 8.5
            0.009255462182712732917728637,  // 9.0
            0.008768700134139385462952823,  // 9.5
            0.008330563433362871256469318,  // 10.0
            0.007934114564314020547248100,  // 10.5
            0.007573675487951840794972024,  // 11.0
            0.007244554301320383179543912,  // 11.5
            0.006942840107209529865664152,  // 12.0
            0.006665247032707682442354394,  // 12.5
            0.006408994188004207068439631,  // 13.0
            0.006171712263039457647532867,  // 13.5
            0.005951370112758847735624416,  // 14.0
            0.005746216513010115682023589,  // 14.5
            0.005554733551962801371038690   // 15.0
        };

        double nn;

        if (value <= 23.5)
        {
            nn = value + value;

            int nn_int = static_cast<int>(nn);

            if (value <= 15 && nn == nn_int)
            {
                return stirling_error_stored_values[nn_int];
            }
            else if (value <= 5.25)
            {
                if (value >= 1)
                {
                    double ln_value = log(value);
                    return lgamma(value) + value * (1 - ln_value) + ldexp(ln_value - ln_two_pi, -1);
                }
                else
                {
                    return ln_gamma_value_plus_one(value) - (value + 0.5) * log(value) + value - ln_sqrt_two_pi;
                }
            }

            nn = value * value;

            if (value > 12.8) return (s0-(s1-(s2-(s3-(s4-(s5-s6/nn)/nn)/nn)/nn)/nn)/nn)/value;
            if (value > 12.3) return (s0-(s1-(s2-(s3-(s4-(s5-(s6-s7/nn)/nn)/nn)/nn)/nn)/nn)/nn)/value;
            if (value > 8.9)  return (s0-(s1-(s2-(s3-(s4-(s5-(s6-(s7-s8/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/value;
            if (value > 7.3)  return (s0-(s1-(s2-(s3-(s4-(s5-(s6-(s7-(s8-(s9-s10/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/value;
            if (value > 6.6)  return (s0-(s1-(s2-(s3-(s4-(s5-(s6-(s7-(s8-(s9-(s10-(s11-s12/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/value;
            if (value > 6.1)  return (s0-(s1-(s2-(s3-(s4-(s5-(s6-(s7-(s8-(s9-(s10-(s11-(s12-(s13-s14/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/value;

            return (s0-(s1-(s2-(s3-(s4-(s5-(s6-(s7-(s8-(s9-(s10-(s11-(s12-(s13-(s14-(s15-s16/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/nn)/value;
        }
        else
        {
            nn = value * value;

            if (value > 15.7e6)	return s0/value;
            if (value > 6180)	return (s0-s1/nn)/value;
            if (value > 205)    return (s0-(s1-s2/nn)/nn)/value;
            if (value > 86)	    return (s0-(s1-(s2-s3/nn)/nn)/nn)/value;
            if (value > 27)	    return (s0-(s1-(s2-(s3-s4/nn)/nn)/nn)/nn)/value;

            return (s0-(s1-(s2-(s3-(s4-s5/nn)/nn)/nn)/nn)/nn)/value;
        }
    }

    // CONSTRAINT 8 <= alpha, beta
    static inline double stirling_correction_difference(double alpha, double beta) noexcept
    {
        double parameter_min = std::min(alpha, beta);
        double parameter_max = std::max(alpha, beta);

        double ratio = parameter_min / parameter_max;

        double reciprocal_sum = 1 / (ratio + 1);
        double normalized_ratio = ratio / (ratio + 1);
        double reciprocal_sum_squared = reciprocal_sum * reciprocal_sum;

        double partial_series[5];

        /* 3  */ partial_series[0] = reciprocal_sum + reciprocal_sum_squared + 1;
        /* 5  */ partial_series[1] = reciprocal_sum + reciprocal_sum_squared * partial_series[0] + 1;
        /* 7  */ partial_series[2] = reciprocal_sum + reciprocal_sum_squared * partial_series[1] + 1;
        /* 9  */ partial_series[3] = reciprocal_sum + reciprocal_sum_squared * partial_series[2] + 1;
        /* 11 */ partial_series[4] = reciprocal_sum + reciprocal_sum_squared * partial_series[3] + 1;

        double parameter_min_correction_difference = 0;
        double parameter_min_reciprocal = 1 / parameter_min;
        double min_scale = parameter_min_reciprocal * parameter_min_reciprocal;

        double parameter_max_correction_difference = 0;
        double parameter_max_reciprocal = 1 / parameter_max;
        double max_scale = parameter_max_reciprocal * parameter_max_reciprocal;

        for (unsigned i = 5; i > 0; --i)
        {
            parameter_min_correction_difference += sc[i];
            parameter_min_correction_difference *= min_scale;

            parameter_max_correction_difference += sc[i] * partial_series[i];
            parameter_max_correction_difference *= max_scale;
        }

        parameter_min_correction_difference += sc[0];
        parameter_min_correction_difference /= parameter_min;

        parameter_max_correction_difference += sc[0];
        parameter_max_correction_difference *= normalized_ratio / parameter_max;

        return parameter_max_correction_difference + parameter_min_correction_difference;
    }

    static inline double ln_beta(double alpha, double beta) noexcept
    {
        double parameter_min = std::min(alpha, beta);
        double parameter_max = std::max(alpha, beta);

        if (parameter_min < 1)
        {
            return parameter_max < 8 ?
                ln_gamma(parameter_min) + (ln_gamma(parameter_max) - ln_gamma(parameter_min + parameter_max)) :
                ln_gamma(parameter_min) + ln_gamma_right_over_gamma_sum(parameter_min, parameter_max);
        }
        else if (parameter_min < 2)
        {
            if (parameter_max <= 2)
            {
                return ln_gamma(parameter_min) + ln_gamma(parameter_max) - ln_gamma_sum(parameter_min, parameter_max);
            }

            if (parameter_max < 8)
            {
                double max_adjustment = 1;
                auto iterations = static_cast<unsigned>(parameter_max - 1);

                for (unsigned i = 1; i <= iterations; ++i)
                {
                    parameter_max += -1;
                    max_adjustment *= parameter_max / (parameter_min + parameter_max);
                }

                return log(max_adjustment) + (ln_gamma(parameter_min) + (ln_gamma(parameter_max) - ln_gamma_sum(parameter_min, parameter_max)));
            }

            return ln_gamma(parameter_min) + ln_gamma_right_over_gamma_sum(parameter_min, parameter_max);
        }
        else if (parameter_min < 8 && parameter_max <= 1'000)
        {
            double min_adjustment = 1;
            auto iterations = static_cast<unsigned>(parameter_min - 1);

            for (unsigned i = 1; i <= iterations; ++i)
            {
                parameter_min += -1;
                double ratio = parameter_min / parameter_max;
                min_adjustment *= ratio / (ratio + 1);
            }

            min_adjustment = log(min_adjustment);

            if (parameter_max >= 8)
            {
                return min_adjustment + ln_gamma(parameter_min)
                       + ln_gamma_right_over_gamma_sum(parameter_min, parameter_max);
            }

            double max_adjustment = 1;
            iterations = static_cast<unsigned>(parameter_max - 1);

            for (unsigned i = 1; i <= iterations; ++i)
            {
                parameter_max += -1;
                max_adjustment *= parameter_max / (parameter_min + parameter_max);
            }

            return min_adjustment
                + log(max_adjustment) + (ln_gamma(parameter_min)
                + (ln_gamma(parameter_max) - ln_gamma_sum(parameter_min, parameter_max)));
        }
        else if (parameter_min < 8)
        {
            double alpha_adjustment = 1;
            auto iterations = static_cast<unsigned>(parameter_min - 1);

            for (unsigned i = 1; i <= iterations; ++i)
            {
                parameter_min += -1.;
                alpha_adjustment *= parameter_min / (parameter_min / parameter_max + 1.);
            }

            return log(alpha_adjustment)
                - iterations * log(parameter_max) + (ln_gamma(parameter_min)
                                                     + ln_gamma_right_over_gamma_sum(parameter_min, parameter_max));
        }
        else
        {
            double // TODO rename
                    w = stirling_correction_difference(parameter_min, parameter_max),
                    ratio = parameter_min / parameter_max,
                    u = -(parameter_min - 0.5) * log(ratio / (ratio + 1)),
                    v = parameter_max * ln_value_plus_one(ratio);

            return u > v ?
                   log(parameter_max) * -0.5 + ln_sqrt_two_pi + w - v - u :
                   log(parameter_max) * -0.5 + ln_sqrt_two_pi + w - u - v ;
        }
    }

    // upper_limit^alpha * limit_compliment^beta / beta(alpha, beta)
    static inline double power_over_beta(double alpha, double beta, double upper_limit, double limit_compliment, bool log_scale) noexcept
    {
        if (upper_limit == 0 || limit_compliment == 0)
        {
            return numeric_zero(log_scale);
        }

        double parameter_min = std::min(alpha, beta);
        double parameter_max = std::max(alpha, beta);

        double c, t, u, v, z;

        if (parameter_min >= 8)
        {
            double h, x0, y0, lambda;

            if (alpha > beta)
            {
                h = beta / alpha;
                x0 = 1 / (h + 1);
                y0 = h / (h + 1);
                lambda = (alpha + beta) * limit_compliment - beta;
            }
            else
            {
                h = alpha / beta;
                x0 = h / (h + 1);
                y0 = 1 / (h + 1);
                lambda = alpha - (alpha + beta) * upper_limit;
            }

            double e = -lambda / alpha;

            u = fabs(e) > 0.6 ? e - log(upper_limit / x0) : value_minus_ln_value_plus_one(e);

            e = lambda / beta;

            v = fabs(e) > 0.6 ? e - log(limit_compliment / y0) : value_minus_ln_value_plus_one(e);

            z = log_scale ? -(alpha * u + beta * v) : exp(-(alpha * u + beta * v));

            return log_scale ?
                -ln_sqrt_two_pi + 0.5 * log(beta * x0) + z - stirling_correction_difference(alpha, beta) :
                inverse_sqrt_two_pi * sqrt(beta * x0) * z * exp(-stirling_correction_difference(alpha, beta));
        }

        double lnx, lny;

        if (upper_limit <= 0.375)
        {
            lnx = log(upper_limit);
            lny = ln_value_plus_one(-limit_compliment);
        }
        else if (limit_compliment > 0.375)
        {
            lnx = log(upper_limit);
            lny = log(limit_compliment);
        }
        else
        {
            lnx = ln_value_plus_one(-limit_compliment);
            lny = log(limit_compliment);
        }

        z = alpha * lnx + beta * lny;

        if (parameter_min >= 1)
        {
            z -= ln_beta(alpha, beta);

            return log_scale ? z : exp(z);
        }
        else if (parameter_max >= 8)
        {
            u = ln_gamma_value_plus_one(parameter_min)
                + ln_gamma_right_over_gamma_sum(parameter_min, parameter_max);

            return log_scale ? log(parameter_min) + (z - u) : parameter_min * exp(z - u);
        }
        else if (parameter_max <= 1)
        {
            double ans = log_scale ? z : exp(z);

            if (ans == numeric_zero(log_scale))
            {
                return ans;
            }

            z = alpha + beta > 1 ?
                (inverse_gamma_value_plus_one(alpha + beta - 1) + 1) / (alpha + beta) :
                inverse_gamma_value_plus_one(alpha + beta) + 1;

            c = (inverse_gamma_value_plus_one(alpha) + 1) * (inverse_gamma_value_plus_one(beta) + 1) / z;

            return log_scale ?
                ans + log(parameter_min * c) - log1p(parameter_min / parameter_max) :
                ans * (parameter_min * c) / (parameter_min / parameter_max + 1);
        }
        else
        {
            u = ln_gamma_value_plus_one(parameter_min);
            auto n = static_cast<int>(parameter_max - 1);

            if (n >= 1)
            {
                c = 1;

                for (unsigned i = 1; i <= n; ++i)
                {
                    parameter_max += -1;
                    c *= parameter_max / (parameter_min + parameter_max);
                }

                u += log(c);
            }

            z -= u;
            parameter_max += -1;

            t = alpha + beta > 1 ?
                (inverse_gamma_value_plus_one(parameter_min + parameter_max - 1) + 1) / (alpha + beta) :
                inverse_gamma_value_plus_one(alpha + beta) + 1;

            return log_scale ?
                log(parameter_min) + z + log1p(inverse_gamma_value_plus_one(parameter_max)) - log(t) :
                parameter_min * exp(z) * (inverse_gamma_value_plus_one(parameter_max) + 1) / t;
        }
    }

    // exp(mu) * upper_limit^alpha * limit_compliment^beta / beta(alpha, beta)
    static inline double exp_power_over_beta(double alpha, double beta,
                                             double upper_limit, double limit_compliment, int mu, bool log_scale) noexcept
    {
        double parameter_min = std::min(alpha, beta);
        double parameter_max = std::max(alpha, beta);

        double c, t, u, v, z;

        if (parameter_min >= 8)
        {
            double h, x0, y0, lambda;

            if (alpha > beta)
            {
                h = beta / alpha;
                x0 = 1 / (h + 1);
                y0 = h / (h + 1);
                lambda = (alpha + beta) * limit_compliment - beta;
            }
            else
            {
                h = alpha / beta;
                x0 = h / (h + 1);
                y0 = 1 / (h + 1);
                lambda = alpha - (alpha + beta) * upper_limit;
            }

            double lx0 = -log1p(beta / alpha);

            double e = -lambda / alpha;

            u = fabs(e) > 0.6 ? e - log(upper_limit / x0) : value_minus_ln_value_plus_one(e);

            e = lambda / beta;

            v = fabs(e) > 0.6 ? e - log(upper_limit / y0) : value_minus_ln_value_plus_one(e);

            z = exp_sum(mu, -(alpha * u + beta * v), log_scale);

            return log_scale ?
                log(inverse_sqrt_two_pi)+ (log(beta) + lx0) / 2 + z - stirling_correction_difference(alpha, beta):
                inverse_sqrt_two_pi * sqrt(beta * x0) * z * exp(-stirling_correction_difference(alpha, beta));
        }

        double lnx, lny;

        if (upper_limit <= 0.375)
        {
            lnx = log(upper_limit);
            lny = ln_value_plus_one(-upper_limit);
        }
        else if (limit_compliment > 0.375)
        {
            lnx = log(upper_limit);
            lny = log(limit_compliment);
        }
        else
        {
            lnx = ln_value_plus_one(-limit_compliment);
            lny = log(limit_compliment);
        }

        z = alpha * lnx + beta * lny;

        if (parameter_min >= 1)
        {
            z -= ln_beta(alpha, beta);

            return exp_sum(mu, z, log_scale);
        }
        else if (parameter_max >= 8)
        {
            u = ln_gamma_value_plus_one(parameter_min)
                + ln_gamma_right_over_gamma_sum(parameter_min, parameter_max);

            return log_scale ?
                log(parameter_min) + exp_sum(mu, z - u, true) :
                parameter_min  * exp_sum(mu, z - u, false);
        }
        else if (parameter_max <= 1)
        {
            double ans = exp_sum(mu, z, log_scale);

            if (ans == numeric_zero(log_scale))
            {
                return ans;
            }

            if (alpha + beta > 1)
            {
                u = alpha + beta - 1;
                z = (inverse_gamma_value_plus_one(u) + 1) / (alpha + beta);
            }
            else
            {
                z = inverse_gamma_value_plus_one(alpha + beta) + 1;
            }

            if (log_scale)
            {
                c = log1p(inverse_gamma_value_plus_one(alpha))
                        + log1p(inverse_gamma_value_plus_one(beta)) - log(z);

                return ans + log(parameter_min) + c - log1p(parameter_min / parameter_max);
            }
            else
            {
                c = (inverse_gamma_value_plus_one(alpha) + 1) * (inverse_gamma_value_plus_one(beta) + 1) / z;

                return ans * (parameter_min * c) / (parameter_min / parameter_max + 1);
            }
        }
        else
        {
            u = ln_gamma_value_plus_one(parameter_min);
            auto n = static_cast<int>(parameter_max - 1);

            if (n >= 1)
            {
                c = 1;

                for (unsigned i = 1; i <= n; ++i)
                {
                    parameter_max += -1;
                    c *= parameter_max / (parameter_min + parameter_max);
                }

                u += log(c);
            }

            z -= u;
            parameter_max += -1;

            t = parameter_min + parameter_max > 1 ?
                (inverse_gamma_value_plus_one(parameter_min + parameter_max - 1) + 1) / (alpha + parameter_max) :
                inverse_gamma_value_plus_one(parameter_min + parameter_max) + 1;

            return log_scale ?
                log(parameter_min) + exp_sum(mu, z, true) + log1p(inverse_gamma_value_plus_one(parameter_max)) - log(t) :
                parameter_min * exp_sum(mu, z, false) * (inverse_gamma_value_plus_one(parameter_max) + 1) / t;
        }
    }

    // CONSTRAINT min(tolerance * beta, tolerance) >= alpha
    // CONSTRAINT 1 >= beta * upper_limit
    // CONSTRAINT 0.5 >= upper_limit
    // i.e. when alpha is very small
    static inline double beta_asymptotic_power_series(double upper_limit, double alpha, double beta, double tolerance) noexcept
    {
        const double euler_constant = 0.577215664901533;

        double transform = upper_limit - (upper_limit * beta);

        double scale = beta * tolerance <= 0.02 ? // TODO rename
            log(upper_limit) + derivative_ln_gamma(beta) + euler_constant + transform :
            log(upper_limit * beta) + euler_constant + transform;

        double scaled_tolerance = tolerance * 5 * fabs(scale);

        double adjustment, sum = 0, iteration = 1;

        do
        {
            iteration += 1;
            transform *= upper_limit - (upper_limit * beta) / (iteration + 1);
            adjustment = transform / (iteration + 1);
            sum += adjustment;
        }
        while (iteration < max_iteration_count && fabs(adjustment) > scaled_tolerance);

        return -alpha * (scale + sum);
    }

    // CONSTRAINT beta <= 1 && 15 <= alpha
    static inline double beta_asymptotic_expansion(double alpha, double beta, double upper_limit, double limit_compliment,
                                                   double p_lower_tail, double tolerance, bool log_scale) noexcept
    {
        const unsigned iterations = 30;

        double c[iterations], d[iterations];

        double beta_minus_one = beta - 0.5 - 0.5;
        double nu = alpha + beta_minus_one * 0.5;
        double lnx = (limit_compliment > 0.375) ? log(upper_limit) : ln_value_plus_one(-limit_compliment);
        double upper_limit_adjusted = -nu * lnx;

        if (beta * upper_limit_adjusted == 0)
        {
            std::cout << "underflow in computing beta_asymptotic_expansion \n";
            return 1;
        }

        double log_r = log(beta) + log1p(inverse_gamma_value_plus_one(beta)) + beta * log(upper_limit_adjusted) + nu * lnx;
        double log_u = log_r - (ln_gamma_right_over_gamma_sum(beta, alpha) + beta * log(nu));
        double u = exp(log_u);

        if (log_u == negative_infinity)
        {
            std::cout << "underflow in computing beta_asymptotic_expansion \n";
            return 2;
        }

        bool u_0 = u == 0;

        double l = log_scale ?
                   p_lower_tail == negative_infinity ? 0 : exp(p_lower_tail - log_u) :
                   p_lower_tail == 0 ? 0 : exp(log(p_lower_tail) - log_u);

        double q_r = scaled_complement_incomplete_gamma_ratio(beta, upper_limit_adjusted, tolerance, log_r);
        double v = 0.25 / (nu * nu);
        double t2 = lnx * 0.25 * lnx;
        double j = q_r;
        double sum = j;
        double t = 1, cn = 1, n2 = 0;

        for (int n = 1; n <= iterations; ++n)
        {
            double beta_plus_n2 = beta + n2;
            j = (beta_plus_n2 * (beta_plus_n2 + 1) * j + (upper_limit_adjusted + beta_plus_n2 + 1) * t) * v;
            n2 += 2;
            t *= t2;
            cn /= n2 * (n2 + 1);
            int n_minus_1 = n - 1;
            c[n_minus_1] = cn;
            double s = 0;

            if (n > 1)
            {
                double coefficient = beta - n;

                for (int i = 1; i <= n_minus_1; ++i)
                {
                    s += coefficient * c[i - 1] * d[n_minus_1 - i];
                    coefficient += beta;
                }
            }

            d[n_minus_1] = beta_minus_one * cn + s / n;
            double dj = d[n_minus_1] * j;
            sum += dj;

            if (sum <= 0.)
            {
                std::cout << "the expansion cannot be computed in beta_asymptotic_expansion \n";
                return 3;
            }
            
            if (fabs(dj) <= tolerance * (sum + l)) break;

            if (n == iterations)
            {
                std::cout << "the expansion failed to converge in beta_asymptotic_expansion \n";
                return 4;
            }
        }

        return log_scale ? ln_sum_of_ln(p_lower_tail, log_u + log(sum)) : p_lower_tail + (u_0 ? exp(log_u + log(sum)) : u * sum);
    }

    // CONSTRAINT 15 <= alpha, beta && 0 < lambda
    static inline double beta_asymptotic_expansion_large(double alpha, double beta, double lambda, double tolerance, bool log_scale) noexcept
    {// TODO clean
        const unsigned iterations = 20;

        const double two_over_sqrt_pi = 1.12837916709551;
        const double ln_two_over_sqrt_pi = 0.120782237635245;
        const double two_power_minus_three_over_two = .353553390593274;

        double a0[iterations + 1], b0[iterations + 1], c[iterations + 1], d[iterations + 1];

        double f = alpha * value_minus_ln_value_plus_one(-lambda / alpha)
                   + beta * value_minus_ln_value_plus_one(lambda / beta);

        double t = log_scale ? -f : exp(-f);

        if (!log_scale && t == 0)
        {
            return 0;
        }

        double z0 = sqrt(f),
               z = z0 / two_power_minus_three_over_two * 0.5,
               z2 = f + f,
               h, r0, r1, w0;

        if (alpha < beta)
        {
            h = alpha / beta;
            r0 = 1 / (h + 1);
            r1 = (beta - alpha) / beta;
            w0 = 1 / sqrt(alpha * (h + 1));
        }
        else
        {
            h = beta / alpha;
            r0 = 1 / (h + 1);
            r1 = (beta - alpha) / alpha;
            w0 = 1 / sqrt(beta * (h + 1));
        }

        a0[0] = r1 * .66666666666666663;
        c[0] = a0[0] * -0.5;
        d[0] = -c[0];

        double j0 = 0.5 / two_over_sqrt_pi * scaled_complementary_error_function(true, z0),
                j1 = two_power_minus_three_over_two,
                sum = j0 + d[0] * w0 * j1;

        double s = 1,
               h2 = h * h,
               hn = 1,
               w = w0,
               znm1 = z,
               zn = z2;

        for (int n = 2; n <= iterations; n += 2)
        {
            hn *= h2;
            a0[n - 1] = r0 * 2 * (h * hn + 1) / (n + 2);
            int np1 = n + 1;
            s += hn;
            a0[np1 - 1] = r1 * 2 * s / (n + 3);

            for (int i = n; i <= np1; ++i)
            {
                double r = (i + 1) * -0.5;
                b0[0] = r * a0[0];

                for (int m = 2; m <= i; ++m)
                {
                    double b_sum = 0.;

                    for (int j = 1; j <= m - 1; ++j)
                    {
                        int mmj = m - j;
                        b_sum += (j * r - mmj) * a0[j - 1] * b0[mmj - 1];
                    }

                    b0[m - 1] = r * a0[m - 1] + b_sum / m;
                }

                c[i - 1] = b0[i - 1] / (i + 1.);

                double d_sum = 0.;

                for (int j = 1; j <= i - 1; ++j)
                {
                    d_sum += d[i - j - 1] * c[j - 1];
                }

                d[i - 1] = -(d_sum + c[i - 1]);
            }

            j0 = two_power_minus_three_over_two * znm1 + (n - 1.) * j0;
            j1 = two_power_minus_three_over_two * zn + n * j1;
            znm1 = z2 * znm1;
            zn = z2 * zn;
            w *= w0;
            double t0 = d[n - 1] * w * j0;
            w *= w0;
            double t1 = d[np1 - 1] * w * j1;
            sum += t0 + t1;

            if (fabs(t0) + fabs(t1) <= tolerance * sum) break;
        }

        return log_scale ?
            ln_two_over_sqrt_pi + t - stirling_correction_difference(alpha, beta) + log(sum) :
            two_over_sqrt_pi * t * exp(-stirling_correction_difference(alpha, beta)) * sum;
    }

    // CONSTRAINT min(tolerance, tolerance * alpha) > beta
    // CONSTRAINT 0.5 >= upper_limit
    static inline double beta_front_power_series(double upper_limit, double alpha, double beta, double tolerance, bool log_scale) noexcept
    {
        double result;

        if (log_scale)
        {
            result = alpha * log(upper_limit);
        }
        else if (alpha > tolerance * 0.001)
        {
            double transform = alpha * log(upper_limit);

            if (transform < max_exp_argument(true))
            {
                return 0;
            }

            result = exp(transform);
        }
        else
        {
            result = 1;
        }

        if (log_scale)
        {
            result += log(beta) - log(alpha);
        }
        else
        {
            result *= beta / alpha;
        }

        double scaled_tolerance = tolerance / alpha;

        double adjustment, sum = 0, transform, iteration = 0;

        transform = upper_limit;
        sum = transform / (alpha + 1);

        do
        {
            iteration += 1;
            transform *= upper_limit;
            adjustment = transform / (alpha + 1 + iteration);
            sum += adjustment;
        }
        while (iteration < max_iteration_count && fabs(adjustment) > scaled_tolerance);

        if (log_scale)
        {
            result += log1p(alpha * sum);
        }
        else
        {
            result *= alpha * sum + 1.;
        }

        return result;
    }

    // CONSTRAINT 1 >= beta or 0.7 >= beta * upper_limit
    static inline double beta_power_series(double upper_limit, double alpha, double beta, double tolerance, bool log_scale) noexcept
    {
        if (upper_limit == 0) return numeric_zero(log_scale);

        double result;

        double scale, adjustment;

        double parameter_min = std::min(alpha, beta);
        double parameter_max = std::max(alpha, beta);

        if (parameter_min >= 1)
        {
            double transform = alpha * log(upper_limit) - ln_beta(alpha, beta);
            result = log_scale ? transform - log(alpha) : exp(transform) / alpha;
        }
        else if (parameter_max <= 1)
        {
            if (log_scale)
            {
                result = alpha * log(upper_limit);
            }
            else
            {
                result = pow(upper_limit, alpha);

                if (result == 0) return result;
            }

            double transform = alpha + beta > 1 ?
                (inverse_gamma_value_plus_one(alpha + beta - 1) + 1) / (alpha + beta):
                inverse_gamma_value_plus_one(alpha + beta) + 1;

            scale = (inverse_gamma_value_plus_one(alpha) + 1) * (inverse_gamma_value_plus_one(beta) + 1) / transform;

            if (log_scale)
            {
                result += log(scale * (beta / (alpha + beta)));
            }
            else
            {
                result *= scale * (beta / (alpha + beta));
            }
        }
        else if (parameter_max < 8)
        {
            adjustment = ln_gamma_value_plus_one(parameter_min);

            auto adjustment_count = static_cast<int>(parameter_max - 1);

            if (adjustment_count >= 1)
            {
                scale = 1;

                for (unsigned i = 1; i <= adjustment_count; ++i)
                {
                    parameter_max += -1;
                    scale *= parameter_max / (parameter_min + parameter_max);
                }

                adjustment += log(scale);
            }

            double transform = alpha * log(upper_limit) - adjustment;

            parameter_max += -1;

            double term = parameter_min + parameter_max > 1 ?
                (inverse_gamma_value_plus_one(parameter_min + parameter_max - 1) + 1) / (parameter_min + parameter_max) :
                inverse_gamma_value_plus_one(parameter_min + parameter_max) + 1;

            if(log_scale)
            {
                result = transform + log(parameter_min / alpha) + log1p(inverse_gamma_value_plus_one(parameter_max)) - log(term);
            }
            else
            {
                result = exp(transform) * (parameter_min / alpha) * (inverse_gamma_value_plus_one(parameter_max) + 1) / term;
            }
        }
        else
        {
            adjustment = ln_gamma_value_plus_one(parameter_min) +
                    ln_gamma_right_over_gamma_sum(parameter_min, parameter_max);

            double transform = alpha * log(upper_limit) - adjustment;

            if(log_scale)
            {
                result = transform + log(parameter_min / alpha);
            }
            else
            {
                result = parameter_min / alpha * exp(transform);
            }
        }

        if (result == numeric_zero(log_scale) || (!log_scale && alpha <= tolerance * 0.1)) return result;

        scale = 1;
        double scaled_tolerance = tolerance / alpha, iteration = 0, sum = 0;

        do
        {
            iteration += 1;
            scale *= (0.5 - beta / iteration + 0.5) * upper_limit;
            adjustment = scale / (alpha + iteration);
            sum += adjustment;
        }
        while (iteration < max_iteration_count && fabs(adjustment) > scaled_tolerance);

        if (iteration == max_iteration_count) { std::cout << "beta_power_series failed to converge \n"; }

        if (log_scale)
        {
            if (alpha * sum > -1) result += log1p(alpha * sum);
            else result = negative_infinity;
        }
        else if (alpha * sum > -1)
        {
            result *= (alpha * sum + 1);
        }
        else
        {
            result = 0;
        }

        return result;
    }

    // CONSTRAINT 1 < alpha, beta
    static inline double beta_continued_fraction_expansion(double upper_limit, double limit_compliment,
                                                           double alpha, double beta, double lambda,
                                                           double tolerance, bool log_scale) noexcept
    {
        double brc = power_over_beta(alpha, beta, upper_limit, limit_compliment, log_scale);

        if (!log_scale && brc == 0) return 0;

        double lambda_plus_one = lambda + 1;
        double beta_alpha_ratio = beta / alpha;
        double one_plus_inverse_alpha = 1 / alpha + 1;
        double limit_compliment_plus_one = limit_compliment + 1;

        double iteration = 0, transform = 1, weight, scale;

        double sum = alpha + 1;
        double alpha_update, beta_update;
        double numerator_previous = 0, denominator_previous = 1;
        double ratio = one_plus_inverse_alpha / lambda_plus_one, ratio_previous;
        double numerator = 1, denominator = lambda_plus_one / one_plus_inverse_alpha;

        do
        {
            iteration += 1;
            weight = iteration * (beta - iteration) * upper_limit;
            scale = alpha / sum;
            alpha_update = transform * (transform + beta_alpha_ratio) * scale * scale * (weight * upper_limit);

            transform = 1 + iteration / alpha;
            scale = transform / (one_plus_inverse_alpha + 2 * (transform - 1));

            beta_update = iteration + weight / sum + scale * (lambda_plus_one + iteration * limit_compliment_plus_one);

            sum += 2;

            auto temp = alpha_update * numerator_previous + beta_update * numerator;
            numerator_previous = numerator; numerator = temp;

            temp = alpha_update * denominator_previous + beta_update * denominator;
            denominator_previous = denominator; denominator = temp;

            ratio_previous = ratio;
            ratio = numerator / denominator;

            if (fabs(ratio - ratio_previous) <= tolerance * ratio) break;

            numerator_previous /= denominator;
            denominator_previous /= denominator;

            numerator = ratio;
            denominator = 1;
        }
        while (iteration < 10'000);

        return (log_scale ? brc + log(ratio) : brc * ratio);
    }

    static inline double beta_update(double upper_limit, double limit_compliment, double alpha, double beta,
                                     int increment, double tolerance, bool log_scale) noexcept
    {
        double result;
        int scaling_factor;
        double exp_scaling_factor;

        if (increment > 1 && alpha >= 1 && alpha + beta >= (alpha + 1) * 1.1)
        {
            scaling_factor = std::min(
                static_cast<int>(max_exp_argument(false)),
                static_cast<int>(fabs(max_exp_argument(true)))
            );

            exp_scaling_factor = exp(-static_cast<double>(scaling_factor));
        }
        else
        {
            scaling_factor = 0;
            exp_scaling_factor = 1;
        }

        result = log_scale ?
             exp_power_over_beta(alpha, beta, upper_limit, limit_compliment, scaling_factor, true) - log(alpha) :
             exp_power_over_beta(alpha, beta, upper_limit, limit_compliment, scaling_factor, false) / alpha;

        if (increment == 1 || result == numeric_zero(log_scale)) return result; // TODO test

        double accumulated_weights = exp_scaling_factor;

        unsigned initial_term_limit = 0;

        if (beta > 1)
        {
            if (limit_compliment > 1e-4)
            {
                auto significant_term_cutoff = static_cast<int>((beta - 1) * upper_limit / limit_compliment - alpha);

                if (significant_term_cutoff >= 1)
                {
                    initial_term_limit = std::min(significant_term_cutoff, increment - 1);
                }
            }
            else
            {
                initial_term_limit = increment - 1;
            }

            for (unsigned iteration = 0; iteration < initial_term_limit; ++iteration)
            {
                auto _iteration = static_cast<double>(iteration);

                exp_scaling_factor *= (alpha + beta + _iteration) / (alpha + 1 + _iteration) * upper_limit;
                accumulated_weights += exp_scaling_factor;
            }
        }

        for (unsigned iteration = initial_term_limit; iteration < increment - 1; ++iteration)
        {
            auto _iteration = static_cast<double>(iteration);

            exp_scaling_factor *= (alpha + beta + _iteration) / (alpha + 1 + _iteration) * upper_limit;
            accumulated_weights += exp_scaling_factor;

            if (exp_scaling_factor <= tolerance * accumulated_weights) break;
        }

        if(log_scale)
        {
            result += log(accumulated_weights);
        }
        else
        {
            result *= accumulated_weights;
        }

        return result;
    }

    struct IncompleteBetaFunctionSolver
    {
        int n = 0;
        double alpha, beta;
        bool log_scale, swap = false, beta_updated = false;
        double upper_limit, limit_compliment, p_lower_tail = 1e-15, p_upper_tail = 1e-15, tolerance = 1e-15;
        double alpha_working = 0, beta_working = 0, upper_limit_working = 0, limit_compliment_working = 0, lambda = 0;

        explicit IncompleteBetaFunctionSolver(double upper_limit, double alpha, double beta, bool log_scale) noexcept;

        inline void set_swap() noexcept;
        inline void set_no_swap() noexcept;

        inline void reverse_swap() noexcept;

        inline void compute_from_lower_tail() noexcept;
        inline void compute_from_lower_tail_beta_power_series() noexcept;
        inline void compute_from_upper_tail() noexcept;
        inline void compute_from_upper_tail_beta_power_series() noexcept;
        inline void compute_from_upper_tail_log() noexcept;

        inline void compute_from_very_small_alpha_and_beta() noexcept;
        inline void compute_from_large_alpha_and_beta() noexcept;
        inline void compute_from_small_alpha_or_beta() noexcept;
    };

    IncompleteBetaFunctionSolver::IncompleteBetaFunctionSolver(double upper_limit, double alpha, double beta, bool log_scale) noexcept
        : alpha(alpha), beta(beta), log_scale(log_scale), upper_limit(upper_limit), limit_compliment(0.5 - upper_limit + 0.5)
        , p_lower_tail(log_scale ? negative_infinity : 0), p_upper_tail(log_scale ? negative_infinity : 0)
    {
        if (std::max(alpha, beta) < tolerance * 0.001)
        {
            compute_from_very_small_alpha_and_beta();
        }
        else if (std::min(alpha, beta) <= 1)
        {
            compute_from_small_alpha_or_beta();
        }
        else
        {
            compute_from_large_alpha_and_beta();
        }
    }

    void IncompleteBetaFunctionSolver::set_swap() noexcept
    {
        alpha_working = beta; upper_limit_working = limit_compliment;
        beta_working = alpha; limit_compliment_working = upper_limit;
    }

    void IncompleteBetaFunctionSolver::set_no_swap() noexcept
    {
        alpha_working = alpha; upper_limit_working = upper_limit;
        beta_working = beta; limit_compliment_working = limit_compliment;
    }

    void IncompleteBetaFunctionSolver::reverse_swap() noexcept
    {
        if (!swap) return;

        std::swap(p_lower_tail, p_upper_tail);
    }

    void IncompleteBetaFunctionSolver::compute_from_lower_tail() noexcept
    {
        if(log_scale)
        {
            p_upper_tail = log1p(-p_lower_tail);
            p_lower_tail = log(p_lower_tail);
        }
        else
        {
            p_upper_tail = 0.5 - p_lower_tail + 0.5;
        }

        return reverse_swap();
    }

    void IncompleteBetaFunctionSolver::compute_from_lower_tail_beta_power_series() noexcept
    {
        p_lower_tail = beta_power_series(upper_limit_working, alpha_working, beta_working, tolerance, log_scale);
        p_upper_tail = log_scale ? stable_ln_of_one_minus_e_to_the_x(p_lower_tail) : 0.5 - p_lower_tail + 0.5;

        return reverse_swap();
    }

    void IncompleteBetaFunctionSolver::compute_from_upper_tail_beta_power_series() noexcept
    {
        p_upper_tail = beta_power_series(limit_compliment_working, beta_working, alpha_working, tolerance, log_scale);
        p_lower_tail  = log_scale ? stable_ln_of_one_minus_e_to_the_x(p_upper_tail) : 0.5 - p_upper_tail + 0.5;

        return reverse_swap();
    }

    void IncompleteBetaFunctionSolver::compute_from_upper_tail() noexcept
    {
        if(log_scale)
        {
            p_lower_tail = log1p(-p_upper_tail);
            p_upper_tail = log(p_upper_tail);
        }
        else
        {
            p_lower_tail = 0.5 - p_upper_tail + 0.5;
        }

        return reverse_swap();
    }

    void IncompleteBetaFunctionSolver::compute_from_upper_tail_log() noexcept
    {
        if(log_scale)
        {
            p_lower_tail = stable_ln_of_one_minus_e_to_the_x(p_upper_tail);
        }
        else
        {
            p_lower_tail  = -stable_e_to_the_x_minus_one(p_upper_tail);
            p_upper_tail = exp(p_upper_tail);
        }

        return reverse_swap();
    }

    void IncompleteBetaFunctionSolver::compute_from_very_small_alpha_and_beta() noexcept
    {
        if(!log_scale)
        {
            p_lower_tail = beta / (alpha + beta);
            p_upper_tail = alpha / (alpha + beta);
        }
        else if(alpha < beta)
        {
            p_lower_tail = log1p(-alpha / (alpha + beta));
            p_upper_tail = log(alpha / (alpha + beta));
        }
        else
        {
            p_lower_tail = log(beta / (alpha + beta));
            p_upper_tail = log1p(-beta / (alpha + beta));
        }
    }

    void IncompleteBetaFunctionSolver::compute_from_small_alpha_or_beta() noexcept
    {
        swap = upper_limit > 0.5;

        if (swap)
        {
            set_swap();
        }
        else
        {
            set_no_swap();
        }

        if (beta_working < std::min(tolerance, tolerance * alpha_working))
        {
            p_lower_tail = beta_front_power_series(upper_limit_working, alpha_working, beta_working, tolerance, log_scale);
            p_upper_tail = log_scale ? stable_ln_of_one_minus_e_to_the_x(p_lower_tail) : 0.5 - p_lower_tail + 0.5;

            return reverse_swap();
        }

        if (alpha_working < std::min(tolerance, tolerance * beta_working) && beta_working * upper_limit_working <= 1)
        {
            p_upper_tail = beta_asymptotic_power_series(upper_limit_working, alpha_working, beta_working, tolerance);

            return compute_from_upper_tail();
        }

        if (std::max(alpha_working, beta_working) > 1)
        {
            if (beta_working <= 1)
            {
                return compute_from_lower_tail_beta_power_series();
            }

            if (upper_limit_working >= 0.29)
            {
                return compute_from_upper_tail_beta_power_series();
            }

            if (upper_limit_working < 0.1 && pow(upper_limit_working * beta_working, alpha_working) <= 0.7)
            {
                return compute_from_lower_tail_beta_power_series();
            }

            if (beta_working > 15)
            {
                p_upper_tail = 0;
                beta_updated = false;
            }
            else
            {
                beta_updated = true;
            }
        }
        else
        {
            if (alpha_working >= std::min(0.2, beta_working))
            {
                return compute_from_lower_tail_beta_power_series();
            }

            if (pow(upper_limit_working, alpha_working) <= 0.9)
            {
                return compute_from_lower_tail_beta_power_series();
            }

            if (upper_limit_working >= 0.3)
            {
                return compute_from_upper_tail_beta_power_series();
            }

            beta_updated = true;
        }

        if (beta_updated)
        {
            n = 20;

            p_upper_tail = beta_update(
                limit_compliment_working,
                upper_limit_working,
                beta_working,
                alpha_working,
                n, tolerance, false
            );

            beta_working += n;
        }

        p_upper_tail = beta_asymptotic_expansion(
            beta_working,
            alpha_working,
            limit_compliment_working,
            upper_limit_working,
            p_upper_tail,
            tolerance * 15, false
        );

        if (p_upper_tail == 0 || (0 < p_upper_tail && p_upper_tail < DBL_MIN))
        {
            if (beta_updated)
            {
                p_upper_tail = beta_update(
                    beta_working - n,
                    alpha_working,
                    limit_compliment_working,
                    upper_limit_working,
                    n, tolerance, true
                );
            }
            else
            {
                p_upper_tail = negative_infinity;
            }

            p_upper_tail = beta_asymptotic_expansion(
                beta_working,
                alpha_working,
                limit_compliment_working,
                upper_limit_working,
                p_upper_tail,
                tolerance * 15, true
            );

            return compute_from_upper_tail_log();
        }

        return compute_from_upper_tail();
    }

    void IncompleteBetaFunctionSolver::compute_from_large_alpha_and_beta() noexcept
    {
        lambda = is_finite(alpha + beta)
                 ? ((alpha > beta) ? (alpha + beta) * limit_compliment - beta : alpha - (alpha + beta) * upper_limit)
                 : alpha * limit_compliment - beta * upper_limit;

        swap = lambda < 0;

        if (swap)
        {
            lambda = -lambda;
            set_swap();
        }
        else
        {
            set_no_swap();
        }

        if (beta_working < 40)
        {
            if (beta_working * upper_limit_working <= 0.7 || (log_scale && lambda > 650))
            {
                return compute_from_lower_tail_beta_power_series();
            }
            else
            {
                n = static_cast<int>(beta_working);
                beta_working -= n;

                if (beta_working == 0)
                {
                    n -= 1;
                    beta_working = 1;
                }

                p_lower_tail = beta_update(
                    limit_compliment_working,
                    upper_limit_working,
                    beta_working,
                    alpha_working,
                    n,
                    tolerance,
                    false
                );

                if (p_lower_tail < DBL_MIN && log_scale)
                {
                    beta_working += n;

                    return compute_from_lower_tail_beta_power_series();
                }

                if (upper_limit_working <= 0.7)
                {
                    p_lower_tail += beta_power_series(
                        upper_limit_working,
                        alpha_working,
                        beta_working,
                        tolerance,
                        false
                    );

                    return compute_from_lower_tail();
                }

                if (alpha_working <= 15)
                {
                    n = 20;

                    p_lower_tail += beta_update(
                        upper_limit_working,
                        limit_compliment_working,
                        alpha_working,
                        beta_working,
                        n, tolerance, false
                    );

                    alpha_working += n;
                }

                p_lower_tail = beta_asymptotic_expansion(
                    alpha_working,
                    beta_working,
                    upper_limit_working,
                    limit_compliment_working,
                    p_lower_tail,
                    tolerance * 15,
                    false
                );

                return compute_from_lower_tail();
            }
        }
        else if ((alpha_working > beta_working && (beta_working <= 100 || lambda > beta_working * 0.03)) ||
                (alpha_working <= beta_working && (alpha_working <= 100 || lambda > alpha_working * 0.03)))
        {
            p_lower_tail = beta_continued_fraction_expansion(
                upper_limit_working,
                limit_compliment_working,
                alpha_working,
                beta_working,
                lambda,
                tolerance * 15,
                log_scale
            );

            p_upper_tail = log_scale ?
               stable_ln_of_one_minus_e_to_the_x(p_lower_tail) :
               0.5 - p_lower_tail + 0.5;

            return reverse_swap();
        }

        p_lower_tail = beta_asymptotic_expansion_large(
            alpha_working,
            beta_working,
            lambda,
            tolerance * 100,
            log_scale
        );

        p_upper_tail = log_scale ? stable_ln_of_one_minus_e_to_the_x(p_lower_tail) : 0.5 - p_lower_tail + 0.5;

        return reverse_swap();
    }

    struct NormalDistributionSolver
    {
        bool log_scale;
        double value, distance;
        double p_lower_tail = 0, p_upper_tail = 0;
        double tolerance = DBL_EPSILON * 0.5;

        explicit NormalDistributionSolver(double value, bool log_scale) noexcept;

        inline void solve_small() noexcept;
        inline void solve_medium() noexcept;
        inline void solve_large() noexcept;
        inline void solve_extremes() noexcept;

        inline void swap_tail() noexcept;
        inline void decompose_and_compute(double partial_value, double adjustment) noexcept;

        //<editor-fold desc="constants">
        #pragma region constants

        const double a[5] =
        {
            2.2352520354606839287,
            161.02823106855587881,
            1067.6894854603709582,
            18154.981253343561249,
            0.065682337918207449113
        };

        const double b[4] =
        {
            47.20258190468824187,
            976.09855173777669322,
            10260.932208618978205,
            45507.789335026729956
        };

        const double c[9] =
        {
            0.39894151208813466764,
            8.8831497943883759412,
            93.506656132177855979,
            597.27027639480026226,
            2494.5375852903726711,
            6848.1904505362823326,
            11602.651437647350124,
            9842.7148383839780218,
            1.0765576773720192317e-8
        };

        const double d[8] =
        {
            22.266688044328115691,
            235.38790178262499861,
            1519.377599407554805,
            6485.558298266760755,
            18615.571640885098091,
            34900.952721145977266,
            38912.003286093271411,
            19685.429676859990727
        };

        const double p[6] =
        {
            0.21589853405795699,
            0.1274011611602473639,
            0.022235277870649807,
            0.001421619193227893466,
            2.9112874951168792e-5,
            0.02307344176494017303
        };

        const double q[5] =
        {
            1.28426009614491121,
            0.468238212480865118,
            0.0659881378689285515,
            0.00378239633202758244,
            7.29751555083966205e-5
        };

        #pragma endregion
        //</editor-fold>
    };

    NormalDistributionSolver::NormalDistributionSolver(double value, bool log_scale) noexcept
        : value(value), distance(std::fabs(value)), log_scale(log_scale)
    {
        if (distance <= 0.67448975)
        {
            solve_small();
        }
        else if (distance <= sqrt_32)
        {
            solve_medium();
        }
        else if ((log_scale && distance < 1e170) || (distance < 37.5193))
        {
            solve_large();
        }
        else
        {
            solve_extremes();
        }
    }

    void NormalDistributionSolver::solve_small() noexcept
    {
        double numerator = 0;
        double denominator = 0;

        if (distance > tolerance)
        {
            double squared = value * value;

            numerator = a[4] * squared;
            denominator = squared;

            for (unsigned i = 0; i < 3; ++i)
            {
                numerator = (numerator + a[i]) * squared;
                denominator = (denominator + b[i]) * squared;
            }
        }

        double adjustment = value * (numerator + a[3]) / (denominator + b[3]);

        p_lower_tail = 0.5 + adjustment;
        p_upper_tail = 0.5 - adjustment;

        if (log_scale)
        {
            p_lower_tail = log(p_lower_tail);
            p_upper_tail = log(p_upper_tail);
        }
    }

    void NormalDistributionSolver::solve_medium() noexcept
    {
        double numerator = c[8] * distance;
        double denominator = distance;

        for (unsigned i = 0; i < 7; ++i)
        {
            numerator = (numerator + c[i]) * distance;
            denominator = (denominator + d[i]) * distance;
        }

        double adjustment = (numerator + c[7]) / (denominator + d[7]);

        decompose_and_compute(distance, adjustment);
        swap_tail();
    }

    void NormalDistributionSolver::solve_large() noexcept
    {
        double inverse_square = 1.0 / (value * value);
        double numerator = p[5] * inverse_square;
        double denominator = inverse_square;

        for (unsigned i = 0; i < 4; ++i)
        {
            numerator = (numerator + p[i]) * inverse_square;
            denominator = (denominator + q[i]) * inverse_square;
        }

        double adjustment = inverse_square * (numerator + p[4]) / (denominator + q[4]);
        adjustment = (inverse_sqrt_two_pi - adjustment) / distance;

        decompose_and_compute(value, adjustment);
        swap_tail();
    }

    void NormalDistributionSolver::solve_extremes() noexcept
    {
        if (value > 0)
        {
            p_lower_tail = numeric_one(log_scale);
            p_upper_tail = numeric_zero(log_scale);
        }
        else
        {
            p_lower_tail = numeric_zero(log_scale);
            p_upper_tail = numeric_one(log_scale);
        }
    }

    void NormalDistributionSolver::swap_tail() noexcept
    {
        if (value <= 0) return;

        std::swap(p_lower_tail, p_upper_tail);
    }

    void NormalDistributionSolver::decompose_and_compute(double partial_value, double adjustment) noexcept
    {
        double truncated = trunc(partial_value * 16) / 16;
        double residual = (partial_value - truncated) * (partial_value + truncated);

        if (log_scale)
        {
            p_lower_tail = (-truncated * truncated * 0.5) + (-residual * 0.5) + log(adjustment);
            p_upper_tail = log1p(-exp(-truncated * truncated * 0.5) * exp(-residual * 0.5) * adjustment);
        }
        else
        {
            p_lower_tail = exp(-truncated * truncated * 0.5) * exp(-residual * 0.5) * adjustment;
            p_upper_tail = 1.0 - p_lower_tail;
        }
    }

    double normal_distribution_pdf(double value, double mean, double standard_deviation, bool lower_tail, bool log_scale)
    {
        double value_working, value_working_compliment, adjusted_value, r;

        if (log_scale)
        {
	        if (value > 0)
            {
                return NAN;
            }

	        if (value == 0)
            {
	            return lower_tail ? positive_infinity : negative_infinity;
            }

            if (value == negative_infinity)
            {
                return lower_tail ? negative_infinity : positive_infinity;
            }
        }
        else
        {
            if (value < 0 || value > 1)
            {
                return NAN;
            }

            if (value == 0)
            {
                return lower_tail ? negative_infinity : positive_infinity;
            }

            if (value == 1)
            {
	            return lower_tail ? positive_infinity : negative_infinity;
            }
        }

        if (standard_deviation < 0) return NAN;
        if (standard_deviation == 0) return mean;

        if (log_scale)
        {
            value_working = lower_tail ? exp(value) : - expm1(value);
        }
        else
        {
            value_working = lower_tail ? value : 0.5 - value + 0.5;
        }

        value_working_compliment = value_working - 0.5;

        if (fabs(value_working_compliment) <= 0.425)
        {
            r = 0.180625 - value_working_compliment * value_working_compliment;
            adjusted_value = value_working_compliment * (((((((r * 2509.0809287301226727 +
                                                               33430.575583588128105) * r + 67265.770927008700853) * r +
                                                             45921.953931549871457) * r + 13731.693765509461125) * r +
                                                           1971.5909503065514427) * r + 133.14166789178437745) * r +
                                                         3.387132872796366608)
                             / (((((((r * 5226.495278852854561 +
                                      28729.085735721942674) * r + 39307.89580009271061) * r +
                                    21213.794301586595867) * r + 5394.1960214247511077) * r +
                                  687.1870074920579083) * r + 42.313330701600911252) * r + 1);
        }
        else
        {
            if (log_scale && ((lower_tail && value_working_compliment <= 0) || (!lower_tail && value_working_compliment > 0)))
            {
                r = value;
            }
            else if (value_working_compliment > 0)
            {
                if (log_scale)
                {
                    r = log(lower_tail ? -expm1(value) : exp(value));
                }
                else
                {
                    r = log(lower_tail ? 0.5 - value + 0.5 : value);
                }
            }
            else
            {
                r = log(value_working);
            }

            r = sqrt(-r);

            if (r <= 5)
            {
                r += -1.6;
                adjusted_value = (((((((r * 7.7454501427834140764e-4 +
                                        .0227238449892691845833) * r + .24178072517745061177) *
                                      r + 1.27045825245236838258) * r +
                                     3.64784832476320460504) * r + 5.7694972214606914055) *
                                   r + 4.6303378461565452959) * r +
                                  1.42343711074968357734)
                                 / (((((((r *
                                          1.05075007164441684324e-9 + 5.475938084995344946e-4) *
                                         r + .0151986665636164571966) * r +
                                        .14810397642748007459) * r + .68976733498510000455) *
                                      r + 1.6763848301838038494) * r +
                                     2.05319162663775882187) * r + 1);
            }
            else if (r >= 816)
            {
                adjusted_value = r * M_SQRT2;
            }
            else
            {
                r += -5;
                adjusted_value = (((((((r * 2.01033439929228813265e-7 +
                                        2.71155556874348757815e-5) * r +
                                       .0012426609473880784386) * r + .026532189526576123093) *
                                     r + .29656057182850489123) * r +
                                    1.7848265399172913358) * r + 5.4637849111641143699) *
                                  r + 6.6579046435011037772)
                                 / (((((((r *
                                          2.04426310338993978564e-15 + 1.4215117583164458887e-7) *
                                         r + 1.8463183175100546818e-5) * r +
                                        7.868691311456132591e-4) * r + .0148753612908506148525)
                                      * r + .13692988092273580531) * r +
                                     .59983220655588793769) * r + 1);
            }

            if (value_working_compliment < 0.0)
            {
                adjusted_value = -adjusted_value;
            }
        }

        return mean + standard_deviation * adjusted_value;
    }
}

namespace nml
{
    using namespace cumulative_distribution_functions_internal;

    double Distribution::f_cdf(float f_statistic, float df_model, float df_error, bool lower_tail, bool log_scale) noexcept
    {
        if (df_model <= 0 || df_error <= 0)
        {

        }

        if (f_statistic <= 0) return numeric_zero(log_scale);
        if (f_statistic >= positive_infinity) return numeric_one(log_scale);

//        if (df2 == ML_POSINF) {
//            if (df1 == ML_POSINF) {
//                if(x <  1.) return R_DT_0;
//                if(x == 1.) return (log_p ? -M_LN2 : 0.5);
//                if(x >  1.) return R_DT_1;
//            }
//
//            return pchisq(x * df1, df1, lower_tail, log_p);
//        }
//
//        if (df1 == ML_POSINF)/* was "fudge"	'df1 > 4e5' in 2.0.x */
//            return pchisq(df2 / x , df2, !lower_tail, log_p);


        return df_model * f_statistic > df_error ?
               beta_cdf(df_error / (df_error + df_model * f_statistic), df_error / 2, df_model / 2, !lower_tail,
                        log_scale) :
               beta_cdf(df_model * f_statistic / (df_error + df_model * f_statistic), df_model / 2, df_error / 2,
                        lower_tail, log_scale);
    }

    double Distribution::beta_cdf(double upper_limit, double alpha, double beta, bool lower_tail, bool log_scale) noexcept
    {
        if (alpha < 0 || beta < 0)
        {

        }

        if (upper_limit <= 0) return numeric_zero(log_scale);
        else if (upper_limit >= 1) return numeric_one(log_scale);

        auto solver = IncompleteBetaFunctionSolver(upper_limit, alpha, beta, log_scale);

        return lower_tail ? solver.p_lower_tail : solver.p_upper_tail;
    }

    double Distribution::t_pdf(float t_statistic, float df, bool log_scale) noexcept
    {
        if (df <= 0)
        {

        }

        if (t_statistic <= negative_infinity || t_statistic >= positive_infinity)
        {
            return numeric_zero(log_scale);
        }

        auto df_double = static_cast<double>(df);
        auto t_double = static_cast<double>(t_statistic);

        double t_squared = t_double * t_double;
        double t_squared_over_df = t_squared / df_double;

        double t_adjustment, abs_t = 0, ln_t_squared_over_df;

        double t = -log_likelihood_ratio_deviance(df_double / 2, (df_double + 1) / 2)
                + log_stirling_error((df_double + 1) / 2) - log_stirling_error(df_double / 2);

        bool large = t_squared_over_df > 1 / DBL_EPSILON;

        if (large)
        {
            abs_t = fabs(t_double);
            ln_t_squared_over_df = log(abs_t) - log(df_double) / 2;
            t_adjustment = df_double * ln_t_squared_over_df;
        }
        else if (t_squared_over_df > 0.2)
        {
            ln_t_squared_over_df = log(1 + t_squared_over_df) / 2;
            t_adjustment = df_double * ln_t_squared_over_df;
        }
        else
        {
            ln_t_squared_over_df = log1p(t_squared_over_df) / 2;
            t_adjustment = -log_likelihood_ratio_deviance(df_double / 2, (df_double + t_squared) / 2) + t_squared / 2;
        }

        if (log_scale)
        {
            return t - t_adjustment - (ln_sqrt_two_pi + ln_t_squared_over_df);
        }
        else
        {
            double inverse_sqrt_t_factor = large ? sqrt(df_double) / abs_t : exp(-ln_t_squared_over_df);

            return exp(t - t_adjustment) * inverse_sqrt_two_pi * inverse_sqrt_t_factor;
        }
    }

    double Distribution::t_cdf(float t_statistic, float df, bool lower_tail, bool log_scale) noexcept
    {
        if (df <= 0)
        {

        }

        if (t_statistic <= negative_infinity) return numeric_zero(log_scale);
        else if (t_statistic >= positive_infinity) return numeric_one(log_scale);

//        if(!R_FINITE(n))
//            return pnorm(x, 0.0, 1.0, lower_tail, log_p);
//
        double nx = 1 + (t_statistic / df) * t_statistic;

        double val = (df > t_statistic * t_statistic)
                 ? beta_cdf(t_statistic * t_statistic / (df + t_statistic * t_statistic), 0.5, df / 2, false, log_scale)
                 : beta_cdf(1 / nx, df / 2, 0.5, true, log_scale);

        if (t_statistic <= 0)
        {
            lower_tail = !lower_tail;
        }

        if (!log_scale)
        {
            return lower_tail ? (0.5 - (val / 2) + 0.5) : val / 2;
        }
        else
        {
            return lower_tail ? log1p(-0.5 * exp(val)) : val - M_LN2;
        }
    }

    double Distribution::normal_cdf(double value, double mean, double standard_deviation, bool lower_tail, bool log_scale) noexcept
    {
        if (!is_finite(value) && mean == value)
        {
            return NAN;
        }

        if (standard_deviation <= 0)
        {
            if (standard_deviation < 0)
            {
                return NAN;
            }

            return (value < mean) ? numeric_zero(log_scale) : numeric_one(log_scale);
        }

        double adjusted_value = (value - mean) / standard_deviation;

        if (!is_finite(adjusted_value))
        {
            return (value < mean) ? numeric_zero(log_scale) : numeric_one(log_scale);
        }

        auto solver = NormalDistributionSolver(adjusted_value, log_scale);

        return lower_tail ? solver.p_lower_tail : solver.p_upper_tail;
    }

    double Distribution::normal_pdf(double value, double mean, double standard_deviation, bool lower_tail, bool log_scale) noexcept
    {
        return normal_distribution_pdf(value, mean, standard_deviation, lower_tail, log_scale);
    }
}

#endif //NML_DISTRIBUTION_FUNCTIONS_H