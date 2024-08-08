//
// Created by nik on 6/9/2024.
//

#ifndef NML_REGULAR_EXPRESSIONS_H
#define NML_REGULAR_EXPRESSIONS_H

#include "result.h"

namespace nml::regular_expressions_internal
{
    constexpr unsigned MAX_BRANCHES = 10;
    constexpr unsigned MAX_CAPTURE_GROUPS = 10;
}

namespace nml
{
    using namespace nml::regular_expressions_internal;

    struct Regex
    {
        struct Match;
        struct CaptureGroup;
        struct MatchIterator;

        enum class Option : char
        {
            NONE        = 0,
            MULTI_LINE  = 1 << 0,
            SINGLE_LINE = 1 << 1,
            INSENSITIVE = 1 << 2,
        };

        enum class ErrorCode : char
        {
            NO_MATCH = 1,
            UNKNOWN,
            UNMATCHED_BRACKETS,
            UNMATCHED_PARENTHESIS,
            INVALID_METACHARACTER,
            EXCEEDED_MAX_BRANCHES,
            EXCEEDED_MAX_CAPTURE_GROUPS,
        };

        explicit Regex(const char* regex, Option options = Option::NONE);
        static Result<Regex, ErrorCode> compile(const char* regex, Option options = Option::NONE) noexcept;

        Match match(const char* string, unsigned length = 0) const noexcept;
        bool is_match(const char* string, unsigned length = 0) const noexcept;
        MatchIterator matches(const char* string, unsigned length = 0) const noexcept;
        int find_matches(const char* string, unsigned length, Match* match) const noexcept;

    private:

        struct MetaData
        {
            struct Branch
            {
                unsigned regex_offset;
                unsigned group_offset;
            };

            struct CaptureGroup
            {
                unsigned length;
                unsigned branch_ct;
                unsigned regex_offset;
                unsigned branch_offset;
            };

            unsigned group_ct;
            CaptureGroup groups[MAX_CAPTURE_GROUPS];

            unsigned branch_ct;
            Branch branches[MAX_BRANCHES];

            bool multi_line;
            bool insensitive;
            bool single_line;
        };

        const char* _regex;
        MetaData _meta_data;
        unsigned _regex_length;

        explicit Regex(const char* regex, unsigned regex_length, MetaData meta_data) noexcept
            : _regex(regex), _meta_data(meta_data), _regex_length(regex_length)
        { }

//        [[nodiscard]] int find_matches(const char* string, unsigned length, Match* match) noexcept;
        [[nodiscard]] int match_character(unsigned regex_offset, char character) const noexcept;
        [[nodiscard]] int match_alternation(unsigned regex_offset, unsigned regex_length, const char* string) const  noexcept;
        [[nodiscard]] int match_capture_group(const char* string, unsigned length, unsigned cg_offset, Match* match) const  noexcept;
        [[nodiscard]] int match_pattern(unsigned regex_offset, unsigned regex_length, const char* string, unsigned length, unsigned cg_offset, Match* match) const  noexcept;
    };
}

namespace nml::regular_expressions_internal
{
    static inline bool is_metacharacter(char ch) noexcept
    {
        constexpr char metacharacters[] = "^$().[]*+?|\\Ssdbfnrtv";

        auto start = std::begin(metacharacters);
        auto end = std::end(metacharacters);

        while (start < end && ch != *start) { ++start; }

        return start < end;
    }

    static inline unsigned regex_token_length(const char* token) noexcept
    {
        return token[0] == '\\' && token[1] == 'x' ? 4 : token[0] == '\\' ? 2 : 1;
    }

    static inline unsigned alternation_length(const char* alternation, unsigned regex_length) noexcept
    {
        unsigned offset = 1;

        while (offset < regex_length && alternation[offset] != ']')
        {
            offset += regex_token_length(alternation + offset);
        }

        return offset <= regex_length ? offset + 1 : 0;
    }

    static bool is_quantifier(const char ch) noexcept
    {
        return ch == '*' || ch == '+' || ch == '?';
    }

    static inline unsigned operation_length(const char* regex, unsigned regex_length) noexcept
    {
        return regex[0] == '[' ?
            alternation_length(regex, regex_length):
            regex_token_length(regex);
    }

    static int hex_char_to_int(int hex_char) noexcept
    {
        return isdigit(hex_char) ? hex_char - '0' : hex_char - 'W';
    }

    static inline int hex_to_char(const char* hex_string) noexcept
    {
        return (hex_char_to_int(tolower(hex_string[0])) << 4)
               | hex_char_to_int(tolower(hex_string[1]));
    }

    static inline int err(Regex::ErrorCode error) noexcept
    {
        return -static_cast<int>(error);
    }
}

namespace nml
{
    using namespace nml::regular_expressions_internal;

    typedef Result<Regex, Regex::ErrorCode> RegexResult;

    struct Regex::CaptureGroup
    {
        unsigned length;
        const char* capture;
    };

    struct Regex::Match
    {
        unsigned length;
        const char* match;

        unsigned group_ct;
        CaptureGroup groups[MAX_CAPTURE_GROUPS];

        explicit Match() noexcept
            : length(0), match(nullptr), group_ct(0), groups()
        { }

        [[nodiscard]] bool is_match() const noexcept
        {
            return match != nullptr;
        }

        [[nodiscard]] bool has_groups() const noexcept
        {
            return group_ct > 0;
        }

        void copy_match(char* buffer) const noexcept
        {
            unsigned offset = 0;

            while (offset < length)
            {
                buffer[offset] = match[offset++];
            }

            buffer[offset] = 0;
        }

        void copy_group(unsigned group, char* buffer) noexcept
        {
            unsigned offset = 0;

            while (offset < groups[group].length)
            {
                buffer[offset] = groups[group].capture[offset++];
            }

            buffer[offset] = 0;
        }

        void print() const
        {
            if (!is_match())
            {
                std::cout << "No Matches\n";
                return;
            }

            std::cout << "Full Match:\n    ";

            for (unsigned i = 0; i < length; ++i)
            {
                std::cout << match[i];
            }

            std::cout << '\n';

            if (!has_groups())
            {
                std::cout << "No Groups\n";
                return;
            }

            std::cout << "Groups:\n";

            for (unsigned i = 0; i < group_ct; ++i)
            {
                std::cout << "    " << i << ": ";

                for (unsigned j = 0; j < groups[i].length; ++j)
                {
                    std::cout << groups[i].capture[j];
                }

                std::cout << "\n";
            }

        }

    };

    class Regex::MatchIterator
    {
        friend Regex;

        const Regex& _regex;

        Match _next;
        unsigned _index;
        const char* _string;
        unsigned _match_count;
        unsigned _string_length;

        explicit MatchIterator(const Regex& regex, const char* string, unsigned string_length = 0) noexcept
            : _index(0)
            , _regex(regex)
            , _next(Match())
            , _string(string)
            , _match_count(0)
            , _string_length(string_length == 0 ? strlen(string) : string_length)
        { }

    public:

        [[nodiscard]] bool is_end() const noexcept
        {
            return _index >= _string_length;
        }

        [[nodiscard]] unsigned match_count() const noexcept
        {
            return _match_count;
        }

        [[nodiscard]] const Match& next() const noexcept
        {
            return _next;
        }

        bool has_next() noexcept
        {
            if (is_end()) return false;

            auto next = _regex.match(_string + _index, _string_length - _index);

            if (!next.is_match())
            {
                _index = _string_length;

                return false;
            }

            _next = next;

            _match_count += 1;

            _index = 1 + _next.length + _next.match - _string;

            return true;
        }

        void reset() noexcept
        {
            _index = 0;
            _match_count = 0;
            _next = Match();
        }
    };

    inline Regex::Option operator|(Regex::Option lhs, Regex::Option rhs)
    {
        return static_cast<Regex::Option>(
                static_cast<std::underlying_type_t<Regex::Option>>(lhs) |
                static_cast<std::underlying_type_t<Regex::Option>>(rhs)
        );
    }

    inline Regex::Option& operator|=(Regex::Option& lhs, Regex::Option rhs)
    {
        lhs = lhs | rhs;
        return lhs;
    }

    inline Regex::Option operator&(Regex::Option lhs, Regex::Option rhs)
    {
        return static_cast<Regex::Option>(
            static_cast<std::underlying_type_t<Regex::Option>>(lhs) &
            static_cast<std::underlying_type_t<Regex::Option>>(rhs)
        );
    }

    inline Regex::Option& operator&=(Regex::Option& lhs, Regex::Option rhs)
    {
        lhs = lhs & rhs;
        return lhs;
    }

    Regex::Regex(const char* regex, Option options) : _regex(regex), _regex_length(0), _meta_data({})
    {
        auto compiled_result = Regex::compile(regex, options);

        if (compiled_result.is_err())
        {
            // panic
        }

        Regex compiled = compiled_result.ok();

        _meta_data = compiled._meta_data;
        _regex_length = compiled._regex_length;
    }

    Result<Regex, Regex::ErrorCode> Regex::compile(const char *regex, Option options) noexcept
    {
        unsigned stride, depth = 0;
        const unsigned regex_length = strlen(regex);

        MetaData meta_data{};
        meta_data.group_ct = 1;
        meta_data.branch_ct = 0;
        meta_data.groups[0] =
        {
            .length = regex_length,
            .branch_ct = 0,
            .regex_offset = 0,
            .branch_offset = 0,
        };

        for (unsigned i = 0; i < regex_length; i += stride)
        {
            stride = 1;

            if (regex[i] == '[')
            {
                while (i + stride < regex_length && regex[i + stride] != ']')
                {
                    stride += regex_token_length(regex + i + stride);
                }

                if (i + stride > regex_length)
                {
                    return RegexResult::err(ErrorCode::UNMATCHED_BRACKETS);
                }

                stride += 1;
            }
            else if (regex[i] == '|')
            {
                if (meta_data.branch_ct >= MAX_BRANCHES)
                {
                    return RegexResult::err(ErrorCode::EXCEEDED_MAX_BRANCHES);
                }

                meta_data.branches[meta_data.branch_ct].group_offset =
                        meta_data.groups[meta_data.group_ct - 1].length == 0 ?
                        meta_data.group_ct - 1 : depth;

                meta_data.branches[meta_data.branch_ct++].regex_offset = i;
            }
            else if (regex[i] == '\\')
            {
                if (i + 1 >= regex_length)
                {
                    return RegexResult::err(ErrorCode::INVALID_METACHARACTER);
                }

                if (regex[i + 1] == 'x')
                {
                    if (i + 3 >= regex_length || !isxdigit(regex[i + 2]) || !isxdigit(regex[i + 3]))
                    {
                        return RegexResult::err(ErrorCode::INVALID_METACHARACTER);
                    }

                    stride = 4;
                }
                else if (is_metacharacter(regex[i + 1]))
                {
                    stride = 2;
                }
                else
                {
                    return RegexResult::err(ErrorCode::INVALID_METACHARACTER);
                }
            }
            else if (regex[i] == '(')
            {
                if (meta_data.group_ct >= MAX_CAPTURE_GROUPS)
                {
                    return RegexResult::err(ErrorCode::EXCEEDED_MAX_CAPTURE_GROUPS);
                }

                meta_data.groups[meta_data.group_ct].regex_offset = i + 1;
                meta_data.groups[meta_data.group_ct++].length = 0;

                depth++;
            }
            else if (regex[i] == ')')
            {
                if (depth == 0)
                {
                    return RegexResult::err(ErrorCode::UNMATCHED_PARENTHESIS);
                }

                auto& group = meta_data.groups[meta_data.group_ct - 1];

                unsigned ind = group.length == 0 ? meta_data.group_ct - 1 : depth;

                meta_data.groups[ind].length = i - meta_data.groups[ind].regex_offset;

                depth--;
            }
        }

        if (depth > 0)
        {
            return RegexResult::err(ErrorCode::UNMATCHED_PARENTHESIS);
        }

        for (unsigned left = 0; left < meta_data.branch_ct; ++left)
        {
            for (unsigned right = left + 1; right < meta_data.branch_ct; ++right)
            {
                if (meta_data.branches[left].group_offset > meta_data.branches[right].group_offset)
                {
                    std::swap(meta_data.branches[left], meta_data.branches[right]);
                }
            }
        }

        for (unsigned cg = 0, br = 0; cg < meta_data.group_ct; ++cg)
        {
            meta_data.groups[cg].branch_ct = 0;
            meta_data.groups[cg].branch_offset = br;

            while (br < meta_data.branch_ct && meta_data.branches[br].group_offset == cg)
            {
                meta_data.groups[cg].branch_ct++; br++;
            }
        }

        meta_data.multi_line = (options & Option::MULTI_LINE) != Option::NONE;
        meta_data.single_line = (options & Option::SINGLE_LINE) != Option::NONE;
        meta_data.insensitive = (options & Option::INSENSITIVE) != Option::NONE;

        return RegexResult::ok(Regex(regex, regex_length, meta_data));
    }

    Regex::Match Regex::match(const char* string, unsigned length) const noexcept
    {
        Match match;

        find_matches(string, length, &match);

        return match;
    }

    bool Regex::is_match(const char* string, unsigned length) const noexcept
    {
        return find_matches(string, length, nullptr) >= 0;
    }

    Regex::MatchIterator Regex::matches(const char* string, unsigned length) const noexcept
    {
        return Regex::MatchIterator(*this, string, length);
    }

    int Regex::find_matches(const char* string, unsigned length, Match* match) const noexcept
    {
        int match_offset = -1;

        if (match != nullptr)
        {
            match->length = 0;
            match->group_ct = 0;
            match->match = nullptr;
        }

        length = length == 0 ? strlen(string) : length;

        bool is_anchored = _regex[_meta_data.branches[0].regex_offset] == '^';

        for (unsigned i = 0; i <= length; ++i)
        {
            match_offset = match_capture_group(string + i, length - i, 0, match);

            if (match_offset >= 0)
            {
                if (match != nullptr)
                {
                    match->match = string + i;
                    match->length = match_offset;
                }

                return match_offset + static_cast<int>(i);
            }

            if (is_anchored) break;
        }

        return match_offset;
    }

    int Regex::match_capture_group(const char* string, unsigned length, unsigned cg_offset, Match* match) const noexcept
    {
        int result, iteration = 0;

        const MetaData::CaptureGroup& cg = _meta_data.groups[cg_offset];

        do
        {
            unsigned regex_offset = iteration == 0 ? cg.regex_offset :
                            _meta_data.branches[cg.branch_offset + iteration - 1].regex_offset + 1;

            unsigned regex_length = cg.branch_ct == 0 ? cg.length : iteration == cg.branch_ct ?
                                    cg.regex_offset + cg.length - regex_offset :
                                    _meta_data.branches[cg.branch_offset + iteration].regex_offset - regex_offset;

            result = match_pattern(regex_offset, regex_length, string, length, cg_offset, match);
        }
        while (result <= 0 && iteration++ < cg.branch_ct);

        return result;
    }

    int Regex::match_pattern(unsigned regex_offset, unsigned regex_length, const char* string,
                             unsigned length, unsigned cg_offset, Match* match) const noexcept
    {
        unsigned stride, string_offset = 0;

        const char* re = &_regex[regex_offset];

        for (unsigned ro = 0; ro < regex_length && string_offset <= length; ro += stride)
        {
            stride = re[ro] == '(' ?
                     _meta_data.groups[cg_offset + 1].length + 2 :
                    operation_length(&re[ro], regex_length);

            if (stride + ro < regex_length && is_quantifier(re[stride + ro]))
            {
                char next_ch = re[stride + ro];

                if (next_ch == '?')
                {
                    int result = match_pattern(
                        regex_offset + ro,
                        stride,
                        string + string_offset,
                        length - string_offset,
                        cg_offset,
                        match
                    );

                    ++ro; string_offset += std::max(result, 0);
                }
                else if (next_ch == '+' || next_ch == '*')
                {
                    bool non_greedy = false;

                    int match_offset_increment, after_match_increment = -1;
                    unsigned match_offset = string_offset, after_match_offset = string_offset;

                    unsigned after_regex_offset = stride + ro + 1;

                    if (after_regex_offset < regex_length && _regex[regex_offset + after_regex_offset] == '?')
                    {
                        non_greedy = true;
                        after_regex_offset += 1;
                    }

                    do
                    {
                        match_offset_increment = match_pattern(
                            regex_offset + ro,
                            stride,
                            string + match_offset,
                            length - match_offset,
                            cg_offset,
                            match
                        );

                        if (match_offset_increment > 0)
                        {
                            match_offset += match_offset_increment;
                        }
                        else if (next_ch == '+' && match_offset_increment < 0)
                        {
                            break;
                        }

                        if (after_regex_offset >= regex_length)
                        {
                            after_match_offset = match_offset;
                        }
                        else
                        {
                            after_match_increment = match_pattern(
                                regex_offset + after_regex_offset,
                                regex_length - after_regex_offset,
                                string + match_offset,
                                length - match_offset,
                                cg_offset,
                                match
                           );

                            if (after_match_increment >= 0)
                            {
                                after_match_offset = match_offset + after_match_increment;
                            }
                        }
                    }
                    while (match_offset_increment > 0 && !(after_match_offset > string_offset && non_greedy));

                    if (match_offset_increment < 0 && after_match_increment < 0 && next_ch == '*')
                    {
                        after_match_increment = match_pattern(
                            regex_offset + after_regex_offset,
                            regex_length - after_regex_offset,
                            string + string_offset,
                            length - string_offset,
                            cg_offset,
                            match
                        );

                        if (after_match_increment > 0)
                        {
                            after_match_offset = string_offset + after_match_increment;
                        }
                    }

                    if (next_ch == '+' && after_match_offset == string_offset)
                    {
                        return err(ErrorCode::NO_MATCH);
                    }

                    if (after_match_offset == string_offset && after_regex_offset < regex_length && after_match_increment < 0)
                    {
                        return err(ErrorCode::NO_MATCH);
                    }

                    return static_cast<int>(after_match_offset);
                }
            }
            else if (re[ro] == '[')
            {
                int match_offset = match_alternation(
                    regex_offset + ro + 1,
                    regex_length - ro - 2,
                    string + string_offset
                );

                if (match_offset <= 0) return match_offset;

                string_offset += match_offset;
            }
            else if (re[ro] == '(')
            {
                cg_offset += 1;

                int match_offset = err(ErrorCode::NO_MATCH);

                if (ro + stride >= regex_length)
                {
                    match_offset = match_capture_group(
                        string + string_offset,
                        length - string_offset,
                        cg_offset,
                        match
                    );
                }
                else
                {
                    for (unsigned i = 0; i <= length - string_offset; ++i)
                    {
                        match_offset = match_capture_group(
                            string + string_offset,
                            length - string_offset - i,
                            cg_offset,
                            match
                        );

                        if (match_offset >= 0)
                        {
                            unsigned offset = ro + stride;

                            auto should_break = match_pattern(
                                regex_offset + offset,
                                regex_length - offset,
                                string + string_offset + match_offset,
                                length - string_offset - match_offset,
                                cg_offset,
                                match
                            );

                            if (should_break >= 0) break;
                        }
                    }
                }

                if (match_offset < 0) return match_offset;

                if (match_offset > 0 && match != nullptr)
                {
                    match->group_ct = cg_offset;

                    match->groups[cg_offset - 1] = Regex::CaptureGroup
                    {
                        .length = static_cast<unsigned>(match_offset),
                        .capture = string + string_offset
                    };
                }

                string_offset += match_offset;
            }
            else if (re[ro] == '^')
            {
                if (_meta_data.multi_line)
                {
                    if (!(string_offset == 0 || string[string_offset - 1] == '\n')) return err(ErrorCode::NO_MATCH);
                }
                else
                {
                    if (string_offset != 0) return err(ErrorCode::NO_MATCH);
                }
            }
            else if (re[ro] == '$')
            {
                char character = string[string_offset];

                if (_meta_data.multi_line)
                {
                    if (!(character == '\n' || character == '\0')) return err(ErrorCode::NO_MATCH);
                }
                else
                {
                    if (character != '\0') return err(ErrorCode::NO_MATCH);
                }
            }
            else
            {
                if (string_offset >= length) return err(ErrorCode::NO_MATCH);

                int match_offset = match_character(regex_offset + ro, string[string_offset]);

                if (match_offset <= 0) return match_offset;

                string_offset += match_offset;
            }
        }

        return static_cast<int>(string_offset);
    }

    int Regex::match_alternation(unsigned regex_offset, unsigned regex_length, const char* string) const noexcept
    {
        int result = -1;
        unsigned offset = 0;
        bool invert = _regex[regex_offset] == '^';

        if (invert)
        {
            regex_offset += 1;
            regex_length -= 1;
        }

        const char* alternation = &_regex[regex_offset];

        while (offset <= regex_length && alternation[offset] != ']' && result <= 0)
        {
            bool is_range = alternation[offset] != '-' && alternation[offset + 1] == '-';

            if (is_range && alternation[offset + 2] != ']' && alternation[offset + 2] != '\0')
            {
                if (_meta_data.insensitive)
                {
                    int lower_ch = tolower(*string), lower_re = tolower(alternation[offset]);

                    result = lower_ch >= lower_re && lower_ch <= tolower(alternation[offset + 2]);
                }
                else
                {
                    result = *string >= alternation[offset] && *string <= alternation[offset + 2];
                }

                offset += 3;
            }
            else
            {
                result = match_character(regex_offset + offset, *string);
                
                offset += regex_token_length(alternation + offset);
            }
        }

        return (!invert && result > 0) || (invert && result <= 0) ? 1 : -1;
    }

    int Regex::match_character(unsigned regex_offset, char character) const noexcept
    {
        char regex_ch = _regex[regex_offset];

        if (regex_ch == '|') return err(ErrorCode::UNKNOWN);
        else if (regex_ch == '$') return err(ErrorCode::NO_MATCH);
        else if (regex_ch == '\\')
        {
            char escaped_ch = _regex[regex_offset + 1];

            switch (escaped_ch)
            {
                case 'S': return isspace(character)  ? err(ErrorCode::NO_MATCH) : 1;
                case 's': return !isspace(character) ? err(ErrorCode::NO_MATCH) : 1;
                case 'd': return !isdigit(character) ? err(ErrorCode::NO_MATCH) : 1;
                case 'b': return character != '\b'      ? err(ErrorCode::NO_MATCH) : 1;
                case 'f': return character != '\f'      ? err(ErrorCode::NO_MATCH) : 1;
                case 'n': return character != '\n'      ? err(ErrorCode::NO_MATCH) : 1;
                case 'r': return character != '\r'      ? err(ErrorCode::NO_MATCH) : 1;
                case 't': return character != '\t'      ? err(ErrorCode::NO_MATCH) : 1;
                case 'v': return character != '\v'      ? err(ErrorCode::NO_MATCH) : 1;
                case 'x': return character != hex_to_char(&_regex[regex_offset + 2])
                                                        ? err(ErrorCode::NO_MATCH) : 1;
                default: return character != escaped_ch ? err(ErrorCode::NO_MATCH) : 1;
            }
        }
        else if(regex_ch == '.')
        {
            if (_meta_data.single_line)
            {
                return 1;
            }
            else
            {
                return character != '\n' ? 1 : err(ErrorCode::NO_MATCH);
            }
        }
        else if (_meta_data.insensitive)
        {
            return tolower(regex_ch) != tolower(character) ? err(ErrorCode::NO_MATCH) : 1;
        }
        else
        {
            return regex_ch != character ? err(ErrorCode::NO_MATCH) : 1;
        }
    }
}

#endif //NML_REGULAR_EXPRESSIONS_H