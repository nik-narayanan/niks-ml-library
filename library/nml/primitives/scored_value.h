//
// Created by nik on 7/9/2024.
//

#ifndef NML_SCORED_VALUE_H
#define NML_SCORED_VALUE_H

namespace nml
{
    template<typename TValue, typename TScore = float>
    struct ScoredValue
    {
        TScore score; TValue value;

        bool operator<(const ScoredValue& rhs) const { return score < rhs.score; }
        bool operator>(const ScoredValue& rhs) const { return rhs.score < score; }
        bool operator==(const ScoredValue& rhs) const { return score == rhs.score; }
        bool operator<=(const ScoredValue& rhs) const { return rhs.score >= score; }
        bool operator>=(const ScoredValue& rhs) const { return score >= rhs.score; }

        template<typename T>
        friend std::ostream& operator<<(std::ostream& os, const ScoredValue<T>& rhs);
    };

    template<typename TValue>
    std::ostream& operator<<(std::ostream& os, const ScoredValue<TValue>& rhs)
    {
        os << "(" << rhs.value << ", " << rhs.score << ")";
        return os;
    }
}

#endif //NML_SCORED_VALUE_H
