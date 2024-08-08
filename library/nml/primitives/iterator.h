//
// Created by nik on 6/23/2024.
//

#ifndef NML_ITERATOR_H
#define NML_ITERATOR_H

namespace nml
{
    template<typename T, typename TContainer>
    struct Iterator
    {
        TContainer container;

        explicit Iterator(const TContainer& container)
            : container(container)
        { }

        // < 0 implies not cached
        [[nodiscard]] inline int64_t length() const noexcept
        {
            return container.length();
        }

        [[nodiscard]] inline bool is_end() const noexcept
        {
            return container.is_end();
        }

        [[nodiscard]] inline T next() const noexcept
        {
            return container.next();
        }

        [[nodiscard]] inline bool has_next() noexcept
        {
            return container.has_next();
        }

        inline void reset() noexcept
        {
            container.reset();
        }
    };
}

#endif //NML_ITERATOR_H