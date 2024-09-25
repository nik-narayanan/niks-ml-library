//
// Created by nik on 6/16/2024.
//

#ifndef NML_HASH_H
#define NML_HASH_H

#include <functional>
#include <string_view>

#include "span.h"

namespace nml::hash_internal { template<typename T> struct HashTable; }

namespace nml
{
    template<typename T>
    struct HashSet
    {
        HashSet() noexcept;

        struct Iterator;
        Iterator end() const noexcept;
        Iterator begin() const noexcept;
        bool insert(const T& value) noexcept;
        bool remove(const T& value) noexcept;
        [[nodiscard]] uint64_t count() const noexcept;
        [[nodiscard]] bool contains(const T& value) const noexcept;

    private:
        hash_internal::HashTable<T> _table;
    };

    template<typename TKey, typename TValue>
    struct HashMap
    {
        HashMap() noexcept;

        struct Iterator;
        Iterator end() const noexcept;
        Iterator begin() const noexcept;
        bool remove(const TKey& key) noexcept;
        [[nodiscard]] uint64_t count() const noexcept;
        [[nodiscard]] TValue* get_value(const TKey& key) noexcept;
        [[nodiscard]] TValue* operator[](const TKey& key) noexcept;
        TValue* insert(const TKey& key, const TValue& value) noexcept;
        [[nodiscard]] bool contains_key(const TKey& key) const noexcept;
        [[nodiscard]] TValue& get_value_unsafe(const TKey& key) noexcept;
        [[nodiscard]] bool try_get_value(const TKey& key, TValue*& value_out) const noexcept;

    private:
        hash_internal::HashTable<std::pair<TKey, TValue>> _table;
    };
}

namespace nml::hash_internal
{
    template<typename T>
    static inline uint64_t hash_value(const T& value) noexcept
    {
        return std::hash<T>{}(value);
    }

    static inline uint64_t hash_value(const char* string) noexcept
    {
        uint64_t hash = 0xCBF29CE484222325;

        while (*string)
        {
            hash ^= static_cast<uint64_t>((unsigned char)*string++);
            hash *= 0x100000001B3;
        }

        return hash;
    }

    static inline bool compare_values(const char* left, const char* right) noexcept
    {
        return strcmp(left, right) == 0;
    }

    template<typename T>
    static inline bool compare_values(const T& left, const T& right) noexcept
    {
        return left == right;
    }

    static constexpr int8_t AVAILABLE = -1;

    template<typename T>
    struct Entry
    {
        int8_t offset{AVAILABLE};
        T data;
    };

    struct TableLocation
    {
        int8_t offset;
        uint64_t start;

        static TableLocation starting(uint64_t start) noexcept
        {
            return { .offset = 0, .start = start };
        }

        static TableLocation not_found() noexcept
        {
            return { .offset = -1, .start = 0 };
        }
    };

    template<typename T>
    struct is_pair : std::false_type {};

    template<typename TKey, typename TValue>
    struct is_pair<std::pair<TKey, TValue>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_pair_v = is_pair<T>::value;

    template<typename T>
    struct HashTable
    {
        HashTable() noexcept;
        ~HashTable() noexcept;

        HashTable(const HashTable&) = delete;
        HashTable& operator=(const HashTable&) = delete;

        HashTable(HashTable&& move) noexcept;
        HashTable& operator=(HashTable&& move) noexcept;

        uint64_t key_ct;
        uint64_t buckets_mask;
        Span<Entry<T>> entries;
        const float load_factor = 0.5;
        const uint8_t max_displacement = 4;

        bool resize() noexcept;
        T* find(const T& value) noexcept;
        bool insert(const T& value) noexcept;
        bool remove(const T& value) noexcept;
        T* insert_return(const T& value) noexcept;
        bool insert_unique(const T& value) noexcept;
        bool contains(const T& value) const noexcept;
        [[nodiscard]] uint64_t bucket_ct() const noexcept;
        static uint64_t hash_key(const T& value) noexcept;
        TableLocation find_location(const T& value) noexcept;
        T* insert_lookup(T value, TableLocation location) noexcept;
        static bool compare(const T& value, const T& other) noexcept;
    };

    template<typename T>
    HashTable<T>::HashTable(HashTable&& move) noexcept
        : key_ct(move.key_ct), entries(move.entries), buckets_mask(move.buckets_mask)
    {
        move.key_ct = 0;
        move.buckets_mask = 0;
        move.entries = Span<Entry<T>>(nullptr, 0);
    }

    template<typename T>
    HashTable<T> &HashTable<T>::operator=(HashTable&& move) noexcept
    {
        if (this != &move)
        {
            this->~HashTable();
            new(this) HashTable(std::move(move));
        }

        return *this;
    }

    template<typename T>
    HashTable<T>::HashTable() noexcept
        : key_ct(0), entries(), buckets_mask(8 - 1)
    {
        uint64_t total_entries = bucket_ct() + max_displacement + 1;

        auto new_memory = MemorySpan(entries.required_bytes(total_entries));

        new_memory.fill(AVAILABLE);

        entries = Span<Entry<T>>(new_memory, total_entries);
    }

    template<typename T>
    HashTable<T>::~HashTable() noexcept
    {
        std::free(entries.get_pointer());
    }

    template<typename T>
    bool HashTable<T>::resize() noexcept
    {
        auto old_key_ct = key_ct;
        auto old_entries = entries;
        auto old_buckets_ct = bucket_ct();
        auto old_buckets_mask = buckets_mask;

        uint64_t new_bucket_ct = old_buckets_ct * 2;

        while (true)
        {
            uint64_t total_entries = new_bucket_ct + max_displacement + 1;

            auto new_memory = MemorySpan(entries.required_bytes(total_entries));

            new_memory.fill(AVAILABLE);
            key_ct = 0, buckets_mask = new_bucket_ct - 1;
            entries = Span<Entry<T>>(new_memory, total_entries);

            for (uint64_t bucket = 0; bucket < old_entries.length; ++bucket)
            {
                if (old_entries[bucket].offset == AVAILABLE) continue;

                if (!insert_unique(old_entries[bucket].data)) break;
            }

            if (key_ct < old_key_ct)
            {
                std::free(new_memory.get_pointer());

                new_bucket_ct *= 2; continue;
            }

            if (old_buckets_mask)
            {
                std::free(old_entries.get_pointer());
            }

            return true;
        }
    }

    template<typename T>
    bool HashTable<T>::insert(const T& value) noexcept
    {
        uint64_t hash = hash_key(value);
        TableLocation location = TableLocation::starting(hash & buckets_mask);
        Entry<T>* entry = entries.get_pointer(location.start);

        while (entry->offset >= location.offset)
        {
            if (compare(entry->data, value)) return false;

            ++entry, ++location.offset;
        }

        insert_lookup(value, location); return true;
    }

    template<typename T>
    T* HashTable<T>::insert_return(const T& value) noexcept
    {
        uint64_t hash = hash_key(value);
        TableLocation location = TableLocation::starting(hash & buckets_mask);
        Entry<T>* entry = entries.get_pointer(location.start);

        while (entry->offset >= location.offset)
        {
            if (compare(entry->data, value)) return &entry->data;

            ++entry, ++location.offset;
        }

        return insert_lookup(value, location);
    }

    template<typename T>
    T* HashTable<T>::find(const T& value) noexcept
    {
        uint64_t hash = hash_key(value);
        TableLocation location = TableLocation::starting(hash & buckets_mask);
        Entry<T>* entry = entries.get_pointer(location.start);

        while (entry->offset >= location.offset)
        {
            if (compare(entry->data, value)) return &entry->data;

            ++entry, ++location.offset;
        }

        return nullptr;
    }

    template<typename T>
    TableLocation HashTable<T>::find_location(const T& value) noexcept
    {
        uint64_t hash = hash_key(value);
        TableLocation location = TableLocation::starting(hash & buckets_mask);
        Entry<T>* entry = entries.get_pointer(location.start);

        while (entry->offset >= location.offset)
        {
            if (compare(entry->data, value)) return location;

            ++entry, ++location.offset;
        }

        return TableLocation::not_found();
    }

    template<typename T>
    bool HashTable<T>::remove(const T& value) noexcept
    {
        TableLocation location = find_location(value);

        if (location.offset == -1) return false;

        Entry<T>* entry = entries.get_pointer(location.start + location.offset);

        entry->offset = AVAILABLE;

        while ((++entry)->offset > 0)
        {
            (entry - 1)->data = entry->data;
            (entry - 1)->offset = entry->offset - 1;

            entry->offset = AVAILABLE;
        }

        key_ct -= 1; return true;
    }

    template<typename T>
    bool HashTable<T>::contains(const T& value) const noexcept
    {
        uint64_t hash = hash_key(value);
        TableLocation location = TableLocation::starting(hash & buckets_mask);
        Entry<T>* entry = entries.get_pointer(location.start);

        while (entry->offset >= location.offset)
        {
            if (compare(entry->data, value)) return &entry->data;

            ++entry, ++location.offset;
        }

        return false;
    }

    template<typename T>
    T* HashTable<T>::insert_lookup(T value, TableLocation location) noexcept
    {
        if (key_ct > buckets_mask * load_factor) { resize(); return insert_return(value); }

        auto* entry = entries.get_pointer(location.start + location.offset);

        do
        {
            if (entry->offset == AVAILABLE)
            {
                entry->data = value;
                entry->offset = location.offset;

                key_ct += 1; return &entry->data;
            }

            if (entry->offset < location.offset)
            {
                std::swap(value, entry->data);
                std::swap(location.offset, entry->offset);
            }

            ++entry, ++location.offset;
        }
        while (location.offset < max_displacement);

        resize(); return insert_return(value);
    }

    template<typename T>
    bool HashTable<T>::insert_unique(const T& value) noexcept
    {
        uint64_t hash = hash_key(value);
        TableLocation location = TableLocation::starting(hash & buckets_mask);

        return insert_lookup(value, location);
    }

    template<typename T>
    bool HashTable<T>::compare(const T& value, const T& other) noexcept
    {
        if constexpr (is_pair_v<T>)
        {
            return compare_values(value.first, other.first);
        }
        else
        {
            return compare_values(value, other);
        }
    }

    template<typename T>
    uint64_t HashTable<T>::hash_key(const T& value) noexcept
    {
        if constexpr (is_pair_v<T>)
        {
            return hash_value(value.first);
        }
        else
        {
            return hash_value(value);
        }
    }

    template<typename T>
    uint64_t HashTable<T>::bucket_ct() const noexcept
    {
        return buckets_mask + (bool)buckets_mask;
    }
}

namespace nml
{
    template<typename T>
    HashSet<T>::HashSet() noexcept
        : _table(hash_internal::HashTable<T>())
    { }

    template<typename T>
    bool HashSet<T>::insert(const T& value) noexcept
    {
        return _table.insert(value);
    }

    template<typename T>
    bool HashSet<T>::remove(const T& value) noexcept
    {
        return _table.remove(value);
    }

    template<typename T>
    bool HashSet<T>::contains(const T& value) const noexcept
    {
        return _table.contains(value);
    }

    template<typename T>
    uint64_t HashSet<T>::count() const noexcept
    {
        return _table.key_ct;
    }

    template<typename T>
    class HashSet<T>::Iterator
    {
        friend HashSet<T>;
        const hash_internal::HashTable<T>& _table;

        uint64_t _position;

        explicit Iterator(const hash_internal::HashTable<T>& table, uint64_t position = 0) noexcept
            : _table(table), _position(position)
        {
            if (_position < _table.entries.length && _table.entries[_position].offset == hash_internal::AVAILABLE)
            {
                ++(*this);
            }
        }

    public:

        Iterator begin() const { return Iterator(_table, 0); }
        Iterator end() const { return Iterator(_table, _table.entries.length); }

        Iterator& operator++()
        {
            while (++_position < _table.entries.length)
            {
                if (_table.entries[_position].offset != hash_internal::AVAILABLE) return *this;
            };

            return *this;
        }

        const T& operator*() const
        {
            return _table.entries[_position].data;
        }

        bool operator==(const Iterator& rhs) const
        {
            return _position == rhs._position;
        }

        bool operator!=(const Iterator& rhs) const
        {
            return _position != rhs._position;
        }
    };

    template<typename T>
    typename HashSet<T>::Iterator HashSet<T>::end() const noexcept
    {
        return Iterator(_table, _table.entries.length);
    }

    template<typename T>
    typename HashSet<T>::Iterator HashSet<T>::begin() const noexcept
    {
        return Iterator(_table);
    }
}


/*******************************************************************************************************************
 HASHMAP
 ******************************************************************************************************************/
namespace nml
{
    template<typename TKey, typename TValue>
    HashMap<TKey, TValue>::HashMap() noexcept
        : _table(hash_internal::HashTable<std::pair<TKey, TValue>>())
    { }

    template<typename TKey, typename TValue>
    uint64_t HashMap<TKey, TValue>::count() const noexcept
    {
        return _table.key_ct;
    }

    template<typename TKey, typename TValue>
    bool HashMap<TKey, TValue>::remove(const TKey& key) noexcept
    {
        std::pair<TKey, TValue> pair;

        pair.first = key;

        return _table.remove(pair);
    }

    template<typename TKey, typename TValue>
    TValue* HashMap<TKey, TValue>::get_value(const TKey& key) noexcept
    {
        std::pair<TKey, TValue> pair;

        pair.first = key;

        return &_table.find(pair)->second;
    }

    template<typename TKey, typename TValue>
    TValue& HashMap<TKey, TValue>::get_value_unsafe(const TKey& key) noexcept
    {
        std::pair<TKey, TValue> pair;

        pair.first = key;

        return _table.find(pair)->second;
    }

    template<typename TKey, typename TValue>
    TValue* HashMap<TKey, TValue>::operator[](const TKey& key) noexcept
    {
        return get_value(key);
    }

    template<typename TKey, typename TValue>
    TValue* HashMap<TKey, TValue>::insert(const TKey& key, const TValue& value) noexcept
    {
        auto pair = std::pair<TKey, TValue>(key, value);

        return &(_table.insert_return(pair)->second);
    }

    template<typename TKey, typename TValue>
    bool HashMap<TKey, TValue>::contains_key(const TKey& key) const noexcept
    {
        std::pair<TKey, TValue> pair;

        pair.first = key;

        return _table.contains(pair);
    }

    template<typename TKey, typename TValue>
    bool HashMap<TKey, TValue>::try_get_value(const TKey& key, TValue*& value_out) const noexcept
    {
        std::pair<TKey, TValue> pair;

        pair.first = key;

        std::pair<TKey, TValue>* out = _table.find();

        if (out == nullptr)
        {
            value_out = nullptr;
            return false;
        }
        else
        {
            value_out = &(out->second);
            return true;
        }
    }

    template<typename TKey, typename TValue>
    class HashMap<TKey, TValue>::Iterator
    {
        friend HashMap<TKey, TValue>;
        const hash_internal::HashTable<std::pair<TKey, TValue>>& _table;

        uint64_t _position;

        explicit Iterator(const hash_internal::HashTable<std::pair<TKey, TValue>>& table, uint64_t position = 0) noexcept
            : _table(table), _position(position)
        {
            if (_position < _table.entries.length && _table.entries[_position].offset == hash_internal::AVAILABLE)
            {
                ++(*this);
            }
        }

    public:

        [[nodiscard]] Iterator begin() const { return Iterator(_table, 0); }
        [[nodiscard]] Iterator end() const { return Iterator(_table, _table.entries.length); }

        Iterator& operator++()
        {
            while (++_position < _table.entries.length)
            {
                if (_table.entries[_position].offset != hash_internal::AVAILABLE) return *this;
            };

            return *this;
        }

        const std::pair<TKey, TValue>& operator*() const
        {
            return _table.entries[_position].data;
        }

        bool operator==(const Iterator& rhs) const
        {
            return _position == rhs._position;
        }

        bool operator!=(const Iterator& rhs) const
        {
            return _position != rhs._position;
        }
    };

    template<typename TKey, typename TValue>
    typename HashMap<TKey, TValue>::Iterator HashMap<TKey, TValue>::begin() const noexcept
    {
        return Iterator(_table);
    }

    template<typename TKey, typename TValue>
    typename HashMap<TKey, TValue>::Iterator HashMap<TKey, TValue>::end() const noexcept
    {
        return Iterator(_table, _table.entries.length);
    }
}

#endif //NML_HASH_H