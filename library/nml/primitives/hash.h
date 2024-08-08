//
// Created by nik on 6/16/2024.
//

#ifndef NML_HASH_H
#define NML_HASH_H

#include "span.h"
#include "iterator.h"

namespace nml::hash_internal
{
    template<typename T>
    struct HashTable;
}

namespace nml
{
    template<typename T>
    class HashSet
    {
        hash_internal::HashTable<T> _table;

    public:

        HashSet() noexcept;

        inline bool insert(T value) noexcept;
        inline uint64_t count() const noexcept;
        inline bool contains(T& value) noexcept;
        inline bool remove(const T& value) noexcept;
        struct Iterator; Iterator to_iterator() const noexcept;
    };

    template<typename TKey, typename TValue>
    class HashMap
    {
        hash_internal::HashTable<std::pair<TKey, TValue>> _table;

    public:

        HashMap() noexcept;

        inline uint64_t count() noexcept;
        inline bool remove(const TKey& key) noexcept;
        inline TValue& insert(const TKey& key, const TValue& value) noexcept;
//        inline TValue& insert(TKey key, TValue value) noexcept;
        inline bool contains_key(const TKey& key) noexcept;
        inline TValue* get_value(const TKey& key) noexcept;
        inline TValue* operator[](const TKey& key) noexcept;
        inline TValue& get_value_unsafe(const TKey& key) noexcept;
        inline bool try_get_value(const TKey& key, TValue*& value_out) const noexcept;
//        struct Iterator; Iterator to_iterator() const noexcept;
    };
}

namespace nml::hash_internal
{
    static inline uint16_t hash_fragment(uint64_t hash) noexcept
    {
        return (hash >> 48) & 0xF000;
    }

    static inline uint64_t hash_value(uint64_t value) noexcept
    {
        value ^= value >> 23;
        value *= 0x2127599bf4325c37ull;
        value ^= value >> 47;

        return value;
    }

    static inline uint64_t hash_value(uint32_t value)
    {
        return hash_value(static_cast<uint64_t>(value));
    }

    static inline uint64_t hash_value(uint16_t value)
    {
        return hash_value(static_cast<uint64_t>(value));
    }

    static inline uint64_t hash_value(int32_t value) noexcept
    {
        uint32_t unsigned_value = reinterpret_cast<uint32_t&>(value);
        return hash_value(unsigned_value);
    }

    static inline uint64_t hash_value(int16_t value) noexcept
    {
        uint16_t unsigned_value = reinterpret_cast<uint16_t&>(value);
        return hash_value(static_cast<uint64_t>(unsigned_value));
    }

    static inline uint64_t hash_value(int64_t value) noexcept
    {
        uint64_t unsigned_value = reinterpret_cast<uint64_t&>(value);
        return hash_value(unsigned_value);
    }

    static inline uint64_t hash_value(float value)
    {
        uint32_t unsigned_value = reinterpret_cast<uint32_t&>(value);
        return hash_value(unsigned_value);
    }

    static inline uint64_t hash_value(double value)
    {
        uint64_t unsigned_value = reinterpret_cast<uint64_t&>(value);
        return hash_value(unsigned_value);
    }

    static inline uint64_t hash_value(const MemorySpan& memory) noexcept
    {
        uint64_t hash = 0xCBF29CE484222325;

        auto bytes = Span<unsigned char>(memory);

        for (uint64_t i = 0; i < bytes.length; ++i)
        {
            hash ^= static_cast<uint64_t>(bytes[i]);
            hash *= 0x100000001B3;
        }

        return hash;
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

    template<typename T>
    static inline uint64_t hash_value(const Span<T>& span) noexcept
    {
        return hash_value(span.to_memory_unsafe());
    }

    template<typename TIterator>
    static inline uint64_t hash_value(Iterator<char, TIterator>& iterator) noexcept
    {
        iterator.reset();

        uint64_t hash = 0xCBF29CE484222325;

        while (iterator.has_next())
        {
            auto next = (unsigned char)iterator.next();

            hash ^= static_cast<uint64_t>(next);
            hash *= 0x100000001B3;
        }

        return hash;
    }

    template<typename T, typename TIterator>
    static inline uint64_t hash_value(Iterator<T, TIterator>& iterator) noexcept
    {
        iterator.reset();

        uint64_t hash = 0xCBF29CE484222325;

        while (iterator.has_next())
        {
            T next = iterator.next();

            auto bytes = reinterpret_cast<const unsigned char*>(&next);
            const unsigned char* end = bytes + sizeof(T);

            while (bytes < end)
            {
                hash ^= static_cast<uint64_t>(*bytes++);
                hash *= 0x100000001B3;
            }
        }

        return hash;
    }


//    template<typename T>
//    static inline T copy_value_and_zero_padding(T value) noexcept
//    {
//        alignas(T) unsigned char memory[sizeof(T)];
//
//        memset(memory, 0, sizeof(T));
//
//        T* ptr = new (memory) T(value);
//
//        T result = *ptr;
//
//        if constexpr (!std::is_trivially_destructible<T>::value)
//        {
//            ptr->~T();
//        }
//
//        return result;
//    }
//
//    template<typename T>
//    static inline uint64_t hash_value(T value) noexcept
//    {
//        alignas(T) unsigned char memory[sizeof(T)];
//
//        memset(memory, 0, sizeof(T));
//
//        T* ptr = new (memory) T(value);
//
//        auto hash = hash_memory(memory, sizeof(T));
//
//        if constexpr (!std::is_trivially_destructible<T>::value)
//        {
//            ptr->~T();
//        }
//
//        return hash;
//    }


    static inline bool compare_values(int16_t left, int16_t right) noexcept
    {
        return left == right;
    }

    static inline bool compare_values(int32_t left, int32_t right) noexcept
    {
        return left == right;
    }

    static inline bool compare_values(int64_t left, int64_t right) noexcept
    {
        return left == right;
    }

    static inline bool compare_values(uint16_t left, uint16_t right) noexcept
    {
        return left == right;
    }

    static inline bool compare_values(uint32_t left, uint32_t right) noexcept
    {
        return left == right;
    }

    static inline bool compare_values(uint64_t left, uint64_t right) noexcept
    {
        return left == right;
    }

    static inline bool compare_values(float left, float right) noexcept
    {
        return left == right;
    }

    static inline bool compare_values(double left, double right) noexcept
    {
        return left == right;
    }

    static inline bool compare_values(const MemorySpan& left, const MemorySpan& right) noexcept
    {
        if (left.bytes != right.bytes) return false;

        return memcmp(left.get_pointer(), right.get_pointer(), left.bytes) == 0;
    }

    static inline bool compare_values(const char* left, const char* right) noexcept
    {
        return strcmp(left, right) == 0;
    }

    template<typename T>
    static inline bool compare_values(const Span<T>& left, const Span<T>& right) noexcept
    {
        return compare_values(left.to_memory_unsafe(), right.to_memory_unsafe());
    }

    template<typename T, typename TIterator>
    static inline bool compare_values(Iterator<T, TIterator>& left, Iterator<T, TIterator>& right) noexcept
    {
        int64_t left_length = left.length(), right_length = right.length();

        if (left_length >= 0 && right_length >= 0 && left_length != right_length)
        {
            return false;
        }

        left.reset(), right.reset();

        while (left.has_next() && right.has_next())
        {
            if (left.next() != right.next())
            {
                return false;
            }
        }

        return true;
    }

    static inline uint64_t quadratic(uint16_t displacement) noexcept
    {
        return ((uint64_t)displacement * displacement + displacement) / 2;
    }

    static const uint16_t zero = 0x0000;

    static inline Span<uint16_t> default_metadata() noexcept
    {
        return Span<uint16_t>((uint16_t*)&zero, 1);
    }

    struct Location
    {
        bool exists;
        uint16_t displacement;
        uint64_t buckets_offset;
    };

    struct MemoryLayout
    {
        uint64_t bucket_bytes;
        uint64_t metadata_bytes;

        [[nodiscard]] uint64_t total_bytes() const noexcept
        {
            return bucket_bytes + metadata_bytes;
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

        struct Iterator;

        Span<T> buckets;
        uint64_t key_ct;
        uint64_t buckets_mask;
        Span<uint16_t> metadata;

        bool resize() noexcept;
        T* find(T& value) noexcept;
        bool evict(size_t bucket) noexcept;
        [[nodiscard]] Iterator iterator() const noexcept;
        static inline uint64_t hash_key(T& value) noexcept;
        static inline bool compare(T& value, T& other) noexcept;
        [[nodiscard]] inline uint64_t bucket_ct() const noexcept;
        T* insert(T& value, bool unique) noexcept;
        inline Location find_first_empty_location(uint64_t home_bucket) noexcept;
        static inline MemoryLayout size_memory_layout(uint64_t bucket_ct) noexcept;
        uint64_t find_insert_location(uint64_t home_bucket, uint16_t displacement_to_empty) noexcept;
    };

    template<typename T>
    bool HashTable<T>::compare(T& value, T& other) noexcept
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
    inline uint64_t HashTable<T>::hash_key(T& value) noexcept
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
    HashTable<T>::HashTable() noexcept
        : key_ct(0), buckets(), metadata(default_metadata()), buckets_mask(32 - 1)
    {
        uint64_t bucket_count = bucket_ct();
        MemoryLayout layout = size_memory_layout(bucket_count);

        uint64_t total_bytes = layout.total_bytes();
        auto new_memory = MemorySpan(static_cast<char*>(std::malloc(total_bytes)), total_bytes);

        buckets = Span<T>(new_memory, bucket_count);
        metadata = Span<uint16_t>(new_memory.offset(layout.bucket_bytes), bucket_count);

        metadata.zero();
        metadata[bucket_count] = 0x01;
    }

    template<typename T>
    HashTable<T>::~HashTable() noexcept
    {
        std::free(buckets.get_pointer());
    }

    template<typename T>
    bool HashTable<T>::resize() noexcept
    {
        auto old_key_ct = key_ct;
        auto old_buckets = buckets;
        auto old_buckets_ct = bucket_ct();
        auto old_metadata = metadata;
        auto old_buckets_mask = buckets_mask;

        uint64_t new_bucket_ct = old_buckets_ct * 2;

        while (true)
        {
            MemoryLayout layout = size_memory_layout(new_bucket_ct);
            uint64_t total_bytes = layout.total_bytes();

            auto new_memory = MemorySpan(static_cast<char*>(std::malloc(total_bytes)), total_bytes);

            buckets_mask = new_bucket_ct - 1;
            buckets = Span<T>(new_memory, new_bucket_ct);
            metadata = Span<uint16_t>(new_memory.offset(layout.bucket_bytes), new_bucket_ct);

            metadata.zero();
            metadata[new_bucket_ct] = 0x01;

            key_ct = 0;

            for (uint64_t bucket = 0; bucket < old_buckets_ct; ++bucket)
            {
                if (old_metadata[bucket] == 0) continue;

                bool inserted = insert(old_buckets[bucket], true) != nullptr;

                if (!inserted) break;
            }

            if (key_ct < old_key_ct)
            {
                std::free(new_memory.get_pointer());

                new_bucket_ct *= 2;
                continue;
            }

            if (old_buckets_mask)
            {
                std::free(old_buckets.get_pointer());
            }

            return true;
        }
    }

    template<typename T>
    MemoryLayout HashTable<T>::size_memory_layout(uint64_t bucket_ct) noexcept
    {
        return MemoryLayout
        {
            .bucket_bytes = (((bucket_ct + 1) * sizeof(T) + sizeof(uint16_t) - 1) / sizeof(uint16_t)) * sizeof(uint16_t),
            .metadata_bytes = (bucket_ct + 1 + 4) * sizeof(uint16_t)
        };
    }

    template<typename T>
    T* HashTable<T>::find(T& value) noexcept
    {
        uint64_t hash = hash_key(value);
        uint64_t home_bucket = hash & buckets_mask;

        if (!(metadata[home_bucket] & 0x0800))
        {
            return nullptr;
        }

        uint16_t fragment = hash_fragment(hash);
        uint64_t bucket = home_bucket;

        while (true)
        {
            if ((metadata[bucket] & 0xF000) == fragment && compare(buckets[bucket], value))
            {
                return buckets.get_pointer(bucket);
            }

            uint16_t displacement = metadata[bucket] & 0x07FF;

            if (displacement == 0x07FF)
            {
                return nullptr;
            }

            bucket = (home_bucket + quadratic(displacement)) & buckets_mask;
        }
    }

    template<typename T>
    uint64_t HashTable<T>::bucket_ct() const noexcept
    {
        return buckets_mask + (bool)buckets_mask;
    }

    template<typename T>
    T* HashTable<T>::insert(T& value, bool unique) noexcept
    {
        uint64_t hash = hash_key(value);
        uint16_t fragment = hash_fragment(hash);
        uint64_t home_bucket = hash & buckets_mask;

        bool bucket_exists = metadata[home_bucket] & 0x0800;

        if (!bucket_exists)
        {
            if (key_ct + 1 > bucket_ct() * 0.6)
            {
                return nullptr;
            }

            if (metadata[home_bucket] != 0x0000)
            {
                bool evicted = evict(home_bucket);

                if (!evicted) return nullptr;
            }

            buckets[home_bucket] = value;
            metadata[home_bucket] = fragment | 0x0800 | 0x07FF;

            key_ct += 1;

            return &buckets[home_bucket];
        }

        if (!unique)
        {
            size_t bucket = home_bucket;

            while (true)
            {
                if ((metadata[bucket] & 0xF000) == fragment && compare(buckets[bucket], value))
                {
                    return &buckets[bucket];
                }

                uint16_t displacement = metadata[bucket] & 0x07FF;

                if (displacement == 0x07FF) break;

                bucket = (home_bucket + quadratic(displacement)) & buckets_mask;
            }
        }

        Location empty_location = find_first_empty_location(home_bucket);

        if (!empty_location.exists) return nullptr;

        uint64_t prev = find_insert_location(home_bucket, empty_location.displacement);

        buckets[empty_location.buckets_offset] = value;

        metadata[empty_location.buckets_offset] = fragment | (metadata[prev] & 0x07FF);
        metadata[prev] = (metadata[prev] & ~0x07FF) | empty_location.displacement;

        key_ct += 1;

        return &buckets[empty_location.buckets_offset];
    }

    template<typename T>
    Location HashTable<T>::find_first_empty_location(uint64_t home_bucket) noexcept
    {
        Location location =
        {
            .exists = false,
            .displacement = 1,
            .buckets_offset = 0,
        };

        uint64_t linear_displacement = 1;

        while (true)
        {
            location.buckets_offset = (home_bucket + linear_displacement) & buckets_mask;

            if (metadata[location.buckets_offset] == 0x0000)
            {
                location.exists = true;

                return location;
            }

            if (++location.displacement == 0x07FF)
            {
                return location;
            }

            linear_displacement += location.displacement;
        }
    }

    template<typename T>
    uint64_t HashTable<T>::find_insert_location(uint64_t home_bucket, uint16_t displacement_to_empty) noexcept
    {
        size_t candidate = home_bucket;

        while (true)
        {
            uint16_t displacement = metadata[candidate] & 0x07FF;

            if (displacement > displacement_to_empty)
            {
                return candidate;
            }

            candidate = (home_bucket + quadratic(displacement)) & buckets_mask;
        }
    }

    template<typename T>
    bool HashTable<T>::evict(size_t bucket) noexcept
    {
        uint64_t home_bucket = hash_key(buckets[bucket]) & buckets_mask;

        uint64_t previous = home_bucket;

        while (true)
        {
            uint64_t next = (home_bucket + quadratic(metadata[previous] & 0x07FF)) & buckets_mask;

            if (next == bucket) break;

            previous = next;
        }

        metadata[previous] = (metadata[previous] & ~0x07FF) | (metadata[bucket] & 0x07FF);

        Location empty_location = find_first_empty_location(home_bucket);

        if (!empty_location.exists) return false;

        previous = find_insert_location(home_bucket, empty_location.displacement);

        buckets[empty_location.buckets_offset] = buckets[bucket];

        metadata[empty_location.buckets_offset] = (metadata[bucket] & 0xF000) | (metadata[previous] & 0x07FF);
        metadata[previous] = (metadata[previous] & ~0x07FF) | empty_location.displacement;

        return true;
    }

    template<typename T>
    struct HashTable<T>::Iterator
    {
        const HashTable<T>& _table;

        T* _next;
        uint64_t _index;
        uint64_t _count;

        explicit Iterator(const HashTable<T>& table) noexcept
            : _next(nullptr), _table(table), _index(0), _count(0)
        { }

        [[nodiscard]] bool is_end() const noexcept
        {
            return _index >= _table.bucket_ct()
                || _count >= _table.key_ct;
        }

        bool has_next() noexcept
        {
            if (is_end()) return false;

            while (_index <= _table.buckets_mask)
            {
                if (_table.metadata[_index])
                {
                    _next = _table.buckets.get_pointer(_index);
                    _index += 1, _count += 1;
                    return true;
                }

                _index++;
            }

            return false;
        }

        void reset() noexcept
        {
            _next = 0;
            _index = 0;
            _count = 0;
        }
    };

    template<typename T>
    HashTable<T>::Iterator HashTable<T>::iterator() const noexcept
    {
        return Iterator(*this);
    }
}

namespace nml
{
    template<typename T>
    HashSet<T>::HashSet() noexcept
        : _table(hash_internal::HashTable<T>())
    { }

    template<typename T>
    class HashSet<T>::Iterator
    {
        friend HashSet<T>;
        hash_internal::HashTable<T>::Iterator _iterator;

        explicit Iterator(const hash_internal::HashTable<T>& set) noexcept
            : _iterator(set.iterator())
        { }

    public:

        [[nodiscard]] inline bool is_end() const noexcept
        {
            return _iterator.is_end();
        }

        [[nodiscard]] inline T next() const noexcept
        {
            return *_iterator._next;
        }

        [[nodiscard]] inline bool has_next() noexcept
        {
            return _iterator.has_next();
        }

        inline void reset() noexcept
        {
            _iterator.reset();
        }
    };

    template<typename T>
    HashSet<T>::Iterator HashSet<T>::to_iterator() const noexcept
    {
        return Iterator(_table);
    }

    template<typename T>
    inline bool HashSet<T>::contains(T& value) noexcept
    {
        return _table.find(value) != nullptr;
    }

    template<typename T>
    bool HashSet<T>::insert(T value) noexcept
    {
        while (true)
        {
            bool inserted = _table.insert(value, false) != nullptr;

            if (inserted) return true;

            _table.resize();
        }
    }

    template<typename T>
    bool HashSet<T>::remove(const T &value) noexcept
    {
        return false;
    }

    template<typename T>
    uint64_t HashSet<T>::count() const noexcept
    {
        return _table.key_ct;
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
    uint64_t HashMap<TKey, TValue>::count() noexcept
    {
        return _table.key_ct;
    }

    template<typename TKey, typename TValue>
    bool HashMap<TKey, TValue>::remove(const TKey& key) noexcept
    {
        return false;
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
    TValue& HashMap<TKey, TValue>::insert(const TKey& key, const TValue& value) noexcept
    {
        auto pair = std::pair<TKey, TValue>(key, value);

        while (true)
        {
            auto inserted = _table.insert(pair, false);

            if (inserted != nullptr)
            {
                return inserted->second;
            }

            _table.resize();
        }
    }

    template<typename TKey, typename TValue>
    bool HashMap<TKey, TValue>::contains_key(const TKey& key) noexcept
    {
        std::pair<TKey, TValue> pair;

        pair.first = key;

        return _table.find(pair) != nullptr;
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
}

#endif //NML_HASH_H