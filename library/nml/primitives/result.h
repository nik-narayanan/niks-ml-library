//
// Created by nik on 3/1/2024.
//

#ifndef RESULT_H
#define RESULT_H

template <typename TData, typename TError>
class Result
{
    union
    {
        TData data;
        TError error;
    };

    const bool _is_ok;  // wrecks alignment for small TData and adds more indirection as tradeoff for noexcept

public:

    explicit Result(TData& data) noexcept : data(data), _is_ok(true) {}
    explicit Result(TData&& data) noexcept : data(std::move(data)), _is_ok(true) {}
    explicit Result(TError& error) noexcept : error(error), _is_ok(false) {}
    explicit Result(TError&& error) noexcept : error(std::move(error)), _is_ok(false) {}

    static Result<TData, TError> ok(TData& data) noexcept { return Result<TData, TError>(data); }
    static Result<TData, TError> ok(TData&& data) noexcept { return Result<TData, TError>(std::move(data)); }
    static Result<TData, TError> err(const TError& error) noexcept { return Result<TData, TError>(error); }
    static Result<TData, TError> err(TError&& error) noexcept { return Result<TData, TError>(std::move(error)); }

    Result(const Result&) = delete;
    float operator=(const Result&) = delete;
    Result& operator=(const Result&&) = delete;

    Result(Result&& other) noexcept : _is_ok(other._is_ok)
    {
        if (is_ok()) new (&data) TData(std::move(other.data));
        else new (&error) TError(std::move(other.error));
    }

    ~Result() noexcept
    {
        if (is_ok()) data.~TData();
        else error.~TError();
    }

    [[nodiscard]] inline bool is_ok() const noexcept { return _is_ok; }
    [[nodiscard]] inline bool is_err() const noexcept { return !_is_ok; }

    TError err() noexcept
    {
        return std::move(error);
    }

    TData ok() noexcept
    {
        return std::move(data);
    }
};

template<typename TData>
using StringResult = Result<TData, std::string>;

#endif //RESULT_H