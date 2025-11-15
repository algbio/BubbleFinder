#pragma once

#include <chrono>
#include <cstdio>
#include "util/context.hpp"

class ScopeTimer
{
public:
    explicit ScopeTimer(const char* label) noexcept
        : label_{label},
          start_{std::chrono::steady_clock::now()}
    {}

    ~ScopeTimer() noexcept
    {
        auto &C = ctx();
        if (!C.timingEnabled) return;

        using namespace std::chrono;
        const long long ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_
            ).count();

        std::fprintf(stderr, "[timer] %s : %lld ms\n", label_, ms);
    }

    ScopeTimer(const ScopeTimer&)            = delete;
    ScopeTimer& operator=(const ScopeTimer&) = delete;

private:
    const char*                                 label_;
    const std::chrono::steady_clock::time_point start_;
};


#if defined(__clang__)
#  define TIMER_DIAG_PUSH  _Pragma("clang diagnostic push")
#  define TIMER_DIAG_IGN   _Pragma("clang diagnostic ignored \"-Wshadow\"")
#  define TIMER_DIAG_POP   _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
#  define TIMER_DIAG_PUSH  _Pragma("GCC diagnostic push")
#  define TIMER_DIAG_IGN   _Pragma("GCC diagnostic ignored \"-Wshadow\"")
#  define TIMER_DIAG_POP   _Pragma("GCC diagnostic pop")
#else
#  define TIMER_DIAG_PUSH
#  define TIMER_DIAG_IGN
#  define TIMER_DIAG_POP
#endif

#define TIME_BLOCK(label) TIMER_DIAG_PUSH TIMER_DIAG_IGN ::ScopeTimer timer_##__COUNTER__{label}; TIMER_DIAG_POP

#define TIME_BLOCK_SCOPE(label, code)                                     \
    do {                                                                  \
        auto _tb_start = std::chrono::high_resolution_clock::now();       \
        code;                                                             \
        auto _tb_end = std::chrono::high_resolution_clock::now();         \
        auto _tb_dur = std::chrono::duration_cast<std::chrono::milliseconds>( \
                           _tb_end - _tb_start)                           \
                           .count();                                      \
        logger::info("{} took {} ms", label, _tb_dur);                    \
    } while (0)

