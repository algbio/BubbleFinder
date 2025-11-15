#pragma once
#include <chrono>
#include <string>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>
#include <unistd.h>
#include "util/logger.hpp"

namespace memtime {

inline size_t parse_kb_from_status(const char* key) {
#ifdef __linux__
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[4096];
    long kb = 0;
    const size_t keylen = std::strlen(key);
    while (fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, key, keylen) == 0) {
            for (char* p = line + keylen; *p; ++p) {
                if (*p >= '0' && *p <= '9') {
                    std::sscanf(p, "%ld", &kb);
                    break;
                }
            }
            break;
        }
    }
    fclose(f);
    return (size_t)kb * 1024ULL;
#else
    (void)key;
    return 0;
#endif
}

inline size_t currentRSSBytes() {
#ifdef __linux__
    return parse_kb_from_status("VmRSS:");
#else
    return 0;
#endif
}

inline size_t peakRSSBytes() {
#ifdef __linux__
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
    #if defined(__linux__)
        return (size_t)ru.ru_maxrss * 1024ULL;
    #else
        return (size_t)ru.ru_maxrss;
    #endif
    }
    return parse_kb_from_status("VmHWM:");
#else
    return 0;
#endif
}

struct Scope {
    std::string name;
    std::chrono::steady_clock::time_point t0;
    size_t rss0;
    size_t hwm0;

    explicit Scope(const char* n)
        : name(n ? n : ""),
          t0(std::chrono::steady_clock::now()),
          rss0(currentRSSBytes()),
          hwm0(peakRSSBytes()) {}

    explicit Scope(const std::string& n)
        : name(n),
          t0(std::chrono::steady_clock::now()),
          rss0(currentRSSBytes()),
          hwm0(peakRSSBytes()) {}

    ~Scope() {
        const auto t1 = std::chrono::steady_clock::now();
        [[maybe_unused]] const auto   ms   =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        [[maybe_unused]] const size_t rss1 = currentRSSBytes();
        [[maybe_unused]] const size_t hwm1 = peakRSSBytes();
        [[maybe_unused]] const double MiB  = 1024.0 * 1024.0;
    }
};

} 


#if defined(__clang__)
#  define MEMTIME_DIAG_PUSH  _Pragma("clang diagnostic push")
#  define MEMTIME_DIAG_IGN   _Pragma("clang diagnostic ignored \"-Wshadow\"")
#  define MEMTIME_DIAG_POP   _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
#  define MEMTIME_DIAG_PUSH  _Pragma("GCC diagnostic push")
#  define MEMTIME_DIAG_IGN   _Pragma("GCC diagnostic ignored \"-Wshadow\"")
#  define MEMTIME_DIAG_POP   _Pragma("GCC diagnostic pop")
#else
#  define MEMTIME_DIAG_PUSH
#  define MEMTIME_DIAG_IGN
#  define MEMTIME_DIAG_POP
#endif

#define MEM_TIME_BLOCK(name_literal_or_string) MEMTIME_DIAG_PUSH MEMTIME_DIAG_IGN memtime::Scope memtime_scope_##__COUNTER__ (name_literal_or_string); MEMTIME_DIAG_POP