#pragma once

#include <atomic>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <algorithm>
#include <string>
#include <cstdio>

#include "util/logger.hpp"
#include "util/mem_time.hpp"

namespace mark {

struct Stat {
    std::atomic<long long> ns{0};
    std::atomic<long long> calls{0};

    std::atomic<long long> mem_calls{0};
    std::atomic<long long> rss_delta_sum{0};
    std::atomic<long long> hwm_delta_sum{0};

    std::atomic<unsigned long long> rss_max{0};
    std::atomic<unsigned long long> hwm_max{0};

    const char* name{nullptr};
};

inline std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

inline std::unordered_map<const char*, Stat*>& registry_map() {
    static std::unordered_map<const char*, Stat*> m;
    return m;
}

inline std::vector<Stat*>& registry_list() {
    static std::vector<Stat*> v;
    return v;
}

inline Stat* register_label(const char* label) {
    auto &m = registry_map();
    std::lock_guard<std::mutex> lk(registry_mutex());
    auto it = m.find(label);
    if (it != m.end()) return it->second;

    Stat* s = new Stat();
    s->name = label;
    m[label] = s;
    registry_list().push_back(s);
    return s;
}

struct Scope {
    Stat* s;
    bool doMem;
    std::chrono::steady_clock::time_point t0;
    size_t rss0{0}, hwm0{0};

    Scope(Stat* statPtr, bool enableMem)
        : s(statPtr), doMem(enableMem),
          t0(std::chrono::steady_clock::now())
    {
        if (doMem) {
            rss0 = memtime::currentRSSBytes();
            hwm0 = memtime::peakRSSBytes();
        }
    }

    ~Scope() {
        const auto t1 = std::chrono::steady_clock::now();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        s->ns.fetch_add(ns, std::memory_order_relaxed);
        s->calls.fetch_add(1, std::memory_order_relaxed);

        if (doMem) {
            const size_t rss1 = memtime::currentRSSBytes();
            const size_t hwm1 = memtime::peakRSSBytes();

            const long long dRss = static_cast<long long>(rss1) - static_cast<long long>(rss0);
            const long long dHwm = static_cast<long long>(hwm1) - static_cast<long long>(hwm0);

            s->rss_delta_sum.fetch_add(dRss, std::memory_order_relaxed);
            s->hwm_delta_sum.fetch_add(dHwm, std::memory_order_relaxed);
            s->mem_calls.fetch_add(1, std::memory_order_relaxed);

            auto prevRss = s->rss_max.load(std::memory_order_relaxed);
            while (prevRss < rss1 &&
                   !s->rss_max.compare_exchange_weak(
                        prevRss,
                        static_cast<unsigned long long>(rss1),
                        std::memory_order_relaxed))
            {}

            auto prevHwm = s->hwm_max.load(std::memory_order_relaxed);
            while (prevHwm < hwm1 &&
                   !s->hwm_max.compare_exchange_weak(
                        prevHwm,
                        static_cast<unsigned long long>(hwm1),
                        std::memory_order_relaxed))
            {}
        }
    }
};

inline void report(std::size_t topN = 0) {
    auto &lst = registry_list();
    std::vector<Stat*> v(lst.begin(), lst.end());
    std::sort(v.begin(), v.end(), [](const Stat* a, const Stat* b) {
        return a->ns.load(std::memory_order_relaxed) > b->ns.load(std::memory_order_relaxed);
    });
    if (topN > 0 && v.size() > topN) v.resize(topN);

    logger::info("Detailed time marks (aggregated):");
    for (auto* s : v) {
        const double secs      = s->ns.load(std::memory_order_relaxed) / 1e9;
        const long long calls  = s->calls.load(std::memory_order_relaxed);
        const long long mcalls = s->mem_calls.load(std::memory_order_relaxed);

        const double rssSumMB  = s->rss_delta_sum.load(std::memory_order_relaxed) / (1024.0 * 1024.0);
        const double hwmSumMB  = s->hwm_delta_sum.load(std::memory_order_relaxed) / (1024.0 * 1024.0);
        const double hwmMaxMB  = s->hwm_max.load(std::memory_order_relaxed) / (1024.0 * 1024.0);

        const char* name = (s->name ? s->name : "(null)");

        logger::info(
            "  {:8.3f}s | {:8} calls | mem {:7} | ΔRSSsum {:9.2f} MiB | ΔHWMsum {:9.2f} MiB | HWMmax {:9.2f} MiB | {}",
            secs, calls, mcalls, rssSumMB, hwmSumMB, hwmMaxMB, name);
    }
}

inline void report_to_json(const std::string& path, std::size_t topN = 0) {
    auto &lst = registry_list();
    std::vector<Stat*> v(lst.begin(), lst.end());
    std::sort(v.begin(), v.end(), [](const Stat* a, const Stat* b) {
        return a->ns.load(std::memory_order_relaxed) > b->ns.load(std::memory_order_relaxed);
    });
    if (topN > 0 && v.size() > topN) v.resize(topN);

    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        logger::info("mark::report_to_json: cannot open {}", path);
        return;
    }

    std::fprintf(f, "{\n  \"marks\": [\n");
    for (size_t i = 0; i < v.size(); ++i) {
        Stat* s = v[i];
        const double secs      = s->ns.load(std::memory_order_relaxed) / 1e9;
        const long long calls  = s->calls.load(std::memory_order_relaxed);
        const long long mcalls = s->mem_calls.load(std::memory_order_relaxed);

        const long long rssSum = s->rss_delta_sum.load(std::memory_order_relaxed);
        const long long hwmSum = s->hwm_delta_sum.load(std::memory_order_relaxed);
        const unsigned long long rssMax = s->rss_max.load(std::memory_order_relaxed);
        const unsigned long long hwmMax = s->hwm_max.load(std::memory_order_relaxed);

        const char* name = (s->name ? s->name : "(null)");

        std::fprintf(
            f,
            "    {\"label\": \"%s\", "
            "\"seconds\": %.9f, "
            "\"calls\": %lld, "
            "\"mem_calls\": %lld, "
            "\"rss_delta_sum_bytes\": %lld, "
            "\"hwm_delta_sum_bytes\": %lld, "
            "\"rss_max_bytes\": %llu, "
            "\"hwm_max_bytes\": %llu}%s\n",
            name, secs, calls, mcalls,
            rssSum, hwmSum, rssMax, hwmMax,
            (i + 1 < v.size()) ? "," : "");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    logger::info("Detailed time marks written to JSON {}", path);
}

} 
#define MARK_JOIN_IMPL(a, b) a##b
#define MARK_JOIN(a, b)      MARK_JOIN_IMPL(a, b)

#if defined(__COUNTER__)
#  define MARK_UNIQ() __COUNTER__
#else
#  define MARK_UNIQ() __LINE__
#endif

#define MARK_SCOPE_MEM(label_literal) \
    MARK_SCOPE_MEM_IMPL(label_literal, MARK_UNIQ())

#define MARK_SCOPE_MEM_IMPL(label_literal, N)                                      \
    static mark::Stat* MARK_JOIN(mark_mstat_, N) = mark::register_label(label_literal); \
    mark::Scope        MARK_JOIN(mark_mscope_, N)(MARK_JOIN(mark_mstat_, N), true)
