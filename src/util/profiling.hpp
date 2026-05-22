// profiling.hpp
#pragma once

#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>

#ifdef ENABLE_PROFILING

class Profiler {
public:
    using Clock = std::chrono::high_resolution_clock;
    using Duration = Clock::duration;

    static Profiler& instance() {
        static Profiler inst;
        return inst;
    }

    void addTime(const std::string& name, Duration dur) {
        std::lock_guard<std::mutex> lock(mu_);
        data_[name] += dur;
    }

    void report() const {
        auto it = data_.find("Total run");
        Duration total = (it != data_.end())
                         ? it->second
                         : [&] {
                               Duration sum = Duration::zero();
                               for (auto const& kv : data_) sum += kv.second;
                               return sum;
                           }();

        double total_s = std::chrono::duration<double>(total).count();
        if (total_s <= 0.0) {
            std::cout << "=== Profiling Report ===\n"
                      << "(no time recorded)\n";
            return;
        }

        std::vector<std::pair<std::string, Duration>> items;
        items.reserve(data_.size());
        for (auto const& kv : data_)
            items.emplace_back(kv.first, kv.second);

        std::sort(items.begin(), items.end(),
                  [](auto const& a, auto const& b) {
                      return a.second > b.second;
                  });

        std::cout << "=== Profiling Report (percent of \"Total run\") ===\n";
        for (auto const& kv : items) {
            double secs = std::chrono::duration<double>(kv.second).count();
            double pct  = (secs / total_s) * 100.0;
            std::cout
                << kv.first
                << ": " << secs << " s"
                << " (" << pct << "%)\n";
        }
    }

private:
    Profiler() = default;
    ~Profiler() = default;

    mutable std::mutex               mu_;
    std::map<std::string, Duration>  data_;
};

struct ProfileBlock {
    ProfileBlock(const char* name)
      : name_(name), start_(Profiler::Clock::now())
    {}

    ~ProfileBlock() {
        auto end = Profiler::Clock::now();
        Profiler::instance().addTime(name_, end - start_);
    }

private:
    const char*                        name_;
    Profiler::Clock::time_point        start_;
};

#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b)      CONCAT_IMPL(a, b)

#define PROFILE_BLOCK(name) \
    ProfileBlock CONCAT(_prof_block_, __LINE__){ name }

#define PROFILE_FUNCTION()  PROFILE_BLOCK(__func__)

#define PROFILING_REPORT()  Profiler::instance().report()

#else  

#define PROFILE_BLOCK(name)
#define PROFILE_FUNCTION()
#define PROFILING_REPORT()

#endif 