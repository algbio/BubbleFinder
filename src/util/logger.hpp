#pragma once
#include <string_view>          // std::string_view
#include <spdlog/spdlog.h>      // brings its own fmt copy
#include <spdlog/fmt/ostr.h>    // enables << for user types

namespace logger   // <- NOT “log”, avoids clash with <math.h>::log
{
using sv = std::string_view;    // shorthand

// ──────────────────────────────────────────────
// Each wrapper converts the string_view to a
// run-time fmt object via spdlog::fmt_lib::runtime.
// This removes the “not a constant expression” error.
// ──────────────────────────────────────────────
template<class... Args>
inline void trace(sv fmt, Args&&... args)
{
    spdlog::trace(spdlog::fmt_lib::runtime(fmt),
                  std::forward<Args>(args)...);
}

template<class... Args>
inline void debug(sv fmt, Args&&... args)
{
    spdlog::debug(spdlog::fmt_lib::runtime(fmt),
                  std::forward<Args>(args)...);
}

template<class... Args>
inline void info(sv fmt, Args&&... args)
{
    spdlog::info(spdlog::fmt_lib::runtime(fmt),
                 std::forward<Args>(args)...);
}

template<class... Args>
inline void warn(sv fmt, Args&&... args)
{
    spdlog::warn(spdlog::fmt_lib::runtime(fmt),
                 std::forward<Args>(args)...);
}

template<class... Args>
inline void error(sv fmt, Args&&... args)
{
    spdlog::error(spdlog::fmt_lib::runtime(fmt),
                  std::forward<Args>(args)...);
}

// optional helpers implemented in logger.cpp
void init();     // configure sinks / level
void flush();    // force-write logs immediately
} // namespace logging
