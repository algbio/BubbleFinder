#pragma once
#include <string_view>          
#include <spdlog/spdlog.h>      
#include <spdlog/fmt/ostr.h>    

namespace logger   
{
using sv = std::string_view;    

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

void init();    
void flush();    
} 