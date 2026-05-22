#include "util/logger.hpp"
#include "util/context.hpp"
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/cfg/env.h> 
#include <memory>

namespace {

std::shared_ptr<spdlog::logger> make_default()
{
    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");


    spdlog::init_thread_pool(8192, 1);

    auto lg = std::make_shared<spdlog::async_logger>(
        "default", spdlog::sinks_init_list{console},
        spdlog::thread_pool(),
        spdlog::async_overflow_policy::overrun_oldest);

    lg->set_level(static_cast<spdlog::level::level_enum>(ctx().logLevel));
    spdlog::set_default_logger(lg);
    spdlog::cfg::load_env_levels();
    return lg;
}

auto dummy = make_default();      
} 

void logger::flush() { spdlog::default_logger()->flush(); }
void logger::init()  { }
