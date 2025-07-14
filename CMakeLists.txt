cmake_minimum_required(VERSION 3.20)
project(sbSPQRnew LANGUAGES CXX)

# ------------------------------------------------------------------#
# Global options                                                    #
# ------------------------------------------------------------------#
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_SANITIZERS
       "Build with AddressSanitizer + UndefinedBehaviourSanitizer" OFF)

# ------------------------------------------------------------------#
# Sanitizer flags (GCC / Clang)                                     #
# ------------------------------------------------------------------#
if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -fsanitize=address
        -fsanitize=undefined
        -fno-omit-frame-pointer    # keep full back-traces
        -fno-pie                   # avoid ASan shadow-range clash
        -g                         # debug symbols
    )
    add_link_options(
        -fsanitize=address
        -fsanitize=undefined
        -no-pie                    # link a non-PIE executable
    )
    message(STATUS "Building with AddressSanitizer + UBSan (no-PIE)")
endif()

# ------------------------------------------------------------------#
# 3rd-party: spdlog (header-only)                                   #
# ------------------------------------------------------------------#
include(FetchContent)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.15.3
)
FetchContent_MakeAvailable(spdlog)

# ------------------------------------------------------------------#
# 3rd-party: OGDF (built/installed elsewhere)                       #
# ------------------------------------------------------------------#
set(OGDF_DIR "/home/politov/OGDF" CACHE PATH "OGDF install prefix")
find_package(OGDF CONFIG REQUIRED
             PATHS "${OGDF_DIR}/lib/cmake")

# ------------------------------------------------------------------#
# Project subdirectories (static libs)                              #
# ------------------------------------------------------------------#
add_subdirectory(src/util)   # → libsbutil
add_subdirectory(src/io)     # → libsbio

# ------------------------------------------------------------------#
# Main executable                                                   #
# ------------------------------------------------------------------#
add_executable(sbfind src/main.cpp)
target_include_directories(sbfind PRIVATE src)   # #include "util/…"
target_link_libraries(sbfind PRIVATE
    sbio
    sbutil
    OGDF
    spdlog::spdlog_header_only
)

# ------------------------------------------------------------------#
# Nice warnings                                                     #
# ------------------------------------------------------------------#
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sbfind PRIVATE -Wall -Wextra -Wshadow -pedantic)
endif()
