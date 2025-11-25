cmake_minimum_required(VERSION 3.21)
project(sbSPQRnew LANGUAGES CXX VERSION 1.0)

# 0. default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 1. C++ standard — use a single standard (OGDF → often C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 2. FetchContent
include(FetchContent)

# spdlog (header-only)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.15.3
)
FetchContent_MakeAvailable(spdlog)

# 3. OGDF – either bundled via FetchContent or found as a system package
option(SB_BUNDLE_OGDF "Download & build OGDF locally" ON)
set(OGDF_TAG "elderberry-202309" CACHE STRING "OGDF tag/branch/commit")

if(SB_BUNDLE_OGDF)
    FetchContent_Declare(
        ogdf
        GIT_REPOSITORY https://github.com/ogdf/ogdf.git
        GIT_TAG        ${OGDF_TAG}
        GIT_SHALLOW    TRUE
    )
    # reduce what OGDF builds
    set(OGDF_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(OGDF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(ogdf)
endif()

find_package(OGDF CONFIG QUIET)

if(TARGET OGDF::OGDF)
    set(OGDF_LIB OGDF::OGDF)
elseif(TARGET ogdf)
    set(OGDF_LIB ogdf)
elseif(TARGET OGDF)
    set(OGDF_LIB OGDF)
else()
    message(FATAL_ERROR "No OGDF library target available.")
endif()

# 4. Subprojects
add_subdirectory(src/util)  # sbutil
add_subdirectory(src/io)    # sbio

# 5. main executable
add_executable(BubbleFinder src/main.cpp src/fas.cpp)
target_include_directories(BubbleFinder PRIVATE src)

target_link_libraries(BubbleFinder PRIVATE
    sbio sbutil
    ${OGDF_LIB}
    spdlog::spdlog_header_only
)

# 6. Sanitizers
option(SB_ENABLE_SANITIZERS "Enable Address/UndefinedBehavior sanitizers (Debug)" OFF)
if(SB_ENABLE_SANITIZERS)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "SB_ENABLE_SANITIZERS is ON but CMAKE_BUILD_TYPE != Debug")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        if(UNIX AND NOT APPLE)
            add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie -g)
            add_link_options   (-fsanitize=address,undefined -no-pie)
        else()
            add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
            add_link_options   (-fsanitize=address,undefined)
        endif()
    endif()
endif()

# 7. LTO / IPO
option(SB_ENABLE_LTO "Enable Link Time Optimization (IPO/LTO)" ON)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok OUTPUT ipo_err)
if(SB_ENABLE_LTO AND ipo_ok AND NOT SB_ENABLE_SANITIZERS)
    foreach(t BubbleFinder sbio sbutil)
        if(TARGET ${t})
            set_property(TARGET ${t} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
        endif()
    endforeach()
endif()

# 8. Per-configuration options and warnings
foreach(t BubbleFinder sbio sbutil)
    if(TARGET ${t})
        target_compile_options(${t} PRIVATE
            $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -flto -pthread>
            $<$<CONFIG:RelWithDebInfo>:-O2 -g -DENABLE_PROFILING -fno-omit-frame-pointer>
            $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer>
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${t} PRIVATE -Wall -Wextra -Wshadow -pedantic)
        endif()
    endif()
endforeach()

# 9. Help messages
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OGDF target: ${OGDF_LIB}")
message(STATUS "Sanitizers: ${SB_ENABLE_SANITIZERS}")
message(STATUS "Enable LTO: ${SB_ENABLE_LTO} (ipo_ok=${ipo_ok})")

# 10. Tests
add_executable(snarls_bf src/snarls_bruteforce.cpp)
add_executable(superbubbles_bf src/superbubbles_bruteforce.cpp)