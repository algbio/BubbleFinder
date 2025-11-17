cmake_minimum_required(VERSION 3.21)
project(sbSPQRnew LANGUAGES CXX VERSION 1.0)

# 0. build-type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 1. C++ standard — utiliser un unique standard (OGDF -> souvent C++17)
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

# 3. OGDF – soit bundle via FetchContent, soit trouve package système
option(SB_BUNDLE_OGDF "Download & build OGDF locally" ON)
set(OGDF_TAG "elderberry-202309" CACHE STRING "OGDF tag/branch/commit")

if(SB_BUNDLE_OGDF)
    FetchContent_Declare(
        ogdf
        GIT_REPOSITORY https://github.com/ogdf/ogdf.git
        GIT_TAG        ${OGDF_TAG}
        GIT_SHALLOW    TRUE
    )
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
    message(FATAL_ERROR "No OGDF library target available. Build with SB_BUNDLE_OGDF=ON or install OGDF and pass its CMake package path.")
endif()

# 4. Subprojects
add_subdirectory(src/util)   # sbutil
add_subdirectory(src/io)     # sbio

# 5. main executable
add_executable(BubbleFinder src/main.cpp src/fas.cpp)
target_include_directories(BubbleFinder PRIVATE src)

target_link_libraries(BubbleFinder PRIVATE
    sbio sbutil
    ${OGDF_LIB}
    spdlog::spdlog_header_only
)

# 10. Brute force tester
add_executable(snarls_bf src/snarls_bruteforce.cpp)

# 6. Sanitizers (Address + UB)
option(SB_ENABLE_SANITIZERS "Enable Address/UndefinedBehavior sanitizers (Debug)" OFF)
if(SB_ENABLE_SANITIZERS)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "SB_ENABLE_SANITIZERS is ON but CMAKE_BUILD_TYPE != Debug — sanitizers are intended for Debug builds")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(SAN_FLAGS "-fsanitize=address,undefined" "-fno-omit-frame-pointer" "-g")
        set(SAN_LINK_FLAGS "-fsanitize=address,undefined")

        if(UNIX AND NOT APPLE)
            list(APPEND SAN_FLAGS "-fno-pie")
            list(APPEND SAN_LINK_FLAGS "-no-pie")
        endif()

        foreach(t BubbleFinder sbio sbutil snarls_bf)
            if(TARGET ${t})
                target_compile_options(${t} PRIVATE ${SAN_FLAGS})
                target_link_options(${t} PRIVATE ${SAN_LINK_FLAGS})
            endif()
        endforeach()

        message(STATUS "Sanitizers enabled: ${SAN_FLAGS}")
    else()
        message(WARNING "Sanitizers requested but compiler is not GNU/Clang — skipping sanitizer flags")
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
else()
    if(SB_ENABLE_LTO AND NOT ipo_ok)
        message(WARNING "IPO/LTO not supported: ${ipo_err}")
    elseif(SB_ENABLE_SANITIZERS)
        message(STATUS "Disabling IPO/LTO because sanitizers are enabled")
    endif()
endif()

# 8. Options par configuration et warnings
foreach(t BubbleFinder sbio sbutil snarls_bf)
    if(TARGET ${t})
        target_compile_options(${t} PRIVATE
            $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -pthread>
            $<$<CONFIG:RelWithDebInfo>:-O2 -g -DENABLE_PROFILING -fno-omit-frame-pointer>
            $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer>
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${t} PRIVATE -Wall -Wextra -Wshadow -pedantic)
        endif()
    endif()
endforeach()

# 9. Messages d’aide
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OGDF target: ${OGDF_LIB}")
message(STATUS "Sanitizers: ${SB_ENABLE_SANITIZERS}")
message(STATUS "Enable LTO: ${SB_ENABLE_LTO} (ipo_ok=${ipo_ok})")
