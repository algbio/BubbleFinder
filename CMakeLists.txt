cmake_minimum_required(VERSION 3.21)
project(sbSPQRnew LANGUAGES CXX VERSION 1.0)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SB_NATIVE_OPTIMIZATIONS "Use -march=native in Release builds" ON)
option(SB_ENABLE_SANITIZERS    "Enable ASan/UBSan (Debug)"           OFF)
option(SB_ENABLE_LTO           "Enable Link Time Optimization"       ON)
option(SB_BUNDLE_OGDF          "Download & build OGDF locally"       ON)
option(BUBBLEFINDER_HAS_GBZ    "Enable GBZ file format support"      OFF)

set(OGDF_TAG "elderberry-202309" CACHE STRING "OGDF tag/branch/commit")


include(FetchContent)

# spdlog (header-only)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.15.3)
FetchContent_MakeAvailable(spdlog)

# OGDF
if(SB_BUNDLE_OGDF)
    FetchContent_Declare(ogdf
        GIT_REPOSITORY https://github.com/ogdf/ogdf.git
        GIT_TAG        ${OGDF_TAG}
        GIT_SHALLOW    TRUE)
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


add_library(sbutil OBJECT
    src/util/context.cpp
    src/util/logger.cpp)
target_include_directories(sbutil
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/util)
target_link_libraries(sbutil PUBLIC ${OGDF_LIB} spdlog::spdlog_header_only)

add_library(sbio OBJECT
    src/io/graph_io.cpp)
target_include_directories(sbio
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/io)
target_link_libraries(sbio PUBLIC sbutil ${OGDF_LIB})


if(BUBBLEFINDER_HAS_GBZ)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/gbz/gbwtgraph/include")
        message(FATAL_ERROR
            "GBZ support requested but submodules not found.\n"
            "Run:\n"
            "  mkdir -p external/gbz && cd external/gbz\n"
            "  git submodule add https://github.com/vgteam/sdsl-lite.git\n"
            "  git submodule add https://github.com/vgteam/libhandlegraph.git\n"
            "  git submodule add https://github.com/jltsiren/gbwt.git\n"
            "  git submodule add https://github.com/jltsiren/gbwtgraph.git")
    endif()

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ZSTD   REQUIRED libzstd)
    pkg_check_modules(CRYPTO REQUIRED libcrypto)
    find_package(OpenMP)

    # sdsl-lite (vgteam fork) — needs policy compat for old libdivsufsort dep
    set(WITH_GOOGLETEST OFF CACHE BOOL "" FORCE)
    set(GENERATE_DOC    OFF CACHE BOOL "" FORCE)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
    add_subdirectory(external/gbz/sdsl-lite EXCLUDE_FROM_ALL)

    # libhandlegraph
    add_subdirectory(external/gbz/libhandlegraph EXCLUDE_FROM_ALL)

    # gbwt (no CMake upstream — build manually)
    add_library(gbwt STATIC
        external/gbz/gbwt/src/algorithms.cpp
        external/gbz/gbwt/src/bwtmerge.cpp
        external/gbz/gbwt/src/cached_gbwt.cpp
        external/gbz/gbwt/src/dynamic_gbwt.cpp
        external/gbz/gbwt/src/fast_locate.cpp
        external/gbz/gbwt/src/files.cpp
        external/gbz/gbwt/src/gbwt.cpp
        external/gbz/gbwt/src/internal.cpp
        external/gbz/gbwt/src/metadata.cpp
        external/gbz/gbwt/src/support.cpp
        external/gbz/gbwt/src/test.cpp
        external/gbz/gbwt/src/utils.cpp
        external/gbz/gbwt/src/variants.cpp)
    target_include_directories(gbwt PUBLIC external/gbz/gbwt/include)
    target_link_libraries(gbwt PUBLIC sdsl ${ZSTD_LIBRARIES})
    target_include_directories(gbwt SYSTEM PUBLIC ${ZSTD_INCLUDE_DIRS})
    target_compile_features(gbwt PUBLIC cxx_std_17)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gbwt PUBLIC OpenMP::OpenMP_CXX)
    endif()

    add_library(gbwtgraph STATIC
        external/gbz/gbwtgraph/src/algorithms.cpp
        external/gbz/gbwtgraph/src/cached_gbwtgraph.cpp
        external/gbz/gbwtgraph/src/gbwtgraph.cpp
        external/gbz/gbwtgraph/src/gbz.cpp
        external/gbz/gbwtgraph/src/gfa.cpp
        external/gbz/gbwtgraph/src/index.cpp
        external/gbz/gbwtgraph/src/internal.cpp
        external/gbz/gbwtgraph/src/minimizer.cpp
        external/gbz/gbwtgraph/src/naive_graph.cpp
        external/gbz/gbwtgraph/src/path_cover.cpp
        external/gbz/gbwtgraph/src/subgraph.cpp
        external/gbz/gbwtgraph/src/utils.cpp)
    target_include_directories(gbwtgraph PUBLIC external/gbz/gbwtgraph/include)
    target_link_libraries(gbwtgraph PUBLIC
        gbwt handlegraph_shared sdsl
        ${ZSTD_LIBRARIES} ${CRYPTO_LIBRARIES})
    target_include_directories(gbwtgraph SYSTEM PUBLIC
        ${ZSTD_INCLUDE_DIRS} ${CRYPTO_INCLUDE_DIRS})
    target_compile_features(gbwtgraph PUBLIC cxx_std_17)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gbwtgraph PUBLIC OpenMP::OpenMP_CXX)
    endif()

    target_link_libraries(sbio PUBLIC gbwtgraph)
    target_compile_definitions(sbio PUBLIC BUBBLEFINDER_HAS_GBZ)

    message(STATUS "GBZ support: ON")
else()
    message(STATUS "GBZ support: OFF (use -DBUBBLEFINDER_HAS_GBZ=ON to enable)")
endif()


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/clsd)

add_executable(BubbleFinder
    src/main.cpp
    src/fas.cpp
    src/util/clsd_interface.cpp
    external/clsd/graph.cc
    external/clsd/dfs.cc
    external/clsd/detect.cc
    external/clsd/cycle.cc
    external/clsd/config.cc)

target_include_directories(BubbleFinder PRIVATE src)

target_link_libraries(BubbleFinder PRIVATE
    sbio sbutil
    ${OGDF_LIB}
    spdlog::spdlog_header_only)


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

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok OUTPUT ipo_err)
if(SB_ENABLE_LTO AND ipo_ok AND NOT SB_ENABLE_SANITIZERS)
    foreach(t BubbleFinder sbio sbutil)
        if(TARGET ${t})
            set_property(TARGET ${t} PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
        endif()
    endforeach()
endif()

foreach(t BubbleFinder sbio sbutil)
    if(TARGET ${t})
        target_compile_options(${t} PRIVATE
            $<$<CONFIG:Release>:
                -O3 -DNDEBUG -flto -pthread
                $<$<BOOL:${SB_NATIVE_OPTIMIZATIONS}>:-march=native>
            >
            $<$<CONFIG:RelWithDebInfo>:-O2 -g -DENABLE_PROFILING -fno-omit-frame-pointer>
            $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer>)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${t} PRIVATE -Wall -Wextra -Wshadow -pedantic)
        endif()
    endif()
endforeach()


message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OGDF target: ${OGDF_LIB}")
message(STATUS "Sanitizers: ${SB_ENABLE_SANITIZERS}")
message(STATUS "Enable LTO: ${SB_ENABLE_LTO} (ipo_ok=${ipo_ok})")
message(STATUS "Native optimizations (-march=native): ${SB_NATIVE_OPTIMIZATIONS}")


add_executable(snarls_bf src/snarls_bruteforce.cpp)
add_executable(superbubbles_bf src/superbubbles_bruteforce.cpp)
add_executable(ultrabubbles_bf src/ultrabubbles_bruteforce.cpp)