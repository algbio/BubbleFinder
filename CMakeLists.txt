cmake_minimum_required(VERSION 3.21)
project(sbSPQRnew LANGUAGES CXX C VERSION 1.0)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SB_NATIVE_OPTIMIZATIONS "Use -march=native in Release builds" ON)
option(SB_ENABLE_SANITIZERS "Enable ASan/UBSan (Debug)" OFF)
option(SB_ENABLE_LTO "Enable Link Time Optimization" ON)


set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)


include(FetchContent)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.15.3)
FetchContent_MakeAvailable(spdlog)


# Auto-fetch external submodules if missing (GBZ + spqr-rust)
set(_needs_fetch FALSE)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/gbz/gbwtgraph/include")
    set(_needs_fetch TRUE)
endif()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/spqr-rust/Cargo.toml")
    set(_needs_fetch TRUE)
endif()

if(_needs_fetch)
    find_package(Git REQUIRED)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.gitmodules")
        message(STATUS "Initializing git submodules...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE _sm_result)
        if(NOT _sm_result EQUAL 0)
            message(FATAL_ERROR "git submodule update failed (code ${_sm_result})")
        endif()
    endif()

    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/spqr-rust/Cargo.toml")
        message(STATUS "Cloning spqr-rust...")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone https://github.com/algbio/spqr-rust.git spqr-rust
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external
            RESULT_VARIABLE _clone_result)
        if(NOT _clone_result EQUAL 0)
            message(FATAL_ERROR "git clone spqr-rust failed (code ${_clone_result})")
        endif()
    endif()

    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/gbz/gbwtgraph/include")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/gbz")
        foreach(repo
            "https://github.com/vgteam/sdsl-lite.git"
            "https://github.com/vgteam/libhandlegraph.git"
            "https://github.com/jltsiren/gbwt.git"
            "https://github.com/jltsiren/gbwtgraph.git")
            get_filename_component(_name ${repo} NAME_WE)
            if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/gbz/${_name}")
                message(STATUS "Cloning ${_name}...")
                execute_process(
                    COMMAND ${GIT_EXECUTABLE} clone ${repo}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/gbz
                    RESULT_VARIABLE _clone_result)
                if(NOT _clone_result EQUAL 0)
                    message(FATAL_ERROR "git clone ${repo} failed (code ${_clone_result})")
                endif()
            endif()
        endforeach()
    endif()
endif()


find_package(PkgConfig QUIET)
set(ZSTD_TARGET "")

if(PkgConfig_FOUND)
    pkg_check_modules(ZSTD_SYS QUIET IMPORTED_TARGET libzstd)
endif()

if(ZSTD_SYS_FOUND)
    message(STATUS "zstd: using system (${ZSTD_SYS_VERSION})")
    set(ZSTD_TARGET PkgConfig::ZSTD_SYS)
else()
    message(STATUS "zstd: not found on system, building from source")
    set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_SHARED   OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_STATIC   ON  CACHE BOOL "" FORCE)
    set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(zstd
        GIT_REPOSITORY https://github.com/facebook/zstd.git
        GIT_TAG        v1.5.6
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  build/cmake)
    FetchContent_MakeAvailable(zstd)

    if(TARGET libzstd_static)
        set(ZSTD_TARGET libzstd_static)
        # Ensure <zstd.h> is findable by downstream targets
        get_target_property(_zstd_inc libzstd_static INTERFACE_INCLUDE_DIRECTORIES)
        if(NOT _zstd_inc)
            target_include_directories(libzstd_static INTERFACE
                ${zstd_SOURCE_DIR}/lib)
        endif()
    else()
        message(FATAL_ERROR "FetchContent built zstd but libzstd_static target not found")
    endif()
endif()


set(CRYPTO_TARGET "")

if(PkgConfig_FOUND)
    pkg_check_modules(CRYPTO_SYS QUIET IMPORTED_TARGET libcrypto)
endif()

if(CRYPTO_SYS_FOUND)
    message(STATUS "libcrypto: using system via pkg-config (${CRYPTO_SYS_VERSION})")
    set(CRYPTO_TARGET PkgConfig::CRYPTO_SYS)
else()
    find_package(OpenSSL QUIET)
    if(OpenSSL_FOUND AND TARGET OpenSSL::Crypto)
        message(STATUS "libcrypto: using system via FindOpenSSL (${OPENSSL_VERSION})")
        set(CRYPTO_TARGET OpenSSL::Crypto)
    else()
        message(FATAL_ERROR
            "libcrypto (OpenSSL) not found. Install it:\n"
            "  conda install -c conda-forge openssl\n"
            "  -or- sudo apt install libssl-dev\n"
            "  -or- sudo yum install openssl-devel\n"
            "OpenSSL is too large to auto-fetch; it is typically pre-installed on all Linux systems.")
    endif()
endif()


find_package(OpenMP QUIET)


add_library(sbutil OBJECT
    src/util/context.cpp
    src/util/logger.cpp)
target_include_directories(sbutil
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/util
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/external/spqr-rust/include)
target_link_libraries(sbutil PUBLIC spdlog::spdlog_header_only)

add_library(sbio OBJECT
    src/io/graph_io.cpp)
target_include_directories(sbio
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/io)
target_link_libraries(sbio PUBLIC sbutil)


# sdsl lite
set(WITH_GOOGLETEST OFF CACHE BOOL "" FORCE)
set(GENERATE_DOC    OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
add_subdirectory(external/gbz/sdsl-lite EXCLUDE_FROM_ALL)

# libhandlegraph 
add_subdirectory(external/gbz/libhandlegraph EXCLUDE_FROM_ALL)

if(TARGET handlegraph_static)
    set(HANDLEGRAPH_LIB handlegraph_static)
elseif(TARGET handlegraph)
    set(HANDLEGRAPH_LIB handlegraph)
elseif(TARGET handlegraph_shared)
    set(HANDLEGRAPH_LIB handlegraph_shared)
else()
    message(FATAL_ERROR "No handlegraph target found")
endif()
message(STATUS "Using handlegraph target: ${HANDLEGRAPH_LIB}")

# gbwt
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
target_link_libraries(gbwt PUBLIC sdsl ${ZSTD_TARGET})
target_compile_features(gbwt PUBLIC cxx_std_17)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gbwt PUBLIC OpenMP::OpenMP_CXX)
endif()

# gbwtgraph
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
    gbwt ${HANDLEGRAPH_LIB} sdsl
    ${ZSTD_TARGET} ${CRYPTO_TARGET})
target_compile_features(gbwtgraph PUBLIC cxx_std_17)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gbwtgraph PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(sbio PUBLIC gbwtgraph)

message(STATUS "GBZ support: ON (always enabled)")


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

find_program(CARGO cargo REQUIRED)
set(SPQR_RUST_DIR "${CMAKE_SOURCE_DIR}/external/spqr-rust")
set(SPQR_RUST_LIB "${SPQR_RUST_DIR}/target/release/libspqr_rust.a")

add_custom_command(
    OUTPUT ${SPQR_RUST_LIB}
    COMMAND ${CARGO} build --release --manifest-path ${SPQR_RUST_DIR}/Cargo.toml
    WORKING_DIRECTORY ${SPQR_RUST_DIR}
    DEPENDS ${SPQR_RUST_DIR}/Cargo.toml
            ${SPQR_RUST_DIR}/src/lib.rs
            ${SPQR_RUST_DIR}/src/biconnected.rs
            ${SPQR_RUST_DIR}/src/connected.rs
            ${SPQR_RUST_DIR}/src/ffi.rs
            ${SPQR_RUST_DIR}/src/spqr_format.rs
            ${SPQR_RUST_DIR}/src/verify.rs
    COMMENT "Building spqr-rust (cargo --release)"
    VERBATIM)

add_custom_target(spqr_rust_build DEPENDS ${SPQR_RUST_LIB})
add_dependencies(BubbleFinder spqr_rust_build)

target_link_libraries(BubbleFinder PRIVATE
    sbio sbutil
    ${SPQR_RUST_LIB}
    spdlog::spdlog_header_only)

# Link OpenMP if available; the snarls solver uses #pragma omp for intra-block
# parallelism (taskloops). When OpenMP is not found, the pragmas degrade to
# no-ops and the code falls back to fully sequential intra-block execution.
if(OpenMP_CXX_FOUND)
    target_link_libraries(BubbleFinder PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(BubbleFinder PRIVATE BF_HAVE_OPENMP=1)
    message(STATUS "BubbleFinder: OpenMP enabled (intra-block parallelism active)")
else()
    target_compile_definitions(BubbleFinder PRIVATE BF_HAVE_OPENMP=0)
    message(STATUS "BubbleFinder: OpenMP not found (intra-block parallelism disabled)")
endif()

set_target_properties(BubbleFinder PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib"
    BUILD_WITH_INSTALL_RPATH FALSE)

option(BUBBLEFINDER_INSTRUMENT
    "Enable BubbleFinder profiling counters and timers (zero overhead when OFF)"
    OFF)
if(BUBBLEFINDER_INSTRUMENT)
    target_compile_definitions(BubbleFinder PRIVATE BUBBLEFINDER_INSTRUMENT)
endif()

# Link against libdl (libspqr_rust.a indirectly calls dlsym).
target_link_libraries(BubbleFinder PRIVATE ${CMAKE_DL_LIBS})


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
message(STATUS "spqr-rust lib: ${SPQR_RUST_LIB}")
message(STATUS "zstd target: ${ZSTD_TARGET}")
message(STATUS "crypto target: ${CRYPTO_TARGET}")
message(STATUS "handlegraph target: ${HANDLEGRAPH_LIB}")
message(STATUS "Sanitizers: ${SB_ENABLE_SANITIZERS}")
message(STATUS "Enable LTO: ${SB_ENABLE_LTO} (ipo_ok=${ipo_ok})")
message(STATUS "Native optimizations (-march=native): ${SB_NATIVE_OPTIMIZATIONS}")

message(STATUS "BubbleFinder instrumentation: ${BUBBLEFINDER_INSTRUMENT}")


add_executable(snarls_bf src/snarls_bruteforce.cpp)
add_executable(superbubbles_bf src/superbubbles_bruteforce.cpp)
add_executable(ultrabubbles_bf src/ultrabubbles_bruteforce.cpp)