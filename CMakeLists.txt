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
    # réduire ce que OGDF construit
    set(OGDF_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(OGDF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(ogdf)  # la cible s'appelle souvent "ogdf"
endif()

# Si un OGDF installé existe, on le préfère
find_package(OGDF CONFIG QUIET)

# Résoudre la cible OGDF effective
if(TARGET OGDF::OGDF)
    set(OGDF_LIB OGDF::OGDF)       # package installé
elseif(TARGET ogdf)
    set(OGDF_LIB ogdf)             # FetchContent a fourni "ogdf"
elseif(TARGET OGDF)
    set(OGDF_LIB OGDF)             # autre alias possible
else()
    message(FATAL_ERROR "No OGDF library target available. Build with SB_BUNDLE_OGDF=ON or install OGDF and pass its CMake package path.")
endif()

# 4. Subprojects (vos libs)
add_subdirectory(src/util)   # ⇒ target sbutil
add_subdirectory(src/io)     # ⇒ target sbio

# 5. main executable
add_executable(BubbleFinder src/main.cpp src/fas.cpp)
target_include_directories(BubbleFinder PRIVATE src)

target_link_libraries(BubbleFinder PRIVATE
    sbio sbutil
    ${OGDF_LIB}
    spdlog::spdlog_header_only
)

# 6. Sanitizers (Address + UB) — activables via option.
# Important: ASan et PIE posent souvent problème -> appliquer -fno-pie/-no-pie sur Linux.
option(SB_ENABLE_SANITIZERS "Enable Address/UndefinedBehavior sanitizers (Debug)" OFF)
if(SB_ENABLE_SANITIZERS)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "SB_ENABLE_SANITIZERS is ON but CMAKE_BUILD_TYPE != Debug — sanitizers are intended for Debug builds")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Sous Linux non-Apple on évite PIE, sinon -no-pie peut être interdit (macOS)
        if(UNIX AND NOT APPLE)
            add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie -g)
            add_link_options   (-fsanitize=address,undefined -no-pie)
        else()
            # macOS et autres : ne pas ajouter -no-pie / -fno-pie qui échouent
            add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
            add_link_options   (-fsanitize=address,undefined)
        endif()
        message(STATUS "Sanitizers enabled: -fsanitize=address,undefined applied")
    else()
        message(WARNING "Sanitizers requested but compiler is not GNU/Clang — skipping sanitizer flags")
    endif()
endif()

# 7. LTO / IPO — n'activer que si supporté et si sanitizers non activés
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

# 8. Options par configuration et warnings (appliqués aux targets existants)
# Définir les options d'optimisation par configuration via generator expressions.
foreach(t BubbleFinder sbio sbutil)
    if(TARGET ${t})
        # optimisation / debug par config
        target_compile_options(${t} PRIVATE
            $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -flto -pthread>
            $<$<CONFIG:RelWithDebInfo>:-O2 -g -DENABLE_PROFILING -fno-omit-frame-pointer>
            $<$<CONFIG:Debug>:-O0 -g -fno-omit-frame-pointer>
        )

        # warnings pour GCC/Clang
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${t} PRIVATE -Wall -Wextra -Wshadow -pedantic)
        endif()
    endif()
endforeach()

# 9. Message d'aide
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OGDF target: ${OGDF_LIB}")
message(STATUS "Sanitizers: ${SB_ENABLE_SANITIZERS}")
message(STATUS "Enable LTO: ${SB_ENABLE_LTO} (ipo_ok=${ipo_ok})")

















# cmake_minimum_required(VERSION 3.20)
# project(sbSPQRnew LANGUAGES CXX)

# # ──────────────────────────────────────────────────────────────────
# # 0. Global compile flags – perf friendly
# #    *  -g                       : full DWARF debug info
# #    *  -fno-omit-frame-pointer : keep frame pointers for perf
# # ──────────────────────────────────────────────────────────────────
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_compile_options(
#     -g
#     -fno-omit-frame-pointer
#     # -fno-inline
# )

# # Release = fast but still symbolised
# # set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2")

# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")
# set(CMAKE_CXX_FLAGS_DEBUG    "-O2 -g -fno-omit-frame-pointer")
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -fno-omit-frame-pointer")


# # ──────────────────────────────────────────────────────────────────
# # 1. Optional sanitiser toggle
# # ──────────────────────────────────────────────────────────────────
# option(ENABLE_SANITIZERS "Build with Address + UB Sanitiser" OFF)
# if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     add_compile_options(
#         -fsanitize=address,undefined
#         -fno-pie                   # ASan shadow-range fix
#     )
#     add_link_options(
#         -fsanitize=address,undefined
#         -no-pie
#     )
# endif()

# # ──────────────────────────────────────────────────────────────────
# # 2. 3rd-party libraries
# # ──────────────────────────────────────────────────────────────────
# include(FetchContent)

# # 2.1 spdlog (header-only)
# FetchContent_Declare(
#     spdlog
#     GIT_REPOSITORY https://github.com/gabime/spdlog.git
#     GIT_TAG        v1.15.3
# )
# FetchContent_MakeAvailable(spdlog)

# # 2.2 OGDF (pre-built – point to your prefix)
# set(OGDF_DIR "/home/politov/OGDF" CACHE PATH "OGDF install prefix")
# find_package(OGDF CONFIG REQUIRED PATHS "${OGDF_DIR}/lib/cmake")

# # 2.3 OpenMP
# find_package(OpenMP REQUIRED)

# # ──────────────────────────────────────────────────────────────────
# # 3. Project libraries
# # ──────────────────────────────────────────────────────────────────
# add_subdirectory(src/util)   # → libsbutil
# add_subdirectory(src/io)     # → libsbio

# # ──────────────────────────────────────────────────────────────────
# # 4. Main executable
# # ──────────────────────────────────────────────────────────────────
# add_executable(BubbleFinder src/main.cpp)
# target_include_directories(BubbleFinder PRIVATE src)
# target_compile_options(BubbleFinder PRIVATE -O2  -g -fno-omit-frame-pointer)


# target_link_libraries(BubbleFinder PRIVATE
#     sbio
#     sbutil
#     OGDF
#     spdlog::spdlog_header_only
#     OpenMP::OpenMP_CXX
# )

# # ──────────────────────────────────────────────────────────────────
# # 5. Warnings & optimisation for *this* target only
# # ──────────────────────────────────────────────────────────────────
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     target_compile_options(BubbleFinder PRIVATE
#         -Wall -Wextra -Wshadow -pedantic
#     )
# endif()

	                        
	           





















	                        # cmake_minimum_required(VERSION 3.20)
	                        # project(sbSPQRnew LANGUAGES CXX)

	                        # # ------------------------------------------------------------------#
	                        # # Global options                                                    #
	                        # # ------------------------------------------------------------------#
	                        # set(CMAKE_CXX_STANDARD 20)
	                        # set(CMAKE_CXX_STANDARD_REQUIRED ON)

	                        # option(ENABLE_SANITIZERS
	                        #     "Build with AddressSanitizer + UndefinedBehaviourSanitizer" OFF)




	                        # # ------------------------------------------------------------------#
	                        # # Sanitizer flags (GCC / Clang)                                     #
	                        # # ------------------------------------------------------------------#
	                        # if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	                        #     add_compile_options(
	                        #         -fsanitize=address
	                        #         -fsanitize=undefined
	                        #         -fno-omit-frame-pointer    # keep full back-traces
	                        #         -fno-pie                   # avoid ASan shadow-range clash
	                        #         -g                         # debug symbols
	                        #     )
	                        #     add_link_options(
	                        #         -fsanitize=address
	                        #         -fsanitize=undefined
	                        #         -no-pie                    # link a non-PIE executable
	                        #     )
	                        #     message(STATUS "Building with AddressSanitizer + UBSan (no-PIE)")
	                        # endif()

	                        # # ------------------------------------------------------------------#
	                        # # 3rd-party: spdlog (header-only)                                   #
	                        # # ------------------------------------------------------------------#
	                        # include(FetchContent)
	                        # FetchContent_Declare(
	                        # spdlog
	                        # GIT_REPOSITORY https://github.com/gabime/spdlog.git
	                        # GIT_TAG        v1.15.3
	                        # )
	                        # FetchContent_MakeAvailable(spdlog)

	                        # # ------------------------------------------------------------------#
	                        # # 3rd-party: OGDF (built/installed elsewhere)                       #
	                        # # ------------------------------------------------------------------#
	                        # set(OGDF_DIR "/home/politov/OGDF" CACHE PATH "OGDF install prefix")
	                        # find_package(OGDF CONFIG REQUIRED
	                        #             PATHS "${OGDF_DIR}/lib/cmake")

	                        # find_package(OpenMP REQUIRED)


	                        # # ------------------------------------------------------------------#
	                        # # Project subdirectories (static libs)                              #
	                        # # ------------------------------------------------------------------#
	                        # add_subdirectory(src/util)   # → libsbutil
	                        # add_subdirectory(src/io)     # → libsbio

	                        # # ------------------------------------------------------------------#
	                        # # Main executable                                                   #
	                        # # ------------------------------------------------------------------#
	                        # add_executable(BubbleFinder src/main.cpp)
	                        # target_include_directories(BubbleFinder PRIVATE src)   # #include "util/…"
	                        # target_link_libraries(BubbleFinder PRIVATE
	                        #     sbio
	                        #     sbutil
	                        #     OGDF
	                        #     spdlog::spdlog_header_only
	                        #     OpenMP::OpenMP_CXX
	                        # )

	                        # # target_compile_options(BubbleFinder PRIVATE -fopenmp)
	                        # # target_link_libraries(BubbleFinder PRIVATE -fopenmp)


	                        # # ------------------------------------------------------------------#
	                        # # Nice warnings                                                     #
	                        # # ------------------------------------------------------------------#
	                        # if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	                        #     # target_compile_options(BubbleFinder PRIVATE -O3 -Wall -Wextra -Wshadow -pedantic -fopenmp -g -march=native -flto -fno-omit-frame-pointer)
	                        #     target_compile_options(BubbleFinder PRIVATE -O2 -g -fno-omit-frame-pointer -DNDEBUG)

	                        # endif()
