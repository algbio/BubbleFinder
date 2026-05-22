add_library(sbutil STATIC context.cpp logger.cpp)

# expose BOTH the util folder *and* the project src root
target_include_directories(sbutil
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src      # <-- ✔ lets others write "util/…"
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})   # src/util  (for its own compilation)

target_link_libraries(sbutil PUBLIC OGDF PUBLIC spdlog::spdlog_header_only)
