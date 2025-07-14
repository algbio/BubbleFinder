add_library(sbio STATIC graph_io.cpp)

target_include_directories(sbio
    PUBLIC  ${PROJECT_SOURCE_DIR}/src          # so callers can #include "io/…"
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})       # src/io for its own .cpp

target_link_libraries(sbio
    PUBLIC sbutil OGDF)        # sbio still needs ctx(), logger, OGDF
