#include "context.hpp"

/* Construct the NodeArrays with the graph reference. */
Context::Context()
    : inDeg   (G, 0)
    , outDeg  (G, 0)
    , isEntry (G, false)
    , isExit  (G, false)
    , graphPath ("")
    , outputPath("")
    , gfaInput(false)
    , logLevel(Context::LOG_INFO)
    , timingEnabled(true)
{}




/* “Magic static” – initialised once, thread-safe since C++11.   */
Context& ctx()
{
    static Context instance;         // C++11: guaranteed thread-safe
    return instance;
}
