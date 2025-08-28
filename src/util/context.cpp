#include "context.hpp"

Context::Context()
    : inDeg   (G, 0)
    , outDeg  (G, 0)
    , isEntry (G, false)
    , isExit  (G, false)
    , graphPath ("")
    , outputPath("")
    , gfaInput(false)
    , doubleGraph(false)
    , logLevel(Context::LOG_INFO)
    , timingEnabled(true)
{}


Context& ctx() {
    static Context instance;
    return instance;
}
