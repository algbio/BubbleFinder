#include "context.hpp"

Context::Context()
    : inDeg   (G, 0)
    , outDeg  (G, 0)
    , isEntry (G, false)
    , isExit  (G, false)
    , graphPath("")
    , outputPath("")
    , gfaInput(false)
    , doubleGraph(false)
    , logLevel(Context::LOG_WARN) 
    , timingEnabled(true)
    , threads(1)
    , stackSize(1ULL * 1024ULL * 1024ULL * 1024ULL)
    , bubbleType(Context::SUPERBUBBLE)
    , directedSuperbubbles(false)
    , inputFormat(InputFormat::Auto)
    , compression(Compression::None)
    , _edge2types(G, std::make_pair(EdgePartType::NONE, EdgePartType::NONE))
    , _edge2cnt(G, std::make_pair(0,0))
    , _goodCutVertices(G, false)
    , stats()
{
}

Context& ctx() {
    static Context instance;
    return instance;
}