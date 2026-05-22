#pragma once

#include "util/spqr_rust_all.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <utility>
#include <cstddef>
#include <cstdint>

enum class EdgePartType
{
    PLUS,
    MINUS,
    NONE
};

struct UBEdge
{
    uint32_t neighbor;
    uint8_t type_self;
    uint8_t type_neigh;
};

struct Context
{
    enum LogLevel
    {
        LOG_ERROR = 0,
        LOG_WARN,
        LOG_INFO,
        LOG_DEBUG
    };

    enum BubbleType
    {
        SUPERBUBBLE,
        SNARL,
        ULTRABUBBLE,
        SPQR_TREE_ONLY
    };

    enum class InputFormat
    {
        Auto,
        Gfa,
        GfaDirected,
        Graph
    };

    enum class Compression
    {
        None,
        Gzip,
        Bzip2,
        Xz
    };

    enum class SpCompressMode
    {
        Off,
        On,
        Instrument,
        MacroDirectDebug
    };

    struct PairHash
    {
        std::size_t operator()(const std::pair<int, int> &p) const
        {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };

    spqr_compat::Graph G;
    spqr_compat::NodeArray<int> inDeg;
    spqr_compat::NodeArray<int> outDeg;
    spqr_compat::NodeArray<bool> isEntry;
    spqr_compat::NodeArray<bool> isExit;

    std::string graphPath = "";
    std::string outputPath = "";
    std::string ultrabubbleTreeOutputPath = "";

    std::vector<bool> ubIsTip;

    bool gfaInput = false; // legacy flag
    bool doubleGraph = false;
    bool doubledUltrabubbles = false;

    LogLevel logLevel = LOG_INFO;
    bool timingEnabled = true;
    unsigned threads = 1;

    std::size_t stackSize = 1ULL * 1024ULL * 1024ULL * 1024ULL;

    std::vector<std::pair<std::string, std::string>> ultrabubbleIncidences;
    std::vector<std::string> gfaSegmentIds;
    std::vector<std::string> gfaLinkLines;

    BubbleType bubbleType = SUPERBUBBLE;
    bool directedSuperbubbles = true;

    InputFormat inputFormat = InputFormat::Auto;
    Compression compression = Compression::None;

    bool clsdTrees = false;
    std::string clsdTreesPath;

    bool includeTrivial = false;
    bool compactOutputChains = false;
    bool spqrWeakUltrabubbles = false;
    bool weakSuperbubbles = false;

    SpCompressMode spCompressMode = SpCompressMode::Off;
    std::string spCompressInstrumentCsv = "";

    bool skipCanonicalizeRoot = false;

    spqr_compat::EdgeArray<std::pair<EdgePartType, EdgePartType>> _edge2types;
    spqr_compat::EdgeArray<std::pair<int, int>> _edge2cnt;
    spqr_compat::NodeArray<bool> _goodCutVertices;

    std::unordered_set<std::pair<int, int>, PairHash> _edges;

    std::unordered_map<std::string, spqr_compat::node> name2node;
    std::unordered_map<spqr_compat::node, std::string> node2name;
    std::vector<std::string> nodeNamesByIndex;
    std::vector<std::uint64_t> nodeNumericNamesByIndex;
    std::vector<std::uint8_t> nodeNumericNameValidByIndex;
    std::unordered_map<std::uint32_t, std::string> sparseNodeNamesByIndex;
    std::vector<std::uint8_t> isTrashNodeByIndex;

    std::vector<std::pair<spqr_compat::node, spqr_compat::node>> superbubbles;

    struct VectorStringHash
    {
        std::size_t operator()(const std::vector<std::string> &v) const
        {
            std::size_t h = 0;
            std::hash<std::string> hasher;

            for (const auto &s : v)
            {
                h ^= hasher(s) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }

            return h;
        }
    };

    struct VectorStringEqual
    {
        bool operator()(const std::vector<std::string> &a,
                        const std::vector<std::string> &b) const
        {
            return a == b;
        }
    };

    std::unordered_set<std::vector<std::string>, VectorStringHash, VectorStringEqual> snarls;

    bool fastSnarlPairsEnabled = false;
    std::vector<std::uint64_t> fastSnarlPairs;
    std::vector<std::vector<std::uint64_t>> fastSnarlCliques;

    std::vector<spqr_compat::node> nodeByGlobalId;

    std::vector<std::pair<std::uint32_t, std::uint32_t>> ultrabubbleIncPacked;

    uint32_t ubNumNodes = 0;

    std::vector<std::string> ubNodeNames;
    std::vector<uint32_t> ubOffset;
    std::vector<UBEdge> ubEdges;
    std::vector<std::string> ubClsdText;

    inline const UBEdge* adjBegin(uint32_t v) const { return ubEdges.data() + ubOffset[v]; }
    inline const UBEdge* adjEnd(uint32_t v) const { return ubEdges.data() + ubOffset[v + 1]; }
    inline uint32_t adjDeg(uint32_t v) const { return ubOffset[v + 1] - ubOffset[v]; }

    Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
};

Context &ctx();
