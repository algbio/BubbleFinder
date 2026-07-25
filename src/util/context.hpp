#pragma once

#include "util/spqr_rust_all.hpp"
#include "util/spqr_index.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <utility>
#include <cstddef>
#include <cstdint>
#include <limits>

enum class EdgePartType : uint8_t
{
    PLUS,
    MINUS,
    NONE
};

static_assert(sizeof(EdgePartType) == sizeof(uint8_t),
              "EdgePartType must stay byte-sized");
static_assert(sizeof(std::pair<EdgePartType, EdgePartType>) == 2 * sizeof(uint8_t),
              "edge type pairs must stay compact");

struct UBEdge
{
    uint32_t neighbor;
    uint8_t type_self;
    uint8_t type_neigh;
};

struct InputHaplotypeStep
{
    std::uint32_t node = 0;
    std::uint8_t is_reverse = 0;
};

struct InputHaplotypePath
{
    std::string name;
    std::string sample;
    std::string locus;
    std::uint64_t haplotype = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t phase_block = std::numeric_limits<std::uint64_t>::max();
    std::uint8_t sense = 0;
    std::uint64_t step_begin = 0;
    std::uint64_t step_end = 0;
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
        SPQR_TREE_ONLY,
        SPQR_INDEX
    };

    enum class InputFormat
    {
        Auto,
        Gfa,
        GfaDirected,
        Graph,
        SpqrIndex
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
        MacroDirect
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
    std::string spqrIndexPath = "";
    std::string spqrTreeView = "raw";
    bool spqrTreeViewExplicit = false;
    bool spqrHaplotypes = false;
    std::unique_ptr<spqr_index::IndexedSpqrTree> spqrIndex;
    bool spqrIndexBubbleEdgeTypesChecked = false;
    bool spqrIndexHasCompleteBubbleEdgeTypes = false;
    bool spqrIndexInputLoaded = false;
    std::string spqrIndexInputGraphView = "";
    bool directSpqrInputGraphUsesIndexNames = false;
    bool directSpqrInputGraphEdgesMaterialized = true;
    std::vector<std::uint8_t> directSpqrGraphEdgeTypePairs;
    std::vector<spqr_index::GraphComponentRecord> directSpqrGraphComponents;
    std::vector<std::uint32_t> directSpqrGraphComponentNodes;
    std::vector<std::uint32_t> directSpqrGraphComponentEdges;
    bool directSpqrGraphComponentNodesIdentity = false;
    bool directSpqrGraphComponentEdgesIdentity = false;

    std::vector<bool> ubIsTip;

    bool gfaInput = false; // legacy flag
    bool doubleGraph = false;
    bool doubledUltrabubbles = false;

    LogLevel logLevel = LOG_INFO;
    bool timingEnabled = true;
    unsigned threads = 1;
    bool threadsExplicit = false;

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
    std::vector<InputHaplotypePath> inputHaplotypePaths;
    std::vector<InputHaplotypeStep> inputHaplotypeSteps;

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
    std::vector<std::vector<std::uint32_t>> fastSnarlCliques;

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

inline EdgePartType storedEndpointTypeToPart(std::uint8_t value)
{
    if (value == 1u) return EdgePartType::PLUS;
    if (value == 2u) return EdgePartType::MINUS;
    return EdgePartType::NONE;
}

inline std::uint8_t edgePartTypeToStoredEndpointType(EdgePartType type)
{
    if (type == EdgePartType::PLUS) return 1u;
    if (type == EdgePartType::MINUS) return 2u;
    return 0u;
}

inline std::uint8_t packEdgePartTypes(EdgePartType a, EdgePartType b)
{
    return (static_cast<std::uint8_t>(a) << 2u) |
           static_cast<std::uint8_t>(b);
}

inline std::pair<EdgePartType, EdgePartType>
storedEndpointTypesToParts(std::uint8_t packed)
{
    return {
        storedEndpointTypeToPart(static_cast<std::uint8_t>((packed >> 4u) & 0x0fu)),
        storedEndpointTypeToPart(static_cast<std::uint8_t>(packed & 0x0fu))
    };
}

inline std::pair<EdgePartType, EdgePartType>
edgePartTypes(const Context &C, spqr_compat::edge e)
{
    if (!C.directSpqrGraphEdgeTypePairs.empty() &&
        static_cast<std::size_t>(e.idx) < C.directSpqrGraphEdgeTypePairs.size()) {
        return storedEndpointTypesToParts(C.directSpqrGraphEdgeTypePairs[e.idx]);
    }
    return C._edge2types(e);
}

Context &ctx();
