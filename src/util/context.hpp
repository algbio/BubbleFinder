#pragma once

#include "util/ogdf_all.hpp"

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
struct UBEdge {
    uint32_t neighbor; // index of the neighbor node
    uint8_t  type_self; // EdgePartType at this node  (0=PLUS, 1=MINUS)
    uint8_t  type_neigh; // EdgePartType at the neighbor (0 = PLUS, 1=MINUS)
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

    struct PairHash
    {
        std::size_t operator()(const std::pair<int, int> &p) const
        {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };

    ogdf::Graph G;
    ogdf::NodeArray<int> inDeg;
    ogdf::NodeArray<int> outDeg;
    ogdf::NodeArray<bool> isEntry;
    ogdf::NodeArray<bool> isExit;

    std::string graphPath = "";
    std::string outputPath = "";

    std::string ultrabubbleTreeOutputPath = "";

    bool gfaInput = false; // kept for backward compatibility
    bool doubleGraph = false;
    LogLevel logLevel = LOG_INFO;
    bool timingEnabled = true;
    unsigned threads = 1;
    std::size_t stackSize = 1ULL * 1024ULL * 1024ULL * 1024ULL;


    std::vector<std::pair<std::string, std::string>> ultrabubbleIncidences;
    std::vector<std::string> gfaSegmentIds;
    std::vector<std::string> gfaLinkLines;

    BubbleType bubbleType = SUPERBUBBLE;
    bool directedSuperbubbles = false;
    InputFormat inputFormat = InputFormat::Auto;
    Compression compression = Compression::None;

    bool clsdTrees = false;
    std::string clsdTreesPath;

    bool includeTrivial = false;

    ogdf::EdgeArray<std::pair<EdgePartType, EdgePartType>> _edge2types;
    ogdf::EdgeArray<std::pair<int, int>> _edge2cnt;
    ogdf::NodeArray<bool> _goodCutVertices;

    std::unordered_set<std::pair<int, int>, PairHash> _edges;

    std::unordered_map<std::string, ogdf::node> name2node;
    std::unordered_map<ogdf::node, std::string> node2name;
    std::vector<std::pair<ogdf::node, ogdf::node>> superbubbles;

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

    std::vector<ogdf::node> nodeByGlobalId;

    std::vector<std::pair<std::uint32_t, std::uint32_t>> ultrabubbleIncPacked;

    uint32_t ubNumNodes = 0;
    std::vector<std::string> ubNodeNames;
    std::vector<uint32_t> ubOffset;   
    std::vector<UBEdge> ubEdges;    
    std::vector<std::string> ubClsdText;

    inline const UBEdge* adjBegin(uint32_t v) const { return ubEdges.data() + ubOffset[v]; }
    inline const UBEdge* adjEnd(uint32_t v)   const { return ubEdges.data() + ubOffset[v+1]; }
    inline uint32_t adjDeg(uint32_t v)   const { return ubOffset[v+1] - ubOffset[v]; }

    Context();
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
};

Context &ctx();