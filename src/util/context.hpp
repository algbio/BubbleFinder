#pragma once

#include "util/ogdf_all.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <utility>
#include <cstddef>

enum class EdgePartType { PLUS, MINUS, NONE };

struct Context
{
    enum LogLevel  { LOG_ERROR = 0, LOG_WARN, LOG_INFO, LOG_DEBUG };
    enum BubbleType { SUPERBUBBLE, SNARL, ULTRABUBBLE };

    // Input file format (graph representation)
    enum class InputFormat {
        Auto,        // autodetect from extension
        Gfa,         // GFA (bidirected interpretation)
        GfaDirected, // GFA interpreted as directed graph
        Graph        // internal .graph format (directed)
    };

    // Input file compression (autodetected from file suffix)
    enum class Compression {
        None,
        Gzip,
        Bzip2,
        Xz
    };

    struct PairHash {
        std::size_t operator()(const std::pair<int, int>& p) const {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };

    ogdf::Graph           G;
    ogdf::NodeArray<int>  inDeg;
    ogdf::NodeArray<int>  outDeg;
    ogdf::NodeArray<bool> isEntry;
    ogdf::NodeArray<bool> isExit;

    std::string graphPath   = "";
    std::string outputPath  = "";
    std::string ultrabubbleTreeOutputPath = "";
    bool        gfaInput    = false;  // kept for backward compatibility
    bool        doubleGraph = false;
    LogLevel    logLevel    = LOG_INFO;
    bool        timingEnabled = true;
    unsigned    threads     = 1;
    std::size_t stackSize   = 1ULL * 1024ULL * 1024ULL * 1024ULL; 


    
    std::vector<std::pair<std::string,std::string>> ultrabubbleIncidences;
    std::vector<std::string> gfaSegmentIds;
    std::vector<std::string> gfaLinkLines;



    

    BubbleType  bubbleType  = SUPERBUBBLE;
    bool        directedSuperbubbles = false;
    InputFormat inputFormat = InputFormat::Auto;
    Compression compression = Compression::None;

    ogdf::EdgeArray<std::pair<EdgePartType, EdgePartType>> _edge2types; 
    ogdf::EdgeArray<std::pair<int, int>>                   _edge2cnt;   
    ogdf::NodeArray<bool>                                  _goodCutVertices;

    std::unordered_set<std::pair<int,int>, PairHash> _edges;

    std::unordered_map<std::string, ogdf::node> name2node;
    std::unordered_map<ogdf::node, std::string> node2name;
    std::vector<std::pair<ogdf::node, ogdf::node>> superbubbles;

    struct VectorStringHash {
        std::size_t operator()(const std::vector<std::string>& v) const {
            std::size_t h = 0;
            std::hash<std::string> hasher;
            for (const auto& s : v) {
                h ^= hasher(s) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        }
    };

    struct VectorStringEqual {
        bool operator()(const std::vector<std::string>& a,
                        const std::vector<std::string>& b) const {
            return a == b;
        }
    };

    std::unordered_set<std::vector<std::string>, VectorStringHash, VectorStringEqual> snarls;

    Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
};

Context& ctx();