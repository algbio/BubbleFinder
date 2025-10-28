#pragma once
#include <ogdf/basic/Graph.h>
#include <ogdf/basic/NodeArray.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum class EdgePartType { PLUS, MINUS, NONE };

struct Context
{
    enum LogLevel { LOG_ERROR = 0, LOG_WARN, LOG_INFO, LOG_DEBUG };
    enum BubbleType { SUPERBUBBLE, SNARL };

    struct PairHash {
        std::size_t operator()(const std::pair<int, int>& p) const {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };

    std::string graphPath = "";
    std::string outputPath = "";
    bool gfaInput = false;
    bool doubleGraph = false;
    LogLevel logLevel = LogLevel::LOG_INFO;
    bool timingEnabled = true;

    ogdf::Graph                       G;

    ogdf::NodeArray<int>              inDeg;
    ogdf::NodeArray<int>              outDeg;
    ogdf::NodeArray<bool>             isEntry;
    ogdf::NodeArray<bool>             isExit;

    unsigned threads = 1;

    // int type = 0;

    BubbleType bubbleType = SUPERBUBBLE;

    ogdf::EdgeArray<std::pair<EdgePartType, EdgePartType>> _edge2types; // for bidirected graphs/snarls
    ogdf::EdgeArray<std::pair<int, int>> _edge2cnt; // for bidirected graphs/snarls

    ogdf::NodeArray<bool> _goodCutVertices; // for bidirected graphs/snarls


    std::unordered_set<std::pair<int,int>, PairHash> _edges;

    std::unordered_map<std::string, ogdf::node> name2node;
    std::unordered_map<ogdf::node,std::string>  node2name;
    std::vector<std::pair<ogdf::node,ogdf::node>> superbubbles;

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
};




Context& ctx();
