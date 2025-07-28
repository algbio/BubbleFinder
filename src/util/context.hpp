#pragma once
#include <ogdf/basic/Graph.h>
#include <ogdf/basic/NodeArray.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Context
{
    enum LogLevel { LOG_ERROR = 0, LOG_WARN, LOG_INFO, LOG_DEBUG };

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

    std::unordered_set<std::pair<int,int>, PairHash> _edges;

    std::unordered_map<std::string, ogdf::node> name2node;
    std::unordered_map<ogdf::node,std::string>  node2name;
    std::vector<std::pair<ogdf::node,ogdf::node>> superbubbles;

    Context();
    Context(const Context&) = delete;
};




Context& ctx();
