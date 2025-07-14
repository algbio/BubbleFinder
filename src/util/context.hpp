#pragma once
#include <ogdf/basic/Graph.h>
#include <ogdf/basic/NodeArray.h>
#include <unordered_map>
#include <vector>

struct Context
{
    enum LogLevel { LOG_ERROR = 0, LOG_WARN, LOG_INFO, LOG_DEBUG };


    std::string graphPath = "";
    std::string outputPath = "";
    bool gfaInput = false;
    LogLevel logLevel = LogLevel::LOG_INFO;
    bool timingEnabled = true;

    ogdf::Graph                       G;

    // arrays that must be constructed **after** G
    ogdf::NodeArray<int>              inDeg;
    ogdf::NodeArray<int>              outDeg;
    ogdf::NodeArray<bool>             isEntry;
    ogdf::NodeArray<bool>             isExit;

    std::unordered_map<std::string, ogdf::node> name2node;
    std::unordered_map<ogdf::node,std::string>  node2name;
    std::vector<std::pair<ogdf::node,ogdf::node>> superbubbles;

    Context();                       // ctor defined in .cpp
    Context(const Context&) = delete;
};




Context& ctx();
