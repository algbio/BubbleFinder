#pragma once
#include "util/ogdf_all.hpp"
#include "util/context.hpp"
#include <memory>

// enum EdgePartType { PLUS, MINUS, NONE };


// class BidirectionalGraph {
// private:
//     ogdf::Graph _G;
//     ogdf::EdgeArray<std::pair<EdgePartType, EdgePartType>> _edge2types;
//     std::unique_ptr<ogdf::BCTree> _bctree;
//     ogdf::NodeArray<bool> _goodCutVertices;

//     std::unordered_map<std::string, ogdf::node> name2node;
//     std::unordered_map<ogdf::node,std::string>  node2name;
// public:
//     BidirectionalGraph();
//     ogdf::node addNode(std::string name);
//     ogdf::edge addEdge(ogdf::node u, ogdf::node v, EdgePartType typeU, EdgePartType typeV);
//     std::vector<std::pair<ogdf::node, EdgePartType>> getAdjNodes(ogdf::node from, EdgePartType type);
//     void findGoodCutVertices();
//     bool isGoodCutVertex(ogdf::node v);
// };


// class Graph {
// private:
//     ogdf::Graph _G;
//     ogdf::EdgeArray<std::pair<EdgePartType, EdgePartType>> _edge2types;
//     std::unique_ptr<ogdf::BCTree> _bctree;
//     ogdf::NodeArray<bool> _goodCutVertices;

//     std::unordered_map<std::string, ogdf::node> name2node;
//     std::unordered_map<ogdf::node,std::string>  node2name;

//     bool _isBidirected = false;
// public:
//     Graph(bool isBidirected=false) {
//         _G = ogdf::Graph();
//         _edge2types = ogdf::EdgeArray<std::pair<EdgePartType, EdgePartType>>(_G);
//         _bctree = nullptr;
//         _goodCutVertices = ogdf::NodeArray<bool>(_G, false);
//         _isBidirected = isBidirected;
//     }


//     ogdf::node addNode(std::string name) {
//         auto it = name2node.find(name);
//         if (it != name2node.end()) {
//             return it->second;
//         }
//         auto v = _G.newNode();
//         name2node[name] = v;
//         node2name[v] = name;
//         return v;
//     }


//     ogdf::edge addEdge(const std::string& u_name, const std::string& v_name) {
//         auto u = addNode(u_name);
//         auto v = addNode(v_name);
//         auto e = _G.newEdge(u, v);
//         _edge2types[e] = {NONE, NONE};
//         return e;
//     }

//     ogdf::edge addEdge(const std::string& u_name, const std::string& v_name, EdgePartType type1, EdgePartType type2) {
//         auto u = addNode(u_name);
//         auto v = addNode(v_name);
//         auto e = _G.newEdge(u, v);
//         _edge2types[e] = {type1, type2};
//         return e;
//     }




//     std::vector<std::pair<ogdf::node, EdgePartType>> getAdjNodes(ogdf::node from, EdgePartType type) {
//         std::vector<std::pair<ogdf::node, EdgePartType>> res;
        
//         ogdf::List<ogdf::edge> incidentEdges;
//         from->adjEdges(incidentEdges);

//         for(ogdf::edge e : incidentEdges) {
//             if(e->target() == from) {
//                 res.emplace_back(e->target(), _edge2types[e].second);
//             } else {
//                 res.emplace_back(e->source(), _edge2types[e].first);
//             }
//         }
//         return res;
//     }


//     void findGoodCutVertices() {
//         _bctree = std::make_unique<ogdf::BCTree>(_G);

//         const ogdf::Graph& T = _bctree->bcTree();


//         // for(ogdf::node v : _G.nodes) {
//         //     if(_bctree->typeOfGNode(v) == ogdf::BCTree::GNodeType::CutVertex) {
                

//         //         // if (plusCnt > 1 && minusCnt > 1) {
//         //         //     _goodCutVertices[v] = true;
//         //         //     std::cout << "Good cut vertex: " << node2name[v] << " with + edges: " << plusCnt << ", - edges: " << minusCnt << "\n";
//         //         // }
//         //     }
//         // }
    
//         // for(ogdf::node v : T.nodes) {
//         //     if(_bctree->typeOfBNode(v) == ogdf::BCTree::BNodeType::CComp) {
//         //         for

//         //     }
//         // }
        
//     }



//     std::vector<Graph*> splitIntoWeaklyConnectedComponents() {
//         std::vector<Graph*> res;

//         ogdf::NodeArray<int> compIdx(_G);
//         int nCC;
//         nCC = connectedComponents(_G, compIdx);

//         std::vector<std::vector<ogdf::node>> bucket(nCC);
//         for (ogdf::node v : _G.nodes) {
//             bucket[compIdx[v]].push_back(v);
//         }

//         std::vector<std::vector<ogdf::edge>> edgeBuckets(nCC);

//         for (ogdf::edge e : _G.edges) {
//             edgeBuckets[compIdx[e->source()]].push_back(e);
//         }


//         for(int i=0; i<nCC; i++) {
//             Graph* g = new Graph(_isBidirected);

//             std::unordered_map<ogdf::node, ogdf::node> oldToNew;

//             for(ogdf::node v : bucket[i]) {
//                 auto newV = g->addNode(node2name[v]);
//                 oldToNew[v] = newV;
//             }

//             for(ogdf::edge e : edgeBuckets[i]) {
//                 auto u = e->source();
//                 auto v = e->target();
//                 auto newU = oldToNew[u];
//                 auto newV = oldToNew[v];
//                 if(_isBidirected) g->addEdge(node2name[u], node2name[v], _edge2types[e].first, _edge2types[e].second);
//                 else g->addEdge(node2name[u], node2name[v]);
//             }

//             res.push_back(g);
//         }


//         return res;
//     }

// };


namespace GraphIO {
    void readGraph();

    void readStandard();
    void readGFA();
    void drawGraph(const ogdf::Graph& G, const std::string& file);

    void writeSuperbubbles();
}
