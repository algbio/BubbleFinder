#include <ogdf/basic/graph_generators.h>
#include <ogdf/layered/DfsAcyclicSubgraph.h>
#include <ogdf/fileformats/GraphIO.h>
#include <ogdf/basic/GraphAttributes.h>
#include <ogdf/basic/simple_graph_alg.h>
#include <ogdf/planarity/PlanarizationLayout.h>
#include <ogdf/decomposition/BCTree.h>  
#include <ogdf/decomposition/DynamicSPQRForest.h>
#include <ogdf/decomposition/DynamicSPQRTree.h>
#include <ogdf/augmentation/DfsMakeBiconnected.h>
#include <ogdf/decomposition/StaticSPQRTree.h>
#include <ogdf/decomposition/SPQRTree.h>
#include <ogdf/energybased/FMMMLayout.h>
#include <ogdf/tree/TreeLayout.h>
#include <ogdf/basic/List.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <stack>
#include <cassert>
#include <chrono>
#include <regex>
#include <cassert>
#include <typeinfo>
#include <thread>
#include <mutex>
#include <cstdlib>
#include <numeric>

#include <sys/resource.h>
#include <sys/time.h>


#include "io/graph_io.hpp"
#include "util/timer.hpp"
#include "util/logger.hpp"
#include "util/profiling.hpp"
#include "fas.h"


using namespace ogdf;


static void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " -g <graphFile> -o <outputFile> [--gfa]\n";
    std::exit(EXIT_FAILURE);
}


static std::string nextArgOrDie(const std::vector<std::string>& a, std::size_t& i, const char* flag) {
    if (++i >= a.size() || (a[i][0] == '-' && a[i] != "-")) {
        std::cerr << "Error: missing path after " << flag << "\n";
        usage(a[0].c_str());
    }
    return a[i];
}


void readArgs(int argc, char** argv) {
    auto& C = ctx();

    std::vector<std::string> args(argv, argv+argc);

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& s = args[i];

        if (s == "-g") {
            C.graphPath = nextArgOrDie(args, i, "-g");

        } else if (s == "-o") {
            C.outputPath = nextArgOrDie(args, i, "-o");

        } else if (s == "--gfa") {
            C.gfaInput = true;

        } else if (s == "-h" || s == "--help") {
            usage(argv[0]);

        } else if(s == "-j") {
            C.threads = std::stoi(nextArgOrDie(args, i, "-j"));
        } else {
            std::cerr << "Unknown argument: " << s << "\n";
            usage(argv[0]);
        }
    }
}








namespace solver {
    namespace {
        thread_local std::vector<std::pair<ogdf::node, ogdf::node>> *tls_superbubble_collector = nullptr;
    }

    static bool tryCommitSuperbubble(ogdf::node source, ogdf::node sink) {
        auto &C = ctx();
        if (C.isEntry[source] || C.isExit[sink]) {
            return false;
        }
        C.isEntry[source] = true;
        C.isExit[sink] = true;
        C.superbubbles.emplace_back(source, sink);
        return true;
    }
    struct BlockData {
        std::unique_ptr<Graph> Gblk;  
        ogdf::NodeArray<ogdf::node> toCc;
        // ogdf::NodeArray<ogdf::node> toBlk;
        ogdf::NodeArray<ogdf::node> toOrig;

        std::unique_ptr<ogdf::StaticSPQRTree> spqr;
        std::unordered_map<ogdf::edge, ogdf::edge> skel2tree; // mapping from skeleton virtual edge to tree edge
        ogdf::NodeArray<ogdf::node> parent; // mapping from node to parent in SPQR tree, it is possible since it is rooted, 
                                            // parent of root is nullptr

        ogdf::NodeArray<ogdf::node> blkToSkel;

        ogdf::node bNode {nullptr};

        bool isAcycic {true};


        ogdf::NodeArray<int> inDeg;
        ogdf::NodeArray<int> outDeg;
        // Cached global degrees (per block node) to avoid random access into global NodeArrays
        ogdf::NodeArray<int> globIn;
        ogdf::NodeArray<int> globOut;
    
        BlockData() {}
    };

    struct CcData {
        std::unique_ptr<ogdf::Graph> Gcc;
        ogdf::NodeArray<ogdf::node> toOrig;
        // ogdf::NodeArray<ogdf::node> toCopy;
        // ogdf::NodeArray<ogdf::node> toBlk;

        std::unique_ptr<ogdf::BCTree> bc;
        std::vector<BlockData> blocks;
    };



    void printBlockEdges(std::vector<CcData> &comps) {
        // auto& C = ctx();

        // for (size_t cid = 0; cid < comps.size(); ++cid) {
        //     const CcData &cc = comps[cid];

        //     for (size_t bid = 0; bid < cc.blocks.size(); ++bid) {
        //         const BlockData &blk = cc.blocks[bid];

        //         const Graph &Gb = *blk.Gblk;
        //         for (edge eB : Gb.edges) {
        //             node uB = eB->source();
        //             node vB = eB->target();

                    
        //             node uG = cc.toOrig[ blk.toCc[uB] ];
        //             node vG = cc.toOrig[ blk.toCc[vB] ];

        //         }
        //         // std::cout << '\n';
        //     }
        // }
        // std::cout << "----------------------------------------\n";
    }



    



    // add a new superbubble (if possible)
    void addSuperbubble(ogdf::node source, ogdf::node sink) {
        // If a thread-local collector is active, defer committing and just record the candidate
        if (tls_superbubble_collector) {
            tls_superbubble_collector->emplace_back(source, sink);
                return;
            }
        // Otherwise, commit directly to global state (sequential behavior)
        tryCommitSuperbubble(source, sink);


        // if(C.isEntry[source] || C.isExit[sink]) {
        //     std::cerr << ("Superbubble already exists for source %s and sink %s", C.node2name[source].c_str(), C.node2name[sink].c_str());
        //     return;
        // }
        // C.isEntry[source] = true;
        // C.isExit[sink] = true;
        // C.superbubbles.emplace_back(source, sink);

    }


    namespace SPQRsolve {
        struct EdgeDPState {
            node s{nullptr};      
            node t{nullptr};

            int localOutS{0};
            int localInT{0};
            int localOutT{0};
            int localInS{0};

            bool globalSourceSink{false}; 

            bool directST{false};
            bool directTS{false};

            bool hasLeakage{false};

            bool acyclic{true};

            int getDirection() const {
                if(acyclic && !globalSourceSink && localOutS>0 && localInT>0) return 1; // s -> t
                if(acyclic && !globalSourceSink && localOutT>0 && localInS>0) return -1; // t -> s
                return 0; // no direction ?
            }
        };

        struct NodeDPState {
            int outgoingCyclesCount{0}; 
            node lastCycleNode{nullptr}; 
            int outgoingSourceSinkCount{0};
            node lastSourceSinkNode{nullptr};
            int outgoingLeakageCount{0};
            node lastLeakageNode{nullptr};
        };

        // pair of dp states for each edge for both directions
        struct EdgeDP {
            EdgeDPState down;   // value valid in  parent -> child  direction
            EdgeDPState up;     // value valid in  child -> parent direction
        };


        void printAllStates(const ogdf::EdgeArray<EdgeDP> &edge_dp, const ogdf::NodeArray<NodeDPState> &node_dp,  const Graph &T) {
            auto& C = ctx();


            std::cout << "Edge dp states:" << std::endl;
            for(auto &e:T.edges) {
                {
                    EdgeDPState state = edge_dp[e].down;
                    if(state.s && state.t) {
                        std::cout << "Edge " << e->source() << " -> " << e->target() << ": ";
                        std::cout << "s = " << C.node2name[state.s] << ", ";
                        std::cout << "t = " << C.node2name[state.t] << ", ";
                        std::cout << "acyclic = " << state.acyclic << ", ";
                        std::cout << "global source = " << state.globalSourceSink << ", ";
                        std::cout << "hasLeakage = " << state.hasLeakage << ", ";
                        std::cout << "localInS = " << state.localInS << ", ";
                        std::cout << "localOutS = " << state.localOutS << ", ";
                        std::cout << "localInT = " << state.localInT << ", ";
                        std::cout << "localOutT = " << state.localOutT << ", ";
                        std::cout << "directST = " << state.directST << ", ";
                        std::cout << "directTS = " << state.directTS << ", ";
                        
                        std::cout << std::endl;
                    }
                }

                {
                    EdgeDPState state = edge_dp[e].up;
                    if(state.s && state.t) {
                        std::cout << "Edge " << e->target() << " -> " << e->source() << ": ";
                        std::cout << "s = " << C.node2name[state.s] << ", ";
                        std::cout << "t = " << C.node2name[state.t] << ", ";
                        std::cout << "acyclic = " << state.acyclic << ", ";
                        std::cout << "global source = " << state.globalSourceSink << ", ";
                        std::cout << "hasLeakage = " << state.hasLeakage << ", ";
                        std::cout << "localInS = " << state.localInS << ", ";
                        std::cout << "localOutS = " << state.localOutS << ", ";
                        std::cout << "localInT = " << state.localInT << ", ";
                        std::cout << "localOutT = " << state.localOutT << ", ";
                        std::cout << "directST = " << state.directST << ", ";
                        std::cout << "directTS = " << state.directTS << ", ";
                        
                        std::cout << std::endl;
                    }
                }
            }

            std::cout << "Node dp states: " << std::endl;
            for(node v : T.nodes) {
                std::cout << "Node " << v->index() << ", ";
                std::cout << "outgoingCyclesCount: " << node_dp[v].outgoingCyclesCount << ", ";
                std::cout << "outgoingLeakageCount: " << node_dp[v].outgoingLeakageCount << ", ";
                std::cout << "outgoingSourceSinkCount: " << node_dp[v].outgoingSourceSinkCount << ", ";

                std::cout << std::endl;
                
            }
        }

        void printAllEdgeStates(const ogdf::EdgeArray<EdgeDP> &edge_dp, const Graph &T) {
            auto& C = ctx();


            std::cout << "Edge dp states:" << std::endl;
            for(auto &e:T.edges) {
                {
                    EdgeDPState state = edge_dp[e].down;
                    if(state.s && state.t) {
                        std::cout << "Edge " << e->source() << " -> " << e->target() << ": ";
                        std::cout << "s = " << C.node2name[state.s] << ", ";
                        std::cout << "t = " << C.node2name[state.t] << ", ";
                        std::cout << "acyclic = " << state.acyclic << ", ";
                        std::cout << "global source = " << state.globalSourceSink << ", ";
                        std::cout << "hasLeakage = " << state.hasLeakage << ", ";
                        std::cout << "localInS = " << state.localInS << ", ";
                        std::cout << "localOutS = " << state.localOutS << ", ";
                        std::cout << "localInT = " << state.localInT << ", ";
                        std::cout << "localOutT = " << state.localOutT << ", ";
                        std::cout << "directST = " << state.directST << ", ";
                        std::cout << "directTS = " << state.directTS << ", ";
                        
                        std::cout << std::endl;
                    }
                }

                {
                    EdgeDPState state = edge_dp[e].up;
                    if(state.s && state.t) {
                        std::cout << "Edge " << e->target() << " -> " << e->source() << ": ";
                        std::cout << "s = " << C.node2name[state.s] << ", ";
                        std::cout << "t = " << C.node2name[state.t] << ", ";
                        std::cout << "acyclic = " << state.acyclic << ", ";
                        std::cout << "global source = " << state.globalSourceSink << ", ";
                        std::cout << "hasLeakage = " << state.hasLeakage << ", ";
                        std::cout << "localInS = " << state.localInS << ", ";
                        std::cout << "localOutS = " << state.localOutS << ", ";
                        std::cout << "localInT = " << state.localInT << ", ";
                        std::cout << "localOutT = " << state.localOutT << ", ";
                        std::cout << "directST = " << state.directST << ", ";
                        std::cout << "directTS = " << state.directTS << ", ";
                        
                        std::cout << std::endl;
                    }
                }
            }

        }

        std::string nodeTypeToString(SPQRTree::NodeType t) {
            switch (t) {
                case SPQRTree::NodeType::SNode: return "SNode";
                case SPQRTree::NodeType::PNode: return "PNode";
                case SPQRTree::NodeType::RNode: return "RNode";
                default: return "Unknown";
            }
        }


        void dfsSPQR_order(
            SPQRTree &spqr,
            std::vector<ogdf::edge> &edge_order, // order of edges to process
            std::vector<ogdf::node> &node_order,
            node curr = nullptr,
            node parent = nullptr,
            edge e = nullptr 
        ) {
            //PROFILE_FUNCTION();
            if(curr == nullptr) {
                curr = spqr.rootNode();
                parent = curr;
                dfsSPQR_order(spqr, edge_order, node_order, curr, parent);
                return;
            }



            // std::cout << "Node " << curr->index() << " is " << nodeTypeToString(spqr.typeOf(curr)) << std::endl;
            node_order.push_back(curr);
            for (adjEntry adj : curr->adjEntries) {
                node child = adj->twinNode();
                if (child == parent) continue;
                dfsSPQR_order(spqr, edge_order, node_order, child, curr, adj->theEdge());
            }
            if(curr!=parent) edge_order.push_back(e);
        }



        // process edge in the direction of parent to child
        // Computing A->B (curr_edge)
        void processEdge(ogdf::edge curr_edge, ogdf::EdgeArray<EdgeDP> &dp, NodeArray<NodeDPState> &node_dp, const CcData &cc, BlockData &blk) {
            //PROFILE_FUNCTION();
            auto& C = ctx();

            const ogdf::NodeArray<int> &globIn  = C.inDeg;
            const ogdf::NodeArray<int> &globOut = C.outDeg;
                        
            EdgeDPState &state = dp[curr_edge].down;
            EdgeDPState &back_state = dp[curr_edge].up;
            
            const StaticSPQRTree &spqr = *blk.spqr;
                        
            ogdf::node A = curr_edge->source();
            ogdf::node B = curr_edge->target();
            
            state.localOutS = 0;
            state.localInT  = 0;
            state.localOutT = 0;
            state.localInS  = 0;
            
            const Skeleton &skel = spqr.skeleton(B);
            const Graph &skelGraph = skel.getGraph();

            
            // Building new graph with correct orientation of virtual edges
            Graph newGraph;

            NodeArray<node> skelToNew(skelGraph, nullptr);
            for (node v : skelGraph.nodes) skelToNew[v] = newGraph.newNode();
            NodeArray<node> newToSkel(newGraph, nullptr);
            for (node v : skelGraph.nodes) newToSkel[skelToNew[v]] = v;


            {
                //PROFILE_BLOCK("processNode:: map block to skeleton nodes");
            for (ogdf::node h : skelGraph.nodes) {
                    ogdf::node vB = skel.original(h);
                    blk.blkToSkel[vB] = h;
                }
            }

            
            NodeArray<int> localInDeg(newGraph, 0), localOutDeg(newGraph, 0);




            // auto mapGlobalToNew = [&](ogdf::node vG) -> ogdf::node {
            //     // global -> component
            //     ogdf::node vComp = cc.toCopy[vG];
            //     if (!vComp) return nullptr;

            //     // component -> block
            //     ogdf::node vBlk  = cc.toBlk[vComp];
            //     if (!vBlk)  return nullptr;

            //     // block -> skeleton
            //     ogdf::node vSkel = blk.blkToSkel[vBlk];
            //     if (!vSkel) return nullptr;

            //     return skelToNew[vSkel];
            // };

            auto mapNewToGlobal = [&](ogdf::node vN) -> ogdf::node {
                if (!vN) return nullptr;

                ogdf::node vSkel = newToSkel[vN];
                if (!vSkel) return nullptr;

                ogdf::node vBlk  = skel.original(vSkel);
                if (!vBlk) return nullptr;

                ogdf::node vCc   = blk.toCc[vBlk];
                if (!vCc) return nullptr;

                return cc.toOrig[vCc];
            };


            // auto mapBlkToNew = [&](ogdf::node bV) -> ogdf::node {
            //     if (!bV) return nullptr;

            //     ogdf::node vSkel = newToSkel[vN];
            //     if (!vSkel) return nullptr;

            //     ogdf::node vBlk  = skel.original(vSkel);
            //     if (!vBlk) return nullptr;

            //     ogdf::node vCc   = blk.toCc[vBlk];
            //     if (!vCc) return nullptr;

            //     return cc.toOrig[vCc];
            // };






            // For debug
            auto printDegrees = [&]() {
                for(node vN:newGraph.nodes) {
                    node vG = mapNewToGlobal(vN);

                    // std::cout << C.node2name[vG] << ":    out: " << localOutDeg[vN] << ", in: " << localInDeg[vN] << std::endl;  
                }
            };



            ogdf::node nS, nT;


            for(edge e : skelGraph.edges) {
                node u = e->source();
                node v = e->target();

                node nU = skelToNew[u];
                node nV = skelToNew[v];
                

                if(!skel.isVirtual(e)) {
                    newGraph.newEdge(nU, nV);
                    localOutDeg[nU]++;
                    localInDeg[nV]++;

                    continue;
                }
                
                auto D = skel.twinTreeNode(e);
                

                if(D == A) {
                    ogdf::node vBlk = skel.original(v);
                    ogdf::node uBlk = skel.original(u);
                    
                    // ogdf::node vG  = blk.toOrig[vCc];
                    // ogdf::node uG  = blk.toOrig[uCc];

                    state.s = back_state.s = vBlk;
                    state.t = back_state.t = uBlk;

                    nS = nV;
                    nT = nU;
                    

                    continue;
                }


                edge treeE = blk.skel2tree.at(e);
                OGDF_ASSERT(treeE != nullptr);



                const EdgeDPState child = dp[treeE].down;
                int dir = child.getDirection();

                // ogdf::node nS = mapGlobalToNew(child.s);
                // ogdf::node nT = mapGlobalToNew(child.t);

                ogdf::node nA = skelToNew[blk.blkToSkel[child.s]];
                ogdf::node nB = skelToNew[blk.blkToSkel[child.t]];
                

                

                if(dir==1) {
                    newGraph.newEdge(nA, nB);
                } else if(dir==-1) {
                    newGraph.newEdge(nB, nA);
                } 


                if(nA == nU && nB == nV) {
                    localOutDeg[nA]+=child.localOutS;
                    localInDeg[nA]+=child.localInS;

                    localOutDeg[nB]+=child.localOutT;
                    localInDeg[nB]+=child.localInT;
                } else {
                    localOutDeg[nB]+=child.localOutT; 
                    localInDeg[nB]+=child.localInT;

                    localOutDeg[nA]+=child.localOutS;
                    localInDeg[nA]+=child.localInS;
                }



                state.acyclic &= child.acyclic;
                state.globalSourceSink |= child.globalSourceSink;
                state.hasLeakage |= child.hasLeakage;
            }


            // Direct ST/TS computation(only happens in P nodes)
            if(spqr.typeOf(B) == SPQRTree::NodeType::PNode) {
                for(edge e : skelGraph.edges) {
                    if(skel.isVirtual(e)) continue;
                    node u = e->source();
                    node v = e->target();

                    // node nU = skelToNew[u];
                    // node nV = skelToNew[v];

                    node bU = skel.original(u);
                    node bV = skel.original(v);


                    // if(mapGlobalToNew(state.s) == nU && mapGlobalToNew(state.t) == nV) {
                    //     state.directST = true;
                    // } else if(mapGlobalToNew(state.s) == nV && mapGlobalToNew(state.t) == nU) {
                    //     state.directTS = true;
                    // } else {
                    //     assert(false);
                    // }

                    if(state.s == bU && state.t == bV) {
                        state.directST = true;
                    } else if(state.s == bV && state.t == bU) {
                        state.directTS = true;
                    } else {
                        assert(false);
                    }
                }
            }


            // for (ogdf::node vN : newGraph.nodes) {
            //     ogdf::node vG  = mapNewToGlobal(vN);
            //     assert(vN == mapGlobalToNew(vG));

            //     if (vG == state.s || vG == state.t)
            //         continue;

                
            //     if(globIn[vG] != localInDeg[vN] || globOut[vG] != localOutDeg[vN]) {
            //         state.hasLeakage = true;
            //     }

            //     if (globIn[vG] == 0 || globOut[vG] == 0) {
            //         state.globalSourceSink = true;
            //     }
            // }



             for (ogdf::node nV : newGraph.nodes) {
                ogdf::node sV = newToSkel[nV];
                ogdf::node bV  = skel.original(sV);
                ogdf::node gV  = mapNewToGlobal(nV);

                if (bV == state.s || bV == state.t)
                    continue;

                
                if(globIn[gV] != localInDeg[nV] || globOut[gV] != localOutDeg[nV]) {
                    state.hasLeakage = true;
                }

                if (globIn[gV] == 0 || globOut[gV] == 0) {
                    state.globalSourceSink = true;
                }
            }





            // state.localInS = localInDeg[mapGlobalToNew(state.s)];
            // state.localOutS = localOutDeg[mapGlobalToNew(state.s)];

            // state.localInT = localInDeg[mapGlobalToNew(state.t)];
            // state.localOutT = localOutDeg[mapGlobalToNew(state.t)];


            state.localInS = localInDeg[nS];
            state.localOutS = localOutDeg[nS];

            state.localInT = localInDeg[nT];
            state.localOutT = localOutDeg[nT];



            
            if(state.acyclic) state.acyclic &= isAcyclic(newGraph);


            if(!state.acyclic) {
                node_dp[A].outgoingCyclesCount++;
                node_dp[A].lastCycleNode = B;
            }

            if(state.globalSourceSink) {
                node_dp[A].outgoingSourceSinkCount++;
                node_dp[A].lastSourceSinkNode = B;
            }

            if(state.hasLeakage) {
                node_dp[A].outgoingLeakageCount++;
                node_dp[A].lastLeakageNode = B;
            }
        }


        void processNode(node curr_node, EdgeArray<EdgeDP> &edge_dp, NodeArray<NodeDPState> &node_dp, const CcData &cc, BlockData &blk) {
            //PROFILE_FUNCTION();
            auto& C = ctx();

            const ogdf::NodeArray<int> &globIn  = C.inDeg;
            const ogdf::NodeArray<int> &globOut = C.outDeg;

            ogdf::node A = curr_node;
            
            const Graph &T = blk.spqr->tree();
            
            NodeDPState curr_state = node_dp[A]; 

            const StaticSPQRTree &spqr = *blk.spqr;
                        

            const Skeleton &skel = spqr.skeleton(A);
            const Graph &skelGraph = skel.getGraph();

            
            // Building new graph with correct orientation of virtual edges
            Graph newGraph;

            NodeArray<node> skelToNew(skelGraph, nullptr);
            for (node v : skelGraph.nodes) skelToNew[v] = newGraph.newNode();
            NodeArray<node> newToSkel(newGraph, nullptr);
            for (node v : skelGraph.nodes) newToSkel[skelToNew[v]] = v;

            for (ogdf::node h : skelGraph.nodes) {
                ogdf::node vB = skel.original(h);
                blk.blkToSkel[vB] = h;
            }

            
            NodeArray<int> localInDeg(newGraph, 0), localOutDeg(newGraph, 0);
            
            NodeArray<bool> isSourceSink(newGraph, false);
            int localSourceSinkCount = 0;

            NodeArray<bool> isLeaking(newGraph, false);
            int localLeakageCount = 0;

            EdgeArray<bool> isVirtual(newGraph, false);
            EdgeArray<EdgeDPState*> edgeToDp(newGraph, nullptr);
            EdgeArray<EdgeDPState*> edgeToDpR(newGraph, nullptr);
            EdgeArray<node> edgeChild(newGraph, nullptr);


            std::vector<edge> virtualEdges;


            // auto mapGlobalToNew = [&](ogdf::node vG) -> ogdf::node {
            //     // global -> component
            //     ogdf::node vComp = cc.toCopy[vG];
            //     if (!vComp) return nullptr;
            //     // component -> block
            //     ogdf::node vBlk  = cc.toBlk[vComp];
            //     if (!vBlk)  return nullptr;
            //     // block -> skeleton
            //     ogdf::node vSkel = blk.blkToSkel[vBlk];
            //     if (!vSkel) return nullptr;

            //     return skelToNew[vSkel];
            // };


            auto mapBlockToNew = [&](ogdf::node bV) -> ogdf::node {
                ogdf::node sV = blk.blkToSkel[bV];
                ogdf::node nV = skelToNew[sV];
                return nV;
            };



            auto mapNewToGlobal = [&](ogdf::node vN) -> ogdf::node {
                if (!vN) return nullptr;
                ogdf::node vSkel = newToSkel[vN];
                if (!vSkel) return nullptr;
                ogdf::node vBlk  = skel.original(vSkel);
                if (!vBlk) return nullptr;
                ogdf::node vCc   = blk.toCc[vBlk];
                if (!vCc) return nullptr;
                return cc.toOrig[vCc];
            };




            auto printDegrees = [&]() {
                for(node vN:newGraph.nodes) {
                    node vG = mapNewToGlobal(vN);
                }
            };


            // Building new graph
            {
                //PROFILE_BLOCK("processNode:: build oriented local graph");
            for(edge e : skelGraph.edges) {
                node u = e->source();
                node v = e->target();

                node nU = skelToNew[u];
                node nV = skelToNew[v];
                

                if(!skel.isVirtual(e)) {
                    auto newEdge = newGraph.newEdge(nU, nV);

                    isVirtual[newEdge] = false;

                    localOutDeg[nU]++;
                    localInDeg[nV]++;

                    continue;
                }
                
                auto B = skel.twinTreeNode(e);
                
                edge treeE = blk.skel2tree.at(e);
                OGDF_ASSERT(treeE != nullptr);



                EdgeDPState *child = (B == blk.parent(A) ? &edge_dp[treeE].up : &edge_dp[treeE].down);
                EdgeDPState *edgeToUpdate = (B == blk.parent(A) ? &edge_dp[treeE].down : &edge_dp[treeE].up);
                int dir = child->getDirection();

                // ogdf::node nS = mapGlobalToNew(child->s);
                // ogdf::node nT = mapGlobalToNew(child->t);

                ogdf::node nS = mapBlockToNew(child->s);
                ogdf::node nT = mapBlockToNew(child->t);



                edge newEdge = nullptr;

                if(dir==1 || dir == 0) {
                    newEdge = newGraph.newEdge(nS, nT);
                    
                    isVirtual[newEdge] = true;

                    virtualEdges.push_back(newEdge);

                    edgeToDp[newEdge] = edgeToUpdate;
                    edgeToDpR[newEdge] = child;
                    edgeChild[newEdge] = B;
                } else if(dir==-1) {
                    newEdge = newGraph.newEdge(nT, nS);
                    
                    isVirtual[newEdge] = true;
                    
                    virtualEdges.push_back(newEdge);
                    
                    edgeToDpR[newEdge] = child;
                    edgeToDp[newEdge] = edgeToUpdate;
                    edgeChild[newEdge] = B;

                    
                } else {
                    newEdge = newGraph.newEdge(nS, nT);
                    isVirtual[newEdge] = true;

                    virtualEdges.push_back(newEdge);


                    edgeChild[newEdge] = B;
                    edgeToDpR[newEdge] = child;

                    edgeToDp[newEdge] = edgeToUpdate;

                }

                if(nS == nU && nT == nV) {
                    localOutDeg[nS]+=child->localOutS; 
                    localInDeg[nS]+=child->localInS;

                    localOutDeg[nT]+=child->localOutT;
                    localInDeg[nT]+=child->localInT;
                } else {
                    localOutDeg[nT]+=child->localOutT; 
                    localInDeg[nT]+=child->localInT;

                    localOutDeg[nS]+=child->localOutS;
                    localInDeg[nS]+=child->localInS;
                }
                }
            }

        

            {
                //PROFILE_BLOCK("processNode:: mark source/sink and leakage");
            for(node vN : newGraph.nodes) {
                node vG = mapNewToGlobal(vN);
                    // node vB = skel.original(newToSkel[vN]);
                if(globIn[vG] == 0 || globOut[vG] == 0) {
                    localSourceSinkCount++;
                    isSourceSink[vN] = true;
                }

                if(globIn[vG] != localInDeg[vN] || globOut[vG] != localOutDeg[vN]) {
                    localLeakageCount++;
                    isLeaking[vN] = true;
                    }
                }
            }


            // calculating ingoing dp states of direct st and ts edges in P node
            if (spqr.typeOf(A) == StaticSPQRTree::NodeType::PNode) {
                //PROFILE_BLOCK("processNode:: P-node direct edge analysis");
                node pole0Blk = nullptr, pole1Blk = nullptr;
                {
                    auto it = skelGraph.nodes.begin();
                    if (it != skelGraph.nodes.end()) pole0Blk = skel.original(*it++);
                    if (it != skelGraph.nodes.end()) pole1Blk = skel.original(*it);
                }

                if (!pole0Blk || !pole1Blk)
                    return;

                node gPole0 = cc.toOrig[blk.toCc[pole0Blk]];
                node gPole1 = cc.toOrig[blk.toCc[pole1Blk]];


                int cnt01 = 0, cnt10 = 0;
                for (edge e : skelGraph.edges) {
                    if (!skel.isVirtual(e))
                    {
                        node uG = mapNewToGlobal(skelToNew[e->source()]);
                        node vG = mapNewToGlobal(skelToNew[e->target()]);
                        if (uG == gPole0 && vG == gPole1) ++cnt01;
                        else if (uG == gPole1 && vG == gPole0) ++cnt10;
                    }
                }


                for (edge e : skelGraph.edges) {
                    if (skel.isVirtual(e))
                    {
                        node  B = skel.twinTreeNode(e);
                        edge  treeE = blk.skel2tree.at(e);

                        SPQRsolve::EdgeDPState &st = 
                            (B == blk.parent(A) ? edge_dp[treeE].down
                            : edge_dp[treeE].up);

                        if (st.s == pole0Blk && st.t == pole1Blk) {
                            st.directST |= (cnt01 > 0);
                            st.directTS |= (cnt10 > 0);
                        }
                        else if (st.s == pole1Blk && st.t == pole0Blk) {
                            st.directST |= (cnt10 > 0);
                            st.directTS |= (cnt01 > 0);
                        }
                    }
                }
            } 
    


            // Computing acyclicity
            if(curr_state.outgoingCyclesCount>=2) {
                //PROFILE_BLOCK("processNode:: acyclicity - multi-outgoing case");
                for(edge e : virtualEdges) {
                    if(edgeToDp[e]->acyclic) {
                        node_dp[edgeChild[e]].outgoingCyclesCount++;
                        node_dp[edgeChild[e]].lastCycleNode = curr_node;
                    }
                    edgeToDp[e]->acyclic &= false;
                }
            } else if(node_dp[curr_node].outgoingCyclesCount == 1) {
                //PROFILE_BLOCK("processNode:: acyclicity - single-outgoing case");
                for (edge e : virtualEdges) {
                    if(edgeChild[e] != curr_state.lastCycleNode) {
                        if(edgeToDp[e]->acyclic) {
                            node_dp[edgeChild[e]].outgoingCyclesCount++;
                            node_dp[edgeChild[e]].lastCycleNode = curr_node;
                        }
                        edgeToDp[e]->acyclic &= false;
                    } else {                        
                        node  nU   = e->source();
                        node  nV   = e->target();
                        auto *st  = edgeToDp[e];
                        auto *ts  = edgeToDpR[e];
                        auto *child = edgeChild[e];
                        bool  acyclic = false;

                        newGraph.delEdge(e);
                        acyclic = isAcyclic(newGraph);

                        edge eRest = newGraph.newEdge(nU, nV);
                        isVirtual[eRest] = true;
                        edgeToDp [eRest] = st;
                        edgeToDpR[eRest] = ts;
                        edgeChild[eRest] = child;
            
                        if(edgeToDp[eRest]->acyclic && !acyclic) {
                            node_dp[edgeChild[eRest]].outgoingCyclesCount++;
                            node_dp[edgeChild[eRest]].lastCycleNode = curr_node;
                        }

                        edgeToDp[eRest]->acyclic &= acyclic;
                    }
                }

            } else {
                //PROFILE_BLOCK("processNode:: acyclicity - FAS baseline");

                FeedbackArcSet FAS(newGraph);
                std::vector<edge> fas = FAS.run();
                // find_feedback_arcs(newGraph, fas, toRemove);

                EdgeArray<bool> isFas(newGraph, 0);
                for (edge e : fas) isFas[e] = true;

                for (edge e : virtualEdges) {

                    if(edgeToDp[e]->acyclic && !isFas[e]) {
                        node_dp[edgeChild[e]].outgoingCyclesCount++;
                        node_dp[edgeChild[e]].lastCycleNode = curr_node;
                    }

                    edgeToDp[e]->acyclic &= isFas[e];
                }


                // NodeArray<int> comp(newGraph);
                // int sccs = strongComponents(newGraph, comp);

                // std::vector<int> size(sccs, 0);
                // for (node v : newGraph.nodes) ++size[comp[v]];

                // int trivial = 0, nonTrivial = 0, ntIdx = -1;

                // for (int i = 0; i < sccs; ++i) {
                //     if (size[i] > 1) { ++nonTrivial; ntIdx = i; }
                //     else ++trivial;
                // }

                // if (nonTrivial >= 2){
                //     for (edge e : virtualEdges) {
                //         if(edgeToDp[e]->acyclic) {
                //             node_dp[edgeChild[e]].outgoingCyclesCount++;
                //             node_dp[edgeChild[e]].lastCycleNode = curr_node;
                //         }

                //         edgeToDp[e]->acyclic &= false;
                //     }
                // } else if (nonTrivial == 1) {
                //     // std::vector<node> toRemove;
                //     // for (node v : newGraph.nodes)
                //     //     if (comp[v] != ntIdx) toRemove.push_back(v);

                //     FeedbackArcSet FAS(newGraph);
                //     std::vector<edge> fas = FAS.run();
                //     // find_feedback_arcs(newGraph, fas, toRemove);

                //     EdgeArray<bool> isFas(newGraph, 0);
                //     for (edge e : fas) isFas[e] = true;

                //     for (edge e : virtualEdges) {

                //         if(edgeToDp[e]->acyclic && !isFas[e]) {
                //             node_dp[edgeChild[e]].outgoingCyclesCount++;
                //             node_dp[edgeChild[e]].lastCycleNode = curr_node;
                //         }

                //         edgeToDp[e]->acyclic &= isFas[e];
                //     }
                // }
            }



            // computing global sources/sinks
            {
                //PROFILE_BLOCK("processNode:: compute global source/sink");
            if(curr_state.outgoingSourceSinkCount >= 2) {
                // all ingoing have source
                for(edge e : virtualEdges) {
                    if(!edgeToDp[e]->globalSourceSink) {
                        node_dp[edgeChild[e]].outgoingSourceSinkCount++;
                        node_dp[edgeChild[e]].lastSourceSinkNode = curr_node;
                    }


                    edgeToDp[e]->globalSourceSink |= true;
                }
            } else if(curr_state.outgoingSourceSinkCount == 1) {
                for(edge e : virtualEdges) {
                    // if(!isVirtual[e]) continue;
                    if(edgeChild[e] != curr_state.lastSourceSinkNode) {
                        if(!edgeToDp[e]->globalSourceSink) {
                            node_dp[edgeChild[e]].outgoingSourceSinkCount++;
                            node_dp[edgeChild[e]].lastSourceSinkNode = curr_node;
                        }

                        edgeToDp[e]->globalSourceSink |= true;
                    } else {
                        node vN = e->source(), uN = e->target();
                        if((int)isSourceSink[vN] + (int)isSourceSink[uN] < localSourceSinkCount) {
                            if(!edgeToDp[e]->globalSourceSink) {
                                node_dp[edgeChild[e]].outgoingSourceSinkCount++;
                                node_dp[edgeChild[e]].lastSourceSinkNode = curr_node;
                            }

                            edgeToDp[e]->globalSourceSink |= true;
                        }
                    }
                }
            } else {
                for(edge e : virtualEdges) {
                    // if(!isVirtual[e]) continue;
                    node vN = e->source(), uN = e->target();
                    if((int)isSourceSink[vN] + (int)isSourceSink[uN] < localSourceSinkCount) {
                        if(!edgeToDp[e]->globalSourceSink) {
                            node_dp[edgeChild[e]].outgoingSourceSinkCount++;
                            node_dp[edgeChild[e]].lastSourceSinkNode = curr_node;
                        }

                        edgeToDp[e]->globalSourceSink |= true;
                    }
                    
                    }
                }
            }


            // computing leakage
            {
                //PROFILE_BLOCK("processNode:: compute leakage");
            if(curr_state.outgoingLeakageCount >= 2) {
                for(edge e : virtualEdges) {
                    // if(!isVirtual[e]) continue;

                    if(!edgeToDp[e]->hasLeakage) {
                        node_dp[edgeChild[e]].outgoingLeakageCount++;
                        node_dp[edgeChild[e]].lastLeakageNode = curr_node;
                    }

                    edgeToDp[e]->hasLeakage |= true;
                }
            } else if(curr_state.outgoingLeakageCount == 1) {
                for(edge e : virtualEdges) {
                    // if(!isVirtual[e]) continue;

                    if(edgeChild[e] != curr_state.lastLeakageNode) {
                        if(!edgeToDp[e]->hasLeakage) {
                            node_dp[edgeChild[e]].outgoingLeakageCount++;
                            node_dp[edgeChild[e]].lastLeakageNode = curr_node;
                        }
                        edgeToDp[e]->hasLeakage |= true;
                    } else {
                        node vN = e->source(), uN = e->target();
                        if((int)isLeaking[vN] + (int)isLeaking[uN] < localLeakageCount) {
                            if(!edgeToDp[e]->hasLeakage) {
                                node_dp[edgeChild[e]].outgoingLeakageCount++;
                                node_dp[edgeChild[e]].lastLeakageNode = curr_node;
                            }
                            edgeToDp[e]->hasLeakage |= true;
                        }
                    }
                }
            } else {
                for(edge e : virtualEdges) {
                    // if(!isVirtual[e]) continue;

                    node vN = e->source(), uN = e->target();
                    if((int)isLeaking[vN] + (int)isLeaking[uN] < localLeakageCount) {
                        if(!edgeToDp[e]->hasLeakage) {
                            node_dp[edgeChild[e]].outgoingLeakageCount++;
                            node_dp[edgeChild[e]].lastLeakageNode = curr_node;
                        }
                        edgeToDp[e]->hasLeakage |= true;
                        }
                    }
                }
            }


            // updating local degrees of poles of states going into A
            {
                //PROFILE_BLOCK("processNode:: update DP local degrees at poles");
            for(edge e:virtualEdges) {
                // if(!isVirtual[e]) continue;
                node vN = e->source();
                node uN = e->target();

                EdgeDPState *BA = edgeToDp[e];
                EdgeDPState *AB = edgeToDpR[e];

                BA->localInS = localInDeg[mapBlockToNew(BA->s)] - AB->localInS; 
                BA->localInT = localInDeg[mapBlockToNew(BA->t)] - AB->localInT; 

                BA->localOutS = localOutDeg[mapBlockToNew(BA->s)] - AB->localOutS; 
                BA->localOutT = localOutDeg[mapBlockToNew(BA->t)] - AB->localOutT; 
                }
            }
        }



        void tryBubblePNodeGrouping(
            const node &A,
            const CcData &cc,
            const BlockData &blk,
            const EdgeArray<EdgeDP> &edge_dp
        ) {    
            if(blk.spqr->typeOf(A) != SPQRTree::NodeType::PNode) return;

            const Skeleton &skel = blk.spqr->skeleton(A);
            const Graph &skelGraph = skel.getGraph();


            node bS, bT;
            {
                auto it = skelGraph.nodes.begin();
                if (it != skelGraph.nodes.end()) bS = skel.original(*it++);
                if (it != skelGraph.nodes.end()) bT = skel.original(*it);
            }



            int directST = 0, directTS = 0;
            for(auto &e:skelGraph.edges) {
                if(skel.isVirtual(e)) continue;

                node a = skel.original(e->source()), b = skel.original(e->target());

                if(a == bS && b == bT) directST++;
                else directTS++;
            }


            // printAllEdgeStates(edge_dp, blk.spqr->tree());

            for(int q=0; q<2; q++) {       
                // s -> t
                
                // std::cout << "s: " << ctx().node2name[s] << ", t: " << ctx().node2name[t] << std::endl;
                std::vector<const EdgeDPState*> goodS, goodT;
                
                int localOutSSum=directST, localInTSum=directST;
                
                // std::cout << " at " << A << std::endl;

                for (adjEntry adj : A->adjEntries) {
                    auto e = adj->theEdge();
                    // std::cout << e->source() << " -> " << e->target() << std::endl;
                    auto& state = (e->source() == A ? edge_dp[e].down : edge_dp[e].up);
                    // directST = (state.s == s ? state.directST : state.directTS);
                    // directTS = (state.s == s ? state.directTS : state.directST);
                

                    
                    int localOutS = (state.s==bS ? state.localOutS : state.localOutT), localInT = (state.t==bT ? state.localInT : state.localInS);

                    localOutSSum += localOutS;
                    localInTSum += localInT;
                    // std::cout << adj->twinNode() << " has outS" <<  localOutS << " and outT " << localInT << std::endl; 

                    if(localOutS > 0) {
                        // std::cout << "PUSHING TO GOODs" << (e->source() == A ? e->target(): e->source()) << std::endl;
                        goodS.push_back(&state);
                    }

                    if(localInT > 0) {
                        // std::cout << "PUSHING TO GOODt" << (e->source() == A ? e->target(): e->source()) << std::endl;
                        goodT.push_back(&state);
                    }
                }

                // if(q == 1) std::swap(goodS, goodT);
                // std::cout << "directST: " << directST << ", directTS: " << directTS << std::endl;

                

                // std::cout << ctx().node2name[cc.toOrig[blk.toCc[s]]] << ", " << ctx().node2name[cc.toOrig[blk.toCc[t]]] << " has s:" << goodS.size() << " and t:" << goodT.size() << std::endl;
                bool good = true;
                for(auto &state:goodS) {
                    if((state->s==bS && state->localInS > 0) || (state->s==bT && state->localInT > 0)) {
                        // std::cout << "BAD 1" << std::endl;
                        good = false;
                    } 

                    good &= state->acyclic;
                    good &= !state->globalSourceSink;
                    good &= !state->hasLeakage;
                }

                for(auto &state:goodT) {
                    if((state->t==bT && state->localOutT > 0) || (state->t==bS && state->localOutS > 0)) {
                        // std::cout << "BAD 2" << std::endl;
                        good = false;
                    }
                    
                    
                    good &= state->acyclic;
                    good &= !state->globalSourceSink;
                    good &= !state->hasLeakage;
                }

                good &= directTS == 0;
                good &= goodS == goodT;
                good &= goodS.size() > 0;

                good &= (localOutSSum == ctx().outDeg[cc.toOrig[blk.toCc[bS]]] && localInTSum == ctx().inDeg[cc.toOrig[blk.toCc[bT]]]);

                // std::cout << "localOutSSum: " << localOutSSum << ", localInTSum: " << localInTSum << std::endl;

                // std::cout << ctx().outDeg[cc.toOrig[blk.toCc[s]]] << ", " << 

                // std::cout << "SETS ARE SAME: " << (goodS == goodT) << std::endl;

                if(good) {
                    // std::cout << "ADDING SUPERBUBBLE " << ctx().node2name[bS] << ", " << ctx().node2name[bT] << std::endl;
                    addSuperbubble(cc.toOrig[blk.toCc[bS]], cc.toOrig[blk.toCc[bT]]);
                }

                std::swap(directST, directTS);
                std::swap(bS, bT);

            }

        }

        
        void tryBubble(const EdgeDPState &curr,
                const EdgeDPState &back,
                const BlockData &blk,
                const CcData &cc,
                bool swap, 
                bool additionalCheck
        ) {
            node S = swap ? blk.toOrig[curr.t] : blk.toOrig[curr.s];
            node T = swap ? blk.toOrig[curr.s] : blk.toOrig[curr.t];

            // std::cout << ctx().node2name[S] << " " << ctx().node2name[T] << " " << (additionalCheck) << std::endl;

            
            /* take the counts from the current direction … */

            int outS = swap ? curr.localOutT  : curr.localOutS;
            int outT = swap ? curr.localOutS : curr.localOutT;
            int inS  = swap ? curr.localInT  : curr.localInS;
            int inT  = swap ? curr.localInS : curr.localInT;


            // if(curr.s && curr.t) {
            //     std::cout << "s = " << ctx().node2name[curr.s] << ", ";
            //     std::cout << "t = " << ctx().node2name[curr.t] << ", ";
            //     std::cout << "acyclic = " << curr.acyclic << ", ";
            //     std::cout << "global source = " << curr.globalSourceSink << ", ";
            //     std::cout << "hasLeakage = " << curr.hasLeakage << ", ";
            //     std::cout << "localInS = " << curr.localInS << ", ";
            //     std::cout << "localOutS = " << curr.localOutS << ", ";
            //     std::cout << "localInT = " << curr.localInT << ", ";
            //     std::cout << "localOutT = " << curr.localOutT << ", ";
            //     std::cout << "directST = " << curr.directST << ", ";
            //     std::cout << "directTS = " << curr.directTS << ", ";
                
            //     std::cout << std::endl;
            // }

            // if(back.s && back.t) {
            //     std::cout << "s = " << ctx().node2name[back.s] << ", ";
            //     std::cout << "t = " << ctx().node2name[back.t] << ", ";
            //     std::cout << "acyclic = " << back.acyclic << ", ";
            //     std::cout << "global source = " << back.globalSourceSink << ", ";
            //     std::cout << "hasLeakage = " << back.hasLeakage << ", ";
            //     std::cout << "localInS = " << back.localInS << ", ";
            //     std::cout << "localOutS = " << back.localOutS << ", ";
            //     std::cout << "localInT = " << back.localInT << ", ";
            //     std::cout << "localOutT = " << back.localOutT << ", ";
            //     std::cout << "directST = " << back.directST << ", ";
            //     std::cout << "directTS = " << back.directTS << ", ";
                
            //     std::cout << std::endl;
            // }



            // int outS = swap ? curr.localOutT + (int)back.directST : curr.localOutS + (int)back.directTS;
            // int outT = swap ? curr.localOutS + (int)back.directTS : curr.localOutT + (int)back.directST;
            // int inS  = swap ? curr.localInT + (int)back.directTS : curr.localInS + (int)back.directST;
            // int inT  = swap ? curr.localInS + (int)back.directST: curr.localInT + (int)back.directTS;
            // std::cout << "before: " << std::endl;
            // std::cout << outS << " " << inS << " | " << outT << " " << inT << std::endl; 

            

            if(back.directST) {
                // std::cout << " added because back.directST" << std::endl;
                if(!swap) {
                    outS++;
                    inT++;
                } else {
                    inS++;
                    outT++;
                }
            } 
            if(back.directTS){
                // std::cout << " added because back.directTS" << std::endl;
                if(!swap) {
                    inS++;
                    outT++;    
                } else {
                    outS++;
                    inT++;
                }
            }

            // std::cout << "after" << std::endl;
            // std::cout << outS << " " << inS << " | " << outT << " " << inT << std::endl; 

            bool backGood = true;

            if (back.s == curr.s && back.t == curr.t) {
                backGood &= (!back.directTS);
            } else if (back.s == curr.t && back.t == curr.s) {
                backGood &= (!back.directST);
            }

            bool acyclic = curr.acyclic;
            bool noLeakage = !curr.hasLeakage;
            bool noGSource = !curr.globalSourceSink;



            if (
                !additionalCheck &&
                acyclic &&
                noGSource &&
                noLeakage &&
                backGood &&
                outS > 0 &&
                inT > 0 &&
                ctx().outDeg[S] == outS &&
                ctx().inDeg [T] == inT &&
                !ctx().isEntry[S] &&
                !ctx().isExit [T])
            {
                if(additionalCheck) {
                    if(!swap) {
                        if(back.directST) addSuperbubble(S, T);
                    } else {
                        if(back.directTS) addSuperbubble(S, T);
                    }
                } else {
                    addSuperbubble(S, T);
                }
            }

        }



        void collectSuperbubbles(const CcData &cc, BlockData &blk, EdgeArray<EdgeDP> &edge_dp, NodeArray<NodeDPState> &node_dp) {
            //PROFILE_FUNCTION();
            const Graph &T = blk.spqr->tree();
            // printAllStates(edge_dp, node_dp, T);

            for(edge e : T.edges) {
                // std::cout << "CHECKING FOR " << e->source() << " " << e->target() << std::endl;
                const EdgeDPState &down = edge_dp[e].down;
                const EdgeDPState &up   = edge_dp[e].up;
                

                // if(blk.spqr->typeOf(e->target()) != SPQRTree::NodeType::SNode) {
                //     std::cout << "DOWN" << std::endl;
                bool additionalCheck;

                additionalCheck = (blk.spqr->typeOf(e->source()) == SPQRTree::NodeType::PNode && blk.spqr->typeOf(e->target()) == SPQRTree::NodeType::SNode);
                tryBubble(down, up, blk, cc, false, additionalCheck);
                tryBubble(down, up, blk, cc, true, additionalCheck);
                // }
                
                // if(blk.spqr->typeOf(e->source()) != SPQRTree::NodeType::SNode) {
                    // std::cout << "UP" << std::endl;
                additionalCheck = (blk.spqr->typeOf(e->target()) == SPQRTree::NodeType::PNode && blk.spqr->typeOf(e->source()) == SPQRTree::NodeType::SNode);

                    tryBubble(up, down, blk, cc, false, additionalCheck);
                    tryBubble(up, down, blk, cc, true, additionalCheck);
                // }

                blk.isAcycic &= (down.acyclic && up.acyclic);

            }
            for(node v : T.nodes) {
                tryBubblePNodeGrouping(v, cc, blk, edge_dp);
            } 
        }

    }

    void checkBlockByCutVertices(const BlockData &blk, const CcData &cc)    
    {

        if (!blk.isAcycic) { 
            return;
        }

        //PROFILE_FUNCTION();
        auto start = std::chrono::high_resolution_clock::now();
        auto &C      = ctx();
        const Graph &G = *blk.Gblk;

        auto step1 = std::chrono::high_resolution_clock::now();
        node src=nullptr, snk=nullptr;

        for (node v : G.nodes) {
            node vG   = blk.toOrig[v];
            int inL   = blk.inDeg [v], outL = blk.outDeg[v];
            int inG   = C.inDeg  [vG], outG = C.outDeg[vG];

            bool isSrc = (inL  == 0 && outL == outG);
            bool isSnk = (outL == 0 && inL == inG);

            if (isSrc ^ isSnk) {
                if (isSrc) { 
                    if(src) return; 
                    src=v; 
                } else { 
                    if(snk) return; 
                    snk=v; 
                }
            } else if (!(inL == inG && outL == outG)) {
                return;
            }
        }

        auto step2 = std::chrono::high_resolution_clock::now();
        if (!src || !snk) { return; }

        auto step3 = std::chrono::high_resolution_clock::now();


        auto step4 = std::chrono::high_resolution_clock::now();
        // reachability
        NodeArray<bool> vis(G,false); std::stack<node> S; vis[src]=true; S.push(src);
        bool reach=false;
        while(!S.empty() && !reach){
            node u=S.top();S.pop();
            for(adjEntry a=u->firstAdj();a;a=a->succ())
                if(a->isSource()){
                    node v=a->twinNode();
                    if(!vis[v]){ if(v==snk){reach=true;break;}
                                vis[v]=true; S.push(v);}
                }
        }
        auto step5 = std::chrono::high_resolution_clock::now();
        if(!reach) { return; }

        node srcG = blk.toOrig[src], snkG = blk.toOrig[snk];
        addSuperbubble(srcG, snkG);
        
        auto end = std::chrono::high_resolution_clock::now();
        
        // Debug slow checkBlockByCutVertices calls
        auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // if (totalDuration.count() > 5000) { // > 5ms
        //     auto step1Duration = std::chrono::duration_cast<std::chrono::microseconds>(step2 - step1);
        //     auto step2Duration = std::chrono::duration_cast<std::chrono::microseconds>(step3 - step2);
        //     auto step3Duration = std::chrono::duration_cast<std::chrono::microseconds>(step4 - step3);
        //     auto step4Duration = std::chrono::duration_cast<std::chrono::microseconds>(step5 - step4);
        //     auto step5Duration = std::chrono::duration_cast<std::chrono::microseconds>(end - step5);
            
        //     std::cout << "SLOW checkBlockByCutVertices: " << totalDuration.count() << "μs total" << std::endl;
        //     std::cout << "  - Find src/snk: " << step1Duration.count() << "μs" << std::endl;
        //     std::cout << "  - Check src/snk: " << step2Duration.count() << "μs" << std::endl;
        //     std::cout << "  - isAcyclic: " << step3Duration.count() << "μs" << std::endl;
        //     std::cout << "  - Reachability: " << step4Duration.count() << "μs" << std::endl;
        //     std::cout << "  - Add superbubble: " << step5Duration.count() << "μs" << std::endl;
        //     std::cout << "  - Graph size: " << G.numberOfNodes() << " nodes, " << G.numberOfEdges() << " edges" << std::endl;
        // }
    }






    void solveSPQR(BlockData &blk, const CcData &cc) {
        //PROFILE_FUNCTION();
        if (!blk.spqr || blk.Gblk->numberOfNodes() < 3) {
            return; // No SPQR tree needed for small blocks
        }
        
        auto T = blk.spqr->tree();

        EdgeArray<SPQRsolve::EdgeDP> dp(T);
        NodeArray<SPQRsolve::NodeDPState> node_dp(T);


        std::vector<ogdf::node> nodeOrder;
        std::vector<ogdf::edge> edgeOrder;

        SPQRsolve::dfsSPQR_order(*blk.spqr, edgeOrder, nodeOrder);

        ogdf::NodeArray<ogdf::node> blkToSkel(*blk.Gblk, nullptr);

        blk.blkToSkel = blkToSkel;

        for(auto e:edgeOrder) {
            SPQRsolve::processEdge(e, dp, node_dp, cc, blk);
        }
        
        for(auto v:nodeOrder) {
            SPQRsolve::processNode(v, dp, node_dp, cc, blk);
        }

        SPQRsolve::collectSuperbubbles(cc, blk, dp, node_dp);
    }





    void findMiniSuperbubbles() {
        //PROFILE_FUNCTION();
        // TIME_BLOCK("Finding mini-superbubbles");

        auto& C = ctx();

        logger::info("Finding mini-superbubbles..");

        for(auto &e:C.G.edges) {
            auto a = e->source();
            auto b = e->target();

            if(a->outdeg() == 1 && b->indeg() == 1) {
                bool ok=true;
                for(auto &w:b->adjEntries) {
                    auto e = w->theEdge();
                    auto src = e->source();
                    auto tgt = e->target();
                    if(src == b && tgt == a) {
                        ok = false;
                        break;
                    }
                }
                
                if(ok) {
                    addSuperbubble(a, b);
                } 
            }
        }


        logger::info("Checked for mini-superbubbles");
    }



    // // // BEST
    // static void buildBlockData(
    //     // const std::vector<node>& verts,
    //         const std::unordered_set<node> &verts,
    //         CcData& cc,
    //         BlockData& blk) {        
    //     //PROFILE_FUNCTION();

    //     {
    //         //PROFILE_BLOCK("buildBlockData:: create clear graph");
    //         blk.Gblk = std::make_unique<Graph>();
    //     }

    //     {
    //         //PROFILE_BLOCK("buildBlockData:: blk mappings inits");

    //         blk.toOrig.init(*blk.Gblk, nullptr);
    //         blk.toCc.init(*blk.Gblk, nullptr);
    //         blk.inDeg.init(*blk.Gblk, 0);
    //         blk.outDeg.init(*blk.Gblk, 0);
    //     }

    //     // Use array mapping instead of hash map for speed
    //     NodeArray<node> cc_to_blk(*cc.Gcc, nullptr);

    //     {
    //         //PROFILE_BLOCK("buildBlockData:: create nodes in Gblk");

    //         for (node vCc : verts) {
    //             node vB = blk.Gblk->newNode();
    //             cc_to_blk[vCc] = vB;
    //             blk.toCc[vB] = vCc;
    //             blk.toOrig[vB] = cc.toOrig[vCc];
    //         }
    //     }

    //     {
    //         //PROFILE_BLOCK("buildBlockData:: create edges in Gblk");

    //         for (edge hE : cc.bc->hEdges(blk.bNode)) {
    //             edge eCc = cc.bc->original(hE);
    //             auto src = cc_to_blk[eCc->source()];
    //             auto tgt = cc_to_blk[eCc->target()];
    //             if (src && tgt) {
    //             edge e = blk.Gblk->newEdge(src, tgt);
    //             blk.outDeg[e->source()]++;
    //             blk.inDeg[e->target()]++;
    //             }
    //         }
    //     }
    // }

    static void buildBlockDataParallel(CcData& cc, BlockData& blk) {
        // PROFILE_FUNCTION();
        {
            PROFILE_BLOCK("buildBlockDataParallel:: all but SPQR and parent");
            {
                //PROFILE_BLOCK("buildBlockDataParallel:: create clear graph");
                blk.Gblk = std::make_unique<Graph>();
            }

            {
                blk.toOrig.init(*blk.Gblk, nullptr);
                blk.toCc.init(*blk.Gblk, nullptr);
                blk.inDeg.init(*blk.Gblk, 0);
                blk.outDeg.init(*blk.Gblk, 0);
            }


            std::unordered_set<node> verts;
            {
                //PROFILE_BLOCK("buildBlockDataParallel:: collect vertices");
                for (edge hE : cc.bc->hEdges(blk.bNode)) {
                    edge eC = cc.bc->original(hE);
                    verts.insert(eC->source());
                    verts.insert(eC->target());
                }
            }

            std::unordered_map<node, node> cc_to_blk;
            cc_to_blk.reserve(verts.size());

            {
                //PROFILE_BLOCK("buildBlockDataParallel:: create nodes in Gblk");
                for (node vCc : verts) {
                    node vB = blk.Gblk->newNode();
                    cc_to_blk[vCc] = vB;
                    blk.toCc[vB] = vCc;
                    node vG = cc.toOrig[vCc];
                    blk.toOrig[vB] = vG;
                }
            }

            {
                //PROFILE_BLOCK("buildBlockDataParallel:: create edges in Gblk");
                for (edge hE : cc.bc->hEdges(blk.bNode)) {
                    edge eCc = cc.bc->original(hE);
                    auto srcIt = cc_to_blk.find(eCc->source());
                    auto tgtIt = cc_to_blk.find(eCc->target());
                    if (srcIt != cc_to_blk.end() && tgtIt != cc_to_blk.end()) {
                        edge e = blk.Gblk->newEdge(srcIt->second, tgtIt->second);
                        blk.outDeg[e->source()]++;
                        blk.inDeg[e->target()]++;
                    }
                }
            }


            {
                //PROFILE_BLOCK("buildBlockDataParallel:: init cached global degrees");
                blk.globIn.init(*blk.Gblk, 0);
                blk.globOut.init(*blk.Gblk, 0);
                for (node vB : blk.Gblk->nodes) {
                    node vG = blk.toOrig[vB];
                    blk.globIn[vB] = ctx().inDeg[vG];
                    blk.globOut[vB] = ctx().outDeg[vG];
                }
            }
        }

        if (blk.Gblk->numberOfNodes() >= 3) {
            // PROFILE_BLOCK("buildBlockDataParallel:: spqr building and parent");
            {
                PROFILE_BLOCK("buildBlockDataParallel:: build SPQR tree");
                blk.spqr = std::make_unique<StaticSPQRTree>(*blk.Gblk);
            }
            const Graph& T = blk.spqr->tree();
            blk.skel2tree.reserve(2*T.edges.size());
            blk.parent.init(T, nullptr);

            node root = blk.spqr->rootNode();
            blk.parent[root] = root;

            for (edge te : T.edges) {
                {
                    node u = te->source();
                    node v = te->target();
                    blk.parent[v] = u;
                }

                if (auto eSrc = blk.spqr->skeletonEdgeSrc(te)) {
                    blk.skel2tree[eSrc] = te;
                }
                if (auto eTgt = blk.spqr->skeletonEdgeTgt(te)) {
                    blk.skel2tree[eTgt] = te;
                }
            }
        }
    }



    struct WorkItem {
        CcData* cc;
        BlockData* blockData;
        node bNode;
    };

    struct BlockPrep {
        CcData* cc;
        node bNode;
    };


    struct ThreadBcTreeArgs {
        size_t tid;
        size_t numThreads;
        int nCC;
        size_t* nextIndex;
        std::mutex* workMutex;
        std::vector<std::unique_ptr<CcData>>* components;
        std::vector<BlockPrep>* blockPreps;

    };

    void* worker_bcTree(void* arg) {
        std::unique_ptr<ThreadBcTreeArgs> targs(static_cast<ThreadBcTreeArgs*>(arg));
        size_t tid = targs->tid;
        size_t numThreads = targs->numThreads;
        int nCC = targs->nCC;
        size_t* nextIndex = targs->nextIndex;
        std::mutex* workMutex = targs->workMutex;
        std::vector<std::unique_ptr<CcData>>* components = targs->components;
        std::vector<BlockPrep>* blockPreps = targs->blockPreps;

        // size_t chunkSize = std::max<size_t>(1, nCC / (numThreads * 4));
        size_t chunkSize = 1;
        size_t processed = 0;

        while (true) {
            size_t startIndex, endIndex;
            {
                std::lock_guard<std::mutex> lock(*workMutex);
                if (*nextIndex >= static_cast<size_t>(nCC)) break;
                startIndex = *nextIndex;
                endIndex = std::min(*nextIndex + chunkSize, static_cast<size_t>(nCC));
                *nextIndex = endIndex;
            }

            auto chunkStart = std::chrono::high_resolution_clock::now();
             
            for (size_t cid = startIndex; cid < endIndex; ++cid) {
                CcData* cc = (*components)[cid].get();

                {
                    // PROFILE_BLOCK("solveStreaming:: building bc tree");
                    cc->bc = std::make_unique<BCTree>(*cc->Gcc);
                }

                std::vector<BlockPrep> localPreps;
                {
                    // PROFILE_BLOCK("solveStreaming:: collect bc tree nodes");
                    for (node v : cc->bc->bcTree().nodes) {
                        if (cc->bc->typeOfBNode(v) == BCTree::BNodeType::BComp) {
                            localPreps.push_back({cc, v});
                        }
                    }
                }

                {
                    static std::mutex prepMutex;
                    std::lock_guard<std::mutex> lock(prepMutex);
                    blockPreps->insert(blockPreps->end(), localPreps.begin(), localPreps.end());
                }
                
                ++processed;
            }
            

            auto chunkEnd = std::chrono::high_resolution_clock::now();
            auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);


                                                            
            if (chunkDuration.count() < 1000) {
                // std::cout << tid << " enlarging chunk from " << chunkSize;
                chunkSize = std::min(chunkSize * 2, static_cast<size_t>(blockPreps->size() / numThreads));
                // std::cout << " to " << chunkSize << std::endl;
            } else if (chunkDuration.count() > 5000) {
                // std::cout << tid << " shrinking chunk from " << chunkSize;
                chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                // std::cout << " to " << chunkSize << std::endl;
            }
        }

        std::cout << "Thread " << tid << " built " << processed << " components (bc trees)" << std::endl;
        return nullptr;
    }


    // BEST NOW
    void solveStreaming() {
        //PROFILE_FUNCTION();
        auto& C = ctx();
        Graph& G = C.G;

        NodeArray<int> compIdx(G);
        int nCC;
        {
            //PROFILE_BLOCK("solveStreaming:: ComputeCC");
            nCC = connectedComponents(G, compIdx);
        }

        std::vector<std::vector<node>> bucket(nCC);
        {
            //PROFILE_BLOCK("solveStreaming:: bucket nodes");
            for (node v : G.nodes) {
                bucket[compIdx[v]].push_back(v);
            }
        }

        std::vector<std::vector<edge>> edgeBuckets(nCC);

        {
            //PROFILE_BLOCK("solveStreaming:: bucket edges");
            for (edge e : G.edges) {
                edgeBuckets[compIdx[e->source()]].push_back(e);
            }
        }


        NodeArray<node> orig_to_cc(G, nullptr);


        logger::info("Streaming over {} components", nCC);

        
        std::vector<std::unique_ptr<CcData>> components(nCC);
        std::vector<std::unique_ptr<BlockData>> allBlockData; 
        
        std::vector<WorkItem> workItems;


        std::vector<BlockPrep> blockPreps;

        {
            TIME_BLOCK("solveStreaming:: build components in parallel (Gcc only)");
            PROFILE_BLOCK("solveStreaming:: build components in parallel (Gcc only)");

            size_t numThreads = std::thread::hardware_concurrency();
            // size_t numThreads = 16;
            numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});
            std::vector<std::thread> workers;
            workers.reserve(numThreads);


            std::mutex workMutex;
            size_t nextIndex = 0;

            for (size_t tid = 0; tid < numThreads; ++tid) {
                workers.emplace_back([&, tid]() {
                    size_t chunkSize = std::max<size_t>(1, nCC / (numThreads * 4));
                    size_t processed = 0;
                    while (true) {
                        size_t startIndex, endIndex;
                        {
                            std::lock_guard<std::mutex> lock(workMutex);
                            if (nextIndex >= static_cast<size_t>(nCC)) break;
                            startIndex = nextIndex;
                            endIndex = std::min(nextIndex + chunkSize, static_cast<size_t>(nCC));
                            nextIndex = endIndex;
                        }


                        auto chunkStart = std::chrono::high_resolution_clock::now();
                        
                        
                        for (size_t ci = startIndex; ci < endIndex; ++ci) {
                            int cid = static_cast<int>(ci);
                            auto cc = std::make_unique<CcData>();

                            {
                                PROFILE_BLOCK("solveStreaming:: rebuild cc graph");
                                cc->Gcc = std::make_unique<Graph>();
                                cc->toOrig.init(*cc->Gcc, nullptr);
            
                                std::unordered_map<node, node> orig_to_cc_local;
                                orig_to_cc_local.reserve(bucket[cid].size());

                                for (node vG : bucket[cid]) {
                                    node vC = cc->Gcc->newNode();
                                    cc->toOrig[vC] = vG;
                                    orig_to_cc_local[vG] = vC;
                                }

                                for (edge e : edgeBuckets[cid]) {
                                        cc->Gcc->newEdge(orig_to_cc_local[e->source()], orig_to_cc_local[e->target()]);
                                }
                            }

                            components[cid] = std::move(cc);
                            
                            processed++;
                        }



                        auto chunkEnd = std::chrono::high_resolution_clock::now();
                        auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);


                                                                     
                        if (chunkDuration.count() < 1000) {
                            // std::cout << tid << " enlarging chunk from " << chunkSize;
                            chunkSize = std::min(chunkSize * 2, static_cast<size_t>(blockPreps.size() / numThreads));
                            // std::cout << " to " << chunkSize << std::endl;
                        } else if (chunkDuration.count() > 5000) {
                            // std::cout << tid << " shrinking chunk from " << chunkSize;
                            chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                            // std::cout << " to " << chunkSize << std::endl;
                        }

                    }
                    std::cout << "Thread " << tid << " built " << processed << " components (Gcc)" << std::endl;
                });
            }

            for (auto &t : workers) t.join();
        }

        {
            TIME_BLOCK("solveStreaming:: build BC trees and collect blocks");
            PROFILE_BLOCK("solveStreaming:: build BC trees and collect blocks");


            size_t numThreads = std::thread::hardware_concurrency();
            // size_t numThreads = 16;
            numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});

            std::vector<pthread_t> threads(numThreads);


            std::mutex workMutex;
            size_t nextIndex = 0;


            for (size_t tid = 0; tid < numThreads; ++tid) {
                pthread_attr_t attr;
                pthread_attr_init(&attr);

                size_t stackSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;
                pthread_attr_setstacksize(&attr, stackSize);
                

                ThreadBcTreeArgs* args = new ThreadBcTreeArgs{
                    tid,
                    numThreads,
                    nCC,
                    &nextIndex,
                    &workMutex,
                    &components,
                    &blockPreps
                };

                int ret = pthread_create(&threads[tid], &attr, worker_bcTree, args);
                if (ret != 0) {
                    std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                    delete args;
                }

                pthread_attr_destroy(&attr);
            }

            for (size_t tid = 0; tid < numThreads; ++tid) {
                pthread_join(threads[tid], nullptr);
            }





            // for (int cid = 0; cid < nCC; ++cid) {
                


            //     auto *cc = components[cid].get();
            //     {                    
            //         PROFILE_BLOCK("solveStreaming:: building bc tree");
            //         cc->bc = std::make_unique<BCTree>(*cc->Gcc);
            //     }

            //     {
            //         PROFILE_BLOCK("solveStreaming:: collect bc tree nodes");
            //         for (node v : cc->bc->bcTree().nodes) {
            //             if (cc->bc->typeOfBNode(v) == BCTree::BNodeType::BComp) {
            //                 blockPreps.push_back({cc, v});
            //             }
            //         }
            //     }
            // }
            // });

            // ww[0].join();

        }

        allBlockData.resize(blockPreps.size());
        

        {
            PROFILE_BLOCK("solveStreaming:: build BlockData sequential");

            for (size_t i = 0; i < blockPreps.size(); ++i) {
                const auto &bp = blockPreps[i];

                allBlockData[i] = std::make_unique<BlockData>();
                allBlockData[i]->bNode = bp.bNode;

                {
                    // PROFILE_BLOCK("solveStreaming:: buildBlockData parallel");
                    buildBlockDataParallel(*bp.cc, *allBlockData[i]);
                }
            }

        }


        {
            TIME_BLOCK("solveStreaming:: build BlockData in parallel");
            PROFILE_BLOCK("solveStreaming:: build BlockData in parallel");

            size_t numThreads = std::thread::hardware_concurrency();


            // size_t maxChunkSize = 1;

            numThreads = std::min({(size_t)C.threads, (size_t)blockPreps.size(), numThreads});
            std::vector<std::thread> workers;
            workers.reserve(numThreads);

            std::mutex workMutex;
            size_t nextIndex = 0;

            for (size_t tid = 0; tid < numThreads; ++tid) {
                workers.emplace_back([&, tid]() {

                    // size_t chunkSize = std::max<size_t>(1, blockPreps.size() / (numThreads * numThreads));
                    // size_t chunkSize = 124950;
                    size_t chunkSize = 1;

                    
                    std::cout << "Thread " << tid << " chunk size: " << chunkSize << std::endl;

                    size_t processed = 0;
                    while (true) {
                        size_t startIndex, endIndex;
                        {
                            std::lock_guard<std::mutex> lock(workMutex);
                            if (nextIndex >= blockPreps.size()) break;
                            startIndex = nextIndex;
                            endIndex = std::min(nextIndex + chunkSize, blockPreps.size());
                            nextIndex = endIndex;
                        }


                        auto chunkStart = std::chrono::high_resolution_clock::now();


                        for (size_t i = startIndex; i < endIndex; ++i) {
                            const auto &bp = blockPreps[i];

                            allBlockData[i] = std::make_unique<BlockData>();
                            allBlockData[i]->bNode = bp.bNode;

                            {
                                // PROFILE_BLOCK("solveStreaming:: buildBlockData parallel");
                                buildBlockDataParallel(*bp.cc, *allBlockData[i]);
                            }

                            processed++;
                        }


                        auto chunkEnd = std::chrono::high_resolution_clock::now();
                        auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);


                        // if (chunkSize < 10) {
                        //     chunkSize = std::min(chunkSize * 2, static_cast<size_t>(blockPreps.size() / numThreads));
                        // }


                                                
                        if (chunkDuration.count() < 100) {
                            // std::cout << tid << " enlarging chunk from " << chunkSize;
                            chunkSize = std::min(chunkSize * 2, blockPreps.size() / numThreads);
                            // std::cout << " to " << chunkSize << std::endl;
                        } else if (chunkDuration.count() > 2000) {
                            // std::cout << tid << " shrinking chunk from " << chunkSize;
                            chunkSize = std::max(chunkSize / 2, (size_t)1);
                            // std::cout << " to " << chunkSize << std::endl;
                        }
                        

                    }
                    std::cout << "Thread " << tid << " built " << processed << " BlockData objects" << std::endl;
                });
            }

            for (auto &t : workers) t.join();

            workItems.reserve(workItems.size() + allBlockData.size());
            for (size_t i = 0; i < allBlockData.size(); ++i) {
                workItems.push_back({blockPreps[i].cc, allBlockData[i].get(), blockPreps[i].bNode});
            }
        }

        {
            TIME_BLOCK("solveStreaming:: sort work items by block edges");
            PROFILE_BLOCK("solveStreaming:: sort work items by block edges");
            std::sort(workItems.begin(), workItems.end(), [](const WorkItem &a, const WorkItem &b) {
                const int ea = a.cc->bc->numberOfEdges(a.bNode);
                const int eb = b.cc->bc->numberOfEdges(b.bNode);
                return ea > eb;
            });
        }

        for(int i = 0; i < std::min(10, (int)workItems.size()); ++i) {
            std::cout << "Work item " << i << ": Block with " << workItems[i].blockData->Gblk->numberOfNodes() << " nodes, " << workItems[i].blockData->Gblk->numberOfEdges() << " edges" << std::endl;
        }

        std::cout << "Built " << components.size() << " components and " << allBlockData.size() << " blocks" << std::endl;
        std::cout << "Collected " << workItems.size() << " work items" << std::endl;



        std::vector<std::vector<std::pair<node, node>>> blockResults(workItems.size());
        {
            TIME_BLOCK("solveStreaming:: process blocks in parallel (no buildBlockData)");
            PROFILE_BLOCK("solveStreaming:: process blocks in parallel (no buildBlockData)");
            
            size_t numThreads = std::thread::hardware_concurrency();
            numThreads = std::min({(size_t)C.threads, (size_t)workItems.size(), numThreads});
            std::vector<std::thread> workers;
            workers.reserve(numThreads);
            
            std::mutex workMutex;
            std::mutex debugMutex;
            size_t nextWorkIndex = 0;
            
            std::cout << "Processing " << workItems.size() << " work items with " << numThreads << " threads" << std::endl;
            
            auto phaseStart = std::chrono::high_resolution_clock::now();
            std::vector<std::chrono::high_resolution_clock::time_point> threadLastWork(numThreads);
            std::vector<size_t> threadWorkCounts(numThreads, 0);
            
            for (size_t tid = 0; tid < numThreads; ++tid) {
                workers.emplace_back([&, tid]() {
                    // size_t chunkSize = std::max(workItems.size() / (numThreads * 4), size_t(1));
                    size_t chunkSize = 1;
                    auto lastChunkStart = std::chrono::high_resolution_clock::now();
                    size_t totalProcessed = 0;
                    
                    auto threadStart = std::chrono::high_resolution_clock::now();
                    threadLastWork[tid] = threadStart;

                    while (true) {
                        size_t startIndex, endIndex;
                        {
                            std::lock_guard<std::mutex> lock(workMutex);
                            if (nextWorkIndex >= workItems.size()) {
                                break;
                            }
                            startIndex = nextWorkIndex;
                            endIndex = std::min(nextWorkIndex + chunkSize, workItems.size());
                            nextWorkIndex = endIndex;
                        }

                        
                        auto chunkStart = std::chrono::high_resolution_clock::now();
                        
                        for (size_t i = startIndex; i < endIndex; ++i) {
                            const auto& workItem = workItems[i];
                            threadLastWork[tid] = std::chrono::high_resolution_clock::now();
                            tls_superbubble_collector = &blockResults[i];

                            BlockData *blockData = workItem.blockData;

                            
                            auto spqrStart = std::chrono::high_resolution_clock::now();
                            if (blockData->Gblk->numberOfNodes() >= 3) {
                                solveSPQR(*blockData, *workItem.cc);
                            }
                            auto spqrEnd = std::chrono::high_resolution_clock::now();

                            auto checkStart = std::chrono::high_resolution_clock::now();
                            checkBlockByCutVertices(*blockData, *workItem.cc);
                            auto checkEnd = std::chrono::high_resolution_clock::now();

                            tls_superbubble_collector = nullptr;
                            totalProcessed++;
                            threadWorkCounts[tid]++;

                            auto checkDuration = std::chrono::duration_cast<std::chrono::microseconds>(checkEnd - checkStart);
                            auto spqrDuration = std::chrono::duration_cast<std::chrono::microseconds>(spqrEnd - spqrStart);
                            if (checkDuration.count() > 10000 || spqrDuration.count() > 10000) {
                                std::lock_guard<std::mutex> debugLock(debugMutex);
                                // std::cout << "Thread " << tid << ": Block " << i << " - checkBlock: " << checkDuration.count() << "μs, solveSPQR: " << spqrDuration.count() << "μs" << std::endl;
                            }
                        }
                        
                        auto chunkEnd = std::chrono::high_resolution_clock::now();
                        auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);
                        
                        if (chunkDuration.count() < 100) {
                            chunkSize = std::min(chunkSize * 2, static_cast<size_t>(workItems.size() / numThreads));
                        } else if (chunkDuration.count() > 2000) {
                            chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                        }
                        
                        // Debug output every 100 items processed
                        // if (totalProcessed % 100 == 0) {
                        //     std::cout << "Thread " << tid << " processed " << totalProcessed << " items, chunk size: " << chunkSize << std::endl;
                        // }
                        
                        lastChunkStart = chunkStart;
                    }
                    
                    auto threadEnd = std::chrono::high_resolution_clock::now();
                    auto threadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(threadEnd - threadStart);
                    
                    std::lock_guard<std::mutex> debugLock(debugMutex);
                    std::cout << "Thread " << tid << " finished: " << totalProcessed << " items in " << threadDuration.count() << "ms" << std::endl;
                });
            }
            
            for (auto& worker : workers) {
                worker.join();
            }

            auto phaseEnd = std::chrono::high_resolution_clock::now();
            auto phaseDuration = std::chrono::duration_cast<std::chrono::milliseconds>(phaseEnd - phaseStart);
            
            for (size_t tid = 0; tid < numThreads; ++tid) {
                auto lastWorkDuration = std::chrono::duration_cast<std::chrono::milliseconds>(phaseEnd - threadLastWork[tid]);
                std::cout << "Thread " << tid << ": " << threadWorkCounts[tid] << " items, last work " << lastWorkDuration.count() << "ms ago" << std::endl;
            }
            std::cout << "Total phase time: " << phaseDuration.count() << "ms" << std::endl;
        }

        {
            TIME_BLOCK("solveStreaming:: commit results");
            PROFILE_BLOCK("solveStreaming:: commit results");
            for (const auto& candidates : blockResults) {
                for (const auto& p : candidates) {
                    tryCommitSuperbubble(p.first, p.second);
                }
            }
        }
    }



    void solve() {
        TIME_BLOCK("Finding superbubbles");
        findMiniSuperbubbles();
        // GraphIO::drawGraph(ctx().G, "movedMinis");
        solveStreaming();
    }
}





int main(int argc, char** argv) {
    // rlimit w;
    // w.rlim_cur = RLIM_INFINITY;
    // w.rlim_max = RLIM_INFINITY;
    
    // setrlimit(RLIMIT_STACK, &w);


    rlimit rl;
    rl.rlim_cur = RLIM_INFINITY;
    rl.rlim_max = RLIM_INFINITY;
    if (setrlimit(RLIMIT_STACK, &rl) != 0) {
        perror("setrlimit");
    }


    //PROFILE_BLOCK("Total run");
    //PROFILE_FUNCTION();
    TIME_BLOCK("Starting graph reading...");
    logger::init();

    

    readArgs(argc, argv);
    {
        PROFILE_BLOCK("Graph reading");
    GraphIO::readGraph();
    }
    GraphIO::drawGraph(ctx().G, "input_graph");


    // std::vector<edge> res;
    // solver::run_fas(ctx().G, res);


    // for(auto &e:res) {
    //     std::cout << e->source() << " " << e->target() << std::endl;
    // }

    // return 0;


    solver::solve();

    std::cout << "Superbubbles found:\n";
    std::cout << ctx().superbubbles.size() << std::endl;
    // return 0;
    if(false)
    std::sort(ctx().superbubbles.begin(), ctx().superbubbles.end(), [&](std::pair<node, node> &a, std::pair<node, node> &b) {
        if(std::stoi(ctx().node2name[a.first]) < std::stoi(ctx().node2name[b.first])) return true;
        else if(std::stoi(ctx().node2name[a.first]) == std::stoi(ctx().node2name[b.first])) {
            return std::stoi(ctx().node2name[a.second]) < std::stoi(ctx().node2name[b.second]);
        }
        return false;
    });

    {
        PROFILE_BLOCK("Writing output");
        TIME_BLOCK("Writing output");
    GraphIO::writeSuperbubbles();
    }

    PROFILING_REPORT();

    // for(auto &sb:ctx().superbubbles) {
    //     std::cout << ctx().node2name[sb.first] << " " << ctx().node2name[sb.second] << '\n';
    // }

    // std::cout << "total count: " << ctx().superbubbles.size() << std::endl;
    
    // std::cout << ctx().outDeg[ctx().name2node["153"]] << std::endl;

    return 0;
}