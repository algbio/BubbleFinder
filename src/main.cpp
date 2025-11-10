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
#include <ogdf/tree/LCA.h>



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
#include <queue>

#include <sys/resource.h>
#include <sys/time.h>


#include "io/graph_io.hpp"
#include "util/timer.hpp"
#include "util/logger.hpp"
#include "util/profiling.hpp"
#include "fas.h"

#include "util/mark_scope.hpp"
#include "util/mem_time.hpp"
#include "util/phase_accum.hpp"


using namespace ogdf;

static std::string g_report_json_path;


static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " -g <graphFile> -o <outputFile> [--gfa] "
              << "[--superbubbles | --snarls] "
              << "[-j <threads>] "
              << "[--report-json <file>] "
              << "[-m <stack size in bytes>]\n";

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

        } else if (s == "--report-json") {
            g_report_json_path = nextArgOrDie(args, i, "--report-json");

        } else if (s == "-h" || s == "--help") {
            usage(args[0].c_str());

        } else if (s == "-j") {
            C.threads = std::stoi(nextArgOrDie(args, i, "-j"));

        } else if (s == "--superbubbles") {
            C.bubbleType = Context::BubbleType::SUPERBUBBLE;

        } else if (s == "--snarls") {
            C.bubbleType = Context::BubbleType::SNARL;

        } else if(s == "-m") {
            C.stackSize = std::stoull(nextArgOrDie(args, i, "-m"));

        } else {
            std::cerr << "Unknown argument: " << s << "\n";
            usage(args[0].c_str());
        }
    }
}




size_t snarlsFound = 0;
size_t isolatedNodesCnt = 0;


namespace solver {
    namespace superbubble {
        namespace {
            thread_local std::vector<std::pair<ogdf::node, ogdf::node>> *tls_superbubble_collector = nullptr;
        }

        static bool tryCommitSuperbubble(ogdf::node source, ogdf::node sink) {
            auto &C = ctx();
            if (C.isEntry[source] || C.isExit[sink] || ctx().node2name[source] == "_trash" || ctx().node2name[sink] == "_trash") {
                // std::cout << C.node2name[source] << " " << C.node2name[sink] << " is already superbubble\n";
                return false;
            }
            C.isEntry[source] = true;
            C.isExit[sink] = true;
            C.superbubbles.emplace_back(source, sink);
            // std::cout << "Added " << C.node2name[source] << " " << C.node2name[sink] << " as superbubble\n";
            return true;
        }
        struct BlockData {
            std::unique_ptr<ogdf::Graph> Gblk;  
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

            // BlockData() = default;
            // ~BlockData() = default;

            // // disable copying & moving
            // BlockData(const BlockData&) = delete;
            // BlockData& operator=(const BlockData&) = delete;
            // BlockData(BlockData&&) = delete;
            // BlockData& operator=(BlockData&&) = delete;


            // BlockData() = default;

            // BlockData(const BlockData&) = delete;
            // BlockData& operator=(const BlockData&) = delete;
            // BlockData(BlockData&&) = delete;
            // BlockData& operator=(BlockData&&) = delete;
        };

        struct CcData {
            std::unique_ptr<ogdf::Graph> Gcc;
            ogdf::NodeArray<ogdf::node> toOrig;
            // ogdf::NodeArray<ogdf::node> toCopy;
            // ogdf::NodeArray<ogdf::node> toBlk;

            std::unique_ptr<ogdf::BCTree> bc;
            // std::vector<BlockData> blocks;
            std::vector<std::unique_ptr<BlockData>> blocks; 
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



        

        void addSuperbubble(ogdf::node source, ogdf::node sink) {
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
            MARK_SCOPE_MEM("sb/checkCutVertices");

            if (!isAcyclic(*blk.Gblk)) { 
                return;
            }

            auto &C      = ctx();
            const Graph &G = *blk.Gblk;

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

            if (!src || !snk) { return; }

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
            if(!reach) { return; }

            node srcG = blk.toOrig[src], snkG = blk.toOrig[snk];
            addSuperbubble(srcG, snkG);
        }




        void solveSPQR(BlockData &blk, const CcData &cc) {
            MARK_SCOPE_MEM("sb/solveSPQR");

            if (!blk.spqr || blk.Gblk->numberOfNodes() < 3) {
                return;
            }
            
            const Graph &T = blk.spqr->tree();

            EdgeArray<SPQRsolve::EdgeDP> dp(T);
            NodeArray<SPQRsolve::NodeDPState> node_dp(T);

            std::vector<ogdf::node> nodeOrder;
            std::vector<ogdf::edge> edgeOrder;

            SPQRsolve::dfsSPQR_order(*blk.spqr, edgeOrder, nodeOrder);

            blk.blkToSkel.init(*blk.Gblk, nullptr);

            for(auto e:edgeOrder) {
                SPQRsolve::processEdge(e, dp, node_dp, cc, blk);
            }
            
            for(auto v:nodeOrder) {
                SPQRsolve::processNode(v, dp, node_dp, cc, blk);
            }

            SPQRsolve::collectSuperbubbles(cc, blk, dp, node_dp);
        }




        void findMiniSuperbubbles() {
            MARK_SCOPE_MEM("sb/findMini");

            auto& C = ctx();

            logger::info("Finding mini-superbubbles..");

            for(auto &e:C.G.edges) {
                auto a = e->source();
                auto b = e->target();

                if(a->outdeg() == 1 && b->indeg() == 1) {
                    bool ok=true;
                    for(auto &w:b->adjEntries) {
                        auto e2 = w->theEdge();
                        auto src = e2->source();
                        auto tgt = e2->target();
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


        static void buildBlockDataParallel(const CcData& cc, BlockData& blk) {
            {
                MARK_SCOPE_MEM("sb/blockData/build");
                blk.Gblk = std::make_unique<Graph>();

                blk.toOrig.init(*blk.Gblk, nullptr);
                blk.toCc.init(*blk.Gblk, nullptr);
                blk.inDeg.init(*blk.Gblk, 0);
                blk.outDeg.init(*blk.Gblk, 0);

                std::unordered_set<node> verts;
                for (edge hE : cc.bc->hEdges(blk.bNode)) {
                    edge eC = cc.bc->original(hE);
                    verts.insert(eC->source());
                    verts.insert(eC->target());
                }

                std::unordered_map<node, node> cc_to_blk;
                cc_to_blk.reserve(verts.size());

                for (node vCc : verts) {
                    node vB = blk.Gblk->newNode();
                    cc_to_blk[vCc] = vB;
                    blk.toCc[vB] = vCc;
                    node vG = cc.toOrig[vCc];
                    blk.toOrig[vB] = vG;
                }

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

                blk.globIn.init(*blk.Gblk, 0);
                blk.globOut.init(*blk.Gblk, 0);
                for (node vB : blk.Gblk->nodes) {
                    node vG = blk.toOrig[vB];
                    blk.globIn[vB] = ctx().inDeg[vG];
                    blk.globOut[vB] = ctx().outDeg[vG];
                }
            }

            if (blk.Gblk->numberOfNodes() >= 3) {
                {
                    MARK_SCOPE_MEM("sb/blockData/spqr_build");
                    blk.spqr = std::make_unique<StaticSPQRTree>(*blk.Gblk);
                }
                const Graph& T = blk.spqr->tree();
                blk.skel2tree.reserve(2*T.edges.size());
                blk.parent.init(T, nullptr);

                node root = blk.spqr->rootNode();
                blk.parent[root] = root;

                for (edge te : T.edges) {
                    node u = te->source();
                    node v = te->target();
                    blk.parent[v] = u;

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
            // BlockData* blockData;
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
                        MARK_SCOPE_MEM("sb/worker_bcTree/build");
                        cc->bc = std::make_unique<BCTree>(*cc->Gcc);
                    }

                    std::vector<BlockPrep> localPreps;
                    {
                        MARK_SCOPE_MEM("sb/worker_bcTree/collect_B_nodes");
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
                    chunkSize = std::min(chunkSize * 2, static_cast<size_t>(nCC / numThreads));
                } else if (chunkDuration.count() > 5000) {
                    chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                }
            }

            std::cout << "Thread " << tid << " built " << processed << " components (bc trees)" << std::endl;
            return nullptr;
        }

        struct ThreadBlockBuildArgs {
            size_t tid;
            size_t numThreads;
            size_t nBlocks;
            size_t* nextIndex;
            std::mutex* workMutex;
            std::vector<BlockPrep>* blockPreps;
            std::vector<std::unique_ptr<BlockData>>* allBlockData;
        };

        static void* worker_buildBlockData(void* arg) {
            std::unique_ptr<ThreadBlockBuildArgs> targs(static_cast<ThreadBlockBuildArgs*>(arg));
            size_t tid        = targs->tid;
            size_t numThreads = targs->numThreads;
            size_t nBlocks    = targs->nBlocks;
            size_t* nextIndex = targs->nextIndex;
            std::mutex* workMutex = targs->workMutex;
            auto* blockPreps  = targs->blockPreps;
            auto* allBlockData = targs->allBlockData;
            size_t chunkSize = 1;
            size_t processed = 0;
            while (true) {
                size_t startIndex, endIndex;
                {
                    std::lock_guard<std::mutex> lock(*workMutex);
                    if (*nextIndex >= nBlocks) break;
                    startIndex = *nextIndex;
                    endIndex   = std::min(startIndex + chunkSize, nBlocks);
                    *nextIndex = endIndex;
                }
                auto chunkStart = std::chrono::high_resolution_clock::now();
                for (size_t i = startIndex; i < endIndex; ++i) {
                    const BlockPrep &bp = (*blockPreps)[i];
                    (*allBlockData)[i] = std::make_unique<BlockData>();
                    (*allBlockData)[i]->bNode = bp.bNode;
                    buildBlockDataParallel(*bp.cc, *(*allBlockData)[i]);
                    ++processed;
                }
                auto chunkEnd = std::chrono::high_resolution_clock::now();
                auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);
                if (chunkDuration.count() < 100) {
                    size_t maxPerThread = std::max<size_t>(1, nBlocks / std::max<size_t>(numThreads, 1));
                    chunkSize = std::min(chunkSize * 2, maxPerThread);
                } else if (chunkDuration.count() > 2000) {
                    chunkSize = std::max<size_t>(1, chunkSize / 2);
                }
            }
            std::cout << "Thread " << tid << " built " << processed << " BlockData objects" << std::endl;
            return nullptr;
        }

        struct ThreadProcessArgs {
            size_t tid;
            size_t numThreads;
            size_t nItems;
            size_t* nextIndex;
            std::mutex* workMutex;
            std::vector<WorkItem>* workItems;
            std::vector<std::unique_ptr<BlockData>>* allBlockData;
            std::vector<std::vector<std::pair<ogdf::node, ogdf::node>>>* blockResults;
        };


        static void* worker_processBlocks(void* arg) {
            std::unique_ptr<ThreadProcessArgs> targs(static_cast<ThreadProcessArgs*>(arg));
            size_t* nextIndex   = targs->nextIndex;
            std::mutex* workMux = targs->workMutex;
            auto& items         = *targs->workItems;
            auto& allBlocks     = *targs->allBlockData;
            auto& results       = *targs->blockResults;
            const size_t n      = targs->nItems;
            while (true) {
                size_t i;
                {
                    std::lock_guard<std::mutex> lk(*workMux);
                    if (*nextIndex >= n) break;
                    i = (*nextIndex)++;
                }

                const WorkItem &w = items[i];

                BlockData *blk = allBlocks[i].get();
                if (!blk) {
                    results[i] = {};
                    continue;
                }

                std::vector<std::pair<ogdf::node, ogdf::node>> local;
                tls_superbubble_collector = &local;

                if (blk->Gblk && blk->Gblk->numberOfNodes() >= 3) {
                    solveSPQR(*blk, *w.cc);
                }
                checkBlockByCutVertices(*blk, *w.cc);

                tls_superbubble_collector = nullptr;
                results[i] = std::move(local);
            }
            return nullptr;
        }



                
        // BEST NOW

void solveStreaming() {
    //PROFILE_FUNCTION();
    auto& C = ctx();
    Graph& G = C.G;

    std::vector<WorkItem> workItems;

    std::vector<std::unique_ptr<CcData>> components;
    std::vector<std::unique_ptr<BlockData>> allBlockData;

{
    // PROFILE_BLOCK("solve:: prepare");


    NodeArray<int> compIdx(G);
    int nCC;
    {
        MARK_SCOPE_MEM("sb/phase/ComputeCC");
        //PROFILE_BLOCK("solveStreaming:: ComputeCC");
        nCC = connectedComponents(G, compIdx);
    }

    components.resize(nCC);

    std::vector<std::vector<node>> bucket(nCC);
    {
        MARK_SCOPE_MEM("sb/phase/BucketNodes");
        //PROFILE_BLOCK("solveStreaming:: bucket nodes");
        for (node v : G.nodes) {
            bucket[compIdx[v]].push_back(v);
        }
    }

    std::vector<std::vector<edge>> edgeBuckets(nCC);

    {
        MARK_SCOPE_MEM("sb/phase/BucketEdges");
        //PROFILE_BLOCK("solveStreaming:: bucket edges");
        for (edge e : G.edges) {
            edgeBuckets[compIdx[e->source()]].push_back(e);
        }
    }


    NodeArray<node> orig_to_cc(G, nullptr);


    logger::info("Streaming over {} components", nCC);

    


    std::vector<BlockPrep> blockPreps;

    {
        PROFILE_BLOCK("solve:: building data");
        MEM_TIME_BLOCK("BUILD: BC+SPQR");
        ACCUM_BUILD();

        {
            MARK_SCOPE_MEM("sb/phase/GccBuildParallel");
            size_t numThreads = std::thread::hardware_concurrency();
            numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});
            std::vector<std::thread> workers;
            workers.reserve(numThreads);

            std::mutex workMutex;
            size_t nextIndex = 0;

            for (size_t tid = 0; tid < numThreads; ++tid) {
                workers.emplace_back([&, tid]() {
                    size_t chunkSize = std::max<size_t>(1, nCC / numThreads);
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

                        for (size_t ci = startIndex; ci < endIndex; ++ci) {
                            int cid = static_cast<int>(ci);
                            auto cc = std::make_unique<CcData>();

                            {
                                MARK_SCOPE_MEM("sb/gcc/rebuild");
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
                        auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - std::chrono::high_resolution_clock::now());
                        // chunk size adapt kept as in your code
                        (void)chunkDuration;
                    }
                    std::cout << "Thread " << tid << " built " << processed << " components (Gcc)" << std::endl;
                });
            }

            for (auto &t : workers) t.join();
        }

        {
            MARK_SCOPE_MEM("sb/phase/BCtrees");

            size_t numThreads = std::thread::hardware_concurrency();
            numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});

            std::vector<pthread_t> threads(numThreads);

            std::mutex workMutex;
            size_t nextIndex = 0;

            for (size_t tid = 0; tid < numThreads; ++tid) {
                pthread_attr_t attr;
                pthread_attr_init(&attr);

                // size_t stackSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;
                size_t stackSize = C.stackSize;
                if(pthread_attr_setstacksize(&attr, stackSize) != 0){
                    std::cout << "[Error] pthread_attr_setstacksize" << std::endl;
                }

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
        }

        allBlockData.resize(blockPreps.size());

        {
            MARK_SCOPE_MEM("sb/phase/BlockDataBuildAll");

            size_t numThreads2 = std::thread::hardware_concurrency();
            numThreads2 = std::min({(size_t)C.threads, (size_t)blockPreps.size(), numThreads2});
            std::vector<pthread_t> threads2(numThreads2);

            std::mutex workMutex2;
            size_t nextIndex2 = 0;

            for (size_t tid = 0; tid < numThreads2; ++tid) {
                pthread_attr_t attr;
                pthread_attr_init(&attr);

                // size_t stackSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;
                size_t stackSize = C.stackSize;

                if(pthread_attr_setstacksize(&attr, stackSize) != 0){
                    std::cout << "[Error] pthread_attr_setstacksize" << std::endl;
                }

                ThreadBlockBuildArgs* args = new ThreadBlockBuildArgs{
                    tid,
                    numThreads2,
                    blockPreps.size(),
                    &nextIndex2,
                    &workMutex2,
                    &blockPreps,
                    &allBlockData
                };

                int ret = pthread_create(&threads2[tid], &attr, worker_buildBlockData, args);
                if (ret != 0) {
                    std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                    delete args;
                }

                pthread_attr_destroy(&attr);
            }

            for (size_t tid = 0; tid < numThreads2; ++tid) {
                pthread_join(threads2[tid], nullptr);
            }
        }

        workItems.reserve(allBlockData.size());
        for (size_t i = 0; i < allBlockData.size(); ++i) {
            workItems.push_back({blockPreps[i].cc, blockPreps[i].bNode});
        }
    }
}

    {
        MEM_TIME_BLOCK("LOGIC: solve blocks (pthreads)");
        ACCUM_LOGIC();
        PROFILE_BLOCK("solve:: process blocks (pthreads, large stack)");
        MARK_SCOPE_MEM("sb/phase/SolveBlocks");

        std::vector<std::vector<std::pair<ogdf::node, ogdf::node>>> blockResults(workItems.size());

        size_t numThreads = std::thread::hardware_concurrency();
        numThreads = std::min({(size_t)C.threads, workItems.size(), numThreads});
        if (numThreads == 0) numThreads = 1;

        std::vector<pthread_t> threads(numThreads);
        std::mutex workMutex;
        size_t nextIndex = 0;

        for (size_t tid = 0; tid < numThreads; ++tid) {
            pthread_attr_t attr;
            pthread_attr_init(&attr);
            // size_t stackSize = 1024ULL * 1024ULL * 1024ULL * 20ULL; 
            size_t stackSize = C.stackSize;

            pthread_attr_setstacksize(&attr, stackSize);

            ThreadProcessArgs* args = new ThreadProcessArgs{
                tid,
                numThreads,
                workItems.size(),
                &nextIndex,
                &workMutex,
                &workItems,
                &allBlockData,
                &blockResults
            };

            int ret = pthread_create(&threads[tid], &attr, worker_processBlocks, args);
            if (ret != 0) {
                std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                delete args;
            }
            pthread_attr_destroy(&attr);
        }

        for (size_t tid = 0; tid < numThreads; ++tid) {
            pthread_join(threads[tid], nullptr);
        }

        for (const auto& candidates : blockResults) {
            for (const auto& p : candidates) {
                tryCommitSuperbubble(p.first, p.second);
            }
        }
    }
}



        void solve() {
            TIME_BLOCK("Finding superbubbles in blocks");
            findMiniSuperbubbles();
            solveStreaming();
        }
    }



    namespace snarls {
        namespace {

            thread_local std::vector<std::vector<std::string>>* tls_snarl_buffer = nullptr;

            static std::mutex g_snarls_mtx;

            inline void flushThreadLocalSnarls(std::vector<std::vector<std::string>>& local) {
                auto &C = ctx();
                std::lock_guard<std::mutex> lk(g_snarls_mtx);
                for (auto &s : local) {
                    snarlsFound += s.size() * (s.size() - 1) / 2;
                    C.snarls.insert(s);
                }
                local.clear();
            }

            struct pair_hash {
                size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
                    auto h1 = std::hash<std::string>{}(p.first);
                    auto h2 = std::hash<std::string>{}(p.second);
                    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
                }
            };

            std::unordered_set<std::pair<std::string, std::string>, pair_hash> tls_snarls_collector;
        }

        static void tryCommitSnarl(std::vector<std::string> s) {
            auto &C = ctx();
            // PROFILE_BLOCK("tryCommitSnarl");


            // if(std::count(s[0].begin(), s[0].end(), ':') == 0) {
            std::lock_guard<std::mutex> lk(g_snarls_mtx);

            snarlsFound += s.size()*(s.size()-1)/2;
            // }
            // std::cout << "S SIZE: " << s.size() << std::endl;
            // std::sort(s.begin(), s.end());
            // for (size_t i = 0; i < s.size(); i++)
            // {
            //     for (size_t j = i + 1; j < s.size(); j++)
            //     {

            //         std::string source = s[i], sink = s[j];
            //         if(source == "_trash+" || sink == "_trash+") continue;
            C.snarls.insert(std::move(s));
                    // C.snarls.insert({source, sink});
            //     }
            // }
            
            // C.snarls.push_back(s);


            // if(s.size()==2) {
            //     // string source = s[0], sink = s[1];
            //     // if(source>sink) std::swap(source, sink);
    
            //     // if(tls_snarls_collector.count({source, sink})) return 0;
    
            //     // tls_snarls_collector.insert({source, sink});
            //     // C.isEntry[source] = true;
            //     // C.isExit[sink] = true;
            //     C.snarls.push_back({source, sink});
            //     // std::cout << "Added " << C.node2name[source] << " " << C.node2name[sink] << " as superbubble\n";
            //     return true;
            // } else if(s.size()>2) {
            //     C.snarls.push_back(s);
            // }

        }




        // void addSnarl(std::string source, std::string sink) {
        //     // if (tls_snarls_collector) {
        //     //     tls_superbubble_collector->emplace_back(source, sink);
        //     //         return;
        //     //     }
        //     // Otherwise, commit directly to global state (sequential behavior)
        //     tryCommitSnarl(source, sink);


        //     // if(C.isEntry[source] || C.isExit[sink]) {
        //     //     std::cerr << ("Superbubble already exists for source %s and sink %s", C.node2name[source].c_str(), C.node2name[sink].c_str());
        //     //     return;
        //     // }
        //     // C.isEntry[source] = true;
        //     // C.isExit[sink] = true;
        //     // C.superbubbles.emplace_back(source, sink);

        // }

        void addSnarl(std::vector<std::string> s) {
            // if (tls_snarls_collector) {
            //     tls_superbubble_collector->emplace_back(source, sink);
            //         return;
            //     }
            // Otherwise, commit directly to global state (sequential behavior)
            // tryCommitSnarl(source, sink);
            if (tls_snarl_buffer) {
                tls_snarl_buffer->push_back(std::move(s));
                return;
            }
            tryCommitSnarl(std::move(s));

            // if(C.isEntry[source] || C.isExit[sink]) {
            //     std::cerr << ("Superbubble already exists for source %s and sink %s", C.node2name[source].c_str(), C.node2name[sink].c_str());
            //     return;
            // }
            // C.isEntry[source] = true;
            // C.isExit[sink] = true;
            // C.superbubbles.emplace_back(source, sink);

        }



        struct BlockData {
            std::unique_ptr<ogdf::Graph> Gblk;  
            ogdf::NodeArray<ogdf::node> toCc;
            ogdf::NodeArray<ogdf::node> nodeToOrig;
            ogdf::EdgeArray<ogdf::edge> edgeToOrig;
            

            std::unique_ptr<ogdf::StaticSPQRTree> spqr;
            // std::unique_ptr<ogdf::LCA> lcaSpqrTree;

            ogdf::NodeArray<ogdf::node> blkToSkel;


            std::unordered_map<ogdf::edge, ogdf::edge> skel2tree; // mapping from skeleton virtual edge to tree edge
            ogdf::NodeArray<ogdf::node> parent; // mapping from node to parent in SPQR tree, it is possible since it is rooted, 
                                                // parent of root is nullptr

            // ogdf::NodeArray<ogdf::node> nodeBlkToSkel;
            // ogdf::NodeArray<ogdf::node> edgeBlkToSkel;

            std::unordered_set<std::pair<std::string, std::string>, pair_hash> _adjInS;

            ogdf::node bNode {nullptr};

            bool isAcycic {true};
        
            BlockData() {}
        };

        struct CcData {
            std::unique_ptr<ogdf::Graph> Gcc;
            ogdf::NodeArray<ogdf::node> nodeToOrig;
            ogdf::EdgeArray<ogdf::edge> edgeToOrig;

            ogdf::NodeArray<bool> isTip;

            ogdf::NodeArray<int> degPlus, degMinus;


            ogdf::NodeArray<bool> isCutNode;
            ogdf::NodeArray<bool> isGoodCutNode;

            ogdf::NodeArray<ogdf::node> lastBad; // last bad adjacent block node for cut nodes
            ogdf::NodeArray<int> badCutCount; // number of adjacent bad blocks for cut nodes
            
            ogdf::EdgeArray<ogdf::edge> auxToOriginal;
            // ogdf::NodeArray<std::array<std::vector<ogdf::node>, 3>> cutToBlocks; // 0-all -, 1 - all +, 2 - mixed
            
            
            // ogdf::NodeArray<ogdf::node> toCopy;
            // ogdf::NodeArray<ogdf::node> toBlk;




            std::unique_ptr<ogdf::BCTree> bc;
            std::vector<BlockData> blocks;
        };


        EdgePartType getNodeEdgeType(ogdf::node v, ogdf::edge e) {
            auto &C = ctx();
            OGDF_ASSERT(v != nullptr && e != nullptr);
            OGDF_ASSERT(v->graphOf() == &C.G);
            OGDF_ASSERT(e->graphOf() == &C.G);
            if(e->source() == v) {
                return C._edge2types(e).first;
            } else if(e->target() == v) {
                return C._edge2types(e).second;
            } else {
                OGDF_ASSERT(false);
                return EdgePartType::NONE;
            }
        }

        // Given block "vB" and graph node "uG", find all outgoing edges from "uG" inside the block with out type "type"
        void getOutgoingEdgesInBlock(const CcData& cc, ogdf::node uG, ogdf::node vB, EdgePartType type, std::vector<ogdf::edge>& outEdges) {
            outEdges.clear();
            ogdf::node uB = cc.bc->repVertex(uG, vB);
            
            // std::cout << "Getting outgoing edges in block for graph node " << uG << " in block node " << vB << " " << uB->adjEntries.size() << std::endl;

            for(auto adjE : uB->adjEntries) {
                ogdf::edge eAux = adjE->theEdge();               // edge in auxiliary graph
                ogdf::edge eCc = cc.bc->original(eAux);          // edge in cc.Gcc
                ogdf::edge eG = cc.edgeToOrig[eCc];             // edge in original graph

                auto outType = getNodeEdgeType(cc.nodeToOrig[uG], eG);

                if(outType == type) {
                    outEdges.push_back(eCc);
                }


                // if(eOri->source() == uG) {
                //     EdgePartType outType = ctx()._edge2types(eOri).first;
                //     if(type == outType) {
                //         outEdges.push_back(eCc);
                //     } 
                // } else {
                //     EdgePartType outType = ctx()._edge2types(eOri).second;
                //     if(type == outType) {
                //         outEdges.push_back(eCc);
                //     }
                // }
            }
            // std::cout << "There are " << uB->adjEntries.size() << " adj entries in block node for graph node " << ctx().node2name[uG] << std::endl;
        }

        void getAllOutgoingEdgesOfType(const CcData& cc, ogdf::node uG, EdgePartType type, std::vector<ogdf::AdjElement*>& outEdges) {
            outEdges.clear();
            
            for(auto adjE : uG->adjEntries) {
                ogdf::edge eC = adjE->theEdge(); // edge in cc.Gcc
                ogdf::edge eOrig = cc.edgeToOrig[eC]; // corresponding edge in original graph

                if(eC->source() == uG) {
                    EdgePartType outType = ctx()._edge2types(eOrig).first;
                    if(type == outType) {
                        outEdges.push_back(adjE);
                    } 
                } else {
                    EdgePartType outType = ctx()._edge2types(eOrig).second;
                    if(type == outType) {
                        outEdges.push_back(adjE);
                    }
                }
            }
        }


        namespace SPQRsolve {
            struct EdgeDPState {
                node s{nullptr};      
                node t{nullptr};

                int localPlusS{0};
                int localPlusT{0};
                int localMinusT{0};
                int localMinusS{0};
            };

            struct EdgeDP {
                EdgeDPState down;   // value valid in  parent -> child  direction
                EdgeDPState up;     // value valid in  child -> parent direction
            };

            struct NodeDPState {
                std::vector<ogdf::node> GccCuts_last3; // last three cut nodes in Gcc
                // size_t cutsCnt{0};
            };


            void printAllEdgeStates(const ogdf::EdgeArray<EdgeDP> &edge_dp, BlockData &blk, const Graph &T) {
                auto& C = ctx();


                std::cout << "Edge dp states:" << std::endl;
                for(auto &e:T.edges) {
                    {
                        EdgeDPState state = edge_dp[e].down;
                        if(state.s && state.t) {
                            std::cout << "Edge " << e->source() << " -> " << e->target() << ": ";
                            std::cout << "s = " << C.node2name[blk.nodeToOrig[state.s]] << ", ";
                            std::cout << "t = " << C.node2name[blk.nodeToOrig[state.t]] << ", ";
                            std::cout << "localMinusS: " << state.localMinusS << ", ";
                            std::cout << "localMinusT: " << state.localMinusT << ", ";
                            std::cout << "localPlusS: " << state.localPlusS << ", ";
                            std::cout << "localPlusT: " << state.localPlusT << ", ";               
                            std::cout << std::endl;
                        }
                    }

                    {
                        EdgeDPState state = edge_dp[e].up;
                        if(state.s && state.t) {
                            std::cout << "Edge " << e->target() << " -> " << e->source() << ": ";
                            std::cout << "s = " << C.node2name[blk.nodeToOrig[state.s]] << ", ";
                            std::cout << "t = " << C.node2name[blk.nodeToOrig[state.t]] << ", ";
                            std::cout << "localMinusS: " << state.localMinusS << ", ";
                            std::cout << "localMinusT: " << state.localMinusT << ", ";
                            std::cout << "localPlusS: " << state.localPlusS << ", ";
                            std::cout << "localPlusT: " << state.localPlusT << ", ";               
                            std::cout << std::endl;
                        }
                    }
                }

            }

            void printAllStates(const ogdf::NodeArray<NodeDPState> &node_dp,  const Graph &T) {
                auto& C = ctx();

                std::cout << "Node dp states: " << std::endl;
                for(node v : T.nodes) {
                    std::cout << "Node " << v->index() << ", ";
                    // std::cout << "cutsCnt: " << node_dp[v].cutsCnt << ", ";
                    std::cout << "GccCuts_last3: " << node_dp[v].GccCuts_last3.size();
                    std::cout << std::endl;
                    
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
                PROFILE_FUNCTION();
                if(curr == nullptr) {
                    curr = spqr.rootNode();
                    parent = curr;
                    dfsSPQR_order(spqr, edge_order, node_order, curr, parent);
                    return;
                }

                node_order.push_back(curr);
                for (adjEntry adj : curr->adjEntries) {
                    node child = adj->twinNode();
                    if (child == parent) continue;
                    dfsSPQR_order(spqr, edge_order, node_order, child, curr, adj->theEdge());
                }
                if(curr!=parent) edge_order.push_back(e);
            }

            void processEdge(ogdf::edge curr_edge, ogdf::EdgeArray<EdgeDP> &dp, const CcData &cc, BlockData &blk) {
                //PROFILE_FUNCTION();
                auto& C = ctx();
                            
                EdgeDPState &state = dp[curr_edge].down;
                EdgeDPState &back_state = dp[curr_edge].up;
                
                const StaticSPQRTree &spqr = *blk.spqr;
                            
                ogdf::node A = curr_edge->source();
                ogdf::node B = curr_edge->target();

                // std::cout << "PROCESSING " << A << "->" << B << " EDGE\n";
                
                state.localPlusS = 0;
                state.localPlusT = 0;
                state.localMinusT = 0;
                state.localMinusS = 0;
                
                const Skeleton &skel = spqr.skeleton(B);
                const Graph &skelGraph = skel.getGraph();


                auto mapSkeletonToGlobal = [&](ogdf::node vSkel) -> ogdf::node {
                    if (!vSkel) return nullptr;

                    ogdf::node vBlk  = skel.original(vSkel);
                    if (!vBlk) return nullptr;

                    ogdf::node vCc   = blk.toCc[vBlk];
                    if (!vCc) return nullptr;

                    return cc.nodeToOrig[vCc];
                };


                for(edge e : skelGraph.edges) {
                    node u = e->source();
                    node v = e->target();

                    auto D = skel.twinTreeNode(e);


                    if(D == A) {
                        ogdf::node vBlk = skel.original(v);
                        ogdf::node uBlk = skel.original(u);
                        
                        state.s = back_state.s = vBlk;
                        state.t = back_state.t = uBlk;                        
                        break;
                    }
                }



                for(edge e : skelGraph.edges) {
                    node u = e->source();
                    node v = e->target();

                    
                    ogdf::node uBlk = skel.original(u);
                    ogdf::node vBlk = skel.original(v);


                    
                    
                    if(!skel.isVirtual(e)) {
                        ogdf::edge eG = blk.edgeToOrig[skel.realEdge(e)];

                        ogdf::node uG = eG->source();
                        ogdf::node vG = eG->target();
    

                        // std::cout << "Type of " << C.node2name[uG] << "-" << C.node2name[vG] << " is " << (getNodeEdgeType(uG, eG) == EdgePartType::PLUS ? "+" : "-") << " - " << (getNodeEdgeType(vG, eG) == EdgePartType::PLUS ? "+" : "-") << "\n";
                        if(uG == blk.nodeToOrig[state.s]) {
                            auto t = getNodeEdgeType(uG, eG);
                            if(t== EdgePartType::PLUS) {
                                state.localPlusS++;
                            } else if(t == EdgePartType::MINUS) {
                                state.localMinusS++;
                            }
                        }

                        if(vG == blk.nodeToOrig[state.s]) {
                            auto t = getNodeEdgeType(vG, eG);
                            if(t== EdgePartType::PLUS) {
                                state.localPlusS++;
                            } else if(t == EdgePartType::MINUS) {
                                state.localMinusS++;
                            }
                        }

                        if(uG == blk.nodeToOrig[state.t]) {
                            auto t = getNodeEdgeType(uG, eG);
                            if(t== EdgePartType::PLUS) {
                                state.localPlusT++;
                            } else if(t == EdgePartType::MINUS) {
                                state.localMinusT++;
                            }
                        }

                        if(vG == blk.nodeToOrig[state.t]) {
                            auto t = getNodeEdgeType(vG, eG);
                            if(t== EdgePartType::PLUS) {
                                state.localPlusT++;
                            } else if(t == EdgePartType::MINUS) {
                                state.localMinusT++;
                            }
                        }

                        continue;
                    }
                    
                    auto D = skel.twinTreeNode(e);
                    

                    if(D == A) {
                        continue;
                    }


                    edge treeE = blk.skel2tree.at(e);
                    OGDF_ASSERT(treeE != nullptr);



                    const EdgeDPState child = dp[treeE].down;

                    ogdf::node nS = child.s;
                    ogdf::node nT = child.t;

                    // ogdf::node nA = skelToNew[blk.blkToSkel[child.s]];
                    // ogdf::node nB = skelToNew[blk.blkToSkel[child.t]];

                    

                    if(state.s == child.s) {
                        // std::cout << "Adding " << child.localPlusS << " plus to " << C.node2name[blk.nodeToOrig[state.s]] << std::endl; 
                        state.localPlusS+=child.localPlusS;
                        // std::cout << "Adding " << child.localMinusS << " minus to " << C.node2name[blk.nodeToOrig[state.s]] << std::endl;
                        state.localMinusS+=child.localMinusS;
                    } 

                    if(state.s == child.t) {
                        state.localPlusS+=child.localPlusT;
                        // std::cout << "Adding " << child.localPlusT << " plus to " << C.node2name[blk.nodeToOrig[state.s]] << std::endl; 
                        state.localMinusS+=child.localMinusT;
                        // std::cout << "Adding " << child.localMinusT << " minus to " << C.node2name[blk.nodeToOrig[state.s]] << std::endl; 

                    } 

                    if(state.t == child.t) {
                        state.localPlusT+=child.localPlusT;
                        // std::cout << "Adding " << child.localPlusT << " plus to " << C.node2name[blk.nodeToOrig[state.t]] << std::endl;
                        state.localMinusT+=child.localMinusT;
                        // std::cout << "Adding " << child.localMinusT << " minus to " << C.node2name[blk.nodeToOrig[state.t]] << std::endl; 

                    } 

                    if(state.t == child.s) {
                        state.localPlusT+=child.localPlusS;
                        // std::cout << "Adding " << child.localPlusS << " plus to " << C.node2name[blk.nodeToOrig[state.t]] << std::endl;
                        state.localMinusT+=child.localMinusS;
                        // std::cout << "Adding " << child.localMinusS << " plus to " << C.node2name[blk.nodeToOrig[state.t]] << std::endl;
                    } 

                }

            }


            void processNode(node curr_node, EdgeArray<EdgeDP> &edge_dp, const CcData &cc, BlockData &blk) {
                auto& C = ctx();
                ogdf::node A = curr_node;
                
                const Graph &T = blk.spqr->tree();
                
                const StaticSPQRTree &spqr = *blk.spqr;
                            

                const Skeleton &skel = spqr.skeleton(A);
                const Graph &skelGraph = skel.getGraph();

                
                Graph newGraph;

                NodeArray<node> skelToNew(skelGraph, nullptr);
                for (node v : skelGraph.nodes) skelToNew[v] = newGraph.newNode();
                NodeArray<node> newToSkel(newGraph, nullptr);
                for (node v : skelGraph.nodes) newToSkel[skelToNew[v]] = v;

                for (ogdf::node h : skelGraph.nodes) {
                    ogdf::node vB = skel.original(h);
                }


                for (ogdf::node h : skelGraph.nodes) {
                    ogdf::node vB = skel.original(h);
                    blk.blkToSkel[vB] = h;
                }
            

                // GraphIO::drawGraph(skelGraph, "skel_" + to_string(curr_node->index()));

                
                NodeArray<int> localPlusDeg(newGraph, 0), localMinusDeg(newGraph, 0);
                

                EdgeArray<bool> isVirtual(newGraph, false);
                EdgeArray<EdgeDPState*> edgeToDp(newGraph, nullptr);
                EdgeArray<EdgeDPState*> edgeToDpR(newGraph, nullptr);
                EdgeArray<node> edgeChild(newGraph, nullptr);



                std::vector<edge> virtualEdges;


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
                    return cc.nodeToOrig[vCc];
                };






                // auto printDegrees = [&]() {
                //     for(node vN:newGraph.nodes) {
                //         node vG = mapNewToGlobal(vN);
                //     }
                // };


                // std::cout << "Processing node " << curr_node << "\n";

                // Building new graph
                {
                    //PROFILE_BLOCK("processNode:: build oriented local graph");
                    for(edge e : skelGraph.edges) {
                        node u = e->source();
                        node v = e->target();

                        
                        ogdf::node uBlk = skel.original(u);
                        ogdf::node vBlk = skel.original(v);


                        ogdf::node uG = blk.nodeToOrig[uBlk];
                        ogdf::node vG = blk.nodeToOrig[vBlk];

                        
                        node nU = skelToNew[u];
                        node nV = skelToNew[v];
                        

                        if(!skel.isVirtual(e)) {
                            ogdf::edge eG = blk.edgeToOrig[skel.realEdge(e)];

                            uG = eG->source();
                            vG = eG->target();

                            auto newEdge = newGraph.newEdge(nU, nV);

                            // std::cout << "Type of " << C.node2name[uG] << "-" << C.node2name[vG] << " is " << (getNodeEdgeType(uG, eG) == EdgePartType::PLUS ? "+" : "-") << " - " << (getNodeEdgeType(vG, eG) == EdgePartType::PLUS ? "+" : "-") << "\n";


                            isVirtual[newEdge] = false;

                            if(blk.nodeToOrig[skel.original(newToSkel[nU])] == uG) {
                                localPlusDeg[nU]+=(getNodeEdgeType(uG, eG) == EdgePartType::PLUS);
                                localMinusDeg[nU]+=(getNodeEdgeType(uG, eG) == EdgePartType::MINUS);
                                localPlusDeg[nV]+=(getNodeEdgeType(vG, eG) == EdgePartType::PLUS);
                                localMinusDeg[nV]+=(getNodeEdgeType(vG, eG) == EdgePartType::MINUS);
                            } else {
                                localPlusDeg[nU]+=(getNodeEdgeType(vG, eG) == EdgePartType::PLUS);
                                localMinusDeg[nU]+=(getNodeEdgeType(vG, eG) == EdgePartType::MINUS);
                                localPlusDeg[nV]+=(getNodeEdgeType(uG, eG) == EdgePartType::PLUS);
                                localMinusDeg[nV]+=(getNodeEdgeType(uG, eG) == EdgePartType::MINUS);
                            }



                            continue;
                        }


                        // std::cout << "Type of " << C.node2name[uG] << "-" << C.node2name[vG] << " is virtual\n";


                        
                        auto B = skel.twinTreeNode(e);
                        
                        edge treeE = blk.skel2tree.at(e);
                        OGDF_ASSERT(treeE != nullptr);



                        EdgeDPState *child = (B == blk.parent(A) ? &edge_dp[treeE].up : &edge_dp[treeE].down);
                        EdgeDPState *edgeToUpdate = (B == blk.parent(A) ? &edge_dp[treeE].down : &edge_dp[treeE].up);

                        ogdf::node nS = mapBlockToNew(child->s);
                        ogdf::node nT = mapBlockToNew(child->t);



                        edge newEdge = newGraph.newEdge(nS, nT);
                        
                        isVirtual[newEdge] = true;

                        virtualEdges.push_back(newEdge);

                        edgeToDp[newEdge] = edgeToUpdate;
                        edgeToDpR[newEdge] = child;
                        edgeChild[newEdge] = B;


                        if(nS == nU && nT == nV) {
                            localMinusDeg[nS] += child->localMinusT;
                            localPlusDeg[nS] += child->localPlusT;

                            localMinusDeg[nT] += child->localMinusS;
                            localPlusDeg[nT] += child->localPlusS;
                        } else {
                            localMinusDeg[nS] += child->localMinusS;
                            localPlusDeg[nS] += child->localPlusS;

                            localMinusDeg[nT] += child->localMinusT;
                            localPlusDeg[nT] += child->localPlusT;
                        }
                    //     for(auto &v:newGraph.nodes) {
                    //        std::cout << C.node2name[blk.nodeToOrig[skel.original(newToSkel[v])]] << " has " << localPlusDeg[v] << " local plus and " << localMinusDeg[v] << " local minus. \n";
                    //    }
                    }


                }

                // std::cout << "Built new graph for " << curr_node << "\n";


                // updating local degrees of poles of states going into A
                {
                    // std::cout << "updating local degrees of poles of states going into A " << curr_node << "\n";

                    //PROFILE_BLOCK("processNode:: update DP local degrees at poles");
                    for(edge e:virtualEdges) {
                        // if(!isVirtual[e]) continue;
                        node vN = e->source();
                        node uN = e->target();

                        EdgeDPState *BA = edgeToDp[e];
                        EdgeDPState *AB = edgeToDpR[e];

                        BA->localPlusS = localPlusDeg[mapBlockToNew(BA->s)] - AB->localPlusS; 
                        BA->localPlusT = localPlusDeg[mapBlockToNew(BA->t)] - AB->localPlusT; 

                        BA->localMinusS = localMinusDeg[mapBlockToNew(BA->s)] - AB->localMinusS; 
                        BA->localMinusT = localMinusDeg[mapBlockToNew(BA->t)] - AB->localMinusT; 
                    }
                    // std::cout << "UPDATED local degrees of poles of states going into A " << curr_node << "\n";

                }

                // for(auto &v:newGraph.nodes) {
                //     std::cout << C.node2name[blk.nodeToOrig[skel.original(newToSkel[v])]] << " has " << localPlusDeg[v] << " local plus and " << localMinusDeg[v] << " local minus. \n";
                // }

            }
    
            


            void solveS(ogdf::node sNode, NodeArray<SPQRsolve::NodeDPState> &node_dp, ogdf::EdgeArray<EdgeDP> &dp, BlockData& blk, const CcData& cc) {
                PROFILE_FUNCTION();
                const Skeleton& skel = blk.spqr->skeleton(sNode);
                const Graph& skelGraph = skel.getGraph();
                const Graph& T = blk.spqr->tree();
                
                // std::cout << "Solving S node.." << std::endl;

                std::vector<ogdf::node> nodesInOrderGcc; // In Gcc
                std::vector<ogdf::node> nodesInOrderSkel; // In skel
                
                ogdf::EdgeArray<EdgeDPState*> skelToState(T);

                // std::vector<EdgeDP> edgeT; // for i, nullptr if edge between i and i+1 is real, EdgeDp* if not 




                for(edge e : skelGraph.edges) {
                    if(!skel.isVirtual(e)) continue;
                    auto B = skel.twinTreeNode(e);
                    edge treeE = blk.skel2tree.at(e);

                    EdgeDPState *child = (B == blk.parent(sNode) ? &dp[treeE].up : &dp[treeE].down);
                    skelToState[treeE] = child;
                }

                std::vector<ogdf::edge> adjEdgesG_; // edges are in G
                std::vector<adjEntry> adjEntriesSkel; // edges are in skel



                {
                    std::function<void(ogdf::node, ogdf::node)> dfs = [&](ogdf::node u, ogdf::node prev) {          
                        nodesInOrderGcc.push_back(blk.toCc[skel.original(u)]);
                        nodesInOrderSkel.push_back(u);
                            
                        for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {

                            if(adj->twinNode() == prev) continue;

                            if(adj->twinNode() == skelGraph.firstNode() && u != skelGraph.firstNode()) {
                                if(skel.realEdge(adj->theEdge())) {
                                    adjEdgesG_.push_back(blk.edgeToOrig[skel.realEdge(adj->theEdge())]);
                                } else {
                                    adjEdgesG_.push_back(nullptr);
                                }

                                adjEntriesSkel.push_back(adj);
                            }

                            if(adj->twinNode() == skelGraph.firstNode() || adj->twinNode() == prev) continue;
                            {
                                // if(skel.realEdge(adj->theEdge())) {
                                //     node B = (skel.twinTreeNode(adj->theEdge()));
                                //     if(blk.parent(sNode) == B) skelToState[adj->theEdge()] =
                                //     else  skelToState[adj->theEdge()] =
                                // } else {
                                //     skelToState[adj->theEdge()] = nullptr;
                                // }
                            }

                            if(skel.realEdge(adj->theEdge())) {
                                adjEdgesG_.push_back(blk.edgeToOrig[skel.realEdge(adj->theEdge())]);
                            } else {
                                adjEdgesG_.push_back(nullptr);
                            }

                            adjEntriesSkel.push_back(adj);
                            dfs(adj->twinNode(), u);
                        }
                        
                    };
                    
                    dfs(skelGraph.firstNode(), skelGraph.firstNode()->firstAdj()->twinNode());
                }

                std::vector<bool> cuts(nodesInOrderGcc.size(), false);

                std::vector<std::string> res;


                for (size_t i = 0; i < nodesInOrderGcc.size(); i++) {
                    std::string s = ctx().node2name[cc.nodeToOrig[nodesInOrderGcc[i]]], t = ctx().node2name[cc.nodeToOrig[nodesInOrderGcc[(i+1)%nodesInOrderGcc.size()]]];
                    blk._adjInS.insert({ s, t});
                }


                for (size_t i = 0; i < nodesInOrderGcc.size(); i++) {
                    auto uGcc = nodesInOrderGcc[i];
                    auto uSkel = nodesInOrderSkel[i];

                    std::vector<edge> adjEdgesSkel = {adjEntriesSkel[(i-1+adjEntriesSkel.size())%adjEntriesSkel.size()]->theEdge(), adjEntriesSkel[i]->theEdge()};
                    std::vector<ogdf::edge> adjEdgesG = {adjEdgesG_[(i-1+adjEdgesG_.size())%adjEdgesG_.size()], adjEdgesG_[i]};

                    bool nodeIsCut = ((cc.isCutNode[uGcc] && cc.badCutCount[uGcc] == 1) || (!cc.isCutNode[uGcc]));

                    EdgePartType t0 = EdgePartType::NONE;
                    EdgePartType t1 = EdgePartType::NONE;

                    if(!skel.isVirtual(adjEdgesSkel[0])) {
                        t0 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesG[0]);
                    } else {
                        edge treeE0 = blk.skel2tree.at(adjEdgesSkel[0]);
                        
                        EdgeDPState* state0 = skelToState[treeE0];
            
                        if(blk.toCc[state0->s] == uGcc) {
                            // take state0.s
                            if(state0->localMinusS>0 && state0->localPlusS>0) {}
                            else if(state0->localMinusS == 0 && state0->localPlusS>0) {
                                t0 = EdgePartType::PLUS;
                            } else if(state0->localMinusS > 0 && state0->localPlusS==0) {
                                t0 = EdgePartType::MINUS;
                            } else {
                                assert(false);
                            }
                        } else {
                            // take state0.t
                            if(state0->localMinusT>0 && state0->localPlusT>0) {}
                            else if(state0->localMinusT == 0 && state0->localPlusT>0) {
                                t0 = EdgePartType::PLUS;
                            } else if(state0->localMinusT > 0 && state0->localPlusT==0) {
                                t0 = EdgePartType::MINUS;
                            } else {
                                assert(false);
                            }
                        }
                    }

                    if(!skel.isVirtual(adjEdgesSkel[1])) {
                        t1 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesG[1]);
                    } else {
                        edge treeE1 = blk.skel2tree.at(adjEdgesSkel[1]);
                        
                        EdgeDPState* state1 = skelToState[treeE1];
            
                        if(blk.toCc[state1->s] == uGcc) {
                            // take state1.s
                            if(state1->localMinusS>0 && state1->localPlusS>0) {}
                            else if(state1->localMinusS == 0 && state1->localPlusS>0) {
                                t1 = EdgePartType::PLUS;
                            } else if(state1->localMinusS > 0 && state1->localPlusS==0) {
                                t1 = EdgePartType::MINUS;
                            } else {
                                assert(false);
                            }
                        } else {
                            // take state1.t
                            if(state1->localMinusT>0 && state1->localPlusT>0) {}
                            else if(state1->localMinusT == 0 && state1->localPlusT>0) {
                                t1 = EdgePartType::PLUS;
                            } else if(state1->localMinusT > 0 && state1->localPlusT==0) {
                                t1 = EdgePartType::MINUS;
                            } else {
                                assert(false);
                            }
                        }
                    }

                    nodeIsCut &= (t0 != EdgePartType::NONE && t1!= EdgePartType::NONE && t0 != t1);


                    // if(!skel.isVirtual(adjEdgesSkel[0]) && !skel.isVirtual(adjEdgesSkel[1])) {
                    //     EdgePartType t0 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesG[0]);
                    //     EdgePartType t1 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesG[1]);


                    //     nodeIsCut &= t0 != t1;
                    // } else if(!skel.isVirtual(adjEdgesSkel[0]) && skel.isVirtual(adjEdgesSkel[1])) {
                    //     EdgePartType t0 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesG[0]);
                    //     EdgePartType t1 = EdgePartType::NONE;

                    //     edge treeE1 = blk.skel2tree.at(adjEdgesSkel[1]);

                    //     EdgeDPState* state1 = skelToState[treeE1];

                    //     if(blk.toCc[state1->s] == uGcc) {
                    //         // take state1.s
                    //         if(state1->localMinusS>0 && state1->localPlusS>0) {}
                    //         else if(state1->localMinusS == 0 && state1->localPlusS>0) {
                    //             t1 = EdgePartType::PLUS;
                    //         } else if(state1->localMinusS > 0 && state1->localPlusS==0) {
                    //             t1 = EdgePartType::MINUS;
                    //         } else {
                    //             assert(false);
                    //         }
                    //     } else {
                    //         // take state1.t
                    //         if(state1->localMinusT>0 && state1->localPlusT>0) {}
                    //         else if(state1->localMinusT == 0 && state1->localPlusT>0) {
                    //             t1 = EdgePartType::PLUS;
                    //         } else if(state1->localMinusT > 0 && state1->localPlusT==0) {
                    //             t1 = EdgePartType::MINUS;
                    //         } else {
                    //             assert(false);
                    //         }
                    //     }

                    //     nodeIsCut &= (t0 != EdgePartType::NONE && t1!= EdgePartType::NONE && t0 != t1);
                    // } else if(skel.isVirtual(adjEdgesSkel[0]) && !skel.isVirtual(adjEdgesSkel[1])) {
                    //     // first is virtual, second is real
                    //     // if(toPrint)
                    //     // std::cout << "vir - real, " << std::endl;
                    //     EdgePartType t0 = EdgePartType::NONE;
                    //     EdgePartType t1 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesG[1]);
                        

                    //     // std::cout << "33" << std::endl;
                    //     edge treeE0 = blk.skel2tree.at(adjEdgesSkel[0]);
                        
                    //     EdgeDPState* state0 = skelToState[treeE0];
                        
                    //     // std::cout << "44" << std::endl;
                    //     // std::cout << "S-" << skelToState[treeE1]->localMinusS << " " << "S+"  << skelToState[treeE1]->localPlusS << " " << "T-" << skelToState[treeE1]->localMinusT << " " << "T+" << skelToState[treeE1]->localPlusT << std::endl;
                    //     // node B = skel.twinTreeNode(adjEdges[1])

                    //     if(blk.toCc[state0->s] == uGcc) {
                    //         // take state0.s
                    //         if(state0->localMinusS>0 && state0->localPlusS>0) {}
                    //         else if(state0->localMinusS == 0 && state0->localPlusS>0) {
                    //             t0 = EdgePartType::PLUS;
                    //         } else if(state0->localMinusS > 0 && state0->localPlusS==0) {
                    //             t0 = EdgePartType::MINUS;
                    //         } else {
                    //             assert(false);
                    //         }
                    //     } else {
                    //         // take state0.t
                    //         if(state0->localMinusT>0 && state0->localPlusT>0) {}
                    //         else if(state0->localMinusT == 0 && state0->localPlusT>0) {
                    //             t0 = EdgePartType::PLUS;
                    //         } else if(state0->localMinusT > 0 && state0->localPlusT==0) {
                    //             t0 = EdgePartType::MINUS;
                    //         } else {
                    //             assert(false);
                    //         }
                    //     }

                        


                    //     nodeIsCut &= (t0 != EdgePartType::NONE && t1!= EdgePartType::NONE && t0 != t1);
                    // } else if(skel.isVirtual(adjEdgesSkel[0]) && skel.isVirtual(adjEdgesSkel[1])) {
                    //     // both edges are virtual
                    //     // if(toPrint)
                    //     // std::cout << ctx().node2name[cc.nodeToOrig[uGcc]] << " vir - vir" << std::endl;
                        
                    //     EdgePartType t0 = EdgePartType::NONE;
                    //     EdgePartType t1 = EdgePartType::NONE;
                        
                    //     edge treeE0 = blk.skel2tree.at(adjEdgesSkel[0]);
                    //     edge treeE1 = blk.skel2tree.at(adjEdgesSkel[1]);
                        
                    //     EdgeDPState* state0 = skelToState[treeE0];
                    //     EdgeDPState* state1 = skelToState[treeE1];

                    //     // if(toPrint) {
                    //     //     std::cout << "S-" << skelToState[treeE0]->localMinusS << " " << "S+"  << skelToState[treeE0]->localPlusS << " " << "T-" << skelToState[treeE0]->localMinusT << " " << "T+" << skelToState[treeE0]->localPlusT << std::endl;
                    //     //     std::cout << "S-" << skelToState[treeE1]->localMinusS << " " << "S+"  << skelToState[treeE1]->localPlusS << " " << "T-" << skelToState[treeE1]->localMinusT << " " << "T+" << skelToState[treeE1]->localPlusT << std::endl;
                    //     // // node B = skel.twinTreeNode(adjEdges[1])
                    //     // }
                    //     assert(blk.nodeToOrig[state0->s] == cc.nodeToOrig[uGcc] || cc.nodeToOrig[state0->t] == cc.nodeToOrig[uGcc]);
                    //     assert(blk.nodeToOrig[state1->s] == cc.nodeToOrig[uGcc] || cc.nodeToOrig[state1->t] == cc.nodeToOrig[uGcc]);

                    //     // std::cout << ctx().node2name[cc.nodeToOrig[state0->s]] << " " << ctx().node2name[cc.nodeToOrig[state0->t]] << std::endl;
                    //     // std::cout << ctx().node2name[state1->s] << " " << ctx().node2name[cc.nodeToOrig[state1->t]] << std::endl;

                    //     if(blk.toCc[state0->s] == uGcc) {
                    //         // take state0.s
                    //         if(state0->localMinusS>0 && state0->localPlusS>0) {}
                    //         else if(state0->localMinusS == 0 && state0->localPlusS>0) {
                    //             t0 = EdgePartType::PLUS;
                    //         } else if(state0->localMinusS > 0 && state0->localPlusS==0) {
                    //             t0 = EdgePartType::MINUS;
                    //         } else {
                    //             assert(false);
                    //         }
                    //     } else {
                    //         // take state0.t
                    //         if(state0->localMinusT>0 && state0->localPlusT>0) {}
                    //         else if(state0->localMinusT == 0 && state0->localPlusT>0) {
                    //             t0 = EdgePartType::PLUS;
                    //         } else if(state0->localMinusT > 0 && state0->localPlusT==0) {
                    //             t0 = EdgePartType::MINUS;
                    //         } else {
                    //             assert(false);
                    //         }
                    //     }

                    //     if(blk.toCc[state1->s] == uGcc) {
                    //         // take state1.s
                    //         if(state1->localMinusS>0 && state1->localPlusS>0) {}
                    //         else if(state1->localMinusS == 0 && state1->localPlusS>0) {
                    //             t1 = EdgePartType::PLUS;
                    //         } else if(state1->localMinusS > 0 && state1->localPlusS==0) {
                    //             t1 = EdgePartType::MINUS;
                    //         } else {
                    //             assert(false);
                    //         }
                    //     } else {
                    //         // take state1.t
                    //         if(state1->localMinusT>0 && state1->localPlusT>0) {}
                    //         else if(state1->localMinusT == 0 && state1->localPlusT>0) {
                    //             t1 = EdgePartType::PLUS;
                    //         } else if(state1->localMinusT > 0 && state1->localPlusT==0) {
                    //             t1 = EdgePartType::MINUS;
                    //         } else {
                    //             assert(false);
                    //         }
                    //     }

                    //     // std::cout << (t0 == EdgePartType::NONE ? "NONE" : (t0 == EdgePartType::PLUS ? "PLUS" : "MINUS")) << " - " << (t1 == EdgePartType::NONE ? "NONE" : (t1 == EdgePartType::PLUS ? "PLUS" : "MINUS")) << std::endl;
                        
                    //     nodeIsCut &= (t0 != EdgePartType::NONE && t1!= EdgePartType::NONE && t0 != t1);
                    // }


                    if(nodeIsCut) { 
                        // std::cout << node_dp[sNode].cutsCnt << std::endl;
                        if(node_dp[sNode].GccCuts_last3.size() < 3) {
                            node_dp[sNode].GccCuts_last3.push_back(uGcc);
                        }
                        // node_dp[sNode].cutsCnt++;
                        // if(toPrint)
                        // std::cout << "Found cut at " << ctx().node2name[cc.nodeToOrig[uGcc]] << std::endl;
                        // std::cout << node_dp[sNode].cutsCnt << std::endl;
                        if(!skel.isVirtual(adjEdgesSkel[0])) {
                            EdgePartType t0 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesG[0]);

                            res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] + (t0 == EdgePartType::PLUS ? "+" : "-"));
                        } else {
                            edge treeE0 = blk.skel2tree.at(adjEdgesSkel[0]);
                            EdgeDPState* state0 = skelToState[treeE0];

                            if(uGcc == blk.toCc[state0->s]) {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] + (state0->localPlusS > 0 ? "+" : "-"));
                            } else {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] + (state0->localPlusT > 0 ? "+" : "-"));
                            }
                        }


                        if(!skel.isVirtual(adjEdgesSkel[1])) {
                            EdgePartType t1 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesG[1]);

                            res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] + (t1 == EdgePartType::PLUS ? "+" : "-"));
                        } else {
                            edge treeE1 = blk.skel2tree.at(adjEdgesSkel[1]);
                            EdgeDPState* state1 = skelToState[treeE1];

                            if(uGcc == blk.toCc[state1->s]) {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] + (state1->localPlusS > 0 ? "+" : "-"));
                            } else {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] + (state1->localPlusT > 0 ? "+" : "-"));
                            }
                        }

                    }

                    // if(toPrint)
                    // std::cout << "Node " << ctx().node2name[cc.nodeToOrig[uGcc]] << " " << (nodeIsCut ? "cut" : "not cut") << std::endl;
                }
                

                // std::cout << "done solving S" << std::endl;

                
                assert(res.size()%2==0);

                if(res.size()>2) {
                    for (size_t i = 1; i < res.size(); i+=2)
                    {
                        std::vector<std::string> v = {/*"S"+*/res[i], /*"S"+*/res[(i+1)%res.size()]};
                        // std::cout << "S node snarl: ";
                        // for(auto &s:v) std::cout << s << " ";
                        // std::cout << std::endl;
                        addSnarl(v);

                        // std::cout << res[i] <<":" << res[(i+1)%res.size()] << std::endl; 
                    }
                }
                

                // for(auto &v:res) {
                //     std::cout << v << "  ";
                // }
                // std::cout << std::endl;

            }

            void solveP(ogdf::node pNode, NodeArray<SPQRsolve::NodeDPState> &node_dp, ogdf::EdgeArray<EdgeDP> &edge_dp, BlockData& blk, const CcData& cc) {
                PROFILE_FUNCTION();
                const Skeleton& skel = blk.spqr->skeleton(pNode);
                const Graph& skelGraph = skel.getGraph();

                std::vector<ogdf::adjEntry> edgeOrdering; // pole0Skel -> pole1Skel

                node pole0Skel = nullptr, pole1Skel = nullptr;
                {
                    auto it = skelGraph.nodes.begin();
                    if (it != skelGraph.nodes.end()) pole0Skel = *it++;
                    if (it != skelGraph.nodes.end()) pole1Skel = *it;
                }                


                node pole0Blk = skel.original(pole0Skel), pole1Blk = skel.original(pole1Skel);
                node pole0Gcc = blk.toCc[pole0Blk], pole1Gcc = blk.toCc[pole1Blk];

                // if(ctx().node2name[cc.nodeToOrig[pole0Gcc]] == "3497" || ctx().node2name[cc.nodeToOrig[pole1Gcc]] == "3497") {
                //     std::cout << "Processing P node with pole " << ctx().node2name[cc.nodeToOrig[pole0Gcc]] << " and " << ctx().node2name[cc.nodeToOrig[pole1Gcc]] << std::endl;
                // }


                for (ogdf::adjEntry adj = pole0Skel->firstAdj(); adj; adj = adj->succ()) {
                    edgeOrdering.push_back(adj);
                }

                if(cc.isCutNode[pole0Gcc]) {
                    if(cc.badCutCount[pole0Gcc] >= 2 || (cc.badCutCount[pole0Gcc] == 1 && cc.lastBad[pole0Gcc] != blk.bNode)) return;
                }

                if(cc.isCutNode[pole1Gcc]) {
                    if(cc.badCutCount[pole1Gcc] >= 2 || (cc.badCutCount[pole1Gcc] == 1 && cc.lastBad[pole1Gcc] != blk.bNode)) return;
                }

                for (size_t i = 0; i < edgeOrdering.size(); i++) {
                    edge eSkel = edgeOrdering[i]->theEdge();
                    if(!skel.isVirtual(eSkel)) continue;
                    node B = (blk.skel2tree[eSkel]->source() == pNode ? blk.skel2tree[eSkel]->target() : blk.skel2tree[eSkel]->source());
                    auto state = (blk.parent(pNode) == B ? edge_dp[blk.skel2tree[eSkel]].up : edge_dp[blk.skel2tree[eSkel]].down);
                    if(state.s == pole0Blk) {
                        if((state.localMinusS>0) + (state.localPlusS>0) == 2) return;
                    } else {
                        if((state.localMinusT>0) + (state.localPlusT>0) == 2) return;
                    }

                    if(state.s == pole1Blk) {
                        if((state.localMinusS>0) + (state.localPlusS>0) == 2) return;
                    } else {
                        if((state.localMinusT>0) + (state.localPlusT>0) == 2) return;
                    }
                }
                


                // std::cout << ctx().node2name[cc.nodeToOrig[pole0Gcc]] << " " << ctx().node2name[cc.nodeToOrig[pole1Gcc]] << std::endl;
                // std::cout << "P NODE CHECKING.." << std::endl;
                
                for(auto &left:{EdgePartType::PLUS, EdgePartType::MINUS}) {
                    for(auto &right:{EdgePartType::PLUS, EdgePartType::MINUS}) {
                        std::vector<ogdf::edge> leftPart, rightPart;
                        for (size_t i = 0; i < edgeOrdering.size(); i++) {
                            edge eSkel = edgeOrdering[i]->theEdge();
                            if(!skel.isVirtual(eSkel)) {
                                EdgePartType l=getNodeEdgeType(cc.nodeToOrig[pole0Gcc], blk.edgeToOrig[skel.realEdge(eSkel)]), r = getNodeEdgeType(cc.nodeToOrig[pole1Gcc], blk.edgeToOrig[skel.realEdge(eSkel)]);
                                if(l == left) leftPart.push_back(eSkel);
                                if(r == right) rightPart.push_back(eSkel);
                            } else {    
                                node B = (blk.skel2tree[eSkel]->source() == pNode ? blk.skel2tree[eSkel]->target() : blk.skel2tree[eSkel]->source());
                                auto state = (blk.parent(pNode) == B ? edge_dp[blk.skel2tree[eSkel]].up : edge_dp[blk.skel2tree[eSkel]].down);

                                
                                EdgePartType l, r;
                                if(state.s == pole0Blk) {
                                    l = (state.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                                } else {
                                    l = (state.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                                }

                                if(state.s == pole1Blk) {
                                    r = (state.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                                } else {
                                    r = (state.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                                }

                                if(l == left) leftPart.push_back(eSkel);
                                if(r == right) rightPart.push_back(eSkel);
                            }
                        }

                        if(leftPart.size() > 0 && leftPart == rightPart) {
                            bool ok = true;
                            if(leftPart.size()==1) {
                                node B = (blk.skel2tree[leftPart[0]]->source() == pNode ? blk.skel2tree[leftPart[0]]->target() : blk.skel2tree[leftPart[0]]->source());
                                


                                if(blk.spqr->typeOf(B) == SPQRTree::NodeType::SNode /*&& node_dp[B].cutsCnt >= 3*/) {
                                    for(auto &gccCut:node_dp[B].GccCuts_last3) {
                                        if(gccCut != pole0Gcc && gccCut != pole1Gcc) {
                                            // std::cout << "FAILED due to S node " << ctx().node2name[cc.nodeToOrig[gccCut]] << std::endl;
                                            ok = false;
                                            break;
                                        }
                                    }
                                    // ok = false;
                                }
                            }

                            if(ok) {
                                // std::cout << "SNARL: " << ctx().node2name[cc.nodeToOrig[pole0Gcc]] + (left == EdgePartType::PLUS ? "+" : "-") << ":" << ctx().node2name[cc.nodeToOrig[pole1Gcc]] + (right == EdgePartType::PLUS ? "+" : "-") << std::endl;
                                string s = /*"P"+*/ ctx().node2name[cc.nodeToOrig[pole0Gcc]] + (left == EdgePartType::PLUS ? "+" : "-");
                                string t = /*"P"+*/ctx().node2name[cc.nodeToOrig[pole1Gcc]] + (right == EdgePartType::PLUS ? "+" : "-");
                                
                                std::vector<std::string> v={s,t};
                                // std::cout << "P node snarl: ";
                                // for(auto &s:v) std::cout << s << " ";
                                // std::cout << std::endl;
                                addSnarl(v);

                            }
                        }                        

                    }
                }
            }

            void solveRR(ogdf::edge rrEdge, NodeArray<SPQRsolve::NodeDPState> &node_dp, ogdf::EdgeArray<EdgeDP> &edge_dp, BlockData& blk, const CcData& cc) {
                PROFILE_FUNCTION();
                EdgeDPState &down = edge_dp[rrEdge].down;
                EdgeDPState &up = edge_dp[rrEdge].up;

                node pole0Blk = down.s, pole1Blk = down.t;
                node pole0Gcc = blk.toCc[pole0Blk], pole1Gcc = blk.toCc[pole1Blk];

                // if(ctx().node2name[cc.nodeToOrig[pole0Gcc]] == "3497" || ctx().node2name[cc.nodeToOrig[pole1Gcc]] == "3497") {
                //     std::cout << "Processing RR edge with pole " << ctx().node2name[cc.nodeToOrig[pole0Gcc]] << " and " << ctx().node2name[cc.nodeToOrig[pole1Gcc]] << std::endl;
                // }

                // if(cc.isCutNode[pole0Blk]) {
                //     node pole0CutT = cc.bc->bcproper(pole0Gcc);
                //     if(cc.badCutCount[pole0CutT]>=2 || (cc.badCutCount[pole0CutT] == 1 && cc.lastBad[pole0CutT] != blk.bNode)) return;                     
                // }

                // if(cc.isCutNode[pole1Blk]) {
                //     node pole1CutT = cc.bc->bcproper(pole1Gcc);
                //     if(cc.badCutCount[pole1CutT]>=2 || (cc.badCutCount[pole1CutT] == 1 && cc.lastBad[pole1CutT] != blk.bNode)) return;                     
                // }




                {
                    // if(down.s == pole0Blk) {
                    //     if((down.localMinusS>0) + (down.localPlusS>0) == 2) return;
                    // } else {
                    //     if((down.localMinusT>0) + (down.localPlusT>0) == 2) return;
                    // }

                    // if(up.s == pole0Blk) {
                    //     if((up.localMinusS>0) + (up.localPlusS>0) == 2) return;
                    // } else {
                    //     if((up.localMinusT>0) + (up.localPlusT>0) == 2) return;
                    // }
                    if((up.localMinusS>0) + (up.localPlusS>0) == 2) return;
                    if((up.localMinusT>0) + (up.localPlusT>0) == 2) return;
                    if((down.localMinusS>0) + (down.localPlusS>0) == 2) return;
                    if((down.localMinusT>0) + (down.localPlusT>0) == 2) return;


                }

                EdgePartType pole0DownType = EdgePartType::NONE, pole0UpType = EdgePartType::NONE, pole1DownType = EdgePartType::NONE, pole1UpType = EdgePartType::NONE;

                if(down.s == pole0Blk) {
                    pole0DownType = (down.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                } else {
                    pole0DownType = (down.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                }

                if(up.s == pole0Blk) {
                    pole0UpType = (up.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                } else {
                    pole0UpType = (up.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                }

                if(down.s == pole1Blk) {
                    pole1DownType = (down.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                } else {
                    pole1DownType = (down.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                }

                if(up.s == pole1Blk) {
                    pole1UpType = (up.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                } else {
                    pole1UpType = (up.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                }


                if(pole0DownType == pole0UpType) return;
                if(pole1DownType == pole1UpType) return;



                if(cc.isCutNode[pole0Gcc] || cc.isCutNode[pole1Gcc]) {
                    assert(cc.badCutCount[pole0Gcc] >= 1);
                    if(cc.badCutCount[pole0Gcc]>=2 || cc.badCutCount[pole1Gcc]>=2) return;
                }


                {
                    string s = ctx().node2name[cc.nodeToOrig[pole0Gcc]] + (pole0DownType == EdgePartType::PLUS ? "+" : "-");
                    string t = ctx().node2name[cc.nodeToOrig[pole1Gcc]] + (pole1DownType == EdgePartType::PLUS ? "+" : "-");

                    std::vector<std::string> v={/*"RR"+*/s,/*"RR"+*/t};
                    // std::cout << "RR edge snarl: ";
                    // for(auto &s:v) std::cout << s << " ";
                    // std::cout << std::endl;
                    addSnarl(v);
                }

                {
                    string s = ctx().node2name[cc.nodeToOrig[pole0Gcc]] + (pole0UpType == EdgePartType::PLUS ? "+" : "-");
                    string t = ctx().node2name[cc.nodeToOrig[pole1Gcc]] + (pole1UpType == EdgePartType::PLUS ? "+" : "-");

                    std::vector<std::string> v={/*"RR"+*/s,/*"RR"+*/t};
                    // std::cout << "RR edge snarl: ";
                    // for(auto &s:v) std::cout << s << " ";
                    // std::cout << std::endl;

                    addSnarl(v);
                }




                // if(cc.isCutNode[pole0Blk]) {
                //     node pole0CutT = cc.bc->bcproper(pole0Gcc);
                //     if(cc.badCutCount[pole0CutT]>=2 || (cc.badCutCount[pole0CutT] == 1 && cc.lastBad[pole0CutT] != blk.bNode)) return;        
                //     // if(cc.badCutCount[pole0CutT]>=2 || (cc.badCutCount[pole0CutT] == 1 && cc.lastBad[pole0CutT] != blk.bNode)) return;                     
                // }

                // if(cc.isCutNode[pole1Blk]) {
                //     node pole1CutT = cc.bc->bcproper(pole1Gcc);
                //     if(cc.badCutCount[pole1CutT]>=2 || (cc.badCutCount[pole1CutT] == 1 && cc.lastBad[pole1CutT] != blk.bNode)) return;                     
                // }


                // std::array<EdgeDPState*, 2> states = { &down, &up };
                
                // for(auto &left:{EdgePartType::PLUS, EdgePartType::MINUS}) {
                //     for(auto &right:{EdgePartType::PLUS, EdgePartType::MINUS}) {
                //         std::vector<EdgeDPState*> leftPart, rightPart;



                //         for(auto state : states) {                            
                //             EdgePartType l, r;
                //             if(state->s == pole0Blk) {
                //                 l = (state->localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                //             } else {
                //                 l = (state->localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                //             }

                //             if(state->s == pole1Blk) {
                //                 r = (state->localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                //             } else {
                //                 r = (state->localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                //             }

                //             if(l == left) leftPart.push_back(state);
                            
                //             if(r == right) rightPart.push_back(state);
                //         }


                //         if(leftPart.size() > 0 && leftPart == rightPart) {
                //             string s = ctx().node2name[cc.nodeToOrig[pole0Gcc]] + (left == EdgePartType::PLUS ? "+" : "-");
                //             string t = ctx().node2name[cc.nodeToOrig[pole1Gcc]] + (right == EdgePartType::PLUS ? "+" : "-");
                            
                //             std::vector<std::string> v={s,t};
                //             addSnarl(v);

                //             // std::cout << "SNARL RR: " << ctx().node2name[cc.nodeToOrig[pole0Gcc]] + (left == EdgePartType::PLUS ? "+" : "-") << ":" << ctx().node2name[cc.nodeToOrig[pole1Gcc]] + (right == EdgePartType::PLUS ? "+" : "-") << std::endl;
                //         }                        

                //     }
                // }

            }

            void solveNodes(NodeArray<SPQRsolve::NodeDPState> &node_dp, ogdf::EdgeArray<EdgeDP> &edge_dp, BlockData& blk, const CcData& cc) {
                PROFILE_FUNCTION();
                if(!blk.spqr) return;
                const Graph &T = blk.spqr->tree();

                for(node tNode : T.nodes) {
                    auto tType = blk.spqr->typeOf(tNode);
                    if(tType == StaticSPQRTree::NodeType::SNode) {
                        // solve S node
                        solveS(tNode, node_dp, edge_dp, blk, cc);
                    } 
                }

                for(node tNode : T.nodes) {
                    auto tType = blk.spqr->typeOf(tNode);
                    if(tType == StaticSPQRTree::NodeType::PNode) {
                        // solve P node
                        solveP(tNode, node_dp, edge_dp, blk, cc);
                    } 
                }

                for(edge e: T.edges) {
                    if(blk.spqr->typeOf(e->source()) == SPQRTree::NodeType::RNode && blk.spqr->typeOf(e->target()) == SPQRTree::NodeType::RNode) {
                        solveRR(e, node_dp, edge_dp, blk, cc);
                    }
                }
            }

            void solveSPQR(BlockData& blk, const CcData& cc) {
                MARK_SCOPE_MEM("sn/solveSPQR");

                PROFILE_FUNCTION();
                if(!blk.spqr) return;
                if(!blk.spqr || blk.Gblk->numberOfNodes() < 3) return;

                const Graph &T = blk.spqr->tree();

                EdgeArray<SPQRsolve::EdgeDP> edge_dp(T);
                NodeArray<SPQRsolve::NodeDPState> node_dp(T);

                std::vector<ogdf::node> nodeOrder;
                std::vector<ogdf::edge> edgeOrder;

                SPQRsolve::dfsSPQR_order(*blk.spqr, edgeOrder, nodeOrder);

                ogdf::NodeArray<ogdf::node> blkToSkel(*blk.Gblk, nullptr);
                blk.blkToSkel = blkToSkel;

                for(auto e:edgeOrder) {
                    SPQRsolve::processEdge(e, edge_dp, cc, blk);
                }

                for(auto v:nodeOrder) {
                    SPQRsolve::processNode(v, edge_dp, cc, blk);
                }

                solveNodes(node_dp, edge_dp, blk, cc);


                for(edge eGblk: blk.Gblk->edges) {
                    edge eGcc = blk.edgeToOrig[eGblk];
                    edge eG = cc.edgeToOrig[eGcc];

                    node uGcc = eGcc->source();
                    node vGcc = eGcc->target();

                    node uG = cc.nodeToOrig[uGcc];
                    node vG = cc.nodeToOrig[vGcc];

                    if(ctx().node2name[uG] == "_trash" || ctx().node2name[vG] == "_trash") {
                        continue;
                    }

                    if(cc.isTip[uGcc] || cc.isTip[vGcc]) {
                        continue;
                    }

                    if((cc.isCutNode[uGcc] && cc.badCutCount[uGcc]>0) || (cc.isCutNode[vGcc] && cc.badCutCount[vGcc]>0)) {
                        continue;
                    }

                    int uPlusCnt = cc.degPlus[uGcc] - (getNodeEdgeType(uG, eG) == EdgePartType::PLUS ? 1 : 0), uMinusCnt = cc.degMinus[uGcc] - (getNodeEdgeType(uG, eG) == EdgePartType::MINUS ? 1 : 0);
                    int vPlusCnt = cc.degPlus[vGcc] - (getNodeEdgeType(vG, eG) == EdgePartType::PLUS ? 1 : 0), vMinusCnt = cc.degMinus[vGcc] - (getNodeEdgeType(vG, eG) == EdgePartType::MINUS ? 1 : 0);
                

                    bool ok = false;

                    string s = ctx().node2name[cc.nodeToOrig[uGcc]];
                    string t = ctx().node2name[cc.nodeToOrig[vGcc]];


                    if((uPlusCnt == 0 && uMinusCnt > 0 && vPlusCnt == 0 && vMinusCnt > 0) || (uPlusCnt > 0 && uMinusCnt == 0 && vPlusCnt > 0 && vMinusCnt == 0) || (uPlusCnt > 0 && uMinusCnt == 0 && vPlusCnt == 0 && vMinusCnt > 0) || (uPlusCnt == 0 && uMinusCnt > 0 && vPlusCnt > 0 && vMinusCnt == 0)) {
                        ok = true;
                    }

                    if(ok) {
                        std::vector<std::string> v={/*"E"+*/s + (getNodeEdgeType(uG, eG) == EdgePartType::PLUS ? "+" : "-"),/*"E"+*/t + (getNodeEdgeType(vG, eG) == EdgePartType::PLUS ? "+" : "-")};
                        addSnarl(v);
                        if(!blk._adjInS.count({s, t}) && !blk._adjInS.count({t, s})) {
                            std::vector<std::string> v={/*"E"+*/s + (getNodeEdgeType(uG, eG) == EdgePartType::PLUS ? "-" : "+"),/*"E"+*/t + (getNodeEdgeType(vG, eG) == EdgePartType::PLUS ? "-" : "+")};
                            addSnarl(v);
                        }
                    } 
                }
                    

            }


        }



        void findTips(CcData& cc) {
            MARK_SCOPE_MEM("sn/findTips");
            PROFILE_FUNCTION();
            size_t localIsolated = 0;
            for(node v : cc.Gcc->nodes) {
                int plusCnt = 0, minusCnt = 0;
                node vG = cc.nodeToOrig[v];

                for(auto adjE: v->adjEntries) {
                    ogdf::edge e = cc.edgeToOrig[adjE->theEdge()];
                    EdgePartType eType = getNodeEdgeType(vG, e);
                    if(eType == EdgePartType::PLUS) plusCnt++;
                    else minusCnt++;
                }

                if(plusCnt + minusCnt == 0) {
                    isolatedNodesCnt++;
                }

                if(plusCnt == 0 || minusCnt == 0) {
                    cc.isTip[v] = true;
                } else {
                    cc.isTip[v] = false;
                }
            }

            {
                std::lock_guard<std::mutex> lk(g_snarls_mtx);
                isolatedNodesCnt += localIsolated;
            }
        }


        void processCutNodes(CcData& cc) {
            MARK_SCOPE_MEM("sn/processCutNodes");
            PROFILE_FUNCTION();

            for(node v : cc.Gcc->nodes) {
                if(cc.bc->typeOfGNode(v) == BCTree::GNodeType::CutVertex) {
                    cc.isCutNode[v] = true;

                    bool isGood = true;
                    ogdf::node vT = cc.bc->bcproper(v);

                    for(auto adjV : vT->adjEntries) {
                        node uT = adjV->twinNode();
                        std::vector<ogdf::edge> outPlus, outMinus;
                        getOutgoingEdgesInBlock(cc, v, uT, EdgePartType::PLUS, outPlus);
                        getOutgoingEdgesInBlock(cc, v, uT, EdgePartType::MINUS, outMinus);

                        if(outPlus.size() > 0 && outMinus.size() > 0) {
                            isGood = false;
                            cc.lastBad[v] = uT;
                            cc.badCutCount[v]++;
                        }
                    }
                    cc.isGoodCutNode[v] = isGood;
                }
            }
        }


        void findCutSnarl(CcData &cc) {
            MARK_SCOPE_MEM("sn/findCutSnarl");
            PROFILE_FUNCTION();
            ogdf::NodeArray<std::pair<bool, bool>> visited(*cc.Gcc, {false, false}); // first: minus visited, second: plus visited

            std::function<void(ogdf::node, ogdf::node, EdgePartType, std::vector<std::string> &)>  dfs=[&](ogdf::node node, ogdf::node prev, EdgePartType edgeType, std::vector<std::string> &goodNodes) -> void {
                if((cc.isGoodCutNode[node] || cc.isTip[node]) && ctx().node2name[cc.nodeToOrig[node]] != "_trash") {
                    goodNodes.push_back(ctx().node2name[cc.nodeToOrig[node]]+(edgeType == EdgePartType::PLUS ? "+" : "-"));
                }

                if(!cc.isGoodCutNode[node] && !cc.isTip[node]) {
                    visited[node].first = visited[node].second = true;
                } else {
                    if(edgeType == EdgePartType::MINUS) visited[node].first = true;
                    else visited[node].second = true;
                }

                std::vector<ogdf::AdjElement*> sameOutEdges, otherOutEdges;
                getAllOutgoingEdgesOfType(cc, node, (edgeType == EdgePartType::PLUS ? EdgePartType::PLUS : EdgePartType::MINUS), sameOutEdges);
                getAllOutgoingEdgesOfType(cc, node, (edgeType == EdgePartType::PLUS ? EdgePartType::MINUS : EdgePartType::PLUS), otherOutEdges);

                bool canGoOther = !cc.isGoodCutNode[node] && !cc.isTip[node];

                for(auto &adjE:sameOutEdges) {
                    ogdf::node otherNode = adjE->twinNode();
                    ogdf::edge e = adjE->theEdge();
                    EdgePartType inType = getNodeEdgeType(cc.nodeToOrig[otherNode], cc.edgeToOrig[e]);

                    if((inType == EdgePartType::PLUS && !visited[otherNode].second) || (inType == EdgePartType::MINUS && !visited[otherNode].first)) {
                        dfs(otherNode, node, inType, goodNodes);
                    }
                }

                if(canGoOther) {
                    for(auto &adjE:otherOutEdges) {
                        ogdf::node otherNode = adjE->twinNode();
                        ogdf::edge e = adjE->theEdge();

                        EdgePartType inType = getNodeEdgeType(cc.nodeToOrig[otherNode], cc.edgeToOrig[e]);

                        if((inType == EdgePartType::PLUS && !visited[otherNode].second) || (inType == EdgePartType::MINUS && !visited[otherNode].first)) {
                            dfs(otherNode, node, inType, goodNodes);
                        }
                    }
                }
            };

            for(node v : cc.Gcc->nodes) {
                for(auto &t : {EdgePartType::PLUS, EdgePartType::MINUS}) {
                    if(t == EdgePartType::PLUS && visited[v].second) continue;
                    if(t == EdgePartType::MINUS && visited[v].first) continue;
                    std::vector<std::string> goodNodes;
                    dfs(v, nullptr, t, goodNodes);

                    if(goodNodes.size()>=2) {
                        addSnarl(goodNodes);
                    }
                }
            }   
        }


        void buildBlockData(BlockData& blk, CcData& cc) {
            MARK_SCOPE_MEM("sn/blockData/build");
            PROFILE_FUNCTION();
            {
                blk.Gblk = std::make_unique<Graph>();
            }

            blk.nodeToOrig.init(*blk.Gblk, nullptr);
            blk.edgeToOrig.init(*blk.Gblk, nullptr);
            blk.toCc.init(*blk.Gblk, nullptr);

            std::unordered_set<node> verts;
            {
                for (edge hE : cc.bc->hEdges(blk.bNode)) {
                    edge eC = cc.bc->original(hE);
                    verts.insert(eC->source());
                    verts.insert(eC->target());
                }
            }

            std::unordered_map<node, node> cc_to_blk;
            cc_to_blk.reserve(verts.size());

            {
                for (node vCc : verts) {
                    node vB = blk.Gblk->newNode();
                    cc_to_blk[vCc] = vB;
                    blk.toCc[vB] = vCc;
                    node vG = cc.nodeToOrig[vCc];
                    blk.nodeToOrig[vB] = vG;
                }
            }

            {
                for (edge hE : cc.bc->hEdges(blk.bNode)) {
                    edge eCc = cc.bc->original(hE);
                    auto srcIt = cc_to_blk.find(eCc->source());
                    auto tgtIt = cc_to_blk.find(eCc->target());
                    if (srcIt != cc_to_blk.end() && tgtIt != cc_to_blk.end()) {
                        edge e = blk.Gblk->newEdge(srcIt->second, tgtIt->second);
                        blk.edgeToOrig[e] = cc.edgeToOrig[eCc];
                    }
                }
            }

            if (blk.Gblk->numberOfNodes() >= 3) {
                {
                    MARK_SCOPE_MEM("sn/blockData/spqr_build");
                    blk.spqr = std::make_unique<StaticSPQRTree>(*blk.Gblk);
                }
                const Graph& T = blk.spqr->tree();

                blk.skel2tree.reserve(2*T.edges.size());
                blk.parent.init(T, nullptr);

                node root = blk.spqr->rootNode();
                blk.parent[root] = root;

                for (edge te : T.edges) {
                    node u = te->source();
                    node v = te->target();
                    blk.parent[v] = u;
                
                    if (auto eSrc = blk.spqr->skeletonEdgeSrc(te)) {
                        blk.skel2tree[eSrc] = te;
                    }
                    if (auto eTgt = blk.spqr->skeletonEdgeTgt(te)) {
                        blk.skel2tree[eTgt] = te;
                    }
                }
            }
        }




        struct BlockPrep {
            CcData* cc;
            node bNode;
        };



        struct ThreadComponentArgs {
            size_t tid;
            size_t numThreads;
            int nCC;
            size_t* nextIndex;
            std::mutex* workMutex;
            std::vector<std::vector<node>>* bucket;
            std::vector<std::vector<edge>>* edgeBuckets;
            std::vector<std::unique_ptr<CcData>>* components;
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

        struct ThreadTipsArgs {
            size_t tid;
            size_t numThreads;
            int nCC;
            size_t* nextIndex;
            std::mutex* workMutex;
            std::vector<std::unique_ptr<CcData>>* components;
        };


        struct ThreadBlocksArgs {
            size_t tid;
            size_t numThreads;
            size_t blocks;
            size_t* nextIndex;
            std::mutex* workMutex;
            std::vector<BlockPrep>* blockPreps;
        };




        void* worker_component(void* arg) {
            std::unique_ptr<ThreadComponentArgs> targs(static_cast<ThreadComponentArgs*>(arg));
            size_t tid = targs->tid;
            size_t numThreads = targs->numThreads;
            int nCC = targs->nCC;
            size_t* nextIndex = targs->nextIndex;
            std::mutex* workMutex = targs->workMutex;
            std::vector<std::unique_ptr<CcData>>* components = targs->components;
            std::vector<std::vector<node>>* bucket = targs->bucket;
            std::vector<std::vector<edge>>* edgeBuckets = targs->edgeBuckets;

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

                    (*components)[cid] = std::make_unique<CcData>();

                    {
                        MARK_SCOPE_MEM("sn/worker_component/gcc_rebuild");
                        (*components)[cid]->Gcc = std::make_unique<Graph>();
                        (*components)[cid]->nodeToOrig.init(*(*components)[cid]->Gcc, nullptr);
                        (*components)[cid]->edgeToOrig.init(*(*components)[cid]->Gcc, nullptr);
                        (*components)[cid]->isTip.init(*(*components)[cid]->Gcc, false);
                        (*components)[cid]->isCutNode.init(*(*components)[cid]->Gcc, false);
                        (*components)[cid]->isGoodCutNode.init(*(*components)[cid]->Gcc, false);
                        (*components)[cid]->lastBad.init(*(*components)[cid]->Gcc, nullptr);
                        (*components)[cid]->badCutCount.init(*(*components)[cid]->Gcc, 0);
                        (*components)[cid]->degPlus.init(*(*components)[cid]->Gcc, 0);
                        (*components)[cid]->degMinus.init(*(*components)[cid]->Gcc, 0);

                        std::unordered_map<node, node> orig_to_cc;
                        orig_to_cc.reserve((*bucket)[cid].size());

                        for (node vG : (*bucket)[cid]) {
                            node vC = (*components)[cid]->Gcc->newNode();
                            (*components)[cid]->nodeToOrig[vC] = vG;
                            orig_to_cc[vG] = vC;
                        }

                        for (edge e : (*edgeBuckets)[cid]) {
                            auto eC = (*components)[cid]->Gcc->newEdge(orig_to_cc[e->source()], orig_to_cc[e->target()]);
                            (*components)[cid]->edgeToOrig[eC] = e;
                            
                            (*components)[cid]->degPlus[orig_to_cc[e->source()]] += (getNodeEdgeType(e->source(), e) == EdgePartType::PLUS ? 1 : 0);
                            (*components)[cid]->degMinus[orig_to_cc[e->source()]] += (getNodeEdgeType(e->source(), e) == EdgePartType::MINUS ? 1 : 0);
                            (*components)[cid]->degPlus[orig_to_cc[e->target()]] += (getNodeEdgeType(e->target(), e) == EdgePartType::PLUS ? 1 : 0);
                            (*components)[cid]->degMinus[orig_to_cc[e->target()]] += (getNodeEdgeType(e->target(), e) == EdgePartType::MINUS ? 1 : 0);
                        }
                    }
                    processed++;
                }

                auto chunkEnd = std::chrono::high_resolution_clock::now();
                auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);

                if (chunkDuration.count() < 1000) {
                    chunkSize = std::min(chunkSize * 2, static_cast<size_t>(nCC / numThreads));
                } else if (chunkDuration.count() > 5000) {
                    chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                }
            }

            std::cout << "Thread " << tid << " built " << processed << " components(rebuild cc graph)" << std::endl;
            return nullptr;
        }


        void* worker_bcTree(void* arg) {
            std::unique_ptr<ThreadBcTreeArgs> targs(static_cast<ThreadBcTreeArgs*>(arg));
            size_t tid = targs->tid;
            size_t numThreads = targs->numThreads;
            int nCC = targs->nCC;
            size_t* nextIndex = targs->nextIndex;
            std::mutex* workMutex = targs->workMutex;
            std::vector<std::unique_ptr<CcData>>* components = targs->components;
            std::vector<BlockPrep>* blockPreps = targs->blockPreps;

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
                        MARK_SCOPE_MEM("sn/worker_bcTree/build");
                        cc->bc = std::make_unique<BCTree>(*cc->Gcc);
                    }

                    std::vector<BlockPrep> localPreps;
                    {
                        MARK_SCOPE_MEM("sn/worker_bcTree/collect_B_nodes");
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
                    chunkSize = std::min(chunkSize * 2, static_cast<size_t>(nCC / numThreads));
                } else if (chunkDuration.count() > 5000) {
                    chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                }
            }

            std::cout << "Thread " << tid << " built " << processed << " components (bc trees)" << std::endl;
            return nullptr;
        }

        void* worker_tips(void* arg) {
            std::unique_ptr<ThreadTipsArgs> targs(static_cast<ThreadTipsArgs*>(arg));
            size_t tid = targs->tid;
            size_t numThreads = targs->numThreads;
            int nCC = targs->nCC;
            size_t* nextIndex = targs->nextIndex;
            std::mutex* workMutex = targs->workMutex;
            std::vector<std::unique_ptr<CcData>>* components = targs->components;

            size_t chunkSize = 1;
            size_t processed = 0;

            std::vector<std::vector<std::string>> localSnarls;
            tls_snarl_buffer = &localSnarls;

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

                    findTips(*cc);
                    if (cc->bc->numberOfCComps() > 0) {
                        processCutNodes(*cc);
                    }
                    findCutSnarl(*cc); 

                    ++processed;
                }

                auto chunkEnd = std::chrono::high_resolution_clock::now();
                auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);

                if (chunkDuration.count() < 1000) {
                    chunkSize = std::min(chunkSize * 2, static_cast<size_t>(nCC / numThreads));
                } else if (chunkDuration.count() > 5000) {
                    chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                }
            }

            tls_snarl_buffer = nullptr;
            flushThreadLocalSnarls(localSnarls);

            std::cout << "Thread " << tid << " built " << processed << " components (cuts tips)" << std::endl;
            return nullptr;
        }

        void* worker_block(void* arg) {
            std::unique_ptr<ThreadBlocksArgs> targs(static_cast<ThreadBlocksArgs*>(arg));
            size_t tid = targs->tid;
            size_t numThreads = targs->numThreads;
            size_t blocks = targs->blocks;
            size_t* nextIndex = targs->nextIndex;
            std::mutex* workMutex = targs->workMutex;
            std::vector<BlockPrep>* blockPreps = targs->blockPreps;

            size_t chunkSize = 1;
            size_t processed = 0;

            std::vector<std::vector<std::string>> localSnarls;
            tls_snarl_buffer = &localSnarls;

            while (true) {
                size_t startIndex, endIndex;
                {
                    std::lock_guard<std::mutex> lock(*workMutex);
                    if (*nextIndex >= static_cast<size_t>(blocks)) break;
                    startIndex = *nextIndex;
                    endIndex = std::min(*nextIndex + chunkSize, static_cast<size_t>(blocks));
                    *nextIndex = endIndex;
                }

                auto chunkStart = std::chrono::high_resolution_clock::now();

                for (size_t bid = startIndex; bid < endIndex; ++bid) {
                    BlockData blk;
                    blk.bNode = (*blockPreps)[bid].bNode;

                    // Measure SPQR build per-block
                    {
                        MEM_TIME_BLOCK("SPQR: build (snarl worker)");
                        buildBlockData(blk, *(*blockPreps)[bid].cc);
                    }

                    // Measure solve (snarl detection) per-block
                    {
                        MEM_TIME_BLOCK("Algorithm: snarl solve (worker)");
                        if (blk.Gblk && blk.Gblk->numberOfNodes() >= 3) {
                            SPQRsolve::solveSPQR(blk, *(*blockPreps)[bid].cc); // addSnarl -> buffer TLS
                        }
                    }

                    ++processed;
                }

                auto chunkEnd = std::chrono::high_resolution_clock::now();
                auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);

                if (chunkDuration.count() < 1000) {
                    chunkSize = std::min(chunkSize * 2, static_cast<size_t>(blocks / numThreads));
                } else if (chunkDuration.count() > 5000) {
                    chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                }
            }

            tls_snarl_buffer = nullptr;
            flushThreadLocalSnarls(localSnarls);

            std::cout << "Thread " << tid << " built " << processed << " components (cuts tips)" << std::endl;
            return nullptr;
        }


        void solve() {
            std::cout << "Finding snarls...\n";
            PROFILE_FUNCTION();
            auto& C = ctx();
            Graph& G = C.G;

            MARK_SCOPE_MEM("sn/phase/ComputeCC");
            NodeArray<int> compIdx(G);
            int nCC = connectedComponents(G, compIdx);

            std::vector<std::vector<node>> bucket(nCC);
            {
                MARK_SCOPE_MEM("sn/phase/BucketNodes");
                for (node v : G.nodes) {
                    bucket[compIdx[v]].push_back(v);
                }
            }

            std::vector<std::vector<edge>> edgeBuckets(nCC);
            {
                MARK_SCOPE_MEM("sn/phase/BucketEdges");
                for (edge e : G.edges) {
                    edgeBuckets[compIdx[e->source()]].push_back(e);
                }
            }


            std::vector<std::unique_ptr<CcData>> components(nCC);

            std::vector<BlockPrep> blockPreps;

            {
                MEM_TIME_BLOCK("SNARLS: BUILD (Gcc+BC+SPQR)");
                ACCUM_BUILD();
                MARK_SCOPE_MEM("sn/phase/BUILD_all");

                // build components (Gcc)
                {
                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});

                    std::vector<pthread_t> threads(numThreads);

                    std::mutex workMutex;
                    size_t nextIndex = 0;

                    for (size_t tid = 0; tid < numThreads; ++tid) {
                        pthread_attr_t attr;
                        pthread_attr_init(&attr);

                        // size_t stackSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;
                        size_t stackSize = C.stackSize;

                        pthread_attr_setstacksize(&attr, stackSize);

                        ThreadComponentArgs* args = new ThreadComponentArgs{
                            tid,
                            numThreads,
                            nCC,
                            &nextIndex,
                            &workMutex,
                            &bucket,
                            &edgeBuckets,
                            &components,
                        };

                        int ret = pthread_create(&threads[tid], &attr, worker_component, args);
                        if (ret != 0) {
                            std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                            delete args;
                        }

                        pthread_attr_destroy(&attr);
                    }

                    for (size_t tid = 0; tid < numThreads; ++tid) {
                        pthread_join(threads[tid], nullptr);
                    }
                }

                // BC trees + collect blocks
                {
                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});
                    std::vector<pthread_t> threads(numThreads);

                    std::mutex workMutex;
                    size_t nextIndex = 0;

                    for (size_t tid = 0; tid < numThreads; ++tid) {
                        pthread_attr_t attr;
                        pthread_attr_init(&attr);

                        // size_t stackSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;
                        size_t stackSize = C.stackSize;

                        if(pthread_attr_setstacksize(&attr, stackSize) != 0){
                            std::cout << "[Error] pthread_attr_setstacksize" << std::endl;
                        }

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
                }

                // tips/cuts/cut-snarls
                {
                    MARK_SCOPE_MEM("sn/phase/tips_cuts");

                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});

                    std::vector<pthread_t> threads(numThreads);

                    std::mutex workMutex;
                    size_t nextIndex = 0;

                    for (size_t tid = 0; tid < numThreads; ++tid) {
                        pthread_attr_t attr;
                        pthread_attr_init(&attr);

                        // size_t stackSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;
                        size_t stackSize = C.stackSize;

                        if(pthread_attr_setstacksize(&attr, stackSize) != 0){
                            std::cout << "[Error] pthread_attr_setstacksize" << std::endl;
                        }

                        ThreadTipsArgs* args = new ThreadTipsArgs{
                            tid,
                            numThreads,
                            nCC,
                            &nextIndex,
                            &workMutex,
                            &components
                        };

                        int ret = pthread_create(&threads[tid], &attr, worker_tips, args);
                        if (ret != 0) {
                            std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                            delete args;
                        }

                        pthread_attr_destroy(&attr);
                    }

                    for (size_t tid = 0; tid < numThreads; ++tid) {
                        pthread_join(threads[tid], nullptr);
                    }
                }

                // SPQR inside blocks
                {
                    MARK_SCOPE_MEM("sn/phase/block_SPQR_solve");

                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, (size_t)blockPreps.size(), numThreads});

                    std::vector<pthread_t> threads(numThreads);

                    std::mutex workMutex;
                    size_t nextIndex = 0;

                    for (size_t tid = 0; tid < numThreads; ++tid) {
                        pthread_attr_t attr;
                        pthread_attr_init(&attr);

                        // size_t stackSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;
                        size_t stackSize = C.stackSize;

                        if(pthread_attr_setstacksize(&attr, stackSize) != 0){
                            std::cout << "[Error] pthread_attr_setstacksize" << std::endl;
                        }

                        ThreadBlocksArgs* args = new ThreadBlocksArgs{
                            tid,
                            numThreads,
                            blockPreps.size(),
                            &nextIndex,
                            &workMutex,
                            &blockPreps
                        };

                        int ret = pthread_create(&threads[tid], &attr, worker_block, args);
                        if (ret != 0) {
                            std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                            delete args;
                        }

                        pthread_attr_destroy(&attr);
                    }

                    for (size_t tid = 0; tid < numThreads; ++tid) {
                        pthread_join(threads[tid], nullptr);
                    }
                }
            }
        }


    }
}


int main(int argc, char** argv) {
    rlimit rl;
    rl.rlim_cur = RLIM_INFINITY;
    rl.rlim_max = RLIM_INFINITY;
    if (setrlimit(RLIMIT_STACK, &rl) != 0) {
        perror("setrlimit");
    }

    TIME_BLOCK("Starting graph reading...");
    logger::init();

    readArgs(argc, argv);

    {
        MEM_TIME_BLOCK("I/O: read graph");
        MARK_SCOPE_MEM("io/read_graph");
        PROFILE_BLOCK("Graph reading");
        GraphIO::readGraph();
    }

    if (ctx().bubbleType == Context::BubbleType::SUPERBUBBLE) {
        solver::superbubble::solve();

        std::cout << "Superbubbles found:\n";
        std::cout << ctx().superbubbles.size() << std::endl;
    } else if (ctx().bubbleType == Context::BubbleType::SNARL) {
        solver::snarls::solve();
        std::cout << "Snarls found..\n";
    }

    {
        MEM_TIME_BLOCK("I/O: write output");
        MARK_SCOPE_MEM("io/write_output");
        PROFILE_BLOCK("Writing output");
        TIME_BLOCK("Writing output");
        GraphIO::writeSuperbubbles();
    }

    std::cout << "Snarls found: " << snarlsFound << std::endl;

    PROFILING_REPORT();

    logger::info("Process PeakRSS: {:.2f} GiB", memtime::peakRSSBytes() / (1024.0 * 1024.0 * 1024.0));

    mark::report();
    if (!g_report_json_path.empty()) {
        mark::report_to_json(g_report_json_path);
    }

    return 0;
}
