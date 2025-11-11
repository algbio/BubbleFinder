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

#include <thread>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
  #include <immintrin.h>
  static inline void SPQR_WS_PAUSE() { _mm_pause(); }
  static inline void PAUSE_SPIN()    { _mm_pause(); }
#else
  #include <thread>
  static inline void SPQR_WS_PAUSE() { std::this_thread::yield(); }
  static inline void PAUSE_SPIN()    { std::this_thread::yield(); }
#endif






using namespace ogdf;

static std::string g_report_json_path;




static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " -g <graphFile> -o <outputFile> [--gfa] "
              << "[--superbubbles | --snarls] "
              << "[-j <threads>] "
              << "[--report-json <file>]\n";
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
            case SPQRTree::NodeType::SNode:
                return "SNode";
            case SPQRTree::NodeType::PNode:
                return "PNode";
            case SPQRTree::NodeType::RNode:
                return "RNode";
            default:
                return "Unknown";
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
            if(back.directTS) {
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

            if (!src || !snk) {
                return;
            }

            NodeArray<bool> vis(G,false);
            std::stack<node> S;
            vis[src]=true;
            S.push(src);
            bool reach=false;
            while(!S.empty() && !reach) {
                node u=S.top();
                S.pop();
                for(adjEntry a=u->firstAdj(); a; a=a->succ())
                    if(a->isSource()) {
                        node v=a->twinNode();
                        if(!vis[v]) {
                            if(v==snk) {
                                reach=true;
                                break;
                            }
                            vis[v]=true;
                            S.push(v);
                        }
                    }
            }
            if(!reach) {
                return;
            }

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

                            size_t stackSize = 64ULL * 1024ULL * 1024ULL * 1024ULL;
                            if(pthread_attr_setstacksize(&attr, stackSize) != 0) {
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

                            size_t stackSize = 64ULL * 1024ULL * 1024ULL * 1024ULL;
                            if(pthread_attr_setstacksize(&attr, stackSize) != 0) {
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
                    size_t stackSize = 1024ULL * 1024ULL * 1024ULL * 20ULL;
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
            struct pair_hash {
                size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
                    auto h1 = std::hash<std::string> {}(p.first);
                    auto h2 = std::hash<std::string> {}(p.second);
                    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
                }
            };

            thread_local std::vector<std::vector<std::string>> tls_snarls_buffer;
            std::mutex g_snarls_mutex;

            std::unordered_set<std::pair<std::string, std::string>, pair_hash> tls_snarls_collector;
        }

        static void tryCommitSnarl(std::vector<std::string> s) {
            auto &C = ctx();
            if (s.size() >= 2) {
                snarlsFound += static_cast<uint64_t>(s.size()) * static_cast<uint64_t>(s.size() - 1) / 2;
            }
            C.snarls.insert(std::move(s));
        }


        inline void flushThreadLocalSnarls() {
            if (tls_snarls_buffer.empty()) return;
            std::lock_guard<std::mutex> lk(g_snarls_mutex);
            for (auto &s : tls_snarls_buffer) {
                tryCommitSnarl(std::move(s));
            }
            tls_snarls_buffer.clear();
        }

        // Remplacer l'ancienne addSnarl par cette version (buffer TLS + flush seuil)
        inline void addSnarl(std::vector<std::string> s) {
            tls_snarls_buffer.emplace_back(std::move(s));
            if (tls_snarls_buffer.size() >= 2048) {
                flushThreadLocalSnarls();
            }
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

        // void addSnarl(std::vector<std::string> s) {
        //     // if (tls_snarls_collector) {
        //     //     tls_superbubble_collector->emplace_back(source, sink);
        //     //         return;
        //     //     }
        //     // Otherwise, commit directly to global state (sequential behavior)
        //     // tryCommitSnarl(source, sink);
        //     tryCommitSnarl(s);

        //     // if(C.isEntry[source] || C.isExit[sink]) {
        //     //     std::cerr << ("Superbubble already exists for source %s and sink %s", C.node2name[source].c_str(), C.node2name[sink].c_str());
        //     //     return;
        //     // }
        //     // C.isEntry[source] = true;
        //     // C.isExit[sink] = true;
        //     // C.superbubbles.emplace_back(source, sink);

        // }



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

            ogdf::node bNode {nullptr};

            bool isAcycic {true};

            BlockData() {}
        };

        struct CcData {
            std::unique_ptr<ogdf::Graph> Gcc;
            ogdf::NodeArray<ogdf::node> nodeToOrig;
            ogdf::EdgeArray<ogdf::edge> edgeToOrig;

            ogdf::NodeArray<bool> isTip;

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

        static std::atomic<uint64_t> s_goeot_selfloops_ignored{0};

        void getAllOutgoingEdgesOfType(const CcData& cc,
                                    ogdf::node uG,
                                    EdgePartType type,
                                    std::vector<ogdf::AdjElement*>& outEdges)
        {
            outEdges.clear();

            for (auto adjE : uG->adjEntries) {
                // Ignorer les boucles (self-loops) pour éviter node == otherNode et du bruit
                if (adjE->twinNode() == uG) {
                    s_goeot_selfloops_ignored.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                ogdf::edge eC    = adjE->theEdge();     // arête dans cc.Gcc
                ogdf::edge eOrig = cc.edgeToOrig[eC];   // arête correspondante dans le graphe original

                if (eC->source() == uG) {
                    EdgePartType outType = ctx()._edge2types(eOrig).first;
                    if (type == outType) outEdges.push_back(adjE);
                } else {
                    EdgePartType outType = ctx()._edge2types(eOrig).second;
                    if (type == outType) outEdges.push_back(adjE);
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


            inline void processEdgeAB(ogdf::edge curr_edge,
                                    ogdf::node A, ogdf::node B,
                                    ogdf::EdgeArray<EdgeDP> &dp,
                                    const CcData &cc,
                                    BlockData &blk)
            {
                EdgeDPState &state = dp[curr_edge].down;
                EdgeDPState &back_state = dp[curr_edge].up;

                const ogdf::StaticSPQRTree &spqr = *blk.spqr;
                state.localPlusS = state.localPlusT = state.localMinusS = state.localMinusT = 0;

                const ogdf::Skeleton &skel = spqr.skeleton(B);
                const ogdf::Graph &skelGraph = skel.getGraph();

                // Trouver poles s/t: arête virtuelle pointant vers A (parent)
                for (ogdf::edge e : skelGraph.edges) {
                    ogdf::node u = e->source();
                    ogdf::node v = e->target();
                    if (skel.twinTreeNode(e) == A) {
                        ogdf::node vBlk = skel.original(v);
                        ogdf::node uBlk = skel.original(u);
                        state.s = back_state.s = vBlk;
                        state.t = back_state.t = uBlk;
                        break;
                    }
                }

                for (ogdf::edge e : skelGraph.edges) {
                    if (!skel.isVirtual(e)) {
                        ogdf::edge eG = blk.edgeToOrig[skel.realEdge(e)];
                        ogdf::node uG = eG->source();
                        ogdf::node vG = eG->target();

                        if (uG == blk.nodeToOrig[state.s]) {
                            auto t = getNodeEdgeType(uG, eG);
                            if (t == EdgePartType::PLUS) ++state.localPlusS; else if (t == EdgePartType::MINUS) ++state.localMinusS;
                        }
                        if (vG == blk.nodeToOrig[state.s]) {
                            auto t = getNodeEdgeType(vG, eG);
                            if (t == EdgePartType::PLUS) ++state.localPlusS; else if (t == EdgePartType::MINUS) ++state.localMinusS;
                        }
                        if (uG == blk.nodeToOrig[state.t]) {
                            auto t = getNodeEdgeType(uG, eG);
                            if (t == EdgePartType::PLUS) ++state.localPlusT; else if (t == EdgePartType::MINUS) ++state.localMinusT;
                        }
                        if (vG == blk.nodeToOrig[state.t]) {
                            auto t = getNodeEdgeType(vG, eG);
                            if (t == EdgePartType::PLUS) ++state.localPlusT; else if (t == EdgePartType::MINUS) ++state.localMinusT;
                        }
                        continue;
                    }

                    // Arête virtuelle: agrège enfant sauf vers A
                    if (skel.twinTreeNode(e) == A) continue;

                    ogdf::edge treeE = blk.skel2tree.at(e);
                    const EdgeDPState child = dp[treeE].down;

                    if (state.s == child.s) {
                        state.localPlusS += child.localPlusS;
                        state.localMinusS += child.localMinusS;
                    } else if (state.s == child.t) {
                        state.localPlusS += child.localPlusT;
                        state.localMinusS += child.localMinusT;
                    }

                    if (state.t == child.t) {
                        state.localPlusT += child.localPlusT;
                        state.localMinusT += child.localMinusT;
                    } else if (state.t == child.s) {
                        state.localPlusT += child.localPlusS;
                        state.localMinusT += child.localMinusS;
                    }
                }
            }




            void processNode(node curr_node,
                            EdgeArray<SPQRsolve::EdgeDP> &edge_dp,
                            const CcData &cc,
                            BlockData &blk)
            {
                const StaticSPQRTree &spqr = *blk.spqr;
                const Skeleton &skel       = spqr.skeleton(curr_node);
                const Graph &skelGraph     = skel.getGraph();

                // map 'block node' -> 'skeleton node' for this skeleton
                for (node h : skelGraph.nodes) {
                    node vB = skel.original(h);
                    blk.blkToSkel[vB] = h;
                }

                NodeArray<int> localPlusDeg(skelGraph, 0);
                NodeArray<int> localMinusDeg(skelGraph, 0);

                // 1) Contributions des arêtes réelles du squelette
                for (edge e : skelGraph.edges) {
                    if (skel.isVirtual(e)) continue;

                    edge eG = blk.edgeToOrig[ skel.realEdge(e) ];
                    node uSk = e->source();
                    node vSk = e->target();

                    node uG = blk.nodeToOrig[ skel.original(uSk) ];
                    node vG = blk.nodeToOrig[ skel.original(vSk) ];

                    EdgePartType tU = getNodeEdgeType(uG, eG);
                    EdgePartType tV = getNodeEdgeType(vG, eG);

                    if (tU == EdgePartType::PLUS)  ++localPlusDeg[uSk];
                    else if (tU == EdgePartType::MINUS) ++localMinusDeg[uSk];

                    if (tV == EdgePartType::PLUS)  ++localPlusDeg[vSk];
                    else if (tV == EdgePartType::MINUS) ++localMinusDeg[vSk];
                }

                // 2) Contributions des arêtes virtuelles (états 'child')
                for (edge e : skelGraph.edges) {
                    if (!skel.isVirtual(e)) continue;

                    edge treeE = blk.skel2tree.at(e);
                    OGDF_ASSERT(treeE != nullptr);

                    node A = curr_node;
                    node B = (treeE->source() == A ? treeE->target() : treeE->source());

                    // Etat à utiliser ici pour "contribuer" au degré local
                    SPQRsolve::EdgeDPState *child =
                        (blk.parent(A) == B ? &edge_dp[treeE].up : &edge_dp[treeE].down);

                    node sSk = blk.blkToSkel[ child->s ];
                    node tSk = blk.blkToSkel[ child->t ];

                    // Ajouter la contribution du sous-arbre enfant aux pôles
                    localPlusDeg[sSk]  += child->localPlusS;
                    localMinusDeg[sSk] += child->localMinusS;
                    localPlusDeg[tSk]  += child->localPlusT;
                    localMinusDeg[tSk] += child->localMinusT;
                }

                // 3) Calculer les états "up" pour chaque arête virtuelle de ce squelette
                for (edge e : skelGraph.edges) {
                    if (!skel.isVirtual(e)) continue;

                    edge treeE = blk.skel2tree.at(e);
                    node A = curr_node;
                    node B = (treeE->source() == A ? treeE->target() : treeE->source());

                    // AB := état coté enfant tel qu'il "rentre" dans A
                    // BA := état que l'on doit remplir pour "sortir" de A vers B
                    SPQRsolve::EdgeDPState *AB;
                    SPQRsolve::EdgeDPState *BA;
                    if (blk.parent(A) == B) {
                        AB = &edge_dp[treeE].up;
                        BA = &edge_dp[treeE].down;
                    } else {
                        AB = &edge_dp[treeE].down;
                        BA = &edge_dp[treeE].up;
                    }

                    BA->s = AB->s;
                    BA->t = AB->t;

                    node sSk = blk.blkToSkel[ AB->s ];
                    node tSk = blk.blkToSkel[ AB->t ];

                    BA->localPlusS  = localPlusDeg[sSk]  - AB->localPlusS;
                    BA->localMinusS = localMinusDeg[sSk] - AB->localMinusS;
                    BA->localPlusT  = localPlusDeg[tSk]  - AB->localPlusT;
                    BA->localMinusT = localMinusDeg[tSk] - AB->localMinusT;
                }
            }

            inline void processNode_threadsafe(ogdf::node curr_node,
                                            ogdf::EdgeArray<EdgeDP> &edge_dp,
                                            const CcData &cc,
                                            BlockData &blk)
            {
                const ogdf::StaticSPQRTree &spqr = *blk.spqr;
                const ogdf::Skeleton &skel       = spqr.skeleton(curr_node);
                const ogdf::Graph &skelGraph     = skel.getGraph();

                std::unordered_map<ogdf::node, ogdf::node> blkToSkelLocal;
                blkToSkelLocal.reserve(skelGraph.numberOfNodes());
                for (ogdf::node h : skelGraph.nodes) {
                    blkToSkelLocal[ skel.original(h) ] = h;
                }

                ogdf::NodeArray<int> localPlusDeg(skelGraph, 0);
                ogdf::NodeArray<int> localMinusDeg(skelGraph, 0);

                for (ogdf::edge e : skelGraph.edges) {
                    if (skel.isVirtual(e)) continue;
                    ogdf::edge eG = blk.edgeToOrig[ skel.realEdge(e) ];
                    ogdf::node uSk = e->source();
                    ogdf::node vSk = e->target();
                    ogdf::node uG = blk.nodeToOrig[ skel.original(uSk) ];
                    ogdf::node vG = blk.nodeToOrig[ skel.original(vSk) ];
                    EdgePartType tU = getNodeEdgeType(uG, eG);
                    EdgePartType tV = getNodeEdgeType(vG, eG);
                    if (tU == EdgePartType::PLUS) ++localPlusDeg[uSk]; else if (tU == EdgePartType::MINUS) ++localMinusDeg[uSk];
                    if (tV == EdgePartType::PLUS) ++localPlusDeg[vSk]; else if (tV == EdgePartType::MINUS) ++localMinusDeg[vSk];
                }

                // Ajouter contributions des enfants (AB) dans les degrés locaux
                for (ogdf::edge e : skelGraph.edges) {
                    if (!skel.isVirtual(e)) continue;
                    ogdf::edge treeE = blk.skel2tree.at(e);
                    ogdf::node A = curr_node;
                    ogdf::node B = (treeE->source() == A ? treeE->target() : treeE->source());
                    EdgeDPState *AB = (blk.parent(A) == B ? &edge_dp[treeE].up : &edge_dp[treeE].down);

                    ogdf::node sSk = blkToSkelLocal[ AB->s ];
                    ogdf::node tSk = blkToSkelLocal[ AB->t ];
                    localPlusDeg[sSk]  += AB->localPlusS;
                    localMinusDeg[sSk] += AB->localMinusS;
                    localPlusDeg[tSk]  += AB->localPlusT;
                    localMinusDeg[tSk] += AB->localMinusT;
                }

                // Calculer BA (à partir de AB et des degrés cumulés)
                for (ogdf::edge e : skelGraph.edges) {
                    if (!skel.isVirtual(e)) continue;
                    ogdf::edge treeE = blk.skel2tree.at(e);
                    ogdf::node A = curr_node;
                    ogdf::node B = (treeE->source() == A ? treeE->target() : treeE->source());

                    EdgeDPState *AB;
                    EdgeDPState *BA;
                    if (blk.parent(A) == B) { AB = &edge_dp[treeE].up; BA = &edge_dp[treeE].down; }
                    else                    { AB = &edge_dp[treeE].down; BA = &edge_dp[treeE].up; }

                    BA->s = AB->s; BA->t = AB->t;

                    ogdf::node sSk = blkToSkelLocal[ AB->s ];
                    ogdf::node tSk = blkToSkelLocal[ AB->t ];
                    BA->localPlusS  = localPlusDeg[sSk]  - AB->localPlusS;
                    BA->localMinusS = localMinusDeg[sSk] - AB->localMinusS;
                    BA->localPlusT  = localPlusDeg[tSk]  - AB->localPlusT;
                    BA->localMinusT = localMinusDeg[tSk] - AB->localMinusT;
                }
            }



            void solveS(ogdf::node sNode,
                        ogdf::NodeArray<SPQRsolve::NodeDPState> &node_dp,
                        ogdf::EdgeArray<SPQRsolve::EdgeDP> &dp,
                        BlockData& blk,
                        const CcData& cc)
            {
                const auto &spqr      = *blk.spqr;
                const auto &skel      = spqr.skeleton(sNode);
                const auto &skelGraph = skel.getGraph();

                // Raccourci pour récupérer l’état d’une arête virtuelle du squelette
                auto edge_state = [&](ogdf::edge eSkel) -> const SPQRsolve::EdgeDPState* {
                    ogdf::edge treeE = blk.skel2tree.at(eSkel);
                    ogdf::node B = skel.twinTreeNode(eSkel);
                    return (blk.parent(sNode) == B) ? &dp[treeE].up : &dp[treeE].down;
                };

                std::vector<ogdf::node> nodesInOrderGcc;   // sommets dans Gcc
                std::vector<ogdf::node> nodesInOrderSkel;  // sommets dans le squelette
                std::vector<ogdf::adjEntry> adjEntriesSkel; // les arêtes incidentes (dans l’ordre) au cycle
                std::vector<ogdf::edge> adjEdgesG_;        // arêtes réelles correspondantes (ou nullptr si virtuelles)

                // Parcours du cycle (ton DFS léger d’origine)
                {
                    std::function<void(ogdf::node, ogdf::node)> dfs = [&](ogdf::node u, ogdf::node prev) {
                        nodesInOrderGcc.push_back(blk.toCc[skel.original(u)]);
                        nodesInOrderSkel.push_back(u);

                        for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                            if (adj->twinNode() == prev) continue;

                            if (adj->twinNode() == skelGraph.firstNode() && u != skelGraph.firstNode()) {
                                if (skel.realEdge(adj->theEdge())) adjEdgesG_.push_back(blk.edgeToOrig[skel.realEdge(adj->theEdge())]);
                                else                              adjEdgesG_.push_back(nullptr);
                                adjEntriesSkel.push_back(adj);
                            }

                            if (adj->twinNode() == skelGraph.firstNode() || adj->twinNode() == prev) continue;

                            if (skel.realEdge(adj->theEdge())) adjEdgesG_.push_back(blk.edgeToOrig[skel.realEdge(adj->theEdge())]);
                            else                              adjEdgesG_.push_back(nullptr);

                            adjEntriesSkel.push_back(adj);
                            dfs(adj->twinNode(), u);
                        }
                    };
                    dfs(skelGraph.firstNode(), skelGraph.firstNode()->firstAdj()->twinNode());
                }

                std::vector<std::string> res;

                for (size_t i = 0; i < nodesInOrderGcc.size(); i++) {
                    auto uGcc = nodesInOrderGcc[i];

                    // Les deux arêtes du cycle autour de u
                    ogdf::edge eSkel0 = adjEntriesSkel[(i + adjEntriesSkel.size() - 1) % adjEntriesSkel.size()]->theEdge();
                    ogdf::edge eSkel1 = adjEntriesSkel[i]->theEdge();
                    ogdf::edge eG0 = adjEdgesG_[(i + adjEdgesG_.size() - 1) % adjEdgesG_.size()];
                    ogdf::edge eG1 = adjEdgesG_[i];

                    bool nodeIsCut = ((cc.isCutNode[uGcc] && cc.badCutCount[uGcc] == 1) || (!cc.isCutNode[uGcc]));
                    EdgePartType t0 = EdgePartType::NONE;
                    EdgePartType t1 = EdgePartType::NONE;

                    // Déterminer le signe sur la première arête adjacente
                    if (!skel.isVirtual(eSkel0)) {
                        t0 = getNodeEdgeType(cc.nodeToOrig[uGcc], eG0);
                    } else {
                        auto s = edge_state(eSkel0);
                        if (blk.toCc[s->s] == uGcc) {
                            if      (s->localMinusS == 0 && s->localPlusS > 0) t0 = EdgePartType::PLUS;
                            else if (s->localPlusS  == 0 && s->localMinusS > 0) t0 = EdgePartType::MINUS;
                        } else {
                            if      (s->localMinusT == 0 && s->localPlusT > 0) t0 = EdgePartType::PLUS;
                            else if (s->localPlusT  == 0 && s->localMinusT > 0) t0 = EdgePartType::MINUS;
                        }
                    }

                    // Déterminer le signe sur la seconde arête adjacente
                    if (!skel.isVirtual(eSkel1)) {
                        t1 = getNodeEdgeType(cc.nodeToOrig[uGcc], eG1);
                    } else {
                        auto s = edge_state(eSkel1);
                        if (blk.toCc[s->s] == uGcc) {
                            if      (s->localMinusS == 0 && s->localPlusS > 0) t1 = EdgePartType::PLUS;
                            else if (s->localPlusS  == 0 && s->localMinusS > 0) t1 = EdgePartType::MINUS;
                        } else {
                            if      (s->localMinusT == 0 && s->localPlusT > 0) t1 = EdgePartType::PLUS;
                            else if (s->localPlusT  == 0 && s->localMinusT > 0) t1 = EdgePartType::MINUS;
                        }
                    }

                    nodeIsCut &= (t0 != EdgePartType::NONE && t1 != EdgePartType::NONE && t0 != t1);

                    if (nodeIsCut) {
                        // Empiler (u, signe à gauche) puis (u, signe à droite)
                        auto push_label = [&](EdgePartType t, ogdf::edge eSkel, ogdf::edge eG) {
                            char signChar = (t == EdgePartType::PLUS ? '+' : '-');
                            if (!skel.isVirtual(eSkel)) {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] + std::string(1, signChar));
                            } else {
                                auto st = edge_state(eSkel);
                                bool useS = (blk.toCc[st->s] == uGcc);
                                int plus  = useS ? st->localPlusS  : st->localPlusT;
                                // signe déterminé par la présence de + vs - local (cohérent avec ci-dessus)
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] + std::string(1, (plus > 0 ? '+' : '-')));
                            }
                        };

                        push_label(t0, eSkel0, eG0);
                        push_label(t1, eSkel1, eG1);
                    }
                }

                // Émission des snarls par paires consécutives
                if (res.size() > 2) {
                    for (size_t i = 1; i < res.size(); i += 2) {
                        addSnarl(std::vector<std::string>{res[i], res[(i + 1) % res.size()]});
                    }
                }
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

                for(auto &left: {
                            EdgePartType::PLUS, EdgePartType::MINUS
                        }) {
                    for(auto &right: {
                                EdgePartType::PLUS, EdgePartType::MINUS
                            }) {
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

                                std::vector<std::string> v= {s,t};
                                // std::cout << "P node snarl: ";
                                // for(auto &s:v) std::cout << s << " ";
                                // std::cout << std::endl;
                                addSnarl(v);

                            }
                        }

                    }
                }
            }

            void solveRR(ogdf::edge rrEdge,
                        ogdf::NodeArray<SPQRsolve::NodeDPState> &node_dp,
                        ogdf::EdgeArray<EdgeDP> &edge_dp,
                        BlockData& blk,
                        const CcData& cc)
            {
                PROFILE_FUNCTION();
                EdgeDPState &down = edge_dp[rrEdge].down;
                EdgeDPState &up   = edge_dp[rrEdge].up;

                ogdf::node pole0Blk = down.s, pole1Blk = down.t;
                ogdf::node pole0Gcc = blk.toCc[pole0Blk], pole1Gcc = blk.toCc[pole1Blk];

                // Si l’un des états est incohérent, on sort
                if (!pole0Blk || !pole1Blk || !pole0Gcc || !pole1Gcc) return;

                // Rejeter les cas où un pôle voit simultanément + et - (pour up ou down)
                if (((up.localMinusS  > 0) + (up.localPlusS  > 0)) == 2) return;
                if (((up.localMinusT  > 0) + (up.localPlusT  > 0)) == 2) return;
                if (((down.localMinusS> 0) + (down.localPlusS> 0)) == 2) return;
                if (((down.localMinusT> 0) + (down.localPlusT> 0)) == 2) return;

                // Types au pôle 0 et 1 pour up/down
                EdgePartType pole0DownType = EdgePartType::NONE, pole0UpType = EdgePartType::NONE;
                EdgePartType pole1DownType = EdgePartType::NONE, pole1UpType = EdgePartType::NONE;

                if (down.s == pole0Blk)  pole0DownType = (down.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                else                      pole0DownType = (down.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);

                if (up.s == pole0Blk)    pole0UpType   = (up.localPlusS   > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                else                      pole0UpType   = (up.localPlusT   > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);

                if (down.s == pole1Blk)  pole1DownType = (down.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                else                      pole1DownType = (down.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);

                if (up.s == pole1Blk)    pole1UpType   = (up.localPlusS   > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                else                      pole1UpType   = (up.localPlusT   > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);

                // Les deux directions doivent différer (sinon pas de snarl RR)
                if (pole0DownType == pole0UpType) return;
                if (pole1DownType == pole1UpType) return;

                // Filtrage sur cut-nodes (similaire à solveP)
                if (cc.isCutNode[pole0Gcc]) {
                    if (cc.badCutCount[pole0Gcc] >= 2) return;
                    if (cc.badCutCount[pole0Gcc] == 1 && cc.lastBad[pole0Gcc] != blk.bNode) return;
                }
                if (cc.isCutNode[pole1Gcc]) {
                    if (cc.badCutCount[pole1Gcc] >= 2) return;
                    if (cc.badCutCount[pole1Gcc] == 1 && cc.lastBad[pole1Gcc] != blk.bNode) return;
                }

                // Émettre les deux snarls (down et up)
                {
                    std::string s = ctx().node2name[cc.nodeToOrig[pole0Gcc]] + (pole0DownType == EdgePartType::PLUS ? "+" : "-");
                    std::string t = ctx().node2name[cc.nodeToOrig[pole1Gcc]] + (pole1DownType == EdgePartType::PLUS ? "+" : "-");
                    std::vector<std::string> v{std::move(s), std::move(t)};
                    addSnarl(std::move(v));
                }
                {
                    std::string s = ctx().node2name[cc.nodeToOrig[pole0Gcc]] + (pole0UpType == EdgePartType::PLUS ? "+" : "-");
                    std::string t = ctx().node2name[cc.nodeToOrig[pole1Gcc]] + (pole1UpType == EdgePartType::PLUS ? "+" : "-");
                    std::vector<std::string> v{std::move(s), std::move(t)};
                    addSnarl(std::move(v));
                }
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
                PROFILE_FUNCTION();

                auto envOn = []() -> bool {
                    const char* s = std::getenv("SBFIND_LOG");
                    return s && s[0] != '0';
                };
                const bool LOG = envOn();
                static std::mutex logMutex;

                if (!blk.spqr) return;
                if (blk.Gblk->numberOfNodes() < 3) return;

                auto t0 = std::chrono::high_resolution_clock::now();

                const ogdf::Graph &T = blk.spqr->tree();

                if (LOG) {
                    std::lock_guard<std::mutex> lk(logMutex);
                    std::cout << "[SPQRsolve] start: |V_blk|=" << blk.Gblk->numberOfNodes()
                            << " |E_blk|=" << blk.Gblk->numberOfEdges()
                            << " SPQR nodes=" << T.numberOfNodes()
                            << " edges=" << T.numberOfEdges()
                            << " P=" << blk.spqr->numberOfPNodes()
                            << " S=" << blk.spqr->numberOfSNodes()
                            << " R=" << blk.spqr->numberOfRNodes()
                            << std::endl;
                }

                ogdf::EdgeArray<SPQRsolve::EdgeDP> edge_dp(T);
                ogdf::NodeArray<SPQRsolve::NodeDPState> node_dp(T);

                std::vector<ogdf::node> nodeOrder;
                std::vector<ogdf::edge> edgeOrder;

                auto t1 = std::chrono::high_resolution_clock::now();
                SPQRsolve::dfsSPQR_order(*blk.spqr, edgeOrder, nodeOrder);
                auto t2 = std::chrono::high_resolution_clock::now();

                ogdf::NodeArray<ogdf::node> blkToSkel(*blk.Gblk, nullptr);
                blk.blkToSkel = blkToSkel;

                auto t3 = std::chrono::high_resolution_clock::now();
                for (auto e : edgeOrder) {
                    SPQRsolve::processEdge(e, edge_dp, cc, blk);
                }
                auto t4 = std::chrono::high_resolution_clock::now();

                for (auto v : nodeOrder) {
                    SPQRsolve::processNode(v, edge_dp, cc, blk);
                }
                auto t5 = std::chrono::high_resolution_clock::now();

                solveNodes(node_dp, edge_dp, blk, cc);
                auto t6 = std::chrono::high_resolution_clock::now();

                long long msOrder = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                long long msEdges = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
                long long msNodes = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();
                long long msSolve = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();
                long long msTotal = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t0).count();

                if (LOG) {
                    std::lock_guard<std::mutex> lk(logMutex);
                    std::cout << "[SPQRsolve] timings: order=" << msOrder
                            << " ms, processEdges=" << msEdges
                            << " ms, processNodes=" << msNodes
                            << " ms, solve=" << msSolve
                            << " ms, total=" << msTotal << " ms"
                            << " |edgeOrder|=" << edgeOrder.size()
                            << " |nodeOrder|=" << nodeOrder.size()
                            << std::endl;
                }
            }




            namespace WS {

                enum class TaskKind : uint8_t { Edge, Node, SolveS, SolveP, SolveRR };

                struct TaskRec {
                    TaskKind    kind;
                    ogdf::edge  e;
                    ogdf::node  v;
                    ogdf::node  parent;
                    ogdf::node  child;
                    std::atomic<int> next;

                    TaskRec()
                        : kind(TaskKind::Edge), e(nullptr), v(nullptr),
                        parent(nullptr), child(nullptr), next(-1) {}

                    TaskRec(const TaskRec&) = delete;
                    TaskRec& operator=(const TaskRec&) = delete;

                    TaskRec(TaskRec&& o) noexcept
                        : kind(o.kind), e(o.e), v(o.v), parent(o.parent), child(o.child), next(-1)
                    {
                        next.store(o.next.load(std::memory_order_relaxed), std::memory_order_relaxed);
                    }
                    TaskRec& operator=(TaskRec&& o) noexcept {
                        if (this != &o) {
                            kind   = o.kind;
                            e      = o.e;
                            v      = o.v;
                            parent = o.parent;
                            child  = o.child;
                            next.store(o.next.load(std::memory_order_relaxed), std::memory_order_relaxed);
                        }
                        return *this;
                    }
                };

                struct ReadyStack {
                    std::atomic<int> head{ -1 };
                    std::vector<TaskRec>* tasks{ nullptr };

                    void init(std::vector<TaskRec>* t) {
                        tasks = t;
                        head.store(-1, std::memory_order_relaxed);
                    }
                    void push(int idx) {
                        int old = head.load(std::memory_order_relaxed);
                        do {
                            (*tasks)[idx].next.store(old, std::memory_order_relaxed);
                        } while (!head.compare_exchange_weak(old, idx,
                                std::memory_order_release, std::memory_order_relaxed));
                    }
                    int pop() {
                        int h = head.load(std::memory_order_acquire);
                        while (h != -1) {
                            int nxt = (*tasks)[h].next.load(std::memory_order_relaxed);
                            if (head.compare_exchange_weak(h, nxt,
                                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                                return h;
                            }
                        }
                        return -1;
                    }
                };

                struct BuildCtx {
                    // graph + dp
                    const CcData* cc{nullptr};
                    BlockData* blk{nullptr};
                    const ogdf::Graph* T{nullptr};
                    ogdf::EdgeArray<EdgeDP>* edge_dp{nullptr};
                    ogdf::NodeArray<NodeDPState>* node_dp{nullptr};

                    // indices
                    ogdf::EdgeArray<int> edgeTaskIdx;   // -> index dans tasks
                    ogdf::EdgeArray<int> edgeOrdinal;   // 0..nEdgeTasks-1
                    ogdf::NodeArray<int> nodeTaskIdx;   // -> index dans tasks

                    // orientation
                    ogdf::NodeArray<ogdf::node> parent;
                    ogdf::NodeArray<ogdf::edge> upEdge;
                    ogdf::NodeArray<std::vector<ogdf::node>> children;

                    // dépendances dp
                    std::unique_ptr<std::atomic<int>[]> edgeDeps; // taille nEdgeTasks
                    std::unique_ptr<std::atomic<int>[]> nodeDeps; // taille nNodes
                    int nNodes{0};
                    int nEdgeTasks{0};
                    int edgeBase{0};

                    // gating node -> parent
                    std::unique_ptr<std::atomic<unsigned char>[]> parentReady;
                    std::unique_ptr<std::atomic<unsigned char>[]> depsZero;

                    // solve tasks indices
                    std::vector<int> solveS_idx;
                    std::vector<int> solveP_idx;
                    std::vector<int> solveRR_idx;

                    // tasks + pile
                    std::vector<TaskRec> tasks;
                    ReadyStack ready;

                    // compteurs globaux
                    std::atomic<int> inflight{0};
                    std::atomic<int> nodesRemaining{0};
                    std::atomic<bool> solvePublished{false};
                };

                inline void buildParentChildren(BuildCtx& C) {
                    const ogdf::Graph& T = *C.T;
                    ogdf::node root = C.blk->spqr->rootNode();

                    for (ogdf::node v : T.nodes) {
                        C.parent[v] = nullptr;
                        C.upEdge[v] = nullptr;
                        C.children[v].clear();
                    }
                    C.parent[root] = root;

                    std::queue<ogdf::node> q;
                    q.push(root);
                    while (!q.empty()) {
                        ogdf::node u = q.front(); q.pop();
                        for (auto adj : u->adjEntries) {
                            ogdf::node w = adj->twinNode();
                            if (C.parent[w] != nullptr) continue;
                            C.parent[w] = u;
                            C.upEdge[w] = adj->theEdge();
                            C.children[u].push_back(w);
                            q.push(w);
                        }
                    }
                }

                inline void buildTasks(BuildCtx& C) {
                    const ogdf::Graph& T = *C.T;
                    C.nNodes = (int)T.numberOfNodes();
                    C.nEdgeTasks = C.nNodes - 1;

                    int sCount = 0, pCount = 0, rrCount = 0;
                    for (ogdf::node v : T.nodes) {
                        auto tp = C.blk->spqr->typeOf(v);
                        if (tp == ogdf::SPQRTree::NodeType::SNode) ++sCount;
                        else if (tp == ogdf::SPQRTree::NodeType::PNode) ++pCount;
                    }
                    for (ogdf::edge e : T.edges) {
                        auto t0 = C.blk->spqr->typeOf(e->source());
                        auto t1 = C.blk->spqr->typeOf(e->target());
                        if (t0 == ogdf::SPQRTree::NodeType::RNode && t1 == ogdf::SPQRTree::NodeType::RNode) ++rrCount;
                    }

                    int totalTasks = C.nNodes + C.nEdgeTasks + sCount + pCount + rrCount;
                    C.tasks.clear();
                    C.tasks.resize(totalTasks);

                    C.nodeTaskIdx.init(T, -1);
                    int cur = 0;
                    for (ogdf::node v : T.nodes) {
                        C.nodeTaskIdx[v] = cur;
                        TaskRec &tr = C.tasks[cur++];
                        tr.kind = TaskKind::Node; tr.v = v;
                        tr.e = nullptr; tr.parent = nullptr; tr.child = nullptr;
                        tr.next.store(-1, std::memory_order_relaxed);
                    }
                    C.nodesRemaining.store(C.nNodes, std::memory_order_relaxed);

                    C.edgeBase = cur;
                    C.edgeTaskIdx.init(T, -1);
                    C.edgeOrdinal.init(T, -1);
                    int ordinal = 0;
                    for (ogdf::node v : T.nodes) {
                        if (C.parent[v] == v) continue;
                        ogdf::edge e = C.upEdge[v];
                        C.edgeOrdinal[e] = ordinal++;

                        C.edgeTaskIdx[e] = cur;
                        TaskRec &tr = C.tasks[cur++];
                        tr.kind = TaskKind::Edge; tr.e = e;
                        tr.v = nullptr; tr.parent = C.parent[v]; tr.child = v;
                        tr.next.store(-1, std::memory_order_relaxed);
                    }

                    C.edgeDeps.reset(new std::atomic<int>[C.nEdgeTasks]);
                    for (int i = 0; i < C.nEdgeTasks; ++i) C.edgeDeps[i].store(0, std::memory_order_relaxed);
                    for (ogdf::node v : T.nodes) {
                        if (C.parent[v] == v) continue;
                        ogdf::edge e = C.upEdge[v];
                        int ord = C.edgeOrdinal[e];
                        C.edgeDeps[ord].store((int)C.children[v].size(), std::memory_order_relaxed);
                    }

                    C.nodeDeps.reset(new std::atomic<int>[C.nNodes]);
                    for (ogdf::node v : T.nodes) {
                        int idx = C.nodeTaskIdx[v];
                        C.nodeDeps[idx].store((int)v->degree(), std::memory_order_relaxed);
                    }

                    C.solveS_idx.clear(); C.solveP_idx.clear(); C.solveRR_idx.clear();
                    for (ogdf::node v : T.nodes) {
                        auto tp = C.blk->spqr->typeOf(v);
                        if (tp == ogdf::SPQRTree::NodeType::SNode) {
                            int id = cur++;
                            TaskRec &tr = C.tasks[id];
                            tr.kind = TaskKind::SolveS; tr.v = v;
                            tr.e = nullptr; tr.parent = nullptr; tr.child = nullptr;
                            tr.next.store(-1, std::memory_order_relaxed);
                            C.solveS_idx.push_back(id);
                        } else if (tp == ogdf::SPQRTree::NodeType::PNode) {
                            int id = cur++;
                            TaskRec &tr = C.tasks[id];
                            tr.kind = TaskKind::SolveP; tr.v = v;
                            tr.e = nullptr; tr.parent = nullptr; tr.child = nullptr;
                            tr.next.store(-1, std::memory_order_relaxed);
                            C.solveP_idx.push_back(id);
                        }
                    }
                    for (ogdf::edge e : T.edges) {
                        auto t0 = C.blk->spqr->typeOf(e->source());
                        auto t1 = C.blk->spqr->typeOf(e->target());
                        if (t0 == ogdf::SPQRTree::NodeType::RNode && t1 == ogdf::SPQRTree::NodeType::RNode) {
                            int id = cur++;
                            TaskRec &tr = C.tasks[id];
                            tr.kind = TaskKind::SolveRR; tr.e = e;
                            tr.v = nullptr; tr.parent = nullptr; tr.child = nullptr;
                            tr.next.store(-1, std::memory_order_relaxed);
                            C.solveRR_idx.push_back(id);
                        }
                    }

                    // Gating parentReady/depsZero vecteurs
                    C.parentReady.reset(new std::atomic<unsigned char>[C.nNodes]);
                    C.depsZero.reset(new std::atomic<unsigned char>[C.nNodes]);
                    for (int i = 0; i < C.nNodes; ++i) {
                        C.parentReady[i].store(0, std::memory_order_relaxed);
                        C.depsZero[i].store(0, std::memory_order_relaxed);
                    }
                    {
                        ogdf::node root = C.blk->spqr->rootNode();
                        int rootIdx = C.nodeTaskIdx[root];
                        C.parentReady[rootIdx].store(1, std::memory_order_relaxed);
                    }

                    C.ready.init(&C.tasks);
                    int pushed = 0;
                    for (ogdf::node v : T.nodes) {
                        if (C.parent[v] == v) continue;
                        ogdf::edge e = C.upEdge[v];
                        int ord = C.edgeOrdinal[e];
                        if (C.edgeDeps[ord].load(std::memory_order_relaxed) == 0) {
                            int eidx = C.edgeTaskIdx[e];
                            C.ready.push(eidx);
                            ++pushed;
                        }
                    }
                    C.inflight.store(pushed, std::memory_order_relaxed);
                }

                inline void publishSolveTasks(BuildCtx& C) {
                    if (C.solvePublished.exchange(true, std::memory_order_acq_rel)) return;
                    int cnt = 0;
                    for (int idx : C.solveS_idx) { C.ready.push(idx); ++cnt; }
                    for (int idx : C.solveP_idx) { C.ready.push(idx); ++cnt; }
                    for (int idx : C.solveRR_idx) { C.ready.push(idx); ++cnt; }
                    C.inflight.fetch_add(cnt, std::memory_order_release);
                }

                inline void workerLoop(BuildCtx* C) {
                    auto& tasks = C->tasks;
                    auto& ready = C->ready;

                    for (;;) {
                        int idx = ready.pop();
                        if (idx < 0) {
                            if (C->inflight.load(std::memory_order_acquire) == 0) break;
                            PAUSE_SPIN();
                            continue;
                        }
                        TaskRec& t = tasks[idx];

                        switch (t.kind) {
                            case TaskKind::Edge: {
                                SPQRsolve::processEdgeAB(t.e, t.parent, t.child, *C->edge_dp, *C->cc, *C->blk);

                                // Parent
                                {
                                    int nidxP = C->nodeTaskIdx[t.parent];
                                    if (C->nodeDeps[nidxP].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                                        C->depsZero[nidxP].store(1, std::memory_order_release);
                                        if (C->parentReady[nidxP].load(std::memory_order_acquire)) {
                                            C->ready.push(nidxP);
                                            C->inflight.fetch_add(1, std::memory_order_release);
                                        }
                                    }
                                }
                                // Child
                                {
                                    int nidxB = C->nodeTaskIdx[t.child];
                                    if (C->nodeDeps[nidxB].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                                        C->depsZero[nidxB].store(1, std::memory_order_release);
                                        if (C->parentReady[nidxB].load(std::memory_order_acquire)) {
                                            C->ready.push(nidxB);
                                            C->inflight.fetch_add(1, std::memory_order_release);
                                        }
                                    }
                                }

                                // Edge up(parent)
                                if (C->parent[t.parent] != t.parent) {
                                    ogdf::edge up = C->upEdge[t.parent];
                                    if (up) {
                                        int ordUp = C->edgeOrdinal[up];
                                        if (C->edgeDeps[ordUp].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                                            int upTask = C->edgeTaskIdx[up];
                                            C->ready.push(upTask);
                                            C->inflight.fetch_add(1, std::memory_order_release);
                                        }
                                    }
                                }

                                C->inflight.fetch_sub(1, std::memory_order_release);
                                break;
                            }

                            case TaskKind::Node: {
                                SPQRsolve::processNode_threadsafe(t.v, *C->edge_dp, *C->cc, *C->blk);

                                // Marquer enfants "parentReady" et publier si depsZero déjà vrai
                                for (ogdf::node c : C->children[t.v]) {
                                    int nidxC = C->nodeTaskIdx[c];
                                    C->parentReady[nidxC].store(1, std::memory_order_release);
                                    if (C->depsZero[nidxC].load(std::memory_order_acquire)) {
                                        C->ready.push(nidxC);
                                        C->inflight.fetch_add(1, std::memory_order_release);
                                    }
                                }

                                if (C->nodesRemaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                                    publishSolveTasks(*C);
                                }

                                C->inflight.fetch_sub(1, std::memory_order_release);
                                break;
                            }

                            case TaskKind::SolveS:
                                SPQRsolve::solveS(t.v, *C->node_dp, *C->edge_dp, *C->blk, *C->cc);
                                C->inflight.fetch_sub(1, std::memory_order_release);
                                break;
                            case TaskKind::SolveP:
                                SPQRsolve::solveP(t.v, *C->node_dp, *C->edge_dp, *C->blk, *C->cc);
                                C->inflight.fetch_sub(1, std::memory_order_release);
                                break;
                            case TaskKind::SolveRR:
                                SPQRsolve::solveRR(t.e, *C->node_dp, *C->edge_dp, *C->blk, *C->cc);
                                C->inflight.fetch_sub(1, std::memory_order_release);
                                break;
                        }
                    }

                    solver::snarls::flushThreadLocalSnarls();
                }


            } // namespace WS

            // Entrée publique: solveur SPQR work-stealing
            inline void solveSPQR_WS(BlockData& blk, const CcData& cc, size_t nthreads) {
                if (!blk.spqr) return;
                if (blk.Gblk->numberOfNodes() < 3) return;

                const ogdf::Graph &T = blk.spqr->tree();
                ogdf::EdgeArray<EdgeDP> edge_dp(T);
                ogdf::NodeArray<NodeDPState> node_dp(T);

                WS::BuildCtx Ctx;
                Ctx.cc = &cc;
                Ctx.blk = &blk;
                Ctx.T = &T;
                Ctx.edge_dp = &edge_dp;
                Ctx.node_dp = &node_dp;

                Ctx.edgeTaskIdx.init(T, -1);
                Ctx.edgeOrdinal.init(T, -1);
                Ctx.nodeTaskIdx.init(T, -1);
                Ctx.parent.init(T, nullptr);
                Ctx.upEdge.init(T, nullptr);
                Ctx.children.init(T);

                WS::buildParentChildren(Ctx);
                WS::buildTasks(Ctx);

                if (nthreads == 0) nthreads = 1;
                std::vector<std::thread> pool;
                pool.reserve(nthreads);
                for (size_t i = 0; i < nthreads; ++i) {
                    pool.emplace_back(WS::workerLoop, &Ctx);
                }
                for (auto &th : pool) th.join();
            }


            
        }



        void findTips(CcData& cc) {
            PROFILE_FUNCTION();
            for(node v : cc.Gcc->nodes) {
                int plusCnt = 0, minusCnt = 0;
                node vG = cc.nodeToOrig[v];

                for(auto adjE: v->adjEntries) {
                    ogdf::edge e = cc.edgeToOrig[adjE->theEdge()];
                    EdgePartType eType = getNodeEdgeType(vG, e);
                    if(eType == EdgePartType::PLUS) plusCnt++;
                    else minusCnt++;
                    // if(e->source() == vG) {
                    //     EdgePartType outType = ctx()._edge2types(e).first;
                    //     if(outType == EdgePartType::PLUS) plusCnt++;
                    //     else minusCnt++;
                    // } else {
                    //     EdgePartType outType = ctx()._edge2types(e).second;
                    //     if(outType == EdgePartType::PLUS) plusCnt++;
                    //     else minusCnt++;
                    // }
                }

                // std::cout << "Node " << ctx().node2name[cc.nodeToOrig[v]] << " has " << plusCnt << " plus outgoing and " << minusCnt << " minus outgoing edges" << std::endl;

                if(plusCnt + minusCnt == 0) {
                    isolatedNodesCnt++;
                }

                // if(ctx().node2name[cc.nodeToOrig[v]] == "3497") std::cout << ctx().node2name[cc.nodeToOrig[v]] << " has " << plusCnt << " plus and " << minusCnt << " minus outgoing edges" << std::endl;
                if(plusCnt == 0 || minusCnt == 0) {
                    cc.isTip[v] = true;
                } else {
                    cc.isTip[v] = false;
                }


            }

            // for(node v : cc.Gcc->nodes) {
            //     if(cc.isTip[v]) {
            //         std::cout << "Node " << ctx().node2name[cc.nodeToOrig[v]] << " is a tip" << std::endl;
            //     }
            // }

        }


        void processCutNodes(CcData& cc) {
            PROFILE_FUNCTION();


            // std::cout << "processCutNodes.." << std::endl;

            // std::cout << cc.bc->numberOfCComps() << " " << cc.bc->numberOfBComps() << std::endl;

            for(node v : cc.Gcc->nodes) {

                // std::cout << "Node " << ctx().node2name[cc.nodeToOrig[v]] << " in bc tree node is " << (cc.bc->typeOfGNode(v) == BCTree::GNodeType::CutVertex ? "cut vertex" : "block vertex")  << std::endl;

                // std::cout << 1 << std::endl;
                // std::cout << v << std::endl;
                if(cc.bc->typeOfGNode(v) == BCTree::GNodeType::CutVertex) {
                    // std::cout << 2 << std::endl;
                    cc.isCutNode[v] = true;

                    bool isGood = true;
                    // std::cout << 3 << std::endl;

                    // if(cc.isCutNode[v]) {
                    ogdf::node vT = cc.bc->bcproper(v);
                    // std::cout << "Cut node " << ctx().node2name[cc.nodeToOrig[v]] << " in bc tree node has " << vT->adjEntries.size() << " adj blocks"  << std::endl;
                    for(auto adjV : vT->adjEntries) {
                        node uT = adjV->twinNode();
                        std::vector<ogdf::edge> outPlus, outMinus;
                        getOutgoingEdgesInBlock(cc, v, uT, EdgePartType::PLUS, outPlus);
                        getOutgoingEdgesInBlock(cc, v, uT, EdgePartType::MINUS, outMinus);
                        // std::cout << cc.nodeToOrig[v] << " in block " << uT << " has " << outPlus.size() << " plus outgoing and " << outMinus.size() << " minus outgoing edges" << std::endl;
                        // std::cout << vT << " " << uT << std::endl;
                        // std::cout << "  In block with " << uT->adjEntries.size() << " adj entries, node " << ctx().node2name[cc.nodeToOrig[v]] << " has " << outPlus.size() << " pluses outgoing and " << outMinus.size() << " minus outgoing edges" << std::endl;

                        // if(outPlus.size() == 0 && outMinus.size() > 0) {
                        //     cc.cutToBlocks[v][0].push_back(uT);
                        // }

                        // if(outPlus.size() > 0 && outMinus.size() == 0) {
                        //     cc.cutToBlocks[v][1].push_back(uT);
                        // }

                        if(outPlus.size() > 0 && outMinus.size() > 0) {
                            isGood = false;
                            cc.lastBad[v] = uT;
                            cc.badCutCount[v]++;
                            // cc.cutToBlocks[v][2].push_back(uT);
                        }
                    }
                    // }
                    cc.isGoodCutNode[v] = isGood;
                    // if(isGood) {
                    //     std::cout << "Good cut node: " << ctx().node2name[cc.nodeToOrig[v]] << std::endl;
                    // }
                    // std::cout << "Cut node " << ctx().node2name[cc.nodeToOrig[v]] << " is " << (isGood ? "good" : "bad") << std::endl;
                }
            }



            {
                for(node v : cc.Gcc->nodes) {
                    if(cc.bc->typeOfGNode(v) != BCTree::GNodeType::CutVertex) continue;


                    // std::cout << "Cut node " << ctx().node2name[cc.nodeToOrig[v]] << " is " << (cc.isGoodCutNode[v] ? "good" : "bad") << " and has " << cc.badCutCount[v] << " bad blocks" << std::endl;
                }
            }
        }


        void findCutSnarl(CcData &cc) {
            auto &C = ctx();
            const char* env = std::getenv("SBFIND_LOG");
            const bool LOG = (env && env[0] != '0');

            auto getenvNum = [](const char* name, size_t defVal) -> size_t {
                const char* s = std::getenv(name);
                if (!s) return defVal;
                char* end = nullptr;
                long long v = std::strtoll(s, &end, 10);
                if (end == s || v < 0) return defVal;
                return static_cast<size_t>(v);
            };
            const size_t BIG_STATES = getenvNum("SBFIND_BIG_STATES", 200000);
            const size_t BIG_TIPS   = getenvNum("SBFIND_BIG_TIPS",   20000);
            const size_t EXPAND_UPTO = getenvNum("SBFIND_EXPAND_TIP_PAIRS_UPTO", 0); // 0 => jamais

            static std::mutex logMutex;

            auto tStart = std::chrono::high_resolution_clock::now();

            // Diagnostics CC
            size_t Vn = cc.Gcc->numberOfNodes();
            size_t En = cc.Gcc->numberOfEdges();
            size_t loopsPresent = 0;
            for (ogdf::edge eC : cc.Gcc->edges) if (eC->source() == eC->target()) ++loopsPresent;

            // Précompte deg+ / deg- (ignorer self-loop)
            ogdf::NodeArray<int> plusDeg(*cc.Gcc, 0), minusDeg(*cc.Gcc, 0);
            for (ogdf::node vCc : cc.Gcc->nodes) {
                int p = 0, m = 0;
                for (auto adjE : vCc->adjEntries) {
                    if (adjE->twinNode() == vCc) continue;
                    ogdf::edge eC    = adjE->theEdge();
                    ogdf::edge eOrig = cc.edgeToOrig[eC];
                    EdgePartType tAtV = getNodeEdgeType(cc.nodeToOrig[vCc], eOrig);
                    if      (tAtV == EdgePartType::PLUS)  ++p;
                    else if (tAtV == EdgePartType::MINUS) ++m;
                }
                plusDeg[vCc]  = p;
                minusDeg[vCc] = m;
            }

            auto isSplit = [&](ogdf::node vCc) -> bool {
                return cc.isCutNode[vCc] && cc.isGoodCutNode[vCc];
            };

            size_t cutNodes = 0, goodCutNodes = 0, splitCandidates = 0;
            for (ogdf::node vCc : cc.Gcc->nodes) {
                if (cc.isCutNode[vCc]) ++cutNodes;
                if (cc.isGoodCutNode[vCc]) ++goodCutNodes;
                if (isSplit(vCc)) ++splitCandidates;
            }

            if (LOG) {
                std::lock_guard<std::mutex> lk(logMutex);
                std::cout << "[findCutSnarl] CC summary: |V|=" << Vn
                        << " |E|=" << En
                        << " self-loops present=" << loopsPresent
                        << " cutNodes=" << cutNodes
                        << " goodCutNodes=" << goodCutNodes
                        << " splitCandidates=" << splitCandidates
                        << std::endl;
            }

            // Espace sign-cut: visited par mode (bits: 1=UNIFY, 2=PLUS, 4=MINUS)
            ogdf::NodeArray<uint8_t> visited(*cc.Gcc, 0);

            struct State { ogdf::node v; uint8_t mode; }; // 0=UNIFY, 1=PLUS, 2=MINUS

            auto label_of = [&](ogdf::node vCc, EdgePartType t) -> std::string {
                return C.node2name[cc.nodeToOrig[vCc]] + (t == EdgePartType::PLUS ? "+" : "-");
            };

            auto modeMask = [](uint8_t mode) -> uint8_t {
                return (mode == 0 ? 1u : (mode == 1 ? 2u : 4u));
            };

            auto maybe_add_tip = [&](ogdf::node vCc, uint8_t mode,
                                    std::vector<std::pair<ogdf::node, EdgePartType>>& compTips) {
                if (C.node2name[cc.nodeToOrig[vCc]] == "_trash") return;

                if (isSplit(vCc)) {
                    if (mode == 1 && plusDeg[vCc]  > 0) compTips.emplace_back(vCc, EdgePartType::PLUS);
                    if (mode == 2 && minusDeg[vCc] > 0) compTips.emplace_back(vCc, EdgePartType::MINUS);
                } else {
                    int p = plusDeg[vCc], m = minusDeg[vCc];
                    if (p + m == 0) return;
                    if (p == 0 && m > 0) compTips.emplace_back(vCc, EdgePartType::MINUS);
                    else if (m == 0 && p > 0) compTips.emplace_back(vCc, EdgePartType::PLUS);
                }
            };

            size_t compsExplored = 0;
            size_t totalStatesVisited = 0;
            size_t maxStatesVisited = 0;
            size_t maxTipsInComp = 0;

            auto bfs_component = [&](ogdf::node startV, uint8_t startMode) {
                std::queue<State> q;
                visited[startV] |= modeMask(startMode);
                q.push({startV, startMode});

                size_t statesInComp = 0;
                size_t adjScanned   = 0;
                size_t edgesFilteredBySign = 0;

                std::vector<std::pair<ogdf::node, EdgePartType>> compTips;
                compTips.reserve(256);

                maybe_add_tip(startV, startMode, compTips);

                while (!q.empty()) {
                    State s = q.front(); q.pop();
                    ++statesInComp;

                    for (auto adjE : s.v->adjEntries) {
                        ++adjScanned;

                        ogdf::node w = adjE->twinNode();
                        if (w == s.v) continue;

                        ogdf::edge eC    = adjE->theEdge();
                        ogdf::edge eOrig = cc.edgeToOrig[eC];

                        EdgePartType tAtV = getNodeEdgeType(cc.nodeToOrig[s.v], eOrig);
                        if (s.mode == 1 && tAtV != EdgePartType::PLUS)  { ++edgesFilteredBySign; continue; }
                        if (s.mode == 2 && tAtV != EdgePartType::MINUS) { ++edgesFilteredBySign; continue; }

                        EdgePartType tAtW = getNodeEdgeType(cc.nodeToOrig[w], eOrig);
                        uint8_t nextMode = isSplit(w) ? (tAtW == EdgePartType::PLUS ? 1 : 2) : 0;
                        uint8_t mask = modeMask(nextMode);

                        if ((visited[w] & mask) == 0) {
                            visited[w] |= mask;
                            q.push({w, nextMode});
                            maybe_add_tip(w, nextMode, compTips);
                        }
                    }
                }

                ++compsExplored;
                totalStatesVisited += statesInComp;
                maxStatesVisited = std::max(maxStatesVisited, statesInComp);
                maxTipsInComp = std::max(maxTipsInComp, compTips.size());

                if (LOG && (statesInComp >= BIG_STATES || compTips.size() >= BIG_TIPS)) {
                    std::lock_guard<std::mutex> lk(logMutex);
                    std::cout << "[findCutSnarl] BIG component: states=" << statesInComp
                            << " tips=" << compTips.size()
                            << " adjScanned=" << adjScanned
                            << " filteredBySign=" << edgesFilteredBySign
                            << std::endl;

                    size_t show = std::min<size_t>(compTips.size(), 16);
                    if (show > 0) {
                        std::cout << "  sample tips: ";
                        for (size_t i = 0; i < show; ++i) {
                            std::cout << label_of(compTips[i].first, compTips[i].second)
                                    << (i + 1 < show ? ", " : "");
                        }
                        std::cout << std::endl;
                    }
                }

                // Émission: compacte par défaut
                if (compTips.size() >= 2) {
                    if (EXPAND_UPTO > 0 && compTips.size() <= EXPAND_UPTO) {
                        // Déroulage complet (petits groupes, debug)
                        for (size_t i = 0; i + 1 < compTips.size(); ++i) {
                            for (size_t j = i + 1; j < compTips.size(); ++j) {
                                const auto &a = compTips[i];
                                const auto &b = compTips[j];
                                std::vector<std::string> sn;
                                sn.reserve(2);
                                sn.push_back(label_of(a.first, a.second));
                                sn.push_back(label_of(b.first, b.second));
                                addSnarl(std::move(sn));
                            }
                        }
                    } else {
                        // Mode compact
                        std::vector<std::string> labels;
                        labels.reserve(compTips.size());
                        for (const auto &tp : compTips) {
                            labels.push_back(label_of(tp.first, tp.second));
                        }
                        addSnarl(std::move(labels));
                    }
                }
            };

            // Parcours de l’espace sign-cut
            for (ogdf::node v : cc.Gcc->nodes) {
                if (isSplit(v)) {
                    if (plusDeg[v]  > 0 && (visited[v] & modeMask(1)) == 0) bfs_component(v, 1);
                    if (minusDeg[v] > 0 && (visited[v] & modeMask(2)) == 0) bfs_component(v, 2);
                } else {
                    if ((visited[v] & modeMask(0)) == 0) bfs_component(v, 0);
                }
            }

            auto tEnd = std::chrono::high_resolution_clock::now();
            auto durMs = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();

            if (LOG) {
                std::lock_guard<std::mutex> lk(logMutex);
                std::cout << "[findCutSnarl] CC done: comps=" << compsExplored
                        << " totalStates=" << totalStatesVisited
                        << " maxStates=" << maxStatesVisited
                        << " maxTipsInComp=" << maxTipsInComp
                        << " time=" << durMs << " ms"
                        << std::endl;
            }
        }



        void buildBlockData(BlockData& blk, CcData& cc) {
            PROFILE_FUNCTION();

            auto envOn = []() -> bool {
                const char* s = std::getenv("SBFIND_LOG");
                return s && s[0] != '0';
            };
            const bool LOG = envOn();
            static std::mutex logMutex;

            auto tStart = std::chrono::high_resolution_clock::now();

            {
                blk.Gblk = std::make_unique<ogdf::Graph>();
            }

            {
                blk.nodeToOrig.init(*blk.Gblk, nullptr);
                blk.edgeToOrig.init(*blk.Gblk, nullptr);
                blk.toCc.init(*blk.Gblk, nullptr);
            }

            std::unordered_set<ogdf::node> verts;
            verts.reserve(256);

            // Collecter les sommets CC qui appartiennent à ce block-node (bNode)
            for (ogdf::edge hE : cc.bc->hEdges(blk.bNode)) {
                ogdf::edge eC = cc.bc->original(hE);
                verts.insert(eC->source());
                verts.insert(eC->target());
            }

            std::unordered_map<ogdf::node, ogdf::node> cc_to_blk;
            cc_to_blk.reserve(verts.size());

            // Créer les nœuds du bloc
            for (ogdf::node vCc : verts) {
                ogdf::node vB = blk.Gblk->newNode();
                cc_to_blk[vCc] = vB;
                blk.toCc[vB] = vCc;
                ogdf::node vG = cc.nodeToOrig[vCc];
                blk.nodeToOrig[vB] = vG;
            }

            // Créer les arêtes du bloc
            size_t edgesAdded = 0;
            for (ogdf::edge hE : cc.bc->hEdges(blk.bNode)) {
                ogdf::edge eCc = cc.bc->original(hE);
                auto srcIt = cc_to_blk.find(eCc->source());
                auto tgtIt = cc_to_blk.find(eCc->target());
                if (srcIt != cc_to_blk.end() && tgtIt != cc_to_blk.end()) {
                    ogdf::edge e = blk.Gblk->newEdge(srcIt->second, tgtIt->second);
                    blk.edgeToOrig[e] = cc.edgeToOrig[eCc];
                    ++edgesAdded;
                }
            }

            auto tAfterGraph = std::chrono::high_resolution_clock::now();

            // Construire l'arbre SPQR si bloc assez grand
            if (blk.Gblk->numberOfNodes() >= 3) {
                {
                    PROFILE_BLOCK("buildBlockData:: build SPQR tree");
                    blk.spqr = std::make_unique<ogdf::StaticSPQRTree>(*blk.Gblk);
                }

                const ogdf::Graph& T = blk.spqr->tree();

                blk.skel2tree.reserve(2 * T.numberOfEdges());
                blk.parent.init(T, nullptr);

                ogdf::node root = blk.spqr->rootNode();
                blk.parent[root] = root;

                for (ogdf::edge te : T.edges) {
                    ogdf::node u = te->source();
                    ogdf::node v = te->target();
                    blk.parent[v] = u;

                    if (auto eSrc = blk.spqr->skeletonEdgeSrc(te)) {
                        blk.skel2tree[eSrc] = te;
                    }
                    if (auto eTgt = blk.spqr->skeletonEdgeTgt(te)) {
                        blk.skel2tree[eTgt] = te;
                    }
                }

                if (LOG) {
                    std::lock_guard<std::mutex> lk(logMutex);
                    std::cout << "[buildBlockData] block built: |V_blk|=" << blk.Gblk->numberOfNodes()
                            << " |E_blk|=" << blk.Gblk->numberOfEdges()
                            << " edgesAdded=" << edgesAdded
                            << " SPQR nodes=" << T.numberOfNodes()
                            << " SPQR edges=" << T.numberOfEdges()
                            << " P=" << blk.spqr->numberOfPNodes()
                            << " S=" << blk.spqr->numberOfSNodes()
                            << " R=" << blk.spqr->numberOfRNodes()
                            << std::endl;
                }
            } else {
                if (LOG) {
                    std::lock_guard<std::mutex> lk(logMutex);
                    std::cout << "[buildBlockData] small block: |V_blk|=" << blk.Gblk->numberOfNodes()
                            << " |E_blk|=" << blk.Gblk->numberOfEdges()
                            << " edgesAdded=" << edgesAdded
                            << " (no SPQR built)"
                            << std::endl;
                }
            }

            auto tEnd = std::chrono::high_resolution_clock::now();
            long long msGraph = std::chrono::duration_cast<std::chrono::milliseconds>(tAfterGraph - tStart).count();
            long long msTotal = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();

            if (LOG) {
                std::lock_guard<std::mutex> lk(logMutex);
                std::cout << "[buildBlockData] timing: graph=" << msGraph << " ms, total=" << msTotal << " ms" << std::endl;
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
            std::vector<std::vector<ogdf::node>>* bucket = targs->bucket;
            std::vector<std::vector<ogdf::edge>>* edgeBuckets = targs->edgeBuckets;

            auto envOn = []() -> bool {
                const char* s = std::getenv("SBFIND_LOG");
                return s && s[0] != '0';
            };
            const bool LOG = envOn();
            static std::mutex logMutex;

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

                    size_t loopsSkipped = 0;
                    size_t edgesAdded   = 0;

                    {
                        PROFILE_BLOCK("solve:: rebuild cc graph");
                        (*components)[cid]->Gcc = std::make_unique<ogdf::Graph>();
                        (*components)[cid]->nodeToOrig.init(*(*components)[cid]->Gcc, nullptr);
                        (*components)[cid]->edgeToOrig.init(*(*components)[cid]->Gcc, nullptr);
                        (*components)[cid]->isTip.init(*(*components)[cid]->Gcc, false);
                        (*components)[cid]->isCutNode.init(*(*components)[cid]->Gcc, false);
                        (*components)[cid]->isGoodCutNode.init(*(*components)[cid]->Gcc, false);
                        (*components)[cid]->lastBad.init(*(*components)[cid]->Gcc, nullptr);
                        (*components)[cid]->badCutCount.init(*(*components)[cid]->Gcc, 0);

                        std::unordered_map<ogdf::node, ogdf::node> orig_to_cc;
                        orig_to_cc.reserve((*bucket)[cid].size());

                        for (ogdf::node vG : (*bucket)[cid]) {
                            ogdf::node vC = (*components)[cid]->Gcc->newNode();
                            (*components)[cid]->nodeToOrig[vC] = vG;
                            orig_to_cc[vG] = vC;
                        }

                        for (ogdf::edge e : (*edgeBuckets)[cid]) {
                            if (e->source() == e->target()) {
                                ++loopsSkipped; // self-loop ignorée
                                continue;
                            }
                            auto eC = (*components)[cid]->Gcc->newEdge(
                                orig_to_cc[e->source()],
                                orig_to_cc[e->target()]
                            );
                            (*components)[cid]->edgeToOrig[eC] = e;
                            ++edgesAdded;
                        }
                    }

                    if (LOG) {
                        std::lock_guard<std::mutex> lk(logMutex);
                        std::cout << "[worker_component] T" << tid
                                << " CC#" << cid
                                << " |V|=" << (*components)[cid]->Gcc->numberOfNodes()
                                << " |E|=" << (*components)[cid]->Gcc->numberOfEdges()
                                << " self-loops skipped=" << loopsSkipped
                                << " edges added=" << edgesAdded
                                << std::endl;
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

            {
                std::lock_guard<std::mutex> lk(logMutex);
                std::cout << "Thread " << tid << " built " << processed << " components (rebuild cc graph)" << std::endl;
            }
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

            //size_t chunkSize = std::max<size_t>(1, nCC / numThreads);
            size_t chunkSize = 1;
            size_t processed = 0;

            while (true) {
                size_t startIndex, endIndex;
                // std::cout << chunkSize << std::endl;
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
                    chunkSize = std::min(chunkSize * 2, static_cast<size_t>(nCC / numThreads));
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

            std::cout << "Thread " << tid << " built " << processed << " components (cuts tips)" << std::endl;

            flushThreadLocalSnarls();

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

            auto envOn = []() -> bool {
                const char* s = std::getenv("SBFIND_LOG");
                return s && s[0] != '0';
            };
            auto getenvNum = [](const char* name, size_t defVal) -> size_t {
                const char* s = std::getenv(name);
                if (!s) return defVal;
                char* end = nullptr;
                long long v = std::strtoll(s, &end, 10);
                if (end == s || v < 0) return defVal;
                return static_cast<size_t>(v);
            };
            const bool LOG = envOn();
            const size_t BIGV = getenvNum("SBFIND_BIG_BLOCK_V", 100000);
            const size_t BIGE = getenvNum("SBFIND_BIG_BLOCK_E", 100000);
            static std::mutex logMutex;

            size_t spqrThreads = []{
                if (const char* s = std::getenv("SBFIND_SPQR_THREADS")) {
                    long v = std::strtol(s, nullptr, 10);
                    if (v > 0) return static_cast<size_t>(v);
                }
                return std::max<size_t>(1, ctx().threads);
            }();

            size_t chunkSize = 1;
            size_t processed = 0;

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

                    auto t0 = std::chrono::high_resolution_clock::now();
                    buildBlockData(blk, *(*blockPreps)[bid].cc);
                    auto t1 = std::chrono::high_resolution_clock::now();

                    size_t Vb = blk.Gblk ? blk.Gblk->numberOfNodes() : 0;
                    size_t Eb = blk.Gblk ? blk.Gblk->numberOfEdges()  : 0;
                    long long msBuild = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

                    if (LOG && (Vb >= BIGV || Eb >= BIGE)) {
                        std::lock_guard<std::mutex> lk(logMutex);
                        std::cout << "[worker_block] T" << tid
                                << " Block#" << bid
                                << " |V_blk|=" << Vb
                                << " |E_blk|=" << Eb
                                << " buildBlockData=" << msBuild << " ms"
                                << " spqr=" << (blk.spqr ? "yes" : "no")
                                << std::endl;
                    }

                    if (blk.Gblk && blk.Gblk->numberOfNodes() >= 3) {
                        auto t2 = std::chrono::high_resolution_clock::now();
                        SPQRsolve::solveSPQR_WS(blk, *(*blockPreps)[bid].cc, spqrThreads);
                        auto t3 = std::chrono::high_resolution_clock::now();
                        long long msSPQR = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

                        if (LOG && (Vb >= BIGV || Eb >= BIGE)) {
                            std::lock_guard<std::mutex> lk(logMutex);
                            std::cout << "[worker_block] T" << tid
                                    << " Block#" << bid
                                    << " SPQRsolve_WS time=" << msSPQR << " ms"
                                    << std::endl;
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

            {
                std::lock_guard<std::mutex> lk(logMutex);
                std::cout << "Thread " << tid << " built " << processed << " components (blocks)" << std::endl;
            }

            flushThreadLocalSnarls();

            return nullptr;
        }



        void solve() {
            std::cout << "Finding snarls...\n";
            PROFILE_FUNCTION();
            auto& C = ctx();
            Graph& G = C.G;

            // break into wccs

            NodeArray<int> compIdx(G);
            int nCC;
            {
                PROFILE_BLOCK("solve:: ComputeCC");
                TIME_BLOCK("solve:: ComputeCC");
                nCC = connectedComponents(G, compIdx);
            }

            std::vector<std::vector<node>> bucket(nCC);
            {
                PROFILE_BLOCK("solve:: bucket nodes");
                TIME_BLOCK("solve:: bucket nodes");

                for (node v : G.nodes) {
                    bucket[compIdx[v]].push_back(v);
                }
            }

            std::vector<std::vector<edge>> edgeBuckets(nCC);

            {
                PROFILE_BLOCK("solve:: bucket edges");
                TIME_BLOCK("solve:: bucket edges");

                for (edge e : G.edges) {
                    edgeBuckets[compIdx[e->source()]].push_back(e);
                }
            }



            std::vector<std::unique_ptr<CcData>> components(nCC);

            // std::cout << "1231321" << std::endl;


            {
                // TIME_BLOCK("solveStreaming:: build BC trees and collect blocks");
                // PROFILE_BLOCK("solveStreaming:: build BC trees and collect blocks");
                PROFILE_BLOCK("solve:: building components");


                // std::cout << 123 << std::endl;
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


            // for (size_t i = 0; i < nCC; i++)
            // {
            //     auto cc = std::make_unique<CcData>();

            //     {
            //         PROFILE_BLOCK("solve:: rebuild cc graph");
            //         // TIME_BLOCK("solve:: rebuild cc graph");
            //         cc->Gcc = std::make_unique<Graph>();
            //         cc->nodeToOrig.init(*cc->Gcc, nullptr);
            //         cc->edgeToOrig.init(*cc->Gcc, nullptr);
            //         cc->isTip.init(*cc->Gcc, false);
            //         cc->isCutNode.init(*cc->Gcc, false);
            //         cc->isGoodCutNode.init(*cc->Gcc, false);
            //         cc->lastBad.init(*cc->Gcc, nullptr);
            //         cc->badCutCount.init(*cc->Gcc, 0);

            //         std::unordered_map<node, node> orig_to_cc;
            //         orig_to_cc.reserve(bucket[i].size());

            //         for (node vG : bucket[i]) {
            //             node vC = cc->Gcc->newNode();
            //             cc->nodeToOrig[vC] = vG;
            //             orig_to_cc[vG] = vC;
            //         }

            //         for (edge e : edgeBuckets[i]) {
            //             auto eC = cc->Gcc->newEdge(orig_to_cc[e->source()], orig_to_cc[e->target()]);
            //             cc->edgeToOrig[eC] = e;
            //         }
            //     }

            //     components[i] = std::move(cc);
            // }



            std::vector<BlockPrep> blockPreps;

            {
                // TIME_BLOCK("solveStreaming:: build BC trees and collect blocks");
                // PROFILE_BLOCK("solveStreaming:: build BC trees and collect blocks");
                // PROFILE_BLOCK("solve:: building data");
                PROFILE_BLOCK("solve:: building bc tree");


                // std::cout << 123 << std::endl;
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
            }


            // return;
            // building bc trees
            // for (int cid = 0; cid < nCC; ++cid) {
            //     PROFILE_BLOCK("solve:: building bc tree");
            //     // TIME_BLOCK("solve:: building bc tree");
            //     auto *cc = components[cid].get();
            //     cc->bc = std::make_unique<BCTree>(*cc->Gcc);
            //     // cc->auxToOriginal.init(cc->bc->auxiliaryGraph());
            //     // for(edge eGcc : cc->Gcc->edges) {
            //     //     cc->auxToOriginal[cc->bc->rep(eGcc)] = cc->edgeToOrig[eGcc];
            //     // }

            //     // GraphIO::drawGraph(cc->bc->bcTree(), "bcTree");

            // }





            // return ;
            std::cout << "built bc trees for " << nCC << " components\n";


            {
                // TIME_BLOCK("solveStreaming:: build BC trees and collect blocks");
                // PROFILE_BLOCK("solveStreaming:: build BC trees and collect blocks");
                PROFILE_BLOCK("solve:: processing tips/cuts");


                // std::cout << 123 << std::endl;
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



            // for (int cid = 0; cid < nCC; ++cid) {
            //     PROFILE_BLOCK("solve:: finding tips");
            //     // TIME_BLOCK("solve:: finding tips");
            //     auto *cc = components[cid].get();
            //     findTips(*cc);
            // }




            // for (int cid = 0; cid < nCC; ++cid) {
            //     PROFILE_BLOCK("solve:: process cut nodes");
            //     // TIME_BLOCK("solve:: process cut nodes");

            //     auto *cc = components[cid].get();
            //     if(cc->bc->numberOfCComps() > 0) {
            //         processCutNodes(*cc);
            //         // std::cout << cc->bc->numberOfBComps() << ", " << cc->bc->numberOfCComps() << std::endl;
            //     }
            // }

            // std::cout << "Isolated: " << isolatedNodesCnt << " nodes\n";



            // tip-tip finding in tree-cut components
            // for (int cid = 0; cid < nCC; ++cid) {
            //     PROFILE_BLOCK("solve:: finding tip-tip snarl candidates");
            //     // TIME_BLOCK("solve:: finding tip-tip snarl candidates");
            //     auto *cc = components[cid].get();
            //     // if(cc->bc->numberOfCComps() > 0)
            //     findCutSnarl(*cc);
            // }




            {
                // TIME_BLOCK("solveStreaming:: build BC trees and collect blocks");
                // PROFILE_BLOCK("solveStreaming:: build BC trees and collect blocks");
                PROFILE_BLOCK("solve:: solving in blocks");


                // std::cout << 123 << std::endl;
                size_t numThreads = std::thread::hardware_concurrency();
                // size_t numThreads = 16;
                numThreads = std::min({(size_t)C.threads, (size_t)blockPreps.size(), numThreads});

                std::vector<pthread_t> threads(numThreads);


                std::mutex workMutex;
                size_t nextIndex = 0;


                for (size_t tid = 0; tid < numThreads; ++tid) {
                    pthread_attr_t attr;
                    pthread_attr_init(&attr);

                    size_t stackSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;
                    pthread_attr_setstacksize(&attr, stackSize);


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



            // // inside of block
            // for (int cid = 0; cid < nCC; ++cid) {
            //     std::cout << "Processing " << cid << std::endl;
            //     PROFILE_BLOCK("solve:: building blocks data");
            //     // TIME_BLOCK("solve:: building blocks data");

            //     for(auto &bNode : components[cid]->bc->bcTree().nodes) {
            //         if(components[cid]->bc->typeOfBNode(bNode) == BCTree::BNodeType::CComp) continue;
            //         BlockData blk;
            //         blk.bNode = bNode;
            //         buildBlockData(blk, *components[cid]);
            //         //std::cout << "Added block with " << blk.spqr->numberOfSNodes() << " S nodes" << std::endl;
            //         components[cid]->blocks.push_back(std::move(blk));
            //     }
            // }

            // std::cout << "Done building blocks" << std::endl;

            // for (int cid = 0; cid < nCC; ++cid) {
            //     PROFILE_BLOCK("solve:: finding snarls inside blocks");
            //     for(auto &blk : components[cid]->blocks) {
            //         if(blk.Gblk->numberOfNodes() >= 3) {
            //             SPQRsolve::solveSPQR(blk, *components[cid]);
            //         }
            //         // checkBlockByCutVertices(*blk, *components[cid]);
            //     }
            // }
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