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


#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>
#include <unistd.h>
#ifdef __linux__
#include <cstdio>
#endif
#include <sys/resource.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>


bool VERBOSE = false;
#define VLOG if (VERBOSE) std::cerr

namespace metrics {

    enum class Phase : uint8_t { IO = 0, BUILD = 1, LOGIC = 2, COUNT = 3 };

    struct PhaseState {
        std::atomic<bool> running{false};
        std::atomic<size_t> baseline_rss{0};   // bytes
        std::atomic<size_t> peak_rss_delta{0}; // bytes
        std::atomic<uint64_t> start_ns{0};
        std::atomic<uint64_t> elapsed_ns{0};
    };

    inline uint64_t now_ns() {
        return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Current RSS in bytes
    inline size_t currentRSS() {
    #ifdef __linux__
        // /proc/self/statm: size resident shared text lib data dt
        // We read resident pages (2nd field) * page size
        FILE* f = std::fopen("/proc/self/statm", "r");
        if (!f) {
            // Fallback to ru_maxrss (note: ru_maxrss = max so far; not current)
            struct rusage ru{};
            getrusage(RUSAGE_SELF, &ru);
            // ru_maxrss in kilobytes on Linux; convert to bytes
            return (size_t)ru.ru_maxrss * 1024ull;
        }
        long rss_pages = 0;
        // Ignore first field
        long dummy = 0;
        if (std::fscanf(f, "%ld %ld", &dummy, &rss_pages) != 2) {
            std::fclose(f);
            struct rusage ru{};
            getrusage(RUSAGE_SELF, &ru);
            return (size_t)ru.ru_maxrss * 1024ull;
        }
        std::fclose(f);
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) page_size = 4096;
        return (size_t)rss_pages * (size_t)page_size;
    #else
        // Portable fallback: ru_maxrss is a max, not current; acceptable fallback
        struct rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        return (size_t)ru.ru_maxrss * 1024ull;
    #endif
    }

    inline std::array<PhaseState, (size_t)Phase::COUNT> &states() {
        static std::array<PhaseState, (size_t)Phase::COUNT> s;
        return s;
    }

    inline void beginPhase(Phase p) {
        auto &st = states()[(size_t)p];
        const uint64_t t0 = now_ns();
        const size_t base = currentRSS();
        st.baseline_rss.store(base, std::memory_order_relaxed);
        st.peak_rss_delta.store(0, std::memory_order_relaxed);
        st.start_ns.store(t0, std::memory_order_relaxed);
        st.running.store(true, std::memory_order_release);
    }

    inline void updateRSS(Phase p) {
        auto &st = states()[(size_t)p];
        if (!st.running.load(std::memory_order_acquire)) return;
        const size_t base = st.baseline_rss.load(std::memory_order_relaxed);
        const size_t cur = currentRSS();
        size_t delta = (cur >= base ? (cur - base) : 0);
        size_t prev = st.peak_rss_delta.load(std::memory_order_relaxed);
        while (delta > prev && !st.peak_rss_delta.compare_exchange_weak(prev, delta, std::memory_order_relaxed)) {
        }
    }

    inline void endPhase(Phase p) {
        auto &st = states()[(size_t)p];
        if (!st.running.load(std::memory_order_acquire)) return;
        updateRSS(p);
        const uint64_t t1 = now_ns();
        const uint64_t t0 = st.start_ns.load(std::memory_order_relaxed);
        const uint64_t d = (t1 >= t0 ? (t1 - t0) : 0);
        uint64_t prev = st.elapsed_ns.load(std::memory_order_relaxed);
        st.elapsed_ns.store(prev + d, std::memory_order_relaxed);
        st.running.store(false, std::memory_order_release);
    }

    struct Snapshot {
        uint64_t elapsed_ns;
        size_t peak_rss_delta;
    };
    inline Snapshot snapshot(Phase p) {
        auto &st = states()[(size_t)p];
        return Snapshot{
            st.elapsed_ns.load(std::memory_order_relaxed),
            st.peak_rss_delta.load(std::memory_order_relaxed)
        };
    }
} 

inline void METRICS_PHASE_BEGIN(metrics::Phase p) { metrics::beginPhase(p); }
inline void METRICS_PHASE_END(metrics::Phase p)   { metrics::endPhase(p);   }
inline void PHASE_RSS_UPDATE_IO()    { metrics::updateRSS(metrics::Phase::IO); }
inline void PHASE_RSS_UPDATE_BUILD() { metrics::updateRSS(metrics::Phase::BUILD); }
inline void PHASE_RSS_UPDATE_LOGIC() { metrics::updateRSS(metrics::Phase::LOGIC); }

using namespace ogdf;

static std::string g_report_json_path;




static void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " -g <graphFile> -o <outputFile> [--gfa] "
              << "[--superbubbles | --snarls] "
              << "[--block-stats <file>] "
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







size_t snarlsFound = 0;
size_t isolatedNodesCnt = 0;

static std::atomic<long long> g_ogdf_total_us{0};

static std::atomic<size_t> g_phase_io_max_rss{0};
static std::atomic<size_t> g_phase_build_max_rss{0};
static std::atomic<size_t> g_phase_logic_max_rss{0};

static inline void __phase_rss_update(std::atomic<size_t> &dst) {
    size_t cur = memtime::peakRSSBytes();
    size_t old = dst.load(std::memory_order_relaxed);
    while (cur > old && !dst.compare_exchange_weak(old, cur, std::memory_order_relaxed)) {}
}
#define PHASE_RSS_UPDATE_IO()    __phase_rss_update(g_phase_io_max_rss)
#define PHASE_RSS_UPDATE_BUILD() __phase_rss_update(g_phase_build_max_rss)
#define PHASE_RSS_UPDATE_LOGIC() __phase_rss_update(g_phase_logic_max_rss)

struct OgdfAcc {
    std::chrono::high_resolution_clock::time_point t0;
    OgdfAcc() : t0(std::chrono::high_resolution_clock::now()) {}
    ~OgdfAcc() {
        auto t1 = std::chrono::high_resolution_clock::now();
        g_ogdf_total_us.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
            std::memory_order_relaxed
        );
    }
};

#define OGDF_ACC_SCOPE() OgdfAcc __ogdf_acc_guard;

#define OGDF_EVAL(TAG, EXPR) \
    ([&]() -> decltype(EXPR) { \
        OGDF_ACC_SCOPE(); \
        MEM_TIME_BLOCK(TAG); \
        MARK_SCOPE_MEM(TAG); \
        PROFILE_BLOCK(TAG); \
        return (EXPR); \
    })()

#define OGDF_NEW_UNIQUE(TAG, T, ...) \
    ([&]() { \
        OGDF_ACC_SCOPE(); \
        MEM_TIME_BLOCK(TAG); \
        MARK_SCOPE_MEM(TAG); \
        PROFILE_BLOCK(TAG); \
        return std::make_unique<T>(__VA_ARGS__); \
    })()

#define OGDF_SCOPE(TAG) \
    OGDF_ACC_SCOPE(); \
    MEM_TIME_BLOCK(TAG); \
    MARK_SCOPE_MEM(TAG); \
    PROFILE_BLOCK(TAG)



namespace solver {

    namespace blockstats {

        bool g_run_block_stats = false;      
        std::string g_output_path;           

        void compute_block_sizes_and_write()
        {
            auto &C = ctx();
            ogdf::Graph &G = C.G;

            if (G.numberOfNodes() == 0) {
                std::cerr << "[blockstats] graph is empty, nothing to do\n";
                return;
            }
            // 1) connected components of the overall graph
            ogdf::NodeArray<int> compIdx(G);
            int nCC = ogdf::connectedComponents(G, compIdx);

            std::vector<std::vector<ogdf::node>> bucket(nCC);
            std::vector<std::vector<ogdf::edge>> edgeBuckets(nCC);

            for (ogdf::node v : G.nodes) {
                bucket[compIdx[v]].push_back(v);
            }
            for (ogdf::edge e : G.edges) {
                edgeBuckets[compIdx[e->source()]].push_back(e);
            }

            std::vector<size_t> blockSizes;
            blockSizes.reserve(G.numberOfNodes());  


            // 2) For each component, rebuild Gcc + BCTree and collect the block sizes

            for (int cid = 0; cid < nCC; ++cid) {

                ogdf::Graph Gcc;
                ogdf::NodeArray<ogdf::node> nodeToOrig(Gcc, nullptr);
                ogdf::EdgeArray<ogdf::edge> edgeToOrig(Gcc, nullptr);

                std::unordered_map<ogdf::node, ogdf::node> orig_to_cc;
                orig_to_cc.reserve(bucket[cid].size());

                for (ogdf::node vG : bucket[cid]) {
                    ogdf::node vC = Gcc.newNode();
                    nodeToOrig[vC] = vG;
                    orig_to_cc[vG] = vC;
                }

                for (ogdf::edge eG : edgeBuckets[cid]) {
                    ogdf::node uC = orig_to_cc[eG->source()];
                    ogdf::node vC = orig_to_cc[eG->target()];
                    ogdf::edge eC = Gcc.newEdge(uC, vC);
                    edgeToOrig[eC] = eG;
                }

                ogdf::BCTree bc(Gcc);

                for (ogdf::node bNode : bc.bcTree().nodes) {
                    if (bc.typeOfBNode(bNode) != ogdf::BCTree::BNodeType::BComp)
                        continue;

                    std::vector<ogdf::edge> edgesCc;
                    edgesCc.reserve(bc.hEdges(bNode).size());
                    for (ogdf::edge hE : bc.hEdges(bNode)) {
                        ogdf::edge eCc = bc.original(hE);
                        if (eCc) edgesCc.push_back(eCc);
                    }

                    std::sort(edgesCc.begin(), edgesCc.end(),
                            [](ogdf::edge a, ogdf::edge b) { return a->index() < b->index(); });
                    edgesCc.erase(std::unique(edgesCc.begin(), edgesCc.end(),
                                            [](ogdf::edge a, ogdf::edge b) { return a->index() == b->index(); }),
                                edgesCc.end());

                    std::vector<ogdf::node> vertsCc;
                    vertsCc.reserve(edgesCc.size() * 2);
                    for (ogdf::edge eCc : edgesCc) {
                        vertsCc.push_back(eCc->source());
                        vertsCc.push_back(eCc->target());
                    }
                    std::sort(vertsCc.begin(), vertsCc.end(),
                            [](ogdf::node a, ogdf::node b) { return a->index() < b->index(); });
                    vertsCc.erase(std::unique(vertsCc.begin(), vertsCc.end()), vertsCc.end());

                    size_t blockSize = vertsCc.size() + edgesCc.size();      

                    blockSizes.push_back(blockSize);
                }
            }

            std::ofstream ofs(g_output_path);
            if (!ofs) {
                std::cerr << "[blockstats] cannot open '" << g_output_path
                        << "' for writing block sizes\n";
                return;
            }

            for (size_t s : blockSizes) {
                ofs << s << '\n';
            }

            ofs.close();

            std::cerr << "[blockstats] wrote " << blockSizes.size()
                    << " block sizes to " << g_output_path << "\n";
        }

    } 

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
            ogdf::StaticSPQRTree &spqr,
            std::vector<ogdf::edge> &edge_order, 
            std::vector<ogdf::node> &node_order,
            ogdf::node curr = nullptr,
            ogdf::node parent = nullptr,
            ogdf::edge e = nullptr
        ) {
            PROFILE_FUNCTION();
            if (curr == nullptr) {
                curr = spqr.rootNode();
                parent = curr;
                dfsSPQR_order(spqr, edge_order, node_order, curr, parent);
                return;
            }

            node_order.push_back(curr);

            std::vector<std::pair<ogdf::node, ogdf::edge>> children;
            children.reserve(curr->degree());
            for (ogdf::adjEntry adj = curr->firstAdj(); adj; adj = adj->succ()) {
                ogdf::node child = adj->twinNode();
                if (child == parent) continue;
                children.emplace_back(child, adj->theEdge());
            }
            std::sort(children.begin(), children.end(),
                    [](const auto &A, const auto &B){
                        if (A.first->index() != B.first->index())
                            return A.first->index() < B.first->index();
                        return A.second->index() < B.second->index();
                    });

            for (auto &ch : children) {
                dfsSPQR_order(spqr, edge_order, node_order, ch.first, curr, ch.second);
            }

            if (curr != parent) edge_order.push_back(e);
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
                        cc->bc = OGDF_NEW_UNIQUE("ogdf/BCTree::ctor", BCTree, *cc->Gcc);
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

        void solve() {
            auto& C = ctx();
            Graph& G = C.G;

            int nCC = 0;
            NodeArray<int> compIdx(G);
            std::vector<std::vector<node>> bucket;
            std::vector<std::vector<edge>> edgeBuckets;
            std::vector<std::unique_ptr<CcData>> components;
            std::vector<BlockPrep> blockPreps;
            std::vector<std::unique_ptr<BlockData>> allBlockData;
            std::vector<WorkItem> workItems;

            // ===================== PHASE I/O =====================
            {
                METRICS_PHASE_BEGIN(metrics::Phase::IO);
                MEM_TIME_BLOCK("I/O: superbubbles/cc+buckets");
                MARK_SCOPE_MEM("sb/io/cc_buckets");

                {
                    MARK_SCOPE_MEM("sb/phase/ComputeCC");
                    nCC = OGDF_EVAL("ogdf/connectedComponents", connectedComponents(G, compIdx));
                }

                components.resize(nCC);
                bucket.assign(nCC, {});
                edgeBuckets.assign(nCC, {});

                {
                    MARK_SCOPE_MEM("sb/phase/BucketNodes");
                    for (node v : G.nodes) {
                        bucket[compIdx[v]].push_back(v);
                    }
                }

                {
                    MARK_SCOPE_MEM("sb/phase/BucketEdges");
                    for (edge e : G.edges) {
                        edgeBuckets[compIdx[e->source()]].push_back(e);
                    }
                }

                PHASE_RSS_UPDATE_IO();
                METRICS_PHASE_END(metrics::Phase::IO);
            }


            // ===================== PHASE BUILD =====================
            {
                METRICS_PHASE_BEGIN(metrics::Phase::BUILD);
                MEM_TIME_BLOCK("BUILD: superbubbles/BC+SPQR");
                MARK_SCOPE_MEM("sb/build/all");
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
                            size_t chunkSize = std::max<size_t>(1, nCC / std::max<size_t>(numThreads, 1));
                            size_t processed = 0;
                            while (true) {
                                size_t startIndex, endIndex;
                                {
                                    std::lock_guard<std::mutex> lock(workMutex);
                                    if (nextIndex >= static_cast<size_t>(nCC)) break;
                                    startIndex = nextIndex;
                                    endIndex   = std::min(nextIndex + chunkSize, static_cast<size_t>(nCC));
                                    nextIndex  = endIndex;
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

                // BlockData + SPQR
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

                PHASE_RSS_UPDATE_BUILD();
                METRICS_PHASE_END(metrics::Phase::BUILD);
            }


            // ===================== PHASE LOGIC =====================
            {
                METRICS_PHASE_BEGIN(metrics::Phase::LOGIC);
                MEM_TIME_BLOCK("LOGIC: superbubbles/all");
                MARK_SCOPE_MEM("sb/logic/all");
                ACCUM_LOGIC();

                findMiniSuperbubbles();

                {
                    MEM_TIME_BLOCK("LOGIC: superbubbles/solve blocks (pthreads)");
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

                PHASE_RSS_UPDATE_LOGIC();
                METRICS_PHASE_END(metrics::Phase::LOGIC);
            }
        }
  
    }

    namespace snarls {
        static inline uint64_t nowMicros() {
            using namespace std::chrono;
            return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
        }

        static size_t currentRSSBytes() {
        #if defined(__linux__)
            long rssPages = 0;
            FILE* f = std::fopen("/proc/self/statm", "r");
            if (f) {
                if (std::fscanf(f, "%*s%ld", &rssPages) != 1) {
                    rssPages = 0;
                }
                std::fclose(f);
            }
            long pageSize = sysconf(_SC_PAGESIZE);
            if (pageSize <= 0) pageSize = 4096;
            if (rssPages < 0) rssPages = 0;
            return static_cast<size_t>(rssPages) * static_cast<size_t>(pageSize);
        #elif defined(__APPLE__)
            mach_task_basic_info info;
            mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
            if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
                return 0;
            }
            return static_cast<size_t>(info.resident_size);
        #elif defined(_WIN32)
            PROCESS_MEMORY_COUNTERS pmc;
            if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
                return static_cast<size_t>(pmc.WorkingSetSize);
            }
            return 0;
        #else
            return 0;
        #endif
        }

        struct PhaseStats {
            std::atomic<uint64_t> elapsed_us{0};
            std::atomic<size_t>   peak_rss{0};    
            std::atomic<size_t>   start_rss{0};   
        };


        static PhaseStats g_stats_io;
        static PhaseStats g_stats_build;
        static PhaseStats g_stats_logic;

        class PhaseSampler {
        public:
            explicit PhaseSampler(PhaseStats& stats, uint32_t period_us = 1000)
                : stats_(stats), period_us_(period_us), stop_(false) {
                stats_.start_rss.store(currentRSSBytes(), std::memory_order_relaxed);
                start_us_ = nowMicros();
                sampler_ = std::thread([this]() { this->run(); });
            }
            ~PhaseSampler() {
                stop_ = true;
                if (sampler_.joinable()) sampler_.join();
                uint64_t dur = nowMicros() - start_us_;
                stats_.elapsed_us.store(dur, std::memory_order_relaxed);
            }
        private:
            void run() {
                size_t local_peak = 0;
                while (!stop_) {
                    size_t rss = currentRSSBytes();
                    if (rss > local_peak) local_peak = rss;
                    std::this_thread::sleep_for(std::chrono::microseconds(period_us_));
                }

                size_t rss = currentRSSBytes();
                if (rss > local_peak) local_peak = rss;

                size_t prev = stats_.peak_rss.load(std::memory_order_relaxed);
                while (local_peak > prev &&
                    !stats_.peak_rss.compare_exchange_weak(prev, local_peak, std::memory_order_relaxed)) {

                }
            }

            PhaseStats& stats_;
            uint32_t period_us_;
            std::atomic<bool> stop_;
            std::thread sampler_;
            uint64_t start_us_{0};
        };

        namespace {

            thread_local std::vector<std::vector<std::string>>* tls_snarl_buffer = nullptr;

            static std::mutex g_snarls_mtx;

            static inline bool write_all_fd(int fd, const void* buf, size_t n) {
                const uint8_t* p = static_cast<const uint8_t*>(buf);
                size_t done = 0;
                while (done < n) {
                    ssize_t w = ::write(fd, p + done, n - done);
                    if (w < 0) {
                        if (errno == EINTR) continue;
                        return false;
                    }
                    if (w == 0) return false;
                    done += static_cast<size_t>(w);
                }
                return true;
            }

            static inline bool read_all_fd(int fd, void* buf, size_t n) {
                uint8_t* p = static_cast<uint8_t*>(buf);
                size_t done = 0;
                while (done < n) {
                    ssize_t r = ::read(fd, p + done, n - done);
                    if (r < 0) {
                        if (errno == EINTR) continue;
                        return false;
                    }
                    if (r == 0) return false;
                    done += static_cast<size_t>(r);
                }
                return true;
            }

            static bool g_is_child_process = false;
            static int g_snarl_out_fd = -1;
            static std::string g_snarl_out_path;
            static std::string g_snarl_counts_path;  

            inline void flushThreadLocalSnarls(std::vector<std::vector<std::string>>& local) {
                if (local.empty()) return;

                if (!g_is_child_process) {
                    auto &C = ctx();
                    std::lock_guard<std::mutex> lk(g_snarls_mtx);
                    for (auto &s : local) {
                        std::sort(s.begin(), s.end());
                        snarlsFound += s.size() * (s.size() - 1) / 2;
                        C.snarls.insert(s);
                    }
                    local.clear();
                    return;
                }

                if (g_snarl_out_fd < 0) {
                    std::string base = "/dev/shm";
                    int test = ::access(base.c_str(), W_OK);
                    if (test != 0) base = "/tmp";
                    g_snarl_out_path = base + "/snarls_" + std::to_string(::getpid()) + ".bin";
                    g_snarl_out_fd = ::open(g_snarl_out_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
                    if (g_snarl_out_fd < 0) {
                        std::fprintf(stderr, "[child %d] open(%s) failed: %s\n",
                                    (int)::getpid(), g_snarl_out_path.c_str(), std::strerror(errno));
                        _exit(111);
                    }
                }

                for (auto &sn : local) {
                    std::sort(sn.begin(), sn.end());

                    uint32_t k = static_cast<uint32_t>(sn.size());
                    if (!write_all_fd(g_snarl_out_fd, &k, sizeof(k))) {
                        std::fprintf(stderr, "[child %d] write(k) failed: %s\n", (int)::getpid(), std::strerror(errno));
                        _exit(112);
                    }
                    for (const auto &s : sn) {
                        uint32_t len = static_cast<uint32_t>(s.size());
                        if (!write_all_fd(g_snarl_out_fd, &len, sizeof(len))) {
                            std::fprintf(stderr, "[child %d] write(len) failed: %s\n", (int)::getpid(), std::strerror(errno));
                            _exit(113);
                        }
                        if (len > 0) {
                            if (!write_all_fd(g_snarl_out_fd, s.data(), len)) {
                                std::fprintf(stderr, "[child %d] write(data) failed: %s\n", (int)::getpid(), std::strerror(errno));
                                _exit(114);
                            }
                        }
                    }
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

        static inline void normalize_snarl(std::vector<std::string>& s) {
            std::sort(s.begin(), s.end());
        }
        static void tryCommitSnarl(std::vector<std::string> s) {
            std::sort(s.begin(), s.end()); 

            if (g_is_child_process) {
                if (tls_snarl_buffer) {
                    tls_snarl_buffer->push_back(std::move(s));
                    return;
                }

                if (g_snarl_out_fd < 0) {
                    std::string base = "/dev/shm";
                    int test = ::access(base.c_str(), W_OK);
                    if (test != 0) base = "/tmp";
                    g_snarl_out_path = base + "/snarls_" + std::to_string(::getpid()) + ".bin";
                    g_snarl_out_fd = ::open(g_snarl_out_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
                    if (g_snarl_out_fd < 0) {
                        std::fprintf(stderr, "[child %d] open(%s) failed: %s\n",
                                    (int)::getpid(), g_snarl_out_path.c_str(), std::strerror(errno));
                        _exit(111);
                    }
                }

                uint32_t k = static_cast<uint32_t>(s.size());
                if (!write_all_fd(g_snarl_out_fd, &k, sizeof(k))) {
                    std::fprintf(stderr, "[child %d] write(k) failed: %s\n", (int)::getpid(), std::strerror(errno));
                    _exit(112);
                }
                for (auto &str : s) {
                    uint32_t len = static_cast<uint32_t>(str.size());
                    if (!write_all_fd(g_snarl_out_fd, &len, sizeof(len))) {
                        std::fprintf(stderr, "[child %d] write(len) failed: %s\n", (int)::getpid(), std::strerror(errno));
                        _exit(113);
                    }
                    if (len > 0) {
                        if (!write_all_fd(g_snarl_out_fd, str.data(), len)) {
                            std::fprintf(stderr, "[child %d] write(data) failed: %s\n", (int)::getpid(), std::strerror(errno));
                            _exit(114);
                        }
                    }
                }
                return;
            }

            auto &C = ctx();
            std::lock_guard<std::mutex> lk(g_snarls_mtx);
            snarlsFound += s.size() * (s.size() - 1) / 2;
            C.snarls.insert(std::move(s));
        }


        std::atomic<uint64_t> g_cnt_cut{0}, g_cnt_S{0}, g_cnt_P{0}, g_cnt_RR{0}, g_cnt_E{0};

        struct SnarlTypeCounts {
            uint64_t cut;
            uint64_t S;
            uint64_t P;
            uint64_t RR;
            uint64_t E;

        };  
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


        inline void addSnarlTagged(const char *tag, std::vector<std::string> s) {
            if (tag) {
                if      (std::strcmp(tag, "CUT") == 0) g_cnt_cut++;
                else if (std::strcmp(tag, "S")   == 0) g_cnt_S++;
                else if (std::strcmp(tag, "P")   == 0) g_cnt_P++;
                else if (std::strcmp(tag, "RR")  == 0) g_cnt_RR++;
                else if (std::strcmp(tag, "E")   == 0) g_cnt_E++;
            }

            VLOG << "[SNARL][" << (tag ? tag : "?") << "] ";
            for (auto &x : s) VLOG << x << " ";
            VLOG << "\n";

            addSnarl(std::move(s));
        }
        


        inline void print_snarl_type_counters() {
            if (g_is_child_process) return; 
            std::cout << "[SNARLS] by type: "
                    << "CUT=" << g_cnt_cut.load()
                    << " S="   << g_cnt_S.load()
                    << " P="   << g_cnt_P.load()
                    << " RR="  << g_cnt_RR.load()
                    << " E="   << g_cnt_E.load()
                    << std::endl;
        }


        struct BlockData {
            std::unique_ptr<ogdf::Graph> Gblk;  
            ogdf::NodeArray<ogdf::node> toCc;
            ogdf::NodeArray<ogdf::node> nodeToOrig;
            ogdf::EdgeArray<ogdf::edge> edgeToOrig;

            std::unique_ptr<ogdf::StaticSPQRTree> spqr;
            ogdf::NodeArray<ogdf::node> blkToSkel;

            std::unordered_map<ogdf::edge, ogdf::edge> skel2tree; 
            ogdf::NodeArray<ogdf::node> parent; 

            ogdf::node bNode {nullptr};

            bool isAcycic {true};

            // NOUVEAU : degrés + / - dans ce bloc H
            ogdf::NodeArray<int> degPlusBlk;
            ogdf::NodeArray<int> degMinusBlk;
            
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


        static bool g_bf_adj_built = false;

        struct BfAdjEntry {
            int nb;               
            EdgePartType here;    
            EdgePartType nbType;  
        };


        static bool bfs_split_block(
            const std::vector<std::vector<BfAdjEntry>>& adj,
            int iu, int iv,
            EdgePartType d_u, EdgePartType d_v,
            std::vector<int>* comp 
        ) {
            const int N = (int)adj.size();
            const int X  = iu;       
            const int Y  = iv;       
            const int Xp = N;        
            const int Yp = N + 1;    

            std::vector<char> visited(N + 2, 0);
            std::queue<int> q;

            visited[X] = 1;
            q.push(X);

            if (comp) {
                comp->clear();
                comp->push_back(X);
            }

            auto push_if_new = [&](int w) {
                if (!visited[w]) {
                    visited[w] = 1;
                    q.push(w);
                    if (comp && w != Xp && w != Yp) {
                        comp->push_back(w);
                    }
                }
            };

            while (!q.empty()) {
                int u = q.front();
                q.pop();

                if (u == X) {
                    for (const auto &inc : adj[X]) {
                        int nb = inc.nb;
                        EdgePartType sHere = inc.here;
                        EdgePartType sNb   = inc.nbType;

                        if (sHere != d_u) continue; 

                        int vPhys;
                        if (nb == Y) {
                            vPhys = (sNb == d_v ? Y : Yp);
                        } else {
                            vPhys = nb;
                        }
                        push_if_new(vPhys);
                    }

                } else if (u == Xp) {
                    for (const auto &inc : adj[X]) {
                        int nb = inc.nb;
                        EdgePartType sHere = inc.here;
                        EdgePartType sNb   = inc.nbType;

                        if (sHere == d_u) continue;

                        int vPhys;
                        if (nb == Y) {
                            vPhys = (sNb == d_v ? Y : Yp);
                        } else {
                            vPhys = nb;
                        }
                        push_if_new(vPhys);
                    }

                } else if (u == Y) {
                    for (const auto &inc : adj[Y]) {
                        int nb = inc.nb;
                        EdgePartType sHere = inc.here;
                        EdgePartType sNb   = inc.nbType;

                        if (sHere != d_v) continue;

                        int vPhys;
                        if (nb == X) {
                            vPhys = (sNb == d_u ? X : Xp);
                        } else {
                            vPhys = nb;
                        }
                        push_if_new(vPhys);
                    }

                } else if (u == Yp) {
                    for (const auto &inc : adj[Y]) {
                        int nb = inc.nb;
                        EdgePartType sHere = inc.here;
                        EdgePartType sNb   = inc.nbType;

                        if (sHere == d_v) continue;

                        int vPhys;
                        if (nb == X) {
                            vPhys = (sNb == d_u ? X : Xp);
                        } else {
                            vPhys = nb;
                        }
                        push_if_new(vPhys);
                    }

                } else {
                    for (const auto &inc : adj[u]) {
                        int nb = inc.nb;
                        EdgePartType sHere = inc.here; 
                        EdgePartType sNb   = inc.nbType;

                        int vPhys;
                        if (nb == X) {
                            vPhys = (sNb == d_u ? X : Xp);
                        } else if (nb == Y) {
                            vPhys = (sNb == d_v ? Y : Yp);
                        } else {
                            vPhys = nb;
                        }
                        push_if_new(vPhys);
                    }
                }
            }

            return (visited[Y] && !visited[Xp] && !visited[Yp]);
        }


        static bool is_edge_snarl_block(
            const std::vector<std::vector<BfAdjEntry>>& adj,
            int iu, int iv,
            EdgePartType d_u, EdgePartType d_v
        ) {
            std::vector<int> comp;
            if (!bfs_split_block(adj, iu, iv, d_u, d_v, &comp)) {
                return false; // not separable
            }

            const int N = (int)adj.size();

            for (int z : comp) {
                if (z == iu || z == iv) continue;

                bool hasPlusZ  = false;
                bool hasMinusZ = false;

                for (const auto &inc : adj[z]) {
                    if (inc.here == EdgePartType::PLUS)  hasPlusZ = true;
                    if (inc.here == EdgePartType::MINUS) hasMinusZ = true;
                    if (hasPlusZ && hasMinusZ) break;
                }

                if (hasPlusZ) {
                    if (bfs_split_block(adj, iu, z, d_u, EdgePartType::PLUS, nullptr) &&
                        bfs_split_block(adj, z,  iv, EdgePartType::MINUS, d_v, nullptr)) {
                        return false; // not minimal
                    }
                }
                if (hasMinusZ) {
                    if (bfs_split_block(adj, iu, z, d_u, EdgePartType::MINUS, nullptr) &&
                        bfs_split_block(adj, z,  iv, EdgePartType::PLUS,  d_v, nullptr)) {
                        return false;
                    }
                }
            }

            return true;
        }

        void getOutgoingEdgesInBlock(const CcData& cc, ogdf::node uG, ogdf::node vB, EdgePartType type, std::vector<ogdf::edge>& outEdges) {
            outEdges.clear();
            ogdf::node uB = cc.bc->repVertex(uG, vB);
            
            for(auto adjE : uB->adjEntries) {
                ogdf::edge eAux = adjE->theEdge();               
                ogdf::edge eCc = cc.bc->original(eAux);          
                ogdf::edge eG = cc.edgeToOrig[eCc];            

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
                ogdf::edge eC = adjE->theEdge();
                ogdf::edge eOrig = cc.edgeToOrig[eC]; 

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
                EdgeDPState down;   
                EdgeDPState up;     
            };

            struct NodeDPState {
                std::vector<ogdf::node> GccCuts_last3; 
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
                std::vector<ogdf::edge> &edge_order, 
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


            void processEdge(ogdf::edge curr_edge,
                            ogdf::EdgeArray<EdgeDP> &dp,
                            const CcData &cc,
                            BlockData &blk)
            {
                //PROFILE_FUNCTION();
                auto &C = ctx();

                const StaticSPQRTree &spqr = *blk.spqr;

                ogdf::node A = curr_edge->source();
                ogdf::node B = curr_edge->target();

                EdgeDPState &state      = dp[curr_edge].down;
                EdgeDPState &back_state = dp[curr_edge].up;

                state.s = state.t = nullptr;
                state.localPlusS = state.localPlusT = 0;
                state.localMinusS = state.localMinusT = 0;

                back_state.s = back_state.t = nullptr;
                back_state.localPlusS = back_state.localPlusT = 0;
                back_state.localMinusS = back_state.localMinusT = 0;

                const Skeleton &skel = spqr.skeleton(B);
                const Graph &skelGraph = skel.getGraph();

                for (edge e : skelGraph.edges) {
                    node u = e->source();
                    node v = e->target();

                    auto D = skel.twinTreeNode(e);

                    if (D == A) {
                        ogdf::node vBlk = skel.original(v);
                        ogdf::node uBlk = skel.original(u);

                        state.s = back_state.s = vBlk;
                        state.t = back_state.t = uBlk;
                        break;
                    }
                }

                if (!state.s || !state.t) {
                    VLOG << "[snarls] processEdge: impossible de trouver les pôles pour "
                            << "l'arête SPQR (" << A->index() << " -> " << B->index() << ")\n";
                    return;
                }

                for (edge e : skelGraph.edges) {
                    node u = e->source();
                    node v = e->target();

                    ogdf::node uBlk = skel.original(u);
                    ogdf::node vBlk = skel.original(v);

                    if (!skel.isVirtual(e)) {
                        ogdf::edge eG = blk.edgeToOrig[skel.realEdge(e)];

                        ogdf::node uG = eG->source();
                        ogdf::node vG = eG->target();

                        // Contributions incidentes à state.s
                        if (uG == blk.nodeToOrig[state.s]) {
                            auto t = getNodeEdgeType(uG, eG);
                            if      (t == EdgePartType::PLUS)  state.localPlusS++;
                            else if (t == EdgePartType::MINUS) state.localMinusS++;
                        }
                        if (vG == blk.nodeToOrig[state.s]) {
                            auto t = getNodeEdgeType(vG, eG);
                            if      (t == EdgePartType::PLUS)  state.localPlusS++;
                            else if (t == EdgePartType::MINUS) state.localMinusS++;
                        }

                        if (uG == blk.nodeToOrig[state.t]) {
                            auto t = getNodeEdgeType(uG, eG);
                            if      (t == EdgePartType::PLUS)  state.localPlusT++;
                            else if (t == EdgePartType::MINUS) state.localMinusT++;
                        }
                        if (vG == blk.nodeToOrig[state.t]) {
                            auto t = getNodeEdgeType(vG, eG);
                            if      (t == EdgePartType::PLUS)  state.localPlusT++;
                            else if (t == EdgePartType::MINUS) state.localMinusT++;
                        }

                        continue;
                    }

                    auto D = skel.twinTreeNode(e);
                    if (D == A) {
                        continue;
                    }

                    edge treeE = blk.skel2tree.at(e);
                    OGDF_ASSERT(treeE != nullptr);

                    const EdgeDPState child = dp[treeE].down;

                    if (!child.s || !child.t) {
                        VLOG << "[snarls] processEdge: état DP enfant incomplet sur treeE="
                                << treeE->index() << "\n";
                        continue;
                    }

                    if (state.s == child.s) {
                        state.localPlusS  += child.localPlusS;
                        state.localMinusS += child.localMinusS;
                    }
                    if (state.s == child.t) {
                        state.localPlusS  += child.localPlusT;
                        state.localMinusS += child.localMinusT;
                    }
                    if (state.t == child.t) {
                        state.localPlusT  += child.localPlusT;
                        state.localMinusT += child.localMinusT;
                    }
                    if (state.t == child.s) {
                        state.localPlusT  += child.localPlusS;
                        state.localMinusT += child.localMinusS;
                    }
                }
            }

            void processNode(ogdf::node curr_node,
                            ogdf::EdgeArray<EdgeDP> &edge_dp,
                            const CcData &cc,
                            BlockData &blk)
            {
                auto& C = ctx();
                ogdf::node A = curr_node;

                const ogdf::Graph &T    = blk.spqr->tree();
                const ogdf::StaticSPQRTree &spqr = *blk.spqr;

                const ogdf::Skeleton &skel      = spqr.skeleton(A);
                const ogdf::Graph    &skelGraph = skel.getGraph();

                ogdf::Graph newGraph;

                ogdf::NodeArray<ogdf::node> skelToNew(skelGraph, nullptr);
                for (ogdf::node v : skelGraph.nodes) {
                    skelToNew[v] = newGraph.newNode();
                }
                ogdf::NodeArray<ogdf::node> newToSkel(newGraph, nullptr);
                for (ogdf::node v : skelGraph.nodes) {
                    newToSkel[skelToNew[v]] = v;
                }

                for (ogdf::node h : skelGraph.nodes) {
                    ogdf::node vB = skel.original(h);
                    if (!vB) continue;
                    blk.blkToSkel[vB] = h;
                }

                ogdf::NodeArray<int> localPlusDeg(newGraph,  0);
                ogdf::NodeArray<int> localMinusDeg(newGraph, 0);

                ogdf::EdgeArray<bool>         isVirtual(newGraph, false);
                ogdf::EdgeArray<EdgeDPState*> edgeToDp(newGraph, nullptr);
                ogdf::EdgeArray<EdgeDPState*> edgeToDpR(newGraph, nullptr);
                ogdf::EdgeArray<ogdf::node>   edgeChild(newGraph, nullptr);

                std::vector<ogdf::edge> virtualEdges;

                auto mapBlockToNew = [&](ogdf::node bV) -> ogdf::node {
                    if (!bV) return nullptr;
                    ogdf::node sV = blk.blkToSkel[bV];
                    if (!sV) return nullptr;
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

                auto getTreeEdge = [&](ogdf::edge eSkel) -> ogdf::edge {
                    auto it = blk.skel2tree.find(eSkel);
                    if (it == blk.skel2tree.end() || !(it->second)) {
                        return nullptr;
                    }
                    return it->second;
                };

                for (ogdf::edge e : skelGraph.edges) {
                    ogdf::node u = e->source();
                    ogdf::node v = e->target();

                    ogdf::node uBlk = skel.original(u);
                    ogdf::node vBlk = skel.original(v);
                    if (!uBlk || !vBlk) continue;

                    ogdf::node uG = blk.nodeToOrig[uBlk];
                    ogdf::node vG = blk.nodeToOrig[vBlk];

                    ogdf::node nU = skelToNew[u];
                    ogdf::node nV = skelToNew[v];

                    if (!skel.isVirtual(e)) {
                        ogdf::edge eG = blk.edgeToOrig[skel.realEdge(e)];
                        if (!eG) continue;

                        uG = eG->source();
                        vG = eG->target();

                        ogdf::edge newEdge = newGraph.newEdge(nU, nV);
                        isVirtual[newEdge] = false;

                        if (blk.nodeToOrig[skel.original(newToSkel[nU])] == uG) {
                            localPlusDeg[nU]  += (getNodeEdgeType(uG, eG) == EdgePartType::PLUS);
                            localMinusDeg[nU] += (getNodeEdgeType(uG, eG) == EdgePartType::MINUS);

                            localPlusDeg[nV]  += (getNodeEdgeType(vG, eG) == EdgePartType::PLUS);
                            localMinusDeg[nV] += (getNodeEdgeType(vG, eG) == EdgePartType::MINUS);
                        } else {
                            localPlusDeg[nU]  += (getNodeEdgeType(vG, eG) == EdgePartType::PLUS);
                            localMinusDeg[nU] += (getNodeEdgeType(vG, eG) == EdgePartType::MINUS);

                            localPlusDeg[nV]  += (getNodeEdgeType(uG, eG) == EdgePartType::PLUS);
                            localMinusDeg[nV] += (getNodeEdgeType(uG, eG) == EdgePartType::MINUS);
                        }

                        continue;
                    }

                    ogdf::node B = skel.twinTreeNode(e);
                    ogdf::edge treeE = getTreeEdge(e);
                    if (!treeE) continue;

                    SPQRsolve::EdgeDP &st = edge_dp[treeE];

                    EdgeDPState *child =
                        (B == blk.parent(A) ? &st.up : &st.down);
                    EdgeDPState *edgeToUpdate =
                        (B == blk.parent(A) ? &st.down : &st.up);

                    if (!child || !child->s || !child->t) {
                        continue;
                    }

                    ogdf::node nS = mapBlockToNew(child->s);
                    ogdf::node nT = mapBlockToNew(child->t);
                    if (!nS || !nT) continue;

                    ogdf::edge newEdge = newGraph.newEdge(nS, nT);

                    isVirtual[newEdge] = true;
                    virtualEdges.push_back(newEdge);

                    edgeToDp[newEdge]  = edgeToUpdate;
                    edgeToDpR[newEdge] = child;
                    edgeChild[newEdge] = B;

                    if (nS == nU && nT == nV) {
                        localMinusDeg[nS] += child->localMinusT;
                        localPlusDeg[nS]  += child->localPlusT;

                        localMinusDeg[nT] += child->localMinusS;
                        localPlusDeg[nT]  += child->localPlusS;
                    } else {
                        localMinusDeg[nS] += child->localMinusS;
                        localPlusDeg[nS]  += child->localPlusS;

                        localMinusDeg[nT] += child->localMinusT;
                        localPlusDeg[nT]  += child->localPlusT;
                    }
                }

                for (ogdf::edge e : virtualEdges) {
                    EdgeDPState *BA = edgeToDp[e];
                    EdgeDPState *AB = edgeToDpR[e];

                    if (!BA || !AB || !BA->s || !BA->t || !AB->s || !AB->t) {
                        continue;
                    }

                    ogdf::node sNew = mapBlockToNew(BA->s);
                    ogdf::node tNew = mapBlockToNew(BA->t);
                    if (!sNew || !tNew) continue;

                    BA->localPlusS  = localPlusDeg[sNew]  - AB->localPlusS;
                    BA->localPlusT  = localPlusDeg[tNew]  - AB->localPlusT;
                    BA->localMinusS = localMinusDeg[sNew] - AB->localMinusS;
                    BA->localMinusT = localMinusDeg[tNew] - AB->localMinusT;
                }
            }
            
            void solveS(ogdf::node sNode,
                        ogdf::NodeArray<SPQRsolve::NodeDPState> &node_dp,
                        ogdf::EdgeArray<SPQRsolve::EdgeDP> &dp,
                        BlockData& blk,
                        const CcData& cc)
            {
                PROFILE_FUNCTION();
                auto &C = ctx();

                const Skeleton& skel      = blk.spqr->skeleton(sNode);
                const Graph&    skelGraph = skel.getGraph();
                const Graph&    T         = blk.spqr->tree();
                VLOG << "[SPQR-S] ==== solveS on tree node " << sNode->index() << " ====\n";

                if (skelGraph.numberOfNodes() == 0) {
                    VLOG << "[SPQR-S]  skeleton empty, abort\n";
                    return;
                }

                ogdf::EdgeArray<EdgeDPState*> skelToState(T, nullptr);

                for (edge e : skelGraph.edges) {
                    if (!skel.isVirtual(e)) continue;
                    ogdf::node B  = skel.twinTreeNode(e);
                    auto itTree   = blk.skel2tree.find(e);
                    if (itTree == blk.skel2tree.end() || !(itTree->second)) {
                        VLOG << "[SPQR-S]  virtual edge " << e->index()
                                << " has no tree edge\n";
                        continue;
                    }
                    edge treeE   = itTree->second;
                    EdgeDPState *child =
                        (B == blk.parent[sNode] ? &dp[treeE].up : &dp[treeE].down);
                    skelToState[treeE] = child;
                }

                std::vector<ogdf::node> nodesInOrderGcc;  
                std::vector<ogdf::node> nodesInOrderSkel;  

                std::vector<ogdf::edge>     adjEdgesG_;
                std::vector<ogdf::adjEntry> adjEntriesSkel;

                {
                    ogdf::node rootSkel = skelGraph.firstNode();
                    if (!rootSkel || !rootSkel->firstAdj()) {
                        VLOG << "[SPQR-S]  skeleton has no edges, abort\n";
                        return;
                    }

                    std::function<void(ogdf::node, ogdf::node)> dfs =
                        [&](ogdf::node u, ogdf::node prev)
                    {
                        nodesInOrderGcc.push_back(blk.toCc[skel.original(u)]);
                        nodesInOrderSkel.push_back(u);

                        for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                            ogdf::node v = adj->twinNode();
                            if (v == prev) continue;

                            if (v == skelGraph.firstNode() && u != skelGraph.firstNode()) {
                                if (skel.realEdge(adj->theEdge()))
                                    adjEdgesG_.push_back(
                                        blk.edgeToOrig[skel.realEdge(adj->theEdge())]);
                                else
                                    adjEdgesG_.push_back(nullptr);
                                adjEntriesSkel.push_back(adj);
                            }

                            if (v == skelGraph.firstNode() || v == prev) continue;

                            if (skel.realEdge(adj->theEdge()))
                                adjEdgesG_.push_back(
                                    blk.edgeToOrig[skel.realEdge(adj->theEdge())]);
                            else
                                adjEdgesG_.push_back(nullptr);

                            adjEntriesSkel.push_back(adj);
                            dfs(v, u);
                        }
                    };

                    dfs(rootSkel, rootSkel->firstAdj()->twinNode());
                }

                if (nodesInOrderGcc.empty() ||
                    adjEntriesSkel.size() != nodesInOrderGcc.size() ||
                    adjEdgesG_.size()     != nodesInOrderGcc.size())
                {
                    VLOG << "[SPQR-S]  inconsistent sizes: nodes=" << nodesInOrderGcc.size()
                            << " adjSkel=" << adjEntriesSkel.size()
                            << " adjG="    << adjEdgesG_.size() << "\n";
                    return;
                }

                VLOG << "[SPQR-S]  cycle order (Gcc/orig names):";
                for (size_t i = 0; i < nodesInOrderGcc.size(); ++i) {
                    ogdf::node vGcc = nodesInOrderGcc[i];
                    ogdf::node vG   = cc.nodeToOrig[vGcc];
                    VLOG << " " << C.node2name[vG];
                }
                VLOG << "\n";

                std::vector<std::string> res;

                for (size_t i = 0; i < nodesInOrderGcc.size(); i++) {
                    ogdf::node uGcc  = nodesInOrderGcc[i];
                    ogdf::node uSkel = nodesInOrderSkel[i];
                    ogdf::node uG    = cc.nodeToOrig[uGcc];
                    std::string uname = C.node2name[uG];

                    ogdf::edge eSkelL =
                        adjEntriesSkel[(i + adjEntriesSkel.size() - 1) %
                                    adjEntriesSkel.size()]->theEdge();
                    ogdf::edge eSkelR = adjEntriesSkel[i]->theEdge();

                    ogdf::edge eGL =
                        adjEdgesG_[(i + adjEdgesG_.size() - 1) % adjEdgesG_.size()];
                    ogdf::edge eGR = adjEdgesG_[i];

                    std::array<ogdf::edge,2> adjSkelArr = { eSkelL, eSkelR };
                    std::array<ogdf::edge,2> adjGArr    = { eGL,    eGR    };

                    bool nodeIsCut =
                        ((cc.isCutNode[uGcc] && cc.badCutCount[uGcc] == 1) ||
                        (!cc.isCutNode[uGcc]));

                    EdgePartType t0 = EdgePartType::NONE;
                    EdgePartType t1 = EdgePartType::NONE;

                    auto getTreeEdge = [&](ogdf::edge eSkel) -> ogdf::edge {
                        auto it = blk.skel2tree.find(eSkel);
                        if (it == blk.skel2tree.end() || !(it->second)) return nullptr;
                        return it->second;
                    };

                    if (!skel.isVirtual(adjSkelArr[0])) {
                        if (adjGArr[0]) {
                            t0 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjGArr[0]);
                        }
                    } else {
                        ogdf::edge treeE0 = getTreeEdge(adjSkelArr[0]);
                        if (treeE0) {
                            EdgeDPState* state0 = skelToState[treeE0];
                            if (state0) {
                                if (blk.toCc[state0->s] == uGcc) {
                                    if      (state0->localMinusS == 0 && state0->localPlusS > 0)
                                        t0 = EdgePartType::PLUS;
                                    else if (state0->localMinusS > 0 && state0->localPlusS == 0)
                                        t0 = EdgePartType::MINUS;
                                } else {
                                    if      (state0->localMinusT == 0 && state0->localPlusT > 0)
                                        t0 = EdgePartType::PLUS;
                                    else if (state0->localMinusT > 0 && state0->localPlusT == 0)
                                        t0 = EdgePartType::MINUS;
                                }
                            }
                        }
                    }

                    if (!skel.isVirtual(adjSkelArr[1])) {
                        if (adjGArr[1]) {
                            t1 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjGArr[1]);
                        }
                    } else {
                        ogdf::edge treeE1 = getTreeEdge(adjSkelArr[1]);
                        if (treeE1) {
                            EdgeDPState* state1 = skelToState[treeE1];
                            if (state1) {
                                if (blk.toCc[state1->s] == uGcc) {
                                    if      (state1->localMinusS == 0 && state1->localPlusS > 0)
                                        t1 = EdgePartType::PLUS;
                                    else if (state1->localMinusS > 0 && state1->localPlusS == 0)
                                        t1 = EdgePartType::MINUS;
                                } else {
                                    if      (state1->localMinusT == 0 && state1->localPlusT > 0)
                                        t1 = EdgePartType::PLUS;
                                    else if (state1->localMinusT > 0 && state1->localPlusT == 0)
                                        t1 = EdgePartType::MINUS;
                                }
                            }
                        }
                    }

                    bool before = nodeIsCut;
                    nodeIsCut &= (t0 != EdgePartType::NONE &&
                                t1 != EdgePartType::NONE &&
                                t0 != t1);


                    VLOG << "[SPQR-S]  node " << uname
                            << " (cutNode=" << cc.isCutNode[uGcc]
                            << " badCutCount=" << cc.badCutCount[uGcc]
                            << " isTip=" << cc.isTip[uGcc]
                            << ") t0=" << (t0 == EdgePartType::PLUS  ? "+" :
                                            t0 == EdgePartType::MINUS ? "-" : "0")
                            << " t1=" << (t1 == EdgePartType::PLUS  ? "+" :
                                            t1 == EdgePartType::MINUS ? "-" : "0")
                            << " -> nodeIsCut=" << nodeIsCut
                            << " (before=" << before << ")\n";

                    if (!nodeIsCut) continue;


                    if (node_dp[sNode].GccCuts_last3.size() < 3) {
                        node_dp[sNode].GccCuts_last3.push_back(uGcc);
                    }


                    if (!skel.isVirtual(adjSkelArr[0])) {
                        if (adjGArr[0]) {
                            EdgePartType tt0 =
                                getNodeEdgeType(cc.nodeToOrig[uGcc], adjGArr[0]);
                            res.push_back(
                                C.node2name[cc.nodeToOrig[uGcc]] +
                                (tt0 == EdgePartType::PLUS ? "+" : "-")
                            );
                        }
                    } else {
                        ogdf::edge treeE0 = getTreeEdge(adjSkelArr[0]);
                        if (treeE0) {
                            EdgeDPState* state0 = skelToState[treeE0];
                            if (state0) {
                                if (uGcc == blk.toCc[state0->s]) {
                                    res.push_back(
                                        C.node2name[cc.nodeToOrig[uGcc]] +
                                        (state0->localPlusS > 0 ? "+" : "-")
                                    );
                                } else {
                                    res.push_back(
                                        C.node2name[cc.nodeToOrig[uGcc]] +
                                        (state0->localPlusT > 0 ? "+" : "-")
                                    );
                                }
                            }
                        }
                    }

                    if (!skel.isVirtual(adjSkelArr[1])) {
                        if (adjGArr[1]) {
                            EdgePartType tt1 =
                                getNodeEdgeType(cc.nodeToOrig[uGcc], adjGArr[1]);
                            res.push_back(
                                C.node2name[cc.nodeToOrig[uGcc]] +
                                (tt1 == EdgePartType::PLUS ? "+" : "-")
                            );
                        }
                    } else {
                        ogdf::edge treeE1 = getTreeEdge(adjSkelArr[1]);
                        if (treeE1) {
                            EdgeDPState* state1 = skelToState[treeE1];
                            if (state1) {
                                if (uGcc == blk.toCc[state1->s]) {
                                    res.push_back(
                                        C.node2name[cc.nodeToOrig[uGcc]] +
                                        (state1->localPlusS > 0 ? "+" : "-")
                                    );
                                } else {
                                    res.push_back(
                                        C.node2name[cc.nodeToOrig[uGcc]] +
                                        (state1->localPlusT > 0 ? "+" : "-")
                                    );
                                }
                            }
                        }
                    }
                }

                if (res.size() % 2 != 0) {
                    VLOG << "[SPQR-S]  WARNING: res.size()="
                            << res.size() << " is odd\n";
                    return;
                }

                if (res.size() > 2) {
                    VLOG << "[SPQR-S]  final cut sequence:";
                    for (auto &x : res) VLOG << " " << x;
                    VLOG << "\n";

                    for (size_t i = 1; i < res.size(); i += 2) {
                        std::vector<std::string> v = { res[i], res[(i+1) % res.size()] };
                        VLOG << "[SPQR-S]  emitting S-snarl: "
                                << v[0] << "  " << v[1] << "\n";
                        addSnarlTagged("S", std::move(v));
                    }
                }

                VLOG << "[SPQR-S] ==== end solveS on tree node "
                        << sNode->index() << " ====\n";
            }

            
            void solveP(ogdf::node pNode,
                        ogdf::NodeArray<SPQRsolve::NodeDPState> &node_dp,
                        ogdf::EdgeArray<SPQRsolve::EdgeDP> &edge_dp,
                        BlockData& blk,
                        const CcData& cc)
            {
                PROFILE_FUNCTION();
                auto &C = ctx();

                const ogdf::Skeleton& skel      = blk.spqr->skeleton(pNode);
                const ogdf::Graph&    skelGraph = skel.getGraph();


                std::vector<ogdf::node> poles;
                for (ogdf::node v : skelGraph.nodes) poles.push_back(v);
                std::sort(poles.begin(), poles.end(),
                        [](ogdf::node a, ogdf::node b){ return a->index() < b->index(); });
                if (poles.size() < 2) return;

                ogdf::node pole0Skel = poles[0];
                ogdf::node pole1Skel = poles[1];

                ogdf::node pole0Blk = skel.original(pole0Skel);
                ogdf::node pole1Blk = skel.original(pole1Skel);
                if (!pole0Blk || !pole1Blk) return;

                ogdf::node pole0Gcc = blk.toCc[pole0Blk];
                ogdf::node pole1Gcc = blk.toCc[pole1Blk];
                if (!pole0Gcc || !pole1Gcc) return;

                std::string name0 = C.node2name[cc.nodeToOrig[pole0Gcc]];
                std::string name1 = C.node2name[cc.nodeToOrig[pole1Gcc]];


                VLOG << "[SPQR-P] node " << pNode->index()
                        << " poles: " << name0 << " , " << name1 << "\n";


                std::vector<ogdf::adjEntry> edgeOrdering;
                for (ogdf::adjEntry adj = pole0Skel->firstAdj(); adj; adj = adj->succ())
                    edgeOrdering.push_back(adj);
                std::sort(edgeOrdering.begin(), edgeOrdering.end(),
                        [](ogdf::adjEntry a, ogdf::adjEntry b) {
                            return a->theEdge()->index() < b->theEdge()->index();
                        });

                if (cc.isCutNode[pole0Gcc]) {
                    if (cc.badCutCount[pole0Gcc] >= 2 ||
                        (cc.badCutCount[pole0Gcc] == 1 && cc.lastBad[pole0Gcc] != blk.bNode))
                    {
                        VLOG << "[SPQR-P]  rejected by cut-node filter at pole0\n";
                        return;
                    }
                }
                if (cc.isCutNode[pole1Gcc]) {
                    if (cc.badCutCount[pole1Gcc] >= 2 ||
                        (cc.badCutCount[pole1Gcc] == 1 && cc.lastBad[pole1Gcc] != blk.bNode))
                    {
                        VLOG << "[SPQR-P]  rejected by cut-node filter at pole1\n";
                        return;
                    }
                }

                auto getTreeEdge = [&](ogdf::edge eSkel) -> ogdf::edge {
                    auto it = blk.skel2tree.find(eSkel);
                    if (it == blk.skel2tree.end() || !(it->second)) {
                        return nullptr;
                    }
                    return it->second;
                };

                for (size_t i = 0; i < edgeOrdering.size(); i++) {
                    ogdf::edge eSkel = edgeOrdering[i]->theEdge();
                    if (!skel.isVirtual(eSkel)) continue;

                    ogdf::edge treeE = getTreeEdge(eSkel);
                    if (!treeE) continue;

                    ogdf::node B = (treeE->source() == pNode ? treeE->target() : treeE->source());
                    SPQRsolve::EdgeDP &st = edge_dp[treeE];
                    SPQRsolve::EdgeDPState &state =
                        (blk.parent[pNode] == B ? st.up : st.down);

                    if (!state.s || !state.t) continue;

                    if (state.s == pole0Blk) {
                        if ((state.localMinusS > 0) + (state.localPlusS > 0) == 2) {
                            VLOG << "[SPQR-P]  pole0 mixed sign on edge " << treeE->index() << "\n";
                            return;
                        }
                    } else {
                        if ((state.localMinusT > 0) + (state.localPlusT > 0) == 2) {
                            VLOG << "[SPQR-P]  pole0 mixed sign via T on edge " << treeE->index() << "\n";
                            return;
                        }
                    }

                    if (state.s == pole1Blk) {
                        if ((state.localMinusS > 0) + (state.localPlusS > 0) == 2) {
                            VLOG << "[SPQR-P]  pole1 mixed sign on edge " << treeE->index() << "\n";
                            return;
                        }
                    } else {
                        if ((state.localMinusT > 0) + (state.localPlusT > 0) == 2) {
                            VLOG << "[SPQR-P]  pole1 mixed sign via T on edge " << treeE->index() << "\n";
                            return;
                        }
                    }
                }


                for (auto &left  : {EdgePartType::PLUS, EdgePartType::MINUS}) {
                    for (auto &right : {EdgePartType::PLUS, EdgePartType::MINUS}) {

                        std::vector<ogdf::edge> leftPart, rightPart;

                        for (size_t i = 0; i < edgeOrdering.size(); i++) {
                            ogdf::edge eSkel = edgeOrdering[i]->theEdge();

                            if (!skel.isVirtual(eSkel)) {
                                ogdf::edge eG = blk.edgeToOrig[skel.realEdge(eSkel)];
                                if (!eG) continue;

                                EdgePartType l = getNodeEdgeType(cc.nodeToOrig[pole0Gcc], eG);
                                EdgePartType r = getNodeEdgeType(cc.nodeToOrig[pole1Gcc], eG);

                                if (l == left)  leftPart.push_back(eSkel);
                                if (r == right) rightPart.push_back(eSkel);
                            } else {
                                ogdf::edge treeE = getTreeEdge(eSkel);
                                if (!treeE) continue;

                                ogdf::node B = (treeE->source() == pNode
                                                ? treeE->target()
                                                : treeE->source());
                                SPQRsolve::EdgeDP &st = edge_dp[treeE];
                                SPQRsolve::EdgeDPState &state =
                                    (blk.parent[pNode] == B ? st.up : st.down);

                                if (!state.s || !state.t) continue;

                                EdgePartType l, r;
                                if (state.s == pole0Blk) {
                                    l = (state.localPlusS > 0 ? EdgePartType::PLUS
                                                            : EdgePartType::MINUS);
                                } else {
                                    l = (state.localPlusT > 0 ? EdgePartType::PLUS
                                                            : EdgePartType::MINUS);
                                }

                                if (state.s == pole1Blk) {
                                    r = (state.localPlusS > 0 ? EdgePartType::PLUS
                                                            : EdgePartType::MINUS);
                                } else {
                                    r = (state.localPlusT > 0 ? EdgePartType::PLUS
                                                            : EdgePartType::MINUS);
                                }

                                if (l == left)  leftPart.push_back(eSkel);
                                if (r == right) rightPart.push_back(eSkel);
                            }
                        }

                        if (!leftPart.empty() && leftPart == rightPart) {
                            bool ok = true;
                            if (leftPart.size() == 1) {
                                ogdf::edge eSkel = leftPart[0];
                                ogdf::edge treeE = getTreeEdge(eSkel);
                                if (treeE) {
                                    ogdf::node B = (treeE->source() == pNode
                                                    ? treeE->target()
                                                    : treeE->source());
                                    if (blk.spqr->typeOf(B) ==
                                        ogdf::StaticSPQRTree::NodeType::SNode)
                                    {
                                        for (auto &gccCut : node_dp[B].GccCuts_last3) {
                                            if (gccCut != pole0Gcc && gccCut != pole1Gcc) {
                                                ok = false;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }

                            if (ok) {
                                std::string s =
                                    C.node2name[cc.nodeToOrig[pole0Gcc]] +
                                    (left == EdgePartType::PLUS ? "+" : "-");
                                std::string t =
                                    C.node2name[cc.nodeToOrig[pole1Gcc]] +
                                    (right == EdgePartType::PLUS ? "+" : "-");

                                VLOG << "[SPQR-P]  found P-snarl candidate " << s
                                        << "  " << t << "\n";

                                std::vector<std::string> v = { s, t };
                                addSnarlTagged("P", std::move(v));
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
                auto &C = ctx();

                EdgeDPState &down = edge_dp[rrEdge].down;
                EdgeDPState &up   = edge_dp[rrEdge].up;

                if (!down.s || !down.t || !up.s || !up.t) {
                    return;
                }

                ogdf::node pole0Blk = down.s;
                ogdf::node pole1Blk = down.t;

                ogdf::node pole0Gcc = blk.toCc[pole0Blk];
                ogdf::node pole1Gcc = blk.toCc[pole1Blk];
                if (!pole0Gcc || !pole1Gcc) {
                    return;
                }

                auto hasDanglingOutside = [&](ogdf::node vGcc) {
                    if (!cc.isCutNode[vGcc]) return false;
                    if (cc.badCutCount[vGcc] >= 2) return true;
                    if (cc.badCutCount[vGcc] == 1 && cc.lastBad[vGcc] != blk.bNode) return true;
                    return false;
                };

                if (hasDanglingOutside(pole0Gcc) || hasDanglingOutside(pole1Gcc)) {
                    return;
                }

                if ((up.localMinusS   > 0 && up.localPlusS   > 0) ||
                    (up.localMinusT   > 0 && up.localPlusT   > 0) ||
                    (down.localMinusS > 0 && down.localPlusS > 0) ||
                    (down.localMinusT > 0 && down.localPlusT > 0))
                {
                    return;
                }

                EdgePartType pole0DownType = EdgePartType::NONE;
                EdgePartType pole0UpType   = EdgePartType::NONE;
                EdgePartType pole1DownType = EdgePartType::NONE;
                EdgePartType pole1UpType   = EdgePartType::NONE;

                if (down.s == pole0Blk)
                    pole0DownType = (down.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                else
                    pole0DownType = (down.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);

                if (up.s == pole0Blk)
                    pole0UpType = (up.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                else
                    pole0UpType = (up.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);

                if (down.s == pole1Blk)
                    pole1DownType = (down.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                else
                    pole1DownType = (down.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);

                if (up.s == pole1Blk)
                    pole1UpType = (up.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);
                else
                    pole1UpType = (up.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS);

                if (pole0DownType == pole0UpType) return;
                if (pole1DownType == pole1UpType) return;

                {
                    std::string s =
                        C.node2name[cc.nodeToOrig[pole0Gcc]] +
                        (pole0DownType == EdgePartType::PLUS ? "+" : "-");
                    std::string t =
                        C.node2name[cc.nodeToOrig[pole1Gcc]] +
                        (pole1DownType == EdgePartType::PLUS ? "+" : "-");

                    std::vector<std::string> v = { s, t };
                    addSnarlTagged("RR", std::move(v));
                }

                {
                    std::string s =
                        C.node2name[cc.nodeToOrig[pole0Gcc]] +
                        (pole0UpType == EdgePartType::PLUS ? "+" : "-");
                    std::string t =
                        C.node2name[cc.nodeToOrig[pole1Gcc]] +
                        (pole1UpType == EdgePartType::PLUS ? "+" : "-");

                    std::vector<std::string> v = { s, t };
                    addSnarlTagged("RR", std::move(v));
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

            static void findEdgeSnarlsBlock(BlockData &blk, const CcData &cc) {
                auto &C = ctx();
                if (!blk.Gblk || !blk.spqr) return;

                const ogdf::StaticSPQRTree &spqr = *blk.spqr;
                const ogdf::Graph &T = spqr.tree();

                ogdf::NodeArray<std::vector<ogdf::node>> sNodesPerBlk(*blk.Gblk);
                for (ogdf::node vB : blk.Gblk->nodes) {
                    sNodesPerBlk[vB].clear();
                }

                for (ogdf::node tNode : T.nodes) {
                    if (spqr.typeOf(tNode) != ogdf::StaticSPQRTree::NodeType::SNode)
                        continue;
                    const ogdf::Skeleton &skel = spqr.skeleton(tNode);
                    const ogdf::Graph &skelG  = skel.getGraph();
                    for (ogdf::node vSkel : skelG.nodes) {
                        ogdf::node vB = skel.original(vSkel);
                        if (vB) {
                            sNodesPerBlk[vB].push_back(tNode);
                        }
                    }
                }

                for (ogdf::node vB : blk.Gblk->nodes) {
                    auto &vec = sNodesPerBlk[vB];
                    std::sort(vec.begin(), vec.end(),
                            [](ogdf::node a, ogdf::node b) {
                                return a->index() < b->index();
                            });
                    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
                }

                auto shareSNode = [&](ogdf::node uB, ogdf::node vB) -> bool {
                    const auto &Su = sNodesPerBlk[uB];
                    const auto &Sv = sNodesPerBlk[vB];
                    size_t i = 0, j = 0;
                    while (i < Su.size() && j < Sv.size()) {
                        if (Su[i] == Sv[j]) return true;
                        if (Su[i]->index() < Sv[j]->index()) ++i;
                        else ++j;
                    }
                    return false;
                };

                auto hasDanglingOutside = [&](ogdf::node vB) {
                    ogdf::node vGcc = blk.toCc[vB];
                    if (!vGcc) return false;
                    if (!cc.isCutNode[vGcc]) return false;
                    if (cc.badCutCount[vGcc] >= 2) return true;
                    if (cc.badCutCount[vGcc] == 1 && cc.lastBad[vGcc] != blk.bNode) return true;
                    return false;
                };

                auto opp = [](EdgePartType t) {
                    if (t == EdgePartType::PLUS)  return EdgePartType::MINUS;
                    if (t == EdgePartType::MINUS) return EdgePartType::PLUS;
                    return EdgePartType::NONE;
                };

                for (ogdf::edge eB : blk.Gblk->edges) {
                    ogdf::node uB = eB->source();
                    ogdf::node vB = eB->target();

                    ogdf::node uGcc = blk.toCc[uB];
                    ogdf::node vGcc = blk.toCc[vB];
                    if (!uGcc || !vGcc) continue;

                    ogdf::node uG = cc.nodeToOrig[uGcc];
                    ogdf::node vG = cc.nodeToOrig[vGcc];
                    if (!uG || !vG) continue;

                    if (C.node2name[uG] == "_trash" || C.node2name[vG] == "_trash")
                        continue;

                    if (cc.isTip[uGcc] || cc.isTip[vGcc])
                        continue;

                    if (hasDanglingOutside(uB) || hasDanglingOutside(vB))
                        continue;

                    ogdf::edge eG = blk.edgeToOrig[eB];
                    if (!eG) continue;

                    EdgePartType su = getNodeEdgeType(uG, eG);
                    EdgePartType sv = getNodeEdgeType(vG, eG);
                    if (su == EdgePartType::NONE || sv == EdgePartType::NONE)
                        continue;

                    EdgePartType su_hat = opp(su);
                    EdgePartType sv_hat = opp(sv);

                    // Proposition 4.6 / Alg. FindEdgeSnarls 
                    bool u_only_su =
                        (su == EdgePartType::PLUS  && blk.degPlusBlk[uB]  == 1) ||
                        (su == EdgePartType::MINUS && blk.degMinusBlk[uB] == 1);
                    bool v_only_sv =
                        (sv == EdgePartType::PLUS  && blk.degPlusBlk[vB]  == 1) ||
                        (sv == EdgePartType::MINUS && blk.degMinusBlk[vB] == 1);

                    bool u_hasOpp =
                        (su_hat == EdgePartType::PLUS  ? blk.degPlusBlk[uB]  > 0
                                                    : blk.degMinusBlk[uB] > 0);
                    bool v_hasOpp =
                        (sv_hat == EdgePartType::PLUS  ? blk.degPlusBlk[vB]  > 0
                                                    : blk.degMinusBlk[vB] > 0);

                    bool cond_edge_case = u_only_su && v_only_sv && u_hasOpp && v_hasOpp;
                    if (!cond_edge_case) continue;

                    {
                        std::string s = C.node2name[uG] + (su == EdgePartType::PLUS ? "+" : "-");
                        std::string t = C.node2name[vG] + (sv == EdgePartType::PLUS ? "+" : "-");
                        std::vector<std::string> sn = { s, t };
                        addSnarlTagged("E", std::move(sn));
                    }

                    if (!shareSNode(uB, vB)) {
                        std::string s2 = C.node2name[uG] + (su_hat == EdgePartType::PLUS ? "+" : "-");
                        std::string t2 = C.node2name[vG] + (sv_hat == EdgePartType::PLUS ? "+" : "-");
                        std::vector<std::string> sn2 = { s2, t2 };
                        addSnarlTagged("E", std::move(sn2));
                    }
                }
            }

            void solveSPQR(BlockData& blk, const CcData& cc) {
                MARK_SCOPE_MEM("sn/solveSPQR");
                PROFILE_FUNCTION();

                if (!blk.spqr) return;
                if (!blk.Gblk || blk.Gblk->numberOfNodes() < 3) return;

                auto &C = ctx();
                const ogdf::Graph &T = blk.spqr->tree();

                ogdf::EdgeArray<SPQRsolve::EdgeDP>      edge_dp(T);
                ogdf::NodeArray<SPQRsolve::NodeDPState> node_dp(T);

                std::vector<ogdf::node> nodeOrder;
                std::vector<ogdf::edge> edgeOrder;

                SPQRsolve::dfsSPQR_order(*blk.spqr, edgeOrder, nodeOrder);

                ogdf::NodeArray<ogdf::node> blkToSkel(*blk.Gblk, nullptr);
                blk.blkToSkel = blkToSkel;

                for (ogdf::edge e : edgeOrder) {
                    SPQRsolve::processEdge(e, edge_dp, cc, blk);
                }

                for (ogdf::node v : nodeOrder) {
                    SPQRsolve::processNode(v, edge_dp, cc, blk);
                }

                SPQRsolve::solveNodes(node_dp, edge_dp, blk, cc);

                SPQRsolve::findEdgeSnarlsBlock(blk, cc);
            }



        }


        void findTips(CcData& cc) {
            MARK_SCOPE_MEM("sn/findTips");
            PROFILE_FUNCTION();

            size_t localIsolated = 0;

            for (ogdf::node v : cc.Gcc->nodes) {
                int plusCnt  = 0;
                int minusCnt = 0;
                ogdf::node vG = cc.nodeToOrig[v];

                for (auto adjE : v->adjEntries) {
                    ogdf::edge eC = adjE->theEdge();
                    ogdf::edge eG = cc.edgeToOrig[eC];

                    EdgePartType t = getNodeEdgeType(vG, eG);
                    if (t == EdgePartType::PLUS) {
                        ++plusCnt;
                    } else if (t == EdgePartType::MINUS) {
                        ++minusCnt;
                    }
                }

                if (plusCnt + minusCnt == 0) {
                    ++localIsolated;
                }


                cc.isTip[v] = (plusCnt == 0 || minusCnt == 0);
            }

            {
                std::lock_guard<std::mutex> lk(g_snarls_mtx);
                isolatedNodesCnt += localIsolated;
            }
        }


        void processCutNodes(CcData& cc) {
            MARK_SCOPE_MEM("sn/processCutNodes");
            PROFILE_FUNCTION();


            for (ogdf::node v : cc.Gcc->nodes) {
                cc.isCutNode[v]     = false;
                cc.isGoodCutNode[v] = false;   
                cc.lastBad[v]       = nullptr;
                cc.badCutCount[v]   = 0;
            }


            for (ogdf::node v : cc.Gcc->nodes) {
                if (cc.bc->typeOfGNode(v) != ogdf::BCTree::GNodeType::CutVertex)
                    continue;

                cc.isCutNode[v] = true;

                int mixedBlocks = 0;
                ogdf::node bestMixed = nullptr;

                ogdf::node vT = cc.bc->bcproper(v); 


                for (ogdf::adjEntry adjV = vT->firstAdj(); adjV; adjV = adjV->succ()) {
                    ogdf::node bT = adjV->twinNode();

                    std::vector<ogdf::edge> outPlus, outMinus;
                    getOutgoingEdgesInBlock(cc, v, bT, EdgePartType::PLUS,  outPlus);
                    getOutgoingEdgesInBlock(cc, v, bT, EdgePartType::MINUS, outMinus);

                    if (!outPlus.empty() && !outMinus.empty()) {
                        ++mixedBlocks;
                        if (!bestMixed || bT->index() < bestMixed->index())
                            bestMixed = bT;
                    }
                }

                cc.badCutCount[v] = mixedBlocks;
                cc.lastBad[v]     = bestMixed;


                cc.isGoodCutNode[v] = (mixedBlocks == 0);
            }
        }

        void findCutSnarl(CcData &cc) {
            MARK_SCOPE_MEM("sn/findCutSnarl");
            PROFILE_FUNCTION();


            // visited[v].first = already visited upon arrival with a MINUS sign
            // visited[v].second = already visited upon arrival with a PLUS sign
            ogdf::NodeArray<std::pair<bool, bool>> visited(
                *cc.Gcc, std::make_pair(false, false)
            );

            // DFS on the incidences (v, edgeType) that simulates sign-cut graphs:
            //
            // - if v is a tip -> potential endpoint in the sign-cut graph,
            // - if v is sign-consistent -> potential endpoint (will be split into two tips),
            // and we DO NOT allow a sign change in v (as after split),
            // - otherwise -> v is an internal vertex where we are allowed
            // to change the sign (unsplit component).

            std::function<void(ogdf::node, ogdf::node,
                            EdgePartType, std::vector<std::string>&)>
                dfs = [&](ogdf::node node,
                        ogdf::node prev,
                        EdgePartType edgeType,
                        std::vector<std::string> &goodNodes) -> void
            {
                // Potential endpoint = sign-consistent tip or vertex (goodCutNode)
                if ((cc.isGoodCutNode[node] || cc.isTip[node]) &&
                    ctx().node2name[cc.nodeToOrig[node]] != "_trash")
                {
                    std::string name = ctx().node2name[cc.nodeToOrig[node]];
                    name.push_back(edgeType == EdgePartType::PLUS ? '+' : '-');
                    goodNodes.push_back(std::move(name));
                }

                // Visit marking:
                // - internal vertex (neither tip nor sign-consistent) -> both signs are blocked,
                // we will not revisit this vertex,
                // - otherwise we only block the current sign.
                if (!cc.isGoodCutNode[node] && !cc.isTip[node]) {
                    // Sommet interne, non splitté dans les sign‑cut graphs
                    visited[node].first  = true;
                    visited[node].second = true;
                } else {
                    if (edgeType == EdgePartType::MINUS)
                        visited[node].first = true;
                    else
                        visited[node].second = true;
                }

                // Adjacencies of the same sign (edgeType) and of opposite sign
                std::vector<ogdf::AdjElement*> sameOutEdges, otherOutEdges;
                getAllOutgoingEdgesOfType(
                    cc, node, edgeType, sameOutEdges
                );
                getAllOutgoingEdgesOfType(
                    cc, node,
                    (edgeType == EdgePartType::PLUS ? EdgePartType::MINUS
                                                    : EdgePartType::PLUS),
                    otherOutEdges
                );

                // Sign can only change at a non-tip and non-sign-consistent vertex
                bool canGoOther = (!cc.isGoodCutNode[node] && !cc.isTip[node]);

                // 1) Propagation while maintaining the same sign
                for (auto &adjE : sameOutEdges) {
                    ogdf::node otherNode = adjE->twinNode();
                    ogdf::edge eC        = adjE->theEdge();
                    ogdf::edge eG        = cc.edgeToOrig[eC];

                    EdgePartType inType =
                        getNodeEdgeType(cc.nodeToOrig[otherNode], eG);

                    bool already =
                        (inType == EdgePartType::PLUS  && visited[otherNode].second) ||
                        (inType == EdgePartType::MINUS && visited[otherNode].first);

                    if (!already) {
                        dfs(otherNode, node, inType, goodNodes);
                    }
                }


                if (canGoOther) {
                    for (auto &adjE : otherOutEdges) {
                        ogdf::node otherNode = adjE->twinNode();
                        ogdf::edge eC        = adjE->theEdge();
                        ogdf::edge eG        = cc.edgeToOrig[eC];

                        EdgePartType inType =
                            getNodeEdgeType(cc.nodeToOrig[otherNode], eG);

                        bool already =
                            (inType == EdgePartType::PLUS  && visited[otherNode].second) ||
                            (inType == EdgePartType::MINUS && visited[otherNode].first);

                        if (!already) {
                            dfs(otherNode, node, inType, goodNodes);
                        }
                    }
                }
            };


            for (ogdf::node v : cc.Gcc->nodes) {
                for (auto t : {EdgePartType::PLUS, EdgePartType::MINUS}) {

                    if (t == EdgePartType::PLUS  && visited[v].second) continue;
                    if (t == EdgePartType::MINUS && visited[v].first)  continue;

                    std::vector<std::string> goodNodes;
                    dfs(v, nullptr, t, goodNodes);
                    if (goodNodes.size() >= 2) {
                        addSnarlTagged("CUT", std::move(goodNodes));
                    }
                }
            }
        }


        void buildBlockData(BlockData& blk, CcData& cc) {
            MARK_SCOPE_MEM("sn/blockData/build");
            PROFILE_FUNCTION();

            blk.Gblk = std::make_unique<ogdf::Graph>();

            blk.nodeToOrig.init(*blk.Gblk, nullptr);
            blk.edgeToOrig.init(*blk.Gblk, nullptr);
            blk.toCc.init(*blk.Gblk, nullptr);

            blk.degPlusBlk.init(*blk.Gblk, 0);
            blk.degMinusBlk.init(*blk.Gblk, 0);

            std::vector<ogdf::edge> edgesCc;
            edgesCc.reserve(cc.bc->hEdges(blk.bNode).size());
            for (ogdf::edge hE : cc.bc->hEdges(blk.bNode)) {
                ogdf::edge eCc = cc.bc->original(hE);
                if (eCc) edgesCc.push_back(eCc);
            }
            std::sort(edgesCc.begin(), edgesCc.end(),
                    [](ogdf::edge a, ogdf::edge b) { return a->index() < b->index(); });
            edgesCc.erase(std::unique(edgesCc.begin(), edgesCc.end(),
                                    [](ogdf::edge a, ogdf::edge b) { return a->index() == b->index(); }),
                        edgesCc.end());

            std::vector<ogdf::node> vertsCc;
            vertsCc.reserve(edgesCc.size() * 2);
            for (ogdf::edge eCc : edgesCc) {
                vertsCc.push_back(eCc->source());
                vertsCc.push_back(eCc->target());
            }
            std::sort(vertsCc.begin(), vertsCc.end(),
                    [](ogdf::node a, ogdf::node b) { return a->index() < b->index(); });
            vertsCc.erase(std::unique(vertsCc.begin(), vertsCc.end()), vertsCc.end());

            std::unordered_map<ogdf::node, ogdf::node> cc_to_blk;
            cc_to_blk.reserve(vertsCc.size());

            for (ogdf::node vCc : vertsCc) {
                ogdf::node vB = blk.Gblk->newNode();
                cc_to_blk[vCc] = vB;

                blk.toCc[vB] = vCc;                     // vB -> vCc
                ogdf::node vG = cc.nodeToOrig[vCc];     // vCc -> vG
                blk.nodeToOrig[vB] = vG;                // vB -> vG
            }

            // --- Block edges + update of signed degrees in the block ---
            for (ogdf::edge eCc : edgesCc) {
                auto itS = cc_to_blk.find(eCc->source());
                auto itT = cc_to_blk.find(eCc->target());
                if (itS != cc_to_blk.end() && itT != cc_to_blk.end()) {
                    ogdf::node uB = itS->second;
                    ogdf::node vB = itT->second;

                    ogdf::edge eB = blk.Gblk->newEdge(uB, vB);
                    ogdf::edge eG = cc.edgeToOrig[eCc];
                    blk.edgeToOrig[eB] = eG; // eB -> eG

                    ogdf::node uG = blk.nodeToOrig[uB];
                    ogdf::node vG = blk.nodeToOrig[vB];

                    EdgePartType su = getNodeEdgeType(uG, eG);
                    EdgePartType sv = getNodeEdgeType(vG, eG);

                    if (su == EdgePartType::PLUS)
                        blk.degPlusBlk[uB]++;
                    else if (su == EdgePartType::MINUS)
                        blk.degMinusBlk[uB]++;

                    if (sv == EdgePartType::PLUS)
                        blk.degPlusBlk[vB]++;
                    else if (sv == EdgePartType::MINUS)
                        blk.degMinusBlk[vB]++;
                }
            }

            // --- Construction of the SPQR tree if the block is non-trivial ---
            if (blk.Gblk->numberOfNodes() >= 3) {
                {
                    MARK_SCOPE_MEM("sn/blockData/spqr_build");
                    blk.spqr = std::make_unique<ogdf::StaticSPQRTree>(*blk.Gblk);
                }
                const ogdf::Graph &T = blk.spqr->tree();

                // Skeleton mapping -> SPQR tree edge
                blk.skel2tree.clear();
                blk.skel2tree.reserve(2 * T.numberOfEdges());

                std::vector<ogdf::edge> treeEdges;
                treeEdges.reserve(T.numberOfEdges());
                for (ogdf::edge te : T.edges) treeEdges.push_back(te);
                std::sort(treeEdges.begin(), treeEdges.end(),
                        [](ogdf::edge a, ogdf::edge b) { return a->index() < b->index(); });

                for (ogdf::edge te : treeEdges) {
                    if (auto eSrc = blk.spqr->skeletonEdgeSrc(te))
                        blk.skel2tree[eSrc] = te;
                    if (auto eTgt = blk.spqr->skeletonEdgeTgt(te))
                        blk.skel2tree[eTgt] = te;
                }

                // Parenting in the SPQR tree (for deterministic navigation)
                blk.parent.init(T, nullptr);
                ogdf::node root = blk.spqr->rootNode();
                if (!root) return;
                blk.parent[root] = root;

                std::deque<ogdf::node> q;
                ogdf::NodeArray<bool> seen(T, false);
                q.push_back(root);
                seen[root] = true;

                while (!q.empty()) {
                    ogdf::node cur = q.front();
                    q.pop_front();

                    std::vector<ogdf::node> nbrs;
                    nbrs.reserve(cur->degree());
                    for (ogdf::adjEntry adj = cur->firstAdj(); adj; adj = adj->succ()) {
                        ogdf::node nxt = adj->twinNode();
                        if (nxt) nbrs.push_back(nxt);
                    }
                    std::sort(nbrs.begin(), nbrs.end(),
                            [](ogdf::node a, ogdf::node b) { return a->index() < b->index(); });
                    nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());

                    for (ogdf::node nxt : nbrs) {
                        if (!seen[nxt]) {
                            seen[nxt]  = true;
                            blk.parent[nxt] = cur;
                            q.push_back(nxt);
                        }
                    }
                }
            }
        }



        struct BlockPrep {
            CcData* cc;
            ogdf::node bNode;

            std::unique_ptr<BlockData> blk;

            BlockPrep(CcData* cc_, ogdf::node b) : cc(cc_), bNode(b), blk(nullptr) {}

            BlockPrep() = default;
            BlockPrep(const BlockPrep&) = delete;
            BlockPrep& operator=(const BlockPrep&) = delete;
            BlockPrep(BlockPrep&&) = default;
            BlockPrep& operator=(BlockPrep&&) = default;
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
                        for (ogdf::node v : cc->bc->bcTree().nodes) {
                            if (cc->bc->typeOfBNode(v) == BCTree::BNodeType::BComp) {

                                localPreps.emplace_back(cc, v);
                            }
                        }
                    }

                    {
                        static std::mutex prepMutex;
                        std::lock_guard<std::mutex> lock(prepMutex);
                        blockPreps->reserve(blockPreps->size() + localPreps.size());
                        for (auto &bp : localPreps) {
                            blockPreps->emplace_back(std::move(bp));
                        }
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


        void* worker_block_build(void* arg) {
            std::unique_ptr<ThreadBlocksArgs> targs(static_cast<ThreadBlocksArgs*>(arg));
            size_t tid = targs->tid;
            size_t numThreads = targs->numThreads;
            size_t blocks = targs->blocks;
            size_t* nextIndex = targs->nextIndex;
            std::mutex* workMutex = targs->workMutex;
            std::vector<BlockPrep>* blockPreps = targs->blockPreps;

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
                    blockPreps->at(bid).blk = std::make_unique<BlockData>();
                    BlockData& blk = *blockPreps->at(bid).blk;
                    blk.bNode = (*blockPreps)[bid].bNode;

                    {
                        //MEM_TIME_BLOCK("SPQR: build (snarl worker)");
                        buildBlockData(blk, *(*blockPreps)[bid].cc);
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

            std::cout << "Thread " << tid << " built " << processed << " blocks (SPQR build)\n";
            return nullptr;
        }

        void* worker_block_solve(void* arg) {
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
                    BlockPrep& prep = (*blockPreps)[bid];
                    if (!prep.blk) continue;
                    BlockData& blk = *prep.blk;

                    {
                        //MEM_TIME_BLOCK("Algorithm: snarl solve (worker)");
                        if (blk.Gblk && blk.Gblk->numberOfNodes() >= 3) {
                            SPQRsolve::solveSPQR(blk, *prep.cc);
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

            std::cout << "Thread " << tid << " solved " << processed << " blocks (SPQR solve)\n";
            return nullptr;
        }

        void solve() {
            std::cout << "Finding snarls (multiprocess...)\n";
            PROFILE_FUNCTION();
            auto& C = ctx();
            Graph& G = C.G;


            struct SnarlTypeCounts {
                uint64_t cut;
                uint64_t S;
                uint64_t P;
                uint64_t RR;
                uint64_t E;
            };

            NodeArray<int> compIdx(G);
            int nCC = 0;
            std::vector<std::vector<node>> bucket;
            std::vector<std::vector<edge>> edgeBuckets;

            {
                PhaseSampler io_sampler(g_stats_io);

                MARK_SCOPE_MEM("sn/phase/ComputeCC");
                nCC = connectedComponents(G, compIdx);

                bucket.assign(nCC, {});
                {
                    MARK_SCOPE_MEM("sn/phase/BucketNodes");
                    for (node v : G.nodes) {
                        bucket[compIdx[v]].push_back(v);
                    }
                }

                edgeBuckets.assign(nCC, {});
                {
                    MARK_SCOPE_MEM("sn/phase/BucketEdges");
                    for (edge e : G.edges) {
                        edgeBuckets[compIdx[e->source()]].push_back(e);
                    }
                }
            }

            size_t maxProcs = std::thread::hardware_concurrency();
            if (maxProcs == 0) maxProcs = 1;
            size_t procs = std::min({ (size_t)C.threads, (size_t)std::max(1, nCC), maxProcs });

            struct ChildInfo {
                pid_t pid;
                std::string outPath; 
                std::string cntPath; 
            };

            std::vector<ChildInfo> children;
            children.reserve(procs);

            uint64_t build_start_us = nowMicros();
            g_stats_build.start_rss.store(currentRSSBytes(), std::memory_order_relaxed);

            for (size_t p = 0; p < procs; ++p) {
                size_t start = (size_t)((uint64_t)p * (uint64_t)nCC / (uint64_t)procs);
                size_t end   = (size_t)((uint64_t)(p+1) * (uint64_t)nCC / (uint64_t)procs);
                if (start >= (size_t)nCC) start = (size_t)nCC;
                if (end   >  (size_t)nCC) end   = (size_t)nCC;
                if (start >= end) continue;

                pid_t pid = ::fork();
                if (pid < 0) {
                    std::cerr << "fork failed: " << strerror(errno) << "\n";
                    continue;
                }

                if (pid == 0) {
                    // ============================
                    // ========  child  ==========
                    // ============================
                    g_is_child_process = true;

                    std::vector<std::vector<std::string>> localSnarls;
                    tls_snarl_buffer = &localSnarls;

                    {
                        std::string base = "/dev/shm";
                        int test = ::access(base.c_str(), W_OK);
                        if (test != 0) base = "/tmp";
                        g_snarl_out_path = base + "/snarls_" + std::to_string(::getpid()) + ".bin";
                        g_snarl_out_fd = ::open(g_snarl_out_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
                        if (g_snarl_out_fd < 0) {
                            std::fprintf(stderr, "[child %d] open(%s) failed: %s\n",
                                        (int)::getpid(), g_snarl_out_path.c_str(), std::strerror(errno));
                            _exit(111);
                        }
                    }


                    for (size_t cid = start; cid < end; ++cid) {
                        std::unique_ptr<CcData> cc(new CcData());
                        {
                            MARK_SCOPE_MEM("sn/worker_component/gcc_rebuild");
                            cc->Gcc = std::make_unique<Graph>();
                            cc->nodeToOrig.init(*cc->Gcc, nullptr);
                            cc->edgeToOrig.init(*cc->Gcc, nullptr);
                            cc->isTip.init(*cc->Gcc, false);
                            cc->isCutNode.init(*cc->Gcc, false);
                            cc->isGoodCutNode.init(*cc->Gcc, false);
                            cc->lastBad.init(*cc->Gcc, nullptr);
                            cc->badCutCount.init(*cc->Gcc, 0);
                            cc->degPlus.init(*cc->Gcc, 0);
                            cc->degMinus.init(*cc->Gcc, 0);

                            std::unordered_map<node, node> orig_to_cc;
                            orig_to_cc.reserve(bucket[cid].size());

                            for (node vG : bucket[cid]) {
                                node vC = cc->Gcc->newNode();
                                cc->nodeToOrig[vC] = vG;
                                orig_to_cc[vG] = vC;
                            }

                            for (edge e : edgeBuckets[cid]) {
                                auto eC = cc->Gcc->newEdge(orig_to_cc[e->source()], orig_to_cc[e->target()]);
                                cc->edgeToOrig[eC] = e;

                                cc->degPlus[orig_to_cc[e->source()]] += (getNodeEdgeType(e->source(), e) == EdgePartType::PLUS ? 1 : 0);
                                cc->degMinus[orig_to_cc[e->source()]] += (getNodeEdgeType(e->source(), e) == EdgePartType::MINUS ? 1 : 0);
                                cc->degPlus[orig_to_cc[e->target()]] += (getNodeEdgeType(e->target(), e) == EdgePartType::PLUS ? 1 : 0);
                                cc->degMinus[orig_to_cc[e->target()]] += (getNodeEdgeType(e->target(), e) == EdgePartType::MINUS ? 1 : 0);
                            }
                        }

                        std::vector<BlockPrep> blockPreps;
                        {
                            MARK_SCOPE_MEM("sn/worker_bcTree/build");
                            cc->bc = std::make_unique<BCTree>(*cc->Gcc);

                            MARK_SCOPE_MEM("sn/worker_bcTree/collect_B_nodes");
                            for (ogdf::node v : cc->bc->bcTree().nodes) {
                                if (cc->bc->typeOfBNode(v) == BCTree::BNodeType::BComp) {
                                    blockPreps.emplace_back(cc.get(), v);
                                }
                            }
                        }

                        {
                            MARK_SCOPE_MEM("sn/phase/block_SPQR_build");
                            for (auto &bp : blockPreps) {
                                bp.blk = std::make_unique<BlockData>();
                                BlockData& blk = *bp.blk;
                                blk.bNode = bp.bNode;
                                buildBlockData(blk, *bp.cc);
                            }
                        }

                        {
                            findTips(*cc);
                            if (cc->bc->numberOfCComps() > 0) {
                                processCutNodes(*cc);
                            }
                            findCutSnarl(*cc);
                            flushThreadLocalSnarls(localSnarls);
                        }

                        {
                            MARK_SCOPE_MEM("sn/phase/block_SPQR_solve");
                            for (auto &bp : blockPreps) {
                                if (!bp.blk) continue;
                                BlockData& blk = *bp.blk;
                                if (blk.Gblk && blk.Gblk->numberOfNodes() >= 3) {
                                    SPQRsolve::solveSPQR(blk, *bp.cc);
                                    flushThreadLocalSnarls(localSnarls);
                                }
                            }
                        }
                    }

                    if (tls_snarl_buffer) {
                        flushThreadLocalSnarls(*tls_snarl_buffer);
                        tls_snarl_buffer = nullptr;
                    }
                    if (g_snarl_out_fd >= 0) {
                        ::fsync(g_snarl_out_fd);
                        ::close(g_snarl_out_fd);
                        g_snarl_out_fd = -1;
                    }

                    {
                        SnarlTypeCounts cnts{
                            g_cnt_cut.load(std::memory_order_relaxed),
                            g_cnt_S.load(std::memory_order_relaxed),
                            g_cnt_P.load(std::memory_order_relaxed),
                            g_cnt_RR.load(std::memory_order_relaxed),
                            g_cnt_E.load(std::memory_order_relaxed)
                        };

                        std::string base = "/dev/shm";
                        int test = ::access(base.c_str(), W_OK);
                        if (test != 0) base = "/tmp";
                        std::string cntPath = base + "/snarls_" + std::to_string(::getpid()) + ".cnt";

                        int fdCnt = ::open(cntPath.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
                        if (fdCnt >= 0) {
                            write_all_fd(fdCnt, &cnts, sizeof(cnts));
                            ::fsync(fdCnt);
                            ::close(fdCnt);
                        }
                    }

                    _exit(0);
                } else {
                    // ============================
                    // ========  PARENT  ==========
                    // ============================
                    std::string base = "/dev/shm";
                    int test = ::access(base.c_str(), W_OK);
                    if (test != 0) base = "/tmp";
                    std::string outPath = base + "/snarls_" + std::to_string(pid) + ".bin";
                    std::string cntPath = base + "/snarls_" + std::to_string(pid) + ".cnt";
                    children.push_back({pid, outPath, cntPath});
                }
            }

            uint64_t build_end_us = nowMicros();
            g_stats_build.elapsed_us.store(build_end_us - build_start_us, std::memory_order_relaxed);
            g_stats_build.peak_rss.store(currentRSSBytes(), std::memory_order_relaxed);

            uint64_t logic_start_us = nowMicros();
            g_stats_logic.start_rss.store(currentRSSBytes(), std::memory_order_relaxed);

            for (auto &chi : children) {
                int status = 0;
                pid_t wp = ::waitpid(chi.pid, &status, 0);
                if (wp < 0) {
                    std::cerr << "waitpid failed for " << chi.pid << ": " << strerror(errno) << "\n";
                    continue;
                }

                if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                    std::cerr << "child " << chi.pid << " exited abnormally (status=" << status
                            << "), skipping files " << chi.outPath << " / " << chi.cntPath << "\n";
                    ::unlink(chi.outPath.c_str());
                    ::unlink(chi.cntPath.c_str());
                    continue;
                }

                {
                    int fd = ::open(chi.outPath.c_str(), O_RDONLY);
                    if (fd >= 0) {
                        while (true) {
                            uint32_t k;
                            if (!read_all_fd(fd, &k, sizeof(k))) break;

                            std::vector<std::string> sn;
                            sn.reserve(k);
                            bool ok = true;
                            for (uint32_t i = 0; i < k; ++i) {
                                uint32_t len = 0;
                                if (!read_all_fd(fd, &len, sizeof(len))) { ok = false; break; }
                                std::string s;
                                s.resize(len);
                                if (len > 0) {
                                    if (!read_all_fd(fd, s.data(), len)) { ok = false; break; }
                                }
                                sn.push_back(std::move(s));
                            }
                            if (!ok) break;


                            tryCommitSnarl(std::move(sn));
                        }

                        ::close(fd);
                        ::unlink(chi.outPath.c_str());
                    }
                }


                {
                    int fdCnt = ::open(chi.cntPath.c_str(), O_RDONLY);
                    if (fdCnt >= 0) {
                        SnarlTypeCounts cnts{};
                        if (read_all_fd(fdCnt, &cnts, sizeof(cnts))) {
                            g_cnt_cut.fetch_add(cnts.cut, std::memory_order_relaxed);
                            g_cnt_S.fetch_add(cnts.S, std::memory_order_relaxed);
                            g_cnt_P.fetch_add(cnts.P, std::memory_order_relaxed);
                            g_cnt_RR.fetch_add(cnts.RR, std::memory_order_relaxed);
                            g_cnt_E.fetch_add(cnts.E, std::memory_order_relaxed);
                        }
                        ::close(fdCnt);
                        ::unlink(chi.cntPath.c_str());
                    }
                }
            }

            uint64_t logic_end_us = nowMicros();
            g_stats_logic.elapsed_us.store(logic_end_us - logic_start_us, std::memory_order_relaxed);
            g_stats_logic.peak_rss.store(currentRSSBytes(), std::memory_order_relaxed);

            auto to_ms  = [](uint64_t us){ return us / 1000.0; };
            auto to_mib = [](size_t bytes){ return bytes / (1024.0 * 1024.0); };

            auto print_phase = [&](const char* name, const PhaseStats& st) {
                double t_ms = to_ms(st.elapsed_us.load());
                double peak_mib = to_mib(st.peak_rss.load());
                double delta_mib = to_mib(st.peak_rss.load() > st.start_rss.load()
                                        ? st.peak_rss.load() - st.start_rss.load() : 0);
                std::cout << "[SNARLS] " << name << " : time=" << t_ms
                        << " ms, peakRSS=" << peak_mib << " MiB, peakDelta=" << delta_mib << " MiB\n";
            };

            print_snarl_type_counters();
            print_phase("I/O",   g_stats_io);
            print_phase("BUILD", g_stats_build);
            print_phase("LOGIC", g_stats_logic);
        }

    }

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

        } else if (s == "--block-stats") {
            solver::blockstats::g_run_block_stats = true;
            solver::blockstats::g_output_path =
                nextArgOrDie(args, i, "--block-stats");

        } else if (s == "-m") {
            C.stackSize = std::stoull(nextArgOrDie(args, i, "-m"));

        } else {
            std::cerr << "Unknown argument: " << s << "\n";
            usage(args[0].c_str());
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
        //MEM_TIME_BLOCK("I/O: read graph");
        MARK_SCOPE_MEM("io/read_graph");
        PROFILE_BLOCK("Graph reading");
        GraphIO::readGraph();
    }

    if (solver::blockstats::g_run_block_stats) {
        solver::blockstats::compute_block_sizes_and_write();

        PROFILING_REPORT();

        logger::info("Process PeakRSS: {:.2f} GiB",
                     memtime::peakRSSBytes() / (1024.0 * 1024.0 * 1024.0));

        mark::report();
        if (!g_report_json_path.empty()) {
            mark::report_to_json(g_report_json_path);
        }

        return 0;
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
        //MEM_TIME_BLOCK("I/O: write output");
        MARK_SCOPE_MEM("io/write_output");
        PROFILE_BLOCK("Writing output");
        TIME_BLOCK("Writing output");
        GraphIO::writeSuperbubbles();
    }

    std::cout << "Snarls found: " << snarlsFound << std::endl;

    PROFILING_REPORT();

    logger::info("Process PeakRSS: {:.2f} GiB",
                 memtime::peakRSSBytes() / (1024.0 * 1024.0 * 1024.0));

    mark::report();
    if (!g_report_json_path.empty()) {
        mark::report_to_json(g_report_json_path);
    }

    return 0;
}