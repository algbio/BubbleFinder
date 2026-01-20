#include "util/ogdf_all.hpp"

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
#include <atomic>
#include <array>
#include <cstdint>
#include <queue>
#include <algorithm>
#include <cctype>

#include <sys/resource.h>
#include <sys/time.h>

#include <cerrno>
#include <cstring>

#ifdef __APPLE__
#  include <mach/mach.h>
#endif

#ifdef __linux__
#include <cstdio>
#endif

#include <unistd.h>
#include <sys/resource.h>

#include "io/graph_io.hpp"
#include "util/timer.hpp"
#include "util/logger.hpp"
#include "util/profiling.hpp"
#include "fas.h"

#include "util/mark_scope.hpp"
#include "util/mem_time.hpp"
#include "util/phase_accum.hpp"

#include "util/clsd_interface.hpp"

bool VERBOSE = false;
#define VLOG if (VERBOSE) std::cerr


// -----------------------------------------------------------------------------
// Metrics instrumentation (RSS and timing per phase)
// -----------------------------------------------------------------------------
namespace metrics {

    enum class Phase : uint8_t { IO = 0, BUILD = 1, LOGIC = 2, COUNT = 3 };

    struct PhaseState {
        std::atomic<bool>   running{false};
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
            struct rusage ru{};
            getrusage(RUSAGE_SELF, &ru);
            return (size_t)ru.ru_maxrss * 1024ull;
        }
        long rss_pages = 0;
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
        const size_t cur  = currentRSS();
        size_t delta = (cur >= base ? (cur - base) : 0);
        size_t prev  = st.peak_rss_delta.load(std::memory_order_relaxed);
        while (delta > prev &&
               !st.peak_rss_delta.compare_exchange_weak(prev, delta,
                                                        std::memory_order_relaxed)) {
        }
    }

    inline void endPhase(Phase p) {
        auto &st = states()[(size_t)p];
        if (!st.running.load(std::memory_order_acquire)) return;
        updateRSS(p);
        const uint64_t t1 = now_ns();
        const uint64_t t0 = st.start_ns.load(std::memory_order_relaxed);
        const uint64_t d  = (t1 >= t0 ? (t1 - t0) : 0);
        uint64_t prev = st.elapsed_ns.load(std::memory_order_relaxed);
        st.elapsed_ns.store(prev + d, std::memory_order_relaxed);
        st.running.store(false, std::memory_order_release);
    }

    struct Snapshot {
        uint64_t elapsed_ns;
        size_t   peak_rss_delta;
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


// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------
using namespace ogdf;

static std::string g_report_json_path;


// -----------------------------------------------------------------------------
// CLI helpers
// -----------------------------------------------------------------------------
static void usage(const char* prog, int exitCode) {
    struct CommandHelp {
        const char* name;
        const char* desc;
    };

    struct OptionHelp {
        const char* flag;
        const char* arg;
        const char* desc;
    };

    static const CommandHelp commands[] = {
        { "superbubbles",
          "Bidirected superbubbles (GFA -> bidirected by default)" },
        { "directed-superbubbles",
          "Directed superbubbles (directed graph)" },
        { "snarls",
          "Snarls (typically on bidirected graphs from GFA)" }
    };

    static const OptionHelp options[] = {
        { "-g", "<file>",  "Input graph file (possibly compressed)" },
        { "-o", "<file>",  "Output file" },
        { "-j", "<threads>","Number of threads" },
        { "--gfa", nullptr,          "Force GFA input (bidirected)" },
        { "--gfa-directed", nullptr, "Force GFA input interpreted as directed graph" },
        { "--graph", nullptr,
          "Force .graph text format (see 'Format options' above)" },
        { "--report-json", "<file>", "Write JSON metrics report" },
        { "-m", "<bytes>",           "Stack size in bytes" },
        { "-h, --help", nullptr,     "Show this help message and exit" }
        // -sanity is intentionally undocumented (internal/debug)
    };

    std::cerr << "Usage:\n"
              << "  " << prog
              << " <command> -g <graphFile> -o <outputFile> [options]\n\n";

    std::cerr << "Commands:\n";
    for (const auto &c : commands) {
        std::cerr << "  " << c.name << "\n"
                  << "      " << c.desc << "\n";
    }
    std::cerr << "\n";

    std::cerr << "Format options (input format):\n"
              << "  --gfa\n"
              << "      GFA input (bidirected).\n"
              << "  --gfa-directed\n"
              << "      GFA input interpreted as a directed graph.\n"
              << "  --graph\n"
              << "      .graph text format with one directed edge per line:\n"
              << "        • first line: two integers n and m\n"
              << "            - n = number of distinct node IDs declared\n"
              << "            - m = number of directed edges\n"
              << "        • next m lines: 'u v' (separated by whitespace),\n"
              << "            each describing a directed edge from u to v.\n"
              << "        • u and v are arbitrary node identifiers (strings\n"
              << "            without whitespace).\n"
              << "  If none of these is given, the format is auto-detected\n"
              << "  from the file extension (e.g. .gfa, .graph).\n\n";

    std::cerr << "Compression:\n"
              << "  Compression is auto-detected from the file name suffix:\n"
              << "    .gz / .bgz  -> gzip\n"
              << "    .bz2        -> bzip2\n"
              << "    .xz         -> xz\n\n";

    std::cerr << "General options:\n";
    for (const auto &o : options) {
        std::cerr << "  " << o.flag;
        if (o.arg) {
            std::cerr << " " << o.arg;
        }
        std::cerr << "\n      " << o.desc << "\n";
    }

    std::exit(exitCode);
}


static std::string nextArgOrDie(const std::vector<std::string>& a,
                                std::size_t& i,
                                const char* flag) {
    if (++i >= a.size() || (a[i][0] == '-' && a[i] != "-")) {
        std::cerr << "Error: missing argument after " << flag << "\n";
        usage(a[0].c_str(), 1);
    }
    return a[i];
}
    
static std::string toLowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return s;
}

// Detect compression from file name and return the "core" extension
// Example:
//  foo.gfa.gz  -> compression = Gzip, coreExtOut = "gfa"
//   foo.graph   -> compression = None,  coreExtOut = "graph"
static Context::Compression
detectCompressionAndCoreExt(const std::string &path,
                            std::string &coreExtOut)
{
    std::string filename = path;
    auto slashPos = filename.find_last_of("/\\");
    if (slashPos != std::string::npos) {
        filename = filename.substr(slashPos + 1);
    }

    coreExtOut.clear();

    auto dotPos = filename.find_last_of('.');
    if (dotPos == std::string::npos) {
        return Context::Compression::None;
    }

    std::string lastExt = toLowerCopy(filename.substr(dotPos + 1));
    std::string base    = filename.substr(0, dotPos);

    Context::Compression comp = Context::Compression::None;

    if (lastExt == "gz" || lastExt == "bgz") {
        comp = Context::Compression::Gzip;
    } else if (lastExt == "bz2") {
        comp = Context::Compression::Bzip2;
    } else if (lastExt == "xz") {
        comp = Context::Compression::Xz;
    } else {
        coreExtOut = lastExt;
        return Context::Compression::None;
    }

    auto dotPos2 = base.find_last_of('.');
    if (dotPos2 != std::string::npos) {
        coreExtOut = toLowerCopy(base.substr(dotPos2 + 1));
    } else {
        coreExtOut.clear();
    }

    return comp;
}


// -----------------------------------------------------------------------------
// Path checks (input/output)
// -----------------------------------------------------------------------------

static bool inputFileReadable(const std::string &path, std::string &errOut) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) {
        errOut = std::string("stat failed: ") + std::strerror(errno);
        return false;
    }
    if (!S_ISREG(st.st_mode)) {
        errOut = "path exists but is not a regular file";
        return false;
    }
    if (access(path.c_str(), R_OK) != 0) {
        errOut = std::string("no read permission: ") + std::strerror(errno);
        return false;
    }
    return true;
}

static bool outputParentDirWritable(const std::string &path, std::string &errOut) {
    std::string dir;
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        dir = ".";          
    } else if (pos == 0) {
        dir = "/";          
    } else {
        dir = path.substr(0, pos);
    }

    struct stat st{};
    if (stat(dir.c_str(), &st) != 0) {
        errOut = std::string("cannot stat output directory '") + dir +
                 "': " + std::strerror(errno);
        return false;
    }
    if (!S_ISDIR(st.st_mode)) {
        errOut = "'" + dir + "' is not a directory";
        return false;
    }
    if (access(dir.c_str(), W_OK) != 0) {
        errOut = std::string("no write permission on '") + dir +
                 "': " + std::strerror(errno);
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Argument parsing
// -----------------------------------------------------------------------------
void readArgs(int argc, char** argv) {
    auto &C = ctx();

    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 2) {
        usage(args[0].c_str(), 1);
    }

    std::size_t i = 1;

    // 1) Subcommand
    const std::string cmd = args[i];

    if (cmd == "-h" || cmd == "--help") {
        usage(args[0].c_str(), 0);
    } else if (cmd == "superbubbles") {
        C.bubbleType           = Context::BubbleType::SUPERBUBBLE;
        C.directedSuperbubbles = false;
    } else if (cmd == "directed-superbubbles") {
        C.bubbleType           = Context::BubbleType::SUPERBUBBLE;
        C.directedSuperbubbles = true;
    } else if (cmd == "snarls") {
        C.bubbleType           = Context::BubbleType::SNARL;
        C.directedSuperbubbles = false;
    } else if (cmd == "ultrabubbles") {
        C.bubbleType           = Context::BubbleType::ULTRABUBBLE;
        C.directedSuperbubbles = false;
    } else {
        std::cerr << "Error: unknown command '" << cmd
                  << "'. Expected one of: superbubbles, directed-superbubbles, snarls.\n\n";
        usage(args[0].c_str(), 1);
    }

    ++i;

    // 2) Options
    for (; i < args.size(); ++i) {
        const std::string &s = args[i];

        if (s == "-g") {
            C.graphPath = nextArgOrDie(args, i, "-g");

        } else if (s == "-o") {
            C.outputPath = nextArgOrDie(args, i, "-o");

        } else if (s == "--gfa") {
            if (C.inputFormat != Context::InputFormat::Auto &&
                C.inputFormat != Context::InputFormat::Gfa) {
                std::cerr << "Error: multiple conflicting input format options "
                             "(--gfa / --gfa-directed / --graph).\n";
                std::exit(1);
            }
            C.inputFormat = Context::InputFormat::Gfa;

        } else if (s == "--gfa-directed") {
            if (C.inputFormat != Context::InputFormat::Auto &&
                C.inputFormat != Context::InputFormat::GfaDirected) {
                std::cerr << "Error: multiple conflicting input format options "
                             "(--gfa / --gfa-directed / --graph).\n";
                std::exit(1);
            }
            C.inputFormat = Context::InputFormat::GfaDirected;

        } else if (s == "--graph") {
            if (C.inputFormat != Context::InputFormat::Auto &&
                C.inputFormat != Context::InputFormat::Graph) {
                std::cerr << "Error: multiple conflicting input format options "
                             "(--gfa / --gfa-directed / --graph).\n";
                std::exit(1);
            }
            C.inputFormat = Context::InputFormat::Graph;

        } else if (s == "--report-json") {
            g_report_json_path = nextArgOrDie(args, i, "--report-json");

        } else if (s == "-j") {
            const std::string v = nextArgOrDie(args, i, "-j");
            try {
                C.threads = std::stoi(v);
            } catch (const std::exception &) {
                std::cerr << "Error: invalid value for -j <threads>: '" << v
                          << "'. Expected a positive integer.\n";
                std::exit(1);
            }
            if (C.threads <= 0) {
                std::cerr << "Error: -j <threads> must be a positive integer (got "
                          << C.threads << ").\n";
                std::exit(1);
            }

        } else if (s == "-m") {
            const std::string v = nextArgOrDie(args, i, "-m");
            try {
                C.stackSize = std::stoull(v);
            } catch (const std::exception &) {
                std::cerr << "Error: invalid value for -m <bytes>: '" << v
                          << "'. Expected a positive integer (number of bytes).\n";
                std::exit(1);
            }
            if (C.stackSize == 0) {
                std::cerr << "Error: -m <bytes> must be a positive integer.\n";
                std::exit(1);
            }

        } else if (s == "-sanity") {
            std::exit(0);

        } else if (s == "-h" || s == "--help") {
            usage(args[0].c_str(), 0);

        } else {
            std::cerr << "Unknown argument: " << s << "\n";
            usage(args[0].c_str(), 1);
        }
    }

    // 3) Basic checks
    if (C.graphPath.empty()) {
        std::cerr << "Error: missing -g <graphFile>.\n\n";
        usage(args[0].c_str(), 1);
    }
    if (C.outputPath.empty()) {
        std::cerr << "Error: missing -o <outputFile>.\n\n";
        usage(args[0].c_str(), 1);
    }

    // 4) Auto-detect compression and input format (if still Auto)
    std::string coreExt;
    C.compression = detectCompressionAndCoreExt(C.graphPath, coreExt);

    if (C.inputFormat == Context::InputFormat::Auto) {
        if (coreExt == "gfa" || coreExt == "gfa1" || coreExt == "gfa2") {
            if (C.directedSuperbubbles) {
                C.inputFormat = Context::InputFormat::GfaDirected;
            } else {
                C.inputFormat = Context::InputFormat::Gfa;
            }
        } else if (coreExt == "graph") {
            C.inputFormat = Context::InputFormat::Graph;
        } else {
            std::cerr << "Error: could not autodetect input format from file '"
                      << C.graphPath << "'.\n"
                      << "       Please specify one of --gfa, --gfa-directed or --graph.\n";
            std::exit(1);
        }
    }

    C.gfaInput = (C.inputFormat == Context::InputFormat::Gfa ||
                  C.inputFormat == Context::InputFormat::GfaDirected);
}


// -----------------------------------------------------------------------------
// Global counters / OGDF accounting
// -----------------------------------------------------------------------------
size_t snarlsFound = 0;
size_t isolatedNodesCnt = 0;

static std::atomic<long long> g_ogdf_total_us{0};
static std::atomic<size_t> g_phase_io_max_rss{0};
static std::atomic<size_t> g_phase_build_max_rss{0};
static std::atomic<size_t> g_phase_logic_max_rss{0};

static inline void __phase_rss_update(std::atomic<size_t> &dst) {
    size_t cur = memtime::peakRSSBytes();
    size_t old = dst.load(std::memory_order_relaxed);
    while (cur > old &&
           !dst.compare_exchange_weak(old, cur, std::memory_order_relaxed)) {}
}

#define PHASE_RSS_UPDATE_IO_LEGACY()    __phase_rss_update(g_phase_io_max_rss)
#define PHASE_RSS_UPDATE_BUILD_LEGACY() __phase_rss_update(g_phase_build_max_rss)
#define PHASE_RSS_UPDATE_LOGIC_LEGACY() __phase_rss_update(g_phase_logic_max_rss)


// -----------------------------------------------------------------------------
// OGDF accounting helpers
// -----------------------------------------------------------------------------
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



                

        void solveStreaming() {
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
            constexpr size_t kMinThreadStackSize = 1024 * 1024 * 1024; // 8 MiB
            struct StrPairHash {
                size_t operator()(const std::pair<std::string,std::string> &p) const noexcept {
                    std::hash<std::string> h;
                    size_t seed = h(p.first);
                    seed ^= h(p.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    return seed;
                }
            };

            // For each thread (typically in worker_block_solve), we keep the pairs
            // of vertices (without sign) that already have an S/P/RR snarl.

            thread_local std::unordered_set<std::pair<std::string,std::string>, StrPairHash>
                tls_spqr_seen_endpoint_pairs;

            inline void canonicalize_pair(std::vector<std::string> &v) {
                if (v.size() == 2 && v[1] < v[0]) {
                    std::swap(v[0], v[1]);
                } else if (v.size() > 2) {
                    std::sort(v.begin(), v.end());
                }
            }

            thread_local std::vector<std::vector<std::string>>* tls_snarl_buffer = nullptr;

            static std::mutex g_snarls_mtx;

            inline void flushThreadLocalSnarls(std::vector<std::vector<std::string>>& local) {
                auto &C = ctx();
                std::lock_guard<std::mutex> lk(g_snarls_mtx);
                for (auto &s : local) {
                    snarlsFound += s.size() * (s.size() - 1) / 2;
                    C.snarls.insert(s);


                    // std::sort(s.begin(), s.end());
                    // for(size_t i = 0; i < s.size(); i++) {
                    //     for(size_t j = i + 1; j < s.size(); j++) {
                    //         std::string source = s[i], sink = s[j];
                    //         // if(source == "_trash+" || sink == "_trash+") continue;
                    //         C.snarls.insert({source, sink});
                    //     }
                    // }
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


            std::atomic<uint64_t> g_cnt_cut{0}, g_cnt_S{0}, g_cnt_P{0}, g_cnt_RR{0}, g_cnt_E{0};

            inline void print_snarl_type_counters() {
                std::cout << "[SNARLS] by type: "
                        << "CUT=" << g_cnt_cut.load()
                        << " S="   << g_cnt_S.load()
                        << " P="   << g_cnt_P.load()
                        << " RR="  << g_cnt_RR.load()
                        << " E="   << g_cnt_E.load()
                        << std::endl;
            }
            static std::mutex g_ogdf_mutex;

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


        inline void addSnarlTagged(const char *tag, std::vector<std::string> s) {
            // Comptage par type (inchangé)
            if (tag) {
                if (std::strcmp(tag, "CUT") == 0)      g_cnt_cut++;
                else if (std::strcmp(tag, "S") == 0)   g_cnt_S++;
                else if (std::strcmp(tag, "P") == 0)   g_cnt_P++;
                else if (std::strcmp(tag, "RR") == 0)  g_cnt_RR++;
                else if (std::strcmp(tag, "E") == 0)   g_cnt_E++;
            }

            // Canonicalisation des paires (ordre stable)
            canonicalize_pair(s);

            // Si c'est un snarl de type S/P/RR, on enregistre la paire d'extrémités
            // (sans signe) dans le set TLS, uniquement dans le contexte SPQR (tls_snarl_buffer != nullptr).
            if (tls_snarl_buffer && tag &&
                (std::strcmp(tag, "S") == 0 ||
                std::strcmp(tag, "P") == 0 ||
                std::strcmp(tag, "RR") == 0) &&
                s.size() == 2) {

                // s[0], s[1] sont de la forme "nom+" ou "nom-"
                auto strip_sign = [](const std::string &x) {
                    if (!x.empty() && (x.back() == '+' || x.back() == '-')) {
                        return x.substr(0, x.size() - 1);
                    }
                    return x;
                };

                std::string aName = strip_sign(s[0]);
                std::string bName = strip_sign(s[1]);
                if (aName > bName) std::swap(aName, bName);

                tls_spqr_seen_endpoint_pairs.insert({aName, bName});
            }

            addSnarl(std::move(s));
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

            // NOUVEAU : degrés + / - dans CE bloc uniquement
            ogdf::NodeArray<int> blkDegPlus;
            ogdf::NodeArray<int> blkDegMinus;

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
        void getOutgoingEdgesInBlock(const CcData& cc,
                                    ogdf::node uG,     // node in Gcc
                                    ogdf::node vB,     // B-node in the BC-tree
                                    EdgePartType type,
                                    std::vector<ogdf::edge>& outEdges)
        {
            outEdges.clear();

            // repVertex may return nullptr if uG does not belong to this block
            ogdf::node uB = cc.bc->repVertex(uG, vB);
            if (!uB) {
                return;
            }

            // std::cout << "Getting outgoing edges in block for graph node " << uG << " in block node " << vB << " " << uB->adjEntries.size() << std::endl;

            for (auto adjE : uB->adjEntries) {
                ogdf::edge eAux = adjE->theEdge();           // edge in the auxiliary graph
                ogdf::edge eCc  = cc.bc->original(eAux);     // edge in cc.Gcc

                // Some auxiliary edges may not have an associated original edge
                if (!eCc) {
                    continue;
                }

                ogdf::edge eG = cc.edgeToOrig[eCc];          // edge in the original graph

                auto outType = getNodeEdgeType(cc.nodeToOrig[uG], eG);
                if (outType == type) {
                    outEdges.push_back(eCc);
                }
            }
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


            void printAllStates(const ogdf::NodeArray<NodeDPState> &node_dp,
                                const Graph &T)
            {
                std::cout << "Node dp states: " << std::endl;
                for (node v : T.nodes) {
                    std::cout << "Node " << v->index() << ", ";
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
                auto& C = ctx();

                EdgeDPState &state      = dp[curr_edge].down;
                EdgeDPState &back_state = dp[curr_edge].up;

                const StaticSPQRTree &spqr = *blk.spqr;

                // Determine parent / child in the SPQR tree without relying on source()/target()
                ogdf::node u = curr_edge->source();
                ogdf::node v = curr_edge->target();

                ogdf::node A = nullptr; // parent
                ogdf::node B = nullptr; // child

                if (blk.parent[u] == v) {
                    // v is the parent of u
                    A = v;
                    B = u;
                } else if (blk.parent[v] == u) {
                    // u is the parent of v
                    A = u;
                    B = v;
                } else {
                    // This should never happen if blk.parent correctly encodes the tree
                    OGDF_ASSERT(false);
                    return;
                }

                state.localPlusS  = 0;
                state.localPlusT  = 0;
                state.localMinusS = 0;
                state.localMinusT = 0;

                const Skeleton &skel      = spqr.skeleton(B);   // skeleton of the child
                const Graph    &skelGraph = skel.getGraph();

                auto mapSkeletonToGlobal = [&](ogdf::node vSkel) -> ogdf::node {
                    if (!vSkel) return nullptr;
                    ogdf::node vBlk  = skel.original(vSkel);
                    if (!vBlk) return nullptr;
                    ogdf::node vCc   = blk.toCc[vBlk];
                    if (!vCc) return nullptr;
                    return cc.nodeToOrig[vCc];
                };

                // 1) Find the virtual edge in skel(B) that corresponds to the tree edge (A,B)
                for (ogdf::edge e : skelGraph.edges) {
                    ogdf::node uSk = e->source();
                    ogdf::node vSk = e->target();

                    ogdf::node D = skel.twinTreeNode(e); // SPQR-tree node on the other side

                    if (D == A) {
                        ogdf::node vBlk = skel.original(vSk);
                        ogdf::node uBlk = skel.original(uSk);

                        state.s = back_state.s = vBlk;
                        state.t = back_state.t = uBlk;
                        break;
                    }
                }

                // 2) Accumulate contributions of real edges and SPQR children into state (on B's side)
                for (ogdf::edge e : skelGraph.edges) {
                    ogdf::node uSk = e->source();
                    ogdf::node vSk = e->target();

                    ogdf::node uBlk = skel.original(uSk);
                    ogdf::node vBlk = skel.original(vSk);

                    if (!skel.isVirtual(e)) {
                        // Real edge: locally count plus/minus for S and T
                        ogdf::edge eG = blk.edgeToOrig[skel.realEdge(e)];

                        ogdf::node uG = eG->source();
                        ogdf::node vG = eG->target();

                        if (uG == blk.nodeToOrig[state.s]) {
                            auto t = getNodeEdgeType(uG, eG);
                            if (t == EdgePartType::PLUS)      state.localPlusS++;
                            else if (t == EdgePartType::MINUS) state.localMinusS++;
                        }

                        if (vG == blk.nodeToOrig[state.s]) {
                            auto t = getNodeEdgeType(vG, eG);
                            if (t == EdgePartType::PLUS)      state.localPlusS++;
                            else if (t == EdgePartType::MINUS) state.localMinusS++;
                        }

                        if (uG == blk.nodeToOrig[state.t]) {
                            auto t = getNodeEdgeType(uG, eG);
                            if (t == EdgePartType::PLUS)      state.localPlusT++;
                            else if (t == EdgePartType::MINUS) state.localMinusT++;
                        }

                        if (vG == blk.nodeToOrig[state.t]) {
                            auto t = getNodeEdgeType(vG, eG);
                            if (t == EdgePartType::PLUS)      state.localPlusT++;
                            else if (t == EdgePartType::MINUS) state.localMinusT++;
                        }

                        continue;
                    }

                    // Virtual edge: corresponds to another SPQR node D
                    ogdf::node D = skel.twinTreeNode(e);

                    // Skip the edge that corresponds directly to the parent A (already processed)
                    if (D == A) {
                        continue;
                    }

                    // Retrieve the SPQR-tree edge corresponding to this virtual edge
                    ogdf::edge treeE = blk.skel2tree.at(e);
                    OGDF_ASSERT(treeE != nullptr);

                    // Use the child's "downward" DP.
                    // (The down/up orientation now follows blk.parent)
                    const EdgeDPState &child = dp[treeE].down;

                    // Add the child's subtree contributions to state
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
                            const CcData &/*cc*/,
                            BlockData &blk)
            {
                ogdf::node A = curr_node;
                const StaticSPQRTree &spqr = *blk.spqr;
                const Skeleton &skel      = spqr.skeleton(A);
                const Graph    &skelG     = skel.getGraph();

                Graph newGraph;

                NodeArray<node> skelToNew(skelG, nullptr);
                for (node v : skelG.nodes)
                    skelToNew[v] = newGraph.newNode();

                NodeArray<node> newToSkel(newGraph, nullptr);
                for (node v : skelG.nodes)
                    newToSkel[skelToNew[v]] = v;

                for (ogdf::node h : skelG.nodes) {
                    ogdf::node vB = skel.original(h);
                    blk.blkToSkel[vB] = h;
                }

                NodeArray<int> localPlusDeg (newGraph, 0);
                NodeArray<int> localMinusDeg(newGraph, 0);

                EdgeArray<bool>         isVirtual(newGraph, false);
                EdgeArray<EdgeDPState*> edgeToDp (newGraph, nullptr);
                EdgeArray<EdgeDPState*> edgeToDpR(newGraph, nullptr);
                EdgeArray<node>         edgeChild(newGraph, nullptr);

                std::vector<edge> virtualEdges;

                auto mapBlockToNew = [&](ogdf::node bV) -> ogdf::node {
                    ogdf::node sV = blk.blkToSkel[bV];
                    ogdf::node nV = skelToNew[sV];
                    return nV;
                };

                // construire newGraph
                for (edge e : skelG.edges) {
                    node u  = e->source();
                    node v  = e->target();

                    ogdf::node uBlk = skel.original(u);
                    ogdf::node vBlk = skel.original(v);

                    ogdf::node uG = blk.nodeToOrig[uBlk];
                    ogdf::node vG = blk.nodeToOrig[vBlk];

                    node nU = skelToNew[u];
                    node nV = skelToNew[v];

                    if (!skel.isVirtual(e)) {
                        ogdf::edge eG   = blk.edgeToOrig[skel.realEdge(e)];
                        uG = eG->source();
                        vG = eG->target();

                        auto newEdge = newGraph.newEdge(nU, nV);
                        isVirtual[newEdge] = false;

                        if (blk.nodeToOrig[skel.original(newToSkel[nU])] == uG) {
                            localPlusDeg [nU] += (getNodeEdgeType(uG, eG) == EdgePartType::PLUS);
                            localMinusDeg[nU] += (getNodeEdgeType(uG, eG) == EdgePartType::MINUS);
                            localPlusDeg [nV] += (getNodeEdgeType(vG, eG) == EdgePartType::PLUS);
                            localMinusDeg[nV] += (getNodeEdgeType(vG, eG) == EdgePartType::MINUS);
                        } else {
                            localPlusDeg [nU] += (getNodeEdgeType(vG, eG) == EdgePartType::PLUS);
                            localMinusDeg[nU] += (getNodeEdgeType(vG, eG) == EdgePartType::MINUS);
                            localPlusDeg [nV] += (getNodeEdgeType(uG, eG) == EdgePartType::PLUS);
                            localMinusDeg[nV] += (getNodeEdgeType(uG, eG) == EdgePartType::MINUS);
                        }
                        continue;
                    }

                    // virtual edge
                    auto B     = skel.twinTreeNode(e);
                    edge treeE = blk.skel2tree.at(e);
                    OGDF_ASSERT(treeE != nullptr);

                    EdgeDPState *child        = (B == blk.parent(A) ? &edge_dp[treeE].up   : &edge_dp[treeE].down);
                    EdgeDPState *edgeToUpdate = (B == blk.parent(A) ? &edge_dp[treeE].down : &edge_dp[treeE].up);

                    ogdf::node nS = mapBlockToNew(child->s);
                    ogdf::node nT = mapBlockToNew(child->t);

                    edge newEdge = newGraph.newEdge(nS, nT);
                    isVirtual[newEdge] = true;

                    virtualEdges.push_back(newEdge);
                    edgeToDp [newEdge] = edgeToUpdate;
                    edgeToDpR[newEdge] = child;
                    edgeChild[newEdge] = B;

                    if (nS == nU && nT == nV) {
                        localMinusDeg[nS] += child->localMinusT;
                        localPlusDeg [nS] += child->localPlusT;
                        localMinusDeg[nT] += child->localMinusS;
                        localPlusDeg [nT] += child->localPlusS;
                    } else {
                        localMinusDeg[nS] += child->localMinusS;
                        localPlusDeg [nS] += child->localPlusS;
                        localMinusDeg[nT] += child->localMinusT;
                        localPlusDeg [nT] += child->localPlusT;
                    }
                }

                // Update DP for incoming virtual edges
                for (edge e : virtualEdges) {
                    EdgeDPState *BA = edgeToDp [e];
                    EdgeDPState *AB = edgeToDpR[e];

                    BA->localPlusS  = localPlusDeg [mapBlockToNew(BA->s)] - AB->localPlusS;
                    BA->localPlusT  = localPlusDeg [mapBlockToNew(BA->t)] - AB->localPlusT;
                    BA->localMinusS = localMinusDeg[mapBlockToNew(BA->s)] - AB->localMinusS;
                    BA->localMinusT = localMinusDeg[mapBlockToNew(BA->t)] - AB->localMinusT;
                }
            }

    
            
            void solveS(ogdf::node sNode,
                        NodeArray<NodeDPState> &node_dp,
                        ogdf::EdgeArray<EdgeDP> &dp,
                        BlockData &blk,
                        const CcData &cc)
            {
                const Skeleton &skel    = blk.spqr->skeleton(sNode);
                const Graph    &skelG   = skel.getGraph();
                const Graph    &T       = blk.spqr->tree();

                std::vector<ogdf::node> nodesInOrderGcc;
                std::vector<ogdf::node> nodesInOrderSkel;

                ogdf::EdgeArray<EdgeDPState*> skelToState(T);

                std::vector<ogdf::edge> adjEdgesG;
                std::vector<adjEntry>   adjEntriesSkel;

                // Associate each virtual edge with its DP state
                for (edge e : skelG.edges) {
                    if (!skel.isVirtual(e)) continue;
                    auto B    = skel.twinTreeNode(e);
                    edge treeE = blk.skel2tree.at(e);

                    EdgeDPState *child = (B == blk.parent(sNode) ? &dp[treeE].up : &dp[treeE].down);
                    skelToState[treeE] = child;
                }

                // DFS to number the vertices in order
                {
                    std::function<void(ogdf::node, ogdf::node)> dfs =
                        [&](ogdf::node u, ogdf::node prev) {
                            nodesInOrderGcc .push_back(blk.toCc[skel.original(u)]);
                            nodesInOrderSkel.push_back(u);

                            for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {

                                if (adj->twinNode() == prev) continue;

                                if (adj->twinNode() == skelG.firstNode() && u != skelG.firstNode()) {
                                    if (skel.realEdge(adj->theEdge())) {
                                        adjEdgesG.push_back(blk.edgeToOrig[skel.realEdge(adj->theEdge())]);
                                    } else {
                                        adjEdgesG.push_back(nullptr);
                                    }
                                    adjEntriesSkel.push_back(adj);
                                }

                                if (adj->twinNode() == skelG.firstNode() || adj->twinNode() == prev)
                                    continue;

                                if (skel.realEdge(adj->theEdge())) {
                                    adjEdgesG.push_back(blk.edgeToOrig[skel.realEdge(adj->theEdge())]);
                                } else {
                                    adjEdgesG.push_back(nullptr);
                                }
                                adjEntriesSkel.push_back(adj);

                                dfs(adj->twinNode(), u);
                            }
                        };

                    dfs(skelG.firstNode(), skelG.firstNode()->firstAdj()->twinNode());
                }

                std::vector<bool> cuts(nodesInOrderGcc.size(), false);
                std::vector<std::string> res;

                for (size_t i = 0; i < nodesInOrderGcc.size(); ++i) {
                    auto uGcc = nodesInOrderGcc[i];

                    std::vector<edge>    adjEdgesSkelLoc = {
                        adjEntriesSkel[(i + adjEntriesSkel.size() - 1) % adjEntriesSkel.size()]->theEdge(),
                        adjEntriesSkel[i]->theEdge()
                    };
                    std::vector<ogdf::edge> adjEdgesGLoc = {
                        adjEdgesG[(i + adjEdgesG.size() - 1) % adjEdgesG.size()],
                        adjEdgesG[i]
                    };

                    bool nodeIsCut = ((cc.isCutNode[uGcc] && cc.badCutCount[uGcc] == 1) ||
                                      (!cc.isCutNode[uGcc]));

                    EdgePartType t0 = EdgePartType::NONE;
                    EdgePartType t1 = EdgePartType::NONE;

                    // edge 0
                    if (!skel.isVirtual(adjEdgesSkelLoc[0])) {
                        t0 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesGLoc[0]);
                    } else {
                        edge treeE0        = blk.skel2tree.at(adjEdgesSkelLoc[0]);
                        EdgeDPState *state0 = skelToState[treeE0];
                        if (blk.toCc[state0->s] == uGcc) {
                            if (state0->localMinusS == 0 && state0->localPlusS > 0)
                                t0 = EdgePartType::PLUS;
                            else if (state0->localMinusS > 0 && state0->localPlusS == 0)
                                t0 = EdgePartType::MINUS;
                        } else {
                            if (state0->localMinusT == 0 && state0->localPlusT > 0)
                                t0 = EdgePartType::PLUS;
                            else if (state0->localMinusT > 0 && state0->localPlusT == 0)
                                t0 = EdgePartType::MINUS;
                        }
                    }

                    // edge 1
                    if (!skel.isVirtual(adjEdgesSkelLoc[1])) {
                        t1 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesGLoc[1]);
                    } else {
                        edge treeE1        = blk.skel2tree.at(adjEdgesSkelLoc[1]);
                        EdgeDPState *state1 = skelToState[treeE1];
                        if (blk.toCc[state1->s] == uGcc) {
                            if (state1->localMinusS == 0 && state1->localPlusS > 0)
                                t1 = EdgePartType::PLUS;
                            else if (state1->localMinusS > 0 && state1->localPlusS == 0)
                                t1 = EdgePartType::MINUS;
                        } else {
                            if (state1->localMinusT == 0 && state1->localPlusT > 0)
                                t1 = EdgePartType::PLUS;
                            else if (state1->localMinusT > 0 && state1->localPlusT == 0)
                                t1 = EdgePartType::MINUS;
                        }
                    }

                    nodeIsCut &= (t0 != EdgePartType::NONE &&
                                  t1 != EdgePartType::NONE &&
                                  t0 != t1);

                    if (nodeIsCut) {
                        if (node_dp[sNode].GccCuts_last3.size() < 3)
                            node_dp[sNode].GccCuts_last3.push_back(uGcc);

                        // left side
                        if (!skel.isVirtual(adjEdgesSkelLoc[0])) {
                            EdgePartType tt0 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesGLoc[0]);
                            res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] +
                                          (tt0 == EdgePartType::PLUS ? "+" : "-"));
                        } else {
                            edge treeE0        = blk.skel2tree.at(adjEdgesSkelLoc[0]);
                            EdgeDPState *state0 = skelToState[treeE0];
                            if (uGcc == blk.toCc[state0->s]) {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] +
                                              (state0->localPlusS > 0 ? "+" : "-"));
                            } else {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] +
                                              (state0->localPlusT > 0 ? "+" : "-"));
                            }
                        }

                        // right side
                        if (!skel.isVirtual(adjEdgesSkelLoc[1])) {
                            EdgePartType tt1 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesGLoc[1]);
                            res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] +
                                          (tt1 == EdgePartType::PLUS ? "+" : "-"));
                        } else {
                            edge treeE1        = blk.skel2tree.at(adjEdgesSkelLoc[1]);
                            EdgeDPState *state1 = skelToState[treeE1];
                            if (uGcc == blk.toCc[state1->s]) {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] +
                                              (state1->localPlusS > 0 ? "+" : "-"));
                            } else {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] +
                                              (state1->localPlusT > 0 ? "+" : "-"));
                            }
                        }
                    }
                }

                OGDF_ASSERT(res.size() % 2 == 0);
                if (res.size() > 2) {
                    for (size_t i = 1; i < res.size(); i += 2) {
                        std::vector<std::string> v = { res[i], res[(i + 1) % res.size()] };
                        addSnarlTagged("S", std::move(v));
                    }
                }
            }


            void solveP(ogdf::node pNode,
                        ogdf::NodeArray<SPQRsolve::NodeDPState> &node_dp,
                        ogdf::EdgeArray<EdgeDP> &edge_dp,
                        BlockData& blk,
                        const CcData& cc)
            {
                PROFILE_FUNCTION();
                auto &C = ctx();

                const Skeleton& skel = blk.spqr->skeleton(pNode);
                const Graph& skelGraph = skel.getGraph();
                const Graph& T = blk.spqr->tree();

                VLOG << "[DEBUG][solveP] P-node idx=" << pNode->index()
                    << " skeleton |V|=" << skelGraph.numberOfNodes()
                    << " |E|=" << skelGraph.numberOfEdges() << "\n";

                // A P-node has exactly two vertices in its skeleton.
                ogdf::node pole0Skel = nullptr, pole1Skel = nullptr;
                {
                    auto it = skelGraph.nodes.begin();
                    if (it != skelGraph.nodes.end()) pole0Skel = *it++;
                    if (it != skelGraph.nodes.end()) pole1Skel = *it;
                }
                if (!pole0Skel || !pole1Skel) {
                    VLOG << "[DEBUG][solveP]  P-node has <2 skeleton vertices, skip\n";
                    return;
                }

                ogdf::node pole0Blk = skel.original(pole0Skel);
                ogdf::node pole1Blk = skel.original(pole1Skel);
                if (!pole0Blk || !pole1Blk) {
                    VLOG << "[DEBUG][solveP]  skel.original() returned nullptr pole, skip\n";
                    return;
                }

                ogdf::node pole0Gcc = blk.toCc[pole0Blk];
                ogdf::node pole1Gcc = blk.toCc[pole1Blk];

                VLOG << "[DEBUG][solveP]  poles: "
                    << C.node2name[cc.nodeToOrig[pole0Gcc]] << " (Gcc idx=" << pole0Gcc->index() << "), "
                    << C.node2name[cc.nodeToOrig[pole1Gcc]] << " (Gcc idx=" << pole1Gcc->index() << ")\n";

                // Test for dangling blocks (dangling blocks outside the current block)
                auto hasDanglingOutside = [&](ogdf::node vGcc) {
                    if (!cc.isCutNode[vGcc]) return false;
                    if (cc.badCutCount[vGcc] >= 2) return true;
                    if (cc.badCutCount[vGcc] == 1 && cc.lastBad[vGcc] != blk.bNode) return true;
                    return false;
                };
                if (hasDanglingOutside(pole0Gcc) || hasDanglingOutside(pole1Gcc)) {
                    VLOG << "[DEBUG][solveP]  bad dangling at pole, skip P-node\n";
                    return;
                }

                // Order (circular) of edges incident to pole 0 in the skeleton
                std::vector<ogdf::adjEntry> edgeOrdering;
                for (ogdf::adjEntry adj = pole0Skel->firstAdj(); adj; adj = adj->succ()) {
                    edgeOrdering.push_back(adj);
                }

                // 1) First pass: ensure that no SPQR child gives + and - simultaneously to the same pole                
                for (ogdf::adjEntry adj : edgeOrdering) {
                    ogdf::edge eSkel = adj->theEdge();
                    if (!skel.isVirtual(eSkel)) continue;

                    auto itMap = blk.skel2tree.find(eSkel);
                    if (itMap == blk.skel2tree.end()) continue;
                    ogdf::edge treeE = itMap->second;
                    ogdf::node B = (treeE->source() == pNode ? treeE->target() : treeE->source());

                    EdgeDP &dpVal = edge_dp[treeE];
                    EdgeDPState &state = (blk.parent[pNode] == B ? dpVal.up : dpVal.down);

                    // pôle 0
                    if (state.s == pole0Blk) {
                        if (state.localPlusS > 0 && state.localMinusS > 0) {
                            VLOG << "[DEBUG][solveP]  child gives both signs at pole0, abort P-node\n";
                            return;
                        }
                    } else {
                        if (state.localPlusT > 0 && state.localMinusT > 0) {
                            VLOG << "[DEBUG][solveP]  child gives both signs at pole0 (T side), abort P-node\n";
                            return;
                        }
                    }

                    // pôle 1
                    if (state.s == pole1Blk) {
                        if (state.localPlusS > 0 && state.localMinusS > 0) {
                            VLOG << "[DEBUG][solveP]  child gives both signs at pole1, abort P-node\n";
                            return;
                        }
                    } else {
                        if (state.localPlusT > 0 && state.localMinusT > 0) {
                            VLOG << "[DEBUG][solveP]  child gives both signs at pole1 (T side), abort P-node\n";
                            return;
                        }
                    }
                }

                // 2) For each combination (left,right) ∈ {+,-}², construct the sets E^{left}_{pole0}, E^{right}_{pole1}                
                for (auto left : {EdgePartType::PLUS, EdgePartType::MINUS}) {
                    for (auto right : {EdgePartType::PLUS, EdgePartType::MINUS}) {

                        std::vector<ogdf::edge> leftPart, rightPart;

                        for (ogdf::adjEntry adj : edgeOrdering) {
                            ogdf::edge eSkel = adj->theEdge();

                            EdgePartType lSign = EdgePartType::NONE;
                            EdgePartType rSign = EdgePartType::NONE;

                            if (!skel.isVirtual(eSkel)) {
                                // Actual edge: the sign is read directly from the poles in the original graph.
                                
                                ogdf::edge eB = skel.realEdge(eSkel);
                                ogdf::edge eG = blk.edgeToOrig[eB];

                                ogdf::node pole0G = cc.nodeToOrig[pole0Gcc];
                                ogdf::node pole1G = cc.nodeToOrig[pole1Gcc];

                                lSign = getNodeEdgeType(pole0G, eG);
                                rSign = getNodeEdgeType(pole1G, eG);
                            } else {
                                // Virtual edge: SPQR child, we use the corresponding DP state.
                                auto itMap = blk.skel2tree.find(eSkel);
                                if (itMap == blk.skel2tree.end()) continue;
                                ogdf::edge treeE = itMap->second;
                                ogdf::node B = (treeE->source() == pNode ? treeE->target() : treeE->source());

                                EdgeDP &dpVal = edge_dp[treeE];
                                EdgeDPState &st = (blk.parent[pNode] == B ? dpVal.up : dpVal.down);

                                // Sign at pole 0
                                if (st.s == pole0Blk) {
                                    bool hasPlus  = (st.localPlusS  > 0);
                                    bool hasMinus = (st.localMinusS > 0);
                                    if (hasPlus && !hasMinus)      lSign = EdgePartType::PLUS;
                                    else if (!hasPlus && hasMinus) lSign = EdgePartType::MINUS;
                                    else                           lSign = EdgePartType::NONE;
                                } else {
                                    bool hasPlus  = (st.localPlusT  > 0);
                                    bool hasMinus = (st.localMinusT > 0);
                                    if (hasPlus && !hasMinus)      lSign = EdgePartType::PLUS;
                                    else if (!hasPlus && hasMinus) lSign = EdgePartType::MINUS;
                                    else                           lSign = EdgePartType::NONE;
                                }

                                // Signe au pôle 1
                                if (st.s == pole1Blk) {
                                    bool hasPlus  = (st.localPlusS  > 0);
                                    bool hasMinus = (st.localMinusS > 0);
                                    if (hasPlus && !hasMinus)      rSign = EdgePartType::PLUS;
                                    else if (!hasPlus && hasMinus) rSign = EdgePartType::MINUS;
                                    else                           rSign = EdgePartType::NONE;
                                } else {
                                    bool hasPlus  = (st.localPlusT  > 0);
                                    bool hasMinus = (st.localMinusT > 0);
                                    if (hasPlus && !hasMinus)      rSign = EdgePartType::PLUS;
                                    else if (!hasPlus && hasMinus) rSign = EdgePartType::MINUS;
                                    else                           rSign = EdgePartType::NONE;
                                }
                            }
                            
                            // The edge is only added to E^{left}_pole0 / E^{right}_pole1
                            // if the sign matches. NONE edges are not added to any set.
                            if (lSign == left)  leftPart.push_back(eSkel);
                            if (rSign == right) rightPart.push_back(eSkel);
                        }

                        // Separability conditions (Proposition P-node of the paper):
                        //  - E^{left}_{pole0} not empty
                        //  - E^{left}_{pole0} = E^{right}_{pole1}
                        if (leftPart.empty() || leftPart != rightPart) continue;

                        // Minimality: according to the paper, if |E^{left}_u| > 1, minimality
                        // follows from the structure P and it is not necessary to filter by S-nodes.
                        // Filtering via GccCuts_last3 should only apply if leftPart.size() == 1.
                        bool ok = true;
                        if (leftPart.size() == 1) {
                            ogdf::edge eSkel = leftPart[0];
                            if (skel.isVirtual(eSkel)) {
                                ogdf::node B = skel.twinTreeNode(eSkel);
                                if (blk.spqr->typeOf(B) == StaticSPQRTree::NodeType::SNode) {
                                    for (ogdf::node gccCut : node_dp[B].GccCuts_last3) {
                                        if (gccCut != pole0Gcc && gccCut != pole1Gcc) {
                                            ok = false;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        if (!ok) continue;

                        // We have a minimal snarl P between the two poles, with the signs (left, right).
                        std::string sName = C.node2name[cc.nodeToOrig[pole0Gcc]] +
                                            (left  == EdgePartType::PLUS ? "+" : "-");
                        std::string tName = C.node2name[cc.nodeToOrig[pole1Gcc]] +
                                            (right == EdgePartType::PLUS ? "+" : "-");

                        std::vector<std::string> v = { sName, tName };
                        VLOG << "[DEBUG][solveP]  addSnarlTagged P: "
                            << v[0] << " " << v[1] << "\n";
                        addSnarlTagged("P", std::move(v));
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

                // Incomplete states => no attempt made
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

                // Dangling test relative to the current block blk.bNode
                auto hasDanglingOutside = [&](ogdf::node vGcc) {
                    if (!cc.isCutNode[vGcc]) return false;
                    if (cc.badCutCount[vGcc] >= 2) return true;
                    if (cc.badCutCount[vGcc] == 1 && cc.lastBad[vGcc] != blk.bNode) return true;
                    return false;
                };

                if (hasDanglingOutside(pole0Gcc) || hasDanglingOutside(pole1Gcc)) {
                    return;
                }

                // States where a pole sees + and - at the same time are rejected                
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

                // We want a change of type between up and down for each pole.
                if (pole0DownType == pole0UpType) return;
                if (pole1DownType == pole1UpType) return;

                // Snarl for the “down” state
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

                // Snarl for the "up" state
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


            void solveNodes(NodeArray<SPQRsolve::NodeDPState> &node_dp,
                            ogdf::EdgeArray<EdgeDP> &edge_dp,
                            BlockData& blk,
                            const CcData& cc)
            {
                PROFILE_FUNCTION();
                if (!blk.spqr) return;

                const Graph &T = blk.spqr->tree();

                VLOG << "[DEBUG][solveNodes] start, |T.nodes|=" << T.numberOfNodes()
                    << " |T.edges|=" << T.numberOfEdges() << "\n";

                // 1) S-nodes
                for (node tNode : T.nodes) {
                    auto tType = blk.spqr->typeOf(tNode);
                    if (tType == StaticSPQRTree::NodeType::SNode) {
                        VLOG << "[DEBUG][solveNodes] S-node idx=" << tNode->index()
                            << " -> solveS()\n";
                        solveS(tNode, node_dp, edge_dp, blk, cc);
                    }
                }

                // 2) P-nodes
                for (node tNode : T.nodes) {
                    auto tType = blk.spqr->typeOf(tNode);
                    if (tType == StaticSPQRTree::NodeType::PNode) {
                        VLOG << "[DEBUG][solveNodes] P-node idx=" << tNode->index()
                            << " -> solveP()\n";
                        solveP(tNode, node_dp, edge_dp, blk, cc);
                    }
                }

                // 3) R-R edges
                for (edge e : T.edges) {
                    auto srcT = blk.spqr->typeOf(e->source());
                    auto dstT = blk.spqr->typeOf(e->target());
                    if (srcT == SPQRTree::NodeType::RNode &&
                        dstT == SPQRTree::NodeType::RNode) {
                        VLOG << "[DEBUG][solveNodes] R-R edge idx=" << e->index()
                            << " -> solveRR()\n";
                        solveRR(e, node_dp, edge_dp, blk, cc);
                    }
                }

                VLOG << "[DEBUG][solveNodes] end\n";
            }
            void solveSPQR(BlockData& blk, const CcData& cc) {
                MARK_SCOPE_MEM("sn/solveSPQR");
                PROFILE_FUNCTION();

                if (!blk.spqr) return;
                if (!blk.Gblk || blk.Gblk->numberOfNodes() < 3) return;

                auto &C = ctx();
                const ogdf::Graph &T = blk.spqr->tree();

                // DP on the edges of the SPQR tree
                ogdf::EdgeArray<EdgeDP>      edge_dp(T);
                ogdf::NodeArray<NodeDPState> node_dp(T);

                std::vector<ogdf::node> nodeOrder;
                std::vector<ogdf::edge> edgeOrder;

                dfsSPQR_order(*blk.spqr, edgeOrder, nodeOrder);

                blk.blkToSkel.init(*blk.Gblk, nullptr);

                // Down phase on the edges
                for (ogdf::edge e : edgeOrder) {
                    processEdge(e, edge_dp, cc, blk);
                }

                // Local “node” phase on each node of the tree
                for (ogdf::node v : nodeOrder) {
                    processNode(v, edge_dp, cc, blk);
                }

                // Resolution of S/P/RR snarls
                solveNodes(node_dp, edge_dp, blk, cc);

                // Pre-calculation: for each vertex of the block, list of S-nodes
                // in which it appears (to filter case B of Prop. 3.16).
                ogdf::NodeArray<std::vector<ogdf::node>> vertexInSnodes(*blk.Gblk);
                for (ogdf::node vB : blk.Gblk->nodes) {
                    vertexInSnodes[vB].clear();
                }

                for (ogdf::node mu : T.nodes) {
                    if (blk.spqr->typeOf(mu) != ogdf::StaticSPQRTree::NodeType::SNode)
                        continue;
                    const ogdf::Skeleton &skel = blk.spqr->skeleton(mu);
                    const ogdf::Graph    &skelG = skel.getGraph();
                    for (ogdf::node vSk : skelG.nodes) {
                        ogdf::node vB = skel.original(vSk);
                        vertexInSnodes[vB].push_back(mu);
                    }
                }

                auto shareSnode = [&](ogdf::node aB, ogdf::node bB) -> bool {
                    const auto &La = vertexInSnodes[aB];
                    const auto &Lb = vertexInSnodes[bB];
                    if (La.empty() || Lb.empty()) return false;
                    // Naive intersection, but the lists are very small in practice [TO OPTIMIZE ?]
                    for (ogdf::node x : La) {
                        for (ogdf::node y : Lb) {
                            if (x == y) return true;
                        }
                    }
                    return false;
                };

                // Test “dangling” relative to the current block
                auto hasDanglingOutside = [&](ogdf::node vGcc) {
                    if (!cc.isCutNode[vGcc]) return false;
                    if (cc.badCutCount[vGcc] >= 2) return true;
                    if (cc.badCutCount[vGcc] == 1 && cc.lastBad[vGcc] != blk.bNode) return true;
                    return false;
                };

                // ----------------------
                // Case E: single-edge snarls
                // ----------------------
                std::vector<ogdf::edge> edgesSorted;
                edgesSorted.reserve(blk.Gblk->numberOfEdges());
                for (ogdf::edge eB : blk.Gblk->edges) edgesSorted.push_back(eB);

                std::sort(edgesSorted.begin(), edgesSorted.end(),
                        [](ogdf::edge a, ogdf::edge b){ return a->index() < b->index(); });

                for (ogdf::edge eB : edgesSorted) {
                    ogdf::edge eG = blk.edgeToOrig[eB];

                    ogdf::node uB = eB->source();
                    ogdf::node vB = eB->target();

                    ogdf::node uGcc = blk.toCc[uB];
                    ogdf::node vGcc = blk.toCc[vB];

                    ogdf::node uG = cc.nodeToOrig[uGcc];
                    ogdf::node vG = cc.nodeToOrig[vGcc];

                    // We ignore edges incident to _trash
                    if (C.node2name[uG] == "_trash" || C.node2name[vG] == "_trash")
                        continue;

                    // We want two non-tips in this block
                    if (cc.isTip[uGcc] || cc.isTip[vGcc])
                        continue;

                    // No dangling blocks outside this block
                    if (hasDanglingOutside(uGcc) || hasDanglingOutside(vGcc))
                        continue;

                    // Sign of this edge in the original graph
                    EdgePartType edgeSignU = getNodeEdgeType(uG, eG); // sign to u
                    EdgePartType edgeSignV = getNodeEdgeType(vG, eG); // sign to v

                    auto flipSign = [](EdgePartType t) {
                        return (t == EdgePartType::PLUS ? EdgePartType::MINUS : EdgePartType::PLUS);
                    };

                    // Checks the condition of Prop. 3.16 for ONE vertex:                    
                    // - if eSign == sign  → case A: e = {u d, ...}, the other incidences                    
                    //  must be of opposite sign to sign.                    
                    // - if eSign != sign  → case B: e = {u hat(d), ...}, the other incidences                    
                    // must have the same sign as sign.
                    auto check_one_vertex = [&](ogdf::node vB,
                                                EdgePartType sign,      // snarl sign at this node
                                                EdgePartType eSign) {   // sign of the eG ridge at this node
                        int totPlus  = blk.blkDegPlus[vB];
                        int totMinus = blk.blkDegMinus[vB];

                        if (sign == EdgePartType::PLUS) {
                            if (eSign == EdgePartType::PLUS) {
                                // Case A: e = {u+, ...}, others must be '-'
                                int othersPlus  = totPlus - 1;
                                int othersMinus = totMinus;
                                return (othersPlus == 0 && othersMinus > 0);
                            } else {
                                // Case B: e = {u-, ...}, others must be '+'
                                int othersPlus  = totPlus;
                                int othersMinus = totMinus - 1;
                                return (othersMinus == 0 && othersPlus > 0);
                            }
                        } else { // sign == MINUS
                            if (eSign == EdgePartType::MINUS) {
                                // Case A: e = {u-, ...}, others must be '+'
                                int othersMinus = totMinus - 1;
                                int othersPlus  = totPlus;
                                return (othersMinus == 0 && othersPlus > 0);
                            } else {
                                // Case B: e = {u+, ...}, others must be '-'
                                int othersMinus = totMinus;
                                int othersPlus  = totPlus - 1;
                                return (othersPlus == 0 && othersMinus > 0);
                            }
                        }
                    };

                    auto testCandidate = [&](EdgePartType signU,
                                            EdgePartType signV,
                                            bool isFlipCase) {
                        if (isFlipCase && shareSnode(uB, vB))
                            return;

                        if (!check_one_vertex(uB, signU, edgeSignU)) return;
                        if (!check_one_vertex(vB, signV, edgeSignV)) return;

                        std::string s =
                            C.node2name[uG] + (signU == EdgePartType::PLUS ? "+" : "-");
                        std::string t =
                            C.node2name[vG] + (signV == EdgePartType::PLUS ? "+" : "-");

                        addSnarlTagged("E", {s, t});
                    };

                    testCandidate(edgeSignU, edgeSignV, false);
                    testCandidate(flipSign(edgeSignU), flipSign(edgeSignV), true);
                }
            }



        }

        void findTips(CcData& cc) {
            MARK_SCOPE_MEM("sn/findTips");
            PROFILE_FUNCTION();
            size_t localIsolated = 0;
            auto &C = ctx();

            VLOG << "[DEBUG][findTips] -----\n";

            for (node v : cc.Gcc->nodes) {
                int plusCnt = 0, minusCnt = 0;
                node vG = cc.nodeToOrig[v];
                const std::string &name = C.node2name[vG];

                for (auto adjE : v->adjEntries) {
                    ogdf::edge e = cc.edgeToOrig[adjE->theEdge()];
                    EdgePartType eType = getNodeEdgeType(vG, e);
                    if (eType == EdgePartType::PLUS) plusCnt++;
                    else if (eType == EdgePartType::MINUS) minusCnt++;
                }

                if (plusCnt + minusCnt == 0) {
                    localIsolated++;
                }

                if (plusCnt == 0 || minusCnt == 0) {
                    cc.isTip[v] = true;
                } else {
                    cc.isTip[v] = false;
                }

                VLOG << "[DEBUG][findTips] node " << name
                        << " (Gcc idx=" << v->index() << ")"
                        << " plusCnt=" << plusCnt
                        << " minusCnt=" << minusCnt
                        << " isTip=" << (cc.isTip[v] ? "true" : "false")
                        << "\n";
            }

            {
                std::lock_guard<std::mutex> lk(g_snarls_mtx);
                isolatedNodesCnt += localIsolated;
            }

            VLOG << "[DEBUG][findTips] localIsolated=" << localIsolated
                    << " totalIsolated=" << isolatedNodesCnt << "\n";
        }

        void processCutNodes(CcData& cc) {
            MARK_SCOPE_MEM("sn/processCutNodes");
            PROFILE_FUNCTION();
            auto &C = ctx();

            VLOG << "[DEBUG][processCutNodes] -----\n";

            for (node v : cc.Gcc->nodes) {
                node vG = cc.nodeToOrig[v];
                const std::string &name = C.node2name[vG];

                if (cc.bc->typeOfGNode(v) == BCTree::GNodeType::CutVertex) {
                    cc.isCutNode[v] = true;

                    bool isGood = true;
                    ogdf::node vT = cc.bc->bcproper(v);

                    for (auto adjV : vT->adjEntries) {
                        node uT = adjV->twinNode();
                        std::vector<ogdf::edge> outPlus, outMinus;
                        getOutgoingEdgesInBlock(cc, v, uT, EdgePartType::PLUS, outPlus);
                        getOutgoingEdgesInBlock(cc, v, uT, EdgePartType::MINUS, outMinus);

                        if (outPlus.size() > 0 && outMinus.size() > 0) {
                            isGood = false;
                            cc.lastBad[v] = uT;
                            cc.badCutCount[v]++;
                        }
                    }
                    cc.isGoodCutNode[v] = isGood;
                }

                VLOG << "[DEBUG][processCutNodes] node " << name
                        << " (Gcc idx=" << v->index() << ")"
                        << " isCutNode=" << (cc.isCutNode[v] ? "true" : "false")
                        << " badCutCount=" << cc.badCutCount[v]
                        << " isGoodCutNode=" << (cc.isGoodCutNode[v] ? "true" : "false");
                if (cc.lastBad[v] != nullptr) {
                    VLOG << " lastBad(B-node idx)=" << cc.lastBad[v]->index();
                }
                VLOG<< "\n";
            }
        }

        void findCutSnarl(CcData &cc)
        {
            MARK_SCOPE_MEM("sn/findCutSnarl");

            // visited[v].first = visited with a path ending in MINUS
            // visited[v].second = visited with a path ending in PLUS
            ogdf::NodeArray<std::pair<bool, bool>> visited(
                *cc.Gcc, {false, false}); // (minusVisited, plusVisited)

            for (ogdf::node start : cc.Gcc->nodes) {
                for (auto t : {EdgePartType::PLUS, EdgePartType::MINUS}) {

                    if (t == EdgePartType::PLUS && visited[start].second) continue;
                    if (t == EdgePartType::MINUS && visited[start].first)  continue;

                    std::vector<std::string> goodNodes;

                    struct Frame {
                        ogdf::node      v;
                        EdgePartType    edgeType;
                    };
                    std::stack<Frame> st;
                    st.push({start, t});

                    while (!st.empty()) {
                        auto [v, edgeType] = st.top();
                        st.pop();

                        bool &minusVisited = visited[v].first;
                        bool &plusVisited  = visited[v].second;
                        bool isGoodOrTip   = (cc.isGoodCutNode[v] || cc.isTip[v]);

                        if (!isGoodOrTip) {
                            if (minusVisited && plusVisited) {
                                continue; 
                            }
                        } else {
                            if (edgeType == EdgePartType::MINUS && minusVisited) continue;
                            if (edgeType == EdgePartType::PLUS  && plusVisited)  continue;
                        }

                        if (isGoodOrTip &&
                            ctx().node2name[cc.nodeToOrig[v]] != "_trash") {

                            goodNodes.push_back(
                                ctx().node2name[cc.nodeToOrig[v]] +
                                (edgeType == EdgePartType::PLUS ? "+" : "-"));
                        }

                        if (!isGoodOrTip) {
                            minusVisited = true;
                            plusVisited  = true;
                        } else {
                            if (edgeType == EdgePartType::MINUS)
                                minusVisited = true;
                            else
                                plusVisited  = true;
                        }


                        std::vector<ogdf::AdjElement*> sameOutEdges, otherOutEdges;
                        getAllOutgoingEdgesOfType(
                            cc, v,
                            (edgeType == EdgePartType::PLUS ? EdgePartType::PLUS
                                                            : EdgePartType::MINUS),
                            sameOutEdges);
                        getAllOutgoingEdgesOfType(
                            cc, v,
                            (edgeType == EdgePartType::PLUS ? EdgePartType::MINUS
                                                            : EdgePartType::PLUS),
                            otherOutEdges);

                        bool canGoOther = !cc.isGoodCutNode[v] && !cc.isTip[v];


                        for (auto &adjE : sameOutEdges) {
                            ogdf::node otherNode = adjE->twinNode();
                            ogdf::edge eCc       = adjE->theEdge();
                            ogdf::edge eOrig     = cc.edgeToOrig[eCc];

                            EdgePartType inType =
                                getNodeEdgeType(cc.nodeToOrig[otherNode], eOrig);

                            if ((inType == EdgePartType::PLUS && !visited[otherNode].second) ||
                                (inType == EdgePartType::MINUS && !visited[otherNode].first)) {
                                st.push({otherNode, inType});
                            }
                        }

                        if (canGoOther) {
                            for (auto &adjE : otherOutEdges) {
                                ogdf::node otherNode = adjE->twinNode();
                                ogdf::edge eCc       = adjE->theEdge();
                                ogdf::edge eOrig     = cc.edgeToOrig[eCc];

                                EdgePartType inType =
                                    getNodeEdgeType(cc.nodeToOrig[otherNode], eOrig);

                                if ((inType == EdgePartType::PLUS && !visited[otherNode].second) ||
                                    (inType == EdgePartType::MINUS && !visited[otherNode].first)) {
                                    st.push({otherNode, inType});
                                }
                            }
                        }
                    }

                    if (goodNodes.size() >= 2) {
                        addSnarlTagged("CUT", std::move(goodNodes));
                    }
                }
            }
        }


        void buildBlockData(BlockData& blk, CcData& cc) {
            MARK_SCOPE_MEM("sn/blockData/build");
            PROFILE_FUNCTION();

            auto &C = ctx();

            VLOG << "[DEBUG][buildBlockData] --------\n";
            VLOG << "[DEBUG][buildBlockData] B-node index=" << blk.bNode->index() << "\n";

            blk.Gblk = std::make_unique<ogdf::Graph>();

            blk.nodeToOrig.init(*blk.Gblk, nullptr);
            blk.edgeToOrig.init(*blk.Gblk, nullptr);
            blk.toCc.init(*blk.Gblk, nullptr);

            blk.blkDegPlus.init(*blk.Gblk, 0);
            blk.blkDegMinus.init(*blk.Gblk, 0);

            std::vector<ogdf::node> verts_vec;
            {
                size_t approx = 0;
                for (ogdf::edge hE : cc.bc->hEdges(blk.bNode)) {
                    (void)hE; ++approx;
                }
                verts_vec.reserve(2 * approx);

                VLOG << "[DEBUG][buildBlockData]  hEdges for this B-node:\n";
                for (ogdf::edge hE : cc.bc->hEdges(blk.bNode)) {
                    ogdf::edge eCc = cc.bc->original(hE);
                    ogdf::node uC = eCc->source();
                    ogdf::node vC = eCc->target();
                    ogdf::node uG = cc.nodeToOrig[uC];
                    ogdf::node vG = cc.nodeToOrig[vC];
                    const std::string &uName = C.node2name[uG];
                    const std::string &vName = C.node2name[vG];

                    VLOG << "    Gcc edge: " << uName << " -- " << vName
                        << " (Gcc idx " << uC->index() << " -- " << vC->index() << ")\n";

                    verts_vec.push_back(uC);
                    verts_vec.push_back(vC);
                }

                std::sort(verts_vec.begin(), verts_vec.end(),
                        [](ogdf::node a, ogdf::node b){ return a->index() < b->index(); });
                verts_vec.erase(std::unique(verts_vec.begin(), verts_vec.end()), verts_vec.end());
            }

            std::unordered_map<ogdf::node, ogdf::node> cc_to_blk;
            cc_to_blk.reserve(verts_vec.size());

            VLOG << "[DEBUG][buildBlockData]  Block vertices (Gcc -> Gblk):\n";
            for (ogdf::node vCc : verts_vec) {
                ogdf::node vB = blk.Gblk->newNode();
                cc_to_blk[vCc] = vB;
                blk.toCc[vB] = vCc;
                ogdf::node vG = cc.nodeToOrig[vCc];
                blk.nodeToOrig[vB] = vG;

                VLOG << "    Gcc idx " << vCc->index()
                    << " -> Gblk idx " << vB->index()
                    << " name=" << C.node2name[vG] << "\n";
            }

            VLOG << "[DEBUG][buildBlockData]  Block edges (Gblk):\n";
            for (ogdf::edge hE : cc.bc->hEdges(blk.bNode)) {
                ogdf::edge eCc = cc.bc->original(hE);
                auto srcIt = cc_to_blk.find(eCc->source());
                auto tgtIt = cc_to_blk.find(eCc->target());
                if (srcIt != cc_to_blk.end() && tgtIt != cc_to_blk.end()) {
                    ogdf::edge eB = blk.Gblk->newEdge(srcIt->second, tgtIt->second);
                    blk.edgeToOrig[eB] = cc.edgeToOrig[eCc];

                    ogdf::node uB = srcIt->second;
                    ogdf::node vB = tgtIt->second;
                    ogdf::node uG = blk.nodeToOrig[uB];
                    ogdf::node vG = blk.nodeToOrig[vB];

                    VLOG << "    add Gblk edge: "
                        << C.node2name[uG] << " -- " << C.node2name[vG]
                        << " (Gblk idx " << eB->index() << ")\n";

                    EdgePartType tU = getNodeEdgeType(uG, blk.edgeToOrig[eB]);
                    EdgePartType tV = getNodeEdgeType(vG, blk.edgeToOrig[eB]);

                    if (tU == EdgePartType::PLUS)  blk.blkDegPlus[uB]++; 
                    else                           blk.blkDegMinus[uB]++;

                    if (tV == EdgePartType::PLUS)  blk.blkDegPlus[vB]++;
                    else                           blk.blkDegMinus[vB]++;
                }
            }

            VLOG << "[DEBUG][buildBlockData]  |V(Gblk)|=" << blk.Gblk->numberOfNodes()
                << " |E(Gblk)|=" << blk.Gblk->numberOfEdges() << "\n";

            if (blk.Gblk->numberOfNodes() >= 3) {
                {
                    MARK_SCOPE_MEM("sn/blockData/spqr_build");

                    OGDF_ASSERT(blk.Gblk != nullptr);
                    OGDF_ASSERT(blk.Gblk->numberOfNodes() > 0);

                    blk.spqr = std::make_unique<ogdf::StaticSPQRTree>(*blk.Gblk);
                }

                OGDF_ASSERT(blk.spqr != nullptr);

                const ogdf::Graph& T = blk.spqr->tree();

                blk.skel2tree.clear();
                blk.skel2tree.reserve(2 * T.numberOfEdges());
                for (ogdf::edge te : T.edges) {
                    if (auto eSrc = blk.spqr->skeletonEdgeSrc(te)) {
                        blk.skel2tree[eSrc] = te;
                    }
                    if (auto eTgt = blk.spqr->skeletonEdgeTgt(te)) {
                        blk.skel2tree[eTgt] = te;
                    }
                }

                blk.parent.init(T, nullptr);
                ogdf::node root = blk.spqr->rootNode();
                blk.parent[root] = root;

                std::stack<ogdf::node> st;
                st.push(root);

                while (!st.empty()) {
                    ogdf::node u = st.top();
                    st.pop();

                    for (ogdf::adjEntry adj = u->firstAdj(); adj; adj = adj->succ()) {
                        ogdf::node v = adj->twinNode();
                        if (blk.parent[v] == nullptr) {
                            blk.parent[v] = u;
                            st.push(v);
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

            ogdf::NodeArray<int> blkDegPlus, blkDegMinus;
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
                    if (!cc) continue;
                    if (!cc->Gcc) continue;

                    {
                        MARK_SCOPE_MEM("sn/worker_bcTree/build");

                        // Sanity
                        OGDF_ASSERT(cc->Gcc->numberOfNodes() > 0);

                        {
                            cc->bc = std::make_unique<BCTree>(*cc->Gcc);
                        }
                    }

                    std::vector<BlockPrep> localPreps;
                    {
                        MARK_SCOPE_MEM("sn/worker_bcTree/collect_B_nodes");
                        VLOG << "[DEBUG][worker_bcTree] CC #" << cid
                            << " BC-tree has " << cc->bc->bcTree().numberOfNodes()
                            << " nodes\n";

                        // On ne fait qu'énumérer les B-nodes de type BComp (blocs)
                        for (ogdf::node v : cc->bc->bcTree().nodes) {
                            if (cc->bc->typeOfBNode(v) == BCTree::BNodeType::BComp) {
                                VLOG << "  [DEBUG][worker_bcTree]  B-node "
                                    << v->index() << " (block)\n";
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

            tls_spqr_seen_endpoint_pairs.clear();

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
            std::cout << "Finding snarls...\n";
            PROFILE_FUNCTION();
            auto& C = ctx();
            Graph& G = C.G;

            // -------------------------------
            // Phase I/O: CC + Bucketing
            // -------------------------------
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

            std::vector<std::unique_ptr<CcData>> components(nCC);
            std::vector<BlockPrep> blockPreps;
            {
                PhaseSampler build_sampler(g_stats_build);

                // 1) rebuild cc graphs (worker_component) : parallel
                {
                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});

                    if (numThreads <= 1) {
                        std::mutex workMutex;
                        size_t nextIndex = 0;
                        ThreadComponentArgs* args = new ThreadComponentArgs{
                            0,
                            1,
                            nCC,
                            &nextIndex,
                            &workMutex,
                            &bucket,
                            &edgeBuckets,
                            &components,
                        };
                        worker_component(static_cast<void*>(args));
                    } else {
                        std::vector<pthread_t> threads(numThreads);
                        std::mutex workMutex;
                        size_t nextIndex = 0;

                        for (size_t tid = 0; tid < numThreads; ++tid) {
                            pthread_attr_t attr;
                            pthread_attr_init(&attr);

                            size_t stackSize = C.stackSize;
                            if (stackSize < kMinThreadStackSize)
                                stackSize = kMinThreadStackSize;
                            int err = pthread_attr_setstacksize(&attr, stackSize);
                            if (err != 0) {
                                std::cerr << "[Error] pthread_attr_setstacksize("
                                        << stackSize << "): " << strerror(err) << std::endl;
                            }

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
                }

                // 2) build BC-trees and collect blocks (worker_bcTree)
                //    IMPORTANT : still MONO‑THREAD (OGDF/BCTree is not thread‑safe)
                {
                    size_t numThreads = 1; // security

                    if (numThreads <= 1) {
                        std::mutex workMutex;
                        size_t nextIndex = 0;
                        ThreadBcTreeArgs* args = new ThreadBcTreeArgs{
                            0,
                            1,
                            nCC,
                            &nextIndex,
                            &workMutex,
                            &components,
                            &blockPreps
                        };
                        worker_bcTree(static_cast<void*>(args));
                    } else {
                        // (never reach, numThreads=1)
                    }
                }

                // 3) build SPQR for blocks (worker_block_build)
                //    IMPORTANT : still MONO‑THREAD regards OGDF::StaticSPQRTree
                {
                    MARK_SCOPE_MEM("sn/phase/block_SPQR_build");

                    size_t numThreads = 1; // secutiry

                    if (numThreads <= 1) {
                        std::mutex workMutex;
                        size_t nextIndex = 0;
                        ThreadBlocksArgs* args = new ThreadBlocksArgs{
                            0,
                            1,
                            blockPreps.size(),
                            &nextIndex,
                            &workMutex,
                            &blockPreps
                        };
                        worker_block_build(static_cast<void*>(args));
                    } else {
                        // (never reach, numThreads=1)
                    }
                }
            }

            // ----------------------------------------------------
            // Phase LOGIC : tips/cuts + solve SPQR on blocks
            // ----------------------------------------------------
            {
                PhaseSampler logic_sampler(g_stats_logic);

                // 4) tips & cuts (worker_tips) : parallel
                {
                    MARK_SCOPE_MEM("sn/phase/tips_cuts");

                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});

                    if (numThreads <= 1) {
                        std::mutex workMutex;
                        size_t nextIndex = 0;
                        ThreadTipsArgs* args = new ThreadTipsArgs{
                            0,
                            1,
                            nCC,
                            &nextIndex,
                            &workMutex,
                            &components
                        };
                        worker_tips(static_cast<void*>(args));
                    } else {
                        std::vector<pthread_t> threads(numThreads);

                        std::mutex workMutex;
                        size_t nextIndex = 0;

                        for (size_t tid = 0; tid < numThreads; ++tid) {
                            pthread_attr_t attr;
                            pthread_attr_init(&attr);

                            size_t stackSize = C.stackSize;
                            if (stackSize < kMinThreadStackSize)
                                stackSize = kMinThreadStackSize;
                            int err = pthread_attr_setstacksize(&attr, stackSize);
                            if (err != 0) {
                                std::cerr << "[Error] pthread_attr_setstacksize("
                                        << stackSize << "): " << strerror(err) << std::endl;
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
                }

                // 5) SPQR solve for blocks (worker_block_solve)
                //    IMPORTANT : MONO‑THREAD to avoid issues with OGDF
                {
                    MARK_SCOPE_MEM("sn/phase/block_SPQR_solve");

                    size_t numThreads = 1; // security

                    if (numThreads <= 1) {
                        std::mutex workMutex;
                        size_t nextIndex = 0;
                        ThreadBlocksArgs* args = new ThreadBlocksArgs{
                            0,
                            1,
                            blockPreps.size(),
                            &nextIndex,
                            &workMutex,
                            &blockPreps
                        };
                        worker_block_solve(static_cast<void*>(args));
                    } else {
                        // (never reach, numThreads=1)
                    }
                }
            }

            // -------------------------------
            // Stats
            // -------------------------------
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

    namespace ultrabubble {


        static inline EdgePartType getNodeEdgeTypeCached(
            ogdf::node v,
            ogdf::edge e,
            const ogdf::EdgeArray<std::pair<EdgePartType,EdgePartType>> &etype
        ) {
            OGDF_ASSERT(v != nullptr && e != nullptr);

            if (e->source() == v) return etype[e].first;
            if (e->target() == v) return etype[e].second;

            OGDF_ASSERT(false);
            return EdgePartType::NONE;
        }

        struct DirectedEdgeBuilder {
            int nextId; 
            std::vector<std::pair<int,int>> edges;

            explicit DirectedEdgeBuilder(int original_n, size_t reserve_edges = 0)
                : nextId(original_n)
            {
                if (reserve_edges) edges.reserve(reserve_edges);
            }

            inline int newIntermediate() { return nextId++; }

            inline void addEdge(int a, int b) {
                edges.emplace_back(a, b);
            }
        };

        static inline void emit_oriented_edge_once_local(
            ogdf::node v,
            ogdf::node u,
            ogdf::edge e,
            EdgePartType sign_at_v,
            EdgePartType sign_at_u,
            const ogdf::NodeArray<int> &plus_dir,
            const ogdf::NodeArray<int> &nodeIdGlobal, 
            const ogdf::NodeArray<int> &localId,    
            DirectedEdgeBuilder &out
        ) {
            const int vg = nodeIdGlobal[v];
            const int ug = nodeIdGlobal[u];

            if (vg >= ug) return;

            const int vl = localId[v];
            const int ul = localId[u];
            OGDF_ASSERT(vl >= 0 && ul >= 0);

            const bool inconsistent = ((sign_at_u == sign_at_v) == (plus_dir[u] == plus_dir[v]));

            if (inconsistent) {
                int x = out.newIntermediate(); 

                if ((plus_dir[v] == 1) == (sign_at_v == EdgePartType::PLUS)) {
                    // v -> x <- u
                    out.addEdge(vl, x);
                    out.addEdge(ul, x);
                } else {
                    // v <- x -> u
                    out.addEdge(x, vl);
                    out.addEdge(x, ul);
                }
            } else if ((plus_dir[v] == 1) == (sign_at_v == EdgePartType::PLUS)) {
                // orientation consistent: v -> u
                out.addEdge(vl, ul);
            } else {
                // orientation consistent: u -> v
                out.addEdge(ul, vl);
            }
        }

        static void computeConnectedComponents(
            ogdf::Graph &G,
            ogdf::NodeArray<int> &localId, 
            std::vector<std::vector<ogdf::node>> &comps 
        ) {
            ogdf::NodeArray<bool> seen(G, false);
            localId.init(G, -1);
            comps.clear();

            std::vector<ogdf::node> st;
            st.reserve(1024);

            for (ogdf::node s : G.nodes) {
                if (seen[s]) continue;

                comps.emplace_back();
                auto &cc = comps.back();
                cc.reserve(256);

                st.clear();
                st.push_back(s);
                seen[s] = true;

                while (!st.empty()) {
                    ogdf::node v = st.back();
                    st.pop_back();

                    localId[v] = (int)cc.size();
                    cc.push_back(v);

                    for (auto ae : v->adjEntries) {
                        ogdf::node u = ae->twinNode();
                        if (!seen[u]) {
                            seen[u] = true;
                            st.push_back(u);
                        }
                    }
                }
            }
        }

        static void orient_emit_iterative_cc(
            ogdf::node start,
            bool plus_enter,
            ogdf::NodeArray<int> &plus_dir,
            const ogdf::NodeArray<int> &nodeIdGlobal,
            const ogdf::NodeArray<int> &localId,
            const ogdf::EdgeArray<std::pair<EdgePartType,EdgePartType>> &etype,
            DirectedEdgeBuilder &out
        ) {
            OGDF_ASSERT(start != nullptr);
            OGDF_ASSERT(plus_dir[start] != 0);

            struct Frame {
                ogdf::node v{nullptr};

                EdgePartType order[2]{EdgePartType::PLUS, EdgePartType::MINUS};
                int order_idx{0};

                ogdf::adjEntry it{nullptr};

                bool pending{false};
                ogdf::node pending_u{nullptr};
                ogdf::edge pending_e{nullptr};
                EdgePartType pending_sign_v{EdgePartType::NONE};
                EdgePartType pending_sign_u{EdgePartType::NONE};
            };

            auto make_frame = [&](ogdf::node v, bool plus_enter_local) -> Frame {
                Frame f;
                f.v = v;
                f.order[0] = EdgePartType::PLUS;
                f.order[1] = EdgePartType::MINUS;
                if (plus_enter_local) std::swap(f.order[0], f.order[1]);
                f.order_idx = 0;
                f.it = v->firstAdj();
                f.pending = false;
                return f;
            };

            std::vector<Frame> st;
            st.reserve(1024);
            st.push_back(make_frame(start, plus_enter));

            while (!st.empty()) {
                Frame &f = st.back();
                ogdf::node v = f.v;

                if (f.pending) {
                    emit_oriented_edge_once_local(
                        v, f.pending_u, f.pending_e,
                        f.pending_sign_v, f.pending_sign_u,
                        plus_dir, nodeIdGlobal, localId, out
                    );
                    f.pending = false;
                    continue;
                }

                if (f.order_idx >= 2) {
                    st.pop_back();
                    continue;
                }

                const EdgePartType wanted_sign = f.order[f.order_idx];

                while (f.it != nullptr) {
                    ogdf::adjEntry ae = f.it;
                    f.it = ae->succ();

                    ogdf::edge e = ae->theEdge();
                    ogdf::node u = ae->twinNode();

                    EdgePartType sign_v = getNodeEdgeTypeCached(v, e, etype);
                    if (sign_v != wanted_sign) continue;

                    EdgePartType sign_u = getNodeEdgeTypeCached(u, e, etype);

                    if (!plus_dir[u]) {
                        if (plus_dir[v] == 1) {
                            plus_dir[u] = 1 + (sign_v == sign_u);
                        } else {
                            plus_dir[u] = 1 + (sign_v != sign_u);
                        }

                        f.pending = true;
                        f.pending_u = u;
                        f.pending_e = e;
                        f.pending_sign_v = sign_v;
                        f.pending_sign_u = sign_u;

                        const bool child_plus_enter = (sign_u == EdgePartType::PLUS);
                        st.push_back(make_frame(u, child_plus_enter));
                        goto next_iteration; 
                    } else {
                        emit_oriented_edge_once_local(
                            v, u, e, sign_v, sign_u,
                            plus_dir, nodeIdGlobal, localId, out
                        );
                    }
                }

                f.order_idx++;
                f.it = v->firstAdj();

            next_iteration:
                continue;
            }
        }

        void solve() {
            std::cout << "Finding ultrabubbles...\n";
            PROFILE_FUNCTION();

            auto &C = ctx();
            ogdf::Graph &G = C.G;

            ogdf::EdgeArray<std::pair<EdgePartType,EdgePartType>> etype(G);
            for (ogdf::edge e : G.edges) etype[e] = C._edge2types(e);

            const int original_n = G.numberOfNodes();
            ogdf::NodeArray<int> nodeId(G, -1);
            {
                int id = 0;
                for (ogdf::node v : G.nodes) nodeId[v] = id++;
                OGDF_ASSERT(id == original_n);
            }

            ogdf::NodeArray<bool> is_tip(G, false);
            for (ogdf::node v : G.nodes) {
                bool saw_plus=false, saw_minus=false;
                for (auto ae : v->adjEntries) {
                    EdgePartType t = getNodeEdgeTypeCached(v, ae->theEdge(), etype);
                    if (t == EdgePartType::PLUS)  saw_plus = true;
                    if (t == EdgePartType::MINUS) saw_minus = true;
                    if (saw_plus && saw_minus) break;
                }
                is_tip[v] = !(saw_plus && saw_minus);
            }

            ogdf::NodeArray<int> localId(G, -1);
            std::vector<std::vector<ogdf::node>> comps;
            computeConnectedComponents(G, localId, comps);

            ogdf::NodeArray<int> plus_dir(G, 0);

            std::vector<std::vector<std::pair<std::string,std::string>>> incidencesByCC(comps.size());

            std::atomic<size_t> next{0};
            std::atomic<bool> abort{false};
            std::exception_ptr eptr = nullptr;
            std::mutex ep_mtx;


            int T = std::min<int>(C.threads, (int)comps.size());
            if (T <= 0) T = 1;

            std::vector<std::thread> threads;
            threads.reserve(T);

            for (int t = 0; t < T; ++t) {
                threads.emplace_back([&]() {
                    try {
                        while (!abort.load(std::memory_order_relaxed)) {
                            size_t ci = next.fetch_add(1);
                            if (ci >= comps.size()) break;

                            auto &cc = comps[ci];
                            const int k = (int)cc.size();

                            DirectedEdgeBuilder out(k, /*reserve_edges=*/0);

                            for (ogdf::node v : cc) {
                                if (!plus_dir[v] && is_tip[v]) {
                                    plus_dir[v] = 1;
                                    orient_emit_iterative_cc(
                                        v, true,
                                        plus_dir,
                                        nodeId,   
                                        localId,  
                                        etype,
                                        out
                                    );
                                }
                            }

                            for (ogdf::node v : cc) {
                                if (!plus_dir[v]) {
                                    throw std::runtime_error(
                                        "Ultrabubble: orientation failed (unoriented node: " + C.node2name[v] + "). "
                                        "Veuillez vérifier que chaque CC contient au moins un tip."
                                    );
                                }
                            }

                            // dedup edges
                            auto &directed_edges = out.edges;
                            std::sort(directed_edges.begin(), directed_edges.end());
                            directed_edges.erase(std::unique(directed_edges.begin(), directed_edges.end()),
                                                directed_edges.end());

                            std::vector<std::pair<int,int>> superbubbles;
                            // { std::lock_guard<std::mutex> lk(clsd_mtx);
                            //   superbubbles = compute_superbubbles_from_edges(directed_edges);
                            // }

                            std::vector<supertree*> trees;
                            superbubbles = compute_superbubbles_from_edges(directed_edges, &trees);

                            

                            // for(auto &t:trees) {
                            //     Config::outputTree(t, C.ultrabubbleTreeOutputPath);
                            // }

                            if(trees.size() > 0) {
                                auto hierarchy = [&](auto&& self, supertree* t, std::string &res) -> void {
                                    int xid = std::stoi(t->entrance->getID());
                                    int yid = std::stoi(t->exit->getID());

                                    bool valid = (xid >= 0 && xid < k) && (yid >= 0 && yid < k);


                                    std::string X="", Y="";

                                    if(valid) {
                                        ogdf::node x = cc[xid];
                                        ogdf::node y = cc[yid];

                                        std::string xname = C.node2name[x];
                                        std::string yname = C.node2name[y];

                                        char xsign = "-+"[ plus_dir[x] == 1 ];
                                        char ysign = "+-"[ plus_dir[y] == 1 ];

                                        X = xname + xsign;
                                        Y = yname + ysign;
                                    }

                                    if (valid && !t->childreen.empty()) {
                                        res += "(";
                                    }

                                     for (size_t i = 0; i < t->childreen.size(); ++i) {
                                        self(self, t->childreen[i], res);
                                        if (i != t->childreen.size() - 1 && valid) {
                                            res += ",";
                                        }
                                    }

                                    if (valid && !t->childreen.empty()) {
                                        res += ")";
                                    }

                                    if (valid) {
                                        res += "<" + X + "," + Y + ">";
                                    }

                                    // if (! t->childreen.empty()) {
                                    //     res += "(";
                                    //     for(long unsigned int i = 0; i < t->childreen.size(); i++) {
                                    //         self(self, t->childreen[i], res);
                                    //         if (i != t->childreen.size()-1) {
                                    //             res += ",";
                                    //         }
                                    //     }
                                    //     res += ")";
                                    // }
                                    // res += "<" + t->entrance->getID();
                                    // res += "," + t->exit->getID();
                                    // res += ">";
                                };

                                // auto hierarchy = [&](auto&& self, supertree* t, std::string &res) -> void { 
                                //     if (! t->childreen.empty()) { 
                                //         res += "("; 
                                //         for(long unsigned int i = 0; i < t->childreen.size(); i++) { 
                                //             self(self, t->childreen[i], res); 
                                //             if (i != t->childreen.size()-1) { 
                                //                 res += ","; 
                                //             } 
                                //         } res += ")"; 
                                //     } 
                                //     res += "<" + t->entrance->getID(); 
                                //     res += "," + t->exit->getID(); 
                                //     res += ">"; 
                                // };

                                for(size_t i = 0; i < trees.size(); i++) {
                                    // std::cout << "(" << trees[i]->entrance->getID(),
                                    // std::cout << "," << trees[i]->exit->getID() << ")" << std::endl;
                                    // if(trees[i]->entrance->getID().size()>0) {
                                        std::string res = "";
                                        hierarchy(hierarchy, trees[i], res);
                                        std::cout << res << std::endl;
                                    // }
                                }
                                std::cout << "done" << std::endl;
                            }

                            
                            auto &inc = incidencesByCC[ci];
                            inc.reserve(superbubbles.size());

                            std::cout << 121321 << std::endl;

                            for (auto &sb : superbubbles) {
                                int xid = sb.first;
                                int yid = sb.second;

                                // ignore intermédiaires
                                if (xid < 0 || yid < 0) continue;
                                if (xid >= k || yid >= k) continue;

                                ogdf::node x = cc[xid];
                                ogdf::node y = cc[yid];

                                std::string xname = C.node2name[x];
                                std::string yname = C.node2name[y];

                                char xsign = "-+"[ plus_dir[x] == 1 ];
                                char ysign = "+-"[ plus_dir[y] == 1 ];

                                int xg = nodeId[x];
                                int yg = nodeId[y];

                                if (xg > yg) inc.emplace_back(yname + ysign, xname + xsign);
                                else         inc.emplace_back(xname + xsign, yname + ysign);
                            }
                        }
                    } catch (...) {
                        abort.store(true);
                        std::lock_guard<std::mutex> lk(ep_mtx);
                        if (!eptr) eptr = std::current_exception();
                    }
                });
            }

            for (auto &th : threads) th.join();
            if (eptr) std::rethrow_exception(eptr);

            // merge final
            C.ultrabubbleIncidences.clear();
            size_t total = 0;
            for (auto &v : incidencesByCC) total += v.size();
            C.ultrabubbleIncidences.reserve(total);

            for (size_t ci = 0; ci < incidencesByCC.size(); ++ci) {
                for (auto &p : incidencesByCC[ci]) {
                    C.ultrabubbleIncidences.emplace_back(std::move(p));
                }
            }

            std::cout << "ULTRABUBBLES found: " << C.ultrabubbleIncidences.size() << "\n";
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
        std::string err;

        if (!inputFileReadable(ctx().graphPath, err)) {
            std::cerr << "Error: cannot open input graph file '"
                      << ctx().graphPath << "' for reading: "
                      << err << "\n";
            return 1;
        }

        if (!outputParentDirWritable(ctx().outputPath, err)) {
            std::cerr << "Error: cannot write output file '"
                      << ctx().outputPath << "': "
                      << err << "\n";
            return 1;
        }
    }    

    {
        MARK_SCOPE_MEM("io/read_graph");
        PROFILE_BLOCK("Graph reading");
        ::GraphIO::readGraph();
    }

    if (ctx().bubbleType == Context::BubbleType::SUPERBUBBLE) {
        solver::superbubble::solve();
        VLOG << "[main] Superbubble solve finished. Oriented superbubbles: "
             << ctx().superbubbles.size() << std::endl;
    } else if (ctx().bubbleType == Context::BubbleType::SNARL) {
        solver::snarls::solve();
        VLOG << "[main] Snarl solve finished. Snarls: "
             << ctx().snarls.size() << std::endl;
    } else if (ctx().bubbleType == Context::BubbleType::ULTRABUBBLE) {
        solver::ultrabubble::solve();
        // VLOG << "[main] Ultrabubble solve finished. Ultrabubbles: "
        //      << ctx().ultrabubbles.size() << std::endl;
    } 

    {
        MARK_SCOPE_MEM("io/write_output");
        PROFILE_BLOCK("Writing output");
        TIME_BLOCK("Writing output");
        ::GraphIO::writeSuperbubbles();
    }

    if (ctx().bubbleType == Context::BubbleType::SNARL) {
        std::cout << "Snarls found: " << snarlsFound << std::endl;
    }

    PROFILING_REPORT();

    logger::info("Process PeakRSS: {:.2f} GiB",
                 memtime::peakRSSBytes() / (1024.0 * 1024.0 * 1024.0));

    mark::report();
    if (!g_report_json_path.empty()) {
        mark::report_to_json(g_report_json_path);
    }

    return 0;
}