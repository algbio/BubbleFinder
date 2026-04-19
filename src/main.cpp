#include "util/ogdf_all.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
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
#include <set>      
#include <queue>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>

#include <sys/resource.h>
#include <sys/time.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstring>

#ifdef __APPLE__
#include <mach/mach.h>
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
#include "io/gfa_parser.hpp"

#include "io/gbz_parser.hpp"

bool VERBOSE = false;
#define VLOG     \
    if (VERBOSE) \
    std::cerr

namespace metrics
{

    enum class Phase : uint8_t
    {
        IO = 0,
        BUILD = 1,
        LOGIC = 2,
        COUNT = 3
    };

    struct PhaseState
    {
        std::atomic<bool> running{false};
        std::atomic<size_t> baseline_rss{0};   // bytes
        std::atomic<size_t> peak_rss_delta{0}; // bytes
        std::atomic<uint64_t> start_ns{0};
        std::atomic<uint64_t> elapsed_ns{0};
    };

    inline uint64_t now_ns()
    {
        return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    // Current RSS in bytes
    inline size_t currentRSS()
    {
#ifdef __linux__
        // /proc/self/statm: size resident shared text lib data dt
        // We read resident pages (2nd field) * page size
        FILE *f = std::fopen("/proc/self/statm", "r");
        if (!f)
        {
            struct rusage ru{};
            getrusage(RUSAGE_SELF, &ru);
            return (size_t)ru.ru_maxrss * 1024ull;
        }
        long rss_pages = 0;
        long dummy = 0;
        if (std::fscanf(f, "%ld %ld", &dummy, &rss_pages) != 2)
        {
            std::fclose(f);
            struct rusage ru{};
            getrusage(RUSAGE_SELF, &ru);
            return (size_t)ru.ru_maxrss * 1024ull;
        }
        std::fclose(f);
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0)
            page_size = 4096;
        return (size_t)rss_pages * (size_t)page_size;
#else
        struct rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        return (size_t)ru.ru_maxrss * 1024ull;
#endif
    }

    inline std::array<PhaseState, (size_t)Phase::COUNT> &states()
    {
        static std::array<PhaseState, (size_t)Phase::COUNT> s;
        return s;
    }

    inline void beginPhase(Phase p)
    {
        auto &st = states()[(size_t)p];
        const uint64_t t0 = now_ns();
        const size_t base = currentRSS();
        st.baseline_rss.store(base, std::memory_order_relaxed);
        st.peak_rss_delta.store(0, std::memory_order_relaxed);
        st.start_ns.store(t0, std::memory_order_relaxed);
        st.running.store(true, std::memory_order_release);
    }

    inline void updateRSS(Phase p)
    {
        auto &st = states()[(size_t)p];
        if (!st.running.load(std::memory_order_acquire))
            return;
        const size_t base = st.baseline_rss.load(std::memory_order_relaxed);
        const size_t cur = currentRSS();
        size_t delta = (cur >= base ? (cur - base) : 0);
        size_t prev = st.peak_rss_delta.load(std::memory_order_relaxed);
        while (delta > prev &&
               !st.peak_rss_delta.compare_exchange_weak(prev, delta,
                                                        std::memory_order_relaxed))
        {
        }
    }

    inline void endPhase(Phase p)
    {
        auto &st = states()[(size_t)p];
        if (!st.running.load(std::memory_order_acquire))
            return;
        updateRSS(p);
        const uint64_t t1 = now_ns();
        const uint64_t t0 = st.start_ns.load(std::memory_order_relaxed);
        const uint64_t d = (t1 >= t0 ? (t1 - t0) : 0);
        uint64_t prev = st.elapsed_ns.load(std::memory_order_relaxed);
        st.elapsed_ns.store(prev + d, std::memory_order_relaxed);
        st.running.store(false, std::memory_order_release);
    }

    struct Snapshot
    {
        uint64_t elapsed_ns;
        size_t peak_rss_delta;
    };

    inline Snapshot snapshot(Phase p)
    {
        auto &st = states()[(size_t)p];
        return Snapshot{
            st.elapsed_ns.load(std::memory_order_relaxed),
            st.peak_rss_delta.load(std::memory_order_relaxed)};
    }
}

inline void METRICS_PHASE_BEGIN(metrics::Phase p) { metrics::beginPhase(p); }
inline void METRICS_PHASE_END(metrics::Phase p) { metrics::endPhase(p); }
inline void PHASE_RSS_UPDATE_IO() { metrics::updateRSS(metrics::Phase::IO); }
inline void PHASE_RSS_UPDATE_BUILD() { metrics::updateRSS(metrics::Phase::BUILD); }
inline void PHASE_RSS_UPDATE_LOGIC() { metrics::updateRSS(metrics::Phase::LOGIC); }


using namespace ogdf;

static std::string g_report_json_path;


static void usage(const char *prog, int exitCode)
{
    struct CommandHelp
    {
        const char *name;
        const char *desc;
    };

    struct OptionHelp
    {
        const char *flag;
        const char *arg;
        const char *desc;
    };

    static const CommandHelp commands[] = {
        { "superbubbles",
          "Superbubbles (bidirected by default; use --directed for directed mode)" },
        { "snarls",
          "Snarls (typically on bidirected graphs from GFA)" },
        { "ultrabubbles",
          "Ultrabubbles.\n"
          "      Oriented mode (default): each CC must have at least one tip OR one cut vertex.\n"
          "      Doubled mode (--doubled): no such restriction, but uses more RAM due to graph doubling." },
        { "spqr-tree",
          "Compute and output the SPQR tree of the input graph" }
    };

    static const OptionHelp options[] = {
        { "-g", "<file>",    "Input graph file (possibly compressed)" },
        { "-o", "<file>",    "Output file" },
        { "-j", "<threads>", "Number of threads" },

        { "--gfa", nullptr,          "Force GFA input (bidirected)" },
        { "--gfa-directed", nullptr, "Force GFA input interpreted as directed graph" },
        { "--graph", nullptr,
          "Force .graph text format (see 'Format options' above)" },

        { "--directed", nullptr,
          "Interpret the graph as directed for the superbubbles command (default: bidirected)" },

        { "--doubled", nullptr,
          "Use the doubled-graph algorithm for ultrabubbles (no tip/cut-vertex requirement per CC, higher RAM)" },

        { "--clsd-trees", "<file>",
          "Write CLSD superbubble trees (ultrabubble hierarchy) to <file> (ultrabubbles command only)" },

        { "-T, --include-trivial", nullptr,
          "Include trivial bubbles in output (default: excluded; ultrabubbles, superbubbles and snarls commands)" },

        { "--report-json", "<file>", "Write JSON metrics report" },
        { "-m", "<bytes>",           "Stack size in bytes" },
        { "-h, --help", nullptr,     "Show this help message and exit" }
    };

    std::cerr << "Usage:\n"
              << "  " << prog
              << " <command> -g <graphFile> -o <outputFile> [options]\n\n";

    std::cerr << "Commands:\n";
    for (const auto &c : commands)
    {
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
              << "  from the file extension (.gfa, .gbz, .graph).\n\n";

    std::cerr << "Supported input formats:\n"
              << "  .gfa / .gfa1 / .gfa2   GFA (auto-detected)\n"
              << "  .gbz                    GBZ (vg/gbwtgraph format)\n"
              << "  .graph                  Simple directed edge list\n\n";

    std::cerr << "Compression:\n"
              << "  Compression is auto-detected from the file name suffix:\n"
              << "    .gz / .bgz  -> gzip\n"
              << "    .bz2        -> bzip2\n"
              << "    .xz         -> xz\n\n";

    std::cerr << "General options:\n";
    for (const auto &o : options)
    {
        std::cerr << "  " << o.flag;
        if (o.arg)
        {
            std::cerr << " " << o.arg;
        }
        std::cerr << "\n      " << o.desc << "\n";
    }

    std::exit(exitCode);
}

static std::string nextArgOrDie(const std::vector<std::string> &a,
                                std::size_t &i,
                                const char *flag)
{
    if (++i >= a.size() || (a[i][0] == '-' && a[i] != "-"))
    {
        std::cerr << "Error: missing argument after " << flag << "\n";
        usage(a[0].c_str(), 1);
    }
    return a[i];
}

static std::string toLowerCopy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c)
                   {
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
    if (slashPos != std::string::npos)
    {
        filename = filename.substr(slashPos + 1);
    }

    coreExtOut.clear();

    auto dotPos = filename.find_last_of('.');
    if (dotPos == std::string::npos)
    {
        return Context::Compression::None;
    }

    std::string lastExt = toLowerCopy(filename.substr(dotPos + 1));
    std::string base = filename.substr(0, dotPos);

    Context::Compression comp = Context::Compression::None;

    if (lastExt == "gz" || lastExt == "bgz")
    {
        comp = Context::Compression::Gzip;
    }
    else if (lastExt == "bz2")
    {
        comp = Context::Compression::Bzip2;
    }
    else if (lastExt == "xz")
    {
        comp = Context::Compression::Xz;
    }
    else
    {
        coreExtOut = lastExt;
        return Context::Compression::None;
    }

    auto dotPos2 = base.find_last_of('.');
    if (dotPos2 != std::string::npos)
    {
        coreExtOut = toLowerCopy(base.substr(dotPos2 + 1));
    }
    else
    {
        coreExtOut.clear();
    }

    return comp;
}



static bool inputFileReadable(const std::string &path, std::string &errOut)
{
    struct stat st{};
    if (stat(path.c_str(), &st) != 0)
    {
        errOut = std::string("stat failed: ") + std::strerror(errno);
        return false;
    }
    if (!S_ISREG(st.st_mode))
    {
        errOut = "path exists but is not a regular file";
        return false;
    }
    if (access(path.c_str(), R_OK) != 0)
    {
        errOut = std::string("no read permission: ") + std::strerror(errno);
        return false;
    }
    return true;
}

static bool outputParentDirWritable(const std::string &path, std::string &errOut)
{
    std::string dir;
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos)
    {
        dir = ".";
    }
    else if (pos == 0)
    {
        dir = "/";
    }
    else
    {
        dir = path.substr(0, pos);
    }

    struct stat st{};
    if (stat(dir.c_str(), &st) != 0)
    {
        errOut = std::string("cannot stat output directory '") + dir +
                 "': " + std::strerror(errno);
        return false;
    }
    if (!S_ISDIR(st.st_mode))
    {
        errOut = "'" + dir + "' is not a directory";
        return false;
    }
    if (access(dir.c_str(), W_OK) != 0)
    {
        errOut = std::string("no write permission on '") + dir +
                 "': " + std::strerror(errno);
        return false;
    }
    return true;
}


void readArgs(int argc, char **argv)
{
    auto &C = ctx();

    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 2)
    {
        usage(args[0].c_str(), 1);
    }

    std::size_t i = 1;

    const std::string cmd = args[i];

    if (cmd == "-h" || cmd == "--help")
    {
        usage(args[0].c_str(), 0);
    }
    else if (cmd == "superbubbles")
    {
        C.bubbleType = Context::BubbleType::SUPERBUBBLE;
        C.directedSuperbubbles = false;  // bidirected by default
    }
    else if (cmd == "directed-superbubbles")
    {
        // Backward compatibility
        C.bubbleType = Context::BubbleType::SUPERBUBBLE;
        C.directedSuperbubbles = true;
    }
    else if (cmd == "snarls")
    {
        C.bubbleType = Context::BubbleType::SNARL;
        C.directedSuperbubbles = false;
    }
    else if (cmd == "ultrabubbles")
    {
        C.bubbleType = Context::BubbleType::ULTRABUBBLE;
        C.directedSuperbubbles = false;
        C.doubledUltrabubbles = false;
    }
    else if (cmd == "spqr-tree")
    {
        C.bubbleType = Context::BubbleType::SPQR_TREE_ONLY;
        C.directedSuperbubbles = false;
    }
    else
    {
        std::cerr << "Error: unknown command '" << cmd
                  << "'. Expected one of: superbubbles, snarls, ultrabubbles, spqr-tree.\n\n";
        usage(args[0].c_str(), 1);
    }

    ++i;

    for (; i < args.size(); ++i)
    {
        const std::string &s = args[i];

        if (s == "-g")
        {
            C.graphPath = nextArgOrDie(args, i, "-g");
        }
        else if (s == "-o")
        {
            C.outputPath = nextArgOrDie(args, i, "-o");
        }
        else if (s == "--gfa")
        {
            if (C.inputFormat != Context::InputFormat::Auto &&
                C.inputFormat != Context::InputFormat::Gfa)
            {
                std::cerr << "Error: multiple conflicting input format options "
                             "(--gfa / --gfa-directed / --graph).\n";
                std::exit(1);
            }
            C.inputFormat = Context::InputFormat::Gfa;
        }
        else if (s == "--gfa-directed")
        {
            if (C.inputFormat != Context::InputFormat::Auto &&
                C.inputFormat != Context::InputFormat::GfaDirected)
            {
                std::cerr << "Error: multiple conflicting input format options "
                             "(--gfa / --gfa-directed / --graph).\n";
                std::exit(1);
            }
            C.inputFormat = Context::InputFormat::GfaDirected;
        }
        else if (s == "--graph")
        {
            if (C.inputFormat != Context::InputFormat::Auto &&
                C.inputFormat != Context::InputFormat::Graph)
            {
                std::cerr << "Error: multiple conflicting input format options "
                             "(--gfa / --gfa-directed / --graph).\n";
                std::exit(1);
            }
            C.inputFormat = Context::InputFormat::Graph;
        }
        else if (s == "--directed")
        {
            if (C.bubbleType != Context::BubbleType::SUPERBUBBLE)
            {
                std::cerr << "Error: option '--directed' is only supported with the "
                             "'superbubbles' command.\n";
                std::exit(1);
            }
            C.directedSuperbubbles = true;
        }
        else if (s == "--doubled")
        {
            if (C.bubbleType != Context::BubbleType::ULTRABUBBLE)
            {
                std::cerr << "Error: option '--doubled' is only supported with the "
                             "'ultrabubbles' command.\n";
                std::exit(1);
            }
            C.doubledUltrabubbles = true;
        }
        else if (s == "--report-json")
        {
            g_report_json_path = nextArgOrDie(args, i, "--report-json");
        }
        else if (s == "--clsd-trees")
        {
            if (C.bubbleType != Context::BubbleType::ULTRABUBBLE)
            {
                std::cerr << "Error: option '--clsd-trees' is only supported with the "
                             "'ultrabubbles' command.\n";
                std::exit(1);
            }
            C.clsdTrees = true;
            C.clsdTreesPath = nextArgOrDie(args, i, "--clsd-trees");
            if (C.clsdTreesPath.empty() || C.clsdTreesPath == "-")
            {
                std::cerr << "Error: --clsd-trees requires a real output file path (not '-').\n";
                std::exit(1);
            }
        }
        else if (s == "-T" || s == "--include-trivial")
        {
            if (C.bubbleType != Context::BubbleType::ULTRABUBBLE &&
                C.bubbleType != Context::BubbleType::SUPERBUBBLE &&
                C.bubbleType != Context::BubbleType::SNARL)
            {
                std::cerr << "Error: option '-T' / '--include-trivial' is only supported with the "
                             "'ultrabubbles', 'superbubbles' or 'snarls' command.\n";
                std::exit(1);
            }
            C.includeTrivial = true;
        }
        else if (s == "-j")
        {
            const std::string v = nextArgOrDie(args, i, "-j");
            try
            {
                C.threads = std::stoi(v);
            }
            catch (const std::exception &)
            {
                std::cerr << "Error: invalid value for -j <threads>: '" << v
                          << "'. Expected a positive integer.\n";
                std::exit(1);
            }
            if (C.threads <= 0)
            {
                std::cerr << "Error: -j <threads> must be a positive integer (got "
                          << C.threads << ").\n";
                std::exit(1);
            }
        }
        else if (s == "-m")
        {
            const std::string v = nextArgOrDie(args, i, "-m");
            try
            {
                C.stackSize = std::stoull(v);
            }
            catch (const std::exception &)
            {
                std::cerr << "Error: invalid value for -m <bytes>: '" << v
                          << "'. Expected a positive integer (number of bytes).\n";
                std::exit(1);
            }
            if (C.stackSize == 0)
            {
                std::cerr << "Error: -m <bytes> must be a positive integer.\n";
                std::exit(1);
            }
        }
        else if (s == "-sanity")
        {
            std::exit(0);
        }
        else if (s == "-h" || s == "--help")
        {
            usage(args[0].c_str(), 0);
        }
        else
        {
            std::cerr << "Unknown argument: " << s << "\n";
            usage(args[0].c_str(), 1);
        }
    }

    if (C.graphPath.empty())
    {
        std::cerr << "Error: missing -g <graphFile>.\n\n";
        usage(args[0].c_str(), 1);
    }
    if (C.outputPath.empty())
    {
        std::cerr << "Error: missing -o <outputFile>.\n\n";
        usage(args[0].c_str(), 1);
    }

    if (C.clsdTrees && C.bubbleType != Context::BubbleType::ULTRABUBBLE)
    {
        std::cerr << "Error: option '--clsd-trees' is only supported with the "
                     "'ultrabubbles' command.\n";
        std::exit(1);
    }

    std::string coreExt;
    C.compression = detectCompressionAndCoreExt(C.graphPath, coreExt);

    if (C.inputFormat == Context::InputFormat::Auto)
    {
        if (coreExt == "gfa" || coreExt == "gfa1" || coreExt == "gfa2" || coreExt == "gbz")
        {
            if (C.directedSuperbubbles)
            {
                C.inputFormat = Context::InputFormat::GfaDirected;
            }
            else
            {
                C.inputFormat = Context::InputFormat::Gfa;
            }
        }
        else if (coreExt == "graph")
        {
            C.inputFormat = Context::InputFormat::Graph;
        }
        else
        {
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

static inline void __phase_rss_update(std::atomic<size_t> &dst)
{
    size_t cur = memtime::peakRSSBytes();
    size_t old = dst.load(std::memory_order_relaxed);
    while (cur > old &&
           !dst.compare_exchange_weak(old, cur, std::memory_order_relaxed))
    {
    }
}

#define PHASE_RSS_UPDATE_IO_LEGACY() __phase_rss_update(g_phase_io_max_rss)
#define PHASE_RSS_UPDATE_BUILD_LEGACY() __phase_rss_update(g_phase_build_max_rss)
#define PHASE_RSS_UPDATE_LOGIC_LEGACY() __phase_rss_update(g_phase_logic_max_rss)

struct OgdfAcc
{
    std::chrono::high_resolution_clock::time_point t0;
    OgdfAcc() : t0(std::chrono::high_resolution_clock::now()) {}
    ~OgdfAcc()
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        g_ogdf_total_us.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
            std::memory_order_relaxed);
    }
};

#define OGDF_ACC_SCOPE() OgdfAcc __ogdf_acc_guard;

#define OGDF_EVAL(TAG, EXPR) \
    ([&]() -> decltype(EXPR) { \
        OGDF_ACC_SCOPE(); \
        MEM_TIME_BLOCK(TAG); \
        MARK_SCOPE_MEM(TAG); \
        PROFILE_BLOCK(TAG); \
        return (EXPR); })()

#define OGDF_NEW_UNIQUE(TAG, T, ...) \
    ([&]() { \
        OGDF_ACC_SCOPE(); \
        MEM_TIME_BLOCK(TAG); \
        MARK_SCOPE_MEM(TAG); \
        PROFILE_BLOCK(TAG); \
        return std::make_unique<T>(__VA_ARGS__); })()

#define OGDF_SCOPE(TAG)  \
    OGDF_ACC_SCOPE();    \
    MEM_TIME_BLOCK(TAG); \
    MARK_SCOPE_MEM(TAG); \
    PROFILE_BLOCK(TAG)

namespace solver
{

    // namespace superbubble {
    //     namespace
    //     {
    //         thread_local std::vector<std::pair<ogdf::node, ogdf::node>> *tls_superbubble_collector = nullptr;
    //     }

    //     static bool tryCommitSuperbubble(ogdf::node source, ogdf::node sink)
    //     {
    //         auto &C = ctx();
    //         if (ctx().node2name[source] == "_trash" ||
    //             ctx().node2name[sink] == "_trash")
    //         {
    //             return false;
    //         }
    //         C.superbubbles.emplace_back(source, sink);
    //         return true;
    //     }

    //     void addSuperbubble(ogdf::node source, ogdf::node sink)
    //     {
    //         if (tls_superbubble_collector)
    //         {
    //             tls_superbubble_collector->emplace_back(source, sink);
    //             return;
    //         }
    //         tryCommitSuperbubble(source, sink);
    //     }

    //     void findMiniSuperbubbles()
    //     {
    //         MARK_SCOPE_MEM("sb/findMini");
    //         auto &C = ctx();
    //         if (!C.includeTrivial) return;

    //         logger::info("Finding mini-superbubbles..");
    //         for (auto &e : C.G.edges)
    //         {
    //             auto a = e->source(); auto b = e->target();
    //             if (a->outdeg() == 1 && b->indeg() == 1)
    //             {
    //                 bool ok = true;
    //                 for (auto &w : b->adjEntries)
    //                 {
    //                     auto e2 = w->theEdge();
    //                     if (e2->source() == b && e2->target() == a)
    //                     { ok = false; break; }
    //                 }
    //                 if (ok) addSuperbubble(a, b);
    //             }
    //         }
    //         logger::info("Checked for mini-superbubbles");
    //     }

    //     struct CcWork {
    //         std::vector<ogdf::node> nodes;
    //         std::vector<ogdf::edge> edges;
    //     };

    //     struct ThreadArgs {
    //         size_t tid;
    //         size_t numThreads;
    //         size_t nItems;
    //         std::atomic<size_t> *nextIndex;
    //         std::vector<CcWork> *work;
    //         std::vector<std::vector<std::pair<ogdf::node, ogdf::node>>> *results;
    //     };

    //     static void worker_process_cc(ThreadArgs targs)
    //     {
    //         auto &work = *targs.work;
    //         auto &results = *targs.results;
    //         const size_t n = targs.nItems;
    //         const bool keep_trivial = ctx().includeTrivial;

    //         size_t processed = 0;

    //         while (true)
    //         {
    //             size_t i = targs.nextIndex->fetch_add(1);
    //             if (i >= n) break;

    //             auto &cc = work[i];
    //             const int nNodes = (int)cc.nodes.size();
    //             if (nNodes <= 1) continue;

    //             std::unordered_map<ogdf::node, int> nodeToId;
    //             nodeToId.reserve(nNodes);
    //             std::vector<ogdf::node> idToNode(nNodes);
    //             for (int j = 0; j < nNodes; j++)
    //             {
    //                 nodeToId[cc.nodes[j]] = j;
    //                 idToNode[j] = cc.nodes[j];
    //             }

    //             std::vector<std::pair<int,int>> directed_edges;
    //             directed_edges.reserve(cc.edges.size());
    //             for (ogdf::edge e : cc.edges)
    //             {
    //                 int src = nodeToId[e->source()];
    //                 int tgt = nodeToId[e->target()];
    //                 directed_edges.emplace_back(src, tgt);
    //             }

    //             std::sort(directed_edges.begin(), directed_edges.end());
    //             directed_edges.erase(
    //                 std::unique(directed_edges.begin(), directed_edges.end()),
    //                 directed_edges.end());

    //             auto superbubbles = compute_weak_superbubbles_from_edges(
    //                 nNodes, directed_edges, nullptr);

    //             if (!keep_trivial && !superbubbles.empty())
    //             {
    //                 std::vector<int> odeg(nNodes, 0);
    //                 for (const auto &de : directed_edges)
    //                     odeg[de.first]++;

    //                 superbubbles.erase(
    //                     std::remove_if(superbubbles.begin(),
    //                                 superbubbles.end(),
    //                         [&](const std::pair<int,int> &sb) {
    //                             return odeg[sb.first] == 1 &&
    //                                 std::binary_search(
    //                                     directed_edges.begin(),
    //                                     directed_edges.end(),
    //                                     std::make_pair(sb.first, sb.second));
    //                         }),
    //                     superbubbles.end());
    //             }

    //             auto &local = results[i];
    //             local.reserve(superbubbles.size());

    //             for (auto &sb : superbubbles)
    //             {
    //                 int xid = sb.first;
    //                 int yid = sb.second;

    //                 if (xid < 0 || xid >= nNodes ||
    //                     yid < 0 || yid >= nNodes)
    //                     continue;

    //                 ogdf::node xg = idToNode[xid];
    //                 ogdf::node yg = idToNode[yid];

    //                 const std::string &xName = ctx().node2name[xg];
    //                 const std::string &yName = ctx().node2name[yg];

    //                 if (xName == "_trash" || yName == "_trash")
    //                     continue;

    //                 local.emplace_back(xg, yg);
    //             }

    //             ++processed;
    //         }

    //         std::cout << "Thread " << targs.tid
    //                 << " processed " << processed
    //                 << " CCs on doubled graph" << std::endl;
    //     }

    //     void solveStreaming()
    //     {
    //         auto &C = ctx();
    //         Graph &G = C.G;

    //         NodeArray<int> compIdx(G);
    //         int nCC;
    //         {
    //             MARK_SCOPE_MEM("sb/phase/ComputeCC");
    //             nCC = connectedComponents(G, compIdx);
    //         }

    //         std::vector<CcWork> work(nCC);
    //         {
    //             MARK_SCOPE_MEM("sb/phase/BucketNodesEdges");
    //             for (ogdf::node v : G.nodes)
    //                 work[compIdx[v]].nodes.push_back(v);
    //             for (ogdf::edge e : G.edges)
    //                 work[compIdx[e->source()]].edges.push_back(e);
    //         }

    //         logger::info("Doubled graph: {} CCs, processing each CC entirely via CLSD", nCC);

    //         std::vector<std::vector<std::pair<ogdf::node, ogdf::node>>> results(nCC);
    //         std::atomic<size_t> nextIndex{0};

    //         size_t numThreads = std::thread::hardware_concurrency();
    //         numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});
    //         if (numThreads == 0) numThreads = 1;

    //         {
    //             MARK_SCOPE_MEM("sb/phase/SolveCCs");

    //             std::vector<std::thread> threads;
    //             threads.reserve(numThreads);

    //             for (size_t tid = 0; tid < numThreads; ++tid)
    //             {
    //                 threads.emplace_back(worker_process_cc, ThreadArgs{
    //                     tid, numThreads, (size_t)nCC,
    //                     &nextIndex, &work, &results
    //                 });
    //             }

    //             for (auto &t : threads)
    //                 t.join();
    //         }

    //         {
    //             MARK_SCOPE_MEM("sb/phase/CommitResults");
    //             for (const auto &candidates : results)
    //                 for (const auto &p : candidates)
    //                     tryCommitSuperbubble(p.first, p.second);
    //         }

    //         logger::info("Superbubbles on doubled graph: {} committed",
    //                      C.superbubbles.size());
    //     }

    //     void solve()
    //     {
    //         TIME_BLOCK("Finding superbubbles on doubled graph");
    //         if (ctx().directedSuperbubbles)
    //             findMiniSuperbubbles();
    //         solveStreaming();
    //     }
    // }   

    namespace superbubble {
        namespace
        {
            thread_local std::vector<std::pair<ogdf::node, ogdf::node>> *tls_superbubble_collector = nullptr;
        }

        static bool tryCommitSuperbubble(ogdf::node source, ogdf::node sink)
        {
            auto &C = ctx();
            if (C.isEntry[source] || C.isExit[sink] || ctx().node2name[source] == "_trash" || ctx().node2name[sink] == "_trash")
            {
                // std::cout << C.node2name[source] << " " << C.node2name[sink] << " is already superbubble\n";
                return false;
            }
            C.isEntry[source] = true;
            C.isExit[sink] = true;
            C.superbubbles.emplace_back(source, sink);
            // std::cout << "Added " << C.node2name[source] << " " << C.node2name[sink] << " as superbubble\n";
            return true;
        }
        struct BlockData
        {
            std::unique_ptr<ogdf::Graph> Gblk;
            ogdf::NodeArray<ogdf::node> toCc;
            // ogdf::NodeArray<ogdf::node> toBlk;
            ogdf::NodeArray<ogdf::node> toOrig;

            std::unique_ptr<ogdf::StaticSPQRTree> spqr;
            std::unordered_map<ogdf::edge, ogdf::edge> skel2tree; // mapping from skeleton virtual edge to tree edge
            ogdf::NodeArray<ogdf::node> parent;                   // mapping from node to parent in SPQR tree, it is possible since it is rooted,
                                                                  // parent of root is nullptr

            ogdf::NodeArray<ogdf::node> blkToSkel;

            ogdf::node bNode{nullptr};

            bool isAcycic{true};

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

        struct CcData
        {
            std::unique_ptr<ogdf::Graph> Gcc;
            ogdf::NodeArray<ogdf::node> toOrig;
            // ogdf::NodeArray<ogdf::node> toCopy;
            // ogdf::NodeArray<ogdf::node> toBlk;

            std::unique_ptr<ogdf::BCTree> bc;
            // std::vector<BlockData> blocks;
            std::vector<std::unique_ptr<BlockData>> blocks;
        };

        void printBlockEdges(std::vector<CcData> &comps)
        {
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

        void addSuperbubble(ogdf::node source, ogdf::node sink)
        {
            if (tls_superbubble_collector)
            {
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

        namespace SPQRsolve
        {
            struct EdgeDPState
            {
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

                int getDirection() const
                {
                    if (acyclic && !globalSourceSink && localOutS > 0 && localInT > 0)
                        return 1; // s -> t
                    if (acyclic && !globalSourceSink && localOutT > 0 && localInS > 0)
                        return -1; // t -> s
                    return 0;      // no direction ?
                }
            };

            struct NodeDPState
            {
                int outgoingCyclesCount{0};
                node lastCycleNode{nullptr};
                int outgoingSourceSinkCount{0};
                node lastSourceSinkNode{nullptr};
                int outgoingLeakageCount{0};
                node lastLeakageNode{nullptr};
            };

            // pair of dp states for each edge for both directions
            struct EdgeDP
            {
                EdgeDPState down; // value valid in  parent -> child  direction
                EdgeDPState up;   // value valid in  child -> parent direction
            };

            void printAllStates(const ogdf::EdgeArray<EdgeDP> &edge_dp, const ogdf::NodeArray<NodeDPState> &node_dp, const TreeGraph &T)
            {
                auto &C = ctx();

                std::cout << "Edge dp states:" << std::endl;
                for (auto e : T.edges)
                {
                    {
                        EdgeDPState state = edge_dp[e].down;
                        if (state.s && state.t)
                        {
                            std::cout << "Edge " << T.source(e) << " -> " << T.target(e) << ": ";
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
                        if (state.s && state.t)
                        {
                            std::cout << "Edge " << T.target(e) << " -> " << T.source(e) << ": ";
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
                for (node v : T.nodes)
                {
                    std::cout << "Node " << v.index() << ", ";
                    std::cout << "outgoingCyclesCount: " << node_dp[v].outgoingCyclesCount << ", ";
                    std::cout << "outgoingLeakageCount: " << node_dp[v].outgoingLeakageCount << ", ";
                    std::cout << "outgoingSourceSinkCount: " << node_dp[v].outgoingSourceSinkCount << ", ";

                    std::cout << std::endl;
                }
            }

            void printAllEdgeStates(const ogdf::EdgeArray<EdgeDP> &edge_dp, const TreeGraph &T)
            {
                auto &C = ctx();

                std::cout << "Edge dp states:" << std::endl;
                for (auto e : T.edges)
                {
                    {
                        EdgeDPState state = edge_dp[e].down;
                        if (state.s && state.t)
                        {
                            std::cout << "Edge " << T.source(e) << " -> " << T.target(e) << ": ";
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
                        if (state.s && state.t)
                        {
                            std::cout << "Edge " << T.target(e) << " -> " << T.source(e) << ": ";
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

            std::string nodeTypeToString(SPQRTree::NodeType t)
            {
                switch (t)
                {
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
                edge e = nullptr)
            {
                // PROFILE_FUNCTION();
                if (curr == nullptr)
                {
                    curr = spqr.rootNode();
                    parent = curr;
                    dfsSPQR_order(spqr, edge_order, node_order, curr, parent);
                    return;
                }

                // std::cout << "Node " << curr->index() << " is " << nodeTypeToString(spqr.typeOf(curr)) << std::endl;
                node_order.push_back(curr);
                const TreeGraph &T = spqr.tree();
                T.forEachAdj(curr, [&](node child, edge te) {
                    if (child == parent)
                        return;
                    dfsSPQR_order(spqr, edge_order, node_order, child, curr, te);
                });
                if (curr != parent)
                    edge_order.push_back(e);
            }

            // process edge in the direction of parent to child
            // Computing A->B (curr_edge)
            void processEdge(ogdf::edge curr_edge, ogdf::EdgeArray<EdgeDP> &dp, NodeArray<NodeDPState> &node_dp, const CcData &cc, BlockData &blk)
            {
                // PROFILE_FUNCTION();
                auto &C = ctx();

                const ogdf::NodeArray<int> &globIn = C.inDeg;
                const ogdf::NodeArray<int> &globOut = C.outDeg;

                EdgeDPState &state = dp[curr_edge].down;
                EdgeDPState &back_state = dp[curr_edge].up;

                const StaticSPQRTree &spqr = *blk.spqr;
                const TreeGraph &T = spqr.tree();

                ogdf::node A = T.source(curr_edge);
                ogdf::node B = T.target(curr_edge);

                state.localOutS = 0;
                state.localInT = 0;
                state.localOutT = 0;
                state.localInS = 0;

                const Skeleton &skel = spqr.skeleton(B);
                const auto &skelGraph = skel.getGraph();

                // Building new graph with correct orientation of virtual edges
                Graph newGraph;

                NodeArray<node> skelToNew(skelGraph, nullptr);
                for (node v : skelGraph.nodes)
                    skelToNew[v] = newGraph.newNode();
                NodeArray<node> newToSkel(newGraph, nullptr);
                for (node v : skelGraph.nodes)
                    newToSkel[skelToNew[v]] = v;

                {
                    // PROFILE_BLOCK("processNode:: map block to skeleton nodes");
                    for (ogdf::node h : skelGraph.nodes)
                    {
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

                auto mapNewToGlobal = [&](ogdf::node vN) -> ogdf::node
                {
                    if (!vN)
                        return nullptr;

                    ogdf::node vSkel = newToSkel[vN];
                    if (!vSkel)
                        return nullptr;

                    ogdf::node vBlk = skel.original(vSkel);
                    if (!vBlk)
                        return nullptr;

                    ogdf::node vCc = blk.toCc[vBlk];
                    if (!vCc)
                        return nullptr;

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
                auto printDegrees = [&]()
                {
                    for (node vN : newGraph.nodes)
                    {
                        node vG = mapNewToGlobal(vN);

                        // std::cout << C.node2name[vG] << ":    out: " << localOutDeg[vN] << ", in: " << localInDeg[vN] << std::endl;
                    }
                };

                ogdf::node nS, nT;

                for (edge e : skelGraph.edges)
                {
                    node u = skelGraph.source(e);
                    node v = skelGraph.target(e);

                    node nU = skelToNew[u];
                    node nV = skelToNew[v];

                    if (!skel.isVirtual(e))
                    {
                        newGraph.newEdge(nU, nV);
                        localOutDeg[nU]++;
                        localInDeg[nV]++;

                        continue;
                    }

                    auto D = skel.twinTreeNode(e);

                    if (D == A)
                    {
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

                    if (dir == 1)
                    {
                        newGraph.newEdge(nA, nB);
                    }
                    else if (dir == -1)
                    {
                        newGraph.newEdge(nB, nA);
                    }

                    if (nA == nU && nB == nV)
                    {
                        localOutDeg[nA] += child.localOutS;
                        localInDeg[nA] += child.localInS;

                        localOutDeg[nB] += child.localOutT;
                        localInDeg[nB] += child.localInT;
                    }
                    else
                    {
                        localOutDeg[nB] += child.localOutT;
                        localInDeg[nB] += child.localInT;

                        localOutDeg[nA] += child.localOutS;
                        localInDeg[nA] += child.localInS;
                    }

                    state.acyclic &= child.acyclic;
                    state.globalSourceSink |= child.globalSourceSink;
                    state.hasLeakage |= child.hasLeakage;
                }

                // Direct ST/TS computation(only happens in P nodes)
                if (spqr.typeOf(B) == SPQRTree::NodeType::PNode)
                {
                    for (edge e : skelGraph.edges)
                    {
                        if (skel.isVirtual(e))
                            continue;
                        node u = skelGraph.source(e);
                        node v = skelGraph.target(e);

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

                        if (state.s == bU && state.t == bV)
                        {
                            state.directST = true;
                        }
                        else if (state.s == bV && state.t == bU)
                        {
                            state.directTS = true;
                        }
                        else
                        {
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

                for (ogdf::node nV : newGraph.nodes)
                {
                    ogdf::node sV = newToSkel[nV];
                    ogdf::node bV = skel.original(sV);
                    ogdf::node gV = mapNewToGlobal(nV);

                    if (bV == state.s || bV == state.t)
                        continue;

                    if (globIn[gV] != localInDeg[nV] || globOut[gV] != localOutDeg[nV])
                    {
                        state.hasLeakage = true;
                    }

                    if (globIn[gV] == 0 || globOut[gV] == 0)
                    {
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

                if (state.acyclic)
                    state.acyclic &= isAcyclic(newGraph);

                if (!state.acyclic)
                {
                    node_dp[A].outgoingCyclesCount++;
                    node_dp[A].lastCycleNode = B;
                }

                if (state.globalSourceSink)
                {
                    node_dp[A].outgoingSourceSinkCount++;
                    node_dp[A].lastSourceSinkNode = B;
                }

                if (state.hasLeakage)
                {
                    node_dp[A].outgoingLeakageCount++;
                    node_dp[A].lastLeakageNode = B;
                }
            }

            void processNode(node curr_node, EdgeArray<EdgeDP> &edge_dp, NodeArray<NodeDPState> &node_dp, const CcData &cc, BlockData &blk)
            {
                // PROFILE_FUNCTION();
                auto &C = ctx();

                const ogdf::NodeArray<int> &globIn = C.inDeg;
                const ogdf::NodeArray<int> &globOut = C.outDeg;

                ogdf::node A = curr_node;

                const auto &T = blk.spqr->tree();

                NodeDPState curr_state = node_dp[A];

                const StaticSPQRTree &spqr = *blk.spqr;

                const Skeleton &skel = spqr.skeleton(A);
                const auto &skelGraph = skel.getGraph();

                // Building new graph with correct orientation of virtual edges
                Graph newGraph;

                NodeArray<node> skelToNew(skelGraph, nullptr);
                for (node v : skelGraph.nodes)
                    skelToNew[v] = newGraph.newNode();
                NodeArray<node> newToSkel(newGraph, nullptr);
                for (node v : skelGraph.nodes)
                    newToSkel[skelToNew[v]] = v;

                for (ogdf::node h : skelGraph.nodes)
                {
                    ogdf::node vB = skel.original(h);
                    blk.blkToSkel[vB] = h;
                }

                NodeArray<int> localInDeg(newGraph, 0), localOutDeg(newGraph, 0);

                NodeArray<bool> isSourceSink(newGraph, false);
                int localSourceSinkCount = 0;

                NodeArray<bool> isLeaking(newGraph, false);
                int localLeakageCount = 0;

                EdgeArray<bool> isVirtual(newGraph, false);
                EdgeArray<EdgeDPState *> edgeToDp(newGraph, nullptr);
                EdgeArray<EdgeDPState *> edgeToDpR(newGraph, nullptr);
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

                auto mapBlockToNew = [&](ogdf::node bV) -> ogdf::node
                {
                    ogdf::node sV = blk.blkToSkel[bV];
                    ogdf::node nV = skelToNew[sV];
                    return nV;
                };

                auto mapNewToGlobal = [&](ogdf::node vN) -> ogdf::node
                {
                    if (!vN)
                        return nullptr;
                    ogdf::node vSkel = newToSkel[vN];
                    if (!vSkel)
                        return nullptr;
                    ogdf::node vBlk = skel.original(vSkel);
                    if (!vBlk)
                        return nullptr;
                    ogdf::node vCc = blk.toCc[vBlk];
                    if (!vCc)
                        return nullptr;
                    return cc.toOrig[vCc];
                };

                auto printDegrees = [&]()
                {
                    for (node vN : newGraph.nodes)
                    {
                        node vG = mapNewToGlobal(vN);
                    }
                };

                // Building new graph
                {
                    // PROFILE_BLOCK("processNode:: build oriented local graph");
                    for (edge e : skelGraph.edges)
                    {
                        node u = skelGraph.source(e);
                        node v = skelGraph.target(e);

                        node nU = skelToNew[u];
                        node nV = skelToNew[v];

                        if (!skel.isVirtual(e))
                        {
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

                        if (dir == 1 || dir == 0)
                        {
                            newEdge = newGraph.newEdge(nS, nT);

                            isVirtual[newEdge] = true;

                            virtualEdges.push_back(newEdge);

                            edgeToDp[newEdge] = edgeToUpdate;
                            edgeToDpR[newEdge] = child;
                            edgeChild[newEdge] = B;
                        }
                        else if (dir == -1)
                        {
                            newEdge = newGraph.newEdge(nT, nS);

                            isVirtual[newEdge] = true;

                            virtualEdges.push_back(newEdge);

                            edgeToDpR[newEdge] = child;
                            edgeToDp[newEdge] = edgeToUpdate;
                            edgeChild[newEdge] = B;
                        }
                        else
                        {
                            newEdge = newGraph.newEdge(nS, nT);
                            isVirtual[newEdge] = true;

                            virtualEdges.push_back(newEdge);

                            edgeChild[newEdge] = B;
                            edgeToDpR[newEdge] = child;

                            edgeToDp[newEdge] = edgeToUpdate;
                        }

                        if (nS == nU && nT == nV)
                        {
                            localOutDeg[nS] += child->localOutS;
                            localInDeg[nS] += child->localInS;

                            localOutDeg[nT] += child->localOutT;
                            localInDeg[nT] += child->localInT;
                        }
                        else
                        {
                            localOutDeg[nT] += child->localOutT;
                            localInDeg[nT] += child->localInT;

                            localOutDeg[nS] += child->localOutS;
                            localInDeg[nS] += child->localInS;
                        }
                    }
                }

                {
                    // PROFILE_BLOCK("processNode:: mark source/sink and leakage");
                    for (node vN : newGraph.nodes)
                    {
                        node vG = mapNewToGlobal(vN);
                        // node vB = skel.original(newToSkel[vN]);
                        if (globIn[vG] == 0 || globOut[vG] == 0)
                        {
                            localSourceSinkCount++;
                            isSourceSink[vN] = true;
                        }

                        if (globIn[vG] != localInDeg[vN] || globOut[vG] != localOutDeg[vN])
                        {
                            localLeakageCount++;
                            isLeaking[vN] = true;
                        }
                    }
                }

                // calculating ingoing dp states of direct st and ts edges in P node
                if (spqr.typeOf(A) == StaticSPQRTree::NodeType::PNode)
                {
                    // PROFILE_BLOCK("processNode:: P-node direct edge analysis");
                    node pole0Blk = nullptr, pole1Blk = nullptr;
                    {
                        auto it = skelGraph.nodes.begin();
                        if (it != skelGraph.nodes.end())
                            pole0Blk = skel.original(*it++);
                        if (it != skelGraph.nodes.end())
                            pole1Blk = skel.original(*it);
                    }

                    if (!pole0Blk || !pole1Blk)
                        return;

                    node gPole0 = cc.toOrig[blk.toCc[pole0Blk]];
                    node gPole1 = cc.toOrig[blk.toCc[pole1Blk]];

                    int cnt01 = 0, cnt10 = 0;
                    for (edge e : skelGraph.edges)
                    {
                        if (!skel.isVirtual(e))
                        {
                            node uG = mapNewToGlobal(skelToNew[skelGraph.source(e)]);
                            node vG = mapNewToGlobal(skelToNew[skelGraph.target(e)]);
                            if (uG == gPole0 && vG == gPole1)
                                ++cnt01;
                            else if (uG == gPole1 && vG == gPole0)
                                ++cnt10;
                        }
                    }

                    for (edge e : skelGraph.edges)
                    {
                        if (skel.isVirtual(e))
                        {
                            node B = skel.twinTreeNode(e);
                            edge treeE = blk.skel2tree.at(e);

                            SPQRsolve::EdgeDPState &st =
                                (B == blk.parent(A) ? edge_dp[treeE].down
                                                    : edge_dp[treeE].up);

                            if (st.s == pole0Blk && st.t == pole1Blk)
                            {
                                st.directST |= (cnt01 > 0);
                                st.directTS |= (cnt10 > 0);
                            }
                            else if (st.s == pole1Blk && st.t == pole0Blk)
                            {
                                st.directST |= (cnt10 > 0);
                                st.directTS |= (cnt01 > 0);
                            }
                        }
                    }
                }

                // Computing acyclicity
                if (curr_state.outgoingCyclesCount >= 2)
                {
                    // PROFILE_BLOCK("processNode:: acyclicity - multi-outgoing case");
                    for (edge e : virtualEdges)
                    {
                        if (edgeToDp[e]->acyclic)
                        {
                            node_dp[edgeChild[e]].outgoingCyclesCount++;
                            node_dp[edgeChild[e]].lastCycleNode = curr_node;
                        }
                        edgeToDp[e]->acyclic &= false;
                    }
                }
                else if (node_dp[curr_node].outgoingCyclesCount == 1)
                {
                    // PROFILE_BLOCK("processNode:: acyclicity - single-outgoing case");
                    for (edge e : virtualEdges)
                    {
                        if (edgeChild[e] != curr_state.lastCycleNode)
                        {
                            if (edgeToDp[e]->acyclic)
                            {
                                node_dp[edgeChild[e]].outgoingCyclesCount++;
                                node_dp[edgeChild[e]].lastCycleNode = curr_node;
                            }
                            edgeToDp[e]->acyclic &= false;
                        }
                        else
                        {
                            node nU = newGraph.source(e);
                            node nV = newGraph.target(e);
                            auto *st = edgeToDp[e];
                            auto *ts = edgeToDpR[e];
                            auto child = edgeChild[e];
                            bool acyclic = false;

                            newGraph.delEdge(e);
                            acyclic = isAcyclic(newGraph);

                            edge eRest = newGraph.newEdge(nU, nV);
                            isVirtual[eRest] = true;
                            edgeToDp[eRest] = st;
                            edgeToDpR[eRest] = ts;
                            edgeChild[eRest] = child;

                            if (edgeToDp[eRest]->acyclic && !acyclic)
                            {
                                node_dp[edgeChild[eRest]].outgoingCyclesCount++;
                                node_dp[edgeChild[eRest]].lastCycleNode = curr_node;
                            }

                            edgeToDp[eRest]->acyclic &= acyclic;
                        }
                    }
                }
                else
                {
                    // PROFILE_BLOCK("processNode:: acyclicity - FAS baseline");

                    FeedbackArcSet FAS(newGraph);
                    std::vector<edge> fas = FAS.run();
                    // find_feedback_arcs(newGraph, fas, toRemove);

                    EdgeArray<bool> isFas(newGraph, 0);
                    for (edge e : fas)
                        isFas[e] = true;

                    for (edge e : virtualEdges)
                    {

                        if (edgeToDp[e]->acyclic && !isFas[e])
                        {
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
                    // PROFILE_BLOCK("processNode:: compute global source/sink");
                    if (curr_state.outgoingSourceSinkCount >= 2)
                    {
                        // all ingoing have source
                        for (edge e : virtualEdges)
                        {
                            if (!edgeToDp[e]->globalSourceSink)
                            {
                                node_dp[edgeChild[e]].outgoingSourceSinkCount++;
                                node_dp[edgeChild[e]].lastSourceSinkNode = curr_node;
                            }

                            edgeToDp[e]->globalSourceSink |= true;
                        }
                    }
                    else if (curr_state.outgoingSourceSinkCount == 1)
                    {
                        for (edge e : virtualEdges)
                        {
                            // if(!isVirtual[e]) continue;
                            if (edgeChild[e] != curr_state.lastSourceSinkNode)
                            {
                                if (!edgeToDp[e]->globalSourceSink)
                                {
                                    node_dp[edgeChild[e]].outgoingSourceSinkCount++;
                                    node_dp[edgeChild[e]].lastSourceSinkNode = curr_node;
                                }

                                edgeToDp[e]->globalSourceSink |= true;
                            }
                            else
                            {
                                node vN = newGraph.source(e), uN = newGraph.target(e);
                                if ((int)isSourceSink[vN] + (int)isSourceSink[uN] < localSourceSinkCount)
                                {
                                    if (!edgeToDp[e]->globalSourceSink)
                                    {
                                        node_dp[edgeChild[e]].outgoingSourceSinkCount++;
                                        node_dp[edgeChild[e]].lastSourceSinkNode = curr_node;
                                    }

                                    edgeToDp[e]->globalSourceSink |= true;
                                }
                            }
                        }
                    }
                    else
                    {
                        for (edge e : virtualEdges)
                        {
                            // if(!isVirtual[e]) continue;
                            node vN = newGraph.source(e), uN = newGraph.target(e);
                            if ((int)isSourceSink[vN] + (int)isSourceSink[uN] < localSourceSinkCount)
                            {
                                if (!edgeToDp[e]->globalSourceSink)
                                {
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
                    // PROFILE_BLOCK("processNode:: compute leakage");
                    if (curr_state.outgoingLeakageCount >= 2)
                    {
                        for (edge e : virtualEdges)
                        {
                            // if(!isVirtual[e]) continue;

                            if (!edgeToDp[e]->hasLeakage)
                            {
                                node_dp[edgeChild[e]].outgoingLeakageCount++;
                                node_dp[edgeChild[e]].lastLeakageNode = curr_node;
                            }

                            edgeToDp[e]->hasLeakage |= true;
                        }
                    }
                    else if (curr_state.outgoingLeakageCount == 1)
                    {
                        for (edge e : virtualEdges)
                        {
                            // if(!isVirtual[e]) continue;

                            if (edgeChild[e] != curr_state.lastLeakageNode)
                            {
                                if (!edgeToDp[e]->hasLeakage)
                                {
                                    node_dp[edgeChild[e]].outgoingLeakageCount++;
                                    node_dp[edgeChild[e]].lastLeakageNode = curr_node;
                                }
                                edgeToDp[e]->hasLeakage |= true;
                            }
                            else
                            {
                                node vN = newGraph.source(e), uN = newGraph.target(e);
                                if ((int)isLeaking[vN] + (int)isLeaking[uN] < localLeakageCount)
                                {
                                    if (!edgeToDp[e]->hasLeakage)
                                    {
                                        node_dp[edgeChild[e]].outgoingLeakageCount++;
                                        node_dp[edgeChild[e]].lastLeakageNode = curr_node;
                                    }
                                    edgeToDp[e]->hasLeakage |= true;
                                }
                            }
                        }
                    }
                    else
                    {
                        for (edge e : virtualEdges)
                        {
                            // if(!isVirtual[e]) continue;

                            node vN = newGraph.source(e), uN = newGraph.target(e);
                            if ((int)isLeaking[vN] + (int)isLeaking[uN] < localLeakageCount)
                            {
                                if (!edgeToDp[e]->hasLeakage)
                                {
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
                    // PROFILE_BLOCK("processNode:: update DP local degrees at poles");
                    for (edge e : virtualEdges)
                    {
                        // if(!isVirtual[e]) continue;
                        node vN = newGraph.source(e);
                        node uN = newGraph.target(e);

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
                const EdgeArray<EdgeDP> &edge_dp)
            {
                if (blk.spqr->typeOf(A) != SPQRTree::NodeType::PNode)
                    return;

                const auto &T = blk.spqr->tree();
                const Skeleton &skel = blk.spqr->skeleton(A);
                const auto &skelGraph = skel.getGraph();

                node bS, bT;
                {
                    auto it = skelGraph.nodes.begin();
                    if (it != skelGraph.nodes.end())
                        bS = skel.original(*it++);
                    if (it != skelGraph.nodes.end())
                        bT = skel.original(*it);
                }

                int directST = 0, directTS = 0;
                for (auto e : skelGraph.edges)
                {
                    if (skel.isVirtual(e))
                        continue;

                    node a = skel.original(skelGraph.source(e)), b = skel.original(skelGraph.target(e));

                    if (a == bS && b == bT)
                        directST++;
                    else
                        directTS++;
                }

                // printAllEdgeStates(edge_dp, blk.spqr->tree());

                for (int q = 0; q < 2; q++)
                {
                    // s -> t

                    // std::cout << "s: " << ctx().node2name[s] << ", t: " << ctx().node2name[t] << std::endl;
                    std::vector<const EdgeDPState *> goodS, goodT;

                    int localOutSSum = directST, localInTSum = directST;

                    // std::cout << " at " << A << std::endl;

                    T.forEachAdj(A, [&](node /*other*/, edge e) {
                        // std::cout << T.source(e) << " -> " << T.target(e) << std::endl;
                        auto &state = (T.source(e) == A ? edge_dp[e].down : edge_dp[e].up);
                        // directST = (state.s == s ? state.directST : state.directTS);
                        // directTS = (state.s == s ? state.directTS : state.directST);

                        int localOutS = (state.s == bS ? state.localOutS : state.localOutT), localInT = (state.t == bT ? state.localInT : state.localInS);

                        localOutSSum += localOutS;
                        localInTSum += localInT;
                        // std::cout << other << " has outS" <<  localOutS << " and outT " << localInT << std::endl;

                        if (localOutS > 0)
                        {
                            // std::cout << "PUSHING TO GOODs" << (T.source(e) == A ? T.target(e): T.source(e)) << std::endl;
                            goodS.push_back(&state);
                        }

                        if (localInT > 0)
                        {
                            // std::cout << "PUSHING TO GOODt" << (T.source(e) == A ? T.target(e): T.source(e)) << std::endl;
                            goodT.push_back(&state);
                        }
                    });

                    // if(q == 1) std::swap(goodS, goodT);
                    // std::cout << "directST: " << directST << ", directTS: " << directTS << std::endl;

                    // std::cout << ctx().node2name[cc.toOrig[blk.toCc[s]]] << ", " << ctx().node2name[cc.toOrig[blk.toCc[t]]] << " has s:" << goodS.size() << " and t:" << goodT.size() << std::endl;
                    bool good = true;
                    for (auto &state : goodS)
                    {
                        if ((state->s == bS && state->localInS > 0) || (state->s == bT && state->localInT > 0))
                        {
                            // std::cout << "BAD 1" << std::endl;
                            good = false;
                        }

                        good &= state->acyclic;
                        good &= !state->globalSourceSink;
                        good &= !state->hasLeakage;
                    }

                    for (auto &state : goodT)
                    {
                        if ((state->t == bT && state->localOutT > 0) || (state->t == bS && state->localOutS > 0))
                        {
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

                    if (good)
                    {
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
                           bool additionalCheck)
            {
                node S = swap ? blk.toOrig[curr.t] : blk.toOrig[curr.s];
                node T = swap ? blk.toOrig[curr.s] : blk.toOrig[curr.t];

                // std::cout << ctx().node2name[S] << " " << ctx().node2name[T] << " " << (additionalCheck) << std::endl;

                /* take the counts from the current direction … */

                int outS = swap ? curr.localOutT : curr.localOutS;
                int outT = swap ? curr.localOutS : curr.localOutT;
                int inS = swap ? curr.localInT : curr.localInS;
                int inT = swap ? curr.localInS : curr.localInT;

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

                if (back.directST)
                {
                    // std::cout << " added because back.directST" << std::endl;
                    if (!swap)
                    {
                        outS++;
                        inT++;
                    }
                    else
                    {
                        inS++;
                        outT++;
                    }
                }
                if (back.directTS)
                {
                    // std::cout << " added because back.directTS" << std::endl;
                    if (!swap)
                    {
                        inS++;
                        outT++;
                    }
                    else
                    {
                        outS++;
                        inT++;
                    }
                }

                // std::cout << "after" << std::endl;
                // std::cout << outS << " " << inS << " | " << outT << " " << inT << std::endl;

                bool backGood = true;

                if (back.s == curr.s && back.t == curr.t)
                {
                    backGood &= (!back.directTS);
                }
                else if (back.s == curr.t && back.t == curr.s)
                {
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
                    ctx().inDeg[T] == inT &&
                    !ctx().isEntry[S] &&
                    !ctx().isExit[T])
                {
                    if (additionalCheck)
                    {
                        if (!swap)
                        {
                            if (back.directST)
                                addSuperbubble(S, T);
                        }
                        else
                        {
                            if (back.directTS)
                                addSuperbubble(S, T);
                        }
                    }
                    else
                    {
                        addSuperbubble(S, T);
                    }
                }
            }

            void collectSuperbubbles(const CcData &cc, BlockData &blk, EdgeArray<EdgeDP> &edge_dp, NodeArray<NodeDPState> &node_dp)
            {
                // PROFILE_FUNCTION();
                const auto &T = blk.spqr->tree();
                // printAllStates(edge_dp, node_dp, T);

                for (edge e : T.edges)
                {
                    // std::cout << "CHECKING FOR " << T.source(e) << " " << T.target(e) << std::endl;
                    const EdgeDPState &down = edge_dp[e].down;
                    const EdgeDPState &up = edge_dp[e].up;

                    // if(blk.spqr->typeOf(T.target(e)) != SPQRTree::NodeType::SNode) {
                    //     std::cout << "DOWN" << std::endl;
                    bool additionalCheck;

                    additionalCheck = (blk.spqr->typeOf(T.source(e)) == SPQRTree::NodeType::PNode && blk.spqr->typeOf(T.target(e)) == SPQRTree::NodeType::SNode);
                    tryBubble(down, up, blk, cc, false, additionalCheck);
                    tryBubble(down, up, blk, cc, true, additionalCheck);
                    // }

                    // if(blk.spqr->typeOf(T.source(e)) != SPQRTree::NodeType::SNode) {
                    // std::cout << "UP" << std::endl;
                    additionalCheck = (blk.spqr->typeOf(T.target(e)) == SPQRTree::NodeType::PNode && blk.spqr->typeOf(T.source(e)) == SPQRTree::NodeType::SNode);

                    tryBubble(up, down, blk, cc, false, additionalCheck);
                    tryBubble(up, down, blk, cc, true, additionalCheck);
                    // }

                    blk.isAcycic &= (down.acyclic && up.acyclic);
                }
                for (node v : T.nodes)
                {
                    tryBubblePNodeGrouping(v, cc, blk, edge_dp);
                }
            }

        }

        void checkBlockByCutVertices(const BlockData &blk, const CcData &cc)
        {

            if (!isAcyclic(*blk.Gblk))
            {
                return;
            }

            auto &C = ctx();
            const Graph &G = *blk.Gblk;

            node src = nullptr, snk = nullptr;

            for (node v : G.nodes)
            {
                node vG = blk.toOrig[v];
                int inL = blk.inDeg[v], outL = blk.outDeg[v];
                int inG = C.inDeg[vG], outG = C.outDeg[vG];

                bool isSrc = (inL == 0 && outL == outG);
                bool isSnk = (outL == 0 && inL == inG);

                if (isSrc ^ isSnk)
                {
                    if (isSrc)
                    {
                        if (src)
                            return;
                        src = v;
                    }
                    else
                    {
                        if (snk)
                            return;
                        snk = v;
                    }
                }
                else if (!(inL == inG && outL == outG))
                {
                    return;
                }
            }

            if (!src || !snk)
            {
                return;
            }

            NodeArray<bool> vis(G, false);
            std::stack<node> S;
            vis[src] = true;
            S.push(src);
            bool reach = false;
            while (!S.empty() && !reach)
            {
                node u = S.top();
                S.pop();
                G.forEachAdj(u, [&](node v, edge e) {
                    if (G.source(e) != u)  // only outgoing edges
                        return;
                    if (!vis[v])
                    {
                        if (v == snk)
                        {
                            reach = true;
                            return;
                        }
                        vis[v] = true;
                        S.push(v);
                    }
                });
            }
            if (!reach)
            {
                return;
            }

            node srcG = blk.toOrig[src], snkG = blk.toOrig[snk];
            addSuperbubble(srcG, snkG);
        }

        void solveSPQR(BlockData &blk, const CcData &cc)
        {

            if (!blk.spqr || blk.Gblk->numberOfNodes() < 3)
            {
                return;
            }

            const auto &T = blk.spqr->tree();

            EdgeArray<SPQRsolve::EdgeDP> dp(T);
            NodeArray<SPQRsolve::NodeDPState> node_dp(T);

            std::vector<ogdf::node> nodeOrder;
            std::vector<ogdf::edge> edgeOrder;

            SPQRsolve::dfsSPQR_order(*blk.spqr, edgeOrder, nodeOrder);

            blk.blkToSkel.init(*blk.Gblk, nullptr);

            for (auto e : edgeOrder)
            {
                SPQRsolve::processEdge(e, dp, node_dp, cc, blk);
            }

            for (auto v : nodeOrder)
            {
                SPQRsolve::processNode(v, dp, node_dp, cc, blk);
            }

            SPQRsolve::collectSuperbubbles(cc, blk, dp, node_dp);
        }

        void findMiniSuperbubbles()
        {
            MARK_SCOPE_MEM("sb/findMini");

            auto &C = ctx();

            logger::info("Finding mini-superbubbles..");


            std::vector<ogdf::edge> edges_vec;
            edges_vec.reserve(C.G.numberOfEdges());
            for (auto e : C.G.edges) edges_vec.push_back(e);

            size_t numThreads = std::thread::hardware_concurrency();
            numThreads = std::min({(size_t)C.threads, numThreads});
            if (numThreads == 0) numThreads = 1;

            auto check_and_emit = [&](ogdf::edge e) {
                auto a = C.G.source(e);
                auto b = C.G.target(e);
                if (C.G.outdeg(a) == 1 && C.G.indeg(b) == 1)
                {
                    bool ok = true;
                    C.G.forEachAdj(b, [&](node /*other*/, edge e2) {
                        auto src = C.G.source(e2);
                        auto tgt = C.G.target(e2);
                        if (src == b && tgt == a)
                        {
                            ok = false;
                        }
                    });
                    if (ok)
                    {
                        addSuperbubble(a, b);
                    }
                }
            };

            if (numThreads <= 1)
            {
                for (auto e : edges_vec) check_and_emit(e);
            }
            else
            {
                const size_t n = edges_vec.size();
                std::vector<std::vector<std::pair<ogdf::node, ogdf::node>>> results(numThreads);

                std::vector<std::thread> threads;
                threads.reserve(numThreads);
                for (size_t tid = 0; tid < numThreads; ++tid)
                {
                    const size_t start = (n * tid) / numThreads;
                    const size_t end = (n * (tid + 1)) / numThreads;
                    threads.emplace_back([&, tid, start, end]() {
                        auto &local = results[tid];
                        local.reserve(std::max<size_t>(16, (end - start) / 64));
                        tls_superbubble_collector = &local;
                        for (size_t i = start; i < end; ++i)
                        {
                            check_and_emit(edges_vec[i]);
                        }
                        tls_superbubble_collector = nullptr;
                    });
                }
                for (auto &t : threads) t.join();

                for (auto &local : results)
                {
                    for (auto &p : local) tryCommitSuperbubble(p.first, p.second);
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

        static void buildBlockDataParallel(const CcData &cc, BlockData &blk)
        {
            {
                blk.Gblk = std::make_unique<Graph>();

                blk.toOrig.init(*blk.Gblk, nullptr);
                blk.toCc.init(*blk.Gblk, nullptr);
                blk.inDeg.init(*blk.Gblk, 0);
                blk.outDeg.init(*blk.Gblk, 0);

                std::unordered_set<node> verts;
                for (edge hE : cc.bc->hEdges(blk.bNode))
                {
                    edge eC = cc.bc->original(hE);
                    verts.insert(cc.Gcc->source(eC));
                    verts.insert(cc.Gcc->target(eC));
                }

                std::unordered_map<node, node> cc_to_blk;
                cc_to_blk.reserve(verts.size());

                for (node vCc : verts)
                {
                    node vB = blk.Gblk->newNode();
                    cc_to_blk[vCc] = vB;
                    blk.toCc[vB] = vCc;
                    node vG = cc.toOrig[vCc];
                    blk.toOrig[vB] = vG;
                }

                for (edge hE : cc.bc->hEdges(blk.bNode))
                {
                    edge eCc = cc.bc->original(hE);
                    auto srcIt = cc_to_blk.find(cc.Gcc->source(eCc));
                    auto tgtIt = cc_to_blk.find(cc.Gcc->target(eCc));
                    if (srcIt != cc_to_blk.end() && tgtIt != cc_to_blk.end())
                    {
                        edge e = blk.Gblk->newEdge(srcIt->second, tgtIt->second);
                        blk.outDeg[blk.Gblk->source(e)]++;
                        blk.inDeg[blk.Gblk->target(e)]++;
                    }
                }

                blk.globIn.init(*blk.Gblk, 0);
                blk.globOut.init(*blk.Gblk, 0);
                for (node vB : blk.Gblk->nodes)
                {
                    node vG = blk.toOrig[vB];
                    blk.globIn[vB] = ctx().inDeg[vG];
                    blk.globOut[vB] = ctx().outDeg[vG];
                }
            }

            if (blk.Gblk->numberOfNodes() >= 3)
            {
                {
                    blk.spqr = std::make_unique<StaticSPQRTree>(*blk.Gblk);
                }
                const auto &T = blk.spqr->tree();
                blk.skel2tree.reserve(2 * T.edges.size());
                blk.parent.init(T, nullptr);

                node root = blk.spqr->rootNode();
                blk.parent[root] = root;

                for (edge te : T.edges)
                {
                    node u = T.source(te);
                    node v = T.target(te);
                    blk.parent[v] = u;

                    if (auto eSrc = blk.spqr->skeletonEdgeSrc(te))
                    {
                        blk.skel2tree[eSrc] = te;
                    }
                    if (auto eTgt = blk.spqr->skeletonEdgeTgt(te))
                    {
                        blk.skel2tree[eTgt] = te;
                    }
                }
            }
        }

        struct WorkItem
        {
            CcData *cc;
            // BlockData* blockData;
            node bNode;
        };

        struct BlockPrep
        {
            CcData *cc;
            node bNode;
        };

        struct ThreadBcTreeArgs
        {
            size_t tid;
            size_t numThreads;
            int nCC;
            std::atomic<size_t> *nextIndex;
            std::vector<std::unique_ptr<CcData>> *components;
            std::vector<std::vector<BlockPrep>> *perThreadPreps;
        };

        void *worker_bcTree(void *arg)
        {
            std::unique_ptr<ThreadBcTreeArgs> targs(static_cast<ThreadBcTreeArgs *>(arg));
            size_t tid = targs->tid;
            size_t numThreads = targs->numThreads;
            int nCC = targs->nCC;
            std::atomic<size_t> *nextIndex = targs->nextIndex;
            std::vector<std::unique_ptr<CcData>> *components = targs->components;
            std::vector<BlockPrep> &myPreps = (*targs->perThreadPreps)[tid];

            size_t chunkSize = 1;
            size_t processed = 0;

            while (true)
            {
                size_t startIndex, endIndex;
                {
                    startIndex = nextIndex->fetch_add(chunkSize, std::memory_order_relaxed);
                    if (startIndex >= static_cast<size_t>(nCC))
                        break;
                    endIndex = std::min(startIndex + chunkSize, static_cast<size_t>(nCC));
                }

                auto chunkStart = std::chrono::high_resolution_clock::now();

                for (size_t cid = startIndex; cid < endIndex; ++cid)
                {
                    CcData *cc = (*components)[cid].get();

                    if (!cc) continue;

                    {
                        cc->bc = std::make_unique<BCTree>(*cc->Gcc);
                    }

                    std::vector<BlockPrep> localPreps;
                    {
                        for (node v : cc->bc->bcTree().nodes)
                        {
                            if (cc->bc->typeOfBNode(v) == BCTree::BNodeType::BComp)
                            {
                                localPreps.push_back({cc, v});
                            }
                        }
                    }

                    myPreps.insert(myPreps.end(),
                                   std::make_move_iterator(localPreps.begin()),
                                   std::make_move_iterator(localPreps.end()));

                    ++processed;
                }

                auto chunkEnd = std::chrono::high_resolution_clock::now();
                auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);

                if (chunkDuration.count() < 1000)
                {
                    chunkSize = std::min(chunkSize * 2, static_cast<size_t>(nCC / numThreads));
                }
                else if (chunkDuration.count() > 5000)
                {
                    chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                }
            }

            std::cout << "Thread " << tid << " built " << processed << " components (bc trees)" << std::endl;
            return nullptr;
        }

        struct ThreadBlockBuildArgs
        {
            size_t tid;
            size_t numThreads;
            size_t nBlocks;
            std::atomic<size_t> *nextIndex;
            std::vector<BlockPrep> *blockPreps;
            std::vector<std::unique_ptr<BlockData>> *allBlockData;
        };

        static void *worker_buildBlockData(void *arg)
        {
            std::unique_ptr<ThreadBlockBuildArgs> targs(static_cast<ThreadBlockBuildArgs *>(arg));
            size_t tid = targs->tid;
            size_t numThreads = targs->numThreads;
            size_t nBlocks = targs->nBlocks;
            std::atomic<size_t> *nextIndex = targs->nextIndex;
            auto *blockPreps = targs->blockPreps;
            auto *allBlockData = targs->allBlockData;
            size_t chunkSize = 1;
            size_t processed = 0;
            while (true)
            {
                size_t startIndex, endIndex;
                {
                    startIndex = nextIndex->fetch_add(chunkSize, std::memory_order_relaxed);
                    if (startIndex >= nBlocks)
                        break;
                    endIndex = std::min(startIndex + chunkSize, nBlocks);
                }
                auto chunkStart = std::chrono::high_resolution_clock::now();
                for (size_t i = startIndex; i < endIndex; ++i)
                {
                    const BlockPrep &bp = (*blockPreps)[i];
                    (*allBlockData)[i] = std::make_unique<BlockData>();
                    (*allBlockData)[i]->bNode = bp.bNode;
                    buildBlockDataParallel(*bp.cc, *(*allBlockData)[i]);
                    ++processed;
                }
                auto chunkEnd = std::chrono::high_resolution_clock::now();
                auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);
                if (chunkDuration.count() < 100)
                {
                    size_t maxPerThread = std::max<size_t>(1, nBlocks / std::max<size_t>(numThreads, 1));
                    chunkSize = std::min(chunkSize * 2, maxPerThread);
                }
                else if (chunkDuration.count() > 2000)
                {
                    chunkSize = std::max<size_t>(1, chunkSize / 2);
                }
            }
            std::cout << "Thread " << tid << " built " << processed << " BlockData objects" << std::endl;
            return nullptr;
        }

        struct ThreadProcessArgs
        {
            size_t tid;
            size_t numThreads;
            size_t nItems;
            std::atomic<size_t> *nextIndex;
            std::vector<WorkItem> *workItems;
            std::vector<std::unique_ptr<BlockData>> *allBlockData;
            std::vector<std::vector<std::pair<ogdf::node, ogdf::node>>> *blockResults;
        };

        static void *worker_processBlocks(void *arg)
        {
            std::unique_ptr<ThreadProcessArgs> targs(static_cast<ThreadProcessArgs *>(arg));
            std::atomic<size_t> *nextIndex = targs->nextIndex;
            auto &items = *targs->workItems;
            auto &allBlocks = *targs->allBlockData;
            auto &results = *targs->blockResults;
            const size_t n = targs->nItems;
            while (true)
            {
                size_t i;
                {
                    i = nextIndex->fetch_add(1, std::memory_order_relaxed);
                    if (i >= n)
                        break;
                }

                const WorkItem &w = items[i];

                BlockData *blk = allBlocks[i].get();
                if (!blk)
                {
                    results[i] = {};
                    continue;
                }

                std::vector<std::pair<ogdf::node, ogdf::node>> local;
                tls_superbubble_collector = &local;

                if (blk->Gblk && blk->Gblk->numberOfNodes() >= 3)
                {
                    solveSPQR(*blk, *w.cc);
                }
                checkBlockByCutVertices(*blk, *w.cc);

                tls_superbubble_collector = nullptr;
                results[i] = std::move(local);
                allBlocks[i].reset();
            }
            return nullptr;
        }

        void solveStreaming()
        {
            auto &C = ctx();
            Graph &G = C.G;

            std::vector<WorkItem> workItems;

            std::vector<std::unique_ptr<CcData>> components;
            std::vector<std::unique_ptr<BlockData>> allBlockData;

            {
                // PROFILE_BLOCK("solve:: prepare");

                NodeArray<int> compIdx(G);
                int nCC;
                {
                    MARK_SCOPE_MEM("sb/phase/ComputeCC");
                    // PROFILE_BLOCK("solveStreaming:: ComputeCC");
                    nCC = connectedComponents(G, compIdx);
                }

                components.resize(nCC);

                std::vector<std::vector<node>> bucket(nCC);
                {
                    MARK_SCOPE_MEM("sb/phase/BucketNodes");
                    // PROFILE_BLOCK("solveStreaming:: bucket nodes");
                    for (node v : G.nodes)
                    {
                        bucket[compIdx[v]].push_back(v);
                    }
                }

                std::vector<std::vector<edge>> edgeBuckets(nCC);

                {
                    MARK_SCOPE_MEM("sb/phase/BucketEdges");
                    // PROFILE_BLOCK("solveStreaming:: bucket edges");
                    for (edge e : G.edges)
                    {
                        edgeBuckets[compIdx[G.source(e)]].push_back(e);
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

                        std::atomic<size_t> nextIndex{0};

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            workers.emplace_back([&, tid]()
                                                 {
                                size_t chunkSize = std::max<size_t>(1, nCC / numThreads);
                                size_t processed = 0;
                                while (true) {
                                    size_t startIndex, endIndex;
                                    {
                                        startIndex = nextIndex.fetch_add(chunkSize, std::memory_order_relaxed);
                                        if (startIndex >= static_cast<size_t>(nCC)) break;
                                        endIndex = std::min(startIndex + chunkSize, static_cast<size_t>(nCC));
                                    }

                                    for (size_t ci = startIndex; ci < endIndex; ++ci) {
                                        int cid = static_cast<int>(ci);
                                        if (edgeBuckets[cid].size() < bucket[cid].size()) {
                                            continue;
                                        }

                                        auto cc = std::make_unique<CcData>();

                                        {
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
                                                cc->Gcc->newEdge(orig_to_cc_local[G.source(e)], orig_to_cc_local[G.target(e)]);
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
                                std::cout << "Thread " << tid << " built " << processed << " components (Gcc)" << std::endl; });
                        }

                        for (auto &t : workers)
                            t.join();
                    }

                    {
                        MARK_SCOPE_MEM("sb/phase/BCtrees");

                        size_t numThreads = std::thread::hardware_concurrency();
                        numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});

                        std::vector<pthread_t> threads(numThreads);

                        std::atomic<size_t> nextIndex{0};
                        std::vector<std::vector<BlockPrep>> perThreadPreps(numThreads);

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            pthread_attr_t attr;
                            pthread_attr_init(&attr);

                            // size_t stackSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;
                            size_t stackSize = C.stackSize;
                            if (pthread_attr_setstacksize(&attr, stackSize) != 0)
                            {
                                std::cout << "[Error] pthread_attr_setstacksize" << std::endl;
                            }

                            ThreadBcTreeArgs *args = new ThreadBcTreeArgs{
                                tid,
                                numThreads,
                                nCC,
                                &nextIndex,
                                &components,
                                &perThreadPreps};

                            int ret = pthread_create(&threads[tid], &attr, worker_bcTree, args);
                            if (ret != 0)
                            {
                                std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                                delete args;
                            }

                            pthread_attr_destroy(&attr);
                        }

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            pthread_join(threads[tid], nullptr);
                        }

                        // Flatten per-thread preps into one vector (sequential, post-join)
                        size_t total = 0;
                        for (auto &tp : perThreadPreps) total += tp.size();
                        blockPreps.reserve(total);
                        for (auto &tp : perThreadPreps)
                        {
                            blockPreps.insert(blockPreps.end(),
                                              std::make_move_iterator(tp.begin()),
                                              std::make_move_iterator(tp.end()));
                        }
                    }

                    allBlockData.resize(blockPreps.size());

                    {
                        MARK_SCOPE_MEM("sb/phase/BlockDataBuildAll");

                        size_t numThreads2 = std::thread::hardware_concurrency();
                        numThreads2 = std::min({(size_t)C.threads, (size_t)blockPreps.size(), numThreads2});
                        std::vector<pthread_t> threads2(numThreads2);

                        std::atomic<size_t> nextIndex2{0};

                        for (size_t tid = 0; tid < numThreads2; ++tid)
                        {
                            pthread_attr_t attr;
                            pthread_attr_init(&attr);

                            // size_t stackSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;
                            size_t stackSize = C.stackSize;

                            if (pthread_attr_setstacksize(&attr, stackSize) != 0)
                            {
                                std::cout << "[Error] pthread_attr_setstacksize" << std::endl;
                            }

                            ThreadBlockBuildArgs *args = new ThreadBlockBuildArgs{
                                tid,
                                numThreads2,
                                blockPreps.size(),
                                &nextIndex2,
                                                                &blockPreps,
                                &allBlockData};

                            int ret = pthread_create(&threads2[tid], &attr, worker_buildBlockData, args);
                            if (ret != 0)
                            {
                                std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                                delete args;
                            }

                            pthread_attr_destroy(&attr);
                        }

                        for (size_t tid = 0; tid < numThreads2; ++tid)
                        {
                            pthread_join(threads2[tid], nullptr);
                        }
                    }

                    workItems.reserve(allBlockData.size());
                    for (size_t i = 0; i < allBlockData.size(); ++i)
                    {
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
                if (numThreads == 0)
                    numThreads = 1;

                std::vector<pthread_t> threads(numThreads);
                std::atomic<size_t> nextIndex{0};

                for (size_t tid = 0; tid < numThreads; ++tid)
                {
                    pthread_attr_t attr;
                    pthread_attr_init(&attr);
                    // size_t stackSize = 1024ULL * 1024ULL * 1024ULL * 20ULL;
                    size_t stackSize = C.stackSize;

                    pthread_attr_setstacksize(&attr, stackSize);

                    ThreadProcessArgs *args = new ThreadProcessArgs{
                        tid,
                        numThreads,
                        workItems.size(),
                        &nextIndex,
                                                &workItems,
                        &allBlockData,
                        &blockResults};

                    int ret = pthread_create(&threads[tid], &attr, worker_processBlocks, args);
                    if (ret != 0)
                    {
                        std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                        delete args;
                    }
                    pthread_attr_destroy(&attr);
                }

                for (size_t tid = 0; tid < numThreads; ++tid)
                {
                    pthread_join(threads[tid], nullptr);
                }

                for (const auto &candidates : blockResults)
                {
                    for (const auto &p : candidates)
                    {
                        tryCommitSuperbubble(p.first, p.second);
                    }
                }
            }
        }

        void solve()
        {
            TIME_BLOCK("Finding superbubbles in blocks");
            findMiniSuperbubbles();
            solveStreaming();
        }
    }

    namespace snarls
    {
        static inline uint64_t nowMicros()
        {
            using namespace std::chrono;
            return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
        }

        static size_t currentRSSBytes()
        {
            #if defined(__linux__)
                        long rssPages = 0;
                        FILE *f = std::fopen("/proc/self/statm", "r");
                        if (f)
                        {
                            if (std::fscanf(f, "%*s%ld", &rssPages) != 1)
                            {
                                rssPages = 0;
                            }
                            std::fclose(f);
                        }
                        long pageSize = sysconf(_SC_PAGESIZE);
                        if (pageSize <= 0)
                            pageSize = 4096;
                        if (rssPages < 0)
                            rssPages = 0;
                        return static_cast<size_t>(rssPages) * static_cast<size_t>(pageSize);
            #elif defined(__APPLE__)
                        mach_task_basic_info info;
                        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
                        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
                        {
                            return 0;
                        }
                        return static_cast<size_t>(info.resident_size);
            #elif defined(_WIN32)
                        PROCESS_MEMORY_COUNTERS pmc;
                        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
                        {
                            return static_cast<size_t>(pmc.WorkingSetSize);
                        }
                        return 0;
            #else
                        return 0;
            #endif
        }

        struct PhaseStats
        {
            std::atomic<uint64_t> elapsed_us{0};
            std::atomic<size_t> peak_rss{0};
            std::atomic<size_t> start_rss{0};
        };

        static PhaseStats g_stats_io;
        static PhaseStats g_stats_build;
        static PhaseStats g_stats_logic;

        class PhaseSampler
        {
        public:
            explicit PhaseSampler(PhaseStats &stats, uint32_t period_us = 1000)
                : stats_(stats), period_us_(period_us), stop_(false)
            {
                stats_.start_rss.store(currentRSSBytes(), std::memory_order_relaxed);
                start_us_ = nowMicros();
                sampler_ = std::thread([this]()
                                       { this->run(); });
            }
            ~PhaseSampler()
            {
                stop_ = true;
                if (sampler_.joinable())
                    sampler_.join();
                uint64_t dur = nowMicros() - start_us_;
                stats_.elapsed_us.store(dur, std::memory_order_relaxed);
            }

        private:
            void run()
            {
                size_t local_peak = 0;
                while (!stop_)
                {
                    size_t rss = currentRSSBytes();
                    if (rss > local_peak)
                        local_peak = rss;
                    std::this_thread::sleep_for(std::chrono::microseconds(period_us_));
                }

                size_t rss = currentRSSBytes();
                if (rss > local_peak)
                    local_peak = rss;

                size_t prev = stats_.peak_rss.load(std::memory_order_relaxed);
                while (local_peak > prev &&
                       !stats_.peak_rss.compare_exchange_weak(prev, local_peak, std::memory_order_relaxed))
                {
                }
            }

            PhaseStats &stats_;
            uint32_t period_us_;
            std::atomic<bool> stop_;
            std::thread sampler_;
            uint64_t start_us_{0};
        };

        namespace
        {
            constexpr size_t kMinThreadStackSize = 1024 * 1024 * 1024; // 8 MiB
            struct StrPairHash
            {
                size_t operator()(const std::pair<std::string, std::string> &p) const noexcept
                {
                    std::hash<std::string> h;
                    size_t seed = h(p.first);
                    seed ^= h(p.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    return seed;
                }
            };


            thread_local std::unordered_set<std::pair<std::string, std::string>, StrPairHash>
                tls_spqr_seen_endpoint_pairs;

            inline void canonicalize_pair(std::vector<std::string> &v)
            {
                if (v.size() == 2 && v[1] < v[0])
                {
                    std::swap(v[0], v[1]);
                }
                else if (v.size() > 2)
                {
                    std::sort(v.begin(), v.end());
                }
            }

            thread_local std::vector<std::vector<std::string>> *tls_snarl_buffer = nullptr;

            static std::mutex g_snarls_mtx;

            inline void flushThreadLocalSnarls(std::vector<std::vector<std::string>> &local)
            {
                auto &C = ctx();
                std::lock_guard<std::mutex> lk(g_snarls_mtx);
                for (auto &s : local)
                {
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

            struct pair_hash
            {
                size_t operator()(const std::pair<std::string, std::string> &p) const noexcept
                {
                    auto h1 = std::hash<std::string>{}(p.first);
                    auto h2 = std::hash<std::string>{}(p.second);
                    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
                }
            };

            std::unordered_set<std::pair<std::string, std::string>, pair_hash> tls_snarls_collector;

            std::atomic<uint64_t> g_cnt_cut{0}, g_cnt_S{0}, g_cnt_P{0}, g_cnt_RR{0}, g_cnt_E{0};

            inline void print_snarl_type_counters()
            {
                std::cout << "[SNARLS] by type: "
                          << "CUT=" << g_cnt_cut.load()
                          << " S=" << g_cnt_S.load()
                          << " P=" << g_cnt_P.load()
                          << " RR=" << g_cnt_RR.load()
                          << " E=" << g_cnt_E.load()
                          << std::endl;
            }

        }

        static void tryCommitSnarl(std::vector<std::string> s)
        {
            auto &C = ctx();
            // PROFILE_BLOCK("tryCommitSnarl");

            // if(std::count(s[0].begin(), s[0].end(), ':') == 0) {
            std::lock_guard<std::mutex> lk(g_snarls_mtx);

            snarlsFound += s.size() * (s.size() - 1) / 2;
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

        void addSnarl(std::vector<std::string> s)
        {
            // if (tls_snarls_collector) {
            //     tls_superbubble_collector->emplace_back(source, sink);
            //         return;
            //     }
            // Otherwise, commit directly to global state (sequential behavior)
            // tryCommitSnarl(source, sink);
            if (tls_snarl_buffer)
            {
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

        inline void addSnarlTagged(const char *tag, std::vector<std::string> s)
        {
            // Comptage par type (inchangé)
            if (tag)
            {
                if (std::strcmp(tag, "CUT") == 0)
                    g_cnt_cut++;
                else if (std::strcmp(tag, "S") == 0)
                    g_cnt_S++;
                else if (std::strcmp(tag, "P") == 0)
                    g_cnt_P++;
                else if (std::strcmp(tag, "RR") == 0)
                    g_cnt_RR++;
                else if (std::strcmp(tag, "E") == 0)
                    g_cnt_E++;
            }

            canonicalize_pair(s);

            addSnarl(std::move(s));
        }

        struct BlockData
        {
            std::unique_ptr<ogdf::Graph> Gblk;
            ogdf::NodeArray<ogdf::node> toCc;
            ogdf::NodeArray<ogdf::node> nodeToOrig;
            ogdf::EdgeArray<ogdf::edge> edgeToOrig;

            std::unique_ptr<ogdf::StaticSPQRTree> spqr;
            ogdf::NodeArray<ogdf::node> blkToSkel;

            std::unordered_map<ogdf::edge, ogdf::edge> skel2tree;
            ogdf::NodeArray<ogdf::node> parent;

            ogdf::node bNode{nullptr};

            bool isAcycic{true};

            ogdf::NodeArray<int> blkDegPlus;
            ogdf::NodeArray<int> blkDegMinus;

            BlockData() {}
        };

        struct CcData
        {
            std::unique_ptr<ogdf::Graph> Gcc;
            ogdf::NodeArray<ogdf::node> nodeToOrig;
            ogdf::EdgeArray<ogdf::edge> edgeToOrig;

            ogdf::NodeArray<bool> isTip;

            ogdf::NodeArray<int> degPlus, degMinus;

            ogdf::NodeArray<bool> isCutNode;
            ogdf::NodeArray<bool> isGoodCutNode;

            ogdf::NodeArray<ogdf::node> lastBad;
            ogdf::NodeArray<int> badCutCount;   

            ogdf::EdgeArray<ogdf::edge> auxToOriginal;
            // ogdf::NodeArray<std::array<std::vector<ogdf::node>, 3>> cutToBlocks; // 0-all -, 1 - all +, 2 - mixed

            // ogdf::NodeArray<ogdf::node> toCopy;
            // ogdf::NodeArray<ogdf::node> toBlk;

            std::unique_ptr<ogdf::BCTree> bc;
            std::vector<BlockData> blocks;
        };

        EdgePartType getNodeEdgeType(ogdf::node v, ogdf::edge e)
        {
            auto &C = ctx();
            if (C.G.source(e) == v)
            {
                return C._edge2types(e).first;
            }
            else if (C.G.target(e) == v)
            {
                return C._edge2types(e).second;
            }
            else
            {
                return EdgePartType::NONE;
            }
        }

        void getOutgoingEdgesInBlock(const CcData &cc,
                                     ogdf::node uG, 
                                     ogdf::node vB, 
                                     EdgePartType type,
                                     std::vector<ogdf::edge> &outEdges)
        {
            outEdges.clear();

            for (ogdf::edge eCc : cc.bc->hEdges(vB))
            {
                if (cc.Gcc->source(eCc) != uG && cc.Gcc->target(eCc) != uG)
                    continue;

                ogdf::edge eG = cc.edgeToOrig[eCc];
                if (!eG) continue;

                auto outType = getNodeEdgeType(cc.nodeToOrig[uG], eG);
                if (outType == type)
                {
                    outEdges.push_back(eCc);
                }
            }
        }

        void getAllOutgoingEdgesOfType(const CcData &cc, ogdf::node uG, EdgePartType type, std::vector<ogdf::adjEntry> &outEdges)
        {
            outEdges.clear();

            cc.Gcc->forEachAdj(uG, [&](node neighbor, edge eC) {
                ogdf::edge eOrig = cc.edgeToOrig[eC];

                if (cc.Gcc->source(eC) == uG)
                {
                    EdgePartType outType = ctx()._edge2types(eOrig).first;
                    if (type == outType)
                    {
                        outEdges.push_back(adjEntry{neighbor, eC});
                    }
                }
                else
                {
                    EdgePartType outType = ctx()._edge2types(eOrig).second;
                    if (type == outType)
                    {
                        outEdges.push_back(adjEntry{neighbor, eC});
                    }
                }
            });
        }

        namespace SPQRsolve
        {
            struct EdgeDPState
            {
                node s{nullptr};
                node t{nullptr};

                int localPlusS{0};
                int localPlusT{0};
                int localMinusT{0};
                int localMinusS{0};
            };

            struct EdgeDP
            {
                EdgeDPState down;
                EdgeDPState up;
            };

            struct NodeDPState
            {
                std::vector<ogdf::node> GccCuts_last3;
            };

            void printAllStates(const ogdf::NodeArray<NodeDPState> &node_dp,
                                const TreeGraph &T)
            {
                std::cout << "Node dp states: " << std::endl;
                for (node v : T.nodes)
                {
                    std::cout << "Node " << v.index() << ", ";
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
                edge e = nullptr)
            {
                PROFILE_FUNCTION();
                if (curr == nullptr)
                {
                    curr = spqr.rootNode();
                    parent = curr;
                    dfsSPQR_order(spqr, edge_order, node_order, curr, parent);
                    return;
                }

                node_order.push_back(curr);
                const auto& T = spqr.tree();
                T.forEachAdj(curr, [&](node child, edge adjEdge) {
                    if (child == parent)
                        return;
                    dfsSPQR_order(spqr, edge_order, node_order, child, curr, adjEdge);
                });
                if (curr != parent)
                    edge_order.push_back(e);
            }

            void processEdge(ogdf::edge curr_edge,
                             ogdf::EdgeArray<EdgeDP> &dp,
                             const CcData &cc,
                             BlockData &blk)
            {
                auto &C = ctx();

                EdgeDPState &state = dp[curr_edge].down;
                EdgeDPState &back_state = dp[curr_edge].up;

                const StaticSPQRTree &spqr = *blk.spqr;
                const auto &T = spqr.tree();

                ogdf::node u = T.source(curr_edge);
                ogdf::node v = T.target(curr_edge);

                ogdf::node A = nullptr; 
                ogdf::node B = nullptr; 

                if (blk.parent[u] == v)
                {
                    A = v;
                    B = u;
                }
                else if (blk.parent[v] == u)
                {
                    A = u;
                    B = v;
                }
                else
                {
                    OGDF_ASSERT(false);
                    return;
                }

                state.localPlusS = 0;
                state.localPlusT = 0;
                state.localMinusS = 0;
                state.localMinusT = 0;

                const Skeleton &skel = spqr.skeleton(B); 
                const auto &skelGraph = skel.getGraph();

                auto mapSkeletonToGlobal = [&](ogdf::node vSkel) -> ogdf::node
                {
                    if (!vSkel)
                        return nullptr;
                    ogdf::node vBlk = skel.original(vSkel);
                    if (!vBlk)
                        return nullptr;
                    ogdf::node vCc = blk.toCc[vBlk];
                    if (!vCc)
                        return nullptr;
                    return cc.nodeToOrig[vCc];
                };

                for (ogdf::edge e : skelGraph.edges)
                {
                    ogdf::node uSk = skelGraph.source(e);
                    ogdf::node vSk = skelGraph.target(e);

                    ogdf::node D = skel.twinTreeNode(e);

                    if (D == A)
                    {
                        ogdf::node vBlk = skel.original(vSk);
                        ogdf::node uBlk = skel.original(uSk);

                        state.s = back_state.s = vBlk;
                        state.t = back_state.t = uBlk;
                        break;
                    }
                }

                for (ogdf::edge e : skelGraph.edges)
                {
                    ogdf::node uSk = skelGraph.source(e);
                    ogdf::node vSk = skelGraph.target(e);

                    ogdf::node uBlk = skel.original(uSk);
                    ogdf::node vBlk = skel.original(vSk);

                    if (!skel.isVirtual(e))
                    {
                        ogdf::edge eG = blk.edgeToOrig[skel.realEdge(e)];

                        ogdf::node uG = C.G.source(eG);
                        ogdf::node vG = C.G.target(eG);

                        if (uG == blk.nodeToOrig[state.s])
                        {
                            auto t = getNodeEdgeType(uG, eG);
                            if (t == EdgePartType::PLUS)
                                state.localPlusS++;
                            else if (t == EdgePartType::MINUS)
                                state.localMinusS++;
                        }

                        if (vG == blk.nodeToOrig[state.s])
                        {
                            auto t = getNodeEdgeType(vG, eG);
                            if (t == EdgePartType::PLUS)
                                state.localPlusS++;
                            else if (t == EdgePartType::MINUS)
                                state.localMinusS++;
                        }

                        if (uG == blk.nodeToOrig[state.t])
                        {
                            auto t = getNodeEdgeType(uG, eG);
                            if (t == EdgePartType::PLUS)
                                state.localPlusT++;
                            else if (t == EdgePartType::MINUS)
                                state.localMinusT++;
                        }

                        if (vG == blk.nodeToOrig[state.t])
                        {
                            auto t = getNodeEdgeType(vG, eG);
                            if (t == EdgePartType::PLUS)
                                state.localPlusT++;
                            else if (t == EdgePartType::MINUS)
                                state.localMinusT++;
                        }

                        continue;
                    }

                    ogdf::node D = skel.twinTreeNode(e);

                    if (D == A)
                    {
                        continue;
                    }

                    ogdf::edge treeE = blk.skel2tree.at(e);
                    OGDF_ASSERT(treeE != nullptr);

                    const EdgeDPState &child = dp[treeE].down;

                    if (state.s == child.s)
                    {
                        state.localPlusS += child.localPlusS;
                        state.localMinusS += child.localMinusS;
                    }

                    if (state.s == child.t)
                    {
                        state.localPlusS += child.localPlusT;
                        state.localMinusS += child.localMinusT;
                    }

                    if (state.t == child.t)
                    {
                        state.localPlusT += child.localPlusT;
                        state.localMinusT += child.localMinusT;
                    }

                    if (state.t == child.s)
                    {
                        state.localPlusT += child.localPlusS;
                        state.localMinusT += child.localMinusS;
                    }
                }
            }

            void processNode(ogdf::node curr_node,
                             ogdf::EdgeArray<EdgeDP> &edge_dp,
                             const CcData & /*cc*/,
                             BlockData &blk)
            {
                auto& C = ctx();
                ogdf::node A = curr_node;
                const StaticSPQRTree &spqr = *blk.spqr;
                const Skeleton &skel = spqr.skeleton(A);
                const auto &skelG = skel.getGraph();

                Graph newGraph;

                NodeArray<node> skelToNew(skelG, nullptr);
                for (node v : skelG.nodes)
                    skelToNew[v] = newGraph.newNode();

                NodeArray<node> newToSkel(newGraph, nullptr);
                for (node v : skelG.nodes)
                    newToSkel[skelToNew[v]] = v;

                for (ogdf::node h : skelG.nodes)
                {
                    ogdf::node vB = skel.original(h);
                    blk.blkToSkel[vB] = h;
                }

                NodeArray<int> localPlusDeg(newGraph, 0);
                NodeArray<int> localMinusDeg(newGraph, 0);

                EdgeArray<bool> isVirtual(newGraph, false);
                EdgeArray<EdgeDPState *> edgeToDp(newGraph, nullptr);
                EdgeArray<EdgeDPState *> edgeToDpR(newGraph, nullptr);
                EdgeArray<node> edgeChild(newGraph, nullptr);

                std::vector<edge> virtualEdges;

                auto mapBlockToNew = [&](ogdf::node bV) -> ogdf::node
                {
                    ogdf::node sV = blk.blkToSkel[bV];
                    ogdf::node nV = skelToNew[sV];
                    return nV;
                };

                for (edge e : skelG.edges)
                {
                    node u = skelG.source(e);
                    node v = skelG.target(e);

                    ogdf::node uBlk = skel.original(u);
                    ogdf::node vBlk = skel.original(v);

                    ogdf::node uG = blk.nodeToOrig[uBlk];
                    ogdf::node vG = blk.nodeToOrig[vBlk];

                    node nU = skelToNew[u];
                    node nV = skelToNew[v];

                    if (!skel.isVirtual(e))
                    {
                        ogdf::edge eG = blk.edgeToOrig[skel.realEdge(e)];
                        uG = C.G.source(eG);
                        vG = C.G.target(eG);

                        auto newEdge = newGraph.newEdge(nU, nV);
                        isVirtual[newEdge] = false;

                        if (blk.nodeToOrig[skel.original(newToSkel[nU])] == uG)
                        {
                            localPlusDeg[nU] += (getNodeEdgeType(uG, eG) == EdgePartType::PLUS);
                            localMinusDeg[nU] += (getNodeEdgeType(uG, eG) == EdgePartType::MINUS);
                            localPlusDeg[nV] += (getNodeEdgeType(vG, eG) == EdgePartType::PLUS);
                            localMinusDeg[nV] += (getNodeEdgeType(vG, eG) == EdgePartType::MINUS);
                        }
                        else
                        {
                            localPlusDeg[nU] += (getNodeEdgeType(vG, eG) == EdgePartType::PLUS);
                            localMinusDeg[nU] += (getNodeEdgeType(vG, eG) == EdgePartType::MINUS);
                            localPlusDeg[nV] += (getNodeEdgeType(uG, eG) == EdgePartType::PLUS);
                            localMinusDeg[nV] += (getNodeEdgeType(uG, eG) == EdgePartType::MINUS);
                        }
                        continue;
                    }

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

                    if (nS == nU && nT == nV)
                    {
                        localMinusDeg[nS] += child->localMinusT;
                        localPlusDeg[nS] += child->localPlusT;
                        localMinusDeg[nT] += child->localMinusS;
                        localPlusDeg[nT] += child->localPlusS;
                    }
                    else
                    {
                        localMinusDeg[nS] += child->localMinusS;
                        localPlusDeg[nS] += child->localPlusS;
                        localMinusDeg[nT] += child->localMinusT;
                        localPlusDeg[nT] += child->localPlusT;
                    }
                }

                for (edge e : virtualEdges)
                {
                    EdgeDPState *BA = edgeToDp[e];
                    EdgeDPState *AB = edgeToDpR[e];

                    BA->localPlusS = localPlusDeg[mapBlockToNew(BA->s)] - AB->localPlusS;
                    BA->localPlusT = localPlusDeg[mapBlockToNew(BA->t)] - AB->localPlusT;
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
                const Skeleton &skel = blk.spqr->skeleton(sNode);
                const auto &skelG = skel.getGraph();
                const auto &T = blk.spqr->tree();

                std::vector<ogdf::node> nodesInOrderGcc;
                std::vector<ogdf::node> nodesInOrderSkel;

                std::unordered_map<uint32_t, EdgeDPState *> skelToState;
                skelToState.reserve(8);

                std::vector<ogdf::edge> adjEdgesG;
                std::vector<adjEntry> adjEntriesSkel;

                for (edge e : skelG.edges)
                {
                    if (!skel.isVirtual(e))
                        continue;
                    auto B = skel.twinTreeNode(e);
                    edge treeE = blk.skel2tree.at(e);

                    EdgeDPState *child = (B == blk.parent(sNode) ? &dp[treeE].up : &dp[treeE].down);
                    skelToState[treeE.idx] = child;
                }

                {
                    node firstNode = skelG.firstNode();
                    node secondNode = nullptr;
                    skelG.forEachAdj(firstNode, [&](node neighbor, edge) {
                        if (!secondNode) secondNode = neighbor;
                    });
                    if (secondNode)
                    {
                        ogdf::node u = firstNode;
                        ogdf::node prev = secondNode;

                        while (true)
                        {
                            nodesInOrderGcc.push_back(blk.toCc[skel.original(u)]);
                            nodesInOrderSkel.push_back(u);

                            ogdf::node nextU = nullptr;
                            ogdf::edge nextE = nullptr;
                            bool closing = false;
                            ogdf::edge closingEdge = nullptr;

                            skelG.forEachAdj(u, [&](ogdf::node neighbor, ogdf::edge e) {
                                if (neighbor == prev)
                                    return;
                                if (neighbor == firstNode && u != firstNode)
                                {
                                    closing = true;
                                    closingEdge = e;
                                    return;
                                }
                                if (neighbor == firstNode || neighbor == prev)
                                    return;

                                nextU = neighbor;
                                nextE = e;
                            });

                            if (closing)
                            {
                                if (skel.realEdge(closingEdge))
                                    adjEdgesG.push_back(blk.edgeToOrig[skel.realEdge(closingEdge)]);
                                else
                                    adjEdgesG.push_back(nullptr);
                                adjEntriesSkel.push_back(adjEntry{firstNode, closingEdge});
                                break;
                            }

                            if (!nextU)
                                break;

                            if (skel.realEdge(nextE))
                                adjEdgesG.push_back(blk.edgeToOrig[skel.realEdge(nextE)]);
                            else
                                adjEdgesG.push_back(nullptr);
                            adjEntriesSkel.push_back(adjEntry{nextU, nextE});

                            prev = u;
                            u = nextU;
                        }
                    }
                }

                std::vector<bool> cuts(nodesInOrderGcc.size(), false);
                std::vector<std::string> res;

                for (size_t i = 0; i < nodesInOrderGcc.size(); ++i)
                {
                    auto uGcc = nodesInOrderGcc[i];

                    std::vector<edge> adjEdgesSkelLoc = {
                        adjEntriesSkel[(i + adjEntriesSkel.size() - 1) % adjEntriesSkel.size()].theEdge(),
                        adjEntriesSkel[i].theEdge()};
                    std::vector<ogdf::edge> adjEdgesGLoc = {
                        adjEdgesG[(i + adjEdgesG.size() - 1) % adjEdgesG.size()],
                        adjEdgesG[i]};

                    bool nodeIsCut = ((cc.isCutNode[uGcc] && cc.badCutCount[uGcc] == 1) ||
                                      (!cc.isCutNode[uGcc]));

                    EdgePartType t0 = EdgePartType::NONE;
                    EdgePartType t1 = EdgePartType::NONE;

                    if (!skel.isVirtual(adjEdgesSkelLoc[0]))
                    {
                        t0 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesGLoc[0]);
                    }
                    else
                    {
                        edge treeE0 = blk.skel2tree.at(adjEdgesSkelLoc[0]);
                        EdgeDPState *state0 = skelToState.at(treeE0.idx);
                        if (blk.toCc[state0->s] == uGcc)
                        {
                            if (state0->localMinusS == 0 && state0->localPlusS > 0)
                                t0 = EdgePartType::PLUS;
                            else if (state0->localMinusS > 0 && state0->localPlusS == 0)
                                t0 = EdgePartType::MINUS;
                        }
                        else
                        {
                            if (state0->localMinusT == 0 && state0->localPlusT > 0)
                                t0 = EdgePartType::PLUS;
                            else if (state0->localMinusT > 0 && state0->localPlusT == 0)
                                t0 = EdgePartType::MINUS;
                        }
                    }

                    // edge 1
                    if (!skel.isVirtual(adjEdgesSkelLoc[1]))
                    {
                        t1 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesGLoc[1]);
                    }
                    else
                    {
                        edge treeE1 = blk.skel2tree.at(adjEdgesSkelLoc[1]);
                        EdgeDPState *state1 = skelToState.at(treeE1.idx);
                        if (blk.toCc[state1->s] == uGcc)
                        {
                            if (state1->localMinusS == 0 && state1->localPlusS > 0)
                                t1 = EdgePartType::PLUS;
                            else if (state1->localMinusS > 0 && state1->localPlusS == 0)
                                t1 = EdgePartType::MINUS;
                        }
                        else
                        {
                            if (state1->localMinusT == 0 && state1->localPlusT > 0)
                                t1 = EdgePartType::PLUS;
                            else if (state1->localMinusT > 0 && state1->localPlusT == 0)
                                t1 = EdgePartType::MINUS;
                        }
                    }

                    nodeIsCut &= (t0 != EdgePartType::NONE &&
                                  t1 != EdgePartType::NONE &&
                                  t0 != t1);

                    if (nodeIsCut)
                    {
                        if (node_dp[sNode].GccCuts_last3.size() < 3)
                            node_dp[sNode].GccCuts_last3.push_back(uGcc);

                        if (!skel.isVirtual(adjEdgesSkelLoc[0]))
                        {
                            EdgePartType tt0 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesGLoc[0]);
                            res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] +
                                          (tt0 == EdgePartType::PLUS ? "+" : "-"));
                        }
                        else
                        {
                            edge treeE0 = blk.skel2tree.at(adjEdgesSkelLoc[0]);
                            EdgeDPState *state0 = skelToState.at(treeE0.idx);
                            if (uGcc == blk.toCc[state0->s])
                            {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] +
                                              (state0->localPlusS > 0 ? "+" : "-"));
                            }
                            else
                            {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] +
                                              (state0->localPlusT > 0 ? "+" : "-"));
                            }
                        }

                        if (!skel.isVirtual(adjEdgesSkelLoc[1]))
                        {
                            EdgePartType tt1 = getNodeEdgeType(cc.nodeToOrig[uGcc], adjEdgesGLoc[1]);
                            res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] +
                                          (tt1 == EdgePartType::PLUS ? "+" : "-"));
                        }
                        else
                        {
                            edge treeE1 = blk.skel2tree.at(adjEdgesSkelLoc[1]);
                            EdgeDPState *state1 = skelToState.at(treeE1.idx);
                            if (uGcc == blk.toCc[state1->s])
                            {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] +
                                              (state1->localPlusS > 0 ? "+" : "-"));
                            }
                            else
                            {
                                res.push_back(ctx().node2name[cc.nodeToOrig[uGcc]] +
                                              (state1->localPlusT > 0 ? "+" : "-"));
                            }
                        }
                    }
                }

                OGDF_ASSERT(res.size() % 2 == 0);
                if (res.size() > 2)
                {
                    for (size_t i = 1; i < res.size(); i += 2)
                    {
                        std::vector<std::string> v = {res[i], res[(i + 1) % res.size()]};
                        addSnarlTagged("S", std::move(v));
                    }
                }
            }

            void solveP(ogdf::node pNode,
                        ogdf::NodeArray<SPQRsolve::NodeDPState> &node_dp,
                        ogdf::EdgeArray<EdgeDP> &edge_dp,
                        BlockData &blk,
                        const CcData &cc)
            {
                PROFILE_FUNCTION();
                auto &C = ctx();

                const Skeleton &skel = blk.spqr->skeleton(pNode);
                const auto &skelGraph = skel.getGraph();
                const auto &T = blk.spqr->tree();

                VLOG << "[DEBUG][solveP] P-node idx=" << pNode.index()
                     << " skeleton |V|=" << skelGraph.numberOfNodes()
                     << " |E|=" << skelGraph.numberOfEdges() << "\n";

                ogdf::node pole0Skel = nullptr, pole1Skel = nullptr;
                {
                    auto it = skelGraph.nodes.begin();
                    if (it != skelGraph.nodes.end())
                        pole0Skel = *it++;
                    if (it != skelGraph.nodes.end())
                        pole1Skel = *it;
                }
                if (!pole0Skel || !pole1Skel)
                {
                    VLOG << "[DEBUG][solveP]  P-node has <2 skeleton vertices, skip\n";
                    return;
                }

                ogdf::node pole0Blk = skel.original(pole0Skel);
                ogdf::node pole1Blk = skel.original(pole1Skel);
                if (!pole0Blk || !pole1Blk)
                {
                    VLOG << "[DEBUG][solveP]  skel.original() returned nullptr pole, skip\n";
                    return;
                }

                ogdf::node pole0Gcc = blk.toCc[pole0Blk];
                ogdf::node pole1Gcc = blk.toCc[pole1Blk];

                VLOG << "[DEBUG][solveP]  poles: "
                     << C.node2name[cc.nodeToOrig[pole0Gcc]] << " (Gcc idx=" << pole0Gcc.index() << "), "
                     << C.node2name[cc.nodeToOrig[pole1Gcc]] << " (Gcc idx=" << pole1Gcc.index() << ")\n";

                auto hasDanglingOutside = [&](ogdf::node vGcc)
                {
                    if (!cc.isCutNode[vGcc])
                        return false;
                    if (cc.badCutCount[vGcc] >= 2)
                        return true;
                    if (cc.badCutCount[vGcc] == 1 && cc.lastBad[vGcc] != blk.bNode)
                        return true;
                    return false;
                };
                if (hasDanglingOutside(pole0Gcc) || hasDanglingOutside(pole1Gcc))
                {
                    VLOG << "[DEBUG][solveP]  bad dangling at pole, skip P-node\n";
                    return;
                }

                std::vector<ogdf::adjEntry> edgeOrdering;
                skelGraph.forEachAdj(pole0Skel, [&](node neighbor, edge e) {
                    edgeOrdering.push_back(adjEntry{neighbor, e});
                });

                for (ogdf::adjEntry adj : edgeOrdering)
                {
                    ogdf::edge eSkel = adj.theEdge();
                    if (!skel.isVirtual(eSkel))
                        continue;

                    auto itMap = blk.skel2tree.find(eSkel);
                    if (itMap == blk.skel2tree.end())
                        continue;
                    ogdf::edge treeE = itMap->second;
                    ogdf::node B = (T.source(treeE) == pNode ? T.target(treeE) : T.source(treeE));

                    EdgeDP &dpVal = edge_dp[treeE];
                    EdgeDPState &state = (blk.parent[pNode] == B ? dpVal.up : dpVal.down);

                    // pôle 0
                    if (state.s == pole0Blk)
                    {
                        if (state.localPlusS > 0 && state.localMinusS > 0)
                        {
                            VLOG << "[DEBUG][solveP]  child gives both signs at pole0, abort P-node\n";
                            return;
                        }
                    }
                    else
                    {
                        if (state.localPlusT > 0 && state.localMinusT > 0)
                        {
                            VLOG << "[DEBUG][solveP]  child gives both signs at pole0 (T side), abort P-node\n";
                            return;
                        }
                    }

                    // pôle 1
                    if (state.s == pole1Blk)
                    {
                        if (state.localPlusS > 0 && state.localMinusS > 0)
                        {
                            VLOG << "[DEBUG][solveP]  child gives both signs at pole1, abort P-node\n";
                            return;
                        }
                    }
                    else
                    {
                        if (state.localPlusT > 0 && state.localMinusT > 0)
                        {
                            VLOG << "[DEBUG][solveP]  child gives both signs at pole1 (T side), abort P-node\n";
                            return;
                        }
                    }
                }

                for (auto left : {EdgePartType::PLUS, EdgePartType::MINUS})
                {
                    for (auto right : {EdgePartType::PLUS, EdgePartType::MINUS})
                    {

                        std::vector<ogdf::edge> leftPart, rightPart;

                        for (ogdf::adjEntry adj : edgeOrdering)
                        {
                            ogdf::edge eSkel = adj.theEdge();

                            EdgePartType lSign = EdgePartType::NONE;
                            EdgePartType rSign = EdgePartType::NONE;

                            if (!skel.isVirtual(eSkel))
                            {
                                ogdf::edge eB = skel.realEdge(eSkel);
                                ogdf::edge eG = blk.edgeToOrig[eB];

                                ogdf::node pole0G = cc.nodeToOrig[pole0Gcc];
                                ogdf::node pole1G = cc.nodeToOrig[pole1Gcc];

                                lSign = getNodeEdgeType(pole0G, eG);
                                rSign = getNodeEdgeType(pole1G, eG);
                            }
                            else
                            {
                                auto itMap = blk.skel2tree.find(eSkel);
                                if (itMap == blk.skel2tree.end())
                                    continue;
                                ogdf::edge treeE = itMap->second;
                                ogdf::node B = (T.source(treeE) == pNode ? T.target(treeE) : T.source(treeE));

                                EdgeDP &dpVal = edge_dp[treeE];
                                EdgeDPState &st = (blk.parent[pNode] == B ? dpVal.up : dpVal.down);

                                if (st.s == pole0Blk)
                                {
                                    bool hasPlus = (st.localPlusS > 0);
                                    bool hasMinus = (st.localMinusS > 0);
                                    if (hasPlus && !hasMinus)
                                        lSign = EdgePartType::PLUS;
                                    else if (!hasPlus && hasMinus)
                                        lSign = EdgePartType::MINUS;
                                    else
                                        lSign = EdgePartType::NONE;
                                }
                                else
                                {
                                    bool hasPlus = (st.localPlusT > 0);
                                    bool hasMinus = (st.localMinusT > 0);
                                    if (hasPlus && !hasMinus)
                                        lSign = EdgePartType::PLUS;
                                    else if (!hasPlus && hasMinus)
                                        lSign = EdgePartType::MINUS;
                                    else
                                        lSign = EdgePartType::NONE;
                                }

                                if (st.s == pole1Blk)
                                {
                                    bool hasPlus = (st.localPlusS > 0);
                                    bool hasMinus = (st.localMinusS > 0);
                                    if (hasPlus && !hasMinus)
                                        rSign = EdgePartType::PLUS;
                                    else if (!hasPlus && hasMinus)
                                        rSign = EdgePartType::MINUS;
                                    else
                                        rSign = EdgePartType::NONE;
                                }
                                else
                                {
                                    bool hasPlus = (st.localPlusT > 0);
                                    bool hasMinus = (st.localMinusT > 0);
                                    if (hasPlus && !hasMinus)
                                        rSign = EdgePartType::PLUS;
                                    else if (!hasPlus && hasMinus)
                                        rSign = EdgePartType::MINUS;
                                    else
                                        rSign = EdgePartType::NONE;
                                }
                            }

                            if (lSign == left)
                                leftPart.push_back(eSkel);
                            if (rSign == right)
                                rightPart.push_back(eSkel);
                        }

                        if (leftPart.empty() || leftPart != rightPart)
                            continue;

                        bool ok = true;
                        if (leftPart.size() == 1)
                        {
                            ogdf::edge eSkel = leftPart[0];
                            if (skel.isVirtual(eSkel))
                            {
                                ogdf::node B = skel.twinTreeNode(eSkel);
                                if (blk.spqr->typeOf(B) == StaticSPQRTree::NodeType::SNode)
                                {
                                    for (ogdf::node gccCut : node_dp[B].GccCuts_last3)
                                    {
                                        if (gccCut != pole0Gcc && gccCut != pole1Gcc)
                                        {
                                            ok = false;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        if (!ok)
                            continue;
                        std::string sName = C.node2name[cc.nodeToOrig[pole0Gcc]] +
                                            (left == EdgePartType::PLUS ? "+" : "-");
                        std::string tName = C.node2name[cc.nodeToOrig[pole1Gcc]] +
                                            (right == EdgePartType::PLUS ? "+" : "-");

                        std::vector<std::string> v = {sName, tName};
                        VLOG << "[DEBUG][solveP]  addSnarlTagged P: "
                             << v[0] << " " << v[1] << "\n";
                        addSnarlTagged("P", std::move(v));
                    }
                }
            }

            void solveRR(ogdf::edge rrEdge,
                         ogdf::NodeArray<SPQRsolve::NodeDPState> &node_dp,
                         ogdf::EdgeArray<EdgeDP> &edge_dp,
                         BlockData &blk,
                         const CcData &cc)
            {
                PROFILE_FUNCTION();
                auto &C = ctx();

                EdgeDPState &down = edge_dp[rrEdge].down;
                EdgeDPState &up = edge_dp[rrEdge].up;

                if (!down.s || !down.t || !up.s || !up.t)
                {
                    return;
                }

                ogdf::node pole0Blk = down.s;
                ogdf::node pole1Blk = down.t;

                ogdf::node pole0Gcc = blk.toCc[pole0Blk];
                ogdf::node pole1Gcc = blk.toCc[pole1Blk];
                if (!pole0Gcc || !pole1Gcc)
                {
                    return;
                }

                auto hasDanglingOutside = [&](ogdf::node vGcc)
                {
                    if (!cc.isCutNode[vGcc])
                        return false;
                    if (cc.badCutCount[vGcc] >= 2)
                        return true;
                    if (cc.badCutCount[vGcc] == 1 && cc.lastBad[vGcc] != blk.bNode)
                        return true;
                    return false;
                };

                if (hasDanglingOutside(pole0Gcc) || hasDanglingOutside(pole1Gcc))
                {
                    return;
                }

                if ((up.localMinusS > 0 && up.localPlusS > 0) ||
                    (up.localMinusT > 0 && up.localPlusT > 0) ||
                    (down.localMinusS > 0 && down.localPlusS > 0) ||
                    (down.localMinusT > 0 && down.localPlusT > 0))
                {
                    return;
                }

                EdgePartType pole0DownType = EdgePartType::NONE;
                EdgePartType pole0UpType = EdgePartType::NONE;
                EdgePartType pole1DownType = EdgePartType::NONE;
                EdgePartType pole1UpType = EdgePartType::NONE;

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
                if (pole0DownType == pole0UpType)
                    return;
                if (pole1DownType == pole1UpType)
                    return;

                // Snarl for the “down” state
                {
                    std::string s =
                        C.node2name[cc.nodeToOrig[pole0Gcc]] +
                        (pole0DownType == EdgePartType::PLUS ? "+" : "-");
                    std::string t =
                        C.node2name[cc.nodeToOrig[pole1Gcc]] +
                        (pole1DownType == EdgePartType::PLUS ? "+" : "-");

                    std::vector<std::string> v = {s, t};
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

                    std::vector<std::string> v = {s, t};
                    addSnarlTagged("RR", std::move(v));
                }
            }

            void solveNodes(NodeArray<SPQRsolve::NodeDPState> &node_dp,
                            ogdf::EdgeArray<EdgeDP> &edge_dp,
                            BlockData &blk,
                            const CcData &cc)
            {
                PROFILE_FUNCTION();
                if (!blk.spqr)
                    return;

                const auto &T = blk.spqr->tree();

                VLOG << "[DEBUG][solveNodes] start, |T.nodes|=" << T.numberOfNodes()
                     << " |T.edges|=" << T.numberOfEdges() << "\n";

                // 1) S-nodes
                for (node tNode : T.nodes)
                {
                    auto tType = blk.spqr->typeOf(tNode);
                    if (tType == StaticSPQRTree::NodeType::SNode)
                    {
                        VLOG << "[DEBUG][solveNodes] S-node idx=" << tNode.index()
                             << " -> solveS()\n";
                        solveS(tNode, node_dp, edge_dp, blk, cc);
                    }
                }

                // 2) P-nodes
                for (node tNode : T.nodes)
                {
                    auto tType = blk.spqr->typeOf(tNode);
                    if (tType == StaticSPQRTree::NodeType::PNode)
                    {
                        VLOG << "[DEBUG][solveNodes] P-node idx=" << tNode.index()
                             << " -> solveP()\n";
                        solveP(tNode, node_dp, edge_dp, blk, cc);
                    }
                }

                // 3) R-R edges
                for (edge e : T.edges)
                {
                    auto srcT = blk.spqr->typeOf(T.source(e));
                    auto dstT = blk.spqr->typeOf(T.target(e));
                    if (srcT == SPQRTree::NodeType::RNode &&
                        dstT == SPQRTree::NodeType::RNode)
                    {
                        VLOG << "[DEBUG][solveNodes] R-R edge idx=" << e.idx
                             << " -> solveRR()\n";
                        solveRR(e, node_dp, edge_dp, blk, cc);
                    }
                }

                VLOG << "[DEBUG][solveNodes] end\n";
            }
            void solveSPQR(BlockData &blk, const CcData &cc)
            {
                PROFILE_FUNCTION();

                if (!blk.spqr)
                    return;
                if (!blk.Gblk || blk.Gblk->numberOfNodes() < 3)
                    return;

                auto &C = ctx();
                const auto &T = blk.spqr->tree();

                // DP on the edges of the SPQR tree
                ogdf::EdgeArray<EdgeDP> edge_dp(T);
                ogdf::NodeArray<NodeDPState> node_dp(T);

                std::vector<ogdf::node> nodeOrder;
                std::vector<ogdf::edge> edgeOrder;

                dfsSPQR_order(*blk.spqr, edgeOrder, nodeOrder);

                blk.blkToSkel.init(*blk.Gblk, nullptr);

                // Down phase on the edges
                for (ogdf::edge e : edgeOrder)
                {
                    processEdge(e, edge_dp, cc, blk);
                }

                // Local “node” phase on each node of the tree
                for (ogdf::node v : nodeOrder)
                {
                    processNode(v, edge_dp, cc, blk);
                }

                // Resolution of S/P/RR snarls
                solveNodes(node_dp, edge_dp, blk, cc);

                // Pre-calculation: for each vertex of the block, list of S-nodes
                // in which it appears (to filter case B of Prop. 3.16).
                ogdf::NodeArray<std::vector<ogdf::node>> vertexInSnodes(*blk.Gblk);
                for (ogdf::node vB : blk.Gblk->nodes)
                {
                    vertexInSnodes[vB].clear();
                }

                for (ogdf::node mu : T.nodes)
                {
                    if (blk.spqr->typeOf(mu) != ogdf::StaticSPQRTree::NodeType::SNode)
                        continue;
                    const ogdf::Skeleton &skel = blk.spqr->skeleton(mu);
                    const auto &skelG = skel.getGraph();
                    for (ogdf::node vSk : skelG.nodes)
                    {
                        ogdf::node vB = skel.original(vSk);
                        vertexInSnodes[vB].push_back(mu);
                    }
                }

                auto shareSnode = [&](ogdf::node aB, ogdf::node bB) -> bool
                {
                    const auto &La = vertexInSnodes[aB];
                    const auto &Lb = vertexInSnodes[bB];
                    if (La.empty() || Lb.empty())
                        return false;
                    // Naive intersection, but the lists are very small in practice [TO OPTIMIZE ?]
                    for (ogdf::node x : La)
                    {
                        for (ogdf::node y : Lb)
                        {
                            if (x == y)
                                return true;
                        }
                    }
                    return false;
                };

                // Test “dangling” relative to the current block
                auto hasDanglingOutside = [&](ogdf::node vGcc)
                {
                    if (!cc.isCutNode[vGcc])
                        return false;
                    if (cc.badCutCount[vGcc] >= 2)
                        return true;
                    if (cc.badCutCount[vGcc] == 1 && cc.lastBad[vGcc] != blk.bNode)
                        return true;
                    return false;
                };

                // ----------------------
                // Case E: single-edge snarls
                // ----------------------
                std::vector<ogdf::edge> edgesSorted;
                edgesSorted.reserve(blk.Gblk->numberOfEdges());
                for (ogdf::edge eB : blk.Gblk->edges)
                    edgesSorted.push_back(eB);

                std::sort(edgesSorted.begin(), edgesSorted.end(),
                          [](ogdf::edge a, ogdf::edge b)
                          { return a.idx < b.idx; });

                for (ogdf::edge eB : edgesSorted)
                {
                    ogdf::edge eG = blk.edgeToOrig[eB];

                    ogdf::node uB = blk.Gblk->source(eB);
                    ogdf::node vB = blk.Gblk->target(eB);

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

                    auto flipSign = [](EdgePartType t)
                    {
                        return (t == EdgePartType::PLUS ? EdgePartType::MINUS : EdgePartType::PLUS);
                    };

                    auto check_one_vertex = [&](ogdf::node vB,
                                                EdgePartType sign,   
                                                EdgePartType eSign) { 
                        int totPlus = blk.blkDegPlus[vB];
                        int totMinus = blk.blkDegMinus[vB];

                        if (sign == EdgePartType::PLUS)
                        {
                            if (eSign == EdgePartType::PLUS)
                            {
                                // Case A: e = {u+, ...}, others must be -
                                int othersPlus = totPlus - 1;
                                int othersMinus = totMinus;
                                return (othersPlus == 0 && othersMinus > 0);
                            }
                            else
                            {
                                // Case B: e = {u-, ...}, others must be +
                                int othersPlus = totPlus;
                                int othersMinus = totMinus - 1;
                                return (othersMinus == 0 && othersPlus > 0);
                            }
                        }
                        else
                        { // sign == MINUS
                            if (eSign == EdgePartType::MINUS)
                            {
                                // Case A: e = {u-, ...}, others must be +
                                int othersMinus = totMinus - 1;
                                int othersPlus = totPlus;
                                return (othersMinus == 0 && othersPlus > 0);
                            }
                            else
                            {
                                // Case B: e = {u+, ...}, others must be -
                                int othersMinus = totMinus;
                                int othersPlus = totPlus - 1;
                                return (othersPlus == 0 && othersMinus > 0);
                            }
                        }
                    };

                    auto testCandidate = [&](EdgePartType signU,
                                             EdgePartType signV,
                                             bool isFlipCase)
                    {
                        if (isFlipCase && shareSnode(uB, vB))
                            return;

                        if (!check_one_vertex(uB, signU, edgeSignU))
                            return;
                        if (!check_one_vertex(vB, signV, edgeSignV))
                            return;

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

        void findTips(CcData &cc)
        {
            MARK_SCOPE_MEM("sn/findTips");
            PROFILE_FUNCTION();
            size_t localIsolated = 0;
            auto &C = ctx();

            VLOG << "[DEBUG][findTips] -----\n";

            for (node v : cc.Gcc->nodes)
            {
                int plusCnt = 0, minusCnt = 0;
                node vG = cc.nodeToOrig[v];
                const std::string &name = C.node2name[vG];

                cc.Gcc->forEachAdj(v, [&](node /*neighbor*/, edge eAdj) {
                    ogdf::edge e = cc.edgeToOrig[eAdj];
                    EdgePartType eType = getNodeEdgeType(vG, e);
                    if (eType == EdgePartType::PLUS)
                        plusCnt++;
                    else if (eType == EdgePartType::MINUS)
                        minusCnt++;
                });

                if (plusCnt + minusCnt == 0)
                {
                    localIsolated++;
                }

                if (plusCnt == 0 || minusCnt == 0)
                {
                    cc.isTip[v] = true;
                }
                else
                {
                    cc.isTip[v] = false;
                }

                VLOG << "[DEBUG][findTips] node " << name
                     << "(Gcc idx=" << v.index() << ")"
                     << "plusCnt=" << plusCnt
                     << "minusCnt=" << minusCnt
                     << "isTip=" << (cc.isTip[v] ? "true" : "false")
                     << "\n";
            }

            {
                std::lock_guard<std::mutex> lk(g_snarls_mtx);
                isolatedNodesCnt += localIsolated;
            }

            VLOG << "[DEBUG][findTips] localIsolated=" << localIsolated
                 << " totalIsolated=" << isolatedNodesCnt << "\n";
        }

        void processCutNodes(CcData &cc)
        {
            MARK_SCOPE_MEM("sn/processCutNodes");
            PROFILE_FUNCTION();
            auto &C = ctx();

            VLOG << "[DEBUG][processCutNodes] -----\n";

            const uint32_t numB = cc.bc->numberOfBComps();
            ogdf::EdgeArray<uint32_t> edge2block(*cc.Gcc, UINT32_MAX);
            for (uint32_t bIdx = 0; bIdx < numB; ++bIdx)
            {
                ogdf::node bNode{bIdx};
                for (ogdf::edge eCc : cc.bc->hEdges(bNode))
                {
                    edge2block[eCc] = bIdx;
                }
            }

            for (node v : cc.Gcc->nodes)
            {
                node vG = cc.nodeToOrig[v];
                const std::string &name = C.node2name[vG];
                if (cc.bc->typeOfGNode(v) == BCTree::GNodeType::CutVertex)
                {
                    cc.isCutNode[v] = true;

                    struct BlockFlags { uint32_t bIdx; bool hasPlus; bool hasMinus; };
                    std::vector<BlockFlags> blocks;
                    blocks.reserve(8);

                    cc.Gcc->forEachAdj(v, [&](ogdf::node /*nb*/, ogdf::edge eCc) {
                        ogdf::edge eG = cc.edgeToOrig[eCc];
                        if (!eG) return;
                        uint32_t bIdx = edge2block[eCc];
                        if (bIdx == UINT32_MAX) return;  // edge not in any block (shouldn't happen)
                        EdgePartType outType = getNodeEdgeType(vG, eG);

                        BlockFlags *slot = nullptr;
                        for (auto &bf : blocks) {
                            if (bf.bIdx == bIdx) { slot = &bf; break; }
                        }
                        if (!slot) {
                            blocks.push_back({bIdx, false, false});
                            slot = &blocks.back();
                        }
                        if (outType == EdgePartType::PLUS)  slot->hasPlus  = true;
                        if (outType == EdgePartType::MINUS) slot->hasMinus = true;
                    });

                    bool isGood = true;
                    for (auto &bf : blocks) {
                        if (bf.hasPlus && bf.hasMinus) {
                            isGood = false;
                            cc.lastBad[v] = ogdf::node{bf.bIdx};
                            cc.badCutCount[v]++;
                        }
                    }
                    cc.isGoodCutNode[v] = isGood;
                }
                VLOG << "[DEBUG][processCutNodes] node " << name
                     << "(Gcc idx=" << v.index() << ")"
                     << "isCutNode=" << (cc.isCutNode[v] ? "true" : "false")
                     << "badCutCount=" << cc.badCutCount[v]
                     << "isGoodCutNode=" << (cc.isGoodCutNode[v] ? "true" : "false");
                if (cc.lastBad[v] != nullptr)
                {
                    VLOG << "lastBad(B-node idx)=" << cc.lastBad[v].index();
                }
                VLOG << "\n";
            }
        }

        void findCutSnarl(CcData &cc)
        {
            MARK_SCOPE_MEM("sn/findCutSnarl");

            ogdf::NodeArray<std::pair<bool, bool>> visited(
                *cc.Gcc, {false, false}); 

            for (ogdf::node start : cc.Gcc->nodes)
            {
                for (auto t : {EdgePartType::PLUS, EdgePartType::MINUS})
                {

                    if (t == EdgePartType::PLUS && visited[start].second)
                        continue;
                    if (t == EdgePartType::MINUS && visited[start].first)
                        continue;

                    std::vector<std::string> goodNodes;

                    struct Frame
                    {
                        ogdf::node v;
                        EdgePartType edgeType;
                    };
                    std::stack<Frame> st;
                    st.push({start, t});

                    while (!st.empty())
                    {
                        auto [v, edgeType] = st.top();
                        st.pop();

                        bool &minusVisited = visited[v].first;
                        bool &plusVisited = visited[v].second;
                        bool isGoodOrTip = (cc.isGoodCutNode[v] || cc.isTip[v]);

                        if (!isGoodOrTip)
                        {
                            if (minusVisited && plusVisited)
                            {
                                continue;
                            }
                        }
                        else
                        {
                            if (edgeType == EdgePartType::MINUS && minusVisited)
                                continue;
                            if (edgeType == EdgePartType::PLUS && plusVisited)
                                continue;
                        }

                        if (isGoodOrTip &&
                            ctx().node2name[cc.nodeToOrig[v]] != "_trash")
                        {

                            goodNodes.push_back(
                                ctx().node2name[cc.nodeToOrig[v]] +
                                (edgeType == EdgePartType::PLUS ? "+" : "-"));
                        }

                        if (!isGoodOrTip)
                        {
                            minusVisited = true;
                            plusVisited = true;
                        }
                        else
                        {
                            if (edgeType == EdgePartType::MINUS)
                                minusVisited = true;
                            else
                                plusVisited = true;
                        }

                        std::vector<ogdf::adjEntry> sameOutEdges, otherOutEdges;
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

                        for (auto &adjE : sameOutEdges)
                        {
                            ogdf::node otherNode = adjE.twinNode();
                            ogdf::edge eCc = adjE.theEdge();
                            ogdf::edge eOrig = cc.edgeToOrig[eCc];

                            EdgePartType inType =
                                getNodeEdgeType(cc.nodeToOrig[otherNode], eOrig);

                            if ((inType == EdgePartType::PLUS && !visited[otherNode].second) ||
                                (inType == EdgePartType::MINUS && !visited[otherNode].first))
                            {
                                st.push({otherNode, inType});
                            }
                        }

                        if (canGoOther)
                        {
                            for (auto &adjE : otherOutEdges)
                            {
                                ogdf::node otherNode = adjE.twinNode();
                                ogdf::edge eCc = adjE.theEdge();
                                ogdf::edge eOrig = cc.edgeToOrig[eCc];

                                EdgePartType inType =
                                    getNodeEdgeType(cc.nodeToOrig[otherNode], eOrig);

                                if ((inType == EdgePartType::PLUS && !visited[otherNode].second) ||
                                    (inType == EdgePartType::MINUS && !visited[otherNode].first))
                                {
                                    st.push({otherNode, inType});
                                }
                            }
                        }
                    }

                    if (goodNodes.size() >= 2)
                    {
                        addSnarlTagged("CUT", std::move(goodNodes));
                    }
                }
            }
        }

        void buildBlockData(BlockData &blk, CcData &cc)
        {
            PROFILE_FUNCTION();

            auto &C = ctx();

            VLOG << "[DEBUG][buildBlockData] --------\n";
            VLOG << "[DEBUG][buildBlockData] B-node index=" << blk.bNode.index() << "\n";

            blk.Gblk = std::make_unique<ogdf::Graph>();

            blk.nodeToOrig.init(*blk.Gblk, nullptr);
            blk.edgeToOrig.init(*blk.Gblk, nullptr);
            blk.toCc.init(*blk.Gblk, nullptr);

            blk.blkDegPlus.init(*blk.Gblk, 0);
            blk.blkDegMinus.init(*blk.Gblk, 0);

            struct EdgeRec { ogdf::edge eCc; ogdf::node uC; ogdf::node vC; };
            std::vector<EdgeRec> edges_vec;
            for (ogdf::edge hE : cc.bc->hEdges(blk.bNode))
            {
                ogdf::edge eCc = cc.bc->original(hE);
                edges_vec.push_back({eCc, cc.Gcc->source(eCc), cc.Gcc->target(eCc)});
            }

            std::vector<ogdf::node> verts_vec;
            verts_vec.reserve(2 * edges_vec.size());
            for (const auto &er : edges_vec)
            {
                verts_vec.push_back(er.uC);
                verts_vec.push_back(er.vC);
            }
            std::sort(verts_vec.begin(), verts_vec.end(),
                      [](ogdf::node a, ogdf::node b)
                      { return a.index() < b.index(); });
            verts_vec.erase(std::unique(verts_vec.begin(), verts_vec.end()), verts_vec.end());

            std::unordered_map<ogdf::node, ogdf::node> cc_to_blk;
            cc_to_blk.reserve(verts_vec.size());

            for (ogdf::node vCc : verts_vec)
            {
                ogdf::node vB = blk.Gblk->newNode();
                cc_to_blk[vCc] = vB;
                blk.toCc[vB] = vCc;
                blk.nodeToOrig[vB] = cc.nodeToOrig[vCc];
            }

            for (const auto &er : edges_vec)
            {
                auto srcIt = cc_to_blk.find(er.uC);
                auto tgtIt = cc_to_blk.find(er.vC);
                if (srcIt != cc_to_blk.end() && tgtIt != cc_to_blk.end())
                {
                    ogdf::edge eB = blk.Gblk->newEdge(srcIt->second, tgtIt->second);
                    blk.edgeToOrig[eB] = cc.edgeToOrig[er.eCc];

                    ogdf::node uB = srcIt->second;
                    ogdf::node vB = tgtIt->second;
                    ogdf::node uG = blk.nodeToOrig[uB];
                    ogdf::node vG = blk.nodeToOrig[vB];

                    EdgePartType tU = getNodeEdgeType(uG, blk.edgeToOrig[eB]);
                    EdgePartType tV = getNodeEdgeType(vG, blk.edgeToOrig[eB]);

                    if (tU == EdgePartType::PLUS)
                        blk.blkDegPlus[uB]++;
                    else
                        blk.blkDegMinus[uB]++;

                    if (tV == EdgePartType::PLUS)
                        blk.blkDegPlus[vB]++;
                    else
                        blk.blkDegMinus[vB]++;
                }
            }

            VLOG << "[DEBUG][buildBlockData]  |V(Gblk)|=" << blk.Gblk->numberOfNodes()
                 << " |E(Gblk)|=" << blk.Gblk->numberOfEdges() << "\n";

            if (blk.Gblk->numberOfNodes() >= 3)
            {
                {

                    OGDF_ASSERT(blk.Gblk != nullptr);
                    OGDF_ASSERT(blk.Gblk->numberOfNodes() > 0);

                    blk.spqr = std::make_unique<ogdf::StaticSPQRTree>(*blk.Gblk);
                }

                OGDF_ASSERT(blk.spqr != nullptr);

                const auto &T = blk.spqr->tree();

                blk.skel2tree.clear();
                blk.skel2tree.reserve(2 * T.numberOfEdges());
                for (ogdf::edge te : T.edges)
                {
                    if (auto eSrc = blk.spqr->skeletonEdgeSrc(te))
                    {
                        blk.skel2tree[eSrc] = te;
                    }
                    if (auto eTgt = blk.spqr->skeletonEdgeTgt(te))
                    {
                        blk.skel2tree[eTgt] = te;
                    }
                }

                blk.parent.init(T, nullptr);
                ogdf::node root = blk.spqr->rootNode();
                blk.parent[root] = root;

                std::stack<ogdf::node> st;
                st.push(root);

                while (!st.empty())
                {
                    ogdf::node u = st.top();
                    st.pop();

                    T.forEachAdj(u, [&](node v, edge /*e*/) {
                        if (blk.parent[v] == nullptr)
                        {
                            blk.parent[v] = u;
                            st.push(v);
                        }
                    });
                }
            }
        }

        struct BlockPrep
        {
            CcData *cc;
            ogdf::node bNode;

            std::unique_ptr<BlockData> blk;

            BlockPrep(CcData *cc_, ogdf::node b) : cc(cc_), bNode(b), blk(nullptr) {}

            BlockPrep() = default;
            BlockPrep(const BlockPrep &) = delete;
            BlockPrep &operator=(const BlockPrep &) = delete;
            BlockPrep(BlockPrep &&) = default;
            BlockPrep &operator=(BlockPrep &&) = default;

            ogdf::NodeArray<int> blkDegPlus, blkDegMinus;
        };

        struct ThreadComponentArgs
        {
            size_t tid;
            size_t numThreads;
            int nCC;
            std::atomic<size_t> *nextIndex;
            std::vector<std::vector<node>> *bucket;
            std::vector<std::vector<edge>> *edgeBuckets;
            std::vector<std::unique_ptr<CcData>> *components;
        };

        struct ThreadBcTreeArgs
        {
            size_t tid;
            size_t numThreads;
            int nCC;
            std::atomic<size_t> *nextIndex;
            std::vector<std::unique_ptr<CcData>> *components;
            std::vector<std::vector<BlockPrep>> *perThreadPreps;
        };

        struct ThreadTipsArgs
        {
            size_t tid;
            size_t numThreads;
            int nCC;
            std::atomic<size_t> *nextIndex;
            std::vector<std::unique_ptr<CcData>> *components;
        };

        struct ThreadBlocksArgs
        {
            size_t tid;
            size_t numThreads;
            size_t blocks;
            std::atomic<size_t> *nextIndex;
            std::vector<BlockPrep> *blockPreps;
        };

        void *worker_component(void *arg)
        {
            std::unique_ptr<ThreadComponentArgs> targs(static_cast<ThreadComponentArgs *>(arg));
            size_t tid = targs->tid;
            size_t numThreads = targs->numThreads;
            int nCC = targs->nCC;
            std::atomic<size_t> *nextIndex = targs->nextIndex;
            std::vector<std::unique_ptr<CcData>> *components = targs->components;
            std::vector<std::vector<node>> *bucket = targs->bucket;
            std::vector<std::vector<edge>> *edgeBuckets = targs->edgeBuckets;

            size_t chunkSize = 1;
            size_t processed = 0;

            while (true)
            {
                size_t startIndex, endIndex;
                {
                    startIndex = nextIndex->fetch_add(chunkSize, std::memory_order_relaxed);
                    if (startIndex >= static_cast<size_t>(nCC))
                        break;
                    endIndex = std::min(startIndex + chunkSize, static_cast<size_t>(nCC));
                }

                auto chunkStart = std::chrono::high_resolution_clock::now();

                for (size_t cid = startIndex; cid < endIndex; ++cid)
                {

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

                        for (node vG : (*bucket)[cid])
                        {
                            node vC = (*components)[cid]->Gcc->newNode();
                            (*components)[cid]->nodeToOrig[vC] = vG;
                            orig_to_cc[vG] = vC;
                        }

                        auto& G = ctx().G;
                        for (edge e : (*edgeBuckets)[cid])
                        {
                            auto eC = (*components)[cid]->Gcc->newEdge(orig_to_cc[G.source(e)], orig_to_cc[G.target(e)]);
                            (*components)[cid]->edgeToOrig[eC] = e;

                            (*components)[cid]->degPlus[orig_to_cc[G.source(e)]] += (getNodeEdgeType(G.source(e), e) == EdgePartType::PLUS ? 1 : 0);
                            (*components)[cid]->degMinus[orig_to_cc[G.source(e)]] += (getNodeEdgeType(G.source(e), e) == EdgePartType::MINUS ? 1 : 0);
                            (*components)[cid]->degPlus[orig_to_cc[G.target(e)]] += (getNodeEdgeType(G.target(e), e) == EdgePartType::PLUS ? 1 : 0);
                            (*components)[cid]->degMinus[orig_to_cc[G.target(e)]] += (getNodeEdgeType(G.target(e), e) == EdgePartType::MINUS ? 1 : 0);
                        }
                    }
                    processed++;
                }

                auto chunkEnd = std::chrono::high_resolution_clock::now();
                auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);

                if (chunkDuration.count() < 1000)
                {
                    chunkSize = std::min(chunkSize * 2, static_cast<size_t>(nCC / numThreads));
                }
                else if (chunkDuration.count() > 5000)
                {
                    chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                }
            }

            std::cout << "Thread " << tid << " built " << processed << " components(rebuild cc graph)" << std::endl;
            return nullptr;
        }
        void *worker_bcTree(void *arg)
        {
            std::unique_ptr<ThreadBcTreeArgs> targs(static_cast<ThreadBcTreeArgs *>(arg));
            size_t tid = targs->tid;
            size_t numThreads = targs->numThreads;
            int nCC = targs->nCC;
            std::atomic<size_t> *nextIndex = targs->nextIndex;
            std::vector<std::unique_ptr<CcData>> *components = targs->components;
            std::vector<BlockPrep> &myPreps = (*targs->perThreadPreps)[tid];

            size_t chunkSize = 1;
            size_t processed = 0;

            while (true)
            {
                size_t startIndex, endIndex;
                {
                    startIndex = nextIndex->fetch_add(chunkSize, std::memory_order_relaxed);
                    if (startIndex >= static_cast<size_t>(nCC))
                        break;
                    endIndex = std::min(startIndex + chunkSize, static_cast<size_t>(nCC));
                }

                auto chunkStart = std::chrono::high_resolution_clock::now();

                for (size_t cid = startIndex; cid < endIndex; ++cid)
                {
                    CcData *cc = (*components)[cid].get();
                    if (!cc)
                        continue;
                    if (!cc->Gcc)
                        continue;

                    {
                        MARK_SCOPE_MEM("sn/worker_bcTree/build");

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

                        for (ogdf::node v : cc->bc->bcTree().nodes)
                        {
                            if (cc->bc->typeOfBNode(v) == BCTree::BNodeType::BComp)
                            {
                                VLOG << "  [DEBUG][worker_bcTree]  B-node "
                                     << v.index() << " (block)\n";
                                localPreps.emplace_back(cc, v);
                            }
                        }
                    }

                    {
                        myPreps.reserve(myPreps.size() + localPreps.size());
                        for (auto &bp : localPreps)
                        {
                            myPreps.emplace_back(std::move(bp));
                        }
                    }

                    ++processed;
                }

                auto chunkEnd = std::chrono::high_resolution_clock::now();
                auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);

                if (chunkDuration.count() < 1000)
                {
                    chunkSize = std::min(chunkSize * 2, static_cast<size_t>(nCC / numThreads));
                }
                else if (chunkDuration.count() > 5000)
                {
                    chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                }
            }

            std::cout << "Thread " << tid << " built " << processed << " components (bc trees)" << std::endl;
            return nullptr;
        }
        void *worker_tips(void *arg)
        {
            std::unique_ptr<ThreadTipsArgs> targs(static_cast<ThreadTipsArgs *>(arg));
            size_t tid = targs->tid;
            size_t numThreads = targs->numThreads;
            int nCC = targs->nCC;
            std::atomic<size_t> *nextIndex = targs->nextIndex;
            std::vector<std::unique_ptr<CcData>> *components = targs->components;

            size_t chunkSize = 1;
            size_t processed = 0;

            std::vector<std::vector<std::string>> localSnarls;
            tls_snarl_buffer = &localSnarls;

            while (true)
            {
                size_t startIndex, endIndex;
                {
                    startIndex = nextIndex->fetch_add(chunkSize, std::memory_order_relaxed);
                    if (startIndex >= static_cast<size_t>(nCC))
                        break;
                    endIndex = std::min(startIndex + chunkSize, static_cast<size_t>(nCC));
                }

                auto chunkStart = std::chrono::high_resolution_clock::now();

                for (size_t cid = startIndex; cid < endIndex; ++cid)
                {
                    CcData *cc = (*components)[cid].get();

                    findTips(*cc);
                    if (cc->bc->numberOfCComps() > 0)
                    {
                        processCutNodes(*cc);
                    }
                    findCutSnarl(*cc);

                    ++processed;
                }

                auto chunkEnd = std::chrono::high_resolution_clock::now();
                auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);

                if (chunkDuration.count() < 1000)
                {
                    chunkSize = std::min(chunkSize * 2, static_cast<size_t>(nCC / numThreads));
                }
                else if (chunkDuration.count() > 5000)
                {
                    chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                }
            }

            tls_snarl_buffer = nullptr;
            flushThreadLocalSnarls(localSnarls);

            std::cout << "Thread " << tid << " built " << processed << " components (cuts tips)" << std::endl;
            return nullptr;
        }

        void *worker_block_build(void *arg)
        {
            std::unique_ptr<ThreadBlocksArgs> targs(static_cast<ThreadBlocksArgs *>(arg));
            size_t tid = targs->tid;
            size_t numThreads = targs->numThreads;
            size_t blocks = targs->blocks;
            std::atomic<size_t> *nextIndex = targs->nextIndex;
            std::vector<BlockPrep> *blockPreps = targs->blockPreps;

            size_t chunkSize = 1;
            size_t processed = 0;

            while (true)
            {
                size_t startIndex, endIndex;
                {
                    startIndex = nextIndex->fetch_add(chunkSize, std::memory_order_relaxed);
                    if (startIndex >= static_cast<size_t>(blocks))
                        break;
                    endIndex = std::min(startIndex + chunkSize, static_cast<size_t>(blocks));
                }

                auto chunkStart = std::chrono::high_resolution_clock::now();

                for (size_t bid = startIndex; bid < endIndex; ++bid)
                {
                    blockPreps->at(bid).blk = std::make_unique<BlockData>();
                    BlockData &blk = *blockPreps->at(bid).blk;
                    blk.bNode = (*blockPreps)[bid].bNode;

                    {
                        // MEM_TIME_BLOCK("SPQR: build (snarl worker)");
                        buildBlockData(blk, *(*blockPreps)[bid].cc);
                    }

                    ++processed;
                }

                auto chunkEnd = std::chrono::high_resolution_clock::now();
                auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);

                if (chunkDuration.count() < 1000)
                {
                    chunkSize = std::min(chunkSize * 2, static_cast<size_t>(blocks / numThreads));
                }
                else if (chunkDuration.count() > 5000)
                {
                    chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                }
            }

            std::cout << "Thread " << tid << " built " << processed << " blocks (SPQR build)\n";
            return nullptr;
        }

        void *worker_block_solve(void *arg)
        {
            std::unique_ptr<ThreadBlocksArgs> targs(static_cast<ThreadBlocksArgs *>(arg));
            size_t tid = targs->tid;
            size_t numThreads = targs->numThreads;
            size_t blocks = targs->blocks;
            std::atomic<size_t> *nextIndex = targs->nextIndex;
            std::vector<BlockPrep> *blockPreps = targs->blockPreps;

            size_t chunkSize = 1;
            size_t processed = 0;

            std::vector<std::vector<std::string>> localSnarls;
            tls_snarl_buffer = &localSnarls;

            tls_spqr_seen_endpoint_pairs.clear();

            while (true)
            {
                size_t startIndex, endIndex;
                {
                    startIndex = nextIndex->fetch_add(chunkSize, std::memory_order_relaxed);
                    if (startIndex >= static_cast<size_t>(blocks))
                        break;
                    endIndex = std::min(startIndex + chunkSize, static_cast<size_t>(blocks));
                }

                auto chunkStart = std::chrono::high_resolution_clock::now();

                for (size_t bid = startIndex; bid < endIndex; ++bid)
                {
                    BlockPrep &prep = (*blockPreps)[bid];
                    if (!prep.blk)
                        continue;
                    BlockData &blk = *prep.blk;

                    {
                        if (blk.Gblk && blk.Gblk->numberOfNodes() >= 3)
                        {
                            SPQRsolve::solveSPQR(blk, *prep.cc);
                        }
                    }

                    prep.blk.reset();

                    ++processed;
                }

                auto chunkEnd = std::chrono::high_resolution_clock::now();
                auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(chunkEnd - chunkStart);

                if (chunkDuration.count() < 1000)
                {
                    chunkSize = std::min(chunkSize * 2, static_cast<size_t>(blocks / numThreads));
                }
                else if (chunkDuration.count() > 5000)
                {
                    chunkSize = std::max(chunkSize / 2, static_cast<size_t>(1));
                }
            }

            tls_snarl_buffer = nullptr;
            flushThreadLocalSnarls(localSnarls);

            std::cout << "Thread " << tid << " solved " << processed << " blocks (SPQR solve)\n";
            return nullptr;
        }

        void solve()
        {
            std::cout << "Finding snarls...\n";
            PROFILE_FUNCTION();
            auto &C = ctx();
            Graph &G = C.G;
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
                    for (node v : G.nodes)
                    {
                        bucket[compIdx[v]].push_back(v);
                    }
                }

                edgeBuckets.assign(nCC, {});
                {
                    MARK_SCOPE_MEM("sn/phase/BucketEdges");
                    for (edge e : G.edges)
                    {
                        edgeBuckets[compIdx[G.source(e)]].push_back(e);
                    }
                }
            }

            std::vector<std::unique_ptr<CcData>> components(nCC);
            std::vector<BlockPrep> blockPreps;
            {
                PhaseSampler build_sampler(g_stats_build);
                {
                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});

                    if (numThreads <= 1)
                    {
                        std::atomic<size_t> nextIndex{0};
                        ThreadComponentArgs *args = new ThreadComponentArgs{
                            0,
                            1,
                            nCC,
                            &nextIndex,
                                                        &bucket,
                            &edgeBuckets,
                            &components,
                        };
                        worker_component(static_cast<void *>(args));
                    }
                    else
                    {
                        std::vector<pthread_t> threads(numThreads);
                        std::atomic<size_t> nextIndex{0};

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            pthread_attr_t attr;
                            pthread_attr_init(&attr);

                            size_t stackSize = C.stackSize;
                            if (stackSize < kMinThreadStackSize)
                                stackSize = kMinThreadStackSize;
                            int err = pthread_attr_setstacksize(&attr, stackSize);
                            if (err != 0)
                            {
                                std::cerr << "[Error] pthread_attr_setstacksize("
                                          << stackSize << "): " << strerror(err) << std::endl;
                            }

                            ThreadComponentArgs *args = new ThreadComponentArgs{
                                tid,
                                numThreads,
                                nCC,
                                &nextIndex,
                                                                &bucket,
                                &edgeBuckets,
                                &components,
                            };

                            int ret = pthread_create(&threads[tid], &attr, worker_component, args);
                            if (ret != 0)
                            {
                                std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                                delete args;
                            }

                            pthread_attr_destroy(&attr);
                        }

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            pthread_join(threads[tid], nullptr);
                        }
                    }
                }

                {
                    MARK_SCOPE_MEM("sn/phase/bcTrees");

                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});

                    std::vector<std::vector<BlockPrep>> perThreadPreps(
                        std::max<size_t>(numThreads, 1));

                    if (numThreads <= 1)
                    {
                        std::atomic<size_t> nextIndex{0};
                        ThreadBcTreeArgs *args = new ThreadBcTreeArgs{
                            0,
                            1,
                            nCC,
                            &nextIndex,
                            &components,
                            &perThreadPreps};
                        worker_bcTree(static_cast<void *>(args));
                    }
                    else
                    {
                        std::vector<pthread_t> threads(numThreads);

                        std::atomic<size_t> nextIndex{0};

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            pthread_attr_t attr;
                            pthread_attr_init(&attr);

                            size_t stackSize = C.stackSize;
                            if (stackSize < kMinThreadStackSize)
                                stackSize = kMinThreadStackSize;
                            int err = pthread_attr_setstacksize(&attr, stackSize);
                            if (err != 0)
                            {
                                std::cerr << "[Error] pthread_attr_setstacksize("
                                          << stackSize << "): " << strerror(err) << std::endl;
                            }

                            ThreadBcTreeArgs *args = new ThreadBcTreeArgs{
                                tid,
                                numThreads,
                                nCC,
                                &nextIndex,
                                &components,
                                &perThreadPreps};

                            int ret = pthread_create(&threads[tid], &attr, worker_bcTree, args);
                            if (ret != 0)
                            {
                                std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                                delete args;
                            }

                            pthread_attr_destroy(&attr);
                        }

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            pthread_join(threads[tid], nullptr);
                        }
                    }

                    size_t total = 0;
                    for (auto &tp : perThreadPreps) total += tp.size();
                    blockPreps.reserve(total);
                    for (auto &tp : perThreadPreps)
                    {
                        blockPreps.insert(blockPreps.end(),
                                          std::make_move_iterator(tp.begin()),
                                          std::make_move_iterator(tp.end()));
                    }
                }

                {
                    MARK_SCOPE_MEM("sn/phase/block_SPQR_build");

                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, blockPreps.size(), numThreads});

                    if (numThreads <= 1)
                    {
                        std::atomic<size_t> nextIndex{0};
                        ThreadBlocksArgs *args = new ThreadBlocksArgs{
                            0,
                            1,
                            blockPreps.size(),
                            &nextIndex,
                                                        &blockPreps};
                        worker_block_build(static_cast<void *>(args));
                    }
                    else
                    {
                        std::vector<pthread_t> threads(numThreads);

                        std::atomic<size_t> nextIndex{0};

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            pthread_attr_t attr;
                            pthread_attr_init(&attr);

                            size_t stackSize = C.stackSize;
                            if (stackSize < kMinThreadStackSize)
                                stackSize = kMinThreadStackSize;
                            int err = pthread_attr_setstacksize(&attr, stackSize);
                            if (err != 0)
                            {
                                std::cerr << "[Error] pthread_attr_setstacksize("
                                          << stackSize << "): " << strerror(err) << std::endl;
                            }

                            ThreadBlocksArgs *args = new ThreadBlocksArgs{
                                tid,
                                numThreads,
                                blockPreps.size(),
                                &nextIndex,
                                                                &blockPreps};

                            int ret = pthread_create(&threads[tid], &attr, worker_block_build, args);
                            if (ret != 0)
                            {
                                std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                                delete args;
                            }

                            pthread_attr_destroy(&attr);
                        }

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            pthread_join(threads[tid], nullptr);
                        }
                    }
                }
            }

            {
                PhaseSampler logic_sampler(g_stats_logic);
                {
                    MARK_SCOPE_MEM("sn/phase/tips_cuts");

                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});

                    if (numThreads <= 1)
                    {
                        std::atomic<size_t> nextIndex{0};
                        ThreadTipsArgs *args = new ThreadTipsArgs{
                            0,
                            1,
                            nCC,
                            &nextIndex,
                                                        &components};
                        worker_tips(static_cast<void *>(args));
                    }
                    else
                    {
                        std::vector<pthread_t> threads(numThreads);

                        std::atomic<size_t> nextIndex{0};

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            pthread_attr_t attr;
                            pthread_attr_init(&attr);

                            size_t stackSize = C.stackSize;
                            if (stackSize < kMinThreadStackSize)
                                stackSize = kMinThreadStackSize;
                            int err = pthread_attr_setstacksize(&attr, stackSize);
                            if (err != 0)
                            {
                                std::cerr << "[Error] pthread_attr_setstacksize("
                                          << stackSize << "): " << strerror(err) << std::endl;
                            }

                            ThreadTipsArgs *args = new ThreadTipsArgs{
                                tid,
                                numThreads,
                                nCC,
                                &nextIndex,
                                                                &components};

                            int ret = pthread_create(&threads[tid], &attr, worker_tips, args);
                            if (ret != 0)
                            {
                                std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                                delete args;
                            }

                            pthread_attr_destroy(&attr);
                        }

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            pthread_join(threads[tid], nullptr);
                        }
                    }
                }

                {
                    MARK_SCOPE_MEM("sn/phase/block_SPQR_solve");

                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, blockPreps.size(), numThreads});

                    if (numThreads <= 1)
                    {
                        std::atomic<size_t> nextIndex{0};
                        ThreadBlocksArgs *args = new ThreadBlocksArgs{
                            0,
                            1,
                            blockPreps.size(),
                            &nextIndex,
                                                        &blockPreps};
                        worker_block_solve(static_cast<void *>(args));
                    }
                    else
                    {
                        std::vector<pthread_t> threads(numThreads);

                        std::atomic<size_t> nextIndex{0};

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            pthread_attr_t attr;
                            pthread_attr_init(&attr);

                            size_t stackSize = C.stackSize;
                            if (stackSize < kMinThreadStackSize)
                                stackSize = kMinThreadStackSize;
                            int err = pthread_attr_setstacksize(&attr, stackSize);
                            if (err != 0)
                            {
                                std::cerr << "[Error] pthread_attr_setstacksize("
                                          << stackSize << "): " << strerror(err) << std::endl;
                            }

                            ThreadBlocksArgs *args = new ThreadBlocksArgs{
                                tid,
                                numThreads,
                                blockPreps.size(),
                                &nextIndex,
                                                                &blockPreps};

                            int ret = pthread_create(&threads[tid], &attr, worker_block_solve, args);
                            if (ret != 0)
                            {
                                std::cerr << "Error creating pthread " << tid << ": " << strerror(ret) << std::endl;
                                delete args;
                            }

                            pthread_attr_destroy(&attr);
                        }

                        for (size_t tid = 0; tid < numThreads; ++tid)
                        {
                            pthread_join(threads[tid], nullptr);
                        }
                    }
                }
            }

            auto to_ms = [](uint64_t us)
            { return us / 1000.0; };
            auto to_mib = [](size_t bytes)
            { return bytes / (1024.0 * 1024.0); };

            auto print_phase = [&](const char *name, const PhaseStats &st)
            {
                double t_ms = to_ms(st.elapsed_us.load());
                double peak_mib = to_mib(st.peak_rss.load());
                double delta_mib = to_mib(st.peak_rss.load() > st.start_rss.load()
                                              ? st.peak_rss.load() - st.start_rss.load()
                                              : 0);
                std::cout << "[SNARLS] " << name << " : time=" << t_ms
                          << " ms, peakRSS=" << peak_mib << " MiB, peakDelta=" << delta_mib << " MiB\n";
            };

            print_snarl_type_counters();
            print_phase("I/O", g_stats_io);
            print_phase("BUILD", g_stats_build);
            print_phase("LOGIC", g_stats_logic);
        }

        void output_spqr_tree_only()
        {

            // Progress logging goes to stderr to avoid corrupting the .spqr output
            // when writing to stdout.
            std::cerr << "[spqr-tree] Writing SPQR-tree representation of the graph\n";

            auto &C = ctx();
            std::ostream *out_ptr = nullptr;
            std::ofstream out_file;

            std::vector<char> outBuffer;

            if (C.outputPath.empty())
            {
                out_ptr = &std::cout;
                std::cerr << "[spqr-tree] Output: stdout\n";
            }
            else
            {
                out_file.open(C.outputPath, std::ios::out | std::ios::binary);
                if (!out_file)
                {
                    throw std::runtime_error("Failed to open output file '" +
                                            C.outputPath + "' for writing");
                }

                outBuffer.resize(64ull * 1024ull * 1024ull);
                out_file.rdbuf()->pubsetbuf(outBuffer.data(),
                                            static_cast<std::streamsize>(outBuffer.size()));

                out_ptr = &out_file;
                std::cerr << "[spqr-tree] Output: " << C.outputPath << "\n";
            }

            std::ostream &out = *out_ptr;

            // Write header (v0.4)
            out << "H v0.4 https://github.com/sebschmi/SPQR-tree-file-format\n";


            std::string prefix = "__BF__";
            auto startsWith = [](const std::string &s, const std::string &p) -> bool
            {
                return s.size() >= p.size() && std::memcmp(s.data(), p.data(), p.size()) == 0;
            };

            bool conflict = true;
            while (conflict)
            {
                conflict = false;
                for (ogdf::node v : C.G.nodes)
                {
                    const std::string &name = C.node2name[v];
                    if (startsWith(name, prefix))
                    {
                        conflict = true;
                        break;
                    }
                }
                if (conflict)
                {
                    prefix.push_back('_');
                }
            }

            auto makeId = [&](const char *tag, uint64_t x) -> std::string
            {
                return prefix + tag + std::to_string(x);
            };

            uint64_t compCtr = 0;
            uint64_t blockCtr = 0;
            uint64_t spqrNodeCtr = 0;
            uint64_t virtEdgeCtr = 0;
            uint64_t edgeCtr = 0;

            std::string line;
            line.reserve(256);

            auto writeE = [&](const std::string &edgeName,
                            const std::string &container,
                            const std::string &uName,
                            const std::string &vName)
            {
                // E <edge> <container> <u> <v>\n
                line.clear();
                line.append("E ");
                line.append(edgeName);
                line.push_back(' ');
                line.append(container);
                line.push_back(' ');
                line.append(uName);
                line.push_back(' ');
                line.append(vName);
                line.push_back('\n');
                out.write(line.data(), static_cast<std::streamsize>(line.size()));
            };

            auto writeV = [&](const std::string &vName,
                            const std::string &a,
                            const std::string &b,
                            const std::string &uName,
                            const std::string &vName2)
            {
                // V <virtEdge> <spqrNode> <spqrNode> <u> <v>\n
                line.clear();
                line.append("V ");
                line.append(vName);
                line.push_back(' ');
                line.append(a);
                line.push_back(' ');
                line.append(b);
                line.push_back(' ');
                line.append(uName);
                line.push_back(' ');
                line.append(vName2);
                line.push_back('\n');
                out.write(line.data(), static_cast<std::streamsize>(line.size()));
            };

            ogdf::NodeArray<int> component(C.G, -1);
            int numCC = ogdf::connectedComponents(C.G, component);
            std::cerr << "[spqr-tree] Graph has " << numCC << " connected components.\n";

            // Group original nodes by component
            std::vector<std::vector<ogdf::node>> ccNodes(numCC);
            for (ogdf::node v : C.G.nodes)
            {
                ccNodes[component[v]].push_back(v);
            }
            
            const int kBlockProgressStep = 256;
            const int kLogEachBlockIfLeq = 50;
            const int kLargeBlockNodes = 200000; 
            const int kLargeBlockEdges = 500000;

            // Process each connected component
            for (int ccIdx = 0; ccIdx < numCC; ++ccIdx)
            {
                std::string compName = makeId("G", compCtr++);

                std::cerr << "[spqr-tree] CC " << (ccIdx + 1) << "/" << numCC
                        << " (" << compName << "), nodes=" << ccNodes[ccIdx].size()
                        << " ...\n";

                out << "G " << compName;
                for (ogdf::node v : ccNodes[ccIdx])
                {
                    out << " " << C.node2name[v];
                }
                out << "\n";

                ogdf::Graph ccGraph;
                ogdf::NodeArray<ogdf::node> ccToOrig(ccGraph);
                ogdf::NodeArray<ogdf::node> origToCc(C.G, nullptr);

                for (ogdf::node vOrig : ccNodes[ccIdx])
                {
                    ogdf::node vCc = ccGraph.newNode();
                    ccToOrig[vCc] = vOrig;
                    origToCc[vOrig] = vCc;
                }

                for (ogdf::node vOrig : ccNodes[ccIdx])
                {
                    C.G.forEachAdj(vOrig, [&](node /*neighbor*/, edge e) {
                        if (C.G.source(e) != vOrig)  // Only process from source side
                            return;

                        ogdf::node src = C.G.source(e);
                        ogdf::node tgt = C.G.target(e);

                        ogdf::node srcCc = origToCc[src];
                        ogdf::node tgtCc = origToCc[tgt];
                        if (srcCc && tgtCc)
                        {
                            ccGraph.newEdge(srcCc, tgtCc);
                        }
                        else
                        {
                            assert(false && "Edge with endpoint outside connected component");
                        }
                    });
                }

                std::cerr << "[spqr-tree]   subgraph |V|=" << ccGraph.numberOfNodes()
                        << " |E|=" << ccGraph.numberOfEdges() << "\n";

                if (ccGraph.numberOfNodes() == 1)
                {
                    // In v0.4: components with a single node have no blocks/cut nodes
                    // Edges are assigned to the component.
                    int localE = 0;
                    for (ogdf::edge eCc : ccGraph.edges)
                    {
                        ogdf::node uOrig = ccToOrig[ccGraph.source(eCc)];
                        ogdf::node vOrig = ccToOrig[ccGraph.target(eCc)];
                        (void)localE;

                        std::string eName = makeId("E", edgeCtr++);
                        writeE(eName, compName, C.node2name[uOrig], C.node2name[vOrig]);
                    }

                    std::cerr << "[spqr-tree]   CC done (single-node component)\n";
                    continue;
                }

                std::cerr << "[spqr-tree]   computing BC-tree...\n";
                ogdf::BCTree bc(ccGraph);

                std::map<ogdf::node, std::string> bcNodeToBlockName;

                ogdf::NodeArray<int> markCc(ccGraph, 0);
                int stampCc = 1;
                std::vector<ogdf::node> tmpNodes;
                tmpNodes.reserve(1024);

                for (ogdf::node bNode : bc.bcTree().nodes)
                {
                    if (bc.typeOfBNode(bNode) != ogdf::BCTree::BNodeType::BComp)
                        continue;

                    std::string blockName = makeId("B", blockCtr++);
                    bcNodeToBlockName[bNode] = blockName;

                    out << "B " << blockName << " " << compName;

                    tmpNodes.clear();
                    ++stampCc;

                    for (ogdf::edge hEdge : bc.hEdges(bNode))
                    {
                        ogdf::edge eCc = bc.original(hEdge);
                        if (!eCc)
                            continue;

                        ogdf::node a = ccGraph.source(eCc);
                        ogdf::node b = ccGraph.target(eCc);

                        if (markCc[a] != stampCc)
                        {
                            markCc[a] = stampCc;
                            tmpNodes.push_back(a);
                        }
                        if (markCc[b] != stampCc)
                        {
                            markCc[b] = stampCc;
                            tmpNodes.push_back(b);
                        }
                    }

                    for (ogdf::node vCc : tmpNodes)
                    {
                        ogdf::node vOrig = ccToOrig[vCc];
                        out << " " << C.node2name[vOrig];
                    }
                    out << "\n";
                }

                std::cerr << "[spqr-tree]   blocks: " << bcNodeToBlockName.size() << "\n";

                // Write C-lines (cut nodes)
                for (ogdf::node vCc : ccGraph.nodes)
                {
                    if (bc.typeOfGNode(vCc) == ogdf::BCTree::GNodeType::CutVertex)
                    {
                        ogdf::node vOrig = ccToOrig[vCc];
                        out << "C " << C.node2name[vOrig];

                        // Iterate over all B-nodes and check if this cut vertex has edges there
                        for (uint32_t bIdx = 0; bIdx < bc.numberOfBComps(); ++bIdx)
                        {
                            ogdf::node bNode{bIdx};
                            // Check if this cut vertex has edges in this block
                            bool hasEdgesInBlock = false;
                            for (ogdf::edge eCc : bc.hEdges(bNode))
                            {
                                if (ccGraph.source(eCc) == vCc || ccGraph.target(eCc) == vCc)
                                {
                                    hasEdgesInBlock = true;
                                    break;
                                }
                            }
                            if (hasEdgesInBlock)
                            {
                                auto it = bcNodeToBlockName.find(bNode);
                                assert(it != bcNodeToBlockName.end());
                                out << " " << it->second;
                            }
                        }
                        out << "\n";
                    }
                }

                const int totalBlocks = (int)bcNodeToBlockName.size();
                int processedBlocks = 0;

                for (ogdf::node bNode : bc.bcTree().nodes)
                {
                    if (bc.typeOfBNode(bNode) != ogdf::BCTree::BNodeType::BComp)
                        continue;

                    ++processedBlocks;
                    if (totalBlocks >= kBlockProgressStep &&
                        (processedBlocks % kBlockProgressStep == 0 || processedBlocks == totalBlocks))
                    {
                        std::cerr << "[spqr-tree]   processed blocks " << processedBlocks << "/" << totalBlocks << "\n";
                    }

                    const std::string &blockName = bcNodeToBlockName[bNode];

                    tmpNodes.clear();
                    ++stampCc;

                    size_t edgeCountApprox = 0;
                    for (ogdf::edge hEdge : bc.hEdges(bNode))
                    {
                        ++edgeCountApprox;
                        ogdf::edge eCc = bc.original(hEdge);
                        if (!eCc)
                            continue;

                        ogdf::node a = ccGraph.source(eCc);
                        ogdf::node b = ccGraph.target(eCc);

                        if (markCc[a] != stampCc)
                        {
                            markCc[a] = stampCc;
                            tmpNodes.push_back(a);
                        }
                        if (markCc[b] != stampCc)
                        {
                            markCc[b] = stampCc;
                            tmpNodes.push_back(b);
                        }
                    }

                    if (edgeCountApprox < 1)
                        continue;

                    // Blocks with <=2 nodes: no SPQR nodes/virtual edges in v0.4 edges assigned to the block
                    if (tmpNodes.size() < 3)
                    {
                        for (ogdf::edge hEdge : bc.hEdges(bNode))
                        {
                            ogdf::edge eCc = bc.original(hEdge);
                            if (!eCc)
                                continue;

                            ogdf::node uOrig = ccToOrig[ccGraph.source(eCc)];
                            ogdf::node vOrig = ccToOrig[ccGraph.target(eCc)];

                            std::string eName = makeId("E", edgeCtr++);
                            writeE(eName, blockName, C.node2name[uOrig], C.node2name[vOrig]);
                        }
                        continue;
                    }

                    // Build block graph
                    ogdf::Graph blockGraph;
                    ogdf::NodeArray<ogdf::node> blockToCC(blockGraph);
                    ogdf::EdgeArray<ogdf::edge> blockEdgeToCC(blockGraph);

                    // Map cc node -> block node using an unordered_map sized to block
                    // (ccGraph is huge; we cannot use NodeArray per block)
                    std::unordered_map<ogdf::node, ogdf::node> ccToBlock;
                    ccToBlock.reserve(tmpNodes.size() * 2);

                    for (ogdf::node vCc : tmpNodes)
                    {
                        ogdf::node vB = blockGraph.newNode();
                        blockToCC[vB] = vCc;
                        ccToBlock.emplace(vCc, vB);
                    }

                    // Second pass: add edges
                    for (ogdf::edge hEdge : bc.hEdges(bNode))
                    {
                        ogdf::edge eCc = bc.original(hEdge);
                        if (!eCc)
                            continue;

                        auto itS = ccToBlock.find(ccGraph.source(eCc));
                        auto itT = ccToBlock.find(ccGraph.target(eCc));
                        if (itS == ccToBlock.end() || itT == ccToBlock.end())
                            continue;

                        ogdf::edge eB = blockGraph.newEdge(itS->second, itT->second);
                        blockEdgeToCC[eB] = eCc;
                    }

                    const bool logThisBlock =
                        (totalBlocks <= kLogEachBlockIfLeq) ||
                        (blockGraph.numberOfNodes() >= kLargeBlockNodes) ||
                        (blockGraph.numberOfEdges() >= kLargeBlockEdges);

                    try
                    {
                        if (logThisBlock)
                        {
                            std::cerr << "[spqr-tree]   block " << blockName
                                    << " |V|=" << blockGraph.numberOfNodes()
                                    << " |E|=" << blockGraph.numberOfEdges()
                                    << " (computing SPQR...)\n";
                        }

                        ogdf::StaticSPQRTree spqr(blockGraph);

                        std::map<ogdf::node, std::string> spqrNodeNames;

                        ogdf::NodeArray<int> markBlk(blockGraph, 0);
                        int stampBlk = 1;

                        // Write S/P/R-lines
                        for (ogdf::node treeNode : spqr.tree().nodes)
                        {
                            char typeChar;
                            switch (spqr.typeOf(treeNode))
                            {
                            case ogdf::SPQRTree::NodeType::SNode:
                                typeChar = 'S';
                                break;
                            case ogdf::SPQRTree::NodeType::PNode:
                                typeChar = 'P';
                                break;
                            case ogdf::SPQRTree::NodeType::RNode:
                                typeChar = 'R';
                                break;
                            default:
                                typeChar = 'S';
                                break;
                            }

                            std::string spqrName = prefix + std::string(1, typeChar) + std::to_string(spqrNodeCtr++);
                            spqrNodeNames[treeNode] = spqrName;

                            out << typeChar << " " << spqrName << " " << blockName;

                            ++stampBlk;

                            const auto &skelG = spqr.skeleton(treeNode).getGraph();
                            const ogdf::Skeleton &skel = spqr.skeleton(treeNode);

                            for (ogdf::node h : skelG.nodes)
                            {
                                ogdf::node vB = skel.original(h);
                                if (!vB)
                                    continue;

                                if (markBlk[vB] == stampBlk)
                                    continue;
                                markBlk[vB] = stampBlk;

                                ogdf::node vCc = blockToCC[vB];
                                ogdf::node vOrig = ccToOrig[vCc];
                                out << " " << C.node2name[vOrig];
                            }
                            out << "\n";
                        }

                        // Write V-lines (virtual edges in SPQR tree)
                        const auto& spqrTree = spqr.tree();
                        for (ogdf::edge te : spqrTree.edges)
                        {
                            ogdf::node src = spqrTree.source(te);
                            ogdf::node tgt = spqrTree.target(te);

                            std::string vName = makeId("V", virtEdgeCtr++);

                            ogdf::edge eSrc = spqr.skeletonEdgeSrc(te);

                            if (eSrc)
                            {
                                const ogdf::Skeleton &skelSrc = spqr.skeleton(src);
                                const auto& skelGraph = skelSrc.getGraph();

                                ogdf::node uSk = skelGraph.source(eSrc);
                                ogdf::node vSk = skelGraph.target(eSrc);

                                ogdf::node uB = skelSrc.original(uSk);
                                ogdf::node vB = skelSrc.original(vSk);

                                if (uB && vB)
                                {
                                    ogdf::node uCc = blockToCC[uB];
                                    ogdf::node vCc = blockToCC[vB];
                                    ogdf::node uOrig = ccToOrig[uCc];
                                    ogdf::node vOrig = ccToOrig[vCc];

                                    writeV(vName,
                                        spqrNodeNames[src],
                                        spqrNodeNames[tgt],
                                        C.node2name[uOrig],
                                        C.node2name[vOrig]);
                                }
                                continue;
                            }

                            // Fallback (should be rare): scan src skeleton for the virtual edge to tgt
                            const ogdf::Skeleton &skelSrc = spqr.skeleton(src);
                            const auto& skelGraphSrc = skelSrc.getGraph();
                            ogdf::edge virtualEdge = nullptr;
                            for (ogdf::edge e : skelGraphSrc.edges)
                            {
                                if (skelSrc.isVirtual(e) && skelSrc.twinTreeNode(e) == tgt)
                                {
                                    virtualEdge = e;
                                    break;
                                }
                            }
                            if (!virtualEdge)
                                continue;

                            ogdf::node uB = skelSrc.original(skelGraphSrc.source(virtualEdge));
                            ogdf::node vB = skelSrc.original(skelGraphSrc.target(virtualEdge));
                            if (!uB || !vB)
                                continue;

                            ogdf::node uCc = blockToCC[uB];
                            ogdf::node vCc = blockToCC[vB];
                            ogdf::node uOrig = ccToOrig[uCc];
                            ogdf::node vOrig = ccToOrig[vCc];

                            writeV(vName,
                                spqrNodeNames[src],
                                spqrNodeNames[tgt],
                                C.node2name[uOrig],
                                C.node2name[vOrig]);
                        }

                        // Write E-lines (real edges), assigned to their containing SPQR node
                        for (ogdf::node treeNode : spqr.tree().nodes)
                        {
                            const ogdf::Skeleton &skel = spqr.skeleton(treeNode);
                            for (ogdf::edge skelEdge : skel.getGraph().edges)
                            {
                                if (skel.isVirtual(skelEdge))
                                    continue;

                                ogdf::edge eB = skel.realEdge(skelEdge);
                                if (!eB)
                                    continue;

                                ogdf::edge eCc = blockEdgeToCC[eB];
                                if (!eCc)
                                    continue;

                                ogdf::node uOrig = ccToOrig[ccGraph.source(eCc)];
                                ogdf::node vOrig = ccToOrig[ccGraph.target(eCc)];

                                std::string eName = makeId("E", edgeCtr++);
                                writeE(eName,
                                    spqrNodeNames[treeNode],
                                    C.node2name[uOrig],
                                    C.node2name[vOrig]);
                            }
                        }

                        if (logThisBlock)
                        {
                            std::cerr << "[spqr-tree]   block " << blockName << " done\n";
                        }
                    }
                    catch (...)
                    {
                        if (logThisBlock)
                        {
                            std::cerr << "[spqr-tree]   block " << blockName << " SPQR failed, skipping\n";
                        }
                        continue;
                    }
                }

                std::cerr << "[spqr-tree] CC " << (ccIdx + 1) << "/" << numCC << " done\n";
            }

            if (!out)
            {
                throw std::runtime_error("Error while writing SPQR tree to output");
            }

            std::cerr << "[spqr-tree] Finished\n";
        }

    }

    namespace ultrabubble {
        static constexpr uint8_t UB_PLUS = (uint8_t)EdgePartType::PLUS;
        static constexpr uint8_t UB_MINUS = (uint8_t)EdgePartType::MINUS;
        static void computeCutVertices(
            const Context &C,
            uint32_t N,
            std::vector<bool> &is_cut)
        {
            is_cut.assign(N, false);
            std::vector<int> disc(N, -1);
            std::vector<int> low(N, -1);

            int timer = 0;

            struct Frame {
                uint32_t v;
                uint32_t parent;      
                uint32_t edge_pos;     
                int child_count;  
            };

            std::vector<Frame> stk;
            stk.reserve(1024);

            for (uint32_t start = 0; start < N; start++) {
                if (disc[start] >= 0) continue;

                if (C.ubOffset[start] == C.ubOffset[start + 1]) {
                    disc[start] = timer++;
                    low[start] = disc[start];
                    continue;
                }

                stk.clear();
                disc[start] = low[start] = timer++;
                stk.push_back({start, UINT32_MAX, C.ubOffset[start], 0});

                while (!stk.empty()) {
                    Frame &f = stk.back();
                    uint32_t v = f.v;
                    uint32_t adj_end = C.ubOffset[v + 1];

                    if (f.edge_pos < adj_end) {
                        uint32_t u = C.ubEdges[f.edge_pos].neighbor;
                        f.edge_pos++;

                        if (disc[u] < 0) {
                            disc[u] = low[u] = timer++;
                            f.child_count++;
                            stk.push_back({u, v, C.ubOffset[u], 0});
                        } else if (u != f.parent) {
                            if (disc[u] < low[v])
                                low[v] = disc[u];
                        }
                    } else {
                        stk.pop_back();

                        if (!stk.empty()) {
                            Frame &pf = stk.back();
                            uint32_t pv = pf.v;

                            if (low[v] < low[pv])
                                low[pv] = low[v];
                            if (pf.parent == UINT32_MAX) {
                                if (pf.child_count >= 2)
                                    is_cut[pv] = true;
                            } else {
                                if (low[v] >= disc[pv])
                                    is_cut[pv] = true;
                            }
                        }
                    }
                }
            }
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
            uint32_t v,
            uint32_t u,
            uint8_t sign_at_v,
            uint8_t sign_at_u,
            const std::vector<int> &plus_dir,
            const std::vector<int> &localId,
            DirectedEdgeBuilder &out
        ) {
            if (v > u) return;

            const int vl = localId[v];
            const int ul = localId[u];

            const bool inconsistent =
                ((sign_at_u == sign_at_v) == (plus_dir[u] == plus_dir[v]));

            if (inconsistent) {
                int x = out.newIntermediate();
                if ((plus_dir[v] == 1) == (sign_at_v == UB_PLUS)) {
                    out.addEdge(vl, x);
                    out.addEdge(ul, x);
                } else {
                    out.addEdge(x, vl);
                    out.addEdge(x, ul);
                }
            } else if ((plus_dir[v] == 1) == (sign_at_v == UB_PLUS)) {
                out.addEdge(vl, ul);
            } else {
                out.addEdge(ul, vl);
            }
        }

        static void orient(
            uint32_t start,
            bool plus_enter,
            std::vector<int> &plus_dir,
            const std::vector<int> &localId,
            const Context &C,
            DirectedEdgeBuilder &out
        ) {
            struct Frame { // used to simulate the DFS recursive call stack iteratively
                uint32_t v;
                uint8_t order[2];
                int order_idx;
                uint32_t edge_pos;  

                bool pending;
                uint32_t pending_u;
                uint8_t pending_sign_v;
                uint8_t pending_sign_u;
            };

            auto make_frame = [](uint32_t v, bool pe, uint32_t adj_start) -> Frame {
                Frame f{};
                f.v = v;
                f.order[0] = UB_PLUS;
                f.order[1] = UB_MINUS;
                if (pe) std::swap(f.order[0], f.order[1]);
                f.order_idx = 0;
                f.edge_pos = adj_start;
                f.pending = false;
                return f;
            };

            std::vector<Frame> st;
            st.reserve(1024);
            st.push_back(make_frame(start, plus_enter, C.ubOffset[start]));

            while (!st.empty()) {
                Frame &f = st.back();
                uint32_t v = f.v;

                if (f.pending) {
                    emit_oriented_edge_once_local(
                        v, f.pending_u,
                        f.pending_sign_v, f.pending_sign_u,
                        plus_dir, localId, out
                    );
                    f.pending = false;
                    continue;
                }

                if (f.order_idx >= 2) {
                    st.pop_back();
                    continue;
                }

                const uint8_t wanted_sign = f.order[f.order_idx];
                const uint32_t adj_end = C.ubOffset[v + 1];

                while (f.edge_pos < adj_end) {
                    const UBEdge &ae = C.ubEdges[f.edge_pos];
                    f.edge_pos++;

                    if (ae.type_self != wanted_sign) continue;

                    uint32_t u = ae.neighbor;
                    uint8_t sign_u = ae.type_neigh;

                    if (!plus_dir[u]) {
                        if (plus_dir[v] == 1) {
                            plus_dir[u] = 1 + (int)(ae.type_self == sign_u);
                        } else {
                            plus_dir[u] = 1 + (int)(ae.type_self != sign_u);
                        }

                        f.pending = true;
                        f.pending_u = u;
                        f.pending_sign_v = ae.type_self;
                        f.pending_sign_u = sign_u;

                        st.push_back(make_frame(u, sign_u == UB_PLUS, C.ubOffset[u]));
                        goto next_iteration;
                    } else {
                        emit_oriented_edge_once_local(
                            v, u, ae.type_self, sign_u,
                            plus_dir, localId, out
                        );
                    }
                }

                f.order_idx++;
                f.edge_pos = C.ubOffset[v];

            next_iteration:
                continue;
            }
        }

        static bool choose_cut_vertex_start(
            uint32_t v,
            const std::vector<uint32_t> &cc,
            const std::vector<int> &localId,
            const Context &C,
            std::vector<int> &comp_of,        
            std::vector<uint32_t> &q)         
        {
            const int k = (int)cc.size();

            comp_of.assign(k, -1);
            comp_of[localId[v]] = -2;

            int n_comps = 0;
            q.clear();

            for (const UBEdge *it = C.adjBegin(v), *end = C.adjEnd(v);
                 it != end; ++it)
            {
                uint32_t u = it->neighbor;
                if (u == v) continue;
                if (comp_of[localId[u]] != -1) 
                continue;

                comp_of[localId[u]] = n_comps;
                q.clear();
                q.push_back(u);
                size_t qi = 0;
                while (qi < q.size()) {
                    uint32_t w = q[qi++];
                    for (const UBEdge *jt = C.adjBegin(w), *jend = C.adjEnd(w);
                         jt != jend; ++jt)
                    {
                        uint32_t x = jt->neighbor;
                        if (x == v) continue;
                        if (comp_of[localId[x]] != -1) continue;
                        comp_of[localId[x]] = n_comps;
                        q.push_back(x);
                    }
                }
                n_comps++;
            }

            if (n_comps <= 1) return true; 
            int plus_count = 0, minus_count = 0;
            std::vector<bool> plus_seen(n_comps, false);
            std::vector<bool> minus_seen(n_comps, false);

            for (const UBEdge *it = C.adjBegin(v), *end = C.adjEnd(v);
                 it != end; ++it)
            {
                uint32_t u = it->neighbor;
                if (u == v) 
                continue;
                int c = comp_of[localId[u]];
                if (c < 0) 
                continue;
                if (it->type_self == UB_PLUS) {
                    if (!plus_seen[c]) { 
                        plus_seen[c] = true; 
                        plus_count++; 
                    }
                } else {
                    if (!minus_seen[c]) { 
                        minus_seen[c] = true; 
                        minus_count++; 
                    }
                }
            }


            if (plus_count == 1 && minus_count > 1)
            return false;   
            if (minus_count == 1 && plus_count > 1)
            return true;    

            return true;
        }

        void solve() {
            std::cout << "Finding ultrabubbles...\n";
            PROFILE_FUNCTION();

            auto &C = ctx();

            const uint32_t N  = C.ubNumNodes;
            const auto &names = C.ubNodeNames;

            std::vector<bool> is_tip(N, false);
            {
                size_t tip_count = 0;
                for (uint32_t v = 0; v < N; v++) {
                    bool saw_plus = false, saw_minus = false;
                    for (const UBEdge *it = C.adjBegin(v), *end = C.adjEnd(v);
                         it != end; ++it) {
                        if (it->type_self == UB_PLUS)  saw_plus  = true;
                        if (it->type_self == UB_MINUS) saw_minus = true;
                        if (saw_plus && saw_minus) break;
                    }
                    is_tip[v] = !(saw_plus && saw_minus);
                    if (is_tip[v]) tip_count++;
                }
                std::cout << "  Tips: " << tip_count << "\n";
            }

            std::vector<bool> is_cut;
            {
                computeCutVertices(C, N, is_cut);
                size_t cut_count = 0;
                for (uint32_t v = 0; v < N; v++)
                    if (is_cut[v]) cut_count++;
                std::cout << "  Cut vertices: " << cut_count << "\n";
            }

            std::vector<bool> is_tip_saved(is_tip);

            std::vector<bool> &can_start = is_tip;
            {
                size_t start_count = 0;
                for (uint32_t v = 0; v < N; v++) {
                    can_start[v] = is_tip_saved[v] || is_cut[v];
                    if (can_start[v]) start_count++;
                }
                std::cout << "  orientation start candidates : " << start_count
                          << " / " << N << "\n";
            }

            { std::vector<bool>().swap(is_cut); }

            std::vector<int> localId(N, -1);
            std::vector<std::vector<uint32_t>> comps;
            {
                std::vector<bool> seen(N, false);
                std::vector<uint32_t> stk;
                stk.reserve(1024);
                for (uint32_t s = 0; s < N; s++) {
                    if (seen[s]) continue;
                    comps.emplace_back();
                    auto &cc = comps.back();
                    cc.reserve(256);
                    stk.clear();
                    stk.push_back(s);
                    seen[s] = true;
                    while (!stk.empty()) {
                        uint32_t v = stk.back(); stk.pop_back();
                        localId[v] = (int)cc.size();
                        cc.push_back(v);
                        for (const UBEdge *it = C.adjBegin(v),
                                          *end = C.adjEnd(v);
                             it != end; ++it) {
                            if (!seen[it->neighbor]) {
                                seen[it->neighbor] = true;
                                stk.push_back(it->neighbor);
                            }
                        }
                    }
                }
            }

            std::vector<int> plus_dir(N, 0);

            using PackedInc = std::pair<std::uint32_t, std::uint32_t>;
            std::vector<std::vector<PackedInc>> incidencesByCC(comps.size());

            std::vector<std::string> clsdTextByCC;
            if (C.clsdTrees) clsdTextByCC.resize(comps.size());

            std::atomic<size_t> next{0};
            std::atomic<bool> abort_flag{false};
            std::atomic<size_t> total_conflict_vertices{0};
            std::exception_ptr eptr = nullptr;
            std::mutex ep_mtx;

            int T = std::min<int>(C.threads, (int)comps.size());
            if (T <= 0) T = 1;

            const bool keep_trivial = C.includeTrivial;

            std::vector<std::thread> threads;
            threads.reserve(T);

            for (int t = 0; t < T; ++t) {
                threads.emplace_back([&, keep_trivial]() {
                    std::vector<int> cut_comp_scratch;
                    std::vector<uint32_t> cut_q_scratch;

                    try {
                        while (!abort_flag.load(std::memory_order_relaxed)) {
                            size_t ci = next.fetch_add(1);
                            if (ci >= comps.size()) break;

                            auto &cc = comps[ci];
                            const int k = (int)cc.size();

                            DirectedEdgeBuilder out(k);

                            for (uint32_t v : cc) {
                                if (!plus_dir[v] && can_start[v]) {
                                    bool pe;
                                    if (is_tip_saved[v]) {
                                        pe = true;
                                    } else {
                                        pe = choose_cut_vertex_start(
                                            v, cc, localId, C,
                                            cut_comp_scratch, cut_q_scratch);
                                    }
                                    plus_dir[v] = pe ? 1 : 2;
                                    orient(
                                        v, pe,
                                        plus_dir, localId, C, out
                                    );
                                }
                            }

                            for (uint32_t v : cc) {
                                if (!plus_dir[v]) {
                                    throw std::runtime_error(
                                        "Ultrabubble: orientation failed "
                                        "(unoriented node: " + names[v] + "). "
                                        "Each connected component must contain "
                                        "at least one tip or cut vertex."
                                    );
                                }
                            }

                            total_conflict_vertices.fetch_add(out.nextId - k);

                            auto &directed_edges = out.edges;
                            std::sort(directed_edges.begin(),
                                      directed_edges.end());
                            directed_edges.erase(
                                std::unique(directed_edges.begin(),
                                            directed_edges.end()),
                                directed_edges.end());

                            std::vector<ClsdTree> trees;
                            std::vector<ClsdTree>* trees_ptr =
                                (C.clsdTrees ? &trees : nullptr);
                            auto superbubbles =
                                compute_weak_superbubbles_from_edges(
                                    out.nextId, directed_edges, trees_ptr);

                            if (!keep_trivial && !superbubbles.empty()) {
                                std::vector<int> odeg(out.nextId, 0);
                                for (const auto &de : directed_edges)
                                    odeg[de.first]++;

                                superbubbles.erase(
                                    std::remove_if(superbubbles.begin(),
                                                   superbubbles.end(),
                                        [&](const std::pair<int,int> &sb) {
                                            return sb.first >= 0 &&
                                                   sb.first < (int)odeg.size() &&
                                                   odeg[sb.first] == 1 &&
                                                   std::binary_search(
                                                       directed_edges.begin(),
                                                       directed_edges.end(),
                                                       std::make_pair(sb.first, sb.second));
                                        }),
                                    superbubbles.end());
                            }

                            std::ostringstream clsd_buf;

                            if (C.clsdTrees && !trees.empty()) {
                                auto hierarchy = [&](auto&& self,
                                                     const ClsdTree& tr)
                                    -> std::vector<std::string>
                                {
                                    int xid = tr.entrance;
                                    int yid = tr.exit;

                                    const bool valid =
                                        (xid >= 0 && xid < k) &&
                                        (yid >= 0 && yid < k);

                                    std::vector<std::string> children_ser;
                                    children_ser.reserve(tr.children.size());
                                    for (const auto& ch : tr.children) {
                                        auto sub = self(self, ch);
                                        for (auto &s : sub)
                                            children_ser.emplace_back(
                                                std::move(s));
                                    }

                                    if (!valid) return children_ser;

                                    uint32_t x = cc[xid];
                                    uint32_t y = cc[yid];

                                    const std::string &xname = names[x];
                                    const std::string &yname = names[y];

                                    if (xname == "_trash" ||
                                        yname == "_trash")
                                        return children_ser;

                                    char xsign =
                                        "-+"[ plus_dir[x] == 1 ];
                                    char ysign =
                                        "+-"[ plus_dir[y] == 1 ];

                                    std::string X = xname + xsign;
                                    std::string Y = yname + ysign;

                                    std::string res;
                                    if (!children_ser.empty()) {
                                        res += "(";
                                        for (size_t i = 0;
                                             i < children_ser.size(); ++i) {
                                            res += children_ser[i];
                                            if (i + 1 < children_ser.size())
                                                res += ",";
                                        }
                                        res += ")";
                                    }
                                    res += "<" + X + "," + Y + ">";
                                    return std::vector<std::string>{
                                        std::move(res)};
                                };

                                for (const auto& tr : trees) {
                                    auto lines = hierarchy(hierarchy, tr);
                                    for (const auto& s : lines)
                                        clsd_buf << s << "\n";
                                }
                            }

                            if (C.clsdTrees)
                                clsdTextByCC[ci] = clsd_buf.str();

                            auto &inc = incidencesByCC[ci];
                            inc.reserve(superbubbles.size());

                            for (auto &sb : superbubbles) {
                                int xid = sb.first;
                                int yid = sb.second;

                                if (xid < 0 || yid < 0) continue;
                                if (xid >= k || yid >= k) continue;

                                uint32_t x = cc[xid];
                                uint32_t y = cc[yid];

                                if (names[x] == "_trash" ||
                                    names[y] == "_trash")
                                    continue;

                                const bool xplus =
                                    (plus_dir[x] == 1);
                                const bool yplus =
                                    (plus_dir[y] != 1);

                                const std::uint32_t xpack =
                                    (std::uint32_t(x) << 1) |
                                    (xplus ? 1u : 0u);
                                const std::uint32_t ypack =
                                    (std::uint32_t(y) << 1) |
                                    (yplus ? 1u : 0u);

                                const std::uint32_t xpack_flip = xpack ^ 1u;
                                const std::uint32_t ypack_flip = ypack ^ 1u;

                                std::uint32_t a1, b1;
                                if (x <= y) { a1 = xpack;      b1 = ypack; }
                                else { a1 = ypack;      b1 = xpack; }

                                std::uint32_t a2, b2;
                                if (x <= y) { a2 = xpack_flip; b2 = ypack_flip; }
                                else { a2 = ypack_flip; b2 = xpack_flip; }

                                if (std::tie(a2, b2) < std::tie(a1, b1)) {
                                    inc.emplace_back(a2, b2);
                                } else {
                                    inc.emplace_back(a1, b1);
                                }
                            }
                        }
                    } catch (...) {
                        abort_flag.store(true);
                        std::lock_guard<std::mutex> lk(ep_mtx);
                        if (!eptr) eptr = std::current_exception();
                    }
                });
            }

            for (auto &th : threads) th.join();
            if (eptr) std::rethrow_exception(eptr);

            if (C.clsdTrees) {
                std::ofstream outFile(C.clsdTreesPath);
                if (!outFile)
                    throw std::runtime_error(
                        "Cannot open CLSD trees output file: " +
                        C.clsdTreesPath);
                for (size_t ci = 0; ci < clsdTextByCC.size(); ++ci)
                    outFile << clsdTextByCC[ci];
            }

            C.ultrabubbleIncPacked.clear();
            size_t total = 0;
            for (auto &v : incidencesByCC) total += v.size();
            C.ultrabubbleIncPacked.reserve(total);

            for (size_t ci = 0; ci < incidencesByCC.size(); ++ci)
                for (auto &p : incidencesByCC[ci])
                    C.ultrabubbleIncPacked.emplace_back(p);

            std::cout << "  Conflict vertices: " << total_conflict_vertices.load() << "\n";
            std::cout << "ULTRABUBBLES found: "
                      << C.ultrabubbleIncPacked.size()
                      << (keep_trivial ? " (trivial included)" : " (trivial excluded)")
                      << "\n";
        }

    }

    namespace ultrabubble_doubled {

        static bool tryCommitSuperbubble(ogdf::node source, ogdf::node sink)
        {
            auto &C = ctx();
            if (C.node2name[source] == "_trash" ||
                C.node2name[sink] == "_trash")
            {
                return false;
            }
            C.superbubbles.emplace_back(source, sink);
            return true;
        }

        struct CcWork {
            std::vector<ogdf::node> nodes;
            std::vector<ogdf::edge> edges;
        };

        struct ThreadArgs {
            size_t tid;
            size_t numThreads;
            size_t nItems;
            std::atomic<size_t> *nextIndex;
            std::vector<CcWork> *work;
            std::vector<std::vector<std::pair<ogdf::node, ogdf::node>>> *results;
            std::vector<std::string> *clsdTextByCC;
        };

        static void worker_process_cc(ThreadArgs targs)
        {
            auto &work = *targs.work;
            auto &results = *targs.results;
            const size_t n = targs.nItems;
            const bool keep_trivial = ctx().includeTrivial;

            size_t processed = 0;

            while (true)
            {
                size_t i = targs.nextIndex->fetch_add(1);
                if (i >= n) break;

                auto &cc = work[i];
                const int nNodes = (int)cc.nodes.size();
                if (nNodes <= 1) continue;

                std::unordered_map<ogdf::node, int> nodeToId;
                nodeToId.reserve(nNodes);
                std::vector<ogdf::node> idToNode(nNodes);
                for (int j = 0; j < nNodes; j++)
                {
                    nodeToId[cc.nodes[j]] = j;
                    idToNode[j] = cc.nodes[j];
                }

                std::vector<std::pair<int,int>> directed_edges;
                directed_edges.reserve(cc.edges.size());
                auto& G = ctx().G;
                for (ogdf::edge e : cc.edges)
                {
                    int src = nodeToId[G.source(e)];
                    int tgt = nodeToId[G.target(e)];
                    directed_edges.emplace_back(src, tgt);
                }

                std::sort(directed_edges.begin(), directed_edges.end());
                directed_edges.erase(
                    std::unique(directed_edges.begin(), directed_edges.end()),
                    directed_edges.end());

                std::vector<ClsdTree> trees;
                std::vector<ClsdTree>* trees_ptr =
                    (targs.clsdTextByCC ? &trees : nullptr);
                auto superbubbles = compute_weak_superbubbles_from_edges(
                    nNodes, directed_edges, trees_ptr);

                if (!keep_trivial && !superbubbles.empty())
                {
                    std::vector<int> odeg(nNodes, 0);
                    for (const auto &de : directed_edges)
                        odeg[de.first]++;

                    superbubbles.erase(
                        std::remove_if(superbubbles.begin(),
                                       superbubbles.end(),
                            [&](const std::pair<int,int> &sb) {
                                return odeg[sb.first] == 1 &&
                                    std::binary_search(
                                        directed_edges.begin(),
                                        directed_edges.end(),
                                        std::make_pair(sb.first, sb.second));
                            }),
                        superbubbles.end());
                }

                auto &local = results[i];
                local.reserve(superbubbles.size());

                for (auto &sb : superbubbles)
                {
                    int xid = sb.first;
                    int yid = sb.second;

                    if (xid < 0 || xid >= nNodes ||
                        yid < 0 || yid >= nNodes)
                        continue;

                    ogdf::node xg = idToNode[xid];
                    ogdf::node yg = idToNode[yid];

                    const std::string &xName = ctx().node2name[xg];
                    const std::string &yName = ctx().node2name[yg];

                    if (xName == "_trash" || yName == "_trash")
                        continue;

                    local.emplace_back(xg, yg);
                }

                if (targs.clsdTextByCC && !trees.empty()) {
                    std::ostringstream clsd_buf;
                    auto hierarchy = [&](auto&& self,
                                         const ClsdTree& tr)
                        -> std::vector<std::string>
                    {
                        int xid = tr.entrance;
                        int yid = tr.exit;

                        const bool valid =
                            (xid >= 0 && xid < nNodes) &&
                            (yid >= 0 && yid < nNodes);

                        std::vector<std::string> children_ser;
                        children_ser.reserve(tr.children.size());
                        for (const auto& ch : tr.children) {
                            auto sub = self(self, ch);
                            for (auto &s : sub)
                                children_ser.emplace_back(
                                    std::move(s));
                        }

                        if (!valid) return children_ser;

                        ogdf::node xg = idToNode[xid];
                        ogdf::node yg = idToNode[yid];

                        const std::string &xName = ctx().node2name[xg];
                        const std::string &yName = ctx().node2name[yg];

                        if (xName == "_trash" ||
                            yName == "_trash")
                            return children_ser;

                        std::string res;
                        if (!children_ser.empty()) {
                            res += "(";
                            for (size_t j = 0;
                                 j < children_ser.size(); ++j) {
                                res += children_ser[j];
                                if (j + 1 < children_ser.size())
                                    res += ",";
                            }
                            res += ")";
                        }
                        res += "<" + xName + "," + yName + ">";
                        return std::vector<std::string>{
                            std::move(res)};
                    };

                    for (const auto& tr : trees) {
                        auto lines = hierarchy(hierarchy, tr);
                        for (const auto& s : lines)
                            clsd_buf << s << "\n";
                    }
                    (*targs.clsdTextByCC)[i] = clsd_buf.str();
                }

                ++processed;
            }

            std::cout << "Thread " << targs.tid
                      << " processed " << processed
                      << " CCs (doubled ultrabubbles)" << std::endl;
        }

        void solve()
        {
            std::cout << "Finding ultrabubbles (doubled mode)...\n";
            PROFILE_FUNCTION();

            auto &C = ctx();
            Graph &G = C.G;

            if (C.includeTrivial)
            {
                MARK_SCOPE_MEM("ub_doubled/findMini");
                logger::info("Finding mini-superbubbles (trivial)..");
                for (auto e : G.edges)
                {
                    auto a = G.source(e);
                    auto b = G.target(e);
                    if (G.outdeg(a) == 1 && G.indeg(b) == 1)
                    {
                        bool ok = true;
                        G.forEachAdj(b, [&](node neighbor, edge e2) {
                            if (G.source(e2) == b && G.target(e2) == a)
                            { ok = false; }
                        });
                        if (ok) tryCommitSuperbubble(a, b);
                    }
                }
                logger::info("Checked for mini-superbubbles");
            }

            NodeArray<int> compIdx(G);
            int nCC;
            {
                MARK_SCOPE_MEM("ub_doubled/ComputeCC");
                nCC = connectedComponents(G, compIdx);
            }

            std::vector<CcWork> work(nCC);
            {
                MARK_SCOPE_MEM("ub_doubled/BucketNodesEdges");
                for (ogdf::node v : G.nodes)
                    work[compIdx[v]].nodes.push_back(v);
                for (ogdf::edge e : G.edges)
                    work[compIdx[G.source(e)]].edges.push_back(e);
            }

            logger::info("Doubled ultrabubbles: {} CCs", nCC);

            std::vector<std::vector<std::pair<ogdf::node, ogdf::node>>> results(nCC);
            std::atomic<size_t> nextIndex{0};

            std::vector<std::string> clsdTextByCC;
            if (C.clsdTrees) clsdTextByCC.resize(nCC);

            size_t numThreads = std::thread::hardware_concurrency();
            numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});
            if (numThreads == 0) numThreads = 1;

            {
                MARK_SCOPE_MEM("ub_doubled/SolveCCs");

                std::vector<std::thread> threads;
                threads.reserve(numThreads);

                for (size_t tid = 0; tid < numThreads; ++tid)
                {
                    threads.emplace_back(worker_process_cc, ThreadArgs{
                        tid, numThreads, (size_t)nCC,
                        &nextIndex, &work, &results,
                        C.clsdTrees ? &clsdTextByCC : nullptr
                    });
                }

                for (auto &t : threads)
                    t.join();
            }

            {
                MARK_SCOPE_MEM("ub_doubled/CommitResults");
                for (const auto &candidates : results)
                    for (const auto &p : candidates)
                        tryCommitSuperbubble(p.first, p.second);
            }

            if (C.clsdTrees) {
                std::ofstream outFile(C.clsdTreesPath);
                if (!outFile)
                    throw std::runtime_error(
                        "Cannot open CLSD trees output file: " +
                        C.clsdTreesPath);
                for (size_t ci = 0; ci < clsdTextByCC.size(); ++ci)
                    outFile << clsdTextByCC[ci];
            }

            std::cout << "ULTRABUBBLES (doubled) found: "
                      << C.superbubbles.size()
                      << (C.includeTrivial ? " (trivial included)" : " (trivial excluded)")
                      << "\n";
        }

    }

}





int main(int argc, char **argv)
{
    rlimit rl;
    rl.rlim_cur = RLIM_INFINITY;
    rl.rlim_max = RLIM_INFINITY;
    if (setrlimit(RLIMIT_STACK, &rl) != 0)
    {
        perror("setrlimit");
    }

    TIME_BLOCK("Starting graph reading...");
    logger::init();

    readArgs(argc, argv);

    {
        std::string err;

        if (!inputFileReadable(ctx().graphPath, err))
        {
            std::cerr << "Error: cannot open input graph file '"
                      << ctx().graphPath << "' for reading: "
                      << err << "\n";
            return 1;
        }

        if (!outputParentDirWritable(ctx().outputPath, err))
        {
            std::cerr << "Error: cannot write output file '"
                      << ctx().outputPath << "': "
                      << err << "\n";
            return 1;
        }

        if (ctx().clsdTrees)
        {
            if (!outputParentDirWritable(ctx().clsdTreesPath, err))
            {
                std::cerr << "Error: cannot write CLSD trees file '"
                          << ctx().clsdTreesPath << "': "
                          << err << "\n";
                return 1;
            }
        }
    }

    {
        MARK_SCOPE_MEM("io/read_graph");
        PROFILE_BLOCK("Graph reading");
        ::GraphIO::readGraph();
    }

    if (ctx().bubbleType == Context::BubbleType::SUPERBUBBLE)
    {
        solver::superbubble::solve();
        VLOG << "[main] Superbubble solve finished. Superbubbles: "
             << ctx().superbubbles.size() << std::endl;
    }
    else if (ctx().bubbleType == Context::BubbleType::SNARL)
    {
        solver::snarls::solve();
        VLOG << "[main] Snarl solve finished. Snarls: "
             << ctx().snarls.size() << std::endl;
    }
    else if (ctx().bubbleType == Context::BubbleType::ULTRABUBBLE)
    {
        if (ctx().doubledUltrabubbles)
        {
            solver::ultrabubble_doubled::solve();
            ctx().bubbleType = Context::BubbleType::SUPERBUBBLE;
        }
        else
        {
            solver::ultrabubble::solve();
        }
    }
    else if (ctx().bubbleType == Context::BubbleType::SPQR_TREE_ONLY)
    {
        solver::snarls::output_spqr_tree_only();
        VLOG << "[main] SPQR tree solve finished." << std::endl;
    }

    {
        MARK_SCOPE_MEM("io/write_output");
        PROFILE_BLOCK("Writing output");
        TIME_BLOCK("Writing output");
        if (ctx().bubbleType == Context::BubbleType::SPQR_TREE_ONLY)
        {
        }
        else
        {
            ::GraphIO::writeSuperbubbles();
        }
    }

    if (ctx().bubbleType == Context::BubbleType::SNARL)
    {
        std::cout << "Snarls found: " << snarlsFound << std::endl;
    }

    PROFILING_REPORT();

    logger::info("Process PeakRSS: {:.2f} GiB",
                 memtime::peakRSSBytes() / (1024.0 * 1024.0 * 1024.0));

    mark::report();
    if (!g_report_json_path.empty())
    {
        mark::report_to_json(g_report_json_path);
    }

    return 0;
}