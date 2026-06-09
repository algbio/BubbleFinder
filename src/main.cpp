#include "util/spqr_rust_all.hpp"

#if defined(BF_HAVE_OPENMP) && BF_HAVE_OPENMP
    #include <omp.h>
#endif

#include <iostream>
#include <iomanip>
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
#include <memory>
#include <optional>
#include <iterator>
#include <functional>
#include <initializer_list>
#include <limits>

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
#include "util/profiling_macros.hpp"
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



using namespace spqr_compat;

static std::string g_report_json_path;

#ifndef BUBBLEFINDER_VERSION
#define BUBBLEFINDER_VERSION "unknown"
#endif

static void printVersion()
{
    std::cout << "BubbleFinder " << BUBBLEFINDER_VERSION << "\n";
}


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
        { "--spqr-weak-superbubbles", nullptr,
          "Use the SPQR-tree weak superbubble backend for ultrabubbles" },

        { "-T, --include-trivial", nullptr,
          "Include trivial bubbles in output (default: excluded; ultrabubbles, superbubbles and snarls commands)" },
        { "--compact-output-chains", nullptr,
          "Compact consecutive output bubbles into maximal chains" },

        { "--sp-compress", "<mode>",
          "Snarls SPQR compression mode: off, on, instrument, macro-direct (snarls default: macro-direct)" },
        { "--no-canonicalize-root", nullptr,
          "Skip SPQR root canonicalization (snarls default)" },
        { "--canonicalize-root", nullptr,
          "Run SPQR root canonicalization even for snarls" },

        { "--report-json", "<file>", "Write JSON metrics report" },
        { "--version", nullptr,       "Show version and exit" },
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

static void setEnvDefault(const char *name, const char *value)
{
    if (std::getenv(name) == nullptr)
    {
        setenv(name, value, 0);
    }
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
    if (args.size() == 2 && args[1] == "--version")
    {
        printVersion();
        std::exit(0);
    }

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
        C.spCompressMode = Context::SpCompressMode::MacroDirectDebug;
        C.skipCanonicalizeRoot = true;
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
        else if (s == "--spqr-weak-superbubbles")
        {
            if (C.bubbleType != Context::BubbleType::ULTRABUBBLE)
            {
                std::cerr << "Error: option '--spqr-weak-superbubbles' is only supported with the "
                             "'ultrabubbles' command.\n";
                std::exit(1);
            }
            C.spqrWeakUltrabubbles = true;
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
        else if (s == "--compact-output-chains")
        {
            C.compactOutputChains = true;
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
        else if (s == "--sp-compress")
        {
            const std::string v = nextArgOrDie(args, i, "--sp-compress");
            if (v == "on" || v == "1" || v == "true")
            {
                C.spCompressMode = Context::SpCompressMode::On;
            }
            else if (v == "off" || v == "0" || v == "false")
            {
                C.spCompressMode = Context::SpCompressMode::Off;
            }
            else if (v == "instrument")
            {
                C.spCompressMode = Context::SpCompressMode::Instrument;
            }
            else if (v == "macro-direct-debug" || v == "macro-direct")
            {
                C.spCompressMode = Context::SpCompressMode::MacroDirectDebug;
            }
            else
            {
                std::cerr << "Error: --sp-compress expects 'on', 'off', "
                             "'instrument', or 'macro-direct', got '" << v << "'.\n";
                std::exit(1);
            }
            if (C.bubbleType != Context::BubbleType::SNARL)
            {
                std::cerr << "Warning: --sp-compress only affects the 'snarls' command; "
                             "ignored for the current command.\n";
            }
        }
        else if (s == "--sp-compress-csv")
        {
            C.spCompressInstrumentCsv = nextArgOrDie(args, i, "--sp-compress-csv");
        }
        else if (s == "--no-canonicalize-root")
        {
            C.skipCanonicalizeRoot = true;
        }
        else if (s == "--canonicalize-root")
        {
            C.skipCanonicalizeRoot = false;
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

    if (C.spqrWeakUltrabubbles)
    {
        if (C.doubledUltrabubbles)
        {
            std::cerr << "Error: option '--spqr-weak-superbubbles' is not supported with '--doubled'.\n";
            std::exit(1);
        }
        if (C.clsdTrees)
        {
            std::cerr << "Error: option '--clsd-trees' is only supported by the CLSD ultrabubble backend.\n";
            std::exit(1);
        }
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

    if (C.bubbleType == Context::BubbleType::SNARL)
    {
        if (C.gfaInput)
        {
            setEnvDefault("BF_GFA_NUMERIC_PARSE", "1");
        }
        if (C.spCompressMode == Context::SpCompressMode::MacroDirectDebug)
        {
            setEnvDefault("BF_SPQR_PAR_COMBINE", "1");
            setEnvDefault("BF_MACRO_DIRECT_EMIT_MACRO_SERIES_S", "1");
            setEnvDefault("BF_MACRO_DIRECT_WITH_GCC_CUTS", "1");
        }
    }
}

// -----------------------------------------------------------------------------
// Global counters / spqr-rust accounting
// -----------------------------------------------------------------------------
static std::atomic<uint64_t> snarlsFound{0};
size_t isolatedNodesCnt = 0;

static inline void addSnarlsFound(size_t count)
{
    snarlsFound.fetch_add(static_cast<uint64_t>(count), std::memory_order_relaxed);
}

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

namespace solver
{

    // namespace superbubble {
    //     namespace
    //     {
    //         thread_local std::vector<std::pair<spqr_compat::node, spqr_compat::node>> *tls_superbubble_collector = nullptr;
    //     }

    //     static bool tryCommitSuperbubble(spqr_compat::node source, spqr_compat::node sink)
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

    //     void addSuperbubble(spqr_compat::node source, spqr_compat::node sink)
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
    //         std::vector<spqr_compat::node> nodes;
    //         std::vector<spqr_compat::edge> edges;
    //     };

    //     struct ThreadArgs {
    //         size_t tid;
    //         size_t numThreads;
    //         size_t nItems;
    //         std::atomic<size_t> *nextIndex;
    //         std::vector<CcWork> *work;
    //         std::vector<std::vector<std::pair<spqr_compat::node, spqr_compat::node>>> *results;
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

    //             std::unordered_map<spqr_compat::node, int> nodeToId;
    //             nodeToId.reserve(nNodes);
    //             std::vector<spqr_compat::node> idToNode(nNodes);
    //             for (int j = 0; j < nNodes; j++)
    //             {
    //                 nodeToId[cc.nodes[j]] = j;
    //                 idToNode[j] = cc.nodes[j];
    //             }

    //             std::vector<std::pair<int,int>> directed_edges;
    //             directed_edges.reserve(cc.edges.size());
    //             for (spqr_compat::edge e : cc.edges)
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

    //                 spqr_compat::node xg = idToNode[xid];
    //                 spqr_compat::node yg = idToNode[yid];

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
    //             for (spqr_compat::node v : G.nodes)
    //                 work[compIdx[v]].nodes.push_back(v);
    //             for (spqr_compat::edge e : G.edges)
    //                 work[compIdx[e->source()]].edges.push_back(e);
    //         }

    //         logger::info("Doubled graph: {} CCs, processing each CC entirely via CLSD", nCC);

    //         std::vector<std::vector<std::pair<spqr_compat::node, spqr_compat::node>>> results(nCC);
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
            thread_local std::vector<std::pair<spqr_compat::node, spqr_compat::node>> *tls_superbubble_collector = nullptr;
            std::mutex risky_weak_superbubble_mutex;
            std::vector<std::pair<spqr_compat::node, spqr_compat::node>> risky_weak_superbubbles;
            std::atomic<uint64_t> tip_diag_spqr_blocks{0};
            std::atomic<uint64_t> tip_diag_spqr_verts{0};
            std::atomic<uint64_t> tip_diag_internal_ss{0};
            std::atomic<uint64_t> tip_diag_blocks_with_ss{0};
            std::atomic<uint64_t> tip_diag_max_ss_one_block{0};
        }

        void clearRiskyWeakSuperbubbles()
        {
            std::lock_guard<std::mutex> lock(risky_weak_superbubble_mutex);
            risky_weak_superbubbles.clear();
        }

        void markRiskyWeakSuperbubble(spqr_compat::node source, spqr_compat::node sink)
        {
            std::lock_guard<std::mutex> lock(risky_weak_superbubble_mutex);
            risky_weak_superbubbles.emplace_back(source, sink);
        }

        std::vector<std::pair<spqr_compat::node, spqr_compat::node>> getRiskyWeakSuperbubbles()
        {
            std::lock_guard<std::mutex> lock(risky_weak_superbubble_mutex);
            return risky_weak_superbubbles;
        }

        namespace spqra_profile
        {
            inline const char *executionMode()
            {
                const char *mode = std::getenv("BF_SPQRA_EXECUTION_MODE");
                return mode ? mode : "";
            }

            inline bool aqContextRequested()
            {
                return std::strcmp(executionMode(), "aq_context") == 0;
            }

            inline bool portKernelRequested()
            {
                return std::strcmp(executionMode(), "port_kernel") == 0 ||
                       std::getenv("BF_SPQRA_PORT_KERNEL") != nullptr;
            }

            inline bool aqDirectCoreScadRequested()
            {
                const char *mode = executionMode();
                return std::strcmp(mode, "aq_context_direct") == 0 ||
                       portKernelRequested() ||
                       std::getenv("BF_SPQRA_DIRECT_CORE_SCAD") != nullptr;
            }

            inline bool aqAtomsRequested()
            {
                const char *mode = executionMode();
                return std::strcmp(mode, "aq_atoms") == 0 ||
                       std::getenv("BF_SPQRA_AQ_ATOMS") != nullptr;
            }

            inline bool aqAtomsUseSpqrTreeFallback()
            {
                return std::getenv("BF_SPQRA_ATOMS_USE_SPQRTREE") != nullptr;
            }


        }

        static bool isTrashSuperbubbleNode(spqr_compat::node v)
        {
            auto &C = ctx();
            const size_t idx = static_cast<size_t>(v.index());
            if (idx < C.isTrashNodeByIndex.size())
                return C.isTrashNodeByIndex[idx] != 0;
            auto it = C.node2name.find(v);
            return it != C.node2name.end() && it->second == "_trash";
        }

        static bool tryCommitSuperbubble(spqr_compat::node source, spqr_compat::node sink)
        {
            auto &C = ctx();
            if (isTrashSuperbubbleNode(source) || isTrashSuperbubbleNode(sink))
            {
                return false;
            }
            if (!C.weakSuperbubbles)
            {
                if (C.isEntry[source] || C.isExit[sink])
                {
                    return false;
                }
                C.isEntry[source] = true;
                C.isExit[sink] = true;
            }
            C.superbubbles.emplace_back(source, sink);
            // std::cout << "Added " << C.node2name[source] << " " << C.node2name[sink] << " as superbubble\n";
            return true;
        }
        struct SpCompressHandleDeleter
        {
            void operator()(SpCompressHandle *h) const
            {
                if (h)
                    sp_compress_free(h);
            }
        };

        struct BlockData
        {
            std::unique_ptr<spqr_compat::Graph> Gblk;
            spqr_compat::NodeArray<spqr_compat::node> toCc;
            // spqr_compat::NodeArray<spqr_compat::node> toBlk;
            spqr_compat::NodeArray<spqr_compat::node> toOrig;

            std::unique_ptr<spqr_compat::StaticSPQRTree> spqr;
            std::unique_ptr<SpCompressHandle, SpCompressHandleDeleter> spCompressHandle;
            SpCompressTreeView macroTreeView{};
            const SpqrTree *coreSpqrTree{nullptr};
            CoreScadView coreScadView{};
            SpqraMinimizerView spqraMinimizerView{};
            SpqraBehaviorAtomView spqraBehaviorAtomView{};
            const uint32_t *coreNodeInv{nullptr};
            uint32_t coreNodeInvLen{0};
            std::vector<uint32_t> spqraMinimizerLocalBlockIds;
            std::unordered_map<spqr_compat::edge, spqr_compat::edge> skel2tree; // mapping from skeleton virtual edge to tree edge
            spqr_compat::NodeArray<spqr_compat::node> parent;                   // mapping from node to parent in SPQR tree, it is possible since it is rooted,
                                                                  // parent of root is nullptr
            spqr_compat::node root{nullptr};

            spqr_compat::NodeArray<spqr_compat::node> blkToSkel;

            spqr_compat::node bNode{nullptr};

            bool isAcycic{true};

            spqr_compat::NodeArray<int> inDeg;
            spqr_compat::NodeArray<int> outDeg;
            // Cached global degrees (per block node) to avoid random access into global NodeArrays
            spqr_compat::NodeArray<int> globIn;
            spqr_compat::NodeArray<int> globOut;
            std::vector<uint32_t> edgeSrc;
            std::vector<uint32_t> edgeDst;
            uint32_t flatNodeCount{0};
            uint32_t flatEdgeCount{0};

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

        inline spqr_compat::node blockEdgeSource(const BlockData &blk, spqr_compat::edge e)
        {
            if (e.idx < blk.edgeSrc.size())
                return spqr_compat::node{blk.edgeSrc[e.idx]};
            return blk.Gblk->source(e);
        }

        inline spqr_compat::node blockEdgeTarget(const BlockData &blk, spqr_compat::edge e)
        {
            if (e.idx < blk.edgeDst.size())
                return spqr_compat::node{blk.edgeDst[e.idx]};
            return blk.Gblk->target(e);
        }

        inline uint32_t weakBlockNodeCount(const BlockData &blk)
        {
            return blk.flatNodeCount ? blk.flatNodeCount :
                static_cast<uint32_t>(blk.Gblk ? blk.Gblk->numberOfNodes() : 0);
        }

        inline uint32_t weakBlockEdgeCount(const BlockData &blk)
        {
            return blk.flatEdgeCount ? blk.flatEdgeCount :
                static_cast<uint32_t>(blk.Gblk ? blk.Gblk->numberOfEdges() : 0);
        }

        struct CcData
        {
            std::unique_ptr<spqr_compat::Graph> Gcc;
            spqr_compat::NodeArray<spqr_compat::node> toOrig;
            // spqr_compat::NodeArray<spqr_compat::node> toCopy;
            // spqr_compat::NodeArray<spqr_compat::node> toBlk;

            std::unique_ptr<spqr_compat::BCTree> bc;
            // std::vector<BlockData> blocks;
            std::vector<std::unique_ptr<BlockData>> blocks;
        };

        #ifdef BUBBLEFINDER_INSTRUMENT
                namespace profiling_patch
                {
                    std::atomic<uint64_t> reroot_time_ns{0};
                    std::atomic<uint64_t> reroot_marked_found{0};
                    std::atomic<uint64_t> reroot_fallback{0};
                    std::atomic<uint64_t> reroot_spqr_nodes_scanned{0};

                    std::atomic<uint64_t> phase2_full_count{0};
                    std::atomic<uint64_t> phase2_light_count{0};
                    std::atomic<uint64_t> phase2_pruned_tip{0};
                    std::atomic<uint64_t> phase2_pruned_cycle{0};
                    std::atomic<uint64_t> phase2_pruned_both{0};

                    std::atomic<uint64_t> blocks_with_spqr{0};
                    std::atomic<uint64_t> total_tree_nodes{0};

                    std::atomic<uint64_t> phase1_edge_dp_time_ns{0};
                    std::atomic<uint64_t> phase2_node_dp_time_ns{0};
                    std::atomic<uint64_t> phase3_collect_time_ns{0};

                    std::atomic<uint64_t> phase1_calls{0};
                    std::atomic<uint64_t> phase2_calls{0};
                    std::atomic<uint64_t> phase3_calls{0};

                    std::atomic<uint64_t> pn_A_setup_ns{0};
                    std::atomic<uint64_t> pn_B_build_ns{0};
                    std::atomic<uint64_t> pn_C_mark_ns{0};
                    std::atomic<uint64_t> pn_D_pnode_ns{0};
                    std::atomic<uint64_t> pn_E1_branch_ns{0};
                    std::atomic<uint64_t> pn_E2_branch_ns{0};
                    std::atomic<uint64_t> pn_E3_branch_ns{0};
                    std::atomic<uint64_t> pn_F_gss_ns{0};
                    std::atomic<uint64_t> pn_G_leak_ns{0};
                    std::atomic<uint64_t> pn_H_poles_ns{0};

                    std::atomic<uint64_t> pn_E1_calls{0};
                    std::atomic<uint64_t> pn_E2_calls{0};
                    std::atomic<uint64_t> pn_E3_calls{0};
                    std::atomic<uint64_t> pn_total_calls{0};
                    std::atomic<uint64_t> pn_pnode_early_returns{0};

                    std::atomic<uint64_t> pe_A_setup_ns{0};
                    std::atomic<uint64_t> pe_B_build_ns{0};
                    std::atomic<uint64_t> pe_C_pnode_extra_ns{0};
                    std::atomic<uint64_t> pe_D_leakage_ns{0};
                    std::atomic<uint64_t> pe_E_acyclic_ns{0};
                    std::atomic<uint64_t> pe_total_calls{0};

                    std::atomic<uint64_t> pn_fastpath_calls{0};

                    std::atomic<uint64_t> pn_E3_fas_dag_skipped{0};
                }
        #endif 

        static spqr_compat::node chooseSPQRRootForPruning(BlockData &blk)
        {
            if (!blk.spqr)
                return nullptr;

            const auto &T = blk.spqr->tree();
            uint64_t localScanned = 0;

            for (spqr_compat::node mu : T.nodes)
            {
                const Skeleton &skel = blk.spqr->skeleton(mu);
                const auto &skelGraph = skel.getGraph();
                ++localScanned;  // counts as one getGraph() call

                for (spqr_compat::node h : skelGraph.nodes)
                {
                    spqr_compat::node vB = skel.original(h);
                    if (vB != nullptr && (blk.globIn[vB] == 0 || blk.globOut[vB] == 0))
                    {
                        BF_INSTR(
                        profiling_patch::reroot_spqr_nodes_scanned.fetch_add(localScanned, std::memory_order_relaxed);
                        profiling_patch::reroot_marked_found.fetch_add(1, std::memory_order_relaxed);
                        )
                        return mu;
                    }
                }
            }

            BF_INSTR(
            profiling_patch::reroot_spqr_nodes_scanned.fetch_add(localScanned, std::memory_order_relaxed);
            profiling_patch::reroot_fallback.fetch_add(1, std::memory_order_relaxed);
            )
            return blk.spqr->rootNode();
        }

        static void rootSPQRTreeForDP(BlockData &blk)
        {
            SPQR_RUST_ASSERT(blk.spqr != nullptr);

            BF_INSTR(auto __t0_reroot = std::chrono::high_resolution_clock::now();)

            const auto &T = blk.spqr->tree();

            blk.root = chooseSPQRRootForPruning(blk);
            if (!blk.root)
                blk.root = blk.spqr->rootNode();

            blk.parent.init(T, nullptr);
            blk.parent[blk.root] = blk.root;

            std::vector<spqr_compat::node> stack;
            stack.reserve(T.edges.size() + 1);
            stack.push_back(blk.root);

            while (!stack.empty())
            {
                spqr_compat::node u = stack.back();
                stack.pop_back();

                T.forEachAdj(u, [&](spqr_compat::node v, spqr_compat::edge /*te*/) {
                    if (v == blk.parent[u])
                        return;
                    if (blk.parent[v] != nullptr)
                        return;

                    blk.parent[v] = u;
                    stack.push_back(v);
                });
            }

            BF_INSTR(
            auto __t1_reroot = std::chrono::high_resolution_clock::now();
            profiling_patch::reroot_time_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(__t1_reroot - __t0_reroot).count(),
                std::memory_order_relaxed);
            profiling_patch::blocks_with_spqr.fetch_add(1, std::memory_order_relaxed);
            profiling_patch::total_tree_nodes.fetch_add(T.numberOfNodes(), std::memory_order_relaxed);
            )
        }

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

        void addSuperbubble(spqr_compat::node source, spqr_compat::node sink)
        {
            if (tls_superbubble_collector)
            {
                tls_superbubble_collector->emplace_back(source, sink);
                return;
            }
            tryCommitSuperbubble(source, sink);
        }


        struct FlatDigraphScratch
        {
            static constexpr uint32_t invalid = UINT32_MAX;

            uint32_t n{0};
            std::vector<uint32_t> src;
            std::vector<uint32_t> dst;
            std::vector<uint32_t> next;
            std::vector<uint32_t> head;
            std::vector<int> indeg;
            std::vector<uint32_t> queue;

            void reset(uint32_t nodeCount, uint32_t edgeReserve)
            {
                n = nodeCount;
                src.clear();
                dst.clear();
                if (src.capacity() < edgeReserve) src.reserve(edgeReserve);
                if (dst.capacity() < edgeReserve) dst.reserve(edgeReserve);
                if (head.size() < n) head.resize(n);
                if (indeg.size() < n) indeg.resize(n);
                if (queue.size() < n) queue.resize(n);
            }

            spqr_compat::edge addEdge(spqr_compat::node u, spqr_compat::node v)
            {
                const uint32_t id = static_cast<uint32_t>(src.size());
                src.push_back(u.idx);
                dst.push_back(v.idx);
                return spqr_compat::edge{id};
            }

            spqr_compat::edge newEdge(spqr_compat::node u, spqr_compat::node v)
            {
                return addEdge(u, v);
            }

            uint32_t numberOfEdges() const
            {
                return static_cast<uint32_t>(src.size());
            }

            bool isAcyclic(uint32_t skippedEdge = invalid,
                           uint32_t skippedFrom = invalid,
                           uint32_t skippedTo = invalid)
            {
                const uint32_t m = numberOfEdges();
                if (next.size() < m) next.resize(m);
                std::fill_n(head.begin(), n, invalid);
                std::fill_n(indeg.begin(), n, 0);

                for (uint32_t i = 0; i < m; ++i)
                {
                    if (i == skippedEdge)
                        continue;
                    if (skippedFrom != invalid && src[i] == skippedFrom && dst[i] == skippedTo)
                        continue;
                    next[i] = head[src[i]];
                    head[src[i]] = i;
                    ++indeg[dst[i]];
                }

                uint32_t qHead = 0;
                uint32_t qTail = 0;
                for (uint32_t v = 0; v < n; ++v)
                    if (indeg[v] == 0)
                        queue[qTail++] = v;

                uint32_t seen = 0;
                while (qHead < qTail)
                {
                    const uint32_t u = queue[qHead++];
                    ++seen;
                    for (uint32_t e = head[u]; e != invalid; e = next[e])
                    {
                        const uint32_t v = dst[e];
                        if (--indeg[v] == 0)
                            queue[qTail++] = v;
                    }
                }
                return seen == n;
            }

            void materialize(Graph &g) const
            {
                for (uint32_t i = 0; i < n; ++i)
                    g.newNode();
                const uint32_t m = numberOfEdges();
                for (uint32_t i = 0; i < m; ++i)
                    g.newEdge(spqr_compat::node{src[i]}, spqr_compat::node{dst[i]});
            }
        };

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

            struct BranchRef
            {
                enum class Kind : uint8_t { None = 0, TreeNode = 1, MacroChild = 2 };
                Kind kind{Kind::None};
                uint32_t id{SPQR_INVALID};

                static BranchRef tree(uint32_t id) { return BranchRef{Kind::TreeNode, id}; }
                static BranchRef macro(uint32_t id) { return BranchRef{Kind::MacroChild, id}; }
                bool valid() const { return kind != Kind::None && id != SPQR_INVALID; }
            };

            inline spqr_compat::node parentOnTreeEdge(const TreeGraph &T, const BlockData &blk, spqr_compat::edge te)
            {
                spqr_compat::node a = T.source(te);
                spqr_compat::node b = T.target(te);

                if (blk.parent[a] == b)
                    return b;

                SPQR_RUST_ASSERT(blk.parent[b] == a);
                return a;
            }

            inline spqr_compat::node childOnTreeEdge(const TreeGraph &T, const BlockData &blk, spqr_compat::edge te)
            {
                spqr_compat::node a = T.source(te);
                spqr_compat::node b = T.target(te);

                if (blk.parent[a] == b)
                    return a;

                SPQR_RUST_ASSERT(blk.parent[b] == a);
                return b;
            }

            inline const EdgeDPState &stateLeavingNode(
                const EdgeArray<EdgeDP> &dp,
                const TreeGraph &T,
                const BlockData &blk,
                spqr_compat::node from,
                spqr_compat::edge te)
            {
                SPQR_RUST_ASSERT(T.source(te) == from || T.target(te) == from);
                spqr_compat::node to = (T.source(te) == from ? T.target(te) : T.source(te));
                return (blk.parent[to] == from ? dp[te].down : dp[te].up);
            }

            enum Phase2PruneMask : unsigned char
            {
                P2None = 0,
                P2Tip = 1u << 0,
                P2Cycle = 1u << 1
            };

            inline unsigned char pruneMaskFromState(const EdgeDPState &st)
            {
                unsigned char m = P2None;
                if (st.globalSourceSink) m |= P2Tip;
                if (!st.acyclic) m |= P2Cycle;
                return m;
            }

            struct LightVirtualRef
            {
                spqr_compat::node other{nullptr};
                EdgeDPState *toUpdate{nullptr};
                EdgeDPState *opposite{nullptr};
            };

            void tryBubble(const EdgeDPState &curr,
                           const EdgeDPState &back,
                           const BlockData &blk,
                           const CcData &cc,
                           bool swap,
                           bool additionalCheck);

            void processNodePrunedLight(
                spqr_compat::node curr_node,
                EdgeArray<EdgeDP> &edge_dp,
                BlockData &blk,
                unsigned char pruneMask)
            {
                SPQR_RUST_ASSERT(pruneMask != P2None);

                const StaticSPQRTree &spqr = *blk.spqr;
                const Skeleton &skel = spqr.skeleton(curr_node);
                const uint32_t nSkel = skel.numberOfNodes();



                thread_local std::vector<int> tls_localInDeg;
                thread_local std::vector<int> tls_localOutDeg;
                thread_local std::vector<LightVirtualRef> tls_virtualEdges;

                if (tls_localInDeg.size() < nSkel) {
                    tls_localInDeg.resize(nSkel);
                    tls_localOutDeg.resize(nSkel);
                }
                std::fill_n(tls_localInDeg.begin(), nSkel, 0);
                std::fill_n(tls_localOutDeg.begin(), nSkel, 0);
                tls_virtualEdges.clear();



                for (uint32_t i = 0; i < nSkel; ++i) {
                    spqr_compat::node h{i};
                    blk.blkToSkel[skel.original(h)] = h;
                }

                const spqr_compat::node parent_of_curr = blk.parent[curr_node];



                skel.forEachEdge([&](spqr_compat::edge e, spqr_compat::node u, spqr_compat::node v) {
                    if (!skel.isVirtual(e)) {
                        tls_localOutDeg[u.idx]++;
                        tls_localInDeg[v.idx]++;
                        return;
                    }

                    spqr_compat::node other = skel.twinTreeNode(e);
                    spqr_compat::edge treeE = blk.skel2tree.at(e);
                    SPQR_RUST_ASSERT(treeE != nullptr);

                    EdgeDPState *opposite =
                        (other == parent_of_curr ? &edge_dp[treeE].up
                                                 : &edge_dp[treeE].down);
                    EdgeDPState *toUpdate =
                        (other == parent_of_curr ? &edge_dp[treeE].down
                                                 : &edge_dp[treeE].up);

                    tls_virtualEdges.push_back({other, toUpdate, opposite});

                    SPQR_RUST_ASSERT(opposite->s != nullptr && opposite->t != nullptr);
                    spqr_compat::node sH = blk.blkToSkel[opposite->s];
                    spqr_compat::node tH = blk.blkToSkel[opposite->t];
                    SPQR_RUST_ASSERT(sH != nullptr && tH != nullptr);

                    tls_localOutDeg[sH.idx] += opposite->localOutS;
                    tls_localInDeg[sH.idx] += opposite->localInS;
                    tls_localOutDeg[tH.idx] += opposite->localOutT;
                    tls_localInDeg[tH.idx] += opposite->localInT;
                });



                if (spqr.typeOf(curr_node) == StaticSPQRTree::NodeType::PNode)
                {
                    spqr_compat::node pole0Blk = nullptr, pole1Blk = nullptr;
                    if (nSkel >= 1) pole0Blk = skel.original(spqr_compat::node{0u});
                    if (nSkel >= 2) pole1Blk = skel.original(spqr_compat::node{1u});

                    if (pole0Blk && pole1Blk)
                    {
                        int cnt01 = 0, cnt10 = 0;
                        skel.forEachEdge([&](spqr_compat::edge e, spqr_compat::node u, spqr_compat::node v) {
                            if (skel.isVirtual(e)) return;
                            spqr_compat::node bU = skel.original(u);
                            spqr_compat::node bV = skel.original(v);
                            if (bU == pole0Blk && bV == pole1Blk) ++cnt01;
                            else if (bU == pole1Blk && bV == pole0Blk) ++cnt10;
                        });

                        for (const auto &ve : tls_virtualEdges) {
                            EdgeDPState &st = *ve.toUpdate;
                            if (st.s == pole0Blk && st.t == pole1Blk) {
                                st.directST |= (cnt01 > 0);
                                st.directTS |= (cnt10 > 0);
                            } else if (st.s == pole1Blk && st.t == pole0Blk) {
                                st.directST |= (cnt10 > 0);
                                st.directTS |= (cnt01 > 0);
                            }
                        }
                    }
                }

                const bool propagateTip = (pruneMask & P2Tip) != 0;
                const bool propagateCycle = (pruneMask & P2Cycle) != 0;

                for (const auto &ve : tls_virtualEdges)
                {
                    EdgeDPState *BA = ve.toUpdate;
                    EdgeDPState *AB = ve.opposite;

                    if (ve.other != parent_of_curr) {
                        if (propagateCycle) BA->acyclic = false;
                        if (propagateTip) BA->globalSourceSink = true;
                    }

                    SPQR_RUST_ASSERT(BA->s != nullptr && BA->t != nullptr);
                    spqr_compat::node sH = blk.blkToSkel[BA->s];
                    spqr_compat::node tH = blk.blkToSkel[BA->t];
                    SPQR_RUST_ASSERT(sH != nullptr && tH != nullptr);

                    BA->localInS = tls_localInDeg[sH.idx] - AB->localInS;
                    BA->localOutS = tls_localOutDeg[sH.idx] - AB->localOutS;
                    BA->localInT = tls_localInDeg[tH.idx] - AB->localInT;
                    BA->localOutT = tls_localOutDeg[tH.idx] - AB->localOutT;
                }
            }

            void printAllStates(const spqr_compat::EdgeArray<EdgeDP> &edge_dp, const spqr_compat::NodeArray<NodeDPState> &node_dp, const TreeGraph &T)
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

            void printAllEdgeStates(const spqr_compat::EdgeArray<EdgeDP> &edge_dp, const TreeGraph &T)
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
                std::vector<spqr_compat::edge> &edge_order, // order of edges to process
                std::vector<spqr_compat::node> &node_order,
                const spqr_compat::NodeArray<spqr_compat::node> &parent,
                node curr)
            {
                // PROFILE_FUNCTION();
                node_order.push_back(curr);
                const TreeGraph &T = spqr.tree();
                T.forEachAdj(curr, [&](node child, edge te) {
                    if (child == parent[curr])
                        return;
                    dfsSPQR_order(spqr, edge_order, node_order, parent, child);
                    edge_order.push_back(te);
                });
            }

            inline void resetMacroState(EdgeDPState &st, node s, node t)
            {
                st = EdgeDPState{};
                st.s = s;
                st.t = t;
            }

            inline void mergeSubState(EdgeDPState &st, const EdgeDPState &sub)
            {
                if (sub.s == st.s)
                {
                    st.localOutS += sub.localOutS;
                    st.localInS += sub.localInS;
                }
                else if (sub.t == st.s)
                {
                    st.localOutS += sub.localOutT;
                    st.localInS += sub.localInT;
                }
                if (sub.s == st.t)
                {
                    st.localOutT += sub.localOutS;
                    st.localInT += sub.localInS;
                }
                else if (sub.t == st.t)
                {
                    st.localOutT += sub.localOutT;
                    st.localInT += sub.localInT;
                }
                if (sub.s == st.s && sub.t == st.t)
                {
                    st.directST |= sub.directST;
                    st.directTS |= sub.directTS;
                }
                else if (sub.s == st.t && sub.t == st.s)
                {
                    st.directST |= sub.directTS;
                    st.directTS |= sub.directST;
                }
                st.acyclic &= sub.acyclic;
                st.globalSourceSink |= sub.globalSourceSink;
                st.hasLeakage |= sub.hasLeakage;
            }

            inline void addBlockEdgeToState(EdgeDPState &st, const BlockData &blk, edge e)
            {
                node u = blockEdgeSource(blk, e);
                node v = blockEdgeTarget(blk, e);
                if (u == st.s) st.localOutS++;
                if (v == st.s) st.localInS++;
                if (u == st.t) st.localOutT++;
                if (v == st.t) st.localInT++;
                if (u == st.s && v == st.t) st.directST = true;
                if (u == st.t && v == st.s) st.directTS = true;
            }

            struct MacroEndpointDeg
            {
                uint32_t a{SPQR_INVALID};
                uint32_t b{SPQR_INVALID};
                int outA{0};
                int inA{0};
                int outB{0};
                int inB{0};
            };

            inline MacroEndpointDeg childEndpointDeg(uint32_t cref,
                                                     const std::vector<EdgeDPState> &states,
                                                     const BlockData &blk)
            {
                MacroEndpointDeg d;
                if (SP_COMPRESS_CHILD_IS_EDGE(cref))
                {
                    edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
                    d.a = blockEdgeSource(blk, e).idx;
                    d.b = blockEdgeTarget(blk, e).idx;
                    d.outA = 1;
                    d.inB = 1;
                }
                else
                {
                    const EdgeDPState &sub = states[SP_COMPRESS_CHILD_AS_MACRO(cref)];
                    d.a = sub.s.idx;
                    d.b = sub.t.idx;
                    d.outA = sub.localOutS;
                    d.inA = sub.localInS;
                    d.outB = sub.localOutT;
                    d.inB = sub.localInT;
                }
                return d;
            }

            inline bool computeSeriesLeakagePath(const SpCompressTreeView &view,
                                                 const SpCompressNode &m,
                                                 const std::vector<EdgeDPState> &states,
                                                 const BlockData &blk,
                                                 EdgeDPState &st)
            {
                if (m.children_count == 0)
                    return true;

                uint32_t cur = m.left;
                int pendingOut = 0;
                int pendingIn = 0;

                auto checkInternal = [&](uint32_t v, int out, int in) {
                    if (v == m.left || v == m.right)
                        return;
                    node bv{v};
                    if (blk.globIn[bv] != in || blk.globOut[bv] != out)
                        st.hasLeakage = true;
                    if (blk.globIn[bv] == 0 || blk.globOut[bv] == 0)
                        st.globalSourceSink = true;
                };

                for (uint32_t i = 0; i < m.children_count; ++i)
                {
                    uint32_t cref = view.children_ptr[m.children_offset + i];
                    MacroEndpointDeg d = childEndpointDeg(cref, states, blk);
                    uint32_t next = SPQR_INVALID;
                    int entryOut = 0, entryIn = 0, exitOut = 0, exitIn = 0;

                    if (d.a == cur)
                    {
                        next = d.b;
                        entryOut = d.outA;
                        entryIn = d.inA;
                        exitOut = d.outB;
                        exitIn = d.inB;
                    }
                    else if (d.b == cur)
                    {
                        next = d.a;
                        entryOut = d.outB;
                        entryIn = d.inB;
                        exitOut = d.outA;
                        exitIn = d.inA;
                    }
                    else
                    {
                        return false;
                    }

                    pendingOut += entryOut;
                    pendingIn += entryIn;
                    checkInternal(cur, pendingOut, pendingIn);
                    cur = next;
                    pendingOut = exitOut;
                    pendingIn = exitIn;
                }

                if (cur != m.right)
                    return false;
                return true;
            }

            inline void computeSeriesLeakageFallback(const SpCompressTreeView &view,
                                                     const SpCompressNode &m,
                                                     const std::vector<EdgeDPState> &states,
                                                     const BlockData &blk,
                                                     EdgeDPState &st)
            {
                thread_local std::vector<std::tuple<uint32_t, int, int>> degs;
                degs.clear();
                degs.reserve(static_cast<size_t>(m.children_count) * 2);
                auto addDeg = [&](uint32_t v, int out, int in) {
                    degs.emplace_back(v, out, in);
                };
                for (uint32_t i = 0; i < m.children_count; ++i)
                {
                    uint32_t cref = view.children_ptr[m.children_offset + i];
                    if (SP_COMPRESS_CHILD_IS_EDGE(cref))
                    {
                        edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
                        addDeg(blockEdgeSource(blk, e).idx, 1, 0);
                        addDeg(blockEdgeTarget(blk, e).idx, 0, 1);
                    }
                    else
                    {
                        const EdgeDPState &sub = states[SP_COMPRESS_CHILD_AS_MACRO(cref)];
                        addDeg(sub.s.idx, sub.localOutS, sub.localInS);
                        addDeg(sub.t.idx, sub.localOutT, sub.localInT);
                    }
                }
                std::sort(degs.begin(), degs.end(),
                          [](const auto &a, const auto &b) { return std::get<0>(a) < std::get<0>(b); });
                for (size_t i = 0; i < degs.size(); )
                {
                    uint32_t v = std::get<0>(degs[i]);
                    int out = 0, in = 0;
                    while (i < degs.size() && std::get<0>(degs[i]) == v)
                    {
                        out += std::get<1>(degs[i]);
                        in += std::get<2>(degs[i]);
                        ++i;
                    }
                    if (v == m.left || v == m.right)
                        continue;
                    node bv{v};
                    if (blk.globIn[bv] != in || blk.globOut[bv] != out)
                        st.hasLeakage = true;
                    if (blk.globIn[bv] == 0 || blk.globOut[bv] == 0)
                        st.globalSourceSink = true;
                }
            }

            enum class PFixedAggMode : uint8_t
            {
                Macro,
                Core
            };

            struct PFixedAggAtom
            {
                uint32_t goodSCount{0};
                uint32_t goodTCount{0};
                bool singleGoodValid{false};
                bool singleGoodSeriesLike{false};
                int64_t localOutSSum{0};
                int64_t localInTSum{0};
                bool good{true};
                bool sameSetOk{true};
                uint32_t branchCount{0};
                int badSourcePoleNoWeakBack{0};
                int badSinkPoleNoWeakBack{0};
                int badAcyclicNoWeakBack{0};
                int badGlobalSourceSink{0};
                int badLeakage{0};

                void addBranch(const EdgeDPState &state,
                               node bS,
                               node bT,
                               bool branchIsSeriesLike,
                               PFixedAggMode mode)
                {
                    ++branchCount;
                    const int localOutS = (state.s == bS ? state.localOutS : state.localOutT);
                    const int localInT = (state.t == bT ? state.localInT : state.localInS);
                    localOutSSum += localOutS;
                    localInTSum += localInT;

                    const int localInS = (state.s == bS ? state.localInS : state.localInT);
                    const int localOutT = (state.t == bT ? state.localOutT : state.localOutS);
                    const bool weakBackBranch =
                        ctx().weakSuperbubbles && localOutS == 0 && localInT == 0 &&
                        localInS > 0 && localOutT > 0 &&
                        state.acyclic && !state.globalSourceSink && !state.hasLeakage;
                    if (weakBackBranch)
                        return;

                    const bool hasGoodS = localOutS > 0;
                    const bool hasGoodT = localInT > 0;
                    const bool observed = hasGoodS || hasGoodT;
                    sameSetOk &= (hasGoodS == hasGoodT);
                    if (hasGoodS)
                    {
                        ++goodSCount;
                        if (goodSCount == 1)
                        {
                            singleGoodValid = true;
                            singleGoodSeriesLike = branchIsSeriesLike;
                        }
                        else
                        {
                            singleGoodValid = false;
                        }
                    }
                    if (hasGoodT)
                        ++goodTCount;

                    if (mode == PFixedAggMode::Core && !observed)
                        return;

                    const bool directFlag = state.directTS || state.directST;
	                    if ((mode == PFixedAggMode::Macro || hasGoodS) &&
	                        ((state.s == bS && state.localInS > 0) ||
	                         (state.s == bT && state.localInT > 0)))
                    {
                        good = false;
                        if (!directFlag)
                            ++badSourcePoleNoWeakBack;
                    }
                    if ((mode == PFixedAggMode::Macro || hasGoodT) &&
                        ((state.t == bT && state.localOutT > 0) ||
                         (state.t == bS && state.localOutS > 0)))
                    {
                        good = false;
                        if (!directFlag)
                            ++badSinkPoleNoWeakBack;
                    }
                    if (!state.acyclic)
                    {
                        good = false;
                        if (!directFlag)
                            ++badAcyclicNoWeakBack;
                    }
                    if (state.globalSourceSink)
                    {
                        good = false;
                        ++badGlobalSourceSink;
                    }
                    if (state.hasLeakage)
                    {
                        good = false;
                        ++badLeakage;
                    }
                }
            };

            struct PFixedAggEntry
            {
                PFixedAggAtom fwd;
                PFixedAggAtom rev;
                int64_t directST{0};
                int64_t directTS{0};
                uint8_t valid{0};
            };

            inline bool pFixedAggTracingDisabled()
            {
                return true;
            }

            inline bool corePFixedAggRequested()
            {
                return pFixedAggTracingDisabled() &&
                       (std::getenv("BF_SPQRA_CORE_P_FIXED_AGG") != nullptr ||
                        std::getenv("BF_SPQRA_P_FIXED_AGG") != nullptr);
            }

            inline bool macroPAtomTracingDisabled()
            {
                return pFixedAggTracingDisabled();
            }

            inline bool emitPFixedAggCandidate(const PFixedAggAtom &branchAgg,
                                               node bS,
                                               node bT,
                                               int64_t directST,
                                               int64_t directTS,
                                               bool qPossible,
                                               const CcData &cc,
                                               BlockData &blk)
            {
                if (!qPossible)
                    return false;

                PFixedAggAtom agg = branchAgg;
                agg.localOutSSum += directST;
                agg.localInTSum += directST;

                bool good = agg.good;
                const bool directOk = (ctx().weakSuperbubbles || directTS == 0);
                const bool nonEmptyOk = (agg.goodSCount > 0 || directST > 1);
                good &= directOk;
                const bool sameSetOk = agg.sameSetOk && (agg.goodSCount == agg.goodTCount);
                good &= sameSetOk;
                good &= nonEmptyOk;

                bool singletonOk = true;
                if (agg.goodSCount == 1 && agg.singleGoodValid)
                    singletonOk = !agg.singleGoodSeriesLike || (ctx().weakSuperbubbles && directST > 0);
                good &= singletonOk;

                const bool countsGood =
                    (agg.localOutSSum == blk.globOut[bS] && agg.localInTSum == blk.globIn[bT]);
                good &= countsGood;

                const bool onlyWeakBackFailures =
                    agg.badGlobalSourceSink == 0 && agg.badLeakage == 0 &&
                    agg.badAcyclicNoWeakBack == 0 &&
                    agg.badSourcePoleNoWeakBack == 0 &&
                    agg.badSinkPoleNoWeakBack == 0;
                if (ctx().weakSuperbubbles && !good && countsGood &&
                    directOk && sameSetOk && nonEmptyOk && singletonOk &&
                    agg.goodSCount > 1 && onlyWeakBackFailures)
                {
                    good = true;
                }

                if (good)
                {
                    node rawS = cc.toOrig[blk.toCc[bS]];
                    node rawT = cc.toOrig[blk.toCc[bT]];
                    addSuperbubble(rawS, rawT);
                }
                return true;
            }

	            inline std::vector<EdgeDPState> computeMacroStates(const SpCompressTreeView &view,
	                                                               BlockData &blk,
	                                                               const CcData *cc = nullptr,
	                                                               bool emitMacroPAtoms = false,
	                                                               std::vector<PFixedAggEntry> *macroPAgg = nullptr)
	            {
	                std::vector<EdgeDPState> states(view.macros_len);
	                if (macroPAgg)
	                    macroPAgg->assign(view.macros_len, PFixedAggEntry{});
	                for (uint32_t mid = 0; mid < view.macros_len; ++mid)
	                {
                    const SpCompressNode &m = view.macros_ptr[mid];
                    EdgeDPState &st = states[mid];
	                    resetMacroState(st, node{m.left}, node{m.right});
	                    bool pFwd = false, pBack = false;
	                    int directST = 0, directTS = 0;
	                    PFixedAggAtom atomFwd;
	                    PFixedAggAtom atomRev;
	                    const bool needPFixedAgg =
	                        m.kind == SP_COMPRESS_KIND_PARALLEL &&
	                        (emitMacroPAtoms || macroPAgg);
	                    for (uint32_t i = 0; i < m.children_count; ++i)
	                    {
                        uint32_t cref = view.children_ptr[m.children_offset + i];
                        if (SP_COMPRESS_CHILD_IS_EDGE(cref))
                        {
                            edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
                            addBlockEdgeToState(st, blk, e);
                            node u = blockEdgeSource(blk, e);
                            node v = blockEdgeTarget(blk, e);
                            if (u == st.s && v == st.t) pFwd = true;
                            if (u == st.t && v == st.s) pBack = true;
                            if (m.kind == SP_COMPRESS_KIND_PARALLEL)
                            {
                                if (u == st.s && v == st.t) directST++;
                                else directTS++;
                            }
                        }
                        else
                        {
                            const uint32_t child = SP_COMPRESS_CHILD_AS_MACRO(cref);
                            const EdgeDPState &sub = states[child];
                            mergeSubState(st, sub);
                            int dir = sub.getDirection();
                            if (sub.s == st.t && sub.t == st.s)
                                dir = -dir;
                            if (dir > 0) pFwd = true;
                            if (dir < 0) pBack = true;
	                            if (m.kind == SP_COMPRESS_KIND_PARALLEL)
	                            {
	                                const bool childIsSeries =
	                                    view.macros_ptr[child].kind == SP_COMPRESS_KIND_SERIES;
	                                if (needPFixedAgg)
	                                {
	                                    atomFwd.addBranch(sub, node{m.left}, node{m.right},
	                                                      childIsSeries, PFixedAggMode::Macro);
	                                    atomRev.addBranch(sub, node{m.right}, node{m.left},
	                                                      childIsSeries, PFixedAggMode::Macro);
	                                }
	                            }
	                        }
	                    }
	                    if (m.kind == SP_COMPRESS_KIND_PARALLEL && pFwd && pBack)
	                        st.acyclic = false;
	                    if (macroPAgg && m.kind == SP_COMPRESS_KIND_PARALLEL)
	                    {
	                        PFixedAggEntry &entry = (*macroPAgg)[mid];
	                        entry.fwd = atomFwd;
	                        entry.rev = atomRev;
	                        entry.directST = directST;
	                        entry.directTS = directTS;
	                        entry.valid = 1;
	                    }
	                    if (emitMacroPAtoms && cc && m.kind == SP_COMPRESS_KIND_PARALLEL)
	                    {
	                        const bool qPossible0 = st.localOutS == blk.globOut[node{m.left}] &&
	                                                st.localInT == blk.globIn[node{m.right}];
	                        const bool qPossible1 = st.localOutT == blk.globOut[node{m.right}] &&
	                                                st.localInS == blk.globIn[node{m.left}];
	                        emitPFixedAggCandidate(atomFwd,
	                                               node{m.left},
	                                               node{m.right},
	                                               directST,
	                                               directTS,
	                                               qPossible0,
	                                               *cc,
	                                               blk);
	                        emitPFixedAggCandidate(atomRev,
	                                               node{m.right},
	                                               node{m.left},
	                                               directTS,
	                                               directST,
	                                               qPossible1,
	                                               *cc,
	                                               blk);
	                    }
                    if (m.kind == SP_COMPRESS_KIND_SERIES)
                    {
                        if (!computeSeriesLeakagePath(view, m, states, blk, st))
                            computeSeriesLeakageFallback(view, m, states, blk, st);
                    }
                }
                return states;
            }

            struct CoreContext
            {
                const SpqrTree *T{nullptr};
                uint32_t len{0};
                uint32_t root{0};
                const uint8_t *types{nullptr};
                const uint32_t *parents{nullptr};
                const uint32_t *childrenOffsets{nullptr};
                const uint32_t *children{nullptr};
                const uint32_t *skelOffsets{nullptr};
                const SkeletonEdge *skelEdges{nullptr};
                const uint32_t *mapOffsets{nullptr};
                const uint32_t *mapping{nullptr};
                const uint32_t *coreNodeInv{nullptr};
                std::vector<uint8_t> ownedTypes;
                std::vector<uint32_t> ownedParents;
                std::vector<uint32_t> ownedChildrenOffsets;
                std::vector<uint32_t> ownedChildren;
                std::vector<uint32_t> ownedSkelOffsets;
                std::vector<SkeletonEdge> ownedSkelEdges;
                std::vector<uint32_t> ownedMapOffsets;
                std::vector<uint32_t> ownedMapping;
                std::vector<uint32_t> post;
            };

            inline bool buildCoreContext(BlockData &blk, CoreContext &out)
            {
                if (!blk.coreSpqrTree || !blk.coreNodeInv)
                    return false;
                out.T = blk.coreSpqrTree;
                out.coreNodeInv = blk.coreNodeInv;
                spqr_tree_info(out.T, &out.len, &out.root);
                if (out.len == 0)
                    return false;
                out.types = spqr_tree_node_types_raw(out.T);
                out.parents = spqr_tree_node_parents_raw(out.T);
                out.childrenOffsets = spqr_tree_children_offsets_raw(out.T);
                uint32_t clen = 0;
                out.children = spqr_tree_children_raw(out.T, &clen);
                out.skelOffsets = spqr_tree_skeleton_offsets_raw(out.T);
                uint32_t elen = 0;
                out.skelEdges = spqr_tree_skeleton_edges_raw(out.T, &elen);
                uint32_t mlen = 0;
                spqr_tree_node_mapping_raw(out.T, &out.mapOffsets, &out.mapping, &mlen);
                if (!out.types || !out.parents || !out.childrenOffsets || !out.skelOffsets ||
                    !out.skelEdges || !out.mapOffsets || !out.mapping)
                    return false;
                out.post.clear();
                out.post.reserve(out.len);
                std::vector<uint8_t> entered(out.len, 0);
                std::vector<uint32_t> stack;
                stack.reserve(out.len);
                stack.push_back(out.root);
                while (!stack.empty())
                {
                    uint32_t tn = stack.back();
                    if (!entered[tn])
                    {
                        entered[tn] = 1;
                        for (uint32_t i = out.childrenOffsets[tn]; i < out.childrenOffsets[tn + 1]; ++i)
                            stack.push_back(out.children[i]);
                    }
                    else
                    {
                        stack.pop_back();
                        out.post.push_back(tn);
                    }
                }
                return true;
            }

            inline bool buildCoreContextFromScadView(BlockData &blk, CoreContext &out)
            {
                const CoreScadView &scad = blk.coreScadView;
                if (!blk.coreNodeInv || blk.coreNodeInvLen == 0 ||
                    !scad.components_ptr || scad.components_len == 0 ||
                    !scad.edges_ptr || !scad.node_mapping_ptr)
                {
                    return false;
                }

                out = CoreContext{};
                out.T = nullptr;
                out.coreNodeInv = blk.coreNodeInv;
                out.len = scad.components_len;
                out.root = 0;

                out.ownedTypes.assign(out.len, 0);
                out.ownedParents.assign(out.len, SPQR_INVALID);
                out.ownedChildrenOffsets.assign(out.len + 1, 0);
                out.ownedSkelOffsets.assign(out.len + 1, 0);
                out.ownedMapOffsets.assign(out.len + 1, 0);
                out.ownedSkelEdges.assign(scad.edges_len, SkeletonEdge{
                                                          SPQR_INVALID,
                                                          SPQR_INVALID,
                                                          SPQR_INVALID,
                                                          SPQR_INVALID,
                                                          SPQR_INVALID,
                                                          SPQR_INVALID});
                out.ownedMapping.assign(scad.node_mapping_ptr,
                                        scad.node_mapping_ptr + scad.node_mapping_len);

                for (uint32_t ci = 0; ci < scad.components_len; ++ci)
                {
                    const FfiScadComponent &fc = scad.components_ptr[ci];
                    if (fc.edge_begin > fc.edge_end || fc.edge_end > scad.edges_len ||
                        fc.inc_begin > fc.inc_end || fc.inc_end > scad.incidences_len ||
                        fc.node_begin > fc.node_end || fc.node_end > scad.node_mapping_len)
                    {
                        return false;
                    }
                    out.ownedTypes[ci] =
                        fc.kind == 1 ? SPQR_NODE_TYPE_S :
                        fc.kind == 2 ? SPQR_NODE_TYPE_P :
                        fc.kind == 3 ? SPQR_NODE_TYPE_R : SPQR_NODE_TYPE_R;
                    out.ownedSkelOffsets[ci] = fc.edge_begin;
                    out.ownedMapOffsets[ci] = fc.node_begin;
                    if (ci + 1 == scad.components_len)
                    {
                        out.ownedSkelOffsets[ci + 1] = fc.edge_end;
                        out.ownedMapOffsets[ci + 1] = fc.node_end;
                    }
                }

                for (uint32_t ei = 0; ei < scad.edges_len; ++ei)
                {
                    const FfiScadEdge &fe = scad.edges_ptr[ei];
                    SkeletonEdge se{};
                    se.src = fe.src_local;
                    se.dst = fe.dst_local;
                    se.real_edge = (fe.kind == 3) ? SPQR_INVALID : fe.original_edge_id;
                    se.virtual_id = (fe.kind == 3) ? fe.virtual_id : SPQR_INVALID;
                    se.twin_tree_node = SPQR_INVALID;
                    se.twin_edge_idx = SPQR_INVALID;
                    out.ownedSkelEdges[ei] = se;
                }

                std::vector<std::pair<uint32_t, uint32_t>> treePairs;
                treePairs.reserve(scad.incidences_len / 2);
                if (scad.incidences_ptr)
                {
                    for (uint32_t ii = 0; ii < scad.incidences_len; ++ii)
                    {
                        const FfiScadIncidence &inc = scad.incidences_ptr[ii];
                        if (inc.component_id >= out.len ||
                            inc.twin_incidence >= scad.incidences_len)
                        {
                            return false;
                        }
                        const FfiScadIncidence &tw = scad.incidences_ptr[inc.twin_incidence];
                        if (tw.twin_incidence != ii || tw.component_id >= out.len)
                        {
                            return false;
                        }

                        const FfiScadComponent &c0 = scad.components_ptr[inc.component_id];
                        const FfiScadComponent &c1 = scad.components_ptr[tw.component_id];
                        const uint32_t ge0 = c0.edge_begin + inc.local_edge_id;
                        const uint32_t ge1 = c1.edge_begin + tw.local_edge_id;
                        if (ge0 >= scad.edges_len || ge1 >= scad.edges_len)
                            return false;
                        out.ownedSkelEdges[ge0].twin_tree_node = tw.component_id;
                        out.ownedSkelEdges[ge0].twin_edge_idx = tw.local_edge_id;

                        if (ii > inc.twin_incidence)
                            treePairs.emplace_back(inc.component_id, tw.component_id);
                    }
                }

                std::vector<uint32_t> adjCount(out.len, 0);
                for (auto [a, b] : treePairs)
                {
                    if (a == b || a >= out.len || b >= out.len)
                        return false;
                    ++adjCount[a];
                    ++adjCount[b];
                }
                std::vector<uint32_t> adjOffsets(out.len + 1, 0);
                for (uint32_t i = 0; i < out.len; ++i)
                    adjOffsets[i + 1] = adjOffsets[i] + adjCount[i];
                std::vector<uint32_t> adj(adjOffsets[out.len], SPQR_INVALID);
                std::vector<uint32_t> write = adjOffsets;
                for (auto [a, b] : treePairs)
                {
                    adj[write[a]++] = b;
                    adj[write[b]++] = a;
                }

                std::vector<uint8_t> seen(out.len, 0);
                std::vector<uint32_t> stack;
                stack.reserve(out.len);
                stack.push_back(out.root);
                seen[out.root] = 1;
                while (!stack.empty())
                {
                    uint32_t v = stack.back();
                    stack.pop_back();
                    for (uint32_t ai = adjOffsets[v]; ai < adjOffsets[v + 1]; ++ai)
                    {
                        uint32_t u = adj[ai];
                        if (seen[u])
                            continue;
                        seen[u] = 1;
                        out.ownedParents[u] = v;
                        stack.push_back(u);
                    }
                }
                for (uint32_t i = 0; i < out.len; ++i)
                    if (!seen[i] && i != out.root)
                        out.ownedParents[i] = out.root;

                std::vector<uint32_t> childCount(out.len, 0);
                for (uint32_t i = 0; i < out.len; ++i)
                {
                    uint32_t p = out.ownedParents[i];
                    if (p != SPQR_INVALID && p < out.len)
                        ++childCount[p];
                }
                for (uint32_t i = 0; i < out.len; ++i)
                    out.ownedChildrenOffsets[i + 1] = out.ownedChildrenOffsets[i] + childCount[i];
                out.ownedChildren.assign(out.ownedChildrenOffsets[out.len], SPQR_INVALID);
                write = out.ownedChildrenOffsets;
                for (uint32_t i = 0; i < out.len; ++i)
                {
                    uint32_t p = out.ownedParents[i];
                    if (p != SPQR_INVALID && p < out.len)
                        out.ownedChildren[write[p]++] = i;
                }

                out.types = out.ownedTypes.data();
                out.parents = out.ownedParents.data();
                out.childrenOffsets = out.ownedChildrenOffsets.data();
                out.children = out.ownedChildren.empty() ? nullptr : out.ownedChildren.data();
                out.skelOffsets = out.ownedSkelOffsets.data();
                out.skelEdges = out.ownedSkelEdges.empty() ? nullptr : out.ownedSkelEdges.data();
                out.mapOffsets = out.ownedMapOffsets.data();
                out.mapping = out.ownedMapping.empty() ? nullptr : out.ownedMapping.data();
                out.post.clear();
                out.post.reserve(out.len);
                std::vector<uint8_t> entered(out.len, 0);
                stack.clear();
                stack.push_back(out.root);
                while (!stack.empty())
                {
                    uint32_t tn = stack.back();
                    if (!entered[tn])
                    {
                        entered[tn] = 1;
                        for (uint32_t i = out.childrenOffsets[tn]; i < out.childrenOffsets[tn + 1]; ++i)
                            stack.push_back(out.children[i]);
                    }
                    else
                    {
                        stack.pop_back();
                        out.post.push_back(tn);
                    }
                }

                return true;
            }

            inline uint8_t scadKindToSpqrType(uint8_t kind)
            {
                return kind == 1 ? SPQR_NODE_TYPE_S :
                       kind == 2 ? SPQR_NODE_TYPE_P :
                       kind == 3 ? SPQR_NODE_TYPE_R : SPQR_NODE_TYPE_R;
            }

            inline bool scadTreeEdgeSideCanEmit(uint8_t currType, uint8_t backType)
            {
                return (currType == SPQR_NODE_TYPE_R && backType != SPQR_NODE_TYPE_P) ||
                       (currType == SPQR_NODE_TYPE_P && backType == SPQR_NODE_TYPE_S) ||
                       (currType == SPQR_NODE_TYPE_S && backType == SPQR_NODE_TYPE_P);
            }

            inline uint64_t scadTreeEdgeSideObservableMessages(uint8_t currType,
                                                               uint8_t backType)
            {
                return scadTreeEdgeSideCanEmit(currType, backType) ? 2ull : 0ull;
            }

            inline bool coreSkeletonTwin(const CoreContext &c,
                                         const SkeletonEdge &se,
                                         uint32_t &other);

            inline bool scadValidTwinPair(const CoreScadView &scad,
                                          uint32_t incidenceId,
                                          uint32_t &twinId)
            {
                if (!scad.incidences_ptr || incidenceId >= scad.incidences_len)
                    return false;
                const FfiScadIncidence &inc = scad.incidences_ptr[incidenceId];
                if (inc.component_id >= scad.components_len ||
                    inc.twin_incidence >= scad.incidences_len)
                    return false;
                const FfiScadIncidence &tw = scad.incidences_ptr[inc.twin_incidence];
                if (tw.component_id >= scad.components_len ||
                    tw.twin_incidence != incidenceId)
                    return false;
                twinId = inc.twin_incidence;
                return true;
            }

            inline uint32_t coreLocalBlockId(const CoreContext &c, uint32_t tn, uint32_t local)
            {
                uint32_t p = c.mapOffsets[tn] + local;
                return c.coreNodeInv[c.mapping[p]];
            }

            inline bool coreSkeletonChildRef(const SpCompressTreeView &view,
                                             const SkeletonEdge &se,
                                             uint32_t &cref)
            {
                if (se.real_edge == SPQR_INVALID ||
                    se.real_edge >= view.core_edges_len ||
                    !view.core_edges_ptr)
                {
                    return false;
                }
                cref = view.core_edges_ptr[se.real_edge].child;
                return true;
            }

	            inline bool coreSkeletonTwin(const CoreContext &c,
	                                         const SkeletonEdge &se,
	                                         uint32_t &other)
	            {
	                if (se.real_edge != SPQR_INVALID || se.twin_tree_node >= c.len)
                    return false;
	                other = se.twin_tree_node;
	                return true;
	            }

	            inline void buildCorePFixedAggEntry(uint32_t A,
	                                                const CoreContext &c,
	                                                const SpCompressTreeView &view,
	                                                const std::vector<EdgeDPState> &macroStates,
	                                                const std::vector<EdgeDPState> &down,
	                                                const std::vector<EdgeDPState> &up,
	                                                const BlockData &blk,
	                                                PFixedAggEntry &entry)
	            {
	                entry = PFixedAggEntry{};
	                if (A >= c.len || c.types[A] != SPQR_NODE_TYPE_P)
	                    return;

	                const node pole0{coreLocalBlockId(c, A, 0)};
	                const node pole1{coreLocalBlockId(c, A, 1)};
	                const uint32_t start = c.skelOffsets[A];
	                const uint32_t end = c.skelOffsets[A + 1];
	                for (uint32_t i = start; i < end; ++i)
	                {
	                    const SkeletonEdge &se = c.skelEdges[i];
	                    uint32_t cref = SPQR_INVALID;
	                    if (coreSkeletonChildRef(view, se, cref))
	                    {
	                        if (SP_COMPRESS_CHILD_IS_EDGE(cref))
	                        {
	                            edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
	                            node u = blockEdgeSource(blk, e);
	                            node v = blockEdgeTarget(blk, e);
	                            if (u == pole0 && v == pole1)
	                                ++entry.directST;
	                            else
	                                ++entry.directTS;
	                        }
	                        else
	                        {
	                            const uint32_t mid = SP_COMPRESS_CHILD_AS_MACRO(cref);
	                            const EdgeDPState &state = macroStates[mid];
	                            const bool seriesLike =
	                                mid < view.macros_len &&
	                                view.macros_ptr[mid].kind == SP_COMPRESS_KIND_SERIES;
	                            entry.fwd.addBranch(state, pole0, pole1, seriesLike,
	                                                PFixedAggMode::Core);
	                            entry.rev.addBranch(state, pole1, pole0, seriesLike,
	                                                PFixedAggMode::Core);
	                        }
	                        continue;
	                    }

	                    uint32_t other = SPQR_INVALID;
	                    if (!coreSkeletonTwin(c, se, other))
	                        continue;
	                    const EdgeDPState &state = (c.parents[other] == A ? down[other] : up[A]);
	                    const bool seriesLike = other < c.len && c.types[other] == SPQR_NODE_TYPE_S;
	                    entry.fwd.addBranch(state, pole0, pole1, seriesLike,
	                                        PFixedAggMode::Core);
	                    entry.rev.addBranch(state, pole1, pole0, seriesLike,
	                                        PFixedAggMode::Core);
		                }
		                entry.valid = 1;
		            }

	            inline bool tryBubbleCorePNodeGroupingFixed(uint32_t A,
	                                                         const CoreContext &c,
	                                                         const std::vector<PFixedAggEntry> *corePAgg,
	                                                         const CcData &cc,
	                                                         BlockData &blk)
	            {
	                if (!corePAgg || !pFixedAggTracingDisabled() ||
	                    A >= corePAgg->size() || A >= c.len ||
	                    c.types[A] != SPQR_NODE_TYPE_P)
	                {
	                    return false;
	                }
	                const PFixedAggEntry &entry = (*corePAgg)[A];
	                if (!entry.valid)
	                    return false;

	                const node pole0{coreLocalBlockId(c, A, 0)};
	                const node pole1{coreLocalBlockId(c, A, 1)};
	                emitPFixedAggCandidate(entry.fwd,
	                                       pole0,
	                                       pole1,
	                                       entry.directST,
	                                       entry.directTS,
	                                       true,
	                                       cc,
	                                       blk);
	                emitPFixedAggCandidate(entry.rev,
	                                       pole1,
	                                       pole0,
	                                       entry.directTS,
	                                       entry.directST,
	                                       true,
	                                       cc,
	                                       blk);
	                return true;
	            }

            inline uint32_t localEndpointOnSkeletonEdge(const CoreContext &c,
                                                        uint32_t tn,
                                                        const SkeletonEdge &se,
                                                        node b)
            {
                if (coreLocalBlockId(c, tn, se.src) == b.idx)
                    return se.src;
                if (coreLocalBlockId(c, tn, se.dst) == b.idx)
                    return se.dst;
                return SPQR_INVALID;
            }

            inline void addStateCountsAtLoc(const EdgeDPState &sub,
                                            uint32_t sLoc,
                                            uint32_t tLoc,
                                            std::vector<int> &inDeg,
                                            std::vector<int> &outDeg)
            {
                outDeg[sLoc] += sub.localOutS;
                inDeg[sLoc] += sub.localInS;
                outDeg[tLoc] += sub.localOutT;
                inDeg[tLoc] += sub.localInT;
            }

            inline void addChildRefToState(EdgeDPState &st,
                                           uint32_t cref,
                                           const std::vector<EdgeDPState> &macroStates,
                                           BlockData &blk)
            {
                if (SP_COMPRESS_CHILD_IS_EDGE(cref))
                    addBlockEdgeToState(st, blk, edge{SP_COMPRESS_CHILD_AS_EDGE(cref)});
                else
                    mergeSubState(st, macroStates[SP_COMPRESS_CHILD_AS_MACRO(cref)]);
            }

            template <typename GraphLike>
            inline uint32_t addStateToGraphOnSkeletonEdge(const EdgeDPState &sub,
                                                          const CoreContext &c,
                                                          uint32_t tn,
                                                          const SkeletonEdge &se,
                                                          GraphLike &g,
                                                          std::vector<int> &inDeg,
                                                          std::vector<int> &outDeg,
                                                          bool zeroAsForward)
            {
                uint32_t sLoc = localEndpointOnSkeletonEdge(c, tn, se, sub.s);
                uint32_t tLoc = localEndpointOnSkeletonEdge(c, tn, se, sub.t);
                if (sLoc == SPQR_INVALID || tLoc == SPQR_INVALID)
                    return SPQR_INVALID;
                int dir = sub.getDirection();
                uint32_t forwardEdge = SPQR_INVALID;
                if (dir == 1 || (dir == 0 && zeroAsForward))
                {
                    edge ge = g.newEdge(node{sLoc}, node{tLoc});
                    forwardEdge = ge.idx;
                }
                else if (dir == -1)
                {
                    g.newEdge(node{tLoc}, node{sLoc});
                }
                addStateCountsAtLoc(sub, sLoc, tLoc, inDeg, outDeg);
                return forwardEdge;
            }

            template <typename GraphLike>
            inline uint32_t addChildRefToGraphFast(uint32_t cref,
                                                   const std::vector<EdgeDPState> &macroStates,
                                                   const CoreContext &c,
                                                   uint32_t tn,
                                                   const SkeletonEdge &se,
                                                   BlockData &blk,
                                                   GraphLike &g,
                                                   std::vector<int> &inDeg,
                                                   std::vector<int> &outDeg,
                                                   bool zeroAsForward)
            {
                if (SP_COMPRESS_CHILD_IS_EDGE(cref))
                {
                    edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
                    uint32_t uLoc = localEndpointOnSkeletonEdge(c, tn, se, blockEdgeSource(blk, e));
                    uint32_t vLoc = localEndpointOnSkeletonEdge(c, tn, se, blockEdgeTarget(blk, e));
                    if (uLoc == SPQR_INVALID || vLoc == SPQR_INVALID)
                        return SPQR_INVALID;
                    edge ge = g.newEdge(node{uLoc}, node{vLoc});
                    outDeg[uLoc]++;
                    inDeg[vLoc]++;
                    return ge.idx;
                }
                const EdgeDPState &sub = macroStates[SP_COMPRESS_CHILD_AS_MACRO(cref)];
                return addStateToGraphOnSkeletonEdge(sub, c, tn, se, g, inDeg, outDeg, zeroAsForward);
            }

            inline std::vector<EdgeDPState> computeCoreDownStates(const CoreContext &c,
                                                                  const SpCompressTreeView &view,
                                                                  const std::vector<EdgeDPState> &macroStates,
                                                                  BlockData &blk,
                                                                  std::vector<NodeDPState> &nodeDp)
            {
                std::vector<EdgeDPState> down(c.len);
                for (uint32_t tn : c.post)
                {
                    if (tn == c.root)
                        continue;
                    uint32_t parent = c.parents[tn];
                    uint32_t start = c.skelOffsets[tn];
                    uint32_t end = c.skelOffsets[tn + 1];
                    uint32_t parentEdge = SPQR_INVALID;
                    for (uint32_t i = start; i < end; ++i)
                    {
                        const SkeletonEdge &se = c.skelEdges[i];
                        uint32_t other = SPQR_INVALID;
                        if (coreSkeletonTwin(c, se, other) && other == parent)
                        {
                            parentEdge = i;
                            break;
                        }
                    }
                    if (parentEdge == SPQR_INVALID)
                        continue;

                    const SkeletonEdge &pe = c.skelEdges[parentEdge];
                    node s{coreLocalBlockId(c, tn, pe.dst)};
                    node t{coreLocalBlockId(c, tn, pe.src)};
                    EdgeDPState &st = down[tn];
                    resetMacroState(st, s, t);

                    uint32_t nLocal = c.mapOffsets[tn + 1] - c.mapOffsets[tn];
                    thread_local FlatDigraphScratch tls_core_down_graph;
                    thread_local std::vector<int> tls_core_down_inDeg;
                    thread_local std::vector<int> tls_core_down_outDeg;
                    if (tls_core_down_inDeg.size() < nLocal) tls_core_down_inDeg.resize(nLocal);
                    if (tls_core_down_outDeg.size() < nLocal) tls_core_down_outDeg.resize(nLocal);
                    std::fill_n(tls_core_down_inDeg.begin(), nLocal, 0);
                    std::fill_n(tls_core_down_outDeg.begin(), nLocal, 0);
                    tls_core_down_graph.reset(nLocal, end - start);
                    for (uint32_t i = start; i < end; ++i)
                    {
                        if (i == parentEdge)
                            continue;
                        const SkeletonEdge &se = c.skelEdges[i];
                        uint32_t cref = SPQR_INVALID;
	                        if (coreSkeletonChildRef(view, se, cref))
	                        {
	                            addChildRefToState(st, cref, macroStates, blk);
	                            addChildRefToGraphFast(cref, macroStates, c, tn, se, blk,
	                                                   tls_core_down_graph,
	                                                   tls_core_down_inDeg,
	                                                   tls_core_down_outDeg,
	                                                   true);
                            if (c.types[tn] == SPQR_NODE_TYPE_P && SP_COMPRESS_CHILD_IS_EDGE(cref))
                            {
                                edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
                                node u = blockEdgeSource(blk, e);
                                node v = blockEdgeTarget(blk, e);
                                if (u == st.s && v == st.t) st.directST = true;
                                if (u == st.t && v == st.s) st.directTS = true;
                            }
                        }
                        else
                        {
                            uint32_t child = SPQR_INVALID;
	                            if (!coreSkeletonTwin(c, se, child))
	                                continue;
	                            mergeSubState(st, down[child]);
	                            addStateToGraphOnSkeletonEdge(down[child], c, tn, se,
	                                                          tls_core_down_graph,
	                                                          tls_core_down_inDeg,
	                                                          tls_core_down_outDeg,
	                                                          false);
	                        }
	                    }

                    uint32_t sLoc = pe.dst;
                    uint32_t tLoc = pe.src;
	                    st.localOutS = tls_core_down_outDeg[sLoc];
	                    st.localInS = tls_core_down_inDeg[sLoc];
	                    st.localOutT = tls_core_down_outDeg[tLoc];
	                    st.localInT = tls_core_down_inDeg[tLoc];

                    for (uint32_t li = 0; li < nLocal; ++li)
                    {
	                        node b{coreLocalBlockId(c, tn, li)};
	                        if (b == st.s || b == st.t)
	                            continue;
	                        if (blk.globIn[b] != tls_core_down_inDeg[li] ||
	                            blk.globOut[b] != tls_core_down_outDeg[li])
	                            st.hasLeakage = true;
                        if (blk.globIn[b] == 0 || blk.globOut[b] == 0)
                            st.globalSourceSink = true;
                    }

                    if (st.acyclic)
                    {
                        st.acyclic &= tls_core_down_graph.isAcyclic();
                    }
                    if (!st.acyclic)
                    {
                        nodeDp[parent].outgoingCyclesCount++;
                        nodeDp[parent].lastCycleNode = node{tn};
                    }
                    if (st.globalSourceSink)
                    {
                        nodeDp[parent].outgoingSourceSinkCount++;
                        nodeDp[parent].lastSourceSinkNode = node{tn};
                    }
                    if (st.hasLeakage)
                    {
                        nodeDp[parent].outgoingLeakageCount++;
                        nodeDp[parent].lastLeakageNode = node{tn};
                    }
                }
                return down;
            }

	            inline bool coreTreeBubbleCurrentCanEmit(const EdgeDPState &curr,
	                                                     const EdgeDPState &back)
	            {
	                return curr.s && curr.t && back.s && back.t &&
	                       curr.acyclic && !curr.globalSourceSink && !curr.hasLeakage;
	            }

	            inline void tryCoreTreeBubblePair(const EdgeDPState &curr,
	                                              const EdgeDPState &back,
	                                              const CcData &cc,
	                                              BlockData &blk,
	                                              bool additionalCheck)
	            {
	                if (!coreTreeBubbleCurrentCanEmit(curr, back))
	                    return;
	                tryBubble(curr, back, blk, cc, false, additionalCheck);
	                tryBubble(curr, back, blk, cc, true, additionalCheck);
	            }

	            inline void tryCoreTreeEdgeBubbles(uint32_t child,
	                                           uint32_t parent,
	                                           const CoreContext &c,
	                                               const EdgeDPState &downSt,
	                                               const EdgeDPState &upSt,
	                                               const CcData &cc,
                                               BlockData &blk)
            {
                if (child >= c.len || parent >= c.len)
                    return;
                uint8_t parentType = c.types[parent];
                uint8_t childType = c.types[child];
	                if ((childType == SPQR_NODE_TYPE_R && parentType != SPQR_NODE_TYPE_P) ||
	                    (childType == SPQR_NODE_TYPE_P && parentType == SPQR_NODE_TYPE_S))
	                {
	                    bool check = childType == SPQR_NODE_TYPE_P;
	                    tryCoreTreeBubblePair(downSt, upSt, cc, blk,
	                                          check);
	                }
	                if (childType == SPQR_NODE_TYPE_S && parentType == SPQR_NODE_TYPE_P)
	                {
	                    tryCoreTreeBubblePair(downSt, upSt, cc, blk,
	                                          true);
	                }
	                if ((parentType == SPQR_NODE_TYPE_R && childType != SPQR_NODE_TYPE_P) ||
	                    (parentType == SPQR_NODE_TYPE_P && childType == SPQR_NODE_TYPE_S))
	                {
	                    bool check = parentType == SPQR_NODE_TYPE_P;
	                    tryCoreTreeBubblePair(upSt, downSt, cc, blk,
	                                          check);
	                }
	                if (parentType == SPQR_NODE_TYPE_S && childType == SPQR_NODE_TYPE_P)
	                {
	                    tryCoreTreeBubblePair(upSt, downSt, cc, blk,
	                                          true);
		                }
		            }

	            inline bool minimizerDirectCollectRequested()
	            {
	                return std::getenv("BF_SPQRA_MINIMIZER_DIRECT_COLLECT") != nullptr;
	            }

	            inline void tryCoreTreeEdgeBubblesByTypes(uint8_t childType,
	                                                      uint8_t parentType,
	                                                      const EdgeDPState &downSt,
	                                                      const EdgeDPState &upSt,
	                                                      const CcData &cc,
	                                                      BlockData &blk)
	            {
	                if ((childType == SPQR_NODE_TYPE_R && parentType != SPQR_NODE_TYPE_P) ||
	                    (childType == SPQR_NODE_TYPE_P && parentType == SPQR_NODE_TYPE_S))
	                {
	                    const bool check = childType == SPQR_NODE_TYPE_P;
	                    tryCoreTreeBubblePair(downSt, upSt, cc, blk,
	                                          check);
	                }
	                if (childType == SPQR_NODE_TYPE_S && parentType == SPQR_NODE_TYPE_P)
	                {
	                    tryCoreTreeBubblePair(downSt, upSt, cc, blk,
	                                          true);
	                }
	                if ((parentType == SPQR_NODE_TYPE_R && childType != SPQR_NODE_TYPE_P) ||
	                    (parentType == SPQR_NODE_TYPE_P && childType == SPQR_NODE_TYPE_S))
	                {
	                    const bool check = parentType == SPQR_NODE_TYPE_P;
	                    tryCoreTreeBubblePair(upSt, downSt, cc, blk,
	                                          check);
	                }
	                if (parentType == SPQR_NODE_TYPE_S && childType == SPQR_NODE_TYPE_P)
	                {
	                    tryCoreTreeBubblePair(upSt, downSt, cc, blk,
	                                          true);
	                }
	            }

	            inline const EdgeDPState *minimizerPortMessage(uint32_t port,
	                                                           const CoreScadView &scad,
	                                                           const SpqraMinimizerView &minView,
	                                                           const std::vector<EdgeDPState> &down,
	                                                           const std::vector<EdgeDPState> &up)
	            {
	                if (!scad.incidences_ptr || !minView.components_ptr ||
	                    port >= scad.incidences_len)
	                    return nullptr;
	                const FfiScadIncidence &inc = scad.incidences_ptr[port];
	                if (inc.twin_incidence >= scad.incidences_len ||
	                    inc.component_id >= minView.components_len)
	                    return nullptr;
	                const FfiScadIncidence &tw = scad.incidences_ptr[inc.twin_incidence];
	                if (tw.component_id >= minView.components_len)
	                    return nullptr;
	                const uint32_t a = inc.component_id;
	                const uint32_t b = tw.component_id;
	                if (a < down.size() && minView.components_ptr[a].parent == b)
	                    return &down[a];
	                if (b < up.size() && minView.components_ptr[b].parent == a)
	                    return &up[b];
	                return nullptr;
	            }

	            inline const EdgeDPState *minimizerViewEdgeMessage(uint32_t component,
	                                                               const FfiSpqraMinimizerEdge &me,
	                                                               const SpqraMinimizerView &minView,
	                                                               const std::vector<EdgeDPState> &down,
	                                                               const std::vector<EdgeDPState> &up)
	            {
	                if (!minView.components_ptr || component >= minView.components_len ||
	                    (me.flags & SPQRA_MIN_EDGE_VIRTUAL) == 0 ||
	                    me.twin_component >= minView.components_len)
	                {
	                    return nullptr;
	                }
	                const uint32_t other = me.twin_component;
	                if (component < down.size() &&
	                    minView.components_ptr[component].parent == other)
	                {
	                    return &down[component];
	                }
	                if (other < up.size() &&
	                    minView.components_ptr[other].parent == component)
	                {
	                    return &up[other];
	                }
	                return nullptr;
	            }

	            inline uint32_t minimizerLocalBlockId(const SpqraMinimizerView &minView,
	                                                  const BlockData &blk,
	                                                  const FfiSpqraMinimizerComponent &comp,
	                                                  uint32_t local)
	            {
	                if (!minView.node_mapping_ptr || !blk.coreNodeInv ||
	                    comp.node_begin > comp.node_end ||
	                    local >= comp.node_end - comp.node_begin)
	                {
	                    return SPQR_INVALID;
	                }
	                const uint32_t mi = comp.node_begin + local;
	                if (mi < blk.spqraMinimizerLocalBlockIds.size())
	                    return blk.spqraMinimizerLocalBlockIds[mi];
	                if (mi >= minView.node_mapping_len)
	                    return SPQR_INVALID;
	                const uint32_t coreId = minView.node_mapping_ptr[mi];
	                if (coreId >= blk.coreNodeInvLen)
	                    return SPQR_INVALID;
	                return blk.coreNodeInv[coreId];
	            }

	            inline void populateMinimizerLocalBlockIds(BlockData &blk)
	            {
	                blk.spqraMinimizerLocalBlockIds.clear();
	                const SpqraMinimizerView &minView = blk.spqraMinimizerView;
	                if (!minView.node_mapping_ptr || minView.node_mapping_len == 0 ||
	                    !blk.coreNodeInv || blk.coreNodeInvLen == 0)
	                {
	                    return;
	                }
	                blk.spqraMinimizerLocalBlockIds.resize(minView.node_mapping_len,
	                                                       SPQR_INVALID);
	                for (uint32_t i = 0; i < minView.node_mapping_len; ++i)
	                {
	                    const uint32_t coreId = minView.node_mapping_ptr[i];
	                    if (coreId < blk.coreNodeInvLen)
	                        blk.spqraMinimizerLocalBlockIds[i] = blk.coreNodeInv[coreId];
	                }
	            }

	            inline bool minimizerMessageKernelRequested()
	            {
	                return std::getenv("BF_SPQRA_MINIMIZER_MESSAGE_KERNEL") != nullptr;
	            }

	            inline bool minimizerViewCollectRequested()
	            {
	                return std::getenv("BF_SPQRA_MINIMIZER_VIEW_COLLECT") != nullptr;
	            }

	            inline bool emitNestedAtomCoreEdgesRequested()
	            {
	                return std::getenv("BF_SPQRA_EMIT_NESTED_ATOM_CORE_EDGES") != nullptr;
	            }

	            inline uint32_t minimizerLocalEndpointOnEdge(const SpqraMinimizerView &minView,
	                                                         const BlockData &blk,
	                                                         const FfiSpqraMinimizerComponent &comp,
	                                                         const FfiSpqraMinimizerEdge &me,
	                                                         node b)
	            {
	                if (minimizerLocalBlockId(minView, blk, comp, me.src_local) == b.idx)
	                    return me.src_local;
	                if (minimizerLocalBlockId(minView, blk, comp, me.dst_local) == b.idx)
	                    return me.dst_local;
	                return SPQR_INVALID;
	            }

	            template <typename GraphLike>
	            inline uint32_t addStateToGraphOnMinimizerEdge(const EdgeDPState &sub,
	                                                           const SpqraMinimizerView &minView,
	                                                           const BlockData &blk,
	                                                           const FfiSpqraMinimizerComponent &comp,
	                                                           const FfiSpqraMinimizerEdge &me,
	                                                           GraphLike &g,
	                                                           std::vector<int> &inDeg,
	                                                           std::vector<int> &outDeg,
	                                                           bool zeroAsForward)
	            {
	                const uint32_t sLoc =
	                    minimizerLocalEndpointOnEdge(minView, blk, comp, me, sub.s);
	                const uint32_t tLoc =
	                    minimizerLocalEndpointOnEdge(minView, blk, comp, me, sub.t);
	                if (sLoc == SPQR_INVALID || tLoc == SPQR_INVALID)
	                    return SPQR_INVALID;
	                int dir = sub.getDirection();
	                uint32_t forwardEdge = SPQR_INVALID;
	                if (dir == 1 || (dir == 0 && zeroAsForward))
	                {
	                    edge ge = g.newEdge(node{sLoc}, node{tLoc});
	                    forwardEdge = ge.idx;
	                }
	                if (dir == -1)
	                {
	                    g.newEdge(node{tLoc}, node{sLoc});
	                }
	                addStateCountsAtLoc(sub, sLoc, tLoc, inDeg, outDeg);
	                return forwardEdge;
	            }

	            template <typename GraphLike>
	            inline uint32_t addChildRefToGraphFastMinimizer(uint32_t cref,
	                                                            const std::vector<EdgeDPState> &macroStates,
	                                                            const SpqraMinimizerView &minView,
	                                                            const BlockData &blk,
	                                                            const FfiSpqraMinimizerComponent &comp,
	                                                            const FfiSpqraMinimizerEdge &me,
	                                                            GraphLike &g,
	                                                            std::vector<int> &inDeg,
	                                                            std::vector<int> &outDeg,
	                                                            bool zeroAsForward)
	            {
	                if (SP_COMPRESS_CHILD_IS_EDGE(cref))
	                {
	                    edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
	                    const uint32_t uLoc =
	                        minimizerLocalEndpointOnEdge(minView, blk, comp, me,
	                                                     blockEdgeSource(blk, e));
	                    const uint32_t vLoc =
	                        minimizerLocalEndpointOnEdge(minView, blk, comp, me,
	                                                     blockEdgeTarget(blk, e));
	                    if (uLoc == SPQR_INVALID || vLoc == SPQR_INVALID)
	                        return SPQR_INVALID;
	                    edge ge = g.newEdge(node{uLoc}, node{vLoc});
	                    outDeg[uLoc]++;
	                    inDeg[vLoc]++;
	                    return ge.idx;
	                }
	                const uint32_t mid = SP_COMPRESS_CHILD_AS_MACRO(cref);
	                if (mid >= macroStates.size())
	                    return SPQR_INVALID;
	                return addStateToGraphOnMinimizerEdge(macroStates[mid], minView, blk, comp, me,
	                                                      g, inDeg, outDeg, zeroAsForward);
	            }

	            struct SpqraBehaviorAtomStates
	            {
	                std::vector<EdgeDPState> fwd;
	                std::vector<EdgeDPState> rev;
	                std::vector<EdgeDPState> outside;
	                std::vector<uint8_t> outsideParentType;
	            };

			            inline uint32_t atomLocalOfBlock(node b,
			                                             std::vector<uint32_t> &blockIds)
	            {
	                for (uint32_t i = 0; i < blockIds.size(); ++i)
		                {
		                    if (blockIds[i] == b.idx)
		                        return i;
		                }
		                blockIds.push_back(b.idx);
			                return static_cast<uint32_t>(blockIds.size() - 1);
			            }

		            inline node atomCoreToBlock(uint32_t core, const BlockData &blk)
		            {
		                if (!blk.coreNodeInv || core >= blk.coreNodeInvLen)
		                    return node{SPQR_INVALID};
		                return node{blk.coreNodeInv[core]};
		            }

		            template <typename GraphLike>
		            inline uint32_t addAtomStateToGraph(const EdgeDPState &sub,
		                                                GraphLike &g,
		                                                std::vector<int> &inDeg,
		                                                std::vector<int> &outDeg,
		                                                std::vector<uint32_t> &blockIds,
		                                                bool zeroAsForward)
		            {
		                const uint32_t sLoc = atomLocalOfBlock(sub.s, blockIds);
		                const uint32_t tLoc = atomLocalOfBlock(sub.t, blockIds);
		                if (sLoc >= inDeg.size() || tLoc >= inDeg.size())
		                    return SPQR_INVALID;
		                int dir = sub.getDirection();
		                uint32_t forwardEdge = SPQR_INVALID;
		                if (dir == 1 || (dir == 0 && zeroAsForward))
		                {
		                    edge ge = g.newEdge(node{sLoc}, node{tLoc});
		                    forwardEdge = ge.idx;
		                }
		                if (dir == -1)
		                    g.newEdge(node{tLoc}, node{sLoc});
		                addStateCountsAtLoc(sub, sLoc, tLoc, inDeg, outDeg);
		                return forwardEdge;
		            }

		            template <typename GraphLike>
		            inline uint32_t addAtomChildRefToGraph(uint32_t cref,
		                                                    const std::vector<EdgeDPState> &macroStates,
	                                                    BlockData &blk,
	                                                    GraphLike &g,
	                                                    std::vector<int> &inDeg,
	                                                    std::vector<int> &outDeg,
	                                                    std::vector<uint32_t> &blockIds,
	                                                    bool zeroAsForward)
	            {
	                if (SP_COMPRESS_CHILD_IS_EDGE(cref))
	                {
	                    edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
	                    node u = blockEdgeSource(blk, e);
	                    node v = blockEdgeTarget(blk, e);
	                    const uint32_t uLoc = atomLocalOfBlock(u, blockIds);
	                    const uint32_t vLoc = atomLocalOfBlock(v, blockIds);
	                    if (uLoc >= inDeg.size() || vLoc >= inDeg.size())
	                        return SPQR_INVALID;
	                    edge ge = g.newEdge(node{uLoc}, node{vLoc});
	                    ++outDeg[uLoc];
	                    ++inDeg[vLoc];
	                    return ge.idx;
	                }
		                const uint32_t mid = SP_COMPRESS_CHILD_AS_MACRO(cref);
		                if (mid >= macroStates.size())
		                    return SPQR_INVALID;
		                return addAtomStateToGraph(macroStates[mid],
		                                           g,
		                                           inDeg,
		                                           outDeg,
		                                           blockIds,
		                                           zeroAsForward);
		            }

		            inline const EdgeDPState *nestedBehaviorAtomStateForItem(
		                const FfiSpqraBehaviorAtomItem &item,
		                const SpqraBehaviorAtomView &atoms,
		                const std::vector<EdgeDPState> &atomFwd,
		                const std::vector<EdgeDPState> &atomRev,
		                const BlockData &blk,
		                uint32_t currentAtom)
		            {
		                const uint32_t atomId = item.child_ref;
		                if (atomId >= currentAtom ||
		                    atomId >= atoms.atoms_len ||
		                    atomId >= atomFwd.size() ||
		                    atomId >= atomRev.size() ||
		                    !atoms.atoms_ptr)
		                {
		                    return nullptr;
		                }
		                const EdgeDPState &fwd = atomFwd[atomId];
		                const EdgeDPState &rev = atomRev[atomId];
		                if (blk.coreNodeInv &&
		                    item.src_core < blk.coreNodeInvLen &&
		                    item.dst_core < blk.coreNodeInvLen)
		                {
		                    node src{blk.coreNodeInv[item.src_core]};
		                    node dst{blk.coreNodeInv[item.dst_core]};
		                    if (src.idx != SPQR_INVALID && dst.idx != SPQR_INVALID)
		                    {
		                        if (fwd.s == src && fwd.t == dst)
		                            return &fwd;
		                        if (rev.s == src && rev.t == dst)
		                            return &rev;
		                    }
		                }
		                const FfiSpqraBehaviorAtom &nested = atoms.atoms_ptr[atomId];
		                if (item.src_core == nested.port0_core &&
		                    item.dst_core == nested.port1_core)
		                    return &fwd;
		                if (item.src_core == nested.port1_core &&
		                    item.dst_core == nested.port0_core)
		                    return &rev;
		                return nullptr;
		            }

			            inline EdgeDPState computeBehaviorAtomState(const FfiSpqraBehaviorAtom &atom,
			                                                        const SpqraBehaviorAtomView &atoms,
			                                                        const std::vector<EdgeDPState> &macroStates,
			                                                        const std::vector<EdgeDPState> &atomFwd,
			                                                        const std::vector<EdgeDPState> &atomRev,
			                                                        BlockData &blk,
			                                                        uint32_t atomIndex,
			                                                        bool reverse)
	            {
	                EdgeDPState st;
	                if (!atoms.items_ptr || !blk.coreNodeInv ||
	                    atom.port0_core >= blk.coreNodeInvLen ||
	                    atom.port1_core >= blk.coreNodeInvLen)
	                {
	                    return st;
	                }
	                node a{blk.coreNodeInv[atom.port0_core]};
	                node b{blk.coreNodeInv[atom.port1_core]};
	                if (a.idx == SPQR_INVALID || b.idx == SPQR_INVALID)
	                    return st;
	                resetMacroState(st, reverse ? b : a, reverse ? a : b);

		                const uint32_t itemBegin = std::min(atom.item_begin, atoms.items_len);
		                const uint32_t itemEnd = std::min(atom.item_end, atoms.items_len);
		                const uint32_t atomItemCount =
		                    itemEnd >= itemBegin ? itemEnd - itemBegin : 0;
		                std::vector<uint32_t> blockIds;
		                blockIds.reserve(atomItemCount + 2);
		                atomLocalOfBlock(a, blockIds);
		                atomLocalOfBlock(b, blockIds);
		                for (uint32_t ii = itemBegin; ii < itemEnd; ++ii)
	                {
	                    const FfiSpqraBehaviorAtomItem &item = atoms.items_ptr[ii];
		                    if (item.flags & SPQRA_MIN_ATOM_ITEM_CHILD_REF)
		                    {
		                        if (SP_COMPRESS_CHILD_IS_EDGE(item.child_ref))
		                        {
		                            edge e{SP_COMPRESS_CHILD_AS_EDGE(item.child_ref)};
		                            atomLocalOfBlock(blockEdgeSource(blk, e), blockIds);
		                            atomLocalOfBlock(blockEdgeTarget(blk, e), blockIds);
		                        }
		                        else
		                        {
		                            const uint32_t mid = SP_COMPRESS_CHILD_AS_MACRO(item.child_ref);
		                            if (mid < macroStates.size())
		                            {
		                                atomLocalOfBlock(macroStates[mid].s, blockIds);
		                                atomLocalOfBlock(macroStates[mid].t, blockIds);
		                            }
		                        }
		                    }
		                    else if (item.flags & SPQRA_MIN_ATOM_ITEM_BEHAVIOR_ATOM)
		                    {
		                        const EdgeDPState *nested =
		                            nestedBehaviorAtomStateForItem(item,
		                                                           atoms,
		                                                           atomFwd,
		                                                           atomRev,
		                                                           blk,
		                                                           atomIndex);
		                        if (nested)
		                        {
		                            atomLocalOfBlock(nested->s, blockIds);
		                            atomLocalOfBlock(nested->t, blockIds);
		                        }
		                    }
	                }

	                thread_local std::vector<int> tls_atom_inDeg;
	                thread_local std::vector<int> tls_atom_outDeg;
	                const uint32_t nLocal = static_cast<uint32_t>(blockIds.size());
	                if (tls_atom_inDeg.size() < nLocal) tls_atom_inDeg.resize(nLocal);
	                if (tls_atom_outDeg.size() < nLocal) tls_atom_outDeg.resize(nLocal);
	                std::fill_n(tls_atom_inDeg.begin(), nLocal, 0);
	                std::fill_n(tls_atom_outDeg.begin(), nLocal, 0);
		                thread_local FlatDigraphScratch tls_atom_graph;
		                tls_atom_graph.reset(nLocal, atomItemCount);

		                for (uint32_t ii = itemBegin; ii < itemEnd; ++ii)
	                {
	                    const FfiSpqraBehaviorAtomItem &item = atoms.items_ptr[ii];
		                    if (item.flags & SPQRA_MIN_ATOM_ITEM_CHILD_REF)
		                    {
		                        addChildRefToState(st, item.child_ref, macroStates, blk);
		                        addAtomChildRefToGraph(item.child_ref,
		                                               macroStates,
		                                               blk,
		                                               tls_atom_graph,
		                                               tls_atom_inDeg,
		                                               tls_atom_outDeg,
		                                               blockIds,
		                                               true);
		                    }
		                    else if (item.flags & SPQRA_MIN_ATOM_ITEM_BEHAVIOR_ATOM)
		                    {
		                        const EdgeDPState *nested =
		                            nestedBehaviorAtomStateForItem(item,
		                                                           atoms,
		                                                           atomFwd,
		                                                           atomRev,
		                                                           blk,
		                                                           atomIndex);
		                        if (nested)
		                        {
		                            mergeSubState(st, *nested);
		                            addAtomStateToGraph(*nested,
		                                                tls_atom_graph,
		                                                tls_atom_inDeg,
		                                                tls_atom_outDeg,
		                                                blockIds,
		                                                true);
		                        }
		                    }
		                }
	                if (st.acyclic)
	                    st.acyclic &= tls_atom_graph.isAcyclic();

	                uint32_t sLoc = SPQR_INVALID;
	                uint32_t tLoc = SPQR_INVALID;
	                for (uint32_t li = 0; li < nLocal; ++li)
	                {
	                    if (blockIds[li] == st.s.idx) sLoc = li;
	                    if (blockIds[li] == st.t.idx) tLoc = li;
	                }
	                if (sLoc != SPQR_INVALID && tLoc != SPQR_INVALID)
	                {
	                    st.localOutS = tls_atom_outDeg[sLoc];
	                    st.localInS = tls_atom_inDeg[sLoc];
	                    st.localOutT = tls_atom_outDeg[tLoc];
	                    st.localInT = tls_atom_inDeg[tLoc];
	                }
	                for (uint32_t li = 0; li < nLocal; ++li)
	                {
	                    node bnode{blockIds[li]};
	                    if (bnode == st.s || bnode == st.t)
	                        continue;
	                    if (blk.globIn[bnode] != tls_atom_inDeg[li] ||
	                        blk.globOut[bnode] != tls_atom_outDeg[li])
	                        st.hasLeakage = true;
	                    if (blk.globIn[bnode] == 0 || blk.globOut[bnode] == 0)
	                        st.globalSourceSink = true;
		                }
		                return st;
		            }

	            inline EdgeDPState reverseBehaviorAtomState(const EdgeDPState &fwd)
	            {
		                EdgeDPState rev;
		                rev.s = fwd.t;
		                rev.t = fwd.s;
		                rev.localOutS = fwd.localOutT;
		                rev.localInS = fwd.localInT;
		                rev.localOutT = fwd.localOutS;
		                rev.localInT = fwd.localInS;
		                rev.globalSourceSink = fwd.globalSourceSink;
		                rev.directST = fwd.directTS;
		                rev.directTS = fwd.directST;
		                rev.hasLeakage = fwd.hasLeakage;
		                rev.acyclic = fwd.acyclic;
	                return rev;
	            }

	            inline const EdgeDPState *behaviorAtomStateForCoreEndpoints(
	                const SpqraBehaviorAtomStates &atomStates,
	                const SpqraBehaviorAtomView &atoms,
	                uint32_t atomId,
	                uint32_t srcCore,
	                uint32_t dstCore)
	            {
	                if (!atoms.atoms_ptr || atomId >= atoms.atoms_len ||
	                    atomId >= atomStates.fwd.size() ||
	                    atomId >= atomStates.rev.size())
	                    return nullptr;
	                const FfiSpqraBehaviorAtom &atom = atoms.atoms_ptr[atomId];
	                if (srcCore == atom.port0_core && dstCore == atom.port1_core)
	                    return &atomStates.fwd[atomId];
	                if (srcCore == atom.port1_core && dstCore == atom.port0_core)
	                    return &atomStates.rev[atomId];
	                return nullptr;
	            }

		            inline SpqraBehaviorAtomStates computeBehaviorAtomStates(
		                const SpqraBehaviorAtomView &atoms,
		                const std::vector<EdgeDPState> &macroStates,
		                BlockData &blk)
	            {
	                SpqraBehaviorAtomStates out;
		                out.fwd.resize(atoms.atoms_len);
		                out.rev.resize(atoms.atoms_len);
		                out.outside.resize(atoms.atoms_len);
		                out.outsideParentType.assign(atoms.atoms_len, 0);
		                if (!atoms.atoms_ptr)
		                    return out;
			                for (uint32_t ai = 0; ai < atoms.atoms_len; ++ai)
		                {
			                    out.fwd[ai] = computeBehaviorAtomState(atoms.atoms_ptr[ai],
			                                                           atoms,
			                                                           macroStates,
			                                                           out.fwd,
			                                                           out.rev,
			                                                           blk,
			                                                           ai,
			                                                           false);
		                    out.rev[ai] = reverseBehaviorAtomState(out.fwd[ai]);
		                }
			                return out;
			            }

		            inline void computeNestedBehaviorAtomOutsideStates(
		                const SpqraBehaviorAtomView &atoms,
		                const std::vector<EdgeDPState> &macroStates,
		                BlockData &blk,
		                SpqraBehaviorAtomStates &atomStates)
		            {
		                if (!std::getenv("BF_SPQRA_NESTED_ATOM_OUTSIDE") ||
		                    !atoms.atoms_ptr || !atoms.items_ptr)
		                    return;
		                const uint32_t nAtoms = atoms.atoms_len;
		                for (uint32_t parentPlusOne = nAtoms; parentPlusOne > 0; --parentPlusOne)
		                {
		                    const uint32_t parentId = parentPlusOne - 1;
		                    if (parentId >= atomStates.outside.size())
		                        continue;
		                    const EdgeDPState &parentOutside = atomStates.outside[parentId];
		                    if (!parentOutside.s || !parentOutside.t)
		                        continue;
		                    const FfiSpqraBehaviorAtom &parent = atoms.atoms_ptr[parentId];
		                    const uint32_t itemBegin = std::min(parent.item_begin, atoms.items_len);
		                    const uint32_t itemEnd = std::min(parent.item_end, atoms.items_len);
		                    if (itemBegin >= itemEnd)
		                        continue;
		                    const uint8_t parentType = scadKindToSpqrType(parent.kind);

		                    for (uint32_t exclude = itemBegin; exclude < itemEnd; ++exclude)
		                    {
		                        const FfiSpqraBehaviorAtomItem &excludedItem = atoms.items_ptr[exclude];
		                        if ((excludedItem.flags & SPQRA_MIN_ATOM_ITEM_BEHAVIOR_ATOM) == 0)
		                            continue;
		                        const uint32_t childId = excludedItem.child_ref;
		                        if (childId >= parentId ||
		                            childId >= nAtoms ||
		                            childId >= atomStates.fwd.size() ||
		                            childId >= atomStates.rev.size() ||
		                            childId >= atomStates.outside.size())
		                            continue;
		                        const EdgeDPState *child =
		                            nestedBehaviorAtomStateForItem(excludedItem,
		                                                           atoms,
		                                                           atomStates.fwd,
		                                                           atomStates.rev,
		                                                           blk,
		                                                           parentId);
		                        if (!child || !child->s || !child->t)
		                            continue;

		                        EdgeDPState st;
		                        resetMacroState(st, child->s, child->t);
		                        std::vector<uint32_t> blockIds;
		                        blockIds.reserve((itemEnd - itemBegin) + 4);
		                        const FfiSpqraBehaviorAtom &childAtom = atoms.atoms_ptr[childId];
		                        node childA = atomCoreToBlock(childAtom.port0_core, blk);
		                        node childB = atomCoreToBlock(childAtom.port1_core, blk);
		                        if (childA.idx != SPQR_INVALID)
		                            atomLocalOfBlock(childA, blockIds);
		                        if (childB.idx != SPQR_INVALID)
		                            atomLocalOfBlock(childB, blockIds);
		                        atomLocalOfBlock(parentOutside.s, blockIds);
		                        atomLocalOfBlock(parentOutside.t, blockIds);
		                        for (uint32_t ii = itemBegin; ii < itemEnd; ++ii)
		                        {
		                            if (ii == exclude)
		                                continue;
		                            const FfiSpqraBehaviorAtomItem &item = atoms.items_ptr[ii];
		                            if (item.flags & SPQRA_MIN_ATOM_ITEM_CHILD_REF)
		                            {
		                                if (SP_COMPRESS_CHILD_IS_EDGE(item.child_ref))
		                                {
		                                    edge e{SP_COMPRESS_CHILD_AS_EDGE(item.child_ref)};
		                                    atomLocalOfBlock(blockEdgeSource(blk, e), blockIds);
		                                    atomLocalOfBlock(blockEdgeTarget(blk, e), blockIds);
		                                }
		                                else
		                                {
		                                    const uint32_t mid = SP_COMPRESS_CHILD_AS_MACRO(item.child_ref);
		                                    if (mid < macroStates.size())
		                                    {
		                                        atomLocalOfBlock(macroStates[mid].s, blockIds);
		                                        atomLocalOfBlock(macroStates[mid].t, blockIds);
		                                    }
		                                }
		                            }
		                            else if (item.flags & SPQRA_MIN_ATOM_ITEM_BEHAVIOR_ATOM)
		                            {
		                                const EdgeDPState *nested =
		                                    nestedBehaviorAtomStateForItem(item,
		                                                                   atoms,
		                                                                   atomStates.fwd,
		                                                                   atomStates.rev,
		                                                                   blk,
		                                                                   parentId);
		                                if (nested)
		                                {
		                                    atomLocalOfBlock(nested->s, blockIds);
		                                    atomLocalOfBlock(nested->t, blockIds);
		                                }
		                            }
		                        }

		                        thread_local std::vector<int> tls_atom_out_inDeg;
		                        thread_local std::vector<int> tls_atom_out_outDeg;
		                        thread_local FlatDigraphScratch tls_atom_out_graph;
		                        const uint32_t nLocal = static_cast<uint32_t>(blockIds.size());
		                        if (tls_atom_out_inDeg.size() < nLocal)
		                            tls_atom_out_inDeg.resize(nLocal);
		                        if (tls_atom_out_outDeg.size() < nLocal)
		                            tls_atom_out_outDeg.resize(nLocal);
		                        std::fill_n(tls_atom_out_inDeg.begin(), nLocal, 0);
		                        std::fill_n(tls_atom_out_outDeg.begin(), nLocal, 0);
		                        tls_atom_out_graph.reset(nLocal, (itemEnd - itemBegin) + 1);

		                        mergeSubState(st, parentOutside);
		                        addAtomStateToGraph(parentOutside,
		                                            tls_atom_out_graph,
		                                            tls_atom_out_inDeg,
		                                            tls_atom_out_outDeg,
		                                            blockIds,
		                                            true);
		                        for (uint32_t ii = itemBegin; ii < itemEnd; ++ii)
		                        {
		                            if (ii == exclude)
		                                continue;
		                            const FfiSpqraBehaviorAtomItem &item = atoms.items_ptr[ii];
		                            if (item.flags & SPQRA_MIN_ATOM_ITEM_CHILD_REF)
		                            {
		                                addChildRefToState(st, item.child_ref, macroStates, blk);
		                                addAtomChildRefToGraph(item.child_ref,
		                                                       macroStates,
		                                                       blk,
		                                                       tls_atom_out_graph,
		                                                       tls_atom_out_inDeg,
		                                                       tls_atom_out_outDeg,
		                                                       blockIds,
		                                                       true);
		                            }
		                            else if (item.flags & SPQRA_MIN_ATOM_ITEM_BEHAVIOR_ATOM)
		                            {
		                                const EdgeDPState *nested =
		                                    nestedBehaviorAtomStateForItem(item,
		                                                                   atoms,
		                                                                   atomStates.fwd,
		                                                                   atomStates.rev,
		                                                                   blk,
		                                                                   parentId);
		                                if (nested)
		                                {
		                                    mergeSubState(st, *nested);
		                                    addAtomStateToGraph(*nested,
		                                                        tls_atom_out_graph,
		                                                        tls_atom_out_inDeg,
		                                                        tls_atom_out_outDeg,
		                                                        blockIds,
		                                                        true);
		                                }
		                            }
		                        }
		                        if (st.acyclic)
		                            st.acyclic &= tls_atom_out_graph.isAcyclic();

		                        uint32_t sLoc = SPQR_INVALID;
		                        uint32_t tLoc = SPQR_INVALID;
		                        for (uint32_t li = 0; li < nLocal; ++li)
		                        {
		                            if (blockIds[li] == st.s.idx) sLoc = li;
		                            if (blockIds[li] == st.t.idx) tLoc = li;
		                        }
		                        if (sLoc != SPQR_INVALID && tLoc != SPQR_INVALID)
		                        {
		                            st.localOutS = tls_atom_out_outDeg[sLoc];
		                            st.localInS = tls_atom_out_inDeg[sLoc];
		                            st.localOutT = tls_atom_out_outDeg[tLoc];
		                            st.localInT = tls_atom_out_inDeg[tLoc];
		                        }
		                        for (uint32_t li = 0; li < nLocal; ++li)
		                        {
		                            node bnode{blockIds[li]};
		                            if (bnode == st.s || bnode == st.t)
		                                continue;
		                            if (blk.globIn[bnode] != tls_atom_out_inDeg[li] ||
		                                blk.globOut[bnode] != tls_atom_out_outDeg[li])
		                                st.hasLeakage = true;
		                            if (blk.globIn[bnode] == 0 || blk.globOut[bnode] == 0)
		                                st.globalSourceSink = true;
		                        }
		                        atomStates.outside[childId] = st;
		                        if (childId < atomStates.outsideParentType.size())
		                            atomStates.outsideParentType[childId] = parentType;
		                    }
		                }
		            }

	            inline bool emitBehaviorAtomPFixedAgg(uint32_t atomId,
	                                                   const SpqraBehaviorAtomView &atoms,
	                                                   const SpqraBehaviorAtomStates &atomStates,
	                                                   const std::vector<EdgeDPState> &macroStates,
	                                                   const SpCompressTreeView &view,
		                                                   const CcData &cc,
		                                                   BlockData &blk)
		            {
		                if (!atoms.atoms_ptr || !atoms.items_ptr ||
		                    atomId >= atoms.atoms_len ||
		                    atomId >= atomStates.fwd.size() ||
		                    atomId >= atomStates.outside.size())
		                    return false;
		                const FfiSpqraBehaviorAtom &atom = atoms.atoms_ptr[atomId];
		                if (scadKindToSpqrType(atom.kind) != SPQR_NODE_TYPE_P)
		                    return false;
		                const EdgeDPState &inside = atomStates.fwd[atomId];
		                const EdgeDPState &outside = atomStates.outside[atomId];
		                if (!inside.s || !inside.t || !outside.s || !outside.t)
		                    return false;

		                PFixedAggEntry entry;
		                node bS{inside.s.idx};
		                node bT{inside.t.idx};
		                const uint32_t itemBegin = std::min(atom.item_begin, atoms.items_len);
		                const uint32_t itemEnd = std::min(atom.item_end, atoms.items_len);
		                for (uint32_t ii = itemBegin; ii < itemEnd; ++ii)
		                {
		                    const FfiSpqraBehaviorAtomItem &item = atoms.items_ptr[ii];
		                    if (item.flags & SPQRA_MIN_ATOM_ITEM_CHILD_REF)
		                    {
		                        const uint32_t cref = item.child_ref;
		                        if (SP_COMPRESS_CHILD_IS_EDGE(cref))
		                        {
		                            edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
		                            node u = blockEdgeSource(blk, e);
		                            node v = blockEdgeTarget(blk, e);
		                            if (u == bS && v == bT)
		                                ++entry.directST;
		                            else
		                                ++entry.directTS;
		                        }
		                        else
		                        {
		                            const uint32_t mid = SP_COMPRESS_CHILD_AS_MACRO(cref);
		                            if (mid >= macroStates.size() || mid >= view.macros_len)
		                                continue;
		                            const bool seriesLike =
		                                view.macros_ptr[mid].kind == SP_COMPRESS_KIND_SERIES;
		                            entry.fwd.addBranch(macroStates[mid],
		                                                bS,
		                                                bT,
		                                                seriesLike,
		                                                PFixedAggMode::Core);
		                            entry.rev.addBranch(macroStates[mid],
		                                                bT,
		                                                bS,
		                                                seriesLike,
		                                                PFixedAggMode::Core);
		                        }
		                    }
		                    else if (item.flags & SPQRA_MIN_ATOM_ITEM_BEHAVIOR_ATOM)
		                    {
		                        const EdgeDPState *nested =
		                            nestedBehaviorAtomStateForItem(item,
		                                                           atoms,
		                                                           atomStates.fwd,
		                                                           atomStates.rev,
		                                                           blk,
		                                                           atomId);
		                        if (!nested)
		                            continue;
		                        const bool seriesLike =
		                            item.child_ref < atoms.atoms_len &&
		                            scadKindToSpqrType(atoms.atoms_ptr[item.child_ref].kind) ==
		                                SPQR_NODE_TYPE_S;
		                        entry.fwd.addBranch(*nested,
		                                            bS,
		                                            bT,
		                                            seriesLike,
		                                            PFixedAggMode::Core);
		                        entry.rev.addBranch(*nested,
		                                            bT,
		                                            bS,
		                                            seriesLike,
		                                            PFixedAggMode::Core);
		                    }
		                }
		                const uint8_t outsideParentType =
		                    atomId < atomStates.outsideParentType.size()
		                        ? atomStates.outsideParentType[atomId]
		                        : 0;
		                const bool outsideSeriesLike = outsideParentType == SPQR_NODE_TYPE_S;
		                entry.fwd.addBranch(outside,
		                                    bS,
		                                    bT,
		                                    outsideSeriesLike,
		                                    PFixedAggMode::Core);
		                entry.rev.addBranch(outside,
		                                    bT,
		                                    bS,
		                                    outsideSeriesLike,
		                                    PFixedAggMode::Core);
		                emitPFixedAggCandidate(entry.fwd,
		                                       bS,
		                                       bT,
		                                       entry.directST,
		                                       entry.directTS,
		                                       true,
		                                       cc,
		                                       blk);
		                emitPFixedAggCandidate(entry.rev,
		                                       bT,
		                                       bS,
		                                       entry.directTS,
		                                       entry.directST,
		                                       true,
		                                       cc,
		                                       blk);
		                return true;
		            }

	            inline const EdgeDPState *behaviorAtomStateForEdge(
	                const SpqraBehaviorAtomStates &atomStates,
	                const FfiSpqraMinimizerEdge &me,
	                const SpqraMinimizerView &minView,
	                const BlockData &blk,
	                const FfiSpqraMinimizerComponent &comp)
		            {
		                const uint32_t atomId = me.child_ref;
		                if (atomId >= atomStates.fwd.size() || atomId >= atomStates.rev.size())
		                    return nullptr;
		                const EdgeDPState &fwd = atomStates.fwd[atomId];
		                const EdgeDPState &rev = atomStates.rev[atomId];
		                if (blk.spqraBehaviorAtomView.atoms_ptr &&
		                    atomId < blk.spqraBehaviorAtomView.atoms_len &&
		                    minView.node_mapping_ptr)
		                {
		                    if (me.src_local != SPQR_INVALID &&
		                        me.dst_local != SPQR_INVALID)
		                    {
		                        const uint32_t srcMi = comp.node_begin + me.src_local;
		                        const uint32_t dstMi = comp.node_begin + me.dst_local;
		                        if (srcMi < minView.node_mapping_len &&
		                            dstMi < minView.node_mapping_len)
		                        {
		                            const FfiSpqraBehaviorAtom &atom =
		                                blk.spqraBehaviorAtomView.atoms_ptr[atomId];
		                            const uint32_t srcCore = minView.node_mapping_ptr[srcMi];
		                            const uint32_t dstCore = minView.node_mapping_ptr[dstMi];
		                            if (srcCore == atom.port0_core && dstCore == atom.port1_core)
		                                return &fwd;
		                            if (srcCore == atom.port1_core && dstCore == atom.port0_core)
		                                return &rev;
		                        }
		                    }
		                }
		                const uint32_t srcBlock =
		                    minimizerLocalBlockId(minView, blk, comp, me.src_local);
		                const uint32_t dstBlock =
		                    minimizerLocalBlockId(minView, blk, comp, me.dst_local);
		                if (fwd.s.idx == srcBlock && fwd.t.idx == dstBlock)
		                    return &fwd;
		                if (rev.s.idx == srcBlock && rev.t.idx == dstBlock)
		                    return &rev;
	                if (fwd.s.idx == dstBlock && fwd.t.idx == srcBlock)
	                    return &fwd;
	                if (rev.s.idx == dstBlock && rev.t.idx == srcBlock)
	                    return &rev;
	                return &fwd;
	            }

            inline void computeCoreUpStates(const CoreContext &c,
                                            const SpCompressTreeView &view,
                                            const std::vector<EdgeDPState> &macroStates,
                                            BlockData &blk,
	                                            std::vector<EdgeDPState> &down,
	                                            std::vector<EdgeDPState> &up,
	                                            std::vector<NodeDPState> &nodeDp,
	                                            const CcData *directCoreCc = nullptr,
	                                            std::vector<PFixedAggEntry> *corePAgg = nullptr)
            {
                const bool emitCoreEdgesDuringUp =
                    directCoreCc != nullptr &&
                    std::getenv("BF_SPQRA_CORE_EXACT_EMIT_DURING_UP") != nullptr;
                for (auto itPost = c.post.rbegin(); itPost != c.post.rend(); ++itPost)
                {
                    uint32_t A = *itPost;
                    uint32_t nLocal = c.mapOffsets[A + 1] - c.mapOffsets[A];
                    uint32_t start = c.skelOffsets[A];
                    uint32_t end = c.skelOffsets[A + 1];
                    thread_local FlatDigraphScratch tls_core_up_graph;
                    thread_local std::vector<int> tls_core_up_inDeg;
                    thread_local std::vector<int> tls_core_up_outDeg;
                    thread_local std::vector<uint8_t> tls_core_up_isSourceSink;
                    thread_local std::vector<uint8_t> tls_core_up_isLeaking;
                    if (tls_core_up_inDeg.size() < nLocal) tls_core_up_inDeg.resize(nLocal);
                    if (tls_core_up_outDeg.size() < nLocal) tls_core_up_outDeg.resize(nLocal);
                    if (tls_core_up_isSourceSink.size() < nLocal) tls_core_up_isSourceSink.resize(nLocal);
                    if (tls_core_up_isLeaking.size() < nLocal) tls_core_up_isLeaking.resize(nLocal);
                    std::fill_n(tls_core_up_inDeg.begin(), nLocal, 0);
                    std::fill_n(tls_core_up_outDeg.begin(), nLocal, 0);
                    std::fill_n(tls_core_up_isSourceSink.begin(), nLocal, (uint8_t)0);
                    std::fill_n(tls_core_up_isLeaking.begin(), nLocal, (uint8_t)0);
                    tls_core_up_graph.reset(nLocal, end - start);

                    struct VRef
                    {
                        uint32_t other;
                        EdgeDPState *toUpdate;
                        const EdgeDPState *opposite;
                        uint32_t sLoc;
                        uint32_t tLoc;
                        uint32_t forwardEdge;
                    };
	                    thread_local std::vector<VRef> refs;
	                    refs.clear();
	                    for (uint32_t i = start; i < end; ++i)
	                    {
	                        const SkeletonEdge &se = c.skelEdges[i];
	                        uint32_t cref = SPQR_INVALID;
	                        if (coreSkeletonChildRef(view, se, cref))
	                        {
	                            addChildRefToGraphFast(cref, macroStates, c, A, se, blk,
	                                                   tls_core_up_graph,
	                                                   tls_core_up_inDeg,
	                                                   tls_core_up_outDeg,
	                                                   true);
	                            continue;
	                        }
                        uint32_t B = SPQR_INVALID;
                        if (!coreSkeletonTwin(c, se, B))
                            continue;
                        const EdgeDPState *child = (B == c.parents[A] ? &up[A] : &down[B]);
                        EdgeDPState *update = (B == c.parents[A] ? &down[A] : &up[B]);
                        if (B != c.parents[A])
                        {
                            update->s = down[B].s;
                            update->t = down[B].t;
                        }
                        uint32_t sLoc = localEndpointOnSkeletonEdge(c, A, se, child->s);
                        uint32_t tLoc = localEndpointOnSkeletonEdge(c, A, se, child->t);
                        uint32_t forwardEdge = addStateToGraphOnSkeletonEdge(*child, c, A, se,
                                                                              tls_core_up_graph,
                                                                              tls_core_up_inDeg,
                                                                              tls_core_up_outDeg,
                                                                              true);
                        if (sLoc != SPQR_INVALID && tLoc != SPQR_INVALID)
                            refs.push_back({B, update, child, sLoc, tLoc, forwardEdge});
                    }
                    if (refs.empty())
                        continue;

                    int localSourceSinkCount = 0;
                    int localLeakageCount = 0;
                    for (uint32_t li = 0; li < nLocal; ++li)
                    {
                        node b{coreLocalBlockId(c, A, li)};
                        if (blk.globIn[b] == 0 || blk.globOut[b] == 0)
                        {
                            tls_core_up_isSourceSink[li] = 1;
                            localSourceSinkCount++;
                        }
                        if (blk.globIn[b] != tls_core_up_inDeg[li] ||
                            blk.globOut[b] != tls_core_up_outDeg[li])
                        {
                            tls_core_up_isLeaking[li] = 1;
                            localLeakageCount++;
                        }
                    }

                    if (c.types[A] == SPQR_NODE_TYPE_P)
                    {
                        node pole0{coreLocalBlockId(c, A, 0)};
                        node pole1{coreLocalBlockId(c, A, 1)};
                        int cnt01 = 0;
                        int cnt10 = 0;
                        for (uint32_t i = start; i < end; ++i)
                        {
                            const SkeletonEdge &se = c.skelEdges[i];
                            uint32_t cref = SPQR_INVALID;
                            if (!coreSkeletonChildRef(view, se, cref))
                                continue;
                            if (!SP_COMPRESS_CHILD_IS_EDGE(cref))
                                continue;
                            edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
                            node u = blockEdgeSource(blk, e);
                            node v = blockEdgeTarget(blk, e);
                            if (u == pole0 && v == pole1) cnt01++;
                            if (u == pole1 && v == pole0) cnt10++;
                        }
                        for (auto &r : refs)
                        {
                            EdgeDPState &st = *r.toUpdate;
                            if (st.s == pole0 && st.t == pole1)
                            {
                                st.directST |= cnt01 > 0;
                                st.directTS |= cnt10 > 0;
                            }
                            else if (st.s == pole1 && st.t == pole0)
                            {
                                st.directST |= cnt10 > 0;
                                st.directTS |= cnt01 > 0;
                            }
                        }
                    }

                    NodeDPState curr = nodeDp[A];
                    if (curr.outgoingCyclesCount >= 2)
                    {
                        for (auto &r : refs)
                        {
                            if (r.toUpdate->acyclic)
                            {
                                nodeDp[r.other].outgoingCyclesCount++;
                                nodeDp[r.other].lastCycleNode = node{A};
                            }
                            r.toUpdate->acyclic = false;
                        }
                    }
                    else if (curr.outgoingCyclesCount == 1)
                    {
                        for (auto &r : refs)
                        {
                            if (node{r.other} != curr.lastCycleNode)
                            {
                                if (r.toUpdate->acyclic)
                                {
                                    nodeDp[r.other].outgoingCyclesCount++;
                                    nodeDp[r.other].lastCycleNode = node{A};
                                }
                                r.toUpdate->acyclic = false;
                            }
                            else
                            {
                                bool ac = tls_core_up_graph.isAcyclic(r.forwardEdge);
                                if (r.toUpdate->acyclic && !ac)
                                {
                                    nodeDp[r.other].outgoingCyclesCount++;
                                    nodeDp[r.other].lastCycleNode = node{A};
                                }
                                r.toUpdate->acyclic &= ac;
                            }
                        }
                    }
                    else
                    {
                        thread_local std::vector<edge> fasEdges;
                        fasEdges.clear();
                        bool ac = tls_core_up_graph.isAcyclic();
                        if (!ac)
                        {
                            Graph g;
                            tls_core_up_graph.materialize(g);
                            FeedbackArcSet fas(g);
                            ac = fas.run_or_acyclic(fasEdges);
                        }
                        if (!ac)
                        {
                            thread_local std::vector<uint8_t> inFas;
                            const uint32_t nEdges = tls_core_up_graph.numberOfEdges();
                            if (inFas.size() < nEdges) inFas.resize(nEdges);
                            std::fill_n(inFas.begin(), nEdges, (uint8_t)0);
                            for (edge e : fasEdges) inFas[e.idx] = 1;
                            for (auto &r : refs)
                            {
                                bool keep = r.forwardEdge != SPQR_INVALID && inFas[r.forwardEdge];
                                if (r.toUpdate->acyclic && !keep)
                                {
                                    nodeDp[r.other].outgoingCyclesCount++;
                                    nodeDp[r.other].lastCycleNode = node{A};
                                }
                                r.toUpdate->acyclic &= keep;
                            }
                        }
                    }

                    for (auto &r : refs)
                    {
                        if (curr.outgoingSourceSinkCount >= 2 ||
                            (curr.outgoingSourceSinkCount == 1 && node{r.other} != curr.lastSourceSinkNode) ||
                            ((int)tls_core_up_isSourceSink[r.sLoc] +
                             (int)tls_core_up_isSourceSink[r.tLoc] < localSourceSinkCount))
                        {
                            if (!r.toUpdate->globalSourceSink)
                            {
                                nodeDp[r.other].outgoingSourceSinkCount++;
                                nodeDp[r.other].lastSourceSinkNode = node{A};
                            }
                            r.toUpdate->globalSourceSink = true;
                        }
                        if (curr.outgoingLeakageCount >= 2 ||
                            (curr.outgoingLeakageCount == 1 && node{r.other} != curr.lastLeakageNode) ||
                            ((int)tls_core_up_isLeaking[r.sLoc] +
                             (int)tls_core_up_isLeaking[r.tLoc] < localLeakageCount))
                        {
                            if (!r.toUpdate->hasLeakage)
                            {
                                nodeDp[r.other].outgoingLeakageCount++;
                                nodeDp[r.other].lastLeakageNode = node{A};
                            }
                            r.toUpdate->hasLeakage = true;
                        }
                        r.toUpdate->localInS = tls_core_up_inDeg[r.sLoc] - r.opposite->localInS;
                        r.toUpdate->localOutS = tls_core_up_outDeg[r.sLoc] - r.opposite->localOutS;
                        r.toUpdate->localInT = tls_core_up_inDeg[r.tLoc] - r.opposite->localInT;
                        r.toUpdate->localOutT = tls_core_up_outDeg[r.tLoc] - r.opposite->localOutT;
                    }

                    if (emitCoreEdgesDuringUp)
                    {
                        const uint32_t parentOfA = c.parents[A];
                        for (const auto &r : refs)
                        {
                            uint32_t B = r.other;
                            if (B == parentOfA || B >= c.len || c.parents[B] != A)
                                continue;
                            tryCoreTreeEdgeBubbles(B, A, c, down[B], up[B], *directCoreCc,
                                                   blk);
                            blk.isAcycic &= (down[B].acyclic && up[B].acyclic);
	                        }
	                    }
	                }
	                if (corePAgg)
	                {
	                    corePAgg->assign(c.len, PFixedAggEntry{});
	                    for (uint32_t tn = 0; tn < c.len; ++tn)
	                    {
	                        if (c.types[tn] == SPQR_NODE_TYPE_P)
	                            buildCorePFixedAggEntry(tn, c, view, macroStates, down, up, blk,
	                                                     (*corePAgg)[tn]);
	                    }
	                }
	            }

            inline std::vector<EdgeDPState> computeMinimizerDownStates(const SpqraMinimizerView &minView,
                                                                       const std::vector<EdgeDPState> &macroStates,
                                                                       BlockData &blk,
	                                                                       std::vector<NodeDPState> &nodeDp,
	                                                                       const SpqraBehaviorAtomStates *atomStates = nullptr)
	            {
	                std::vector<EdgeDPState> down(minView.components_len);
	                if (!minView.components_ptr || !minView.edges_ptr || !minView.postorder_ptr)
	                    return down;
	                const uint32_t root = minView.summary.root;
	                for (uint32_t pi = 0; pi < minView.postorder_len; ++pi)
	                {
	                    const uint32_t tn = minView.postorder_ptr[pi];
	                    if (tn >= minView.components_len || tn == root)
	                        continue;
	                    const FfiSpqraMinimizerComponent &comp = minView.components_ptr[tn];
	                    const uint32_t parent = comp.parent;
	                    if (parent >= minView.components_len)
	                        continue;

	                    uint32_t parentEdge = SPQR_INVALID;
	                    const uint32_t edgeEnd = std::min(comp.edge_end, minView.edges_len);
	                    for (uint32_t ge = comp.edge_begin; ge < edgeEnd; ++ge)
	                    {
	                        const FfiSpqraMinimizerEdge &me = minView.edges_ptr[ge];
	                        if ((me.flags & SPQRA_MIN_EDGE_VIRTUAL) &&
	                            me.twin_component == parent)
	                        {
	                            parentEdge = ge;
	                            break;
	                        }
	                    }
	                    if (parentEdge == SPQR_INVALID)
	                        continue;

	                    const FfiSpqraMinimizerEdge &pe = minView.edges_ptr[parentEdge];
	                    node s{minimizerLocalBlockId(minView, blk, comp, pe.dst_local)};
	                    node t{minimizerLocalBlockId(minView, blk, comp, pe.src_local)};
	                    if (s.idx == SPQR_INVALID || t.idx == SPQR_INVALID)
	                        continue;
	                    EdgeDPState &st = down[tn];
	                    resetMacroState(st, s, t);

	                    const uint32_t nLocal =
	                        comp.node_end >= comp.node_begin ? comp.node_end - comp.node_begin : 0;
	                    thread_local FlatDigraphScratch tls_min_down_graph;
	                    thread_local std::vector<int> tls_min_down_inDeg;
	                    thread_local std::vector<int> tls_min_down_outDeg;
	                    if (tls_min_down_inDeg.size() < nLocal) tls_min_down_inDeg.resize(nLocal);
	                    if (tls_min_down_outDeg.size() < nLocal) tls_min_down_outDeg.resize(nLocal);
	                    std::fill_n(tls_min_down_inDeg.begin(), nLocal, 0);
	                    std::fill_n(tls_min_down_outDeg.begin(), nLocal, 0);
	                    tls_min_down_graph.reset(nLocal, edgeEnd - comp.edge_begin);

	                    for (uint32_t ge = comp.edge_begin; ge < edgeEnd; ++ge)
	                    {
	                        if (ge == parentEdge)
	                            continue;
	                        const FfiSpqraMinimizerEdge &me = minView.edges_ptr[ge];
	                        if ((me.flags & SPQRA_MIN_EDGE_HAS_BEHAVIOR_ATOM) && atomStates)
	                        {
	                            const EdgeDPState *atom =
	                                behaviorAtomStateForEdge(*atomStates, me, minView, blk, comp);
	                            if (atom)
	                            {
	                                mergeSubState(st, *atom);
	                                addStateToGraphOnMinimizerEdge(*atom, minView, blk, comp, me,
	                                                               tls_min_down_graph,
	                                                               tls_min_down_inDeg,
	                                                               tls_min_down_outDeg,
	                                                               false);
	                            }
	                            continue;
	                        }
	                        if (me.flags & SPQRA_MIN_EDGE_HAS_CHILD_REF)
	                        {
	                            addChildRefToState(st, me.child_ref, macroStates, blk);
	                            addChildRefToGraphFastMinimizer(me.child_ref, macroStates,
	                                                            minView, blk, comp, me,
	                                                            tls_min_down_graph,
	                                                            tls_min_down_inDeg,
	                                                            tls_min_down_outDeg,
	                                                            true);
	                            if (scadKindToSpqrType(comp.kind) == SPQR_NODE_TYPE_P &&
	                                SP_COMPRESS_CHILD_IS_EDGE(me.child_ref))
	                            {
	                                edge e{SP_COMPRESS_CHILD_AS_EDGE(me.child_ref)};
	                                node u = blockEdgeSource(blk, e);
	                                node v = blockEdgeTarget(blk, e);
	                                if (u == st.s && v == st.t) st.directST = true;
	                                if (u == st.t && v == st.s) st.directTS = true;
	                            }
	                            continue;
	                        }
	                        if (me.flags & SPQRA_MIN_EDGE_VIRTUAL)
	                        {
	                            const uint32_t child = me.twin_component;
	                            if (child >= minView.components_len ||
	                                minView.components_ptr[child].parent != tn)
	                                continue;
	                            mergeSubState(st, down[child]);
	                            addStateToGraphOnMinimizerEdge(down[child], minView, blk, comp, me,
	                                                           tls_min_down_graph,
	                                                           tls_min_down_inDeg,
	                                                           tls_min_down_outDeg,
	                                                           false);
	                        }
	                    }

	                    const uint32_t sLoc = pe.dst_local;
	                    const uint32_t tLoc = pe.src_local;
	                    if (sLoc < nLocal && tLoc < nLocal)
	                    {
	                        st.localOutS = tls_min_down_outDeg[sLoc];
	                        st.localInS = tls_min_down_inDeg[sLoc];
	                        st.localOutT = tls_min_down_outDeg[tLoc];
	                        st.localInT = tls_min_down_inDeg[tLoc];
	                    }

	                    for (uint32_t li = 0; li < nLocal; ++li)
	                    {
	                        node b{minimizerLocalBlockId(minView, blk, comp, li)};
	                        if (b.idx == SPQR_INVALID || b == st.s || b == st.t)
	                            continue;
	                        if (blk.globIn[b] != tls_min_down_inDeg[li] ||
	                            blk.globOut[b] != tls_min_down_outDeg[li])
	                            st.hasLeakage = true;
	                        if (blk.globIn[b] == 0 || blk.globOut[b] == 0)
	                            st.globalSourceSink = true;
	                    }

	                    if (st.acyclic)
	                        st.acyclic &= tls_min_down_graph.isAcyclic();
	                    if (!st.acyclic && parent < nodeDp.size())
	                    {
	                        nodeDp[parent].outgoingCyclesCount++;
	                        nodeDp[parent].lastCycleNode = node{tn};
	                    }
	                    if (st.globalSourceSink && parent < nodeDp.size())
	                    {
	                        nodeDp[parent].outgoingSourceSinkCount++;
	                        nodeDp[parent].lastSourceSinkNode = node{tn};
	                    }
	                    if (st.hasLeakage && parent < nodeDp.size())
	                    {
	                        nodeDp[parent].outgoingLeakageCount++;
	                        nodeDp[parent].lastLeakageNode = node{tn};
	                    }
	                }
                return down;
            }

            inline void computeMinimizerUpStates(const SpqraMinimizerView &minView,
                                                 const std::vector<EdgeDPState> &macroStates,
                                                 BlockData &blk,
	                                                 std::vector<EdgeDPState> &down,
	                                                 std::vector<EdgeDPState> &up,
	                                                 std::vector<NodeDPState> &nodeDp,
	                                                 SpqraBehaviorAtomStates *atomStates = nullptr)
	            {
	                if (!minView.components_ptr || !minView.edges_ptr || !minView.postorder_ptr)
	                    return;
	                for (auto itPost = std::make_reverse_iterator(minView.postorder_ptr + minView.postorder_len);
	                     itPost != std::make_reverse_iterator(minView.postorder_ptr); ++itPost)
	                {
	                    const uint32_t A = *itPost;
	                    if (A >= minView.components_len)
	                        continue;
	                    const FfiSpqraMinimizerComponent &comp = minView.components_ptr[A];
	                    const uint32_t nLocal =
	                        comp.node_end >= comp.node_begin ? comp.node_end - comp.node_begin : 0;
	                    const uint32_t edgeEnd = std::min(comp.edge_end, minView.edges_len);
	                    thread_local FlatDigraphScratch tls_min_up_graph;
	                    thread_local std::vector<int> tls_min_up_inDeg;
	                    thread_local std::vector<int> tls_min_up_outDeg;
	                    thread_local std::vector<uint8_t> tls_min_up_isSourceSink;
	                    thread_local std::vector<uint8_t> tls_min_up_isLeaking;
	                    if (tls_min_up_inDeg.size() < nLocal) tls_min_up_inDeg.resize(nLocal);
	                    if (tls_min_up_outDeg.size() < nLocal) tls_min_up_outDeg.resize(nLocal);
	                    if (tls_min_up_isSourceSink.size() < nLocal) tls_min_up_isSourceSink.resize(nLocal);
	                    if (tls_min_up_isLeaking.size() < nLocal) tls_min_up_isLeaking.resize(nLocal);
	                    std::fill_n(tls_min_up_inDeg.begin(), nLocal, 0);
	                    std::fill_n(tls_min_up_outDeg.begin(), nLocal, 0);
	                    std::fill_n(tls_min_up_isSourceSink.begin(), nLocal, (uint8_t)0);
	                    std::fill_n(tls_min_up_isLeaking.begin(), nLocal, (uint8_t)0);
	                    tls_min_up_graph.reset(nLocal, edgeEnd - comp.edge_begin);

	                    struct VRef
	                    {
	                        uint32_t other;
	                        EdgeDPState *toUpdate;
	                        const EdgeDPState *opposite;
	                        uint32_t sLoc;
	                        uint32_t tLoc;
	                        uint32_t forwardEdge;
	                    };
	                    thread_local std::vector<VRef> refs;
	                    refs.clear();
	                    const uint32_t parentA = comp.parent;
	                    for (uint32_t ge = comp.edge_begin; ge < edgeEnd; ++ge)
	                    {
	                        const FfiSpqraMinimizerEdge &me = minView.edges_ptr[ge];
	                        if ((me.flags & SPQRA_MIN_EDGE_HAS_BEHAVIOR_ATOM) && atomStates)
	                        {
	                            const uint32_t atomId = me.child_ref;
	                            if (atomId >= atomStates->outside.size())
	                                continue;
	                            const EdgeDPState *child =
	                                behaviorAtomStateForEdge(*atomStates, me, minView, blk, comp);
	                            if (!child)
	                                continue;
		                            EdgeDPState *update = &atomStates->outside[atomId];
		                            update->s = child->s;
		                            update->t = child->t;
		                            if (atomId < atomStates->outsideParentType.size())
		                                atomStates->outsideParentType[atomId] =
		                                    scadKindToSpqrType(comp.kind);
		                            const uint32_t sLoc =
		                                minimizerLocalEndpointOnEdge(minView, blk, comp, me, child->s);
	                            const uint32_t tLoc =
	                                minimizerLocalEndpointOnEdge(minView, blk, comp, me, child->t);
	                            const uint32_t forwardEdge =
	                                addStateToGraphOnMinimizerEdge(*child, minView, blk, comp, me,
	                                                               tls_min_up_graph,
	                                                               tls_min_up_inDeg,
	                                                               tls_min_up_outDeg,
	                                                               true);
	                            if (sLoc != SPQR_INVALID && tLoc != SPQR_INVALID)
	                                refs.push_back({SPQR_INVALID, update, child, sLoc, tLoc, forwardEdge});
	                            continue;
	                        }
	                        if (me.flags & SPQRA_MIN_EDGE_HAS_CHILD_REF)
	                        {
	                            addChildRefToGraphFastMinimizer(me.child_ref, macroStates,
	                                                            minView, blk, comp, me,
	                                                            tls_min_up_graph,
	                                                            tls_min_up_inDeg,
	                                                            tls_min_up_outDeg,
	                                                            true);
	                            continue;
	                        }
	                        if ((me.flags & SPQRA_MIN_EDGE_VIRTUAL) == 0)
	                            continue;
	                        const uint32_t B = me.twin_component;
	                        if (B >= minView.components_len)
	                            continue;
	                        const EdgeDPState *child = (B == parentA ? &up[A] : &down[B]);
	                        EdgeDPState *update = (B == parentA ? &down[A] : &up[B]);
	                        if (B != parentA)
	                        {
	                            update->s = down[B].s;
	                            update->t = down[B].t;
	                        }
	                        const uint32_t sLoc =
	                            minimizerLocalEndpointOnEdge(minView, blk, comp, me, child->s);
	                        const uint32_t tLoc =
	                            minimizerLocalEndpointOnEdge(minView, blk, comp, me, child->t);
	                        const uint32_t forwardEdge =
	                            addStateToGraphOnMinimizerEdge(*child, minView, blk, comp, me,
	                                                           tls_min_up_graph,
	                                                           tls_min_up_inDeg,
	                                                           tls_min_up_outDeg,
	                                                           true);
	                        if (sLoc != SPQR_INVALID && tLoc != SPQR_INVALID)
	                            refs.push_back({B, update, child, sLoc, tLoc, forwardEdge});
	                    }
	                    if (refs.empty())
	                        continue;

	                    int localSourceSinkCount = 0;
	                    int localLeakageCount = 0;
	                    for (uint32_t li = 0; li < nLocal; ++li)
	                    {
	                        node b{minimizerLocalBlockId(minView, blk, comp, li)};
	                        if (b.idx == SPQR_INVALID)
	                            continue;
	                        if (blk.globIn[b] == 0 || blk.globOut[b] == 0)
	                        {
	                            tls_min_up_isSourceSink[li] = 1;
	                            ++localSourceSinkCount;
	                        }
	                        if (blk.globIn[b] != tls_min_up_inDeg[li] ||
	                            blk.globOut[b] != tls_min_up_outDeg[li])
	                        {
	                            tls_min_up_isLeaking[li] = 1;
	                            ++localLeakageCount;
	                        }
	                    }

	                    if (scadKindToSpqrType(comp.kind) == SPQR_NODE_TYPE_P)
	                    {
	                        node pole0{minimizerLocalBlockId(minView, blk, comp, 0)};
	                        node pole1{minimizerLocalBlockId(minView, blk, comp, 1)};
	                        int cnt01 = 0;
	                        int cnt10 = 0;
	                        for (uint32_t ge = comp.edge_begin; ge < edgeEnd; ++ge)
	                        {
	                            const FfiSpqraMinimizerEdge &me = minView.edges_ptr[ge];
	                            if ((me.flags & SPQRA_MIN_EDGE_HAS_CHILD_REF) == 0 ||
	                                !SP_COMPRESS_CHILD_IS_EDGE(me.child_ref))
	                                continue;
	                            edge e{SP_COMPRESS_CHILD_AS_EDGE(me.child_ref)};
	                            node u = blockEdgeSource(blk, e);
	                            node v = blockEdgeTarget(blk, e);
	                            if (u == pole0 && v == pole1) ++cnt01;
	                            if (u == pole1 && v == pole0) ++cnt10;
	                        }
	                        for (auto &r : refs)
	                        {
	                            EdgeDPState &st = *r.toUpdate;
	                            if (st.s == pole0 && st.t == pole1)
	                            {
	                                st.directST |= cnt01 > 0;
	                                st.directTS |= cnt10 > 0;
	                            }
	                            else if (st.s == pole1 && st.t == pole0)
	                            {
	                                st.directST |= cnt10 > 0;
	                                st.directTS |= cnt01 > 0;
	                            }
	                        }
	                    }

	                    NodeDPState curr = A < nodeDp.size() ? nodeDp[A] : NodeDPState{};
	                    if (curr.outgoingCyclesCount >= 2)
	                    {
	                        for (auto &r : refs)
	                        {
	                            if (r.toUpdate->acyclic && r.other < nodeDp.size())
	                            {
	                                nodeDp[r.other].outgoingCyclesCount++;
	                                nodeDp[r.other].lastCycleNode = node{A};
	                            }
	                            r.toUpdate->acyclic = false;
	                        }
	                    }
	                    else if (curr.outgoingCyclesCount == 1)
	                    {
	                        for (auto &r : refs)
	                        {
	                            if (node{r.other} != curr.lastCycleNode)
	                            {
	                                if (r.toUpdate->acyclic && r.other < nodeDp.size())
	                                {
	                                    nodeDp[r.other].outgoingCyclesCount++;
	                                    nodeDp[r.other].lastCycleNode = node{A};
	                                }
	                                r.toUpdate->acyclic = false;
	                            }
	                            else
	                            {
	                                bool ac = tls_min_up_graph.isAcyclic(r.forwardEdge);
	                                if (r.toUpdate->acyclic && !ac && r.other < nodeDp.size())
	                                {
	                                    nodeDp[r.other].outgoingCyclesCount++;
	                                    nodeDp[r.other].lastCycleNode = node{A};
	                                }
	                                r.toUpdate->acyclic &= ac;
	                            }
	                        }
	                    }
	                    else
	                    {
	                        thread_local std::vector<edge> fasEdges;
	                        fasEdges.clear();
	                        bool ac = tls_min_up_graph.isAcyclic();
	                        if (!ac)
	                        {
	                            Graph g;
	                            tls_min_up_graph.materialize(g);
	                            FeedbackArcSet fas(g);
	                            ac = fas.run_or_acyclic(fasEdges);
	                        }
	                        if (!ac)
	                        {
	                            thread_local std::vector<uint8_t> inFas;
	                            const uint32_t nEdges = tls_min_up_graph.numberOfEdges();
	                            if (inFas.size() < nEdges) inFas.resize(nEdges);
	                            std::fill_n(inFas.begin(), nEdges, (uint8_t)0);
	                            for (edge e : fasEdges) inFas[e.idx] = 1;
	                            for (auto &r : refs)
	                            {
	                                bool keep = r.forwardEdge != SPQR_INVALID && inFas[r.forwardEdge];
	                                if (r.toUpdate->acyclic && !keep && r.other < nodeDp.size())
	                                {
	                                    nodeDp[r.other].outgoingCyclesCount++;
	                                    nodeDp[r.other].lastCycleNode = node{A};
	                                }
	                                r.toUpdate->acyclic &= keep;
	                            }
	                        }
	                    }

	                    for (auto &r : refs)
	                    {
	                        if (curr.outgoingSourceSinkCount >= 2 ||
	                            (curr.outgoingSourceSinkCount == 1 && node{r.other} != curr.lastSourceSinkNode) ||
	                            ((int)tls_min_up_isSourceSink[r.sLoc] +
	                             (int)tls_min_up_isSourceSink[r.tLoc] < localSourceSinkCount))
	                        {
	                            if (!r.toUpdate->globalSourceSink && r.other < nodeDp.size())
	                            {
	                                nodeDp[r.other].outgoingSourceSinkCount++;
	                                nodeDp[r.other].lastSourceSinkNode = node{A};
	                            }
	                            r.toUpdate->globalSourceSink = true;
	                        }
	                        if (curr.outgoingLeakageCount >= 2 ||
	                            (curr.outgoingLeakageCount == 1 && node{r.other} != curr.lastLeakageNode) ||
	                            ((int)tls_min_up_isLeaking[r.sLoc] +
	                             (int)tls_min_up_isLeaking[r.tLoc] < localLeakageCount))
	                        {
	                            if (!r.toUpdate->hasLeakage && r.other < nodeDp.size())
	                            {
	                                nodeDp[r.other].outgoingLeakageCount++;
	                                nodeDp[r.other].lastLeakageNode = node{A};
	                            }
	                            r.toUpdate->hasLeakage = true;
	                        }
	                        r.toUpdate->localInS = tls_min_up_inDeg[r.sLoc] - r.opposite->localInS;
	                        r.toUpdate->localOutS = tls_min_up_outDeg[r.sLoc] - r.opposite->localOutS;
	                        r.toUpdate->localInT = tls_min_up_inDeg[r.tLoc] - r.opposite->localInT;
	                        r.toUpdate->localOutT = tls_min_up_outDeg[r.tLoc] - r.opposite->localOutT;
	                    }
	                }
            }


            inline void tryBubbleCorePNodeGrouping(uint32_t A,
                                                   const CoreContext &c,
                                                   const SpCompressTreeView &view,
                                                   const std::vector<EdgeDPState> &macroStates,
                                                   const std::vector<EdgeDPState> &down,
                                                   const std::vector<EdgeDPState> &up,
                                                   const CcData &cc,
                                                   BlockData &blk)
            {
                if (c.types[A] != SPQR_NODE_TYPE_P)
                    return;
                node bS{coreLocalBlockId(c, A, 0)};
                node bT{coreLocalBlockId(c, A, 1)};
                uint32_t start = c.skelOffsets[A];
                uint32_t end = c.skelOffsets[A + 1];
                int directST = 0, directTS = 0;
                for (uint32_t i = start; i < end; ++i)
                {
                    const SkeletonEdge &se = c.skelEdges[i];
                    uint32_t cref = SPQR_INVALID;
                    if (!coreSkeletonChildRef(view, se, cref))
                        continue;
                    if (!SP_COMPRESS_CHILD_IS_EDGE(cref))
                        continue;
                    edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
                    node u = blockEdgeSource(blk, e);
                    node v = blockEdgeTarget(blk, e);
                    if (u == bS && v == bT) directST++;
                    else directTS++;
                }
                for (int q = 0; q < 2; ++q)
                {
                    std::vector<const EdgeDPState *> goodS, goodT;
                    BranchRef singleGood;
                    int localOutSSum = directST;
                    int localInTSum = directST;
                    auto weakBackBranch = [&](const EdgeDPState &state, int outS, int inT) {
                        int inS = (state.s == bS ? state.localInS : state.localInT);
                        int outT = (state.t == bT ? state.localOutT : state.localOutS);
                        return ctx().weakSuperbubbles && outS == 0 && inT == 0 &&
                               inS > 0 && outT > 0 &&
                               state.acyclic && !state.globalSourceSink && !state.hasLeakage;
                    };
                    for (uint32_t i = start; i < end; ++i)
                    {
                        const SkeletonEdge &se = c.skelEdges[i];
                        uint32_t cref = SPQR_INVALID;
                        if (coreSkeletonChildRef(view, se, cref))
                        {
                            if (SP_COMPRESS_CHILD_IS_EDGE(cref))
                                continue;
                            const EdgeDPState &state = macroStates[SP_COMPRESS_CHILD_AS_MACRO(cref)];
                            int localOutS = (state.s == bS ? state.localOutS : state.localOutT);
                            int localInT = (state.t == bT ? state.localInT : state.localInS);
                            localOutSSum += localOutS;
                            localInTSum += localInT;
                            if (weakBackBranch(state, localOutS, localInT))
                                continue;
                            if (localOutS > 0) {
                                goodS.push_back(&state);
                                singleGood = BranchRef::macro(SP_COMPRESS_CHILD_AS_MACRO(cref));
                            }
                            if (localInT > 0) goodT.push_back(&state);
                            continue;
                        }
                        uint32_t other = SPQR_INVALID;
                        if (!coreSkeletonTwin(c, se, other))
                            continue;
                        const EdgeDPState &state = (c.parents[other] == A ? down[other] : up[A]);
                        int localOutS = (state.s == bS ? state.localOutS : state.localOutT);
                        int localInT = (state.t == bT ? state.localInT : state.localInS);
                        localOutSSum += localOutS;
                        localInTSum += localInT;
                        if (weakBackBranch(state, localOutS, localInT))
                            continue;
                        if (localOutS > 0) {
                            goodS.push_back(&state);
                            singleGood = BranchRef::tree(other);
                        }
                        if (localInT > 0) goodT.push_back(&state);
                    }
                    bool good = true;
                    int badSourcePole = 0;
                    int badSinkPole = 0;
                    int badSAcyclic = 0, badSGlobalSourceSink = 0, badSLeakage = 0;
                    int badTAcyclic = 0, badTGlobalSourceSink = 0, badTLeakage = 0;
                    int badSAcyclicNoWeakBack = 0, badTAcyclicNoWeakBack = 0;
                    int badSourcePoleNoWeakBack = 0, badSinkPoleNoWeakBack = 0;
                    for (auto state : goodS)
                    {
                        if ((state->s == bS && state->localInS > 0) || (state->s == bT && state->localInT > 0))
                        {
                            good = false;
                            badSourcePole++;
                            if (!(state->directTS || state->directST)) badSourcePoleNoWeakBack++;
                        }
                        if (!state->acyclic)
                        {
                            badSAcyclic++;
                            if (!(state->directTS || state->directST)) badSAcyclicNoWeakBack++;
                        }
                        if (state->globalSourceSink) badSGlobalSourceSink++;
                        if (state->hasLeakage) badSLeakage++;
                        good &= state->acyclic;
                        good &= !state->globalSourceSink;
                        good &= !state->hasLeakage;
                    }
                    for (auto state : goodT)
                    {
                        if ((state->t == bT && state->localOutT > 0) || (state->t == bS && state->localOutS > 0))
                        {
                            good = false;
                            badSinkPole++;
                            if (!(state->directTS || state->directST)) badSinkPoleNoWeakBack++;
                        }
                        if (!state->acyclic)
                        {
                            badTAcyclic++;
                            if (!(state->directTS || state->directST)) badTAcyclicNoWeakBack++;
                        }
                        if (state->globalSourceSink) badTGlobalSourceSink++;
                        if (state->hasLeakage) badTLeakage++;
                        good &= state->acyclic;
                        good &= !state->globalSourceSink;
                        good &= !state->hasLeakage;
                    }
                    bool directOk = (ctx().weakSuperbubbles || directTS == 0);
                    bool sameSetOk = (goodS == goodT);
                    bool nonEmptyOk = (!goodS.empty() || directST > 1);
                    good &= directOk;
                    good &= sameSetOk;
                    good &= nonEmptyOk;
                    bool singletonOk = true;
                    if (goodS.size() == 1 && singleGood.valid())
                    {
                        if (singleGood.kind == BranchRef::Kind::TreeNode)
                        {
                            singletonOk = singleGood.id < c.len &&
                                          (c.types[singleGood.id] != SPQR_NODE_TYPE_S ||
                                           (ctx().weakSuperbubbles && directST > 0));
                        }
                        else if (singleGood.kind == BranchRef::Kind::MacroChild)
                        {
                            singletonOk = singleGood.id < view.macros_len;
                            if (singletonOk)
                            {
                                const SpCompressNode &m = view.macros_ptr[singleGood.id];
                                singletonOk = m.kind != SP_COMPRESS_KIND_SERIES || (ctx().weakSuperbubbles && directST > 0);
                            }
                        }
                    }
                    good &= singletonOk;
                    bool corePCountsGood =
                        (localOutSSum == blk.globOut[bS] && localInTSum == blk.globIn[bT]);
                    good &= corePCountsGood;
                    bool onlyCorePWeakBackFailures =
                        badSGlobalSourceSink == 0 && badTGlobalSourceSink == 0 &&
                        badSLeakage == 0 && badTLeakage == 0 &&
                        badSAcyclicNoWeakBack == 0 && badTAcyclicNoWeakBack == 0 &&
                        badSourcePoleNoWeakBack == 0 && badSinkPoleNoWeakBack == 0;
                    if (ctx().weakSuperbubbles && !good && corePCountsGood &&
                        sameSetOk && nonEmptyOk && singletonOk &&
                        goodS.size() > 1 && onlyCorePWeakBackFailures)
                    {
                        good = true;
                    }
                    if (good)
                    {
                        node rawS = cc.toOrig[blk.toCc[bS]];
                        node rawT = cc.toOrig[blk.toCc[bT]];
                        addSuperbubble(rawS, rawT);
                    }
                    std::swap(directST, directTS);
                    std::swap(bS, bT);
                }
            }

            inline void tryBubbleMacroPNodeGrouping(uint32_t mid,
                                                    const SpCompressTreeView &view,
                                                    const std::vector<EdgeDPState> &macroStates,
                                                    const CcData &cc,
                                                    BlockData &blk)
            {
                const SpCompressNode &m = view.macros_ptr[mid];
                if (m.kind != SP_COMPRESS_KIND_PARALLEL)
                    return;
                node bS{m.left};
                node bT{m.right};
                const EdgeDPState &whole = macroStates[mid];
                const bool qPossible0 = whole.localOutS == blk.globOut[bS] &&
                                        whole.localInT == blk.globIn[bT];
                const bool qPossible1 = whole.localOutT == blk.globOut[bT] &&
                                        whole.localInS == blk.globIn[bS];
                if (!qPossible0 && !qPossible1)
                    return;
                int directST = 0, directTS = 0;
                for (uint32_t i = 0; i < m.children_count; ++i)
                {
                    uint32_t cref = view.children_ptr[m.children_offset + i];
                    if (!SP_COMPRESS_CHILD_IS_EDGE(cref))
                        continue;
                    edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
                    node u = blockEdgeSource(blk, e);
                    node v = blockEdgeTarget(blk, e);
                    if (u == bS && v == bT) directST++;
                    else directTS++;
                }
                for (int q = 0; q < 2; ++q)
                {
                    if ((q == 0 && !qPossible0) || (q == 1 && !qPossible1))
                    {
                        std::swap(directST, directTS);
                        std::swap(bS, bT);
                        continue;
                    }
                    uint32_t goodSCount = 0;
                    uint32_t goodTCount = 0;
                    uint32_t singleGood = SPQR_INVALID;
                    int localOutSSum = directST;
                    int localInTSum = directST;
                    bool good = true;
                    bool sameSetOk = true;
                    int badSourcePole = 0;
                    int badSinkPole = 0;
                    int badSourcePoleNoWeakBack = 0;
                    int badSinkPoleNoWeakBack = 0;
                    int badAcyclicNoWeakBack = 0;
                    int badGlobalSourceSink = 0;
                    int badLeakage = 0;
                    auto weakBackBranch = [&](const EdgeDPState &state, int outS, int inT) {
                        int inS = (state.s == bS ? state.localInS : state.localInT);
                        int outT = (state.t == bT ? state.localOutT : state.localOutS);
                        return ctx().weakSuperbubbles && outS == 0 && inT == 0 &&
                               inS > 0 && outT > 0 &&
                               state.acyclic && !state.globalSourceSink && !state.hasLeakage;
                    };
                    for (uint32_t i = 0; i < m.children_count; ++i)
                    {
                        uint32_t cref = view.children_ptr[m.children_offset + i];
                        const EdgeDPState *state = nullptr;
                        if (SP_COMPRESS_CHILD_IS_EDGE(cref))
                        {
                            continue;
                        }
                        else
                        {
                            state = &macroStates[SP_COMPRESS_CHILD_AS_MACRO(cref)];
                        }
                        int localOutS = (state->s == bS ? state->localOutS : state->localOutT);
                        int localInT = (state->t == bT ? state->localInT : state->localInS);
                        localOutSSum += localOutS;
                        localInTSum += localInT;
                        if (weakBackBranch(*state, localOutS, localInT))
                            continue;
                        bool hasGoodS = localOutS > 0;
                        bool hasGoodT = localInT > 0;
                        sameSetOk &= (hasGoodS == hasGoodT);
                        if (hasGoodS)
                        {
                            ++goodSCount;
                            singleGood = i;
                        }
                        if (hasGoodT)
                            ++goodTCount;
                        if ((state->s == bS && state->localInS > 0) || (state->s == bT && state->localInT > 0))
                        {
                            good = false;
                            ++badSourcePole;
                            if (!(state->directTS || state->directST))
                                ++badSourcePoleNoWeakBack;
                        }
                        if ((state->t == bT && state->localOutT > 0) || (state->t == bS && state->localOutS > 0))
                        {
                            good = false;
                            ++badSinkPole;
                            if (!(state->directTS || state->directST))
                                ++badSinkPoleNoWeakBack;
                        }
                        if (!state->acyclic)
                        {
                            good = false;
                            if (!(state->directTS || state->directST))
                                ++badAcyclicNoWeakBack;
                        }
                        if (state->globalSourceSink)
                        {
                            good = false;
                            ++badGlobalSourceSink;
                        }
                        if (state->hasLeakage)
                        {
                            good = false;
                            ++badLeakage;
                        }
                    }
                    bool directOk = (ctx().weakSuperbubbles || directTS == 0);
                    bool nonEmptyOk = (goodSCount > 0 || directST > 1);
                    good &= directOk;
                    good &= sameSetOk;
                    good &= nonEmptyOk;
                    bool singletonOk = true;
                    if (goodSCount == 1 && singleGood != SPQR_INVALID)
                    {
                        uint32_t cref = view.children_ptr[m.children_offset + singleGood];
                        if (SP_COMPRESS_CHILD_IS_MACRO(cref))
                        {
                            const SpCompressNode &sub = view.macros_ptr[SP_COMPRESS_CHILD_AS_MACRO(cref)];
                            singletonOk = sub.kind != SP_COMPRESS_KIND_SERIES || (ctx().weakSuperbubbles && directST > 0);
                        }
                    }
                    good &= singletonOk;
                    bool countsGood = (localOutSSum == blk.globOut[bS] && localInTSum == blk.globIn[bT]);
                    good &= countsGood;
                    bool onlyWeakBackFailures =
                        badGlobalSourceSink == 0 && badLeakage == 0 &&
                        badAcyclicNoWeakBack == 0 &&
                        badSourcePoleNoWeakBack == 0 &&
                        badSinkPoleNoWeakBack == 0;
                    if (ctx().weakSuperbubbles && !good && countsGood &&
                        directOk && sameSetOk && nonEmptyOk && singletonOk &&
                        goodSCount > 1 && onlyWeakBackFailures)
                    {
                        good = true;
                    }
                    if (good)
                    {
                        node rawS = cc.toOrig[blk.toCc[bS]];
                        node rawT = cc.toOrig[blk.toCc[bT]];
                        addSuperbubble(rawS, rawT);
                    }
                    std::swap(directST, directTS);
                    std::swap(bS, bT);
                }
            }

            inline void subtractSubState(EdgeDPState &st, const EdgeDPState &sub)
            {
                if (sub.s == st.s)
                {
                    st.localOutS -= sub.localOutS;
                    st.localInS -= sub.localInS;
                }
                else if (sub.t == st.s)
                {
                    st.localOutS -= sub.localOutT;
                    st.localInS -= sub.localInT;
                }
                if (sub.s == st.t)
                {
                    st.localOutT -= sub.localOutS;
                    st.localInT -= sub.localInS;
                }
                else if (sub.t == st.t)
                {
                    st.localOutT -= sub.localOutT;
                    st.localInT -= sub.localInT;
                }
            }

            inline void subtractChildRefFromState(EdgeDPState &st,
                                                  uint32_t cref,
                                                  const std::vector<EdgeDPState> &macroStates,
                                                  BlockData &blk)
            {
                if (SP_COMPRESS_CHILD_IS_EDGE(cref))
                {
                    edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
                    node u = blockEdgeSource(blk, e);
                    node v = blockEdgeTarget(blk, e);
                    if (u == st.s) st.localOutS--;
                    if (v == st.s) st.localInS--;
                    if (u == st.t) st.localOutT--;
                    if (v == st.t) st.localInT--;
                }
                else
                {
                    subtractSubState(st, macroStates[SP_COMPRESS_CHILD_AS_MACRO(cref)]);
                }
            }

            inline void mergeDirectFlags(EdgeDPState &st, const EdgeDPState &sub)
            {
                if (sub.s == st.s && sub.t == st.t)
                {
                    st.directST |= sub.directST;
                    st.directTS |= sub.directTS;
                }
                else if (sub.s == st.t && sub.t == st.s)
                {
                    st.directST |= sub.directTS;
                    st.directTS |= sub.directST;
                }
            }

            inline uint8_t macroSpqrType(const SpCompressNode &m)
            {
                return m.kind == SP_COMPRESS_KIND_PARALLEL ? SPQR_NODE_TYPE_P : SPQR_NODE_TYPE_S;
            }

            inline bool buildMacroSeriesPath(const SpCompressTreeView &view,
                                             const SpCompressNode &m,
                                             BlockData &blk,
                                             std::vector<uint32_t> &order)
            {
                auto childEndpoints = [&](uint32_t cref, uint32_t &a, uint32_t &b) {
                    if (SP_COMPRESS_CHILD_IS_EDGE(cref))
                    {
                        edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
                        a = blockEdgeSource(blk, e).idx;
                        b = blockEdgeTarget(blk, e).idx;
                    }
                    else
                    {
                        const SpCompressNode &sub = view.macros_ptr[SP_COMPRESS_CHILD_AS_MACRO(cref)];
                        a = sub.left;
                        b = sub.right;
                    }
                };

                const uint32_t k = m.children_count;
                order.clear();
                order.reserve(k);

                uint32_t cur = m.left;
                bool orderedPath = true;
                for (uint32_t i = 0; i < k; ++i)
                {
                    uint32_t cref = view.children_ptr[m.children_offset + i];
                    uint32_t a = SPQR_INVALID, b = SPQR_INVALID;
                    childEndpoints(cref, a, b);
                    if (a == SPQR_INVALID || b == SPQR_INVALID)
                    {
                        orderedPath = false;
                        break;
                    }
                    uint32_t next = SPQR_INVALID;
                    if (a == cur) next = b;
                    else if (b == cur) next = a;
                    else
                    {
                        orderedPath = false;
                        break;
                    }
                    order.push_back(i);
                    cur = next;
                }
                if (orderedPath && cur == m.right)
                    return true;

                struct PathChild
                {
                    uint32_t cref{SPQR_INVALID};
                    uint32_t a{SPQR_INVALID};
                    uint32_t b{SPQR_INVALID};
                };
                thread_local std::vector<PathChild> children;
                thread_local std::vector<std::pair<uint32_t, uint32_t>> incidence;
                thread_local std::vector<uint8_t> visited;
                children.clear();
                incidence.clear();
                children.reserve(k);
                incidence.reserve(static_cast<size_t>(k) * 2);
                order.clear();
                for (uint32_t i = 0; i < k; ++i)
                {
                    uint32_t cref = view.children_ptr[m.children_offset + i];
                    uint32_t a = SPQR_INVALID, b = SPQR_INVALID;
                    childEndpoints(cref, a, b);
                    if (a == SPQR_INVALID || b == SPQR_INVALID)
                        return false;
                    children.push_back(PathChild{cref, a, b});
                    incidence.emplace_back(a, i);
                    incidence.emplace_back(b, i);
                }
                std::sort(incidence.begin(), incidence.end());
                visited.assign(k, 0);
                uint32_t current = m.left;
                uint32_t prev = SPQR_INVALID;
                order.reserve(k);
                for (uint32_t step = 0; step < k; ++step)
                {
                    auto it = std::lower_bound(
                        incidence.begin(), incidence.end(), current,
                        [](const std::pair<uint32_t, uint32_t> &p, uint32_t v) {
                            return p.first < v;
                        });
                    uint32_t chosen = SPQR_INVALID;
                    for (; it != incidence.end() && it->first == current; ++it)
                    {
                        uint32_t idx = it->second;
                        if (idx == prev || visited[idx])
                            continue;
                        chosen = idx;
                        break;
                    }
                    if (chosen == SPQR_INVALID)
                        return false;
                    const PathChild &ch = children[chosen];
                    uint32_t next = SPQR_INVALID;
                    if (ch.a == current) next = ch.b;
                    else if (ch.b == current) next = ch.a;
                    else return false;
                    visited[chosen] = 1;
                    order.push_back(chosen);
                    prev = chosen;
                    current = next;
                }
                if (current != m.right)
                    return false;
                for (uint8_t v : visited)
                    if (!v)
                        return false;
                return true;
            }

            struct MacroContext
            {
                std::vector<EdgeDPState> outside;
                std::vector<uint8_t> hasOutside;
                std::vector<uint32_t> parent;
                std::vector<uint32_t> parentType;
            };

            inline MacroContext computeMacroOutsideStates(const CoreContext &c,
                                                          const SpCompressTreeView &view,
                                                          const std::vector<EdgeDPState> &macroStates,
                                                          const std::vector<EdgeDPState> &down,
                                                          const std::vector<EdgeDPState> &up,
                                                          BlockData &blk)
            {
                const uint32_t M = view.macros_len;
                MacroContext out;
                out.outside.resize(M);
                out.hasOutside.assign(M, 0);
                out.parent.assign(M, SPQR_INVALID);
                out.parentType.assign(M, SPQR_INVALID);
                for (uint32_t mid = 0; mid < M; ++mid)
                {
                    const SpCompressNode &m = view.macros_ptr[mid];
                    for (uint32_t i = 0; i < m.children_count; ++i)
                    {
                        uint32_t cref = view.children_ptr[m.children_offset + i];
                        if (SP_COMPRESS_CHILD_IS_MACRO(cref))
                        {
                            uint32_t sub = SP_COMPRESS_CHILD_AS_MACRO(cref);
                            if (sub < M)
                                out.parent[sub] = mid;
                        }
                    }
                }
                for (uint32_t tn = 0; tn < c.len; ++tn)
                {
                    if (c.types[tn] != SPQR_NODE_TYPE_P)
                        continue;
                    node pole0{coreLocalBlockId(c, tn, 0)};
                    node pole1{coreLocalBlockId(c, tn, 1)};
                    EdgeDPState total;
                    resetMacroState(total, pole0, pole1);
                    uint32_t start = c.skelOffsets[tn], end = c.skelOffsets[tn + 1];
                    for (uint32_t i = start; i < end; ++i)
                    {
                        const SkeletonEdge &se = c.skelEdges[i];
                        uint32_t cref = SPQR_INVALID;
                        if (coreSkeletonChildRef(view, se, cref))
                        {
                            addChildRefToState(total, cref, macroStates, blk);
                            if (SP_COMPRESS_CHILD_IS_EDGE(cref))
                            {
                                edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
                                node u = blockEdgeSource(blk, e);
                                node v = blockEdgeTarget(blk, e);
                                if (u == total.s && v == total.t) total.directST = true;
                                if (u == total.t && v == total.s) total.directTS = true;
                            }
                        }
                        else
                        {
                            uint32_t other = SPQR_INVALID;
                            if (!coreSkeletonTwin(c, se, other))
                                continue;
                            mergeSubState(total, other == c.parents[tn] ? up[tn] : down[other]);
                        }
                    }
                    for (uint32_t i = start; i < end; ++i)
                    {
                        const SkeletonEdge &se = c.skelEdges[i];
                        uint32_t cref = SPQR_INVALID;
                        if (!coreSkeletonChildRef(view, se, cref))
                            continue;
                        if (!SP_COMPRESS_CHILD_IS_MACRO(cref))
                            continue;
                        uint32_t mid = SP_COMPRESS_CHILD_AS_MACRO(cref);
                        if (mid >= M || out.parent[mid] != SPQR_INVALID)
                            continue;
                        const SpCompressNode &m = view.macros_ptr[mid];
                        EdgeDPState &st = out.outside[mid];
                        resetMacroState(st, node{m.left}, node{m.right});
                        mergeSubState(st, total);
                        mergeDirectFlags(st, total);
                        subtractChildRefFromState(st, cref, macroStates, blk);
                        out.hasOutside[mid] = 1;
                        out.parentType[mid] = SPQR_NODE_TYPE_P;
                    }
                }
                std::vector<uint32_t> stack;
                stack.reserve(M);
                for (uint32_t mid = 0; mid < M; ++mid)
                    if (out.parent[mid] == SPQR_INVALID)
                        stack.push_back(mid);
                std::vector<uint32_t> order;
                while (!stack.empty())
                {
                    uint32_t pid = stack.back();
                    stack.pop_back();
                    const SpCompressNode &p = view.macros_ptr[pid];
                    if (p.kind == SP_COMPRESS_KIND_PARALLEL)
                    {
                        EdgeDPState total;
                        resetMacroState(total, node{p.left}, node{p.right});
                        for (uint32_t i = 0; i < p.children_count; ++i)
                        {
                            uint32_t cref = view.children_ptr[p.children_offset + i];
                            addChildRefToState(total, cref, macroStates, blk);
                        }
                        if (out.hasOutside[pid])
                            mergeSubState(total, out.outside[pid]);
                        for (uint32_t i = 0; i < p.children_count; ++i)
                        {
                            uint32_t cref = view.children_ptr[p.children_offset + i];
                            if (!SP_COMPRESS_CHILD_IS_MACRO(cref))
                                continue;
                            uint32_t sub = SP_COMPRESS_CHILD_AS_MACRO(cref);
                            if (sub >= M)
                                continue;
                            const SpCompressNode &m = view.macros_ptr[sub];
                            EdgeDPState &st = out.outside[sub];
                            resetMacroState(st, node{m.left}, node{m.right});
                            mergeSubState(st, total);
                            mergeDirectFlags(st, total);
                            subtractChildRefFromState(st, cref, macroStates, blk);
                            out.hasOutside[sub] = 1;
                            out.parentType[sub] = macroSpqrType(p);
                            stack.push_back(sub);
                        }
                    }
                    else
                    {
                        if (!buildMacroSeriesPath(view, p, blk, order))
                            continue;
                        for (uint32_t i = 0; i < order.size(); ++i)
                        {
                            uint32_t childIdx = order[i];
                            uint32_t cref = view.children_ptr[p.children_offset + childIdx];
                            if (!SP_COMPRESS_CHILD_IS_MACRO(cref))
                                continue;
                            uint32_t sub = SP_COMPRESS_CHILD_AS_MACRO(cref);
                            if (sub >= M)
                                continue;
                            const SpCompressNode &m = view.macros_ptr[sub];
                            EdgeDPState &st = out.outside[sub];
                            resetMacroState(st, node{m.left}, node{m.right});
                            if (i > 0)
                            {
                                uint32_t prevRef = view.children_ptr[p.children_offset + order[i - 1]];
                                addChildRefToState(st, prevRef, macroStates, blk);
                            }
                            if (i + 1 < order.size())
                            {
                                uint32_t nextRef = view.children_ptr[p.children_offset + order[i + 1]];
                                addChildRefToState(st, nextRef, macroStates, blk);
                            }
                            if (out.hasOutside[pid])
                                mergeSubState(st, out.outside[pid]);
                            out.hasOutside[sub] = 1;
                            out.parentType[sub] = macroSpqrType(p);
                            stack.push_back(sub);
                        }
                    }
                }
                return out;
            }

            inline void collectMacroTreeSuperbubbles(const SpCompressTreeView &view,
                                                     const std::vector<EdgeDPState> &macroStates,
                                                     const MacroContext &macroCtx,
                                                     const CcData &cc,
                                                     BlockData &blk)
            {
                for (uint32_t mid = 0; mid < view.macros_len; ++mid)
                {
                    if (!macroCtx.hasOutside[mid])
                        continue;
                    uint32_t parentType = macroCtx.parentType[mid];
                    if (parentType == SPQR_INVALID)
                        continue;
                    uint8_t childType = macroSpqrType(view.macros_ptr[mid]);
                    const EdgeDPState &downSt = macroStates[mid];
                    const EdgeDPState &upSt = macroCtx.outside[mid];
                    if ((childType == SPQR_NODE_TYPE_P && parentType == SPQR_NODE_TYPE_S) ||
                        (childType == SPQR_NODE_TYPE_S && parentType == SPQR_NODE_TYPE_P))
                    {
                        tryBubble(downSt, upSt, blk, cc, false, true);
                        tryBubble(downSt, upSt, blk, cc, true, true);
                    }
                    if ((parentType == SPQR_NODE_TYPE_R && childType != SPQR_NODE_TYPE_P) ||
                        (parentType == SPQR_NODE_TYPE_P && childType == SPQR_NODE_TYPE_S) ||
                        (parentType == SPQR_NODE_TYPE_S && childType == SPQR_NODE_TYPE_P))
                    {
                        tryBubble(upSt, downSt, blk, cc, false, true);
                        tryBubble(upSt, downSt, blk, cc, true, true);
                    }
                }
            }

            inline void collectCoreSuperbubbles(const CoreContext &c,
                                                const SpCompressTreeView &view,
                                                const std::vector<EdgeDPState> &macroStates,
                                                const MacroContext &macroCtx,
                                                const std::vector<EdgeDPState> &down,
                                                const std::vector<EdgeDPState> &up,
                                                const CcData &cc,
                                                BlockData &blk,
                                                const std::vector<PFixedAggEntry> *corePAgg = nullptr)
            {
                for (uint32_t child = 0; child < c.len; ++child)
                {
                    if (child == c.root)
                        continue;
                    uint32_t parent = c.parents[child];
                    const EdgeDPState &downSt = down[child];
                    const EdgeDPState &upSt = up[child];
                    uint8_t parentType = c.types[parent];
                    uint8_t childType = c.types[child];
                    if ((childType == SPQR_NODE_TYPE_R && parentType != SPQR_NODE_TYPE_P) ||
                        (childType == SPQR_NODE_TYPE_P && parentType == SPQR_NODE_TYPE_S))
                    {
                        bool check = childType == SPQR_NODE_TYPE_P;
                        tryBubble(downSt, upSt, blk, cc, false, check);
                        tryBubble(downSt, upSt, blk, cc, true, check);
                    }
                    if (childType == SPQR_NODE_TYPE_S && parentType == SPQR_NODE_TYPE_P)
                    {
                        tryBubble(downSt, upSt, blk, cc, false, true);
                        tryBubble(downSt, upSt, blk, cc, true, true);
                    }
                    if ((parentType == SPQR_NODE_TYPE_R && childType != SPQR_NODE_TYPE_P) ||
                        (parentType == SPQR_NODE_TYPE_P && childType == SPQR_NODE_TYPE_S))
                    {
                        bool check = parentType == SPQR_NODE_TYPE_P;
                        tryBubble(upSt, downSt, blk, cc, false, check);
                        tryBubble(upSt, downSt, blk, cc, true, check);
                    }
                    if (parentType == SPQR_NODE_TYPE_S && childType == SPQR_NODE_TYPE_P)
                    {
                        tryBubble(upSt, downSt, blk, cc, false, true);
                        tryBubble(upSt, downSt, blk, cc, true, true);
                    }
                    blk.isAcycic &= (downSt.acyclic && upSt.acyclic);
                }
                for (uint32_t tn = 0; tn < c.len; ++tn)
                {
                    if (tryBubbleCorePNodeGroupingFixed(tn, c, corePAgg, cc, blk))
                        continue;
                    tryBubbleCorePNodeGrouping(tn, c, view, macroStates, down, up, cc, blk);
                }
                if (!macroCtx.hasOutside.empty())
                    collectMacroTreeSuperbubbles(view, macroStates, macroCtx, cc, blk);
                for (uint32_t mid = 0; mid < view.macros_len; ++mid)
                    tryBubbleMacroPNodeGrouping(mid, view, macroStates, cc, blk);
            }

            struct AQCoreEdgeTask
            {
                uint32_t child{SPQR_INVALID};
                uint32_t parent{SPQR_INVALID};
                bool currDown{true};
                bool swap{false};
                bool additionalCheck{false};
            };

            struct AQMacroEdgeTask
            {
                uint32_t macroId{SPQR_INVALID};
                bool currDown{true};
                bool swap{false};
                bool additionalCheck{true};
            };

            struct AQContext
            {
                std::vector<AQCoreEdgeTask> coreEdgeTasks;
                std::vector<uint32_t> corePNodes;
                std::vector<AQMacroEdgeTask> macroEdgeTasks;
                std::vector<uint32_t> macroPNodes;
            };

            inline AQContext buildAQContextFromSpqrTree(const CoreContext &c,
                                                        const SpCompressTreeView &view,
                                                        const MacroContext &macroCtx,
                                                        bool skipMacroPNodes = false,
                                                        bool skipCoreEdgeTasks = false)
            {
                AQContext aq;
                if (!skipCoreEdgeTasks)
                    aq.coreEdgeTasks.reserve(static_cast<size_t>(c.len) * 2);
                aq.corePNodes.reserve(c.len / 4 + 1);
                aq.macroPNodes.reserve(view.macros_len / 4 + 1);

                auto addCore = [&](uint32_t child,
                                   uint32_t parent,
                                   bool currDown,
                                   bool swap,
                                   bool additionalCheck) {
                    aq.coreEdgeTasks.push_back(AQCoreEdgeTask{
                        child,
                        parent,
                        currDown,
                        swap,
                        additionalCheck});
                };

                if (!skipCoreEdgeTasks)
                for (uint32_t child = 0; child < c.len; ++child)
                {
                    if (child == c.root)
                        continue;
                    uint32_t parent = c.parents[child];
                    uint8_t parentType = c.types[parent];
                    uint8_t childType = c.types[child];
                    if ((childType == SPQR_NODE_TYPE_R && parentType != SPQR_NODE_TYPE_P) ||
                        (childType == SPQR_NODE_TYPE_P && parentType == SPQR_NODE_TYPE_S))
                    {
                        bool check = childType == SPQR_NODE_TYPE_P;
                        addCore(child, parent, true, false, check);
                        addCore(child, parent, true, true, check);
                    }
                    if (childType == SPQR_NODE_TYPE_S && parentType == SPQR_NODE_TYPE_P)
                    {
                        addCore(child, parent, true, false, true);
                        addCore(child, parent, true, true, true);
                    }
                    if ((parentType == SPQR_NODE_TYPE_R && childType != SPQR_NODE_TYPE_P) ||
                        (parentType == SPQR_NODE_TYPE_P && childType == SPQR_NODE_TYPE_S))
                    {
                        bool check = parentType == SPQR_NODE_TYPE_P;
                        addCore(child, parent, false, false, check);
                        addCore(child, parent, false, true, check);
                    }
                    if (parentType == SPQR_NODE_TYPE_S && childType == SPQR_NODE_TYPE_P)
                    {
                        addCore(child, parent, false, false, true);
                        addCore(child, parent, false, true, true);
                    }
                }

                for (uint32_t tn = 0; tn < c.len; ++tn)
                    if (c.types[tn] == SPQR_NODE_TYPE_P)
                        aq.corePNodes.push_back(tn);

                if (!macroCtx.outside.empty())
                {
                    aq.macroEdgeTasks.reserve(view.macros_len / 8 + 1);
                    for (uint32_t mid = 0; mid < view.macros_len; ++mid)
                    {
                        if (!macroCtx.hasOutside[mid])
                            continue;
                        uint32_t parentType = macroCtx.parentType[mid];
                        if (parentType == SPQR_INVALID)
                            continue;
                        uint8_t childType = macroSpqrType(view.macros_ptr[mid]);
                        if ((childType == SPQR_NODE_TYPE_P && parentType == SPQR_NODE_TYPE_S) ||
                            (childType == SPQR_NODE_TYPE_S && parentType == SPQR_NODE_TYPE_P))
                        {
                            aq.macroEdgeTasks.push_back(AQMacroEdgeTask{mid, true, false, true});
                            aq.macroEdgeTasks.push_back(AQMacroEdgeTask{mid, true, true, true});
                        }
                        if ((parentType == SPQR_NODE_TYPE_R && childType != SPQR_NODE_TYPE_P) ||
                            (parentType == SPQR_NODE_TYPE_P && childType == SPQR_NODE_TYPE_S) ||
                            (parentType == SPQR_NODE_TYPE_S && childType == SPQR_NODE_TYPE_P))
                        {
                            aq.macroEdgeTasks.push_back(AQMacroEdgeTask{mid, false, false, true});
                            aq.macroEdgeTasks.push_back(AQMacroEdgeTask{mid, false, true, true});
                        }
                    }
                }

                if (!skipMacroPNodes)
                {
                    for (uint32_t mid = 0; mid < view.macros_len; ++mid)
                        if (view.macros_ptr[mid].kind == SP_COMPRESS_KIND_PARALLEL)
                            aq.macroPNodes.push_back(mid);
                }
                return aq;
            }

            inline void executeAQContextFromSpqrTree(const AQContext &aq,
                                                     const CoreContext &c,
                                                     const SpCompressTreeView &view,
                                                     const std::vector<EdgeDPState> &macroStates,
                                                     const MacroContext &macroCtx,
	                                                     const std::vector<EdgeDPState> &down,
	                                                     const std::vector<EdgeDPState> &up,
	                                                     const CcData &cc,
		                                                     BlockData &blk,
		                                                     const std::vector<PFixedAggEntry> *corePAgg = nullptr)
	            {
	                for (uint32_t child = 0; child < c.len; ++child)
	                {
	                    if (child == c.root)
                        continue;
                    blk.isAcycic &= (down[child].acyclic && up[child].acyclic);
                }
                for (const AQCoreEdgeTask &task : aq.coreEdgeTasks)
                {
                    const EdgeDPState &curr = task.currDown ? down[task.child] : up[task.child];
                    const EdgeDPState &back = task.currDown ? up[task.child] : down[task.child];
	                    tryBubble(curr, back, blk, cc, task.swap, task.additionalCheck);
	                }
		                for (uint32_t tn : aq.corePNodes)
		                {
		                    if (tryBubbleCorePNodeGroupingFixed(tn, c, corePAgg, cc, blk))
	                        continue;
		                    tryBubbleCorePNodeGrouping(tn, c, view, macroStates, down, up, cc, blk);
		                }
	                for (const AQMacroEdgeTask &task : aq.macroEdgeTasks)
	                {
	                    const EdgeDPState &downSt = macroStates[task.macroId];
                    const EdgeDPState &upSt = macroCtx.outside[task.macroId];
                    const EdgeDPState &curr = task.currDown ? downSt : upSt;
                    const EdgeDPState &back = task.currDown ? upSt : downSt;
	                    tryBubble(curr, back, blk, cc, task.swap, task.additionalCheck);
	                }
	                for (uint32_t mid : aq.macroPNodes)
	                    tryBubbleMacroPNodeGrouping(mid, view, macroStates, cc, blk);
		            }

	            inline bool collectAQAtomsFromMinimizerMessages(const CoreScadView &scad,
	                                                            const SpqraMinimizerView &minView,
	                                                            const SpCompressTreeView &view,
	                                                            const std::vector<EdgeDPState> &macroStates,
	                                                            const std::vector<EdgeDPState> &down,
	                                                            const std::vector<EdgeDPState> &up,
	                                                            const CcData &cc,
	                                                            BlockData &blk)
	            {
	                if (!minimizerDirectCollectRequested())
	                {
	                    if (!minimizerMessageKernelRequested())
	                        return false;
	                }
	                if (!scad.components_ptr || !scad.incidences_ptr ||
	                    !minView.components_ptr || !minView.edges_ptr ||
	                    !minView.node_mapping_ptr || minView.components_len == 0 ||
	                    minView.components_len != scad.components_len ||
	                    minView.edges_len != scad.edges_len ||
	                    minView.node_mapping_len == 0 ||
	                    minView.summary.bad_twin_count != 0)
	                {
	                    return false;
	                }

		                const bool coreEdgesAlreadyEmitted =
		                    std::getenv("BF_SPQRA_CORE_EXACT_EMIT_DURING_UP") != nullptr &&
		                    !minimizerMessageKernelRequested();
		                for (uint32_t ii = 0; ii < scad.incidences_len; ++ii)
	                {
	                    uint32_t twin = SPQR_INVALID;
	                    if (!scadValidTwinPair(scad, ii, twin) || ii > twin)
	                        continue;
	                    const uint32_t a = scad.incidences_ptr[ii].component_id;
	                    const uint32_t b = scad.incidences_ptr[twin].component_id;
		                    if (a >= minView.components_len || b >= minView.components_len)
		                    {
		                        continue;
		                    }

	                    uint32_t child = SPQR_INVALID;
	                    uint32_t parent = SPQR_INVALID;
	                    uint32_t childPort = SPQR_INVALID;
	                    uint32_t parentPort = SPQR_INVALID;
	                    if (minView.components_ptr[a].parent == b)
	                    {
	                        child = a;
	                        parent = b;
	                        childPort = ii;
	                        parentPort = twin;
	                    }
	                    else if (minView.components_ptr[b].parent == a)
	                    {
	                        child = b;
	                        parent = a;
	                        childPort = twin;
	                        parentPort = ii;
	                    }
		                    else
		                    {
		                        continue;
		                    }

	                    const EdgeDPState *downSt =
	                        minimizerPortMessage(childPort, scad, minView, down, up);
	                    const EdgeDPState *upSt =
	                        minimizerPortMessage(parentPort, scad, minView, down, up);
		                    if (!downSt || !upSt)
		                    {
		                        continue;
		                    }

	                    const uint8_t childType =
	                        scadKindToSpqrType(minView.components_ptr[child].kind);
	                    const uint8_t parentType =
	                        scadKindToSpqrType(minView.components_ptr[parent].kind);
		                    if (!coreEdgesAlreadyEmitted)
		                    {
		                        tryCoreTreeEdgeBubblesByTypes(childType, parentType,
		                                                      *downSt, *upSt, cc, blk);
		                    }
		                    blk.isAcycic &= (downSt->acyclic && upSt->acyclic);
		                }

		                std::vector<uint32_t> edgeToInc;
	                edgeToInc.assign(scad.edges_len, SPQR_INVALID);
	                for (uint32_t ii = 0; ii < scad.incidences_len; ++ii)
	                {
	                    const FfiScadIncidence &inc = scad.incidences_ptr[ii];
	                    if (inc.component_id >= scad.components_len)
	                        continue;
	                    const FfiScadComponent &comp = scad.components_ptr[inc.component_id];
	                    const uint32_t ge = comp.edge_begin + inc.local_edge_id;
	                    if (ge < scad.edges_len && ge < edgeToInc.size())
	                        edgeToInc[ge] = ii;
	                }

	                for (uint32_t ci = 0; ci < minView.components_len; ++ci)
	                {
	                    const FfiSpqraMinimizerComponent &comp = minView.components_ptr[ci];
		                    if (scadKindToSpqrType(comp.kind) != SPQR_NODE_TYPE_P)
		                        continue;
		                    const uint32_t pole0 = minimizerLocalBlockId(minView, blk, comp, 0);
		                    const uint32_t pole1 = minimizerLocalBlockId(minView, blk, comp, 1);
		                    if (pole0 == SPQR_INVALID || pole1 == SPQR_INVALID)
		                    {
		                        continue;
		                    }

	                    PFixedAggEntry entry;
	                    node bS{pole0};
	                    node bT{pole1};
	                    const uint32_t edgeEnd = std::min(comp.edge_end, minView.edges_len);
	                    for (uint32_t ge = comp.edge_begin; ge < edgeEnd; ++ge)
	                    {
	                        const FfiSpqraMinimizerEdge &me = minView.edges_ptr[ge];
	                        if (me.flags & SPQRA_MIN_EDGE_VIRTUAL)
	                        {
		                            if (me.twin_component >= minView.components_len)
		                            {
		                                continue;
		                            }
	                            const FfiSpqraMinimizerComponent &twComp =
	                                minView.components_ptr[me.twin_component];
	                            const uint32_t twGe =
	                                twComp.edge_begin + me.twin_local_edge;
		                            if (twGe >= edgeToInc.size())
		                            {
		                                continue;
		                            }
	                            const uint32_t twPort = edgeToInc[twGe];
	                            const EdgeDPState *state =
	                                minimizerPortMessage(twPort, scad, minView, down, up);
		                            if (!state)
		                            {
		                                continue;
		                            }
	                            const bool seriesLike =
	                                me.twin_component < minView.components_len &&
	                                scadKindToSpqrType(
	                                    minView.components_ptr[me.twin_component].kind) ==
	                                    SPQR_NODE_TYPE_S;
	                            entry.fwd.addBranch(*state, bS, bT, seriesLike,
	                                                PFixedAggMode::Core);
	                            entry.rev.addBranch(*state, bT, bS, seriesLike,
	                                                PFixedAggMode::Core);
	                            continue;
	                        }

	                        if ((me.flags & SPQRA_MIN_EDGE_HAS_CHILD_REF) == 0)
	                            continue;
	                        const uint32_t cref = me.child_ref;
	                        if (SP_COMPRESS_CHILD_IS_EDGE(cref))
	                        {
	                            edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
	                            node u = blockEdgeSource(blk, e);
	                            node v = blockEdgeTarget(blk, e);
	                            if (u == bS && v == bT)
	                                ++entry.directST;
	                            else
	                                ++entry.directTS;
	                        }
	                        else
	                        {
	                            const uint32_t mid = SP_COMPRESS_CHILD_AS_MACRO(cref);
		                            if (mid >= macroStates.size() || mid >= view.macros_len)
		                            {
		                                continue;
		                            }
	                            const bool seriesLike =
	                                view.macros_ptr[mid].kind == SP_COMPRESS_KIND_SERIES;
	                            entry.fwd.addBranch(macroStates[mid], bS, bT, seriesLike,
	                                                PFixedAggMode::Core);
	                            entry.rev.addBranch(macroStates[mid], bT, bS, seriesLike,
	                                                PFixedAggMode::Core);
	                        }
	                    }
	                    entry.valid = 1;
	                    emitPFixedAggCandidate(entry.fwd,
	                                           bS,
	                                           bT,
	                                           entry.directST,
	                                           entry.directTS,
	                                           true,
	                                           cc,
	                                           blk);
	                    emitPFixedAggCandidate(entry.rev,
	                                           bT,
	                                           bS,
	                                           entry.directTS,
	                                           entry.directST,
	                                           true,
		                                           cc,
		                                           blk);
		                }
		                return true;
		            }

	            inline uint64_t emitBehaviorAtomInternalCoreEdges(
	                const SpqraBehaviorAtomView &atoms,
	                const SpqraBehaviorAtomStates &atomStates,
	                const CcData &cc,
	                BlockData &blk,
	                uint64_t *badPorts = nullptr)
	            {
	                if (!emitNestedAtomCoreEdgesRequested() ||
	                    !atoms.atoms_ptr || !atoms.items_ptr)
	                    return 0;
	                uint64_t taskCount = 0;
	                uint64_t bad = 0;
	                for (uint32_t parentId = 0; parentId < atoms.atoms_len; ++parentId)
	                {
	                    const FfiSpqraBehaviorAtom &parent = atoms.atoms_ptr[parentId];
	                    const uint8_t parentType = scadKindToSpqrType(parent.kind);
	                    const uint32_t itemBegin = std::min(parent.item_begin, atoms.items_len);
	                    const uint32_t itemEnd = std::min(parent.item_end, atoms.items_len);
	                    for (uint32_t ii = itemBegin; ii < itemEnd; ++ii)
	                    {
	                        const FfiSpqraBehaviorAtomItem &item = atoms.items_ptr[ii];
	                        if ((item.flags & SPQRA_MIN_ATOM_ITEM_BEHAVIOR_ATOM) == 0)
	                            continue;
	                        const uint32_t childId = item.child_ref;
	                        if (childId >= atoms.atoms_len ||
	                            childId >= atomStates.fwd.size() ||
	                            childId >= atomStates.rev.size() ||
	                            childId >= atomStates.outside.size())
	                        {
	                            ++bad;
	                            continue;
	                        }
	                        const EdgeDPState *inside =
	                            nestedBehaviorAtomStateForItem(item,
	                                                           atoms,
	                                                           atomStates.fwd,
	                                                           atomStates.rev,
	                                                           blk,
	                                                           parentId);
	                        const EdgeDPState &outside = atomStates.outside[childId];
	                        if (!inside || !inside->s || !inside->t ||
	                            !outside.s || !outside.t)
	                        {
	                            ++bad;
	                            continue;
	                        }
	                        const uint8_t childType =
	                            scadKindToSpqrType(atoms.atoms_ptr[childId].kind);
	                        taskCount +=
	                            scadTreeEdgeSideObservableMessages(childType, parentType) +
	                            scadTreeEdgeSideObservableMessages(parentType, childType);
	                        tryCoreTreeEdgeBubblesByTypes(childType,
	                                                      parentType,
	                                                      *inside,
	                                                      outside,
	                                                      cc,
	                                                      blk);
	                    }
	                }
	                if (badPorts)
	                    *badPorts += bad;
	                return taskCount;
	            }

	            inline bool collectAQAtomsFromMinimizerViewMessages(const SpqraMinimizerView &minView,
	                                                                const SpCompressTreeView &view,
	                                                                const std::vector<EdgeDPState> &macroStates,
	                                                                const std::vector<EdgeDPState> &down,
	                                                                const std::vector<EdgeDPState> &up,
	                                                                const CcData &cc,
	                                                                BlockData &blk,
	                                                                const SpqraBehaviorAtomStates *atomStates = nullptr)
	            {
	                if (!minimizerMessageKernelRequested())
	                    return false;
	                if (!minView.components_ptr || !minView.edges_ptr ||
	                    !minView.node_mapping_ptr || minView.components_len == 0 ||
	                    minView.summary.bad_twin_count != 0)
	                {
	                    return false;
	                }

		                for (uint32_t child = 0; child < minView.components_len; ++child)
	                {
	                    const FfiSpqraMinimizerComponent &comp =
	                        minView.components_ptr[child];
	                    const uint32_t parent = comp.parent;
	                    if (parent == SPQR_INVALID || parent >= minView.components_len)
	                        continue;
		                    if (child >= down.size() || child >= up.size())
		                    {
		                        continue;
		                    }
	                    const EdgeDPState &downSt = down[child];
	                    const EdgeDPState &upSt = up[child];
	                    const uint8_t childType = scadKindToSpqrType(comp.kind);
	                    const uint8_t parentType =
	                        scadKindToSpqrType(minView.components_ptr[parent].kind);
			                    tryCoreTreeEdgeBubblesByTypes(childType, parentType,
			                                                  downSt, upSt, cc, blk);
		                    blk.isAcycic &= (downSt.acyclic && upSt.acyclic);
		                }

		                std::vector<uint8_t> emittedAtomP;
	                if (atomStates && blk.spqraBehaviorAtomView.atoms_ptr)
	                {
	                    emittedAtomP.assign(blk.spqraBehaviorAtomView.atoms_len, 0);
	                    for (uint32_t ci = 0; ci < minView.components_len; ++ci)
	                    {
	                        const FfiSpqraMinimizerComponent &comp = minView.components_ptr[ci];
	                        const uint8_t parentType = scadKindToSpqrType(comp.kind);
	                        const uint32_t edgeEnd = std::min(comp.edge_end, minView.edges_len);
	                        for (uint32_t ge = comp.edge_begin; ge < edgeEnd; ++ge)
	                        {
	                            const FfiSpqraMinimizerEdge &me = minView.edges_ptr[ge];
	                            if ((me.flags & SPQRA_MIN_EDGE_HAS_BEHAVIOR_ATOM) == 0)
	                                continue;
	                            const uint32_t atomId = me.child_ref;
	                            if (atomId >= atomStates->fwd.size() ||
	                                atomId >= atomStates->outside.size() ||
	                                atomId >= blk.spqraBehaviorAtomView.atoms_len)
	                                continue;
	                            const EdgeDPState *inside =
	                                behaviorAtomStateForEdge(*atomStates, me, minView, blk, comp);
	                            if (!inside)
	                                continue;
	                            const EdgeDPState &outside = atomStates->outside[atomId];
	                            const uint8_t atomType =
	                                scadKindToSpqrType(blk.spqraBehaviorAtomView.atoms_ptr[atomId].kind);
		                            tryCoreTreeEdgeBubblesByTypes(atomType,
	                                                          parentType,
	                                                          *inside,
	                                                          outside,
	                                                          cc,
	                                                          blk);
		                            if (atomType == SPQR_NODE_TYPE_P &&
		                                atomId < emittedAtomP.size() &&
		                                !emittedAtomP[atomId])
		                            {
		                                emittedAtomP[atomId] = 1;
			                                const FfiSpqraBehaviorAtom &atom =
			                                    blk.spqraBehaviorAtomView.atoms_ptr[atomId];
				                                if (!blk.spqraBehaviorAtomView.items_ptr)
				                                {
			                                    continue;
			                                }
			                                if (!outside.s || !outside.t)
			                                {
			                                    continue;
			                                }
		                                PFixedAggEntry entry;
		                                node bS{inside->s.idx};
		                                node bT{inside->t.idx};
		                                const uint32_t itemBegin =
		                                    std::min(atom.item_begin,
		                                             blk.spqraBehaviorAtomView.items_len);
		                                const uint32_t itemEnd =
		                                    std::min(atom.item_end,
		                                             blk.spqraBehaviorAtomView.items_len);
		                                for (uint32_t ii = itemBegin; ii < itemEnd; ++ii)
	                                {
	                                    const FfiSpqraBehaviorAtomItem &item =
	                                        blk.spqraBehaviorAtomView.items_ptr[ii];
		                                    if (item.flags & SPQRA_MIN_ATOM_ITEM_CHILD_REF)
		                                    {
		                                        const uint32_t cref = item.child_ref;
		                                        if (SP_COMPRESS_CHILD_IS_EDGE(cref))
		                                        {
		                                            edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
		                                            node u = blockEdgeSource(blk, e);
		                                            node v = blockEdgeTarget(blk, e);
		                                            if (u == bS && v == bT)
		                                                ++entry.directST;
		                                            else
		                                                ++entry.directTS;
		                                        }
		                                        else
		                                        {
		                                            const uint32_t mid =
		                                                SP_COMPRESS_CHILD_AS_MACRO(cref);
			                                            if (mid >= macroStates.size() ||
			                                                mid >= view.macros_len)
			                                            {
			                                                continue;
			                                            }
		                                            const bool seriesLike =
		                                                view.macros_ptr[mid].kind ==
		                                                SP_COMPRESS_KIND_SERIES;
		                                            entry.fwd.addBranch(macroStates[mid],
		                                                                bS,
		                                                                bT,
		                                                                seriesLike,
		                                                                PFixedAggMode::Core);
		                                            entry.rev.addBranch(macroStates[mid],
		                                                                bT,
		                                                                bS,
		                                                                seriesLike,
		                                                                PFixedAggMode::Core);
		                                        }
		                                    }
		                                    else if (item.flags & SPQRA_MIN_ATOM_ITEM_BEHAVIOR_ATOM)
		                                    {
		                                        const EdgeDPState *nested =
		                                            nestedBehaviorAtomStateForItem(item,
		                                                                           blk.spqraBehaviorAtomView,
		                                                                           atomStates->fwd,
		                                                                           atomStates->rev,
		                                                                           blk,
		                                                                           atomId);
			                                        if (!nested)
			                                        {
			                                            continue;
			                                        }
		                                        const bool seriesLike =
		                                            item.child_ref < blk.spqraBehaviorAtomView.atoms_len &&
		                                            scadKindToSpqrType(blk.spqraBehaviorAtomView.atoms_ptr[item.child_ref].kind) ==
		                                                SPQR_NODE_TYPE_S;
		                                        entry.fwd.addBranch(*nested,
		                                                            bS,
		                                                            bT,
		                                                            seriesLike,
		                                                            PFixedAggMode::Core);
		                                        entry.rev.addBranch(*nested,
		                                                            bT,
		                                                            bS,
		                                                            seriesLike,
		                                                            PFixedAggMode::Core);
		                                    }
	                                }
	                                const bool outsideSeriesLike =
	                                    parentType == SPQR_NODE_TYPE_S;
	                                entry.fwd.addBranch(outside,
	                                                    bS,
	                                                    bT,
	                                                    outsideSeriesLike,
	                                                    PFixedAggMode::Core);
		                                entry.rev.addBranch(outside,
		                                                    bT,
		                                                    bS,
		                                                    outsideSeriesLike,
		                                                    PFixedAggMode::Core);
		                                entry.valid = 1;
		                                emitPFixedAggCandidate(entry.fwd,
	                                                       bS,
	                                                       bT,
	                                                       entry.directST,
	                                                       entry.directTS,
	                                                       true,
	                                                       cc,
	                                                       blk);
	                                emitPFixedAggCandidate(entry.rev,
	                                                       bT,
	                                                       bS,
	                                                       entry.directTS,
	                                                       entry.directST,
	                                                       true,
	                                                       cc,
	                                                       blk);
	                            }
	                        }
	                    }
	                }

	                if (atomStates && blk.spqraBehaviorAtomView.atoms_ptr)
	                {
		                    emitBehaviorAtomInternalCoreEdges(
		                        blk.spqraBehaviorAtomView,
		                        *atomStates,
		                        cc,
		                        blk,
		                        nullptr);
	                }

	                if (atomStates && blk.spqraBehaviorAtomView.atoms_ptr &&
	                    std::getenv("BF_SPQRA_NESTED_ATOM_OUTSIDE") != nullptr)
	                {
		                    for (uint32_t atomId = 0;
		                         atomId < blk.spqraBehaviorAtomView.atoms_len &&
		                         atomId < emittedAtomP.size();
		                         ++atomId)
		                    {
		                        if (emittedAtomP[atomId])
		                            continue;
		                        if (emitBehaviorAtomPFixedAgg(atomId,
		                                                      blk.spqraBehaviorAtomView,
		                                                      *atomStates,
		                                                      macroStates,
		                                                      view,
		                                                      cc,
		                                                      blk))
			                        {
				                            emittedAtomP[atomId] = 1;
				                        }
			                    }
		                }

			                for (uint32_t ci = 0; ci < minView.components_len; ++ci)
			                {
	                    const FfiSpqraMinimizerComponent &comp = minView.components_ptr[ci];
		                    if (scadKindToSpqrType(comp.kind) != SPQR_NODE_TYPE_P)
		                        continue;
			                    const uint32_t pole0 = minimizerLocalBlockId(minView, blk, comp, 0);
		                    const uint32_t pole1 = minimizerLocalBlockId(minView, blk, comp, 1);
		                    if (pole0 == SPQR_INVALID || pole1 == SPQR_INVALID)
		                    {
		                        continue;
		                    }

	                    PFixedAggEntry entry;
	                    node bS{pole0};
	                    node bT{pole1};
	                    const uint32_t edgeEnd = std::min(comp.edge_end, minView.edges_len);
	                    for (uint32_t ge = comp.edge_begin; ge < edgeEnd; ++ge)
	                    {
	                        const FfiSpqraMinimizerEdge &me = minView.edges_ptr[ge];
	                        if ((me.flags & SPQRA_MIN_EDGE_HAS_BEHAVIOR_ATOM) && atomStates)
	                        {
	                            const uint32_t atomId = me.child_ref;
		                            if (atomId >= atomStates->fwd.size() ||
		                                atomId >= blk.spqraBehaviorAtomView.atoms_len)
		                            {
		                                continue;
		                            }
	                            const EdgeDPState *state =
	                                behaviorAtomStateForEdge(*atomStates, me, minView, blk, comp);
		                            if (!state)
		                            {
		                                continue;
		                            }
	                            const bool seriesLike =
	                                scadKindToSpqrType(blk.spqraBehaviorAtomView.atoms_ptr[atomId].kind) ==
	                                SPQR_NODE_TYPE_S;
	                            entry.fwd.addBranch(*state, bS, bT, seriesLike,
	                                                PFixedAggMode::Core);
	                            entry.rev.addBranch(*state, bT, bS, seriesLike,
	                                                PFixedAggMode::Core);
	                            continue;
	                        }
		                        if (me.flags & SPQRA_MIN_EDGE_VIRTUAL)
		                        {
			                            if (me.twin_component >= minView.components_len)
		                            {
		                                continue;
		                            }
	                            const FfiSpqraMinimizerComponent &twComp =
	                                minView.components_ptr[me.twin_component];
	                            const uint32_t twGe =
	                                twComp.edge_begin + me.twin_local_edge;
		                            if (twGe >= minView.edges_len)
		                            {
		                                continue;
		                            }
	                            const EdgeDPState *state =
	                                minimizerViewEdgeMessage(me.twin_component,
	                                                         minView.edges_ptr[twGe],
	                                                         minView,
	                                                         down,
	                                                         up);
		                            if (!state)
		                            {
		                                continue;
		                            }
	                            const bool seriesLike =
	                                scadKindToSpqrType(twComp.kind) == SPQR_NODE_TYPE_S;
	                            entry.fwd.addBranch(*state, bS, bT, seriesLike,
	                                                PFixedAggMode::Core);
	                            entry.rev.addBranch(*state, bT, bS, seriesLike,
	                                                PFixedAggMode::Core);
	                            continue;
	                        }

		                        if ((me.flags & SPQRA_MIN_EDGE_HAS_CHILD_REF) == 0)
		                            continue;
		                        const uint32_t cref = me.child_ref;
	                        if (SP_COMPRESS_CHILD_IS_EDGE(cref))
	                        {
	                            edge e{SP_COMPRESS_CHILD_AS_EDGE(cref)};
	                            node u = blockEdgeSource(blk, e);
	                            node v = blockEdgeTarget(blk, e);
	                            if (u == bS && v == bT)
	                                ++entry.directST;
	                            else
	                                ++entry.directTS;
	                        }
	                        else
	                        {
	                            const uint32_t mid = SP_COMPRESS_CHILD_AS_MACRO(cref);
		                            if (mid >= macroStates.size() || mid >= view.macros_len)
		                            {
		                                continue;
		                            }
	                            const bool seriesLike =
	                                view.macros_ptr[mid].kind == SP_COMPRESS_KIND_SERIES;
	                            entry.fwd.addBranch(macroStates[mid], bS, bT, seriesLike,
	                                                PFixedAggMode::Core);
	                            entry.rev.addBranch(macroStates[mid], bT, bS, seriesLike,
	                                                PFixedAggMode::Core);
	                        }
	                    }
	                    entry.valid = 1;
	                    emitPFixedAggCandidate(entry.fwd,
	                                           bS,
	                                           bT,
	                                           entry.directST,
	                                           entry.directTS,
	                                           true,
	                                           cc,
	                                           blk);
	                    emitPFixedAggCandidate(entry.rev,
	                                           bT,
	                                           bS,
	                                           entry.directTS,
	                                           entry.directST,
	                                           true,
		                                           cc,
		                                           blk);
		                }
		                return true;
		            }

	            inline void collectAQAtomsLiveCoreDirect(const CoreContext &c,
	                                                     const SpCompressTreeView &view,
                                                     const std::vector<EdgeDPState> &macroStates,
                                                     const std::vector<EdgeDPState> &down,
	                                                     const std::vector<EdgeDPState> &up,
	                                                     const CcData &cc,
		                                                     BlockData &blk,
		                                                     const std::vector<PFixedAggEntry> *corePAgg = nullptr)
	            {
		                const bool coreEdgesAlreadyEmitted =
	                    std::getenv("BF_SPQRA_CORE_EXACT_EMIT_DURING_UP") != nullptr;

                for (uint32_t child = 0; child < c.len; ++child)
                {
                    if (child == c.root)
                        continue;
	                    uint32_t parent = c.parents[child];
	                    const EdgeDPState &downSt = down[child];
	                    const EdgeDPState &upSt = up[child];
	                    if (!coreEdgesAlreadyEmitted)
	                    {
	                        tryCoreTreeEdgeBubbles(child, parent, c, downSt, upSt, cc, blk);
                    }
	                    blk.isAcycic &= (downSt.acyclic && upSt.acyclic);
	                }

	                for (uint32_t tn = 0; tn < c.len; ++tn)
	                {
		                    if (c.types[tn] != SPQR_NODE_TYPE_P)
	                        continue;
	                    if (tryBubbleCorePNodeGroupingFixed(tn, c, corePAgg, cc, blk))
	                        continue;
	                    tryBubbleCorePNodeGrouping(tn, c, view, macroStates, down, up, cc, blk);
	                }
	            }

            inline void executeCorePortKernelFromScad(const CoreScadView &scad,
                                                      const CoreContext &c,
                                                      const SpCompressTreeView &view,
                                                      const std::vector<EdgeDPState> &macroStates,
                                                      const std::vector<EdgeDPState> &down,
                                                      const std::vector<EdgeDPState> &up,
                                                      const CcData &cc,
                                                      BlockData &blk,
	                                                      bool emitCoreEdges,
		                                                      bool emitCorePNodes,
		                                                      const std::vector<PFixedAggEntry> *corePAgg = nullptr)
	            {
	                if (emitCoreEdges && scad.incidences_ptr && scad.components_ptr)
                {
                    for (uint32_t ii = 0; ii < scad.incidences_len; ++ii)
                    {
                        uint32_t twin = SPQR_INVALID;
                        if (!scadValidTwinPair(scad, ii, twin) || ii > twin)
                            continue;
                        const uint32_t a = scad.incidences_ptr[ii].component_id;
                        const uint32_t b = scad.incidences_ptr[twin].component_id;
                        uint32_t child = SPQR_INVALID;
                        uint32_t parent = SPQR_INVALID;
                        if (a < c.len && b < c.len && a != c.root && c.parents[a] == b)
                        {
                            child = a;
                            parent = b;
                        }
                        else if (a < c.len && b < c.len && b != c.root && c.parents[b] == a)
                        {
                            child = b;
                            parent = a;
                        }
	                        else
	                        {
	                            continue;
	                        }
	                        tryCoreTreeEdgeBubbles(child, parent, c, down[child], up[child],
                                               cc, blk);
	                        blk.isAcycic &= (down[child].acyclic && up[child].acyclic);
	                    }
	                }

	                if (emitCorePNodes)
                {
                    for (uint32_t tn = 0; tn < c.len; ++tn)
                    {
	                        if (c.types[tn] != SPQR_NODE_TYPE_P)
	                            continue;
	                        if (tryBubbleCorePNodeGroupingFixed(tn, c, corePAgg, cc, blk))
                            continue;
	                        tryBubbleCorePNodeGrouping(tn, c, view, macroStates, down, up, cc, blk);
	                    }
	                }
	            }

            inline bool solveSPQRMacroDirect(BlockData &blk, const CcData &cc)
            {
	                if (!blk.spCompressHandle)
	                    return false;
	                const SpCompressTreeView &view = blk.macroTreeView;
                const bool aqAtoms = spqra_profile::aqAtomsRequested();
                const bool atomMacroPEnabled = aqAtoms && macroPAtomTracingDisabled();
                const bool portKernel = spqra_profile::portKernelRequested();
                const bool directCoreScad =
                    spqra_profile::aqDirectCoreScadRequested() ||
                    (aqAtoms && !spqra_profile::aqAtomsUseSpqrTreeFallback());
	                if (aqAtoms && atomMacroPEnabled &&
	                    minimizerMessageKernelRequested() &&
	                    !spqra_profile::aqContextRequested() &&
	                    blk.spqraMinimizerView.components_ptr &&
	                    blk.spqraMinimizerView.edges_ptr &&
	                    blk.spqraMinimizerView.node_mapping_ptr &&
	                    (minimizerViewCollectRequested() ||
		                     blk.coreScadView.incidences_ptr))
		                {
		                    populateMinimizerLocalBlockIds(blk);
		                    auto macroStates = computeMacroStates(view, blk, &cc, true);
		                    SpqraBehaviorAtomStates atomStates =
		                        computeBehaviorAtomStates(blk.spqraBehaviorAtomView,
	                                                  macroStates,
	                                                  blk);
	                    std::vector<NodeDPState> nodeDp(blk.spqraMinimizerView.components_len);
                    auto down = computeMinimizerDownStates(blk.spqraMinimizerView,
                                                           macroStates,
                                                           blk,
	                                                           nodeDp,
	                                                           &atomStates);
		                    std::vector<EdgeDPState> up(blk.spqraMinimizerView.components_len);
		                    computeMinimizerUpStates(blk.spqraMinimizerView,
			                                             macroStates,
			                                             blk,
		                                             down,
		                                             up,
		                                             nodeDp,
		                                             &atomStates);
		                    computeNestedBehaviorAtomOutsideStates(blk.spqraBehaviorAtomView,
		                                                           macroStates,
		                                                           blk,
		                                                           atomStates);
		                    if (minimizerViewCollectRequested())
		                    {
	                        collectAQAtomsFromMinimizerViewMessages(blk.spqraMinimizerView,
	                                                                view,
	                                                                macroStates,
	                                                                down,
	                                                                up,
	                                                                cc,
	                                                                blk,
	                                                                &atomStates);
	                    }
	                    else
	                    {
		                        collectAQAtomsFromMinimizerMessages(blk.coreScadView,
		                                                             blk.spqraMinimizerView,
		                                                             view,
		                                                             macroStates,
		                                                             down,
		                                                             up,
			                                                             cc,
			                                                             blk);
		                    }
		                    return true;
		                }
                if (!blk.coreSpqrTree &&
                    (view.stats.fully_sp_reducible != 0 || view.core_edges_len == 0))
                {
                    auto macroStates = computeMacroStates(view, blk,
                                                          atomMacroPEnabled ? &cc : nullptr,
                                                          atomMacroPEnabled);
                    if (!atomMacroPEnabled)
	                    {
	                        for (uint32_t mid = 0; mid < view.macros_len; ++mid)
	                            tryBubbleMacroPNodeGrouping(mid, view, macroStates, cc, blk);
	                    }
	                    return true;
	                }
	                CoreContext core;
                {
                    bool builtCore = false;
                    if (directCoreScad)
                        builtCore = buildCoreContextFromScadView(blk, core);
                    if (!builtCore)
                        builtCore = buildCoreContext(blk, core);
                    if (!builtCore)
                        return false;
                }
                if (view.macros_len == 0)
                    return false;
                if (!blk.coreSpqrTree && !directCoreScad)
                {
                    auto macroStates = computeMacroStates(view, blk,
                                                          atomMacroPEnabled ? &cc : nullptr,
                                                          atomMacroPEnabled);
                    if (!atomMacroPEnabled)
                    {
                        for (uint32_t mid = 0; mid < view.macros_len; ++mid)
                            tryBubbleMacroPNodeGrouping(mid, view, macroStates, cc, blk);
                    }
	                    return true;
	                }
	                auto macroStates = computeMacroStates(view, blk,
	                                                      atomMacroPEnabled ? &cc : nullptr,
	                                                      atomMacroPEnabled);
		                std::vector<NodeDPState> nodeDp(core.len);
		                auto down = computeCoreDownStates(core, view, macroStates, blk, nodeDp);
		                std::vector<EdgeDPState> up(core.len);
	                std::vector<PFixedAggEntry> corePAgg;
	                std::vector<PFixedAggEntry> *corePAggPtr =
	                    corePFixedAggRequested() ? &corePAgg : nullptr;
		                computeCoreUpStates(core, view, macroStates, blk, down, up, nodeDp,
		                                    &cc, corePAggPtr);
                MacroContext macroCtx;
                if (spqra_profile::aqContextRequested() || directCoreScad || aqAtoms)
                {
                    const bool coreEdgesAlreadyEmitted =
                        std::getenv("BF_SPQRA_CORE_EXACT_EMIT_DURING_UP") != nullptr;
                    if (portKernel && aqAtoms && atomMacroPEnabled && macroCtx.outside.empty() &&
                        !spqra_profile::aqContextRequested())
                    {
		                        executeCorePortKernelFromScad(blk.coreScadView, core, view,
		                                                      macroStates, down, up, cc, blk,
		                                                      !coreEdgesAlreadyEmitted,
		                                                      true,
		                                                      corePAggPtr);
                    }
	                    else if (aqAtoms && atomMacroPEnabled && macroCtx.outside.empty() &&
	                        !spqra_profile::aqContextRequested())
	                    {
		                        if (!collectAQAtomsFromMinimizerMessages(blk.coreScadView,
		                                                                  blk.spqraMinimizerView,
		                                                                  view,
		                                                                  macroStates,
		                                                                  down,
		                                                                  up,
		                                                                  cc,
		                                                                  blk))
		                        {
			                        collectAQAtomsLiveCoreDirect(core, view, macroStates, down, up, cc, blk,
			                                                     corePAggPtr);
		                        }
	                    }
                    else
                    {
                        const bool streamCoreEdgesFromPorts = portKernel && !coreEdgesAlreadyEmitted;
                        if (portKernel)
                        {
		                            executeCorePortKernelFromScad(blk.coreScadView, core, view,
		                                                          macroStates, down, up, cc, blk,
		                                                          streamCoreEdgesFromPorts,
		                                                          false,
		                                                          corePAggPtr);
	                        }
                        AQContext aq = buildAQContextFromSpqrTree(core, view, macroCtx,
                                                                  atomMacroPEnabled,
                                                                  streamCoreEdgesFromPorts);
		                        executeAQContextFromSpqrTree(aq, core, view, macroStates, macroCtx,
		                                                     down, up, cc, blk, corePAggPtr);
	                    }
	                }
	                else
	                {
		                    collectCoreSuperbubbles(core, view, macroStates, macroCtx, down, up,
		                                           cc, blk, corePAggPtr);
		                }
	                return true;
	            }



            void processEdge(spqr_compat::edge curr_edge, spqr_compat::EdgeArray<EdgeDP> &dp, NodeArray<NodeDPState> &node_dp, const CcData &cc, BlockData &blk)
            {
                // PROFILE_FUNCTION();

                BF_INSTR(
                profiling_patch::pe_total_calls.fetch_add(1, std::memory_order_relaxed);
                auto __pe_tA_start = std::chrono::high_resolution_clock::now();
                )

                const spqr_compat::NodeArray<int> &globIn = blk.globIn;
                const spqr_compat::NodeArray<int> &globOut = blk.globOut;

                EdgeDPState &state = dp[curr_edge].down;
                EdgeDPState &back_state = dp[curr_edge].up;

                const StaticSPQRTree &spqr = *blk.spqr;
                const TreeGraph &T = spqr.tree();

                spqr_compat::node A = parentOnTreeEdge(T, blk, curr_edge);
                spqr_compat::node B = childOnTreeEdge(T, blk, curr_edge);

                state.localOutS = 0;
                state.localInT = 0;
                state.localOutT = 0;
                state.localInS = 0;

                const Skeleton &skel = spqr.skeleton(B);
                const uint32_t pe_nSkel = skel.numberOfNodes();
                const uint32_t pe_nSkelEdges = skel.numberOfEdges();

                thread_local std::vector<uint32_t> tls_pe_skelToNew;
                thread_local std::vector<int> tls_pe_localInDeg;
                thread_local std::vector<int> tls_pe_localOutDeg;
                thread_local FlatDigraphScratch tls_pe_graph;

                if (tls_pe_skelToNew.size() < pe_nSkel) tls_pe_skelToNew.resize(pe_nSkel);
                if (tls_pe_localInDeg.size() < pe_nSkel) tls_pe_localInDeg.resize(pe_nSkel);
                if (tls_pe_localOutDeg.size() < pe_nSkel) tls_pe_localOutDeg.resize(pe_nSkel);

                std::fill_n(tls_pe_localInDeg.begin(), pe_nSkel, 0);
                std::fill_n(tls_pe_localOutDeg.begin(), pe_nSkel, 0);
                tls_pe_graph.reset(pe_nSkel, pe_nSkelEdges);

                for (uint32_t i = 0; i < pe_nSkel; ++i) {
                    tls_pe_skelToNew[i] = i;
                }

                for (uint32_t i = 0; i < pe_nSkel; ++i) {
                    spqr_compat::node h{i};
                    spqr_compat::node vB = skel.original(h);
                    blk.blkToSkel[vB] = h;
                }

                // auto mapGlobalToNew = [&](spqr_compat::node vG) -> spqr_compat::node {
                //     // global -> component
                //     spqr_compat::node vComp = cc.toCopy[vG];
                //     if (!vComp) return nullptr;

                //     // component -> block
                //     spqr_compat::node vBlk  = cc.toBlk[vComp];
                //     if (!vBlk)  return nullptr;

                //     // block -> skeleton
                //     spqr_compat::node vSkel = blk.blkToSkel[vBlk];
                //     if (!vSkel) return nullptr;

                //     return skelToNew[vSkel];
                // };


                // auto mapBlkToNew = [&](spqr_compat::node bV) -> spqr_compat::node {
                //     if (!bV) return nullptr;

                //     spqr_compat::node vSkel = newToSkel[vN];
                //     if (!vSkel) return nullptr;

                //     spqr_compat::node vBlk  = skel.original(vSkel);
                //     if (!vBlk) return nullptr;

                //     spqr_compat::node vCc   = blk.toCc[vBlk];
                //     if (!vCc) return nullptr;

                //     return cc.toOrig[vCc];
                // };


                spqr_compat::node nS{nullptr}, nT{nullptr};

                {
                    BF_INSTR(
                    auto __pe_tB_start = std::chrono::high_resolution_clock::now();
                    profiling_patch::pe_A_setup_ns.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            __pe_tB_start - __pe_tA_start).count(),
                        std::memory_order_relaxed);
                    )
                }
                BF_INSTR(auto __pe_tB_start = std::chrono::high_resolution_clock::now();)

                skel.forEachEdge([&](edge e, node u, node v)
                {
                    node nU{tls_pe_skelToNew[u.idx]};
                    node nV{tls_pe_skelToNew[v.idx]};

                    if (!skel.isVirtual(e))
                    {
                        tls_pe_graph.addEdge(nU, nV);
                        tls_pe_localOutDeg[nU.idx]++;
                        tls_pe_localInDeg[nV.idx]++;

                        return;
                    }

                    auto D = skel.twinTreeNode(e);

                    if (D == A)
                    {
                        spqr_compat::node vBlk = skel.original(v);
                        spqr_compat::node uBlk = skel.original(u);

                        // spqr_compat::node vG  = blk.toOrig[vCc];
                        // spqr_compat::node uG  = blk.toOrig[uCc];

                        state.s = back_state.s = vBlk;
                        state.t = back_state.t = uBlk;

                        nS = nV;
                        nT = nU;

                        return;
                    }

                    edge treeE = blk.skel2tree.at(e);
                    SPQR_RUST_ASSERT(treeE != nullptr);

                    const EdgeDPState child = dp[treeE].down;
                    int dir = child.getDirection();

                    // spqr_compat::node nS = mapGlobalToNew(child.s);
                    // spqr_compat::node nT = mapGlobalToNew(child.t);

                    spqr_compat::node nA{tls_pe_skelToNew[blk.blkToSkel[child.s].idx]};
                    spqr_compat::node nB{tls_pe_skelToNew[blk.blkToSkel[child.t].idx]};

                    if (dir == 1)
                    {
                        tls_pe_graph.addEdge(nA, nB);
                    }
                    else if (dir == -1)
                    {
                        tls_pe_graph.addEdge(nB, nA);
                    }

                    if (nA == nU && nB == nV)
                    {
                        tls_pe_localOutDeg[nA.idx] += child.localOutS;
                        tls_pe_localInDeg[nA.idx] += child.localInS;

                        tls_pe_localOutDeg[nB.idx] += child.localOutT;
                        tls_pe_localInDeg[nB.idx] += child.localInT;
                    }
                    else
                    {
                        tls_pe_localOutDeg[nB.idx] += child.localOutT;
                        tls_pe_localInDeg[nB.idx] += child.localInT;

                        tls_pe_localOutDeg[nA.idx] += child.localOutS;
                        tls_pe_localInDeg[nA.idx] += child.localInS;
                    }

                    state.acyclic &= child.acyclic;
                    state.globalSourceSink |= child.globalSourceSink;
                    state.hasLeakage |= child.hasLeakage;
                });

                BF_INSTR(
                auto __pe_tC_start = std::chrono::high_resolution_clock::now();
                profiling_patch::pe_B_build_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        __pe_tC_start - __pe_tB_start).count(),
                    std::memory_order_relaxed);
                )

                // Direct ST/TS computation(only happens in P nodes)
                if (spqr.typeOf(B) == SPQRTree::NodeType::PNode)
                {
                    skel.forEachEdge([&](edge e, node u, node v)
                    {
                        if (skel.isVirtual(e))
                            return;

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
                    });
                }

                // for (spqr_compat::node vN : newGraph.nodes) {
                //     spqr_compat::node vG  = mapNewToGlobal(vN);
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

                BF_INSTR(
                auto __pe_tD_start = std::chrono::high_resolution_clock::now();
                profiling_patch::pe_C_pnode_extra_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        __pe_tD_start - __pe_tC_start).count(),
                    std::memory_order_relaxed);
                )

                for (uint32_t nV = 0; nV < pe_nSkel; ++nV)
                {
                    spqr_compat::node bV = skel.original(spqr_compat::node{nV});

                    if (bV == state.s || bV == state.t)
                        continue;

                    if (globIn[bV] != tls_pe_localInDeg[nV] ||
                        globOut[bV] != tls_pe_localOutDeg[nV])
                    {
                        state.hasLeakage = true;
                    }

                    if (globIn[bV] == 0 || globOut[bV] == 0)
                    {
                        state.globalSourceSink = true;
                    }
                }

                BF_INSTR(
                auto __pe_tE_start = std::chrono::high_resolution_clock::now();
                profiling_patch::pe_D_leakage_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        __pe_tE_start - __pe_tD_start).count(),
                    std::memory_order_relaxed);
                )

                // state.localInS = localInDeg[mapGlobalToNew(state.s)];
                // state.localOutS = localOutDeg[mapGlobalToNew(state.s)];

                // state.localInT = localInDeg[mapGlobalToNew(state.t)];
                // state.localOutT = localOutDeg[mapGlobalToNew(state.t)];

                SPQR_RUST_ASSERT(nS != nullptr && nT != nullptr);
                state.localInS = tls_pe_localInDeg[nS.idx];
                state.localOutS = tls_pe_localOutDeg[nS.idx];

                state.localInT = tls_pe_localInDeg[nT.idx];
                state.localOutT = tls_pe_localOutDeg[nT.idx];

                if (state.acyclic)
                {
                    if (ctx().weakSuperbubbles)
                    {
                        int dir = state.getDirection();
                        if (dir == 1)
                            state.acyclic &= tls_pe_graph.isAcyclic(FlatDigraphScratch::invalid, nT.idx, nS.idx);
                        else if (dir == -1)
                            state.acyclic &= tls_pe_graph.isAcyclic(FlatDigraphScratch::invalid, nS.idx, nT.idx);
                        else
                            state.acyclic &= tls_pe_graph.isAcyclic();
                    }
                    else
                    {
                        state.acyclic &= tls_pe_graph.isAcyclic();
                    }
                }

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

                BF_INSTR(
                auto __pe_tE_end = std::chrono::high_resolution_clock::now();
                profiling_patch::pe_E_acyclic_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        __pe_tE_end - __pe_tE_start).count(),
                    std::memory_order_relaxed);
                )
            }

            void processNode(node curr_node, EdgeArray<EdgeDP> &edge_dp, NodeArray<NodeDPState> &node_dp, const CcData &cc, BlockData &blk)
            {

                BF_INSTR(
                profiling_patch::pn_total_calls.fetch_add(1, std::memory_order_relaxed);
                auto __pn_tA_start = std::chrono::high_resolution_clock::now();
                )

                const spqr_compat::NodeArray<int> &globIn = blk.globIn;
                const spqr_compat::NodeArray<int> &globOut = blk.globOut;

                spqr_compat::node A = curr_node;
                NodeDPState curr_state = node_dp[A];

                const StaticSPQRTree &spqr = *blk.spqr;
                const Skeleton &skel = spqr.skeleton(A);
                const uint32_t nSkel = skel.numberOfNodes();
                const uint32_t nSkelEdges = skel.numberOfEdges();

                thread_local std::vector<uint32_t> tls_pn_skelToNew;
                thread_local std::vector<int> tls_pn_localInDeg;
                thread_local std::vector<int> tls_pn_localOutDeg;
                thread_local std::vector<char> tls_pn_isSrcSink;
                thread_local std::vector<char> tls_pn_isLeaking;
                thread_local std::vector<SPQRsolve::EdgeDPState*> tls_pn_edgeToDp;
                thread_local std::vector<SPQRsolve::EdgeDPState*> tls_pn_edgeToDpR;
                thread_local std::vector<spqr_compat::node> tls_pn_edgeChild;
                thread_local std::vector<spqr_compat::edge> tls_pn_virtualEdges;
                thread_local FlatDigraphScratch tls_pn_graph;

                if (tls_pn_skelToNew.size() < nSkel) tls_pn_skelToNew.resize(nSkel);
                if (tls_pn_localInDeg.size() < nSkel) tls_pn_localInDeg.resize(nSkel);
                if (tls_pn_localOutDeg.size() < nSkel) tls_pn_localOutDeg.resize(nSkel);
                if (tls_pn_isSrcSink.size() < nSkel) tls_pn_isSrcSink.resize(nSkel);
                if (tls_pn_isLeaking.size() < nSkel) tls_pn_isLeaking.resize(nSkel);
                if (tls_pn_edgeToDp.size() < nSkelEdges) tls_pn_edgeToDp.resize(nSkelEdges);
                if (tls_pn_edgeToDpR.size() < nSkelEdges) tls_pn_edgeToDpR.resize(nSkelEdges);
                if (tls_pn_edgeChild.size() < nSkelEdges) tls_pn_edgeChild.resize(nSkelEdges);

                std::fill_n(tls_pn_localInDeg.begin(), nSkel, 0);
                std::fill_n(tls_pn_localOutDeg.begin(), nSkel, 0);
                std::fill_n(tls_pn_isSrcSink.begin(), nSkel, (char)0);
                std::fill_n(tls_pn_isLeaking.begin(), nSkel, (char)0);
                std::fill_n(tls_pn_edgeToDp.begin(), nSkelEdges, (SPQRsolve::EdgeDPState*)nullptr);
                std::fill_n(tls_pn_edgeToDpR.begin(), nSkelEdges, (SPQRsolve::EdgeDPState*)nullptr);
                std::fill_n(tls_pn_edgeChild.begin(), nSkelEdges, spqr_compat::node{});
                tls_pn_virtualEdges.clear();
                tls_pn_graph.reset(nSkel, nSkelEdges);

                for (uint32_t i = 0; i < nSkel; ++i) {
                    tls_pn_skelToNew[i] = i;
                }

                for (uint32_t i = 0; i < nSkel; ++i) {
                    spqr_compat::node h{i};
                    spqr_compat::node vB = skel.original(h);
                    blk.blkToSkel[vB] = h;
                }

                int localSourceSinkCount = 0;
                int localLeakageCount = 0;

                auto mapBlockToNew = [&](spqr_compat::node bV) -> spqr_compat::node {
                    spqr_compat::node sV = blk.blkToSkel[bV];
                    return spqr_compat::node{tls_pn_skelToNew[sV.idx]};
                };

                BF_INSTR(
                auto __pn_tA_end = std::chrono::high_resolution_clock::now();
                profiling_patch::pn_A_setup_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__pn_tA_end - __pn_tA_start).count(),
                    std::memory_order_relaxed);
                auto __pn_tB_start = __pn_tA_end;
                )

                skel.forEachEdge([&](edge e, node u, node v) {
                    spqr_compat::node nU{tls_pn_skelToNew[u.idx]};
                    spqr_compat::node nV{tls_pn_skelToNew[v.idx]};

                    if (!skel.isVirtual(e)) {
                        tls_pn_graph.addEdge(nU, nV);
                        tls_pn_localOutDeg[nU.idx]++;
                        tls_pn_localInDeg[nV.idx]++;
                        return;
                    }

                    auto B = skel.twinTreeNode(e);
                    edge treeE = blk.skel2tree.at(e);
                    SPQR_RUST_ASSERT(treeE != nullptr);

                    SPQRsolve::EdgeDPState *child =
                        (B == blk.parent(A) ? &edge_dp[treeE].up : &edge_dp[treeE].down);
                    SPQRsolve::EdgeDPState *edgeToUpdate =
                        (B == blk.parent(A) ? &edge_dp[treeE].down : &edge_dp[treeE].up);
                    int dir = child->getDirection();

                    spqr_compat::node nS = mapBlockToNew(child->s);
                    spqr_compat::node nT = mapBlockToNew(child->t);

                    edge newEdge;
                    if (dir == 1 || dir == 0) {
                        newEdge = tls_pn_graph.addEdge(nS, nT);
                    } else /* dir == -1 */ {
                        newEdge = tls_pn_graph.addEdge(nT, nS);
                    }
                    tls_pn_edgeToDp[newEdge.idx] = edgeToUpdate;
                    tls_pn_edgeToDpR[newEdge.idx] = child;
                    tls_pn_edgeChild[newEdge.idx] = B;
                    tls_pn_virtualEdges.push_back(newEdge);

                    if (nS == nU && nT == nV) {
                        tls_pn_localOutDeg[nS.idx] += child->localOutS;
                        tls_pn_localInDeg[nS.idx] += child->localInS;
                        tls_pn_localOutDeg[nT.idx] += child->localOutT;
                        tls_pn_localInDeg[nT.idx] += child->localInT;
                    } else {
                        tls_pn_localOutDeg[nT.idx] += child->localOutT;
                        tls_pn_localInDeg[nT.idx] += child->localInT;
                        tls_pn_localOutDeg[nS.idx] += child->localOutS;
                        tls_pn_localInDeg[nS.idx] += child->localInS;
                    }
                });

                BF_INSTR(
                auto __pn_tB_end = std::chrono::high_resolution_clock::now();
                profiling_patch::pn_B_build_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__pn_tB_end - __pn_tB_start).count(),
                    std::memory_order_relaxed);
                auto __pn_tC_start = __pn_tB_end;
                )
                if (tls_pn_virtualEdges.empty()) {
                    BF_INSTR(profiling_patch::pn_fastpath_calls.fetch_add(1, std::memory_order_relaxed);)
                    return;
                }

                for (uint32_t vN = 0; vN < nSkel; ++vN) {
                    node vB = skel.original(spqr_compat::node{vN});
                    if (globIn[vB] == 0 || globOut[vB] == 0) {
                        localSourceSinkCount++;
                        tls_pn_isSrcSink[vN] = (char)1;
                    }
                    if (globIn[vB] != tls_pn_localInDeg[vN] ||
                        globOut[vB] != tls_pn_localOutDeg[vN])
                    {
                        localLeakageCount++;
                        tls_pn_isLeaking[vN] = (char)1;
                    }
                }

                BF_INSTR(
                auto __pn_tC_end = std::chrono::high_resolution_clock::now();
                profiling_patch::pn_C_mark_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__pn_tC_end - __pn_tC_start).count(),
                    std::memory_order_relaxed);
                auto __pn_tD_start = __pn_tC_end;
                )

                // Same semantics and early-return as the original.
                if (spqr.typeOf(A) == StaticSPQRTree::NodeType::PNode)
                {
                    node pole0Blk = nullptr, pole1Blk = nullptr;
                    if (nSkel >= 1) pole0Blk = skel.original(spqr_compat::node{0u});
                    if (nSkel >= 2) pole1Blk = skel.original(spqr_compat::node{1u});

                    if (!pole0Blk || !pole1Blk) {
                        BF_INSTR(
                        auto __pn_t_early = std::chrono::high_resolution_clock::now();
                        profiling_patch::pn_D_pnode_ns.fetch_add(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(__pn_t_early - __pn_tD_start).count(),
                            std::memory_order_relaxed);
                        profiling_patch::pn_pnode_early_returns.fetch_add(1, std::memory_order_relaxed);
                        )
                        return;
                    }

                    int cnt01 = 0, cnt10 = 0;
                    skel.forEachEdge([&](edge e, node u, node v) {
                        if (!skel.isVirtual(e)) {
                            node bU = skel.original(u);
                            node bV = skel.original(v);
                            if      (bU == pole0Blk && bV == pole1Blk) ++cnt01;
                            else if (bU == pole1Blk && bV == pole0Blk) ++cnt10;
                        }
                    });

                    skel.forEachEdge([&](edge e, node, node) {
                        if (skel.isVirtual(e)) {
                            node B = skel.twinTreeNode(e);
                            edge treeE = blk.skel2tree.at(e);

                            SPQRsolve::EdgeDPState &st =
                                (B == blk.parent(A) ? edge_dp[treeE].down
                                                    : edge_dp[treeE].up);

                            if (st.s == pole0Blk && st.t == pole1Blk) {
                                st.directST |= (cnt01 > 0);
                                st.directTS |= (cnt10 > 0);
                            } else if (st.s == pole1Blk && st.t == pole0Blk) {
                                st.directST |= (cnt10 > 0);
                                st.directTS |= (cnt01 > 0);
                            }
                        }
                    });
                }

                BF_INSTR(
                auto __pn_tD_end = std::chrono::high_resolution_clock::now();
                profiling_patch::pn_D_pnode_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__pn_tD_end - __pn_tD_start).count(),
                    std::memory_order_relaxed);
                auto __pn_tE_start = __pn_tD_end;
                )

                if (curr_state.outgoingCyclesCount >= 2)
                {
                    BF_INSTR(profiling_patch::pn_E1_calls.fetch_add(1, std::memory_order_relaxed);)
                    for (edge e : tls_pn_virtualEdges) {
                        if (tls_pn_edgeToDp[e.idx]->acyclic) {
                            node_dp[tls_pn_edgeChild[e.idx]].outgoingCyclesCount++;
                            node_dp[tls_pn_edgeChild[e.idx]].lastCycleNode = curr_node;
                        }
                        tls_pn_edgeToDp[e.idx]->acyclic &= false;
                    }
                }
                else if (node_dp[curr_node].outgoingCyclesCount == 1)
                {
                    BF_INSTR(profiling_patch::pn_E2_calls.fetch_add(1, std::memory_order_relaxed);)
                    for (edge e : tls_pn_virtualEdges) {
                        if (tls_pn_edgeChild[e.idx] != curr_state.lastCycleNode) {
                            if (tls_pn_edgeToDp[e.idx]->acyclic) {
                                node_dp[tls_pn_edgeChild[e.idx]].outgoingCyclesCount++;
                                node_dp[tls_pn_edgeChild[e.idx]].lastCycleNode = curr_node;
                            }
                            tls_pn_edgeToDp[e.idx]->acyclic &= false;
                        } else {
                            const bool acyclic = tls_pn_graph.isAcyclic(e.idx);

                            if (tls_pn_edgeToDp[e.idx]->acyclic && !acyclic) {
                                node_dp[tls_pn_edgeChild[e.idx]].outgoingCyclesCount++;
                                node_dp[tls_pn_edgeChild[e.idx]].lastCycleNode = curr_node;
                            }
                            tls_pn_edgeToDp[e.idx]->acyclic &= acyclic;
                        }
                    }
                }
                else
                {
                    BF_INSTR(profiling_patch::pn_E3_calls.fetch_add(1, std::memory_order_relaxed);)

                    thread_local std::vector<edge> tls_pn_fasResult;
                    tls_pn_fasResult.clear();
                    const bool __pn_isAcyclic = tls_pn_graph.isAcyclic();

                    if (__pn_isAcyclic) {
                        BF_INSTR(
                        profiling_patch::pn_E3_fas_dag_skipped.fetch_add(
                            1, std::memory_order_relaxed);
                        )
                    } else {
                        Graph newGraph;
                        tls_pn_graph.materialize(newGraph);
                        FeedbackArcSet FAS(newGraph);
                        const bool __pn_fasAcyclic = FAS.run_or_acyclic(tls_pn_fasResult);
                        if (__pn_fasAcyclic) {
                            BF_INSTR(
                            profiling_patch::pn_E3_fas_dag_skipped.fetch_add(
                                1, std::memory_order_relaxed);
                            )
                        } else {
                            thread_local std::vector<char> tls_pn_isFas;
                            const uint32_t nNewEdges = tls_pn_graph.numberOfEdges();
                            if (tls_pn_isFas.size() < nNewEdges) tls_pn_isFas.resize(nNewEdges);
                            std::fill_n(tls_pn_isFas.begin(), nNewEdges, (char)0);
                            for (edge e : tls_pn_fasResult) tls_pn_isFas[e.idx] = (char)1;

                            for (edge e : tls_pn_virtualEdges) {
                                if (tls_pn_edgeToDp[e.idx]->acyclic && !tls_pn_isFas[e.idx]) {
                                    node_dp[tls_pn_edgeChild[e.idx]].outgoingCyclesCount++;
                                    node_dp[tls_pn_edgeChild[e.idx]].lastCycleNode = curr_node;
                                }
                                tls_pn_edgeToDp[e.idx]->acyclic &= (bool)tls_pn_isFas[e.idx];
                            }
                        }
                    }
                }

                BF_INSTR(auto __pn_tE_end = std::chrono::high_resolution_clock::now();)
                {
                    BF_INSTR(
                    auto __pn_dE = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        __pn_tE_end - __pn_tE_start).count();
                    )
                    BF_INSTR(
                    if (curr_state.outgoingCyclesCount >= 2)
                        profiling_patch::pn_E1_branch_ns.fetch_add(__pn_dE, std::memory_order_relaxed);
                    else if (curr_state.outgoingCyclesCount == 1)
                        profiling_patch::pn_E2_branch_ns.fetch_add(__pn_dE, std::memory_order_relaxed);
                    else
                        profiling_patch::pn_E3_branch_ns.fetch_add(__pn_dE, std::memory_order_relaxed);
                    )
                }
                BF_INSTR(auto __pn_tF_start = __pn_tE_end;)

                if (curr_state.outgoingSourceSinkCount >= 2) {
                    for (edge e : tls_pn_virtualEdges) {
                        if (!tls_pn_edgeToDp[e.idx]->globalSourceSink) {
                            node_dp[tls_pn_edgeChild[e.idx]].outgoingSourceSinkCount++;
                            node_dp[tls_pn_edgeChild[e.idx]].lastSourceSinkNode = curr_node;
                        }
                        tls_pn_edgeToDp[e.idx]->globalSourceSink |= true;
                    }
                } else if (curr_state.outgoingSourceSinkCount == 1) {
                    for (edge e : tls_pn_virtualEdges) {
                        if (tls_pn_edgeChild[e.idx] != curr_state.lastSourceSinkNode) {
                            if (!tls_pn_edgeToDp[e.idx]->globalSourceSink) {
                                node_dp[tls_pn_edgeChild[e.idx]].outgoingSourceSinkCount++;
                                node_dp[tls_pn_edgeChild[e.idx]].lastSourceSinkNode = curr_node;
                            }
                            tls_pn_edgeToDp[e.idx]->globalSourceSink |= true;
                        } else {
                            node vN{tls_pn_graph.src[e.idx]}, uN{tls_pn_graph.dst[e.idx]};
                            if ((int)tls_pn_isSrcSink[vN.idx] + (int)tls_pn_isSrcSink[uN.idx] < localSourceSinkCount) {
                                if (!tls_pn_edgeToDp[e.idx]->globalSourceSink) {
                                    node_dp[tls_pn_edgeChild[e.idx]].outgoingSourceSinkCount++;
                                    node_dp[tls_pn_edgeChild[e.idx]].lastSourceSinkNode = curr_node;
                                }
                                tls_pn_edgeToDp[e.idx]->globalSourceSink |= true;
                            }
                        }
                    }
                } else {
                    for (edge e : tls_pn_virtualEdges) {
                        node vN{tls_pn_graph.src[e.idx]}, uN{tls_pn_graph.dst[e.idx]};
                        if ((int)tls_pn_isSrcSink[vN.idx] + (int)tls_pn_isSrcSink[uN.idx] < localSourceSinkCount) {
                            if (!tls_pn_edgeToDp[e.idx]->globalSourceSink) {
                                node_dp[tls_pn_edgeChild[e.idx]].outgoingSourceSinkCount++;
                                node_dp[tls_pn_edgeChild[e.idx]].lastSourceSinkNode = curr_node;
                            }
                            tls_pn_edgeToDp[e.idx]->globalSourceSink |= true;
                        }
                    }
                }

                BF_INSTR(
                auto __pn_tF_end = std::chrono::high_resolution_clock::now();
                profiling_patch::pn_F_gss_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__pn_tF_end - __pn_tF_start).count(),
                    std::memory_order_relaxed);
                auto __pn_tG_start = __pn_tF_end;
                )

                if (curr_state.outgoingLeakageCount >= 2) {
                    for (edge e : tls_pn_virtualEdges) {
                        if (!tls_pn_edgeToDp[e.idx]->hasLeakage) {
                            node_dp[tls_pn_edgeChild[e.idx]].outgoingLeakageCount++;
                            node_dp[tls_pn_edgeChild[e.idx]].lastLeakageNode = curr_node;
                        }
                        tls_pn_edgeToDp[e.idx]->hasLeakage |= true;
                    }
                } else if (curr_state.outgoingLeakageCount == 1) {
                    for (edge e : tls_pn_virtualEdges) {
                        if (tls_pn_edgeChild[e.idx] != curr_state.lastLeakageNode) {
                            if (!tls_pn_edgeToDp[e.idx]->hasLeakage) {
                                node_dp[tls_pn_edgeChild[e.idx]].outgoingLeakageCount++;
                                node_dp[tls_pn_edgeChild[e.idx]].lastLeakageNode = curr_node;
                            }
                            tls_pn_edgeToDp[e.idx]->hasLeakage |= true;
                        } else {
                            node vN{tls_pn_graph.src[e.idx]}, uN{tls_pn_graph.dst[e.idx]};
                            if ((int)tls_pn_isLeaking[vN.idx] + (int)tls_pn_isLeaking[uN.idx] < localLeakageCount) {
                                if (!tls_pn_edgeToDp[e.idx]->hasLeakage) {
                                    node_dp[tls_pn_edgeChild[e.idx]].outgoingLeakageCount++;
                                    node_dp[tls_pn_edgeChild[e.idx]].lastLeakageNode = curr_node;
                                }
                                tls_pn_edgeToDp[e.idx]->hasLeakage |= true;
                            }
                        }
                    }
                } else {
                    for (edge e : tls_pn_virtualEdges) {
                        node vN{tls_pn_graph.src[e.idx]}, uN{tls_pn_graph.dst[e.idx]};
                        if ((int)tls_pn_isLeaking[vN.idx] + (int)tls_pn_isLeaking[uN.idx] < localLeakageCount) {
                            if (!tls_pn_edgeToDp[e.idx]->hasLeakage) {
                                node_dp[tls_pn_edgeChild[e.idx]].outgoingLeakageCount++;
                                node_dp[tls_pn_edgeChild[e.idx]].lastLeakageNode = curr_node;
                            }
                            tls_pn_edgeToDp[e.idx]->hasLeakage |= true;
                        }
                    }
                }

                BF_INSTR(
                auto __pn_tG_end = std::chrono::high_resolution_clock::now();
                profiling_patch::pn_G_leak_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__pn_tG_end - __pn_tG_start).count(),
                    std::memory_order_relaxed);
                auto __pn_tH_start = __pn_tG_end;
                )

                for (edge e : tls_pn_virtualEdges) {
                    SPQRsolve::EdgeDPState *BA = tls_pn_edgeToDp[e.idx];
                    SPQRsolve::EdgeDPState *AB = tls_pn_edgeToDpR[e.idx];

                    BA->localInS = tls_pn_localInDeg[mapBlockToNew(BA->s).idx] - AB->localInS;
                    BA->localInT = tls_pn_localInDeg[mapBlockToNew(BA->t).idx] - AB->localInT;
                    BA->localOutS = tls_pn_localOutDeg[mapBlockToNew(BA->s).idx] - AB->localOutS;
                    BA->localOutT = tls_pn_localOutDeg[mapBlockToNew(BA->t).idx] - AB->localOutT;
                }
                BF_INSTR(
                auto __pn_tH_end = std::chrono::high_resolution_clock::now();
                profiling_patch::pn_H_poles_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__pn_tH_end - __pn_tH_start).count(),
                    std::memory_order_relaxed);
                )
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

                for (int q = 0; q < 2; q++)
                {
                    std::vector<const EdgeDPState *> goodS, goodT;
                    node singleGoodS = nullptr;

                    int localOutSSum = directST, localInTSum = directST;

                    T.forEachAdj(A, [&](node other, edge e) {
                        const auto &state = stateLeavingNode(edge_dp, T, blk, A, e);

                        int localOutS = (state.s == bS ? state.localOutS : state.localOutT), localInT = (state.t == bT ? state.localInT : state.localInS);

                        localOutSSum += localOutS;
                        localInTSum += localInT;
                        if (localOutS > 0)
                        {
                            goodS.push_back(&state);
                            singleGoodS = other;
                        }

                        if (localInT > 0)
                        {
                            goodT.push_back(&state);
                        }
                    });

                    bool good = true;
                    for (auto &state : goodS)
                    {
                        if ((state->s == bS && state->localInS > 0) || (state->s == bT && state->localInT > 0))
                        {
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
                            good = false;
                        }

                        good &= state->acyclic;
                        good &= !state->globalSourceSink;
                        good &= !state->hasLeakage;
                    }

                    good &= (ctx().weakSuperbubbles || directTS == 0);
                    good &= goodS == goodT;
                    good &= goodS.size() > 0;
                    good &= (goodS.size() != 1 || blk.spqr->typeOf(singleGoodS) != SPQRTree::NodeType::SNode || directST > 0);

                    good &= (localOutSSum == blk.globOut[bS] && localInTSum == blk.globIn[bT]);

                    if (good)
                    {
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
                if (!curr.s || !curr.t || !back.s || !back.t)
                    return;

                node S = swap ? blk.toOrig[curr.t] : blk.toOrig[curr.s];
                node T = swap ? blk.toOrig[curr.s] : blk.toOrig[curr.t];

                int outS = swap ? curr.localOutT : curr.localOutS;
                int outT = swap ? curr.localOutS : curr.localOutT;
                int inS = swap ? curr.localInT : curr.localInS;
                int inT = swap ? curr.localInS : curr.localInT;

                if (back.directST)
                {
                    if (!swap)
                    {
                        outS++;
                        inT++;
                    }
                    else if (!ctx().weakSuperbubbles)
                    {
                        inS++;
                        outT++;
                    }
                }
                if (back.directTS)
                {
                    if (!swap)
                    {
                        if (!ctx().weakSuperbubbles)
                        {
                            inS++;
                            outT++;
                        }
                    }
                    else
                    {
                        outS++;
                        inT++;
                    }
                }

                bool backGood = true;

                if (!ctx().weakSuperbubbles && back.s == curr.s && back.t == curr.t)
                {
                    backGood &= (!back.directTS);
                }
                else if (!ctx().weakSuperbubbles && back.s == curr.t && back.t == curr.s)
                {
                    backGood &= (!back.directST);
                }

                bool acyclic = curr.acyclic;
                bool noLeakage = !curr.hasLeakage;
                bool noGSource = !curr.globalSourceSink;
                const int globOutS = swap ? blk.globOut[curr.t] : blk.globOut[curr.s];
                const int globInT = swap ? blk.globIn[curr.s] : blk.globIn[curr.t];

                bool endpointCountsGood = (globOutS == outS && globInT == inT);

                bool pass =
                    acyclic &&
                    noGSource &&
                    noLeakage &&
                    backGood &&
                    outS > 0 &&
                    inT > 0 &&
                    inS == 0 &&
                    outT == 0 &&
                    endpointCountsGood &&
                    !ctx().isEntry[S] &&
                    !ctx().isExit[T];
                bool riskyCleanBackSourceSink =
                    ctx().weakSuperbubbles && !additionalCheck &&
                    back.globalSourceSink && back.acyclic && !back.hasLeakage;
                if (pass)
                {
                    if (!additionalCheck)
                    {
                        if (riskyCleanBackSourceSink)
                            markRiskyWeakSuperbubble(S, T);
                        addSuperbubble(S, T);
                    }
                    else if ((!swap && back.directST) || (swap && back.directTS))
                    {
                        addSuperbubble(S, T);
                    }
                }
            }

            void collectSuperbubbles(const CcData &cc, BlockData &blk, EdgeArray<EdgeDP> &edge_dp, NodeArray<NodeDPState> &node_dp)
            {
                const auto &T = blk.spqr->tree();

                for (edge e : T.edges)
                {
                    const spqr_compat::node parent = parentOnTreeEdge(T, blk, e);
                    const spqr_compat::node child = childOnTreeEdge(T, blk, e);

                    const EdgeDPState &down = edge_dp[e].down;
                    const EdgeDPState &up = edge_dp[e].up;
                    const auto parentType = blk.spqr->typeOf(parent);
                    const auto childType = blk.spqr->typeOf(child);

                    if ((childType == SPQRTree::NodeType::RNode && parentType != SPQRTree::NodeType::PNode) ||
                        (childType == SPQRTree::NodeType::PNode &&
                         parentType == SPQRTree::NodeType::SNode))
                    {
                        const bool check = childType == SPQRTree::NodeType::PNode;
                        tryBubble(down, up, blk, cc, false, check);
                        tryBubble(down, up, blk, cc, true, check);
                    }
                    if (childType == SPQRTree::NodeType::SNode &&
                        parentType == SPQRTree::NodeType::PNode)
                    {
                        tryBubble(down, up, blk, cc, false, true);
                        tryBubble(down, up, blk, cc, true, true);
                    }
                    if ((parentType == SPQRTree::NodeType::RNode && childType != SPQRTree::NodeType::PNode) ||
                        (parentType == SPQRTree::NodeType::PNode &&
                         childType == SPQRTree::NodeType::SNode))
                    {
                        const bool check = parentType == SPQRTree::NodeType::PNode;
                        tryBubble(up, down, blk, cc, false, check);
                        tryBubble(up, down, blk, cc, true, check);
                    }
                    if (parentType == SPQRTree::NodeType::SNode &&
                        childType == SPQRTree::NodeType::PNode)
                    {
                        tryBubble(up, down, blk, cc, false, true);
                        tryBubble(up, down, blk, cc, true, true);
                    }
                    blk.isAcycic &= (down.acyclic && up.acyclic);
                }
                for (node v : T.nodes)
                {
                    tryBubblePNodeGrouping(v, cc, blk, edge_dp);
                }
            }

        }

        void checkFlatBlockByCutVertices(const BlockData &blk)
        {
            const uint32_t n = weakBlockNodeCount(blk);
            const uint32_t m = weakBlockEdgeCount(blk);
            if (n == 0 || m == 0 || blk.edgeSrc.size() < m || blk.edgeDst.size() < m)
                return;

            std::vector<uint32_t> off(n + 1, 0);
            for (uint32_t i = 0; i < m; ++i)
                ++off[blk.edgeSrc[i] + 1];
            uint32_t sum = 0;
            for (uint32_t i = 0; i < n; ++i)
            {
                uint32_t count = off[i + 1];
                off[i] = sum;
                sum += count;
            }
            off[n] = sum;

            std::vector<uint32_t> cursor(off.begin(), off.end());
            std::vector<uint32_t> adj(m);
            for (uint32_t i = 0; i < m; ++i)
                adj[cursor[blk.edgeSrc[i]]++] = blk.edgeDst[i];

            auto reaches = [&](uint32_t src, uint32_t snk) -> bool {
                std::vector<uint8_t> vis(n, 0);
                std::vector<uint32_t> stack;
                stack.push_back(src);
                vis[src] = 1;
                while (!stack.empty())
                {
                    uint32_t u = stack.back();
                    stack.pop_back();
                    for (uint32_t p = off[u]; p < off[u + 1]; ++p)
                    {
                        uint32_t v = adj[p];
                        if (vis[v])
                            continue;
                        if (v == snk)
                            return true;
                        vis[v] = 1;
                        stack.push_back(v);
                    }
                }
                return false;
            };

            auto acyclic_without = [&](uint32_t from, uint32_t to) -> bool {
                std::vector<int> indeg(n, 0);
                for (uint32_t i = 0; i < m; ++i)
                    if (!(blk.edgeSrc[i] == from && blk.edgeDst[i] == to))
                        ++indeg[blk.edgeDst[i]];

                std::queue<uint32_t> q;
                for (uint32_t v = 0; v < n; ++v)
                    if (indeg[v] == 0)
                        q.push(v);

                uint32_t seen = 0;
                while (!q.empty())
                {
                    uint32_t u = q.front();
                    q.pop();
                    ++seen;
                    for (uint32_t p = off[u]; p < off[u + 1]; ++p)
                    {
                        uint32_t v = adj[p];
                        if (u == from && v == to)
                            continue;
                        if (--indeg[v] == 0)
                            q.push(v);
                    }
                }
                return seen == n;
            };

            auto try_pair = [&](uint32_t src, uint32_t snk, int backCount) -> bool {
                if (src == snk || src >= n || snk >= n)
                    return false;

                for (uint32_t i = 0; i < n; ++i)
                {
                    node v{i};
                    int inL = blk.inDeg[v], outL = blk.outDeg[v];
                    int inG = blk.globIn[v], outG = blk.globOut[v];

                    if (i == src)
                        inL -= backCount;
                    if (i == snk)
                        outL -= backCount;

                    if (i == src)
                    {
                        if (!(inL == 0 && outL == outG))
                            return false;
                    }
                    else if (i == snk)
                    {
                        if (!(outL == 0 && inL == inG))
                            return false;
                    }
                    else if (!(inL == inG && outL == outG))
                    {
                        return false;
                    }
                }

                if (backCount == 0)
                {
                    if (!acyclic_without(UINT32_MAX, UINT32_MAX))
                        return false;
                }
                else if (!acyclic_without(snk, src))
                {
                    return false;
                }

                if (!reaches(src, snk))
                    return false;

                addSuperbubble(blk.toOrig[node{src}], blk.toOrig[node{snk}]);
                return true;
            };

            uint32_t src = UINT32_MAX, snk = UINT32_MAX;
            for (uint32_t i = 0; i < n; ++i)
            {
                node v{i};
                bool isSrc = (blk.inDeg[v] == 0 && blk.outDeg[v] == blk.globOut[v]);
                bool isSnk = (blk.outDeg[v] == 0 && blk.inDeg[v] == blk.globIn[v]);
                if (isSrc ^ isSnk)
                {
                    if (isSrc)
                    {
                        if (src != UINT32_MAX)
                            return;
                        src = i;
                    }
                    else
                    {
                        if (snk != UINT32_MAX)
                            return;
                        snk = i;
                    }
                }
            }

            if (src != UINT32_MAX && snk != UINT32_MAX && try_pair(src, snk, 0))
                return;

            if (!ctx().weakSuperbubbles)
                return;

            std::unordered_set<uint64_t> tried;
            for (uint32_t i = 0; i < m; ++i)
            {
                uint32_t s = blk.edgeDst[i], t = blk.edgeSrc[i];
                uint64_t key = (static_cast<uint64_t>(s) << 32) | t;
                if (!tried.insert(key).second)
                    continue;
                int backCount = 0;
                for (uint32_t p = off[t]; p < off[t + 1]; ++p)
                    if (adj[p] == s)
                        ++backCount;
                if (backCount > 0 && try_pair(s, t, backCount))
                    return;
            }
        }

        void checkBlockByCutVertices(const BlockData &blk)
        {
            if (blk.flatEdgeCount != 0 && blk.Gblk && blk.Gblk->numberOfEdges() == 0)
            {
                checkFlatBlockByCutVertices(blk);
                return;
            }

            const Graph &G = *blk.Gblk;

            auto reaches = [&](node src, node snk) -> bool {
                NodeArray<bool> vis(G, false);
                std::stack<node> S;
                vis[src] = true;
                S.push(src);
                while (!S.empty())
                {
                    node u = S.top();
                    S.pop();
                    bool found = false;
                    G.forEachAdj(u, [&](node v, edge e) {
                        if (found || blockEdgeSource(blk, e) != u)
                            return;
                        if (!vis[v])
                        {
                            if (v == snk)
                            {
                                found = true;
                                return;
                            }
                            vis[v] = true;
                            S.push(v);
                        }
                    });
                    if (found)
                        return true;
                }
                return false;
            };

            auto acyclic_without = [&](node from, node to) -> bool {
                std::vector<int> indeg(G.numberOfNodes(), 0);
                for (edge e : G.edges)
                    if (!(blockEdgeSource(blk, e) == from && blockEdgeTarget(blk, e) == to))
                        indeg[blockEdgeTarget(blk, e).idx]++;

                std::queue<node> q;
                for (node v : G.nodes)
                    if (indeg[v.idx] == 0)
                        q.push(v);

                uint32_t seen = 0;
                while (!q.empty())
                {
                    node u = q.front();
                    q.pop();
                    seen++;
                    G.forEachAdj(u, [&](node v, edge e) {
                        if (blockEdgeSource(blk, e) != u || (u == from && v == to))
                            return;
                        if (--indeg[v.idx] == 0)
                            q.push(v);
                    });
                }
                return seen == G.numberOfNodes();
            };

            auto try_pair = [&](node src, node snk, int backCount) -> bool {
                if (!src || !snk || src == snk)
                    return false;

                for (node v : G.nodes)
                {
                    int inL = blk.inDeg[v], outL = blk.outDeg[v];
                    int inG = blk.globIn[v], outG = blk.globOut[v];

                    if (v == src)
                        inL -= backCount;
                    if (v == snk)
                        outL -= backCount;

                    if (v == src)
                    {
                        if (!(inL == 0 && outL == outG))
                            return false;
                    }
                    else if (v == snk)
                    {
                        if (!(outL == 0 && inL == inG))
                            return false;
                    }
                    else if (!(inL == inG && outL == outG))
                    {
                        return false;
                    }
                }

                if (backCount == 0)
                {
                    if (!isAcyclic(G))
                        return false;
                }
                else if (!acyclic_without(snk, src))
                {
                    return false;
                }

                if (!reaches(src, snk))
                    return false;

                node srcG = blk.toOrig[src], snkG = blk.toOrig[snk];
                addSuperbubble(srcG, snkG);
                return true;
            };

            node src = nullptr, snk = nullptr;
            for (node v : G.nodes)
            {
                bool isSrc = (blk.inDeg[v] == 0 && blk.outDeg[v] == blk.globOut[v]);
                bool isSnk = (blk.outDeg[v] == 0 && blk.inDeg[v] == blk.globIn[v]);
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
            }

            if (try_pair(src, snk, 0))
                return;

            if (!ctx().weakSuperbubbles)
                return;

            std::unordered_set<uint64_t> tried;
            for (edge e : G.edges)
            {
                node s = blockEdgeTarget(blk, e), t = blockEdgeSource(blk, e);
                uint64_t key = (static_cast<uint64_t>(s.idx) << 32) | t.idx;
                if (!tried.insert(key).second)
                    continue;
                int backCount = 0;
                G.forEachAdj(t, [&](node v, edge h) {
                    if (blockEdgeSource(blk, h) == t && v == s)
                        backCount++;
                });
                if (backCount > 0 && try_pair(s, t, backCount))
                    return;
            }
        }

        void solveSPQR(BlockData &blk, const CcData &cc)
        {

            if (ctx().weakSuperbubbles && SPQRsolve::solveSPQRMacroDirect(blk, cc))
            {
                return;
            }

            if (!blk.spqr || blk.Gblk->numberOfNodes() < 3)
            {
                return;
            }

            const auto &T = blk.spqr->tree();

            EdgeArray<SPQRsolve::EdgeDP> dp(T);
            NodeArray<SPQRsolve::NodeDPState> node_dp(T);

            std::vector<spqr_compat::node> nodeOrder;
            std::vector<spqr_compat::edge> edgeOrder;

            SPQRsolve::dfsSPQR_order(*blk.spqr, edgeOrder, nodeOrder, blk.parent, blk.root);

            blk.blkToSkel.init(*blk.Gblk, nullptr);

            {
                BF_INSTR(auto __t0_ph1 = std::chrono::high_resolution_clock::now();)
                for (auto e : edgeOrder)
                {
                    SPQRsolve::processEdge(e, dp, node_dp, cc, blk);
                }
                BF_INSTR(
                auto __t1_ph1 = std::chrono::high_resolution_clock::now();
                profiling_patch::phase1_edge_dp_time_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__t1_ph1 - __t0_ph1).count(),
                    std::memory_order_relaxed);
                profiling_patch::phase1_calls.fetch_add(1, std::memory_order_relaxed);
                )
            }

            NodeArray<unsigned char> phase2Prune(T, (unsigned char)SPQRsolve::P2None);

            {
                BF_INSTR(auto __t0_ph2 = std::chrono::high_resolution_clock::now();)
                for (auto v : nodeOrder)
                {
                    const unsigned char mask = phase2Prune[v];

                    if (mask == SPQRsolve::P2None)
                    {
                        BF_INSTR(profiling_patch::phase2_full_count.fetch_add(1, std::memory_order_relaxed);)
                        SPQRsolve::processNode(v, dp, node_dp, cc, blk);
                    }
                    else
                    {
                        BF_INSTR(profiling_patch::phase2_light_count.fetch_add(1, std::memory_order_relaxed);)
                        BF_INSTR(
                        if (mask & SPQRsolve::P2Tip)
                            profiling_patch::phase2_pruned_tip.fetch_add(1, std::memory_order_relaxed);
                        )
                        BF_INSTR(
                        if (mask & SPQRsolve::P2Cycle)
                            profiling_patch::phase2_pruned_cycle.fetch_add(1, std::memory_order_relaxed);
                        )
                        BF_INSTR(
                        if ((mask & SPQRsolve::P2Tip) && (mask & SPQRsolve::P2Cycle))
                            profiling_patch::phase2_pruned_both.fetch_add(1, std::memory_order_relaxed);
                        )
                        SPQRsolve::processNodePrunedLight(v, dp, blk, mask);
                    }

                    T.forEachAdj(v, [&](spqr_compat::node child, spqr_compat::edge te) {
                        if (child == blk.parent[v])
                            return;

                        const unsigned char childMask =
                            (unsigned char)(mask | SPQRsolve::pruneMaskFromState(dp[te].up));

                        phase2Prune[child] = childMask;
                    });
                }
                BF_INSTR(
                auto __t1_ph2 = std::chrono::high_resolution_clock::now();
                profiling_patch::phase2_node_dp_time_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__t1_ph2 - __t0_ph2).count(),
                    std::memory_order_relaxed);
                profiling_patch::phase2_calls.fetch_add(1, std::memory_order_relaxed);
                )
            }

            {
                BF_INSTR(auto __t0_ph3 = std::chrono::high_resolution_clock::now();)
                SPQRsolve::collectSuperbubbles(cc, blk, dp, node_dp);
                BF_INSTR(
                auto __t1_ph3 = std::chrono::high_resolution_clock::now();
                profiling_patch::phase3_collect_time_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__t1_ph3 - __t0_ph3).count(),
                    std::memory_order_relaxed);
                profiling_patch::phase3_calls.fetch_add(1, std::memory_order_relaxed);
                )
            }
        }

        void findMiniSuperbubbles()
            {
                MARK_SCOPE_MEM("sb/findMini");

                auto &C = ctx();

                if (C.weakSuperbubbles && !C.includeTrivial)
                    return;

                logger::info("Finding mini-superbubbles..");


            std::vector<spqr_compat::edge> edges_vec;
            edges_vec.reserve(C.G.numberOfEdges());
            for (auto e : C.G.edges) edges_vec.push_back(e);

            size_t numThreads = std::thread::hardware_concurrency();
            numThreads = std::min({(size_t)C.threads, numThreads});
            if (numThreads == 0) numThreads = 1;

            auto check_and_emit = [&](spqr_compat::edge e) {
                auto a = C.G.source(e);
                auto b = C.G.target(e);
                if (C.G.outdeg(a) == 1 && C.G.indeg(b) == 1)
                {
                    bool ok = true;
                    if (!C.weakSuperbubbles)
                    {
                        C.G.forEachAdj(b, [&](node /*other*/, edge e2) {
                            auto src = C.G.source(e2);
                            auto tgt = C.G.target(e2);
                            if (src == b && tgt == a)
                            {
                                ok = false;
                            }
                        });
                    }
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
                std::vector<std::vector<std::pair<spqr_compat::node, spqr_compat::node>>> results(numThreads);

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

        static void configureSpqrRustThreads()
        {
            static std::once_flag rustThreadEnvOnce;
            std::call_once(rustThreadEnvOnce, []() {
                auto &C = ctx();
                if (!std::getenv("BF_SPQR_THREADS"))
                {
                    std::string threads = std::to_string(C.threads > 0 ? C.threads : 1);
                    setenv("BF_SPQR_THREADS", threads.c_str(), 0);
                }
                if (!std::getenv("BF_SPQR_PAR_COMBINE"))
                    setenv("BF_SPQR_PAR_COMBINE", "1", 0);
            });
        }

        static void buildDegreeTwoContractibleMask(const BlockData &blk,
                                                   std::vector<uint8_t> &contractible)
        {
            const uint32_t n_blk = weakBlockNodeCount(blk);
            contractible.assign(n_blk, 0);
            for (uint32_t i = 0; i < n_blk; ++i)
            {
                node vB{i};
                const int totalDeg = blk.inDeg[vB] + blk.outDeg[vB];
                if (totalDeg == 2)
                    contractible[vB.idx] = 1;
            }
        }

        static bool shouldAttemptWeakMacroDirect(const BlockData &blk)
        {
            if (!ctx().weakSuperbubbles || !blk.Gblk)
                return false;

            const uint64_t n_blk = static_cast<uint64_t>(weakBlockNodeCount(blk));
            const uint64_t m_blk = static_cast<uint64_t>(weakBlockEdgeCount(blk));
            if (n_blk < 3 || m_blk < 2)
                return false;

            return m_blk >= 10000000ull;
        }

        static bool buildBlockSpqrCompressed(BlockData &blk,
                                             const std::vector<uint8_t> &contractible)
        {
            const uint32_t n_blk = weakBlockNodeCount(blk);
            const uint32_t edgeCount = weakBlockEdgeCount(blk);
            const bool haveIndexedEdges =
                blk.edgeSrc.size() == edgeCount && blk.edgeDst.size() == edgeCount;
            std::vector<SpCompressInputEdge> edges;
            if (!haveIndexedEdges)
            {
                edges.reserve(edgeCount);
                for (edge e : blk.Gblk->edges)
                {
                    edges.push_back(SpCompressInputEdge{
                        static_cast<uint32_t>(blockEdgeSource(blk, e).idx),
                        static_cast<uint32_t>(blockEdgeTarget(blk, e).idx),
                        static_cast<uint32_t>(e.idx)});
                }
            }

            const uint8_t buildCoreMode =
                (spqra_profile::aqDirectCoreScadRequested() ||
                 (spqra_profile::aqAtomsRequested() &&
                      !spqra_profile::aqAtomsUseSpqrTreeFallback()))
                    ? static_cast<uint8_t>(2)
                    : static_cast<uint8_t>(1);
            SpCompressHandle *raw_handle = nullptr;
            if (haveIndexedEdges)
            {
                raw_handle = sp_compress_indexed_ffi(
                    n_blk,
                    blk.edgeSrc.empty() ? nullptr : blk.edgeSrc.data(),
                    blk.edgeDst.empty() ? nullptr : blk.edgeDst.data(),
                    edgeCount,
                    contractible.data(),
                    static_cast<uint32_t>(contractible.size()),
                    buildCoreMode);
            }
            else
            {
                    raw_handle = sp_compress_ffi(
                        n_blk,
                        edges.empty() ? nullptr : edges.data(),
                        static_cast<uint32_t>(edges.size()),
	                    contractible.data(),
                    static_cast<uint32_t>(contractible.size()),
                    buildCoreMode);
            }

            if (!raw_handle || sp_compress_success(raw_handle) == 0)
            {
                if (raw_handle)
                    sp_compress_free(raw_handle);
                throw std::runtime_error("sp_compress_ffi failed in superbubble macro-direct mode");
            }

            blk.spCompressHandle.reset(raw_handle);
            blk.macroTreeView = sp_compress_get_tree(raw_handle);
            blk.coreSpqrTree = sp_compress_get_core_spqr(raw_handle);
            blk.coreScadView = sp_compress_get_core_scad_export(raw_handle);
            blk.spqraMinimizerView = sp_compress_get_spqra_minimizer_view(raw_handle);
            blk.spqraBehaviorAtomView = sp_compress_get_spqra_behavior_atom_view(raw_handle);
            blk.coreNodeInv = sp_compress_core_node_inv(raw_handle, &blk.coreNodeInvLen);

                const bool macroOnlyCore =
                    blk.macroTreeView.macros_len != 0 &&
                    (blk.macroTreeView.stats.fully_sp_reducible != 0 ||
                     blk.macroTreeView.core_edges_len == 0);
	            const bool usable =
	                blk.macroTreeView.macros_len != 0 &&
	                    (macroOnlyCore ||
	                     ((blk.coreSpqrTree ||
                       (blk.coreScadView.components_len != 0 && blk.coreScadView.components_ptr != nullptr) ||
                       (blk.spqraMinimizerView.components_len != 0 &&
                        blk.spqraMinimizerView.components_ptr != nullptr)) &&
                      blk.coreNodeInv && blk.coreNodeInvLen != 0));

            if (!usable)
            {
                blk.spCompressHandle.reset();
                blk.macroTreeView = SpCompressTreeView{};
                blk.coreSpqrTree = nullptr;
                blk.coreScadView = CoreScadView{};
                blk.spqraMinimizerView = SpqraMinimizerView{};
                blk.coreNodeInv = nullptr;
                blk.coreNodeInvLen = 0;
                return false;
            }
            return true;
        }

        static void materializeFlatBlockEdges(BlockData &blk)
        {
            if (!blk.Gblk || blk.flatEdgeCount == 0 ||
                blk.Gblk->numberOfEdges() == blk.flatEdgeCount)
            {
                return;
            }
            for (uint32_t i = 0; i < blk.flatEdgeCount; ++i)
                blk.Gblk->newEdge(node{blk.edgeSrc[i]}, node{blk.edgeDst[i]});
        }

        static void buildBlockDataParallel(const CcData &cc, BlockData &blk)
        {
            {
                blk.Gblk = std::make_unique<Graph>();

                blk.toOrig.init(*blk.Gblk, nullptr);
                blk.toCc.init(*blk.Gblk, nullptr);
                blk.inDeg.init(*blk.Gblk, 0);
                blk.outDeg.init(*blk.Gblk, 0);

	                auto hEdges = cc.bc->hEdges(blk.bNode);
	                size_t hEdgeCount = 0;
	                for (edge hE : hEdges)
	                {
	                    (void)hE;
	                    ++hEdgeCount;
	                }

                    const bool flatMacroOnly =
                        ctx().weakSuperbubbles &&
                        hEdgeCount >= 10000000ull;

	                if (hEdgeCount >= 1000000)
	                {
                        blk.edgeSrc.reserve(hEdgeCount);
                        blk.edgeDst.reserve(hEdgeCount);
	                    std::vector<int32_t> ccToBlkIdx(cc.Gcc->numberOfNodes(), -1);
	                    auto ensureBlockNode = [&](node vCc) -> node {
	                        int32_t mapped = ccToBlkIdx[vCc.idx];
	                        if (mapped >= 0)
	                            return node{static_cast<uint32_t>(mapped)};
	                        node vB = blk.Gblk->newNode();
	                        ccToBlkIdx[vCc.idx] = static_cast<int32_t>(vB.idx);
	                        blk.toCc[vB] = vCc;
	                        blk.toOrig[vB] = cc.toOrig[vCc];
	                        blk.inDeg[vB] = 0;
	                        blk.outDeg[vB] = 0;
	                        return vB;
	                    };

	                    for (edge hE : hEdges)
	                    {
	                        edge eCc = cc.bc->original(hE);
	                        node src = ensureBlockNode(cc.Gcc->source(eCc));
	                        node tgt = ensureBlockNode(cc.Gcc->target(eCc));
                            if (flatMacroOnly)
                            {
                                blk.edgeSrc.push_back(src.idx);
                                blk.edgeDst.push_back(tgt.idx);
                            }
                            else
	                        {
	                            edge e = blk.Gblk->newEdge(src, tgt);
	                            if (e.idx == blk.edgeSrc.size())
	                            {
	                                blk.edgeSrc.push_back(src.idx);
	                                blk.edgeDst.push_back(tgt.idx);
	                            }
	                        }
	                        blk.outDeg[src]++;
	                        blk.inDeg[tgt]++;
	                    }
                        if (flatMacroOnly)
                        {
                            blk.flatNodeCount = static_cast<uint32_t>(blk.Gblk->numberOfNodes());
                            blk.flatEdgeCount = static_cast<uint32_t>(blk.edgeSrc.size());
                        }
	                }
	                else
	                {
                        blk.edgeSrc.reserve(hEdgeCount);
                        blk.edgeDst.reserve(hEdgeCount);
	                    std::unordered_set<node> verts;
	                    for (edge hE : hEdges)
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
                    blk.inDeg[vB] = 0;
                    blk.outDeg[vB] = 0;
                }

	                    for (edge hE : hEdges)
	                    {
	                        edge eCc = cc.bc->original(hE);
	                        auto srcIt = cc_to_blk.find(cc.Gcc->source(eCc));
	                        auto tgtIt = cc_to_blk.find(cc.Gcc->target(eCc));
	                        if (srcIt != cc_to_blk.end() && tgtIt != cc_to_blk.end())
	                        {
                                if (flatMacroOnly)
                                {
                                    blk.edgeSrc.push_back(srcIt->second.idx);
                                    blk.edgeDst.push_back(tgtIt->second.idx);
                                }
                                else
                                {
	                                edge e = blk.Gblk->newEdge(srcIt->second, tgtIt->second);
	                                if (e.idx == blk.edgeSrc.size())
	                                {
	                                    blk.edgeSrc.push_back(srcIt->second.idx);
	                                    blk.edgeDst.push_back(tgtIt->second.idx);
	                                }
                                }
	                            blk.outDeg[srcIt->second]++;
	                            blk.inDeg[tgtIt->second]++;
	                        }
	                    }
                        if (flatMacroOnly)
                        {
                            blk.flatNodeCount = static_cast<uint32_t>(blk.Gblk->numberOfNodes());
                            blk.flatEdgeCount = static_cast<uint32_t>(blk.edgeSrc.size());
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

	            std::vector<uint8_t> contractible;
            buildDegreeTwoContractibleMask(blk, contractible);

            if (shouldAttemptWeakMacroDirect(blk) &&
                buildBlockSpqrCompressed(blk, contractible))
		                return;

                materializeFlatBlockEdges(blk);

	            if (blk.Gblk->numberOfNodes() >= 3)
	            {
	                {
	                    blk.spqr = std::make_unique<StaticSPQRTree>(
	                        *blk.Gblk,
	                        contractible.data(),
                        static_cast<uint32_t>(contractible.size()));
                }
                const auto &T = blk.spqr->tree();
                blk.skel2tree.reserve(2 * T.edges.size());

                for (edge te : T.edges)
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

                rootSPQRTreeForDP(blk);
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
            std::vector<std::vector<std::pair<spqr_compat::node, spqr_compat::node>>> *blockResults;
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

                std::vector<std::pair<spqr_compat::node, spqr_compat::node>> local;
                tls_superbubble_collector = &local;

                if (blk->Gblk && weakBlockNodeCount(*blk) >= 3)
                {
                    BF_INSTR(
                    {
                        uint64_t blk_ss = 0;
                        uint64_t v_count = 0;
                        for (spqr_compat::node vB : blk->Gblk->nodes) {
                            ++v_count;
                            if (blk->globIn[vB] == 0 || blk->globOut[vB] == 0) ++blk_ss;
                        }
                        tip_diag_spqr_blocks.fetch_add(1, std::memory_order_relaxed);
                        tip_diag_spqr_verts.fetch_add(v_count, std::memory_order_relaxed);
                        tip_diag_internal_ss.fetch_add(blk_ss, std::memory_order_relaxed);
                        if (blk_ss > 0)
                            tip_diag_blocks_with_ss.fetch_add(1, std::memory_order_relaxed);
                        uint64_t prev = tip_diag_max_ss_one_block.load(std::memory_order_relaxed);
                        while (blk_ss > prev &&
                               !tip_diag_max_ss_one_block.compare_exchange_weak(
                                   prev, blk_ss, std::memory_order_relaxed)) {}
                    }
                    )
                    solveSPQR(*blk, *w.cc);
                }
                checkBlockByCutVertices(*blk);

                tls_superbubble_collector = nullptr;
                results[i] = std::move(local);
                allBlocks[i].reset();
            }
            return nullptr;
        }

        struct PrecomputedCcInput
        {
            std::vector<uint32_t> localToGlobal;
            std::vector<std::pair<int, int>> edges;
        };

        static void solvePreparedComponents(std::vector<std::unique_ptr<CcData>> &components)
        {
            auto &C = ctx();
            const int nCC = static_cast<int>(components.size());
            if (nCC == 0)
                return;

            std::vector<WorkItem> workItems;
            std::vector<std::unique_ptr<BlockData>> allBlockData;
            std::vector<BlockPrep> blockPreps;

            {
                PROFILE_BLOCK("solve:: building data");
                MEM_TIME_BLOCK("BUILD: BC+SPQR");
                ACCUM_BUILD();

                {
                    MARK_SCOPE_MEM("sb/phase/BCtrees");

                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, (size_t)nCC, numThreads});
                    if (numThreads == 0)
                        numThreads = 1;

                    std::vector<pthread_t> threads(numThreads);
                    std::atomic<size_t> nextIndex{0};
                    std::vector<std::vector<BlockPrep>> perThreadPreps(numThreads);

                    for (size_t tid = 0; tid < numThreads; ++tid)
                    {
                        pthread_attr_t attr;
                        pthread_attr_init(&attr);

                        if (pthread_attr_setstacksize(&attr, C.stackSize) != 0)
                            std::cout << "[Error] pthread_attr_setstacksize" << std::endl;

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
                        pthread_join(threads[tid], nullptr);

                    size_t total = 0;
                    for (auto &tp : perThreadPreps)
                        total += tp.size();
                    blockPreps.reserve(total);
                    for (auto &tp : perThreadPreps)
                    {
                        blockPreps.insert(blockPreps.end(),
                                          std::make_move_iterator(tp.begin()),
                                          std::make_move_iterator(tp.end()));
                    }
                }

                {
                    allBlockData.resize(blockPreps.size());
                }

                if (!blockPreps.empty())
                {
                    MARK_SCOPE_MEM("sb/phase/BlockDataBuildAll");

                    size_t numThreads2 = std::thread::hardware_concurrency();
                    numThreads2 = std::min({(size_t)C.threads, blockPreps.size(), numThreads2});
                    if (numThreads2 == 0)
                        numThreads2 = 1;

                    std::vector<pthread_t> threads2(numThreads2);
                    std::atomic<size_t> nextIndex2{0};

                    for (size_t tid = 0; tid < numThreads2; ++tid)
                    {
                        pthread_attr_t attr;
                        pthread_attr_init(&attr);

                        if (pthread_attr_setstacksize(&attr, C.stackSize) != 0)
                            std::cout << "[Error] pthread_attr_setstacksize" << std::endl;

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
                        pthread_join(threads2[tid], nullptr);
                }

                {
                    workItems.reserve(allBlockData.size());
                    for (size_t i = 0; i < allBlockData.size(); ++i)
                        workItems.push_back({blockPreps[i].cc, blockPreps[i].bNode});
                }
            }

            {
                MEM_TIME_BLOCK("LOGIC: solve blocks (pthreads)");
                ACCUM_LOGIC();
                PROFILE_BLOCK("solve:: process blocks (pthreads, large stack)");
                MARK_SCOPE_MEM("sb/phase/SolveBlocks");

                std::vector<std::vector<std::pair<spqr_compat::node, spqr_compat::node>>> blockResults(workItems.size());

                size_t numThreads = std::thread::hardware_concurrency();
                numThreads = std::min({(size_t)C.threads, workItems.size(), numThreads});
                if (numThreads == 0)
                    numThreads = 1;

                std::vector<pthread_t> threads(numThreads);
                std::atomic<size_t> nextIndex{0};

                {
                    for (size_t tid = 0; tid < numThreads; ++tid)
                    {
                        pthread_attr_t attr;
                        pthread_attr_init(&attr);
                        pthread_attr_setstacksize(&attr, C.stackSize);

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
                        pthread_join(threads[tid], nullptr);
                }

                {
                    size_t totalCandidates = 0;
                    for (const auto &candidates : blockResults)
                        totalCandidates += candidates.size();
                    C.superbubbles.reserve(C.superbubbles.size() + totalCandidates);
                    if (C.weakSuperbubbles)
                    {
                        for (const auto &candidates : blockResults)
                            for (const auto &p : candidates)
                                if (!isTrashSuperbubbleNode(p.first) &&
                                    !isTrashSuperbubbleNode(p.second))
                                    C.superbubbles.emplace_back(p);
                    }
                    else
                    {
                        for (const auto &candidates : blockResults)
                            for (const auto &p : candidates)
                                tryCommitSuperbubble(p.first, p.second);
                    }
                }
            }

            BF_INSTR(
            {
                const uint64_t b = tip_diag_spqr_blocks.load();
                const uint64_t v = tip_diag_spqr_verts.load();
                const uint64_t ss = tip_diag_internal_ss.load();
                const uint64_t bs = tip_diag_blocks_with_ss.load();
                const uint64_t mx = tip_diag_max_ss_one_block.load();
                std::cout << "\n[tip-diag] SPQR-eligible blocks: " << b
                          << ", verts: " << v
                          << ", source/sink verts: " << ss
                          << " (" << (v ? 100.0 * ss / v : 0.0) << "%)"
                          << ", blocks with >=1: " << bs
                          << " (" << (b ? 100.0 * bs / b : 0.0) << "%)"
                          << ", max in one block: " << mx << "\n";
            }
            )
        }

        void solveStreamingPrecomputed(const std::vector<PrecomputedCcInput> &inputs)
        {
            auto &C = ctx();
            configureSpqrRustThreads();
            {
                findMiniSuperbubbles();
            }
            std::vector<std::unique_ptr<CcData>> components;

            {
                MARK_SCOPE_MEM("sb/phase/GccBuildPrecomputed");
                components.resize(inputs.size());
                size_t numThreads = std::thread::hardware_concurrency();
                numThreads = std::min({(size_t)C.threads, inputs.size(), numThreads});
                if (numThreads == 0)
                    numThreads = 1;

                std::vector<std::thread> workers;
                workers.reserve(numThreads);
                std::atomic<size_t> nextIndex{0};

                for (size_t tid = 0; tid < numThreads; ++tid)
                {
                    workers.emplace_back([&]() {
                        while (true)
                        {
                            size_t ci = nextIndex.fetch_add(1, std::memory_order_relaxed);
                            if (ci >= inputs.size())
                                break;

                            const PrecomputedCcInput &in = inputs[ci];
                            if (in.edges.size() < in.localToGlobal.size())
                                continue;

                            auto cc = std::make_unique<CcData>();
                            cc->Gcc = std::make_unique<Graph>();
                            cc->toOrig.init(*cc->Gcc, nullptr);

                            for (uint32_t global : in.localToGlobal)
                            {
                                node vC = cc->Gcc->newNode();
                                cc->toOrig[vC] = node{global};
                            }

                            for (const auto &e : in.edges)
                            {
                                if (e.first < 0 || e.second < 0)
                                    continue;
                                uint32_t s = static_cast<uint32_t>(e.first);
                                uint32_t t = static_cast<uint32_t>(e.second);
                                if (s < in.localToGlobal.size() && t < in.localToGlobal.size())
                                    cc->Gcc->newEdge(node{s}, node{t});
                            }

                            components[ci] = std::move(cc);
                        }
                    });
                }

                for (auto &t : workers)
                    t.join();
            }

            solvePreparedComponents(components);
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

                std::vector<std::vector<std::pair<spqr_compat::node, spqr_compat::node>>> blockResults(workItems.size());

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

                size_t totalCandidates = 0;
                for (const auto &candidates : blockResults)
                    totalCandidates += candidates.size();
                C.superbubbles.reserve(C.superbubbles.size() + totalCandidates);
                if (C.weakSuperbubbles)
                {
                    for (const auto &candidates : blockResults)
                        for (const auto &p : candidates)
                            if (!isTrashSuperbubbleNode(p.first) &&
                                !isTrashSuperbubbleNode(p.second))
                                C.superbubbles.emplace_back(p);
                }
                else
                {
                    for (const auto &candidates : blockResults)
                        for (const auto &p : candidates)
                            tryCommitSuperbubble(p.first, p.second);
                }
            }


            BF_INSTR(
            {
                const uint64_t b = tip_diag_spqr_blocks.load();
                const uint64_t v = tip_diag_spqr_verts.load();
                const uint64_t ss = tip_diag_internal_ss.load();
                const uint64_t bs = tip_diag_blocks_with_ss.load();
                const uint64_t mx = tip_diag_max_ss_one_block.load();
                std::cout << "\n[tip-diag] SPQR-eligible blocks: " << b
                            << ", verts: " << v
                            << ", source/sink verts: " << ss
                            << " (" << (v ? 100.0 * ss / v : 0.0) << "%)"
                            << ", blocks with >=1: " << bs
                            << " (" << (b ? 100.0 * bs / b : 0.0) << "%)"
                            << ", max in one block: " << mx << "\n";
            }
            )
        }

        #ifdef BUBBLEFINDER_INSTRUMENT
                static void printPatch1Profiling()
                {
                    namespace pp = profiling_patch;
                
                    constexpr double NS_TO_MS = 1e-6;
                    const auto pct = [](double v, double total) {
                        return total > 0.0 ? (100.0 * v / total) : 0.0;
                    };
                
                    const uint64_t n_blocks = pp::blocks_with_spqr.load();
                    const uint64_t n_tnodes = pp::total_tree_nodes.load();
                    const uint64_t n_pe = pp::pe_total_calls.load();
                    const uint64_t n_pn_full = pp::phase2_full_count.load();
                    const uint64_t n_pn_light = pp::phase2_light_count.load();
                    const uint64_t n_pn_total = n_pn_full + n_pn_light;
                
                    const double t_p1 = pp::phase1_edge_dp_time_ns.load() * NS_TO_MS;
                    const double t_p2 = pp::phase2_node_dp_time_ns.load() * NS_TO_MS;
                    const double t_p3 = pp::phase3_collect_time_ns.load() * NS_TO_MS;
                    const double t_phases = t_p1 + t_p2 + t_p3;
                
                    auto &os = std::cout;
                    auto old_flags = os.flags();
                    auto old_precision = os.precision();
                
                    auto p3 = [&]() -> std::ostream& { os << std::fixed << std::setprecision(3); return os; };
                    auto p1 = [&]() -> std::ostream& { os << std::fixed << std::setprecision(1); return os; };
                
                    p3() << "\nSPQR solver profile\n"
                        << n_blocks << " blocks, " << n_tnodes << " tree nodes (avg ";
                    p1() << (n_blocks ? (double)n_tnodes / (double)n_blocks : 0.0)
                        << "/block).\n";
                
                    p3() << "\nPhases:\n"
                        << "  1  edge DP   (processEdge)          " << t_p1 << " ms (";
                    p1() << pct(t_p1, t_phases) << "%) / " << n_pe << " calls\n";
                    p3() << "  2  node DP   (processNode)          " << t_p2 << " ms (";
                    p1() << pct(t_p2, t_phases) << "%) / " << n_pn_total << " calls ("
                        << n_pn_full << " full, " << n_pn_light << " pruned)\n";
                    p3() << "  3  collect   (collectSuperbubbles)  " << t_p3 << " ms (";
                    p1() << pct(t_p3, t_phases) << "%)\n";
                
                    if (n_pn_total > 0) {
                        const uint64_t both = pp::phase2_pruned_both.load();
                        const uint64_t only_src = pp::phase2_pruned_tip.load()   - both;
                        const uint64_t only_cyc = pp::phase2_pruned_cycle.load() - both;
                        os << "\nPhase 2 pruning: " << n_pn_light << "/" << n_pn_total << " (";
                        p1() << pct((double)n_pn_light, (double)n_pn_total) << "%)";
                        os << " " << only_src << " by source/sink ancestor, "
                        << only_cyc << " by cycle, " << both << " by both.\n";
                    }
                
                    if (n_blocks > 0) {
                        const uint64_t scanned = pp::reroot_spqr_nodes_scanned.load();
                        p3() << "\nRoot selection (chooseSPQRRootForPruning): "
                            << (pp::reroot_time_ns.load() * NS_TO_MS) << " ms total. "
                            << pp::reroot_marked_found.load() << "/" << n_blocks
                            << " blocks found marked, "
                            << pp::reroot_fallback.load() << " fell back. "
                            << scanned << " skeletons scanned overall (avg ";
                        p1() << ((double)scanned / (double)n_blocks) << "/block).\n";
                    }
                
                    auto row = [&](const char *label, double t, double sum) {
                        os << "  " << label;
                        p3() << t << " ms (";
                        p1() << pct(t, sum) << "%)\n";
                    };
                
                    if (n_pe > 0) {
                        const double tA = pp::pe_A_setup_ns.load()* NS_TO_MS;
                        const double tB = pp::pe_B_build_ns.load() * NS_TO_MS;
                        const double tC = pp::pe_C_pnode_extra_ns.load() * NS_TO_MS;
                        const double tD = pp::pe_D_leakage_ns.load() * NS_TO_MS;
                        const double tE = pp::pe_E_acyclic_ns.load()* NS_TO_MS;
                        const double sum = tA + tB + tC + tD + tE;
                        os << "\nprocessEdge: " << n_pe << " calls, total ";
                        p3() << sum << " ms, ";
                        p1() << ((sum * 1000.0) / (double)n_pe) << " us avg\n";
                        row("setup            ", tA, sum);
                        row("build + child DP ", tB, sum);
                        row("P-node extras    ", tC, sum);
                        row("leakage          ", tD, sum);
                        row("acyclicity       ", tE, sum);
                    }
                
                    if (n_pn_full > 0) {
                        const double tA = pp::pn_A_setup_ns.load()* NS_TO_MS;
                        const double tB = pp::pn_B_build_ns.load()* NS_TO_MS;
                        const double tC = pp::pn_C_mark_ns.load() * NS_TO_MS;
                        const double tD = pp::pn_D_pnode_ns.load()* NS_TO_MS;
                        const double tE1= pp::pn_E1_branch_ns.load() * NS_TO_MS;
                        const double tE2 = pp::pn_E2_branch_ns.load() * NS_TO_MS;
                        const double tE3 = pp::pn_E3_branch_ns.load() * NS_TO_MS;
                        const double tF = pp::pn_F_gss_ns.load() * NS_TO_MS;
                        const double tG = pp::pn_G_leak_ns.load() * NS_TO_MS;
                        const double tH = pp::pn_H_poles_ns.load() * NS_TO_MS;
                        const double tE = tE1 + tE2 + tE3;
                        const double sum = tA + tB + tC + tD + tE + tF + tG + tH;
                        const uint64_t n_e1 = pp::pn_E1_calls.load();
                        const uint64_t n_e2 = pp::pn_E2_calls.load();
                        const uint64_t n_e3 = pp::pn_E3_calls.load();
                
                        os << "\nprocessNode: " << n_pn_full << " full calls, "
                        << pp::pn_pnode_early_returns.load()
                        << " P-node early returns, total ";
                        p3() << sum << " ms\n";
                        row("setup            ", tA, sum);
                        row("build            ", tB, sum);
                        row("mark src/sink    ", tC, sum);
                        row("P-node directST  ", tD, sum);
                        row("acyclicity check ", tE, sum);
                        os << "    full FAS run on newGraph:           " << n_e3 << " calls";
                        if (n_e3 > 0) {
                            os << " (";
                            p3() << tE3 << " ms, ";
                            p1() << ((tE3 * 1000.0) / (double)n_e3) << " us avg)";
                        }
                        os << "\n    shortcut (1 child has cycle):       " << n_e2 << " calls";
                        if (n_e2 > 0) { os << " ("; p3() << tE2 << " ms)"; }
                        os << "\n    shortcut (>=2 children have cycle): " << n_e1 << " calls";
                        if (n_e1 > 0) { os << " ("; p3() << tE1 << " ms)"; }
                        os << "\n";
                        row("source/sink prop ", tF, sum);
                        row("leakage prop     ", tG, sum);
                        row("pole fixup       ", tH, sum);
                    }
                
                    const uint64_t n_pn_fast = pp::pn_fastpath_calls.load();
                    const uint64_t n_e3_total = pp::pn_E3_calls.load();
                    const uint64_t n_e3_skip = pp::pn_E3_fas_dag_skipped.load();
                    if (n_pn_full > 0 || n_e3_total > 0) {
                        if (n_e3_total > 0) {
                            os << " FAS skipped on DAG " << n_e3_skip << "/" << n_e3_total << " (";
                            p1() << pct((double)n_e3_skip, (double)n_e3_total) << "%)";
                        }
                        os << ".\n";
                    }
                    os << std::endl;
                
                    os.precision(old_precision);
                    os.flags(old_flags);
                }
        #endif  // BUBBLEFINDER_INSTRUMENT
 

        void solve()
        {
            TIME_BLOCK("Finding superbubbles in blocks");
            configureSpqrRustThreads();
            findMiniSuperbubbles();
            solveStreaming();
            BF_INSTR(printPatch1Profiling();)
        }
    }

    namespace snarls
    {
        inline void addSnarlTagged(const char *tag, std::vector<std::string> s);

        #if defined(BF_HAVE_OPENMP) && BF_HAVE_OPENMP
            #define BF_OMP_PRAGMA(x) _Pragma(#x)
        #else
            static inline int omp_get_num_threads() { return 1; }
            static inline int omp_get_max_threads() { return 1; }
            static inline int omp_get_thread_num() { return 0; }
            static inline bool omp_in_parallel() { return false; }
            #define BF_OMP_PRAGMA(x) 
        #endif

        struct IntraPlan
        {
            bool critical = false;
            uint64_t quantum = 1;
            int numThreads = 1;
            std::atomic<int> *activeIntraTaskloops = nullptr;
            size_t bid = static_cast<size_t>(-1);
        };

        static inline uint64_t bf_ceil_div(uint64_t a, uint64_t b)
        {
            return b == 0 ? 0 : (a + b - 1) / b;
        }

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
            thread_local std::vector<uint64_t> *tls_fast_snarl_pair_buffer = nullptr;
            thread_local std::vector<std::vector<uint64_t>> *tls_fast_snarl_clique_buffer = nullptr;

            static std::mutex g_snarls_mtx;

            inline uint64_t pack_fast_snarl_endpoint_key(uint32_t idx, uint8_t sign)
            {
                return (static_cast<uint64_t>(idx) << 1) |
                       static_cast<uint64_t>(sign);
            }

            inline uint64_t pack_fast_snarl_pair_key(uint32_t a_idx, uint8_t a_sign,
                                                     uint32_t b_idx, uint8_t b_sign)
            {
                uint64_t a = pack_fast_snarl_endpoint_key(a_idx, a_sign);
                uint64_t b = pack_fast_snarl_endpoint_key(b_idx, b_sign);
                if (a > b) std::swap(a, b);
                return (a << 32) | b;
            }

            inline bool parse_fast_snarl_endpoint(const std::string &s,
                                                  uint32_t &idx,
                                                  uint8_t &sign)
            {
                if (s.size() < 2) return false;
                const char c = s.back();
                if (c == '+') {
                    sign = 0;
                } else if (c == '-') {
                    sign = 1;
                } else {
                    return false;
                }

                std::string name(s.data(), s.size() - 1);
                if (name == "_trash") return false;
                auto &C = ctx();
                auto it = C.name2node.find(name);
                if (it == C.name2node.end()) return false;
                idx = static_cast<uint32_t>(it->second.idx);
                return true;
            }

            inline bool tryCommitFastSnarlPairs(const std::vector<std::string> &s)
            {
                auto &C = ctx();
                if (!C.fastSnarlPairsEnabled) return false;
                if (s.size() < 2) return true;

                thread_local std::vector<std::pair<uint32_t, uint8_t>> endpoints;
                endpoints.clear();
                endpoints.reserve(s.size());

                for (const std::string &endpoint : s) {
                    uint32_t idx = 0;
                    uint8_t sign = 255;
                    if (!parse_fast_snarl_endpoint(endpoint, idx, sign)) {
                        return false;
                    }
                    endpoints.emplace_back(idx, sign);
                }

                addSnarlsFound(s.size() * (s.size() - 1) / 2);

                if (endpoints.size() == 2) {
                    const uint64_t key = pack_fast_snarl_pair_key(
                        endpoints[0].first, endpoints[0].second,
                        endpoints[1].first, endpoints[1].second);
                    if (tls_fast_snarl_pair_buffer) {
                        tls_fast_snarl_pair_buffer->push_back(key);
                    } else {
                        std::lock_guard<std::mutex> lk(g_snarls_mtx);
                        C.fastSnarlPairs.push_back(key);
                    }
                    return true;
                }

                std::vector<uint64_t> clique;
                clique.reserve(endpoints.size());
                for (const auto &endpoint : endpoints) {
                    clique.push_back(pack_fast_snarl_endpoint_key(endpoint.first,
                                                                  endpoint.second));
                }

                if (tls_fast_snarl_clique_buffer) {
                    tls_fast_snarl_clique_buffer->push_back(std::move(clique));
                } else {
                    std::lock_guard<std::mutex> lk(g_snarls_mtx);
                    C.fastSnarlCliques.push_back(std::move(clique));
                }
                return true;
            }

            struct SnarlEndpoint {
                spqr_compat::node node{nullptr};
                EdgePartType sign{EdgePartType::NONE};
            };

            using SnarlEndpointPair = std::pair<SnarlEndpoint, SnarlEndpoint>;

            struct CutSides {
                SnarlEndpoint prev_side;
                SnarlEndpoint next_side;
            };

            inline bool isTrashOriginalNode(spqr_compat::node v)
            {
                auto &C = ctx();
                const size_t idx = static_cast<size_t>(v.idx);
                if (idx < C.isTrashNodeByIndex.size()) {
                    return C.isTrashNodeByIndex[idx] != 0;
                }
                auto it = C.node2name.find(v);
                return it != C.node2name.end() && it->second == "_trash";
            }

            inline uint8_t endpoint_sign_bit(EdgePartType sign)
            {
                return sign == EdgePartType::MINUS ? 1 : 0;
            }

            inline char endpoint_sign_char(EdgePartType sign)
            {
                return sign == EdgePartType::MINUS ? '-' : '+';
            }

            inline bool debug_tagged_snarls_enabled()
            {
                const char *taggedPath = std::getenv("BF_DEBUG_TAGGED_SNARLS");
                return taggedPath && *taggedPath;
            }

            std::atomic<uint64_t> g_cnt_cut{0}, g_cnt_S{0}, g_cnt_P{0}, g_cnt_RR{0}, g_cnt_E{0};

            inline void countSnarlTag(const char *tag)
            {
                if (!tag) return;
                auto tag_is = [&](const char *kind) {
                    const size_t n = std::strlen(kind);
                    return std::strncmp(tag, kind, n) == 0 &&
                           (tag[n] == '\0' || tag[n] == ':');
                };
                if (tag_is("CUT"))
                    g_cnt_cut++;
                else if (tag_is("S"))
                    g_cnt_S++;
                else if (tag_is("P"))
                    g_cnt_P++;
                else if (tag_is("RR"))
                    g_cnt_RR++;
                else if (tag_is("E"))
                    g_cnt_E++;
            }

            inline void commitFastSnarlPairKey(uint64_t key)
            {
                auto &C = ctx();
                addSnarlsFound(1);
                if (tls_fast_snarl_pair_buffer) {
                    tls_fast_snarl_pair_buffer->push_back(key);
                    return;
                }
                std::lock_guard<std::mutex> lk(g_snarls_mtx);
                C.fastSnarlPairs.push_back(key);
            }

            inline bool tryCommitFastSnarlEndpoints(const char *tag,
                                                    const std::vector<SnarlEndpoint> &endpoints)
            {
                auto &C = ctx();
                if (!C.fastSnarlPairsEnabled || debug_tagged_snarls_enabled())
                    return false;
                if (endpoints.size() < 2)
                    return true;

                for (const SnarlEndpoint &ep : endpoints) {
                    if (!ep.node || ep.sign == EdgePartType::NONE)
                        return false;
                }

                countSnarlTag(tag);

                if (endpoints.size() == 2) {
                    commitFastSnarlPairKey(pack_fast_snarl_pair_key(
                        static_cast<uint32_t>(endpoints[0].node.idx),
                        endpoint_sign_bit(endpoints[0].sign),
                        static_cast<uint32_t>(endpoints[1].node.idx),
                        endpoint_sign_bit(endpoints[1].sign)));
                    return true;
                }

                addSnarlsFound(endpoints.size() * (endpoints.size() - 1) / 2);
                std::vector<uint64_t> clique;
                clique.reserve(endpoints.size());
                for (const SnarlEndpoint &ep : endpoints) {
                    clique.push_back(pack_fast_snarl_endpoint_key(
                        static_cast<uint32_t>(ep.node.idx),
                        endpoint_sign_bit(ep.sign)));
                }

                if (tls_fast_snarl_clique_buffer) {
                    tls_fast_snarl_clique_buffer->push_back(std::move(clique));
                } else {
                    std::lock_guard<std::mutex> lk(g_snarls_mtx);
                    C.fastSnarlCliques.push_back(std::move(clique));
                }
                return true;
            }

            inline void addSnarlTaggedEndpoints(const char *tag,
                                                const std::vector<SnarlEndpoint> &endpoints)
            {
                if (tryCommitFastSnarlEndpoints(tag, endpoints))
                    return;

                std::vector<std::string> s;
                s.reserve(endpoints.size());
                auto &C = ctx();
                for (const SnarlEndpoint &ep : endpoints) {
                    if (!ep.node || ep.sign == EdgePartType::NONE)
                        continue;
                    s.push_back(C.node2name[ep.node] + endpoint_sign_char(ep.sign));
                }
                if (s.size() >= 2)
                    addSnarlTagged(tag, std::move(s));
            }

            inline std::string endpointToString(const SnarlEndpoint &ep)
            {
                auto &C = ctx();
                return C.node2name[ep.node] + endpoint_sign_char(ep.sign);
            }

            inline void addSnarlTaggedPairNodes(const char *tag,
                                                spqr_compat::node a,
                                                EdgePartType aSign,
                                                spqr_compat::node b,
                                                EdgePartType bSign)
            {
                auto &C = ctx();
                if (C.fastSnarlPairsEnabled && !debug_tagged_snarls_enabled() &&
                    a && b && aSign != EdgePartType::NONE && bSign != EdgePartType::NONE) {
                    countSnarlTag(tag);
                    commitFastSnarlPairKey(pack_fast_snarl_pair_key(
                        static_cast<uint32_t>(a.idx), endpoint_sign_bit(aSign),
                        static_cast<uint32_t>(b.idx), endpoint_sign_bit(bSign)));
                    return;
                }

                std::string s = C.node2name[a] + endpoint_sign_char(aSign);
                std::string t = C.node2name[b] + endpoint_sign_char(bSign);
                addSnarlTagged(tag, {std::move(s), std::move(t)});
            }

            inline void flushThreadLocalSnarls(std::vector<std::vector<std::string>> &local)
            {
                auto &C = ctx();
                std::lock_guard<std::mutex> lk(g_snarls_mtx);
                for (auto &s : local)
                {
                    addSnarlsFound(s.size() * (s.size() - 1) / 2);
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

            inline void flushThreadLocalFastSnarlPairs(std::vector<uint64_t> &local)
            {
                if (local.empty()) return;
                auto &C = ctx();
                std::lock_guard<std::mutex> lk(g_snarls_mtx);
                if (C.fastSnarlPairs.empty()) {
                    C.fastSnarlPairs.swap(local);
                } else {
                    C.fastSnarlPairs.insert(C.fastSnarlPairs.end(), local.begin(), local.end());
                    local.clear();
                }
            }

            inline void flushThreadLocalFastSnarlCliques(std::vector<std::vector<uint64_t>> &local)
            {
                if (local.empty()) return;
                auto &C = ctx();
                std::lock_guard<std::mutex> lk(g_snarls_mtx);
                C.fastSnarlCliques.reserve(C.fastSnarlCliques.size() + local.size());
                for (auto &clique : local) {
                    C.fastSnarlCliques.push_back(std::move(clique));
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

            inline bool hasMixedSelfLoopBetween(const SnarlEndpoint& a,
                                                const SnarlEndpoint& b)
            {
                auto &C = ctx();
                if (!a.node || a.node != b.node || a.sign == b.sign ||
                    a.sign == EdgePartType::NONE || b.sign == EdgePartType::NONE) {
                    return false;
                }
                bool found = false;
                C.G.forEachAdj(a.node, [&](spqr_compat::node, spqr_compat::edge e) {
                    if (found || C.G.source(e) != a.node || C.G.target(e) != a.node)
                        return;
                    const auto edgeTypes = C._edge2types(e);
                    found =
                        (edgeTypes.first == a.sign && edgeTypes.second == b.sign) ||
                        (edgeTypes.first == b.sign && edgeTypes.second == a.sign);
                });
                return found;
            }

            inline uint64_t endpointKey(const SnarlEndpoint& ep)
            {
                return pack_fast_snarl_endpoint_key(
                    static_cast<uint32_t>(ep.node.idx),
                    endpoint_sign_bit(ep.sign));
            }

            bool verifySmallSnarlComponentDirect(uint64_t leftKey,
                                                 uint64_t rightKey,
                                                 const size_t maxTouchedNodes)
            {
                auto &C = ctx();
                const uint32_t leftIdx = static_cast<uint32_t>(leftKey >> 1);
                const uint8_t leftSign = static_cast<uint8_t>(leftKey & 1u);
                const uint32_t rightIdx = static_cast<uint32_t>(rightKey >> 1);
                const uint8_t rightSign = static_cast<uint8_t>(rightKey & 1u);

                thread_local std::vector<uint8_t> sideMask;
                thread_local std::vector<uint32_t> touched;
                thread_local std::vector<uint64_t> st;

                auto ensureSize = [&](uint32_t idx) {
                    if (idx >= sideMask.size())
                        sideMask.resize(static_cast<size_t>(idx) + 1, 0);
                };
                ensureSize(std::max(leftIdx, rightIdx));

                auto clearTouched = [&]() {
                    for (uint32_t idx : touched)
                        sideMask[idx] = 0;
                    touched.clear();
                };

                auto setSide = [&](uint32_t idx, uint8_t sign) {
                    ensureSize(idx);
                    const uint8_t bit = static_cast<uint8_t>(1u << sign);
                    if (sideMask[idx] & bit)
                        return false;
                    if (sideMask[idx] == 0)
                        touched.push_back(idx);
                    sideMask[idx] |= bit;
                    return true;
                };

                st.clear();
                st.reserve(64);
                setSide(leftIdx, leftSign);
                st.push_back(leftKey);

                for (size_t pos = 0; pos < st.size(); ++pos)
                {
                    const uint64_t state = st[pos];
                    const uint32_t uIdx = static_cast<uint32_t>(state >> 1);
                    const uint8_t sign = static_cast<uint8_t>(state & 1u);
                    const bool boundary = (uIdx == leftIdx || uIdx == rightIdx);
                    if (!boundary)
                    {
                        const uint8_t otherSign = static_cast<uint8_t>(sign ^ 1u);
                        if (setSide(uIdx, otherSign))
                            st.push_back((static_cast<uint64_t>(uIdx) << 1) | otherSign);
                    }

                    spqr_compat::node u{uIdx};
                    C.G.forEachAdj(u, [&](spqr_compat::node other, spqr_compat::edge e) {
                        const auto edgeTypes = C._edge2types(e);
                        auto pushIfMatches = [&](uint32_t toIdx, uint8_t outSign, uint8_t inSign) {
                            if (outSign != sign)
                                return;
                            if (setSide(toIdx, inSign))
                                st.push_back((static_cast<uint64_t>(toIdx) << 1) | inSign);
                        };

                        if (C.G.source(e) == u && C.G.target(e) == u)
                        {
                            const uint8_t s0 = endpoint_sign_bit(edgeTypes.first);
                            const uint8_t s1 = endpoint_sign_bit(edgeTypes.second);
                            pushIfMatches(uIdx, s0, s1);
                            pushIfMatches(uIdx, s1, s0);
                        }
                        else if (C.G.source(e) == u)
                        {
                            pushIfMatches(static_cast<uint32_t>(other.idx),
                                          endpoint_sign_bit(edgeTypes.first),
                                          endpoint_sign_bit(edgeTypes.second));
                        }
                        else
                        {
                            pushIfMatches(static_cast<uint32_t>(other.idx),
                                          endpoint_sign_bit(edgeTypes.second),
                                          endpoint_sign_bit(edgeTypes.first));
                        }
                    });

                    if (touched.size() > maxTouchedNodes)
                    {
                        clearTouched();
                        return false;
                    }
                }

                const bool reachesRight =
                    rightIdx < sideMask.size() &&
                    (sideMask[rightIdx] & static_cast<uint8_t>(1u << rightSign)) != 0;
                const bool reachesLeftOpp =
                    leftIdx < sideMask.size() &&
                    (sideMask[leftIdx] & static_cast<uint8_t>(1u << (leftSign ^ 1u))) != 0;
                const bool reachesRightOpp =
                    rightIdx < sideMask.size() &&
                    (sideMask[rightIdx] & static_cast<uint8_t>(1u << (rightSign ^ 1u))) != 0;
                clearTouched();
                return reachesRight && !reachesLeftOpp && !reachesRightOpp;
            }

            template <typename EmitFn>
            inline uint32_t emitMixedSelfLoopComposedS(const std::vector<CutSides>& cuts,
                                                       bool cyclic,
                                                       const EmitFn& emit)
            {
                const size_t n = cuts.size();
                if (n < 3)
                    return 0;

                uint32_t emitted = 0;
                auto try_emit = [&](size_t i) {
                    if (!hasMixedSelfLoopBetween(cuts[i].prev_side, cuts[i].next_side))
                        return;
                    const SnarlEndpoint& first = cuts[(i + n - 1) % n].next_side;
                    const SnarlEndpoint& second = cuts[(i + 1) % n].prev_side;
                    if (!first.node || !second.node || first.node == second.node)
                        return;
                    if (!verifySmallSnarlComponentDirect(endpointKey(first),
                                                         endpointKey(second),
                                                         1024))
                        return;
                    emit(first, second);
                    ++emitted;
                };

                if (cyclic) {
                    for (size_t i = 0; i < n; ++i)
                        try_emit(i);
                } else {
                    for (size_t i = 1; i + 1 < n; ++i)
                        try_emit(i);
                }
                return emitted;
            }

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

        
            #ifdef BUBBLEFINDER_INSTRUMENT
                        namespace profiling_patch
                        {
                            std::atomic<uint64_t> blocks_with_spqr{0};
                            std::atomic<uint64_t> total_tree_nodes{0};

                            std::atomic<uint64_t> phase1_edge_dp_time_ns{0};
                            std::atomic<uint64_t> phase2_node_dp_time_ns{0};
                            std::atomic<uint64_t> phase3_solve_time_ns{0};
                            std::atomic<uint64_t> phase4_caseE_time_ns{0};
                            std::atomic<uint64_t> phase1_calls{0};
                            std::atomic<uint64_t> phase2_calls{0};
                            std::atomic<uint64_t> phase3_calls{0};
                            std::atomic<uint64_t> phase4_calls{0};

                            std::atomic<uint64_t> p3_solveS_ns{0};
                            std::atomic<uint64_t> p3_solveP_ns{0};
                            std::atomic<uint64_t> p3_solveRR_ns{0};
                            std::atomic<uint64_t> p3_solveS_calls{0};
                            std::atomic<uint64_t> p3_solveP_calls{0};
                            std::atomic<uint64_t> p3_solveRR_calls{0};

                            std::atomic<uint64_t> pn_total_calls{0};
                            std::atomic<uint64_t> pn_fastpath_calls{0};
                            std::atomic<uint64_t> pn_A_setup_ns{0};
                            std::atomic<uint64_t> pn_B_build_ns{0};
                            std::atomic<uint64_t> pn_C_propagate_ns{0};

                            std::atomic<uint64_t> pe_total_calls{0};
                            std::atomic<uint64_t> pe_A_setup_ns{0};
                            std::atomic<uint64_t> pe_B_build_ns{0};

                            std::atomic<uint64_t> sub_alloc_dp_ns{0};
                            std::atomic<uint64_t> sub_dfs_order_ns{0};
                            std::atomic<uint64_t> sub_blktoskel_init_ns{0};
                            std::atomic<uint64_t> sub_phase4_setup_ns{0};
                            std::atomic<uint64_t> sub_phase4_caseE_body_ns{0};
                            std::atomic<uint64_t> sub_levels_ns{0};
                            std::atomic<uint64_t> sub_destruct_ns{0};
                            std::atomic<uint64_t> sub_solveS_collect_ns{0};
                            std::atomic<uint64_t> sub_caseE_collect_ns{0};

                            std::atomic<uint64_t> taskloops_S_created{0};
                            std::atomic<uint64_t> taskloops_P_created{0};
                            std::atomic<uint64_t> taskloops_caseE_created{0};
                            std::atomic<uint64_t> tasks_requested_S_total{0};
                            std::atomic<uint64_t> tasks_requested_P_total{0};
                            std::atomic<uint64_t> tasks_requested_caseE_total{0};

                            std::atomic<uint64_t> taskloops_S_skipped{0};
                            std::atomic<uint64_t> taskloops_P_skipped{0};
                            std::atomic<uint64_t> taskloops_caseE_skipped{0};

                            struct BlockTiming
                            {
                                size_t bid = static_cast<size_t>(-1);
                                uint64_t logicWeight = 0;
                                uint64_t blockNodes = 0;
                                uint64_t blockEdges = 0;
                                uint64_t spqrTreeNodes = 0;
                                uint32_t spqrSCount = 0;
                                uint32_t spqrPCount = 0;
                                uint32_t spqrRCount = 0;
                                uint32_t maxDepth = 0;
                                uint32_t maxHeight = 0;
                                uint64_t build_ns = 0;
                                uint64_t solve_total_ns = 0;
                                uint64_t sub_alloc_dp_ns = 0;
                                uint64_t sub_dfs_order_ns = 0;
                                uint64_t sub_blktoskel_init_ns = 0;
                                uint64_t sub_levels_ns = 0;
                                uint64_t phase1_ns = 0;
                                uint64_t phase2_ns = 0;
                                uint64_t phase3_ns = 0;
                                uint64_t phase3_S_ns = 0;
                                uint64_t phase3_P_ns = 0;
                                uint64_t phase3_RR_ns = 0;
                                uint64_t phase3_S_collect_ns = 0;
                                uint64_t phase4_setup_ns = 0;
                                uint64_t phase4_caseE_collect_ns = 0;
                                uint64_t phase4_caseE_body_ns = 0;
                                uint64_t destruct_ns = 0;
                                bool critical = false;
                                uint32_t taskloops_S_created = 0;
                                uint32_t taskloops_P_created = 0;
                                uint32_t taskloops_caseE_created = 0;
                                uint64_t tasks_requested_S = 0;
                                uint64_t tasks_requested_P = 0;
                                uint64_t tasks_requested_caseE = 0;
                                std::vector<uint64_t> widthByDepth;
                                std::vector<uint64_t> widthByHeight;
                            };

                            inline std::vector<BlockTiming>& g_block_timings()
                            {
                                static std::vector<BlockTiming> v;
                                return v;
                            }

                            inline std::mutex& g_block_timings_resize_mutex()
                            {
                                static std::mutex m;
                                return m;
                            }

                            inline void reset_block_timings(size_t n)
                            {
                                std::lock_guard<std::mutex> lk(g_block_timings_resize_mutex());
                                auto &v = g_block_timings();
                                v.clear();
                                v.resize(n);
                                for (size_t i = 0; i < n; ++i) v[i].bid = i;
                            }

                            inline BlockTiming* try_get_block_timing(size_t bid)
                            {
                                auto &v = g_block_timings();
                                if (bid >= v.size()) return nullptr;
                                return &v[bid];
                            }
                        }
            #endif

        }

        static void tryCommitSnarl(std::vector<std::string> s)
        {
            auto &C = ctx();
            // PROFILE_BLOCK("tryCommitSnarl");

            // if(std::count(s[0].begin(), s[0].end(), ':') == 0) {
            std::lock_guard<std::mutex> lk(g_snarls_mtx);

            addSnarlsFound(s.size() * (s.size() - 1) / 2);
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
            countSnarlTag(tag);

            canonicalize_pair(s);

            if (const char *taggedPath = std::getenv("BF_DEBUG_TAGGED_SNARLS"))
            {
                if (*taggedPath)
                {
                    static std::mutex tagged_mtx;
                    static std::ofstream tagged_out;
                    static std::string tagged_open_path;
                    std::lock_guard<std::mutex> lk(tagged_mtx);
                    if (!tagged_out.is_open() || tagged_open_path != taggedPath)
                    {
                        tagged_out.close();
                        tagged_out.open(taggedPath);
                        tagged_open_path = taggedPath;
                    }
                    if (tagged_out)
                    {
                        tagged_out << (tag ? tag : "?");
                        for (const auto &x : s)
                        {
                            tagged_out << '\t' << x;
                        }
                        tagged_out << '\n';
                    }
                }
            }

            if (tryCommitFastSnarlPairs(s)) {
                return;
            }

            addSnarl(std::move(s));
        }

        struct SpCompressHandleDeleter
        {
            void operator()(SpCompressHandle *h) const
            {
                if (h)
                {
                    sp_compress_free(h);
                }
            }
        };

        struct BlockData
        {
            std::unique_ptr<spqr_compat::Graph> Gblk;
            spqr_compat::NodeArray<spqr_compat::node> toCc;
            spqr_compat::NodeArray<spqr_compat::node> nodeToOrig;
            spqr_compat::EdgeArray<spqr_compat::edge> edgeToOrig;
            std::vector<spqr_compat::node> flatNodeToOrig;
            std::vector<spqr_compat::node> flatNodeToCc;
            std::vector<spqr_compat::edge> flatEdgeToOrig;
            std::vector<SpCompressInputEdge> flatFfiEdges;
            std::vector<uint32_t> flatEdgeSrc;
            std::vector<uint32_t> flatEdgeDst;
            std::vector<int> flatDegPlus;
            std::vector<int> flatDegMinus;
            uint32_t flatEdgeCount{0};
            bool flatNodes{false};
            bool flatEdges{false};

            std::unique_ptr<spqr_compat::StaticSPQRTree> spqr;
            std::unique_ptr<SpCompressHandle, SpCompressHandleDeleter> spCompressHandle;
            SpCompressTreeView macroTreeView{};
            const SpqrTree *coreSpqrTree{nullptr};
            const uint32_t *coreNodeInv{nullptr};
            uint32_t coreNodeInvLen{0};
            spqr_compat::NodeArray<spqr_compat::node> blkToSkel;

            std::unordered_map<spqr_compat::edge, spqr_compat::edge> skel2tree;
            spqr_compat::NodeArray<spqr_compat::node> parent;

            spqr_compat::node bNode{nullptr};

            bool isAcycic{true};

            spqr_compat::NodeArray<int> blkDegPlus;
            spqr_compat::NodeArray<int> blkDegMinus;

            BlockData() {}
        };

        struct CcData
        {
            std::unique_ptr<spqr_compat::Graph> Gcc;
            spqr_compat::NodeArray<spqr_compat::node> nodeToOrig;
            spqr_compat::EdgeArray<spqr_compat::edge> edgeToOrig;

            spqr_compat::NodeArray<bool> isTip;

            spqr_compat::NodeArray<int> degPlus, degMinus;

            spqr_compat::NodeArray<bool> isCutNode;
            spqr_compat::NodeArray<bool> isGoodCutNode;

            spqr_compat::NodeArray<spqr_compat::node> lastBad;
            spqr_compat::NodeArray<int> badCutCount;   

            spqr_compat::EdgeArray<spqr_compat::edge> auxToOriginal;
            // spqr_compat::NodeArray<std::array<std::vector<spqr_compat::node>, 3>> cutToBlocks; // 0-all -, 1 - all +, 2 - mixed

            // spqr_compat::NodeArray<spqr_compat::node> toCopy;
            // spqr_compat::NodeArray<spqr_compat::node> toBlk;

            std::unique_ptr<spqr_compat::BCTree> bc;
            std::vector<BlockData> blocks;
        };

        inline size_t blockNodeCount(const BlockData &blk)
        {
            return blk.flatNodes ? blk.flatNodeToOrig.size()
                                 : (blk.Gblk ? blk.Gblk->numberOfNodes() : 0);
        }

        inline spqr_compat::node blockNodeToOrig(const BlockData &blk, spqr_compat::node v)
        {
            return blk.flatNodes ? blk.flatNodeToOrig[v.idx] : blk.nodeToOrig[v];
        }

        inline spqr_compat::node &blockNodeToOrig(BlockData &blk, spqr_compat::node v)
        {
            return blk.flatNodes ? blk.flatNodeToOrig[v.idx] : blk.nodeToOrig[v];
        }

        inline spqr_compat::node blockNodeToCc(const BlockData &blk, spqr_compat::node v)
        {
            return blk.flatNodes ? blk.flatNodeToCc[v.idx] : blk.toCc[v];
        }

        inline spqr_compat::node &blockNodeToCc(BlockData &blk, spqr_compat::node v)
        {
            return blk.flatNodes ? blk.flatNodeToCc[v.idx] : blk.toCc[v];
        }

        inline size_t blockEdgeCount(const BlockData &blk)
        {
            return blk.flatEdges ? blk.flatEdgeCount
                                 : (blk.Gblk ? blk.Gblk->numberOfEdges() : 0);
        }

        inline spqr_compat::node blockEdgeSource(const BlockData &blk, spqr_compat::edge e)
        {
            if (!blk.flatEdges) return blk.Gblk->source(e);
            if (!blk.flatFfiEdges.empty()) return spqr_compat::node{blk.flatFfiEdges[e.idx].u};
            return spqr_compat::node{blk.macroTreeView.input_endpoints_ptr[2 * e.idx]};
        }

        inline spqr_compat::node blockEdgeTarget(const BlockData &blk, spqr_compat::edge e)
        {
            if (!blk.flatEdges) return blk.Gblk->target(e);
            if (!blk.flatFfiEdges.empty()) return spqr_compat::node{blk.flatFfiEdges[e.idx].v};
            return spqr_compat::node{blk.macroTreeView.input_endpoints_ptr[2 * e.idx + 1]};
        }

        inline spqr_compat::edge blockEdgeToOrig(const BlockData &blk, spqr_compat::edge e)
        {
            return blk.flatEdges ? blk.flatEdgeToOrig[e.idx] : blk.edgeToOrig[e];
        }

        inline int blockDegPlus(const BlockData &blk, spqr_compat::node v)
        {
            return blk.flatEdges ? blk.flatDegPlus[v.idx] : blk.blkDegPlus[v];
        }

        inline int blockDegMinus(const BlockData &blk, spqr_compat::node v)
        {
            return blk.flatEdges ? blk.flatDegMinus[v.idx] : blk.blkDegMinus[v];
        }

        inline std::vector<spqr_compat::edge> collectBlockEdges(const BlockData &blk)
        {
            std::vector<spqr_compat::edge> edges;
            edges.reserve(blockEdgeCount(blk));
            if (blk.flatEdges)
            {
                for (uint32_t i = 0; i < blk.flatEdgeCount; ++i)
                    edges.push_back(spqr_compat::edge{i});
            }
            else if (blk.Gblk)
            {
                for (spqr_compat::edge e : blk.Gblk->edges)
                    edges.push_back(e);
            }
            return edges;
        }

        EdgePartType getNodeEdgeType(spqr_compat::node v, spqr_compat::edge e)
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

        inline void addIncidentTypeCount(int &plusCnt, int &minusCnt, EdgePartType t)
        {
            if (t == EdgePartType::PLUS)
                plusCnt++;
            else if (t == EdgePartType::MINUS)
                minusCnt++;
        }

        inline void addIncidentTypeCount(spqr_compat::NodeArray<int> &plusCnt,
                                         spqr_compat::NodeArray<int> &minusCnt,
                                         spqr_compat::node v,
                                         EdgePartType t)
        {
            if (t == EdgePartType::PLUS)
                plusCnt[v]++;
            else if (t == EdgePartType::MINUS)
                minusCnt[v]++;
        }

        void getOutgoingEdgesInBlock(const CcData &cc,
                                     spqr_compat::node uG, 
                                     spqr_compat::node vB, 
                                     EdgePartType type,
                                     std::vector<spqr_compat::edge> &outEdges)
        {
            outEdges.clear();

            for (spqr_compat::edge eCc : cc.bc->hEdges(vB))
            {
                if (cc.Gcc->source(eCc) != uG && cc.Gcc->target(eCc) != uG)
                    continue;

                spqr_compat::edge eG = cc.edgeToOrig[eCc];
                if (!eG) continue;

                auto outType = getNodeEdgeType(cc.nodeToOrig[uG], eG);
                if (outType == type)
                {
                    outEdges.push_back(eCc);
                }
            }
        }

        void getAllOutgoingEdgesOfType(const CcData &cc, spqr_compat::node uG, EdgePartType type, std::vector<spqr_compat::adjEntry> &outEdges)
        {
            outEdges.clear();

            cc.Gcc->forEachAdj(uG, [&](node neighbor, edge eC) {
                spqr_compat::edge eOrig = cc.edgeToOrig[eC];

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
                std::vector<spqr_compat::node> GccCuts_last3;
            };

            inline std::vector<EdgeDPState> compute_all_macro_dp_states(
                const SpCompressTreeView &m_view,
                BlockData &blk)
            {
                const uint32_t M = m_view.macros_len;
                std::vector<EdgeDPState> states(M);

                if (M == 0) return states;

                auto &C = ctx();

                for (uint32_t m_id = 0; m_id < M; ++m_id) {
                    const SpCompressNode &m = m_view.macros_ptr[m_id];
                    EdgeDPState &state = states[m_id];
                    state.s = spqr_compat::node{m.left};
                    state.t = spqr_compat::node{m.right};
                    state.localPlusS = 0;
                    state.localPlusT = 0;
                    state.localMinusS = 0;
                    state.localMinusT = 0;

                    spqr_compat::node gS = blockNodeToOrig(blk, state.s);
                    spqr_compat::node gT = blockNodeToOrig(blk, state.t);

                    for (uint32_t i = 0; i < m.children_count; ++i) {
                        uint32_t child_ref = m_view.children_ptr[m.children_offset + i];

                        if (SP_COMPRESS_CHILD_IS_EDGE(child_ref)) {
                            uint32_t block_edge_id = SP_COMPRESS_CHILD_AS_EDGE(child_ref);
                            spqr_compat::edge eBlk{block_edge_id};
                            spqr_compat::edge eG = blockEdgeToOrig(blk, eBlk);
                            spqr_compat::node uG = C.G.source(eG);
                            spqr_compat::node vG = C.G.target(eG);

                            if (uG == gS) {
                                auto t = getNodeEdgeType(uG, eG);
                                if (t == EdgePartType::PLUS) state.localPlusS++;
                                else if (t == EdgePartType::MINUS) state.localMinusS++;
                            }
                            if (vG == gS) {
                                auto t = getNodeEdgeType(vG, eG);
                                if (t == EdgePartType::PLUS) state.localPlusS++;
                                else if (t == EdgePartType::MINUS) state.localMinusS++;
                            }
                            if (uG == gT) {
                                auto t = getNodeEdgeType(uG, eG);
                                if (t == EdgePartType::PLUS) state.localPlusT++;
                                else if (t == EdgePartType::MINUS) state.localMinusT++;
                            }
                            if (vG == gT) {
                                auto t = getNodeEdgeType(vG, eG);
                                if (t == EdgePartType::PLUS) state.localPlusT++;
                                else if (t == EdgePartType::MINUS) state.localMinusT++;
                            }
                        } else {
                            uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(child_ref);
                            const EdgeDPState &sub = states[sub_id];
                            if (sub.s == state.s) {
                                state.localPlusS += sub.localPlusS;
                                state.localMinusS += sub.localMinusS;
                            }
                            if (sub.s == state.t) {
                                state.localPlusT += sub.localPlusS;
                                state.localMinusT += sub.localMinusS;
                            }
                            if (sub.t == state.s) {
                                state.localPlusS += sub.localPlusT;
                                state.localMinusS += sub.localMinusT;
                            }
                            if (sub.t == state.t) {
                                state.localPlusT += sub.localPlusT;
                                state.localMinusT += sub.localMinusT;
                            }
                        }
                    }
                }

                return states;
            }

            inline std::vector<spqr_compat::node> compute_macro_series_GccCuts_last3_one(
                const std::vector<EdgeDPState>& macro_states,
                const SpCompressTreeView& m_view,
                BlockData& blk,
                const CcData& cc,
                uint32_t m_id)
            {
                std::vector<spqr_compat::node> cuts;
                if (m_id >= m_view.macros_len) return cuts;

                const SpCompressNode& m = m_view.macros_ptr[m_id];
                if (m.kind != SP_COMPRESS_KIND_SERIES) return cuts;

                std::vector<std::pair<uint32_t, uint32_t>> incidence;
                incidence.reserve(static_cast<size_t>(m.children_count) * 2);

                for (uint32_t i = 0; i < m.children_count; ++i) {
                    uint32_t cref = m_view.children_ptr[m.children_offset + i];
                    uint32_t e0 = SPQR_INVALID;
                    uint32_t e1 = SPQR_INVALID;
                    if (SP_COMPRESS_CHILD_IS_EDGE(cref)) {
                        uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cref);
                        spqr_compat::edge eBlk{bid};
                        e0 = blockEdgeSource(blk, eBlk).idx;
                        e1 = blockEdgeTarget(blk, eBlk).idx;
                    } else {
                        uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                        if (sub_id >= m_view.macros_len) continue;
                        e0 = m_view.macros_ptr[sub_id].left;
                        e1 = m_view.macros_ptr[sub_id].right;
                    }
                    if (e0 != m.left && e0 != m.right) incidence.emplace_back(e0, i);
                    if (e1 != m.left && e1 != m.right) incidence.emplace_back(e1, i);
                }

                std::sort(incidence.begin(), incidence.end());

                for (size_t k = 0; k < incidence.size(); ) {
                    uint32_t cur_node = incidence[k].first;
                    uint32_t ch0 = incidence[k].second;
                    ++k;
                    uint32_t ch1 = SPQR_INVALID;
                    if (k < incidence.size() && incidence[k].first == cur_node) {
                        ch1 = incidence[k].second;
                        ++k;
                        while (k < incidence.size() && incidence[k].first == cur_node) ++k;
                    }
                    if (ch1 == SPQR_INVALID) continue;

                    spqr_compat::node uBlk{cur_node};
                    spqr_compat::node uG = blockNodeToOrig(blk, uBlk);
                    spqr_compat::node uGcc = blockNodeToCc(blk, uBlk);
                    if (!uGcc) continue;

                    bool nodeIsCut =
                        (cc.isCutNode[uGcc] && cc.badCutCount[uGcc] == 1) ||
                        (!cc.isCutNode[uGcc]);
                    if (!nodeIsCut) continue;

                    auto orient_at = [&](uint32_t child_idx) -> EdgePartType {
                        uint32_t cref = m_view.children_ptr[m.children_offset + child_idx];
                        if (SP_COMPRESS_CHILD_IS_EDGE(cref)) {
                            uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cref);
                            spqr_compat::edge eBlk{bid};
                            spqr_compat::edge eG = blockEdgeToOrig(blk, eBlk);
                            return getNodeEdgeType(uG, eG);
                        }

                        uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                        if (sub_id >= macro_states.size()) return EdgePartType::NONE;
                        const EdgeDPState& sub = macro_states[sub_id];
                        if (sub.s == uBlk) {
                            if (sub.localPlusS > 0 && sub.localMinusS == 0) return EdgePartType::PLUS;
                            if (sub.localPlusS == 0 && sub.localMinusS > 0) return EdgePartType::MINUS;
                        } else if (sub.t == uBlk) {
                            if (sub.localPlusT > 0 && sub.localMinusT == 0) return EdgePartType::PLUS;
                            if (sub.localPlusT == 0 && sub.localMinusT > 0) return EdgePartType::MINUS;
                        }
                        return EdgePartType::NONE;
                    };

                    EdgePartType t0 = orient_at(ch0);
                    EdgePartType t1 = orient_at(ch1);
                    if (t0 != EdgePartType::NONE && t1 != EdgePartType::NONE && t0 != t1) {
                        cuts.push_back(uGcc);
                        if (cuts.size() >= 3) break;
                    }
                }

                return cuts;
            }

            struct MacroSeriesGccCutsCache {
                const std::vector<EdgeDPState>& macro_states;
                const SpCompressTreeView& m_view;
                BlockData& blk;
                const CcData& cc;
                std::vector<std::vector<spqr_compat::node>> cuts;
                std::vector<uint8_t> computed;
                uint64_t requests = 0;
                uint64_t computes = 0;
                uint64_t compute_us = 0;

                MacroSeriesGccCutsCache(
                    const std::vector<EdgeDPState>& states,
                    const SpCompressTreeView& view,
                    BlockData& block,
                    const CcData& component)
                    : macro_states(states),
                      m_view(view),
                      blk(block),
                      cc(component),
                      cuts(view.macros_len),
                      computed(view.macros_len, 0)
                {}

                const std::vector<spqr_compat::node>& get(uint32_t m_id) {
                    static const std::vector<spqr_compat::node> empty;
                    ++requests;
                    if (m_id >= cuts.size()) return empty;
                    if (!computed[m_id]) {
                        auto t0 = std::chrono::steady_clock::now();
                        cuts[m_id] = compute_macro_series_GccCuts_last3_one(
                            macro_states, m_view, blk, cc, m_id);
                        auto t1 = std::chrono::steady_clock::now();
                        compute_us += static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                t1 - t0).count());
                        computed[m_id] = 1;
                        ++computes;
                    }
                    return cuts[m_id];
                }
            };

            inline uint32_t emit_macro_snarls_parallel(
                uint32_t macro_id,
                const std::vector<EdgeDPState> &states,
                const SpCompressTreeView &m_view,
                BlockData &blk,
                const CcData &cc,
	                std::vector<SnarlEndpointPair> *out_snarls,
	                MacroSeriesGccCutsCache* macro_gcc_cuts = nullptr,
                const std::vector<EdgeDPState>* macro_down_states = nullptr,
                const std::vector<uint8_t>* macro_has_down_state = nullptr,
                const std::vector<uint8_t>* macro_down_gcc_cut_count = nullptr,
                const std::vector<std::array<int32_t, 3>>* macro_down_gcc_cut_idx = nullptr,
                const std::vector<uint32_t>* parent_macro = nullptr,
                const std::vector<uint32_t>* parent_child_idx = nullptr)
            {
                const SpCompressNode &m = m_view.macros_ptr[macro_id];
                if (m.kind != SP_COMPRESS_KIND_PARALLEL) return 0;

                auto &C = ctx();

                spqr_compat::node pole0Blk{m.left};
                spqr_compat::node pole1Blk{m.right};

                spqr_compat::node pole0G = blockNodeToOrig(blk, pole0Blk);
                spqr_compat::node pole1G = blockNodeToOrig(blk, pole1Blk);

                spqr_compat::node pole0Gcc = blockNodeToCc(blk, pole0Blk);
                spqr_compat::node pole1Gcc = blockNodeToCc(blk, pole1Blk);

                auto hasDanglingOutside = [&](spqr_compat::node vGcc) {
                    if (!vGcc) return false;
                    if (!cc.isCutNode[vGcc]) return false;
                    if (cc.badCutCount[vGcc] >= 2) return true;
                    if (cc.badCutCount[vGcc] == 1 && cc.lastBad[vGcc] != blk.bNode) return true;
                    return false;
                };
                if (hasDanglingOutside(pole0Gcc) || hasDanglingOutside(pole1Gcc)) {
                    return 0;
                }

                struct ChildOri {
                    EdgePartType lSign;  // at pole0
                    EdgePartType rSign;  // at pole1
                    bool ambiguous_pole0;  // child has both signs at pole0
                    bool ambiguous_pole1;
                };

                auto sign_at = [](int plus, int minus, bool &ambiguous) -> EdgePartType {
                    bool hp = plus > 0, hm = minus > 0;
                    if (hp && hm) { ambiguous = true; return EdgePartType::NONE; }
                    if (hp) return EdgePartType::PLUS;
                    if (hm) return EdgePartType::MINUS;
                    return EdgePartType::NONE;
                };

                auto compute_ori_from_state = [&](const EdgeDPState &st) -> ChildOri {
                    ChildOri o{EdgePartType::NONE, EdgePartType::NONE, false, false};

                    if (st.s == pole0Blk) {
                        o.lSign = sign_at(st.localPlusS, st.localMinusS, o.ambiguous_pole0);
                    } else if (st.t == pole0Blk) {
                        o.lSign = sign_at(st.localPlusT, st.localMinusT, o.ambiguous_pole0);
                    }

                    if (st.s == pole1Blk) {
                        o.rSign = sign_at(st.localPlusS, st.localMinusS, o.ambiguous_pole1);
                    } else if (st.t == pole1Blk) {
                        o.rSign = sign_at(st.localPlusT, st.localMinusT, o.ambiguous_pole1);
                    }

                    return o;
                };

                auto compute_ori = [&](uint32_t cref) -> ChildOri {
                    ChildOri o{EdgePartType::NONE, EdgePartType::NONE, false, false};

                    if (SP_COMPRESS_CHILD_IS_EDGE(cref)) {
                        uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cref);
                        spqr_compat::edge eBlk{bid};
                        spqr_compat::edge eG = blockEdgeToOrig(blk, eBlk);

                        o.lSign = getNodeEdgeType(pole0G, eG);
                        o.rSign = getNodeEdgeType(pole1G, eG);
                    } else {
                        uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                        const EdgeDPState &st = states[sub_id];
                        o = compute_ori_from_state(st);
                    }
                    return o;
                };

                const bool has_down_child =
                    macro_down_states && macro_has_down_state &&
                    macro_id < macro_down_states->size() &&
                    macro_id < macro_has_down_state->size() &&
                    (*macro_has_down_state)[macro_id] != 0;

                EdgeDPState series_context_state;
                bool has_series_context_child = false;
                if (!has_down_child && parent_macro && parent_child_idx &&
                    macro_id < parent_macro->size() &&
                    macro_id < parent_child_idx->size()) {
                    const uint32_t parent_id = (*parent_macro)[macro_id];
                    const uint32_t child_idx = (*parent_child_idx)[macro_id];
                    if (parent_id != SPQR_INVALID && parent_id < m_view.macros_len) {
                        const SpCompressNode& parent = m_view.macros_ptr[parent_id];
                        if (parent.kind == SP_COMPRESS_KIND_SERIES &&
                            child_idx < parent.children_count) {
                            series_context_state.s = pole0Blk;
                            series_context_state.t = pole1Blk;
                            series_context_state.localPlusS = 0;
                            series_context_state.localMinusS = 0;
                            series_context_state.localPlusT = 0;
                            series_context_state.localMinusT = 0;
                            auto merge_context_ref = [&](uint32_t cref) {
                                if (SP_COMPRESS_CHILD_IS_EDGE(cref)) {
                                    uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cref);
                                    spqr_compat::edge eBlk{bid};
                                    spqr_compat::edge eG = blockEdgeToOrig(blk, eBlk);
                                    spqr_compat::node uG = C.G.source(eG);
                                    spqr_compat::node vG = C.G.target(eG);
                                    if (uG == pole0G || vG == pole0G) {
                                        auto t = getNodeEdgeType(pole0G, eG);
                                        if (t == EdgePartType::PLUS) series_context_state.localPlusS++;
                                        else if (t == EdgePartType::MINUS) series_context_state.localMinusS++;
                                    }
                                    if (uG == pole1G || vG == pole1G) {
                                        auto t = getNodeEdgeType(pole1G, eG);
                                        if (t == EdgePartType::PLUS) series_context_state.localPlusT++;
                                        else if (t == EdgePartType::MINUS) series_context_state.localMinusT++;
                                    }
                                    return;
                                }

                                uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                                if (sub_id >= states.size()) return;
                                const EdgeDPState& st = states[sub_id];
                                auto merge_side = [&](spqr_compat::node pole, bool left_side) {
                                    int plus = 0;
                                    int minus = 0;
                                    if (st.s == pole) {
                                        plus = st.localPlusS;
                                        minus = st.localMinusS;
                                    } else if (st.t == pole) {
                                        plus = st.localPlusT;
                                        minus = st.localMinusT;
                                    }
                                    if (left_side) {
                                        series_context_state.localPlusS += plus;
                                        series_context_state.localMinusS += minus;
                                    } else {
                                        series_context_state.localPlusT += plus;
                                        series_context_state.localMinusT += minus;
                                    }
                                };
                                merge_side(pole0Blk, true);
                                merge_side(pole1Blk, false);
                            };

                            uint32_t prev_ref_path = SPQR_INVALID;
                            uint32_t next_ref_path = SPQR_INVALID;
                            bool path_ok = false;
                            {
                                const uint32_t k = parent.children_count;
                                struct PathChild {
                                    uint32_t cref{SPQR_INVALID};
                                    uint32_t a{SPQR_INVALID};
                                    uint32_t b{SPQR_INVALID};
                                };
                                std::vector<PathChild> ch_local;
                                std::vector<std::pair<uint32_t, uint32_t>> incidence;
                                ch_local.reserve(k);
                                incidence.reserve(static_cast<size_t>(k) * 2);

                                bool valid = true;
                                for (uint32_t i = 0; i < k; ++i) {
                                    uint32_t cr2 =
                                        m_view.children_ptr[parent.children_offset + i];
                                    uint32_t a = SPQR_INVALID;
                                    uint32_t b = SPQR_INVALID;
                                    if (SP_COMPRESS_CHILD_IS_EDGE(cr2)) {
                                        uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cr2);
                                        spqr_compat::edge eBlk{bid};
                                        a = blockEdgeSource(blk, eBlk).idx;
                                        b = blockEdgeTarget(blk, eBlk).idx;
                                    } else {
                                        uint32_t mid = SP_COMPRESS_CHILD_AS_MACRO(cr2);
                                        if (mid < m_view.macros_len) {
                                            a = m_view.macros_ptr[mid].left;
                                            b = m_view.macros_ptr[mid].right;
                                        }
                                    }
                                    if (a == SPQR_INVALID || b == SPQR_INVALID) {
                                        valid = false;
                                        break;
                                    }
                                    ch_local.push_back(PathChild{cr2, a, b});
                                    incidence.emplace_back(a, i);
                                    incidence.emplace_back(b, i);
                                }

                                if (valid) {
                                    std::sort(incidence.begin(), incidence.end());
                                    std::vector<uint8_t> visited(k, 0);
                                    uint32_t current = parent.left;
                                    uint32_t prev_idx = SPQR_INVALID;

                                    for (uint32_t step = 0; step < k && valid; ++step) {
                                        auto it = std::lower_bound(
                                            incidence.begin(), incidence.end(), current,
                                            [](const std::pair<uint32_t, uint32_t>& p,
                                               uint32_t v) {
                                                return p.first < v;
                                            });
                                        uint32_t chosen = SPQR_INVALID;
                                        for (; it != incidence.end() && it->first == current;
                                             ++it) {
                                            uint32_t cidx = it->second;
                                            if (cidx == prev_idx) continue;
                                            if (cidx >= k || visited[cidx]) continue;
                                            chosen = cidx;
                                            break;
                                        }
                                        if (chosen == SPQR_INVALID) {
                                            valid = false;
                                            break;
                                        }

                                        const PathChild& ch = ch_local[chosen];
                                        uint32_t next =
                                            (ch.a == current) ? ch.b :
                                            ((ch.b == current) ? ch.a : SPQR_INVALID);
                                        if (next == SPQR_INVALID) {
                                            valid = false;
                                            break;
                                        }

                                        if (chosen == child_idx) {
                                            if (prev_idx != SPQR_INVALID) {
                                                prev_ref_path = ch_local[prev_idx].cref;
                                            }

                                            visited[chosen] = 1;
                                            auto it2 = std::lower_bound(
                                                incidence.begin(), incidence.end(), next,
                                                [](const std::pair<uint32_t, uint32_t>& p,
                                                   uint32_t v) {
                                                    return p.first < v;
                                                });
                                            for (; it2 != incidence.end() && it2->first == next;
                                                 ++it2) {
                                                uint32_t cidx2 = it2->second;
                                                if (cidx2 == chosen) continue;
                                                if (cidx2 >= k || visited[cidx2]) continue;
                                                next_ref_path = ch_local[cidx2].cref;
                                                break;
                                            }
                                            path_ok = true;
                                            break;
                                        }

                                        visited[chosen] = 1;
                                        prev_idx = chosen;
                                        current = next;
                                    }
                                }
                            }

                            bool added = false;
                            if (path_ok) {
                                if (prev_ref_path != SPQR_INVALID) {
                                    merge_context_ref(prev_ref_path);
                                    added = true;
                                }
                                if (next_ref_path != SPQR_INVALID) {
                                    merge_context_ref(next_ref_path);
                                    added = true;
                                }
                            } else {
                                if (child_idx > 0) {
                                    uint32_t prev_ref =
                                        m_view.children_ptr[parent.children_offset + child_idx - 1];
                                    merge_context_ref(prev_ref);
                                    added = true;
                                }
                                if (child_idx + 1 < parent.children_count) {
                                    uint32_t next_ref =
                                        m_view.children_ptr[parent.children_offset + child_idx + 1];
                                    merge_context_ref(next_ref);
                                    added = true;
                                }
                            }
                            has_series_context_child = added;
                        }
                    }
                }

                const uint32_t n_eff =
                    m.children_count +
                    ((has_down_child || has_series_context_child) ? 1u : 0u);

                auto compute_eff_ori = [&](uint32_t idx) -> ChildOri {
                    if (idx < m.children_count) {
                        uint32_t cref = m_view.children_ptr[m.children_offset + idx];
                        return compute_ori(cref);
                    }
                    if (has_down_child) {
                        return compute_ori_from_state((*macro_down_states)[macro_id]);
                    }
                    return compute_ori_from_state(series_context_state);
                };

                for (uint32_t i = 0; i < n_eff; ++i) {
                    ChildOri o = compute_eff_ori(i);
                    if (o.ambiguous_pole0 || o.ambiguous_pole1) {
                        return 0;
                    }
                }

                uint32_t emitted = 0;
                for (auto leftSign : {EdgePartType::PLUS, EdgePartType::MINUS}) {
                    for (auto rightSign : {EdgePartType::PLUS, EdgePartType::MINUS}) {
                        bool any = false;
                        bool equal_parts = true;
                        bool context_only_part = false;
                        uint32_t part_size = 0;
                        uint32_t single_idx = SPQR_INVALID;
                        for (uint32_t i = 0; i < n_eff; ++i) {
                            ChildOri o = compute_eff_ori(i);
                            const bool in_left = (o.lSign == leftSign);
                            const bool in_right = (o.rSign == rightSign);
                            if (in_left != in_right) {
                                equal_parts = false;
                                break;
                            }
                            if (in_left) {
                                part_size++;
                                single_idx = i;
                                if (i < m.children_count) {
                                    any = true;
                                } else {
                                    context_only_part = true;
                                }
                            }
                        }

                        if (part_size == 0 || !equal_parts) {
                            continue;
                        }

                        if (!any && context_only_part) {
                            bool ok = false;
                            if (macro_down_gcc_cut_count && macro_down_gcc_cut_idx &&
                                macro_id < macro_down_gcc_cut_count->size() &&
                                macro_id < macro_down_gcc_cut_idx->size()) {
                                ok = true;
                                const uint8_t n_cuts = (*macro_down_gcc_cut_count)[macro_id];
                                const auto& cuts = (*macro_down_gcc_cut_idx)[macro_id];
                                const int32_t pole0_idx = pole0Gcc ? pole0Gcc.index() : -1;
                                const int32_t pole1_idx = pole1Gcc ? pole1Gcc.index() : -1;
                                for (uint8_t j = 0; j < n_cuts && j < 3; ++j) {
                                    if (cuts[j] != pole0_idx && cuts[j] != pole1_idx) {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            if (!ok) {
                                continue;
                            }
                        }

                        if (macro_gcc_cuts && part_size == 1) {
                            if (single_idx >= m.children_count) {
                            } else {
                                uint32_t cref = m_view.children_ptr[m.children_offset + single_idx];
                                if (SP_COMPRESS_CHILD_IS_MACRO(cref)) {
                                    uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                                    if (m_view.macros_ptr[sub_id].kind == SP_COMPRESS_KIND_SERIES) {
                                        spqr_compat::node p0GccForCuts = blockNodeToCc(blk, pole0Blk);
                                        spqr_compat::node p1GccForCuts = blockNodeToCc(blk, pole1Blk);
                                        bool ok = true;
                                        for (spqr_compat::node gccCut : macro_gcc_cuts->get(sub_id)) {
                                            if (gccCut != p0GccForCuts && gccCut != p1GccForCuts) {
                                                ok = false;
                                                break;
                                            }
                                        }
                                        if (!ok) {
                                            continue;
                                        }
                                    }
                                }
                            }
                        }

                        if (out_snarls) {
                            out_snarls->push_back(
                                {SnarlEndpoint{pole0G, leftSign},
                                 SnarlEndpoint{pole1G, rightSign}});
	                        }
                        emitted++;
                    }
                }

                return emitted;
            }

            inline uint32_t emit_all_parallel_macro_snarls(
                const std::vector<EdgeDPState> &states,
                const SpCompressTreeView &m_view,
                BlockData &blk,
                const CcData &cc,
                std::vector<SnarlEndpointPair> *out_snarls,
                const std::vector<uint8_t>& absorbed_by_tcore = {},
                MacroSeriesGccCutsCache* macro_gcc_cuts = nullptr,
                const std::vector<EdgeDPState>* macro_down_states = nullptr,
                const std::vector<uint8_t>* macro_has_down_state = nullptr,
	                const std::vector<uint8_t>* macro_down_gcc_cut_count = nullptr,
	                const std::vector<std::array<int32_t, 3>>* macro_down_gcc_cut_idx = nullptr)
            {
                const uint32_t M = m_view.macros_len;

                std::vector<uint8_t> absorbed(M, 0);
                std::vector<uint32_t> parent_macro(M, SPQR_INVALID);
                std::vector<uint32_t> parent_child_idx(M, SPQR_INVALID);
                for (uint32_t m_id = 0; m_id < M; ++m_id) {
                    const SpCompressNode &m = m_view.macros_ptr[m_id];
                    for (uint32_t i = 0; i < m.children_count; ++i) {
                        uint32_t cref = m_view.children_ptr[m.children_offset + i];
                        if (!SP_COMPRESS_CHILD_IS_MACRO(cref)) continue;
                        uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                        if (sub_id < M) {
                            parent_macro[sub_id] = m_id;
                            parent_child_idx[sub_id] = i;
                        }
                        if (m.kind != SP_COMPRESS_KIND_PARALLEL) continue;
                        const SpCompressNode &sub = m_view.macros_ptr[sub_id];
                        if (sub.kind != SP_COMPRESS_KIND_PARALLEL) continue;
                        bool same_poles =
                            (sub.left == m.left && sub.right == m.right) ||
                            (sub.left == m.right && sub.right == m.left);
                        if (same_poles) absorbed[sub_id] = 1;
                    }
                }

                const bool have_tcore_filter = !absorbed_by_tcore.empty();

                uint32_t total = 0;
                uint32_t skipped_absorbed_parent = 0;
                uint32_t skipped_absorbed_tcore = 0;
                for (uint32_t m_id = 0; m_id < M; ++m_id) {
                    const SpCompressNode &m = m_view.macros_ptr[m_id];
                    if (m.kind != SP_COMPRESS_KIND_PARALLEL) continue;
                    if (absorbed[m_id]) { skipped_absorbed_parent++; continue; }
                    if (have_tcore_filter && absorbed_by_tcore[m_id]) {
                        skipped_absorbed_tcore++;
                        continue;
                    }
	                    total += emit_macro_snarls_parallel(
	                        m_id, states, m_view, blk, cc, out_snarls,
	                        macro_gcc_cuts, macro_down_states, macro_has_down_state,
                        macro_down_gcc_cut_count, macro_down_gcc_cut_idx,
                        &parent_macro, &parent_child_idx);
                }
                return total;
            }


            inline void merge_real_edge_into_state(
                EdgeDPState& state,
                spqr_compat::node pole0G, spqr_compat::node pole1G,
                spqr_compat::edge eG)
            {
                auto& C = ctx();
                spqr_compat::node uG = C.G.source(eG);
                spqr_compat::node vG = C.G.target(eG);
                if (uG == pole0G) {
                    auto t = getNodeEdgeType(uG, eG);
                    if (t == EdgePartType::PLUS) state.localPlusS++;
                    else if (t == EdgePartType::MINUS) state.localMinusS++;
                }
                if (vG == pole0G) {
                    auto t = getNodeEdgeType(vG, eG);
                    if (t == EdgePartType::PLUS) state.localPlusS++;
                    else if (t == EdgePartType::MINUS) state.localMinusS++;
                }
                if (uG == pole1G) {
                    auto t = getNodeEdgeType(uG, eG);
                    if (t == EdgePartType::PLUS) state.localPlusT++;
                    else if (t == EdgePartType::MINUS) state.localMinusT++;
                }
                if (vG == pole1G) {
                    auto t = getNodeEdgeType(vG, eG);
                    if (t == EdgePartType::PLUS) state.localPlusT++;
                    else if (t == EdgePartType::MINUS) state.localMinusT++;
                }
            }

            inline void merge_substate_into_state(
                EdgeDPState& state,
                spqr_compat::node pole0Blk, spqr_compat::node pole1Blk,
                const EdgeDPState& sub)
            {
                if (sub.s == pole0Blk) {
                    state.localPlusS  += sub.localPlusS;
                    state.localMinusS += sub.localMinusS;
                } else if (sub.t == pole0Blk) {
                    state.localPlusS  += sub.localPlusT;
                    state.localMinusS += sub.localMinusT;
                }
                if (sub.s == pole1Blk) {
                    state.localPlusT  += sub.localPlusS;
                    state.localMinusT += sub.localMinusS;
                } else if (sub.t == pole1Blk) {
                    state.localPlusT  += sub.localPlusT;
                    state.localMinusT += sub.localMinusT;
                }
            }

            inline void subtract_substate_from_state(
                EdgeDPState& state,
                spqr_compat::node pole0Blk, spqr_compat::node pole1Blk,
                const EdgeDPState& sub)
            {
                if (sub.s == pole0Blk) {
                    state.localPlusS  -= sub.localPlusS;
                    state.localMinusS -= sub.localMinusS;
                } else if (sub.t == pole0Blk) {
                    state.localPlusS  -= sub.localPlusT;
                    state.localMinusS -= sub.localMinusT;
                }
                if (sub.s == pole1Blk) {
                    state.localPlusT  -= sub.localPlusS;
                    state.localMinusT -= sub.localMinusS;
                } else if (sub.t == pole1Blk) {
                    state.localPlusT  -= sub.localPlusT;
                    state.localMinusT -= sub.localMinusT;
                }
            }

            inline void reset_state_for_poles(
                EdgeDPState& state,
                spqr_compat::node pole0Blk,
                spqr_compat::node pole1Blk)
            {
                state.s = pole0Blk;
                state.t = pole1Blk;
                state.localPlusS = state.localMinusS = 0;
                state.localPlusT = state.localMinusT = 0;
            }

            inline EdgePartType sign_from_counts(int plus, int minus)
            {
                if (plus > 0 && minus == 0) return EdgePartType::PLUS;
                if (minus > 0 && plus == 0) return EdgePartType::MINUS;
                return EdgePartType::NONE;
            }

            inline EdgePartType state_sign_at_block_node(
                const EdgeDPState& state,
                spqr_compat::node vBlk)
            {
                if (state.s == vBlk) {
                    return sign_from_counts(state.localPlusS, state.localMinusS);
                }
                if (state.t == vBlk) {
                    return sign_from_counts(state.localPlusT, state.localMinusT);
                }
                return EdgePartType::NONE;
            }

            inline bool state_has_ambiguous_sign_at_block_node(
                const EdgeDPState& state,
                spqr_compat::node vBlk)
            {
                if (state.s == vBlk) {
                    return state.localPlusS > 0 && state.localMinusS > 0;
                }
                if (state.t == vBlk) {
                    return state.localPlusT > 0 && state.localMinusT > 0;
                }
                return false;
            }

            inline bool state_has_block_node(const EdgeDPState& state, spqr_compat::node vBlk)
            {
                return state.s == vBlk || state.t == vBlk;
            }

            inline void merge_child_ref_into_state(
                EdgeDPState& state,
                spqr_compat::node pole0Blk,
                spqr_compat::node pole1Blk,
                spqr_compat::node pole0G,
                spqr_compat::node pole1G,
                uint32_t child_ref,
                const std::vector<EdgeDPState>& macro_states,
                const SpCompressTreeView& m_view,
                BlockData& blk)
            {
                (void)m_view;
                if (SP_COMPRESS_CHILD_IS_EDGE(child_ref)) {
                    uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(child_ref);
                    spqr_compat::edge eBlk{bid};
                    spqr_compat::edge eG = blockEdgeToOrig(blk, eBlk);
                    merge_real_edge_into_state(state, pole0G, pole1G, eG);
                } else {
                    uint32_t macro_id = SP_COMPRESS_CHILD_AS_MACRO(child_ref);
                    merge_substate_into_state(state, pole0Blk, pole1Blk,
                                              macro_states[macro_id]);
                }
            }

            inline void subtract_child_ref_from_state(
                EdgeDPState& state,
                spqr_compat::node pole0Blk,
                spqr_compat::node pole1Blk,
                spqr_compat::node pole0G,
                spqr_compat::node pole1G,
                uint32_t child_ref,
                const std::vector<EdgeDPState>& macro_states,
                BlockData& blk)
            {
                if (SP_COMPRESS_CHILD_IS_EDGE(child_ref)) {
                    EdgeDPState tmp;
                    reset_state_for_poles(tmp, pole0Blk, pole1Blk);
                    uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(child_ref);
                    spqr_compat::edge eBlk{bid};
                    spqr_compat::edge eG = blockEdgeToOrig(blk, eBlk);
                    merge_real_edge_into_state(tmp, pole0G, pole1G, eG);
                    state.localPlusS  -= tmp.localPlusS;
                    state.localMinusS -= tmp.localMinusS;
                    state.localPlusT  -= tmp.localPlusT;
                    state.localMinusT -= tmp.localMinusT;
                } else {
                    uint32_t macro_id = SP_COMPRESS_CHILD_AS_MACRO(child_ref);
                    subtract_substate_from_state(state, pole0Blk, pole1Blk,
                                                 macro_states[macro_id]);
                }
            }

            struct TCoreContext {
                const SpqrTree* T = nullptr;
                uint32_t T_len = 0, T_root = 0;
                const uint8_t* node_types = nullptr;
                const uint32_t* skel_offsets = nullptr;
                const SkeletonEdge* skel_edges = nullptr;
                const uint32_t* node_parents = nullptr;
                const uint32_t* children_offsets = nullptr;
                const uint32_t* children_array = nullptr;
                const uint32_t* node_mapping_offsets = nullptr;
                const uint32_t* node_mapping = nullptr;
                const uint32_t* core_node_inv = nullptr;
                std::vector<uint32_t> post_order; 
            };

            inline bool build_T_core_context(BlockData& blk, TCoreContext& ctx_out)
            {
                if (blk.coreSpqrTree) {
                    ctx_out.T = blk.coreSpqrTree;
                } else {
                    ctx_out.T = nullptr;
                }
                if (!ctx_out.T) return false;

                spqr_tree_info(ctx_out.T, &ctx_out.T_len, &ctx_out.T_root);
                if (ctx_out.T_len == 0) return false;

                ctx_out.node_types = spqr_tree_node_types_raw(ctx_out.T);
                ctx_out.skel_offsets = spqr_tree_skeleton_offsets_raw(ctx_out.T);
                uint32_t skel_total = 0;
                ctx_out.skel_edges = spqr_tree_skeleton_edges_raw(ctx_out.T, &skel_total);
                ctx_out.node_parents = spqr_tree_node_parents_raw(ctx_out.T);
                ctx_out.children_offsets = spqr_tree_children_offsets_raw(ctx_out.T);
                uint32_t children_len = 0;
                ctx_out.children_array = spqr_tree_children_raw(ctx_out.T, &children_len);
                uint32_t mapping_len = 0;
                spqr_tree_node_mapping_raw(ctx_out.T,
                                           &ctx_out.node_mapping_offsets,
                                           &ctx_out.node_mapping,
                                           &mapping_len);

                uint32_t inv_len = 0;
                if (blk.coreNodeInv) {
                    ctx_out.core_node_inv = blk.coreNodeInv;
                    inv_len = blk.coreNodeInvLen;
                } else {
                    ctx_out.core_node_inv = nullptr;
                }
                if (!ctx_out.core_node_inv || inv_len == 0) return false;

                // Iterative postorder use a stack with entered flag
                ctx_out.post_order.clear();
                ctx_out.post_order.reserve(ctx_out.T_len);
                std::vector<uint8_t> entered(ctx_out.T_len, 0);
                std::vector<uint32_t> stack;
                stack.reserve(ctx_out.T_len);
                stack.push_back(ctx_out.T_root);
                while (!stack.empty()) {
                    uint32_t tn = stack.back();
                    if (!entered[tn]) {
                        entered[tn] = 1;
                        uint32_t cs = ctx_out.children_offsets[tn];
                        uint32_t ce = ctx_out.children_offsets[tn + 1];
                        for (uint32_t i = cs; i < ce; ++i) {
                            stack.push_back(ctx_out.children_array[i]);
                        }
                    } else {
                        stack.pop_back();
                        ctx_out.post_order.push_back(tn);
                    }
                }
                return true;
            }

            inline uint32_t tcore_local_to_block_id(
                const TCoreContext& tctx,
                uint32_t tree_node,
                uint32_t local_idx)
            {
                uint32_t map_start = tctx.node_mapping_offsets[tree_node];
                uint32_t core_id = tctx.node_mapping[map_start + local_idx];
                return tctx.core_node_inv[core_id];
            }

            inline std::vector<EdgeDPState> compute_T_core_up_states(
                const std::vector<EdgeDPState>& macro_states,
                const SpCompressTreeView& m_view,
                BlockData& blk,
                const TCoreContext& tctx)
            {
                std::vector<EdgeDPState> up_states(tctx.T_len);

                for (uint32_t tn : tctx.post_order) {
                    if (tn == tctx.T_root) continue;

                    uint32_t parent_tn = tctx.node_parents[tn];

                    uint32_t e_start = tctx.skel_offsets[tn];
                    uint32_t e_end = tctx.skel_offsets[tn + 1];
                    uint32_t parent_virt_idx = SPQR_INVALID;
                    uint32_t parent_pole0_local = 0, parent_pole1_local = 1;
                    for (uint32_t i = e_start; i < e_end; ++i) {
                        const SkeletonEdge& se = tctx.skel_edges[i];
                        if (se.real_edge == SPQR_INVALID && se.twin_tree_node == parent_tn) {
                            parent_virt_idx = i;
                            parent_pole0_local = se.src;
                            parent_pole1_local = se.dst;
                            break;
                        }
                    }
                    if (parent_virt_idx == SPQR_INVALID) continue; 

                    uint32_t pole0_blk_id = tcore_local_to_block_id(tctx, tn, parent_pole0_local);
                    uint32_t pole1_blk_id = tcore_local_to_block_id(tctx, tn, parent_pole1_local);
                    spqr_compat::node pole0Blk{pole0_blk_id};
                    spqr_compat::node pole1Blk{pole1_blk_id};
                    spqr_compat::node pole0G = blockNodeToOrig(blk, pole0Blk);
                    spqr_compat::node pole1G = blockNodeToOrig(blk, pole1Blk);

                    EdgeDPState& state = up_states[tn];
                    state.s = pole0Blk;
                    state.t = pole1Blk;
                    state.localPlusS = state.localMinusS = 0;
                    state.localPlusT = state.localMinusT = 0;

                    for (uint32_t i = e_start; i < e_end; ++i) {
                        if (i == parent_virt_idx) continue;
                        const SkeletonEdge& se = tctx.skel_edges[i];

                        if (se.real_edge != SPQR_INVALID) {
                            uint32_t cr = m_view.core_edges_ptr[se.real_edge].child;
                            if (SP_COMPRESS_CHILD_IS_EDGE(cr)) {
                                uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cr);
                                spqr_compat::edge eBlk{bid};
                                spqr_compat::edge eG = blockEdgeToOrig(blk, eBlk);
                                merge_real_edge_into_state(state, pole0G, pole1G, eG);
                            } else {
                                uint32_t macro_id = SP_COMPRESS_CHILD_AS_MACRO(cr);
                                merge_substate_into_state(state, pole0Blk, pole1Blk,
                                                          macro_states[macro_id]);
                            }
                        } else {
                            uint32_t gc = se.twin_tree_node;
                            merge_substate_into_state(state, pole0Blk, pole1Blk, up_states[gc]);
                        }
                    }
                }
                return up_states;
            }

            inline std::vector<EdgeDPState> compute_T_core_down_states(
                const std::vector<EdgeDPState>& up_states,
                const std::vector<EdgeDPState>& macro_states,
                const SpCompressTreeView& m_view,
                BlockData& blk,
                const TCoreContext& tctx)
            {
                std::vector<EdgeDPState> down_states(tctx.T_len);

                struct LocalCounts {
                    int plus{0};
                    int minus{0};
                };

                std::vector<LocalCounts> local_counts;

                auto add_edge_contribution = [&](uint32_t tn,
                                                 const SkeletonEdge& se,
                                                 const EdgeDPState* virtual_state) {
                    const uint32_t pole0_blk_id = tcore_local_to_block_id(tctx, tn, se.src);
                    const uint32_t pole1_blk_id = tcore_local_to_block_id(tctx, tn, se.dst);
                    spqr_compat::node pole0Blk{pole0_blk_id};
                    spqr_compat::node pole1Blk{pole1_blk_id};
                    spqr_compat::node pole0G = blockNodeToOrig(blk, pole0Blk);
                    spqr_compat::node pole1G = blockNodeToOrig(blk, pole1Blk);

                    EdgeDPState tmp;
                    reset_state_for_poles(tmp, pole0Blk, pole1Blk);

                    if (se.real_edge != SPQR_INVALID) {
                        uint32_t cr = m_view.core_edges_ptr[se.real_edge].child;
                        merge_child_ref_into_state(tmp, pole0Blk, pole1Blk,
                                                   pole0G, pole1G, cr,
                                                   macro_states, m_view, blk);
                    } else if (virtual_state) {
                        merge_substate_into_state(tmp, pole0Blk, pole1Blk,
                                                  *virtual_state);
                    }

                    local_counts[se.src].plus += tmp.localPlusS;
                    local_counts[se.src].minus += tmp.localMinusS;
                    local_counts[se.dst].plus += tmp.localPlusT;
                    local_counts[se.dst].minus += tmp.localMinusT;
                };

                auto subtract_edge_contribution = [&](uint32_t tn,
                                                      const SkeletonEdge& se,
                                                      const EdgeDPState& virtual_state,
                                                      EdgeDPState& state) {
                    EdgeDPState tmp;
                    reset_state_for_poles(tmp, state.s, state.t);
                    merge_substate_into_state(tmp, state.s, state.t, virtual_state);
                    state.localPlusS -= tmp.localPlusS;
                    state.localMinusS -= tmp.localMinusS;
                    state.localPlusT -= tmp.localPlusT;
                    state.localMinusT -= tmp.localMinusT;
                    (void)tn;
                    (void)se;
                };

                for (auto it = tctx.post_order.rbegin(); it != tctx.post_order.rend(); ++it) {
                    uint32_t P = *it;
                    const uint32_t local_n =
                        tctx.node_mapping_offsets[P + 1] -
                        tctx.node_mapping_offsets[P];
                    local_counts.assign(local_n, LocalCounts{});

                    const uint32_t e_start = tctx.skel_offsets[P];
                    const uint32_t e_end = tctx.skel_offsets[P + 1];
                    const uint32_t P_parent =
                        (P == tctx.T_root) ? SPQR_INVALID : tctx.node_parents[P];

                    for (uint32_t i = e_start; i < e_end; ++i) {
                        const SkeletonEdge& se = tctx.skel_edges[i];
                        const EdgeDPState* virtual_state = nullptr;
                        if (se.real_edge == SPQR_INVALID) {
                            const uint32_t v = se.twin_tree_node;
                            if (P_parent != SPQR_INVALID && v == P_parent) {
                                virtual_state = &down_states[P];
                            } else {
                                virtual_state = &up_states[v];
                            }
                        }
                        add_edge_contribution(P, se, virtual_state);
                    }

                    for (uint32_t i = e_start; i < e_end; ++i) {
                        const SkeletonEdge& se = tctx.skel_edges[i];
                        if (se.real_edge != SPQR_INVALID) continue;

                        const uint32_t child = se.twin_tree_node;
                        if (child == P_parent || child == SPQR_INVALID) continue;

                        const uint32_t pole0_blk_id = tcore_local_to_block_id(tctx, P, se.src);
                        const uint32_t pole1_blk_id = tcore_local_to_block_id(tctx, P, se.dst);
                        EdgeDPState& state = down_states[child];
                        state.s = spqr_compat::node{pole0_blk_id};
                        state.t = spqr_compat::node{pole1_blk_id};
                        state.localPlusS = local_counts[se.src].plus;
                        state.localMinusS = local_counts[se.src].minus;
                        state.localPlusT = local_counts[se.dst].plus;
                        state.localMinusT = local_counts[se.dst].minus;

                        subtract_edge_contribution(P, se, up_states[child], state);
                    }
                }
                return down_states;
            }

            struct MacroDownContext {
                std::vector<EdgeDPState> states;
                std::vector<uint8_t> has_state;
                std::vector<uint8_t> gcc_cut_count;
                std::vector<std::array<int32_t, 3>> gcc_cut_idx;
                uint32_t seeded_from_tcore{0};
                uint32_t nested_states{0};
            };

            inline void compute_tcore_edge_context_state(
                uint32_t tn,
                uint32_t excluded_skel_idx,
                uint32_t macro_id,
                const std::vector<EdgeDPState>& macro_states,
                const std::vector<EdgeDPState>& tcore_up_states,
                const std::vector<EdgeDPState>& tcore_down_states,
                const TCoreContext& tctx,
                const SpCompressTreeView& m_view,
                BlockData& blk,
                EdgeDPState& state)
            {
                const SpCompressNode& target = m_view.macros_ptr[macro_id];
                spqr_compat::node pole0Blk{target.left};
                spqr_compat::node pole1Blk{target.right};
                spqr_compat::node pole0G = blockNodeToOrig(blk, pole0Blk);
                spqr_compat::node pole1G = blockNodeToOrig(blk, pole1Blk);
                reset_state_for_poles(state, pole0Blk, pole1Blk);

                uint32_t e_start = tctx.skel_offsets[tn];
                uint32_t e_end = tctx.skel_offsets[tn + 1];
                uint32_t parent_tn =
                    (tn == tctx.T_root) ? SPQR_INVALID : tctx.node_parents[tn];

                for (uint32_t i = e_start; i < e_end; ++i) {
                    if (i == excluded_skel_idx) continue;
                    const SkeletonEdge& se = tctx.skel_edges[i];

                    if (se.real_edge != SPQR_INVALID) {
                        uint32_t cr = m_view.core_edges_ptr[se.real_edge].child;
                        merge_child_ref_into_state(state, pole0Blk, pole1Blk,
                                                   pole0G, pole1G, cr,
                                                   macro_states, m_view, blk);
                    } else {
                        uint32_t v = se.twin_tree_node;
                        const EdgeDPState* sub = nullptr;
                        if (parent_tn != SPQR_INVALID && v == parent_tn) {
                            if (tn < tcore_down_states.size()) {
                                sub = &tcore_down_states[tn];
                            }
                        } else if (v < tcore_up_states.size()) {
                            sub = &tcore_up_states[v];
                        }
                        if (sub) {
                            merge_substate_into_state(state, pole0Blk, pole1Blk, *sub);
                        }
                    }
                }
            }

            inline MacroDownContext compute_root_macro_tcore_contexts(
                const std::vector<EdgeDPState>& macro_states,
                const SpCompressTreeView& m_view,
                BlockData& blk,
                const TCoreContext& tctx,
                const std::vector<EdgeDPState>& tcore_up_states,
                const std::vector<EdgeDPState>& tcore_down_states,
                const std::vector<std::vector<spqr_compat::node>>* tcore_s_gcc_cuts = nullptr)
            {
                const uint32_t M = m_view.macros_len;
                MacroDownContext out;
                out.states.resize(M);
                out.has_state.assign(M, 0);
                if (tcore_s_gcc_cuts) {
                    out.gcc_cut_count.assign(M, 0);
                    const std::array<int32_t, 3> empty_cuts{{-1, -1, -1}};
                    out.gcc_cut_idx.assign(M, empty_cuts);
                }
                if (M == 0 || tctx.T_len == 0) return out;

                std::vector<uint8_t> has_macro_parent(M, 0);
                for (uint32_t m_id = 0; m_id < M; ++m_id) {
                    const SpCompressNode& m = m_view.macros_ptr[m_id];
                    for (uint32_t i = 0; i < m.children_count; ++i) {
                        uint32_t cref = m_view.children_ptr[m.children_offset + i];
                        if (!SP_COMPRESS_CHILD_IS_MACRO(cref)) continue;
                        uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                        if (sub_id < M) has_macro_parent[sub_id] = 1;
                    }
                }

                struct LocalCounts {
                    int plus{0};
                    int minus{0};
                };
                std::vector<LocalCounts> local_counts;

                auto edge_local_counts = [&](uint32_t tn,
                                             const SkeletonEdge& se,
                                             const EdgeDPState* virtual_state,
                                             LocalCounts& src_counts,
                                             LocalCounts& dst_counts) {
                    src_counts = LocalCounts{};
                    dst_counts = LocalCounts{};

                    const uint32_t pole0_blk_id = tcore_local_to_block_id(tctx, tn, se.src);
                    const uint32_t pole1_blk_id = tcore_local_to_block_id(tctx, tn, se.dst);
                    spqr_compat::node pole0Blk{pole0_blk_id};
                    spqr_compat::node pole1Blk{pole1_blk_id};
                    spqr_compat::node pole0G = blockNodeToOrig(blk, pole0Blk);
                    spqr_compat::node pole1G = blockNodeToOrig(blk, pole1Blk);

                    EdgeDPState tmp;
                    reset_state_for_poles(tmp, pole0Blk, pole1Blk);
                    if (se.real_edge != SPQR_INVALID) {
                        uint32_t cr = m_view.core_edges_ptr[se.real_edge].child;
                        merge_child_ref_into_state(tmp, pole0Blk, pole1Blk,
                                                   pole0G, pole1G, cr,
                                                   macro_states, m_view, blk);
                    } else if (virtual_state) {
                        merge_substate_into_state(tmp, pole0Blk, pole1Blk, *virtual_state);
                    }

                    src_counts.plus = tmp.localPlusS;
                    src_counts.minus = tmp.localMinusS;
                    dst_counts.plus = tmp.localPlusT;
                    dst_counts.minus = tmp.localMinusT;
                };

                for (uint32_t tn = 0; tn < tctx.T_len; ++tn) {
                    const uint32_t local_n =
                        tctx.node_mapping_offsets[tn + 1] -
                        tctx.node_mapping_offsets[tn];
                    local_counts.assign(local_n, LocalCounts{});

                    uint32_t e_start = tctx.skel_offsets[tn];
                    uint32_t e_end = tctx.skel_offsets[tn + 1];
                    uint32_t parent_tn =
                        (tn == tctx.T_root) ? SPQR_INVALID : tctx.node_parents[tn];

                    for (uint32_t i = e_start; i < e_end; ++i) {
                        const SkeletonEdge& se = tctx.skel_edges[i];
                        const EdgeDPState* virtual_state = nullptr;
                        if (se.real_edge == SPQR_INVALID) {
                            uint32_t v = se.twin_tree_node;
                            if (parent_tn != SPQR_INVALID && v == parent_tn) {
                                if (tn < tcore_down_states.size()) {
                                    virtual_state = &tcore_down_states[tn];
                                }
                            } else if (v < tcore_up_states.size()) {
                                virtual_state = &tcore_up_states[v];
                            }
                        }

                        LocalCounts src_counts;
                        LocalCounts dst_counts;
                        edge_local_counts(tn, se, virtual_state, src_counts, dst_counts);
                        local_counts[se.src].plus += src_counts.plus;
                        local_counts[se.src].minus += src_counts.minus;
                        local_counts[se.dst].plus += dst_counts.plus;
                        local_counts[se.dst].minus += dst_counts.minus;
                    }

                    for (uint32_t i = e_start; i < e_end; ++i) {
                        const SkeletonEdge& se = tctx.skel_edges[i];
                        if (se.real_edge == SPQR_INVALID) continue;
                        uint32_t cr = m_view.core_edges_ptr[se.real_edge].child;
                        if (!SP_COMPRESS_CHILD_IS_MACRO(cr)) continue;
                        uint32_t macro_id = SP_COMPRESS_CHILD_AS_MACRO(cr);
                        if (macro_id >= M || has_macro_parent[macro_id]) continue;

                        const SpCompressNode& macro = m_view.macros_ptr[macro_id];
                        const uint32_t src_blk_id = tcore_local_to_block_id(tctx, tn, se.src);
                        const uint32_t dst_blk_id = tcore_local_to_block_id(tctx, tn, se.dst);

                        uint32_t left_local = SPQR_INVALID;
                        uint32_t right_local = SPQR_INVALID;
                        if (macro.left == src_blk_id) left_local = se.src;
                        else if (macro.left == dst_blk_id) left_local = se.dst;
                        if (macro.right == src_blk_id) right_local = se.src;
                        else if (macro.right == dst_blk_id) right_local = se.dst;
                        if (left_local == SPQR_INVALID || right_local == SPQR_INVALID) {
                            continue;
                        }

                        LocalCounts excl_src;
                        LocalCounts excl_dst;
                        edge_local_counts(tn, se, nullptr, excl_src, excl_dst);
                        auto excluded_at = [&](uint32_t local) -> LocalCounts {
                            if (local == se.src) return excl_src;
                            if (local == se.dst) return excl_dst;
                            return LocalCounts{};
                        };

                        LocalCounts left = local_counts[left_local];
                        LocalCounts right = local_counts[right_local];
                        LocalCounts excl_left = excluded_at(left_local);
                        LocalCounts excl_right = excluded_at(right_local);
                        left.plus -= excl_left.plus;
                        left.minus -= excl_left.minus;
                        right.plus -= excl_right.plus;
                        right.minus -= excl_right.minus;

                        EdgeDPState& state = out.states[macro_id];
                        state.s = spqr_compat::node{macro.left};
                        state.t = spqr_compat::node{macro.right};
                        state.localPlusS = left.plus;
                        state.localMinusS = left.minus;
                        state.localPlusT = right.plus;
                        state.localMinusT = right.minus;
                        if (tcore_s_gcc_cuts &&
                            tctx.node_types[tn] == SPQR_NODE_TYPE_S &&
                            tn < tcore_s_gcc_cuts->size() &&
                            macro_id < out.gcc_cut_count.size() &&
                            macro_id < out.gcc_cut_idx.size()) {
                            const auto& cuts = (*tcore_s_gcc_cuts)[tn];
                            const uint8_t n_cuts =
                                static_cast<uint8_t>(std::min<size_t>(cuts.size(), 3));
                            out.gcc_cut_count[macro_id] = n_cuts;
                            for (uint8_t j = 0; j < n_cuts; ++j) {
                                out.gcc_cut_idx[macro_id][j] =
                                    cuts[j] ? cuts[j].index() : -1;
                            }
                        }
                        if (!out.has_state[macro_id]) {
                            out.seeded_from_tcore++;
                        }
                        out.has_state[macro_id] = 1;
                    }
                }

                return out;
            }

            inline MacroDownContext compute_macro_down_states(
                const std::vector<EdgeDPState>& macro_states,
                const SpCompressTreeView& m_view,
                BlockData& blk,
                const TCoreContext* tctx = nullptr,
                const std::vector<EdgeDPState>* tcore_up_states = nullptr,
                const std::vector<EdgeDPState>* tcore_down_states = nullptr,
                const std::vector<uint8_t>* target_filter = nullptr)
            {
                const uint32_t M = m_view.macros_len;
                MacroDownContext out;
                out.states.resize(M);
                out.has_state.assign(M, 0);
                if (M == 0) return out;

                std::vector<uint32_t> parent_macro(M, SPQR_INVALID);
                for (uint32_t m_id = 0; m_id < M; ++m_id) {
                    const SpCompressNode& m = m_view.macros_ptr[m_id];
                    for (uint32_t i = 0; i < m.children_count; ++i) {
                        uint32_t cref = m_view.children_ptr[m.children_offset + i];
                        if (!SP_COMPRESS_CHILD_IS_MACRO(cref)) continue;
                        uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                        if (sub_id < M) parent_macro[sub_id] = m_id;
                    }
                }

                std::vector<uint8_t> needed;
                if (target_filter) {
                    needed.assign(M, 0);
                    for (uint32_t m_id = 0; m_id < M; ++m_id) {
                        if (m_id >= target_filter->size() || !(*target_filter)[m_id]) continue;
                        uint32_t cur = m_id;
                        while (cur != SPQR_INVALID && cur < M && !needed[cur]) {
                            needed[cur] = 1;
                            cur = parent_macro[cur];
                        }
                    }
                }
                auto is_needed = [&](uint32_t m_id) {
                    return !target_filter || (m_id < needed.size() && needed[m_id]);
                };

                if (tctx && tcore_up_states && tcore_down_states &&
                    tctx->T_len > 0 && !tcore_up_states->empty()) {
                    for (uint32_t tn = 0; tn < tctx->T_len; ++tn) {
                        uint32_t e_start = tctx->skel_offsets[tn];
                        uint32_t e_end = tctx->skel_offsets[tn + 1];
                        for (uint32_t i = e_start; i < e_end; ++i) {
                            const SkeletonEdge& se = tctx->skel_edges[i];
                            if (se.real_edge == SPQR_INVALID) continue;
                            uint32_t cr = m_view.core_edges_ptr[se.real_edge].child;
                            if (!SP_COMPRESS_CHILD_IS_MACRO(cr)) continue;
                            uint32_t macro_id = SP_COMPRESS_CHILD_AS_MACRO(cr);
                            if (macro_id >= M || parent_macro[macro_id] != SPQR_INVALID) continue;
                            if (!is_needed(macro_id)) continue;

                            compute_tcore_edge_context_state(
                                tn, i, macro_id, macro_states,
                                *tcore_up_states, *tcore_down_states,
                                *tctx, m_view, blk, out.states[macro_id]);
                            if (!out.has_state[macro_id]) {
                                out.seeded_from_tcore++;
                            }
                            out.has_state[macro_id] = 1;
                        }
                    }
                }

                std::vector<uint32_t> stack;
                stack.reserve(M);
                for (uint32_t m_id = 0; m_id < M; ++m_id) {
                    if (parent_macro[m_id] == SPQR_INVALID && is_needed(m_id)) {
                        stack.push_back(m_id);
                    }
                }

                while (!stack.empty()) {
                    uint32_t parent_id = stack.back();
                    stack.pop_back();

                    const SpCompressNode& parent = m_view.macros_ptr[parent_id];
                    const bool parent_has_down = out.has_state[parent_id] != 0;

                    auto finish_child = [&](uint32_t sub_id) {
                        if (!out.has_state[sub_id]) {
                            out.nested_states++;
                        }
                        out.has_state[sub_id] = 1;
                        stack.push_back(sub_id);
                    };

                    if (parent.kind == SP_COMPRESS_KIND_PARALLEL) {
                        spqr_compat::node parent0Blk{parent.left};
                        spqr_compat::node parent1Blk{parent.right};
                        spqr_compat::node parent0G = blockNodeToOrig(blk, parent0Blk);
                        spqr_compat::node parent1G = blockNodeToOrig(blk, parent1Blk);

                        EdgeDPState total;
                        reset_state_for_poles(total, parent0Blk, parent1Blk);
                        for (uint32_t k = 0; k < parent.children_count; ++k) {
                            uint32_t cref = m_view.children_ptr[parent.children_offset + k];
                            merge_child_ref_into_state(total, parent0Blk, parent1Blk,
                                                       parent0G, parent1G, cref,
                                                       macro_states, m_view, blk);
                        }
                        if (parent_has_down) {
                            merge_substate_into_state(total, parent0Blk, parent1Blk,
                                                      out.states[parent_id]);
                        }

                        for (uint32_t child_idx = 0; child_idx < parent.children_count; ++child_idx) {
                            uint32_t child_ref =
                                m_view.children_ptr[parent.children_offset + child_idx];
                            if (!SP_COMPRESS_CHILD_IS_MACRO(child_ref)) continue;

                            uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(child_ref);
                            if (sub_id >= M) continue;
                            if (!is_needed(sub_id)) continue;

                            const SpCompressNode& sub = m_view.macros_ptr[sub_id];
                            spqr_compat::node pole0Blk{sub.left};
                            spqr_compat::node pole1Blk{sub.right};
                            spqr_compat::node pole0G = blockNodeToOrig(blk, pole0Blk);
                            spqr_compat::node pole1G = blockNodeToOrig(blk, pole1Blk);

                            EdgeDPState& state = out.states[sub_id];
                            reset_state_for_poles(state, pole0Blk, pole1Blk);
                            merge_substate_into_state(state, pole0Blk, pole1Blk, total);
                            subtract_child_ref_from_state(state, pole0Blk, pole1Blk,
                                                          pole0G, pole1G, child_ref,
                                                          macro_states, blk);
                            finish_child(sub_id);
                        }
                    } else if (parent.kind == SP_COMPRESS_KIND_SERIES) {
                        const uint32_t k = parent.children_count;
                        if (k == 0) {} else {
                            struct PathChild {
                                uint32_t cref{SPQR_INVALID};
                                uint32_t a{SPQR_INVALID};
                                uint32_t b{SPQR_INVALID};
                            };
                            std::vector<PathChild> ch_local;
                            std::vector<std::pair<uint32_t, uint32_t>> incidence;
                            ch_local.reserve(k);
                            incidence.reserve(static_cast<size_t>(k) * 2);

                            auto child_endpoints = [&](uint32_t cref,
                                                       uint32_t& a,
                                                       uint32_t& b) -> bool {
                                if (SP_COMPRESS_CHILD_IS_EDGE(cref)) {
                                    uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cref);
                                    spqr_compat::edge eBlk{bid};
                                    a = blockEdgeSource(blk, eBlk).idx;
                                    b = blockEdgeTarget(blk, eBlk).idx;
                                } else {
                                    uint32_t mid2 = SP_COMPRESS_CHILD_AS_MACRO(cref);
                                    if (mid2 >= M) return false;
                                    const SpCompressNode& sm = m_view.macros_ptr[mid2];
                                    a = sm.left;
                                    b = sm.right;
                                }
                                return a != SPQR_INVALID && b != SPQR_INVALID;
                            };

                            bool valid_path = true;
                            for (uint32_t i = 0; i < k; ++i) {
                                uint32_t cref =
                                    m_view.children_ptr[parent.children_offset + i];
                                uint32_t a = SPQR_INVALID, b = SPQR_INVALID;
                                if (!child_endpoints(cref, a, b)) {
                                    valid_path = false;
                                    break;
                                }
                                ch_local.push_back(PathChild{cref, a, b});
                                incidence.emplace_back(a, i);
                                incidence.emplace_back(b, i);
                            }

                            std::vector<uint32_t> path_order;
                            path_order.reserve(k);
                            std::vector<uint32_t> path_vertices;
                            path_vertices.reserve(static_cast<size_t>(k) + 1);

                            if (valid_path) {
                                std::sort(incidence.begin(), incidence.end());
                                std::vector<uint8_t> visited(k, 0);
                                uint32_t current = parent.left;
                                uint32_t prev_child = SPQR_INVALID;
                                path_vertices.push_back(current);

                                for (uint32_t step = 0; step < k; ++step) {
                                    auto it = std::lower_bound(
                                        incidence.begin(), incidence.end(), current,
                                        [](const std::pair<uint32_t, uint32_t>& p,
                                           uint32_t v) {
                                            return p.first < v;
                                        });
                                    uint32_t chosen = SPQR_INVALID;
                                    for (; it != incidence.end() && it->first == current;
                                         ++it) {
                                        uint32_t cidx = it->second;
                                        if (cidx == prev_child) continue;
                                        if (cidx >= k || visited[cidx]) continue;
                                        chosen = cidx;
                                        break;
                                    }
                                    if (chosen == SPQR_INVALID) {
                                        valid_path = false;
                                        break;
                                    }
                                    const PathChild& ch = ch_local[chosen];
                                    uint32_t next = SPQR_INVALID;
                                    if (ch.a == current) next = ch.b;
                                    else if (ch.b == current) next = ch.a;
                                    else { valid_path = false; break; }
                                    visited[chosen] = 1;
                                    path_order.push_back(chosen);
                                    path_vertices.push_back(next);
                                    prev_child = chosen;
                                    current = next;
                                }
                                if (valid_path && current != parent.right) {
                                    valid_path = false;
                                }
                                if (valid_path) {
                                    for (uint8_t v : visited) {
                                        if (!v) { valid_path = false; break; }
                                    }
                                }
                            }

                            if (!valid_path) {
                                for (uint32_t child_idx = 0; child_idx < k; ++child_idx) {
                                    uint32_t child_ref =
                                        m_view.children_ptr[parent.children_offset + child_idx];
                                    if (!SP_COMPRESS_CHILD_IS_MACRO(child_ref)) continue;
                                    uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(child_ref);
                                    if (sub_id >= M) continue;
                                    if (!is_needed(sub_id)) continue;
                                    const SpCompressNode& sub = m_view.macros_ptr[sub_id];
                                    spqr_compat::node pole0Blk{sub.left};
                                    spqr_compat::node pole1Blk{sub.right};
                                    spqr_compat::node pole0G = blockNodeToOrig(blk, pole0Blk);
                                    spqr_compat::node pole1G = blockNodeToOrig(blk, pole1Blk);
                                    EdgeDPState& state = out.states[sub_id];
                                    reset_state_for_poles(state, pole0Blk, pole1Blk);
                                    if (child_idx > 0) {
                                        uint32_t prev_ref =
                                            m_view.children_ptr[parent.children_offset + child_idx - 1];
                                        merge_child_ref_into_state(state, pole0Blk, pole1Blk,
                                                                   pole0G, pole1G, prev_ref,
                                                                   macro_states, m_view, blk);
                                    }
                                    if (child_idx + 1 < k) {
                                        uint32_t next_ref =
                                            m_view.children_ptr[parent.children_offset + child_idx + 1];
                                        merge_child_ref_into_state(state, pole0Blk, pole1Blk,
                                                                   pole0G, pole1G, next_ref,
                                                                   macro_states, m_view, blk);
                                    }
                                    if (parent_has_down) {
                                        merge_substate_into_state(state, pole0Blk, pole1Blk,
                                                                  out.states[parent_id]);
                                    }
                                    finish_child(sub_id);
                                }
                            } else {
                                for (uint32_t i = 0; i < k; ++i) {
                                    uint32_t child_idx = path_order[i];
                                    const PathChild& ch = ch_local[child_idx];
                                    if (!SP_COMPRESS_CHILD_IS_MACRO(ch.cref)) continue;
                                    uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(ch.cref);
                                    if (sub_id >= M) continue;
                                    if (!is_needed(sub_id)) continue;

                                    const SpCompressNode& sub = m_view.macros_ptr[sub_id];
                                    spqr_compat::node pole0Blk{sub.left};
                                    spqr_compat::node pole1Blk{sub.right};
                                    spqr_compat::node pole0G = blockNodeToOrig(blk, pole0Blk);
                                    spqr_compat::node pole1G = blockNodeToOrig(blk, pole1Blk);

                                    EdgeDPState& state = out.states[sub_id];
                                    reset_state_for_poles(state, pole0Blk, pole1Blk);

                                    if (i > 0) {
                                        uint32_t prev_path_child = path_order[i - 1];
                                        uint32_t prev_ref = ch_local[prev_path_child].cref;
                                        merge_child_ref_into_state(state, pole0Blk, pole1Blk,
                                                                   pole0G, pole1G, prev_ref,
                                                                   macro_states, m_view, blk);
                                    }

                                    if (i + 1 < k) {
                                        uint32_t next_path_child = path_order[i + 1];
                                        uint32_t next_ref = ch_local[next_path_child].cref;
                                        merge_child_ref_into_state(state, pole0Blk, pole1Blk,
                                                                   pole0G, pole1G, next_ref,
                                                                   macro_states, m_view, blk);
                                    }
                                    if (parent_has_down) {
                                        merge_substate_into_state(state, pole0Blk, pole1Blk,
                                                                  out.states[parent_id]);
                                    }
                                    finish_child(sub_id);
                                }
                            }
                        }
                    } else {
                        for (uint32_t child_idx = 0; child_idx < parent.children_count; ++child_idx) {
                            uint32_t child_ref =
                                m_view.children_ptr[parent.children_offset + child_idx];
                            if (!SP_COMPRESS_CHILD_IS_MACRO(child_ref)) continue;

                            uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(child_ref);
                            if (sub_id >= M) continue;
                            if (!is_needed(sub_id)) continue;

                            const SpCompressNode& sub = m_view.macros_ptr[sub_id];
                            spqr_compat::node pole0Blk{sub.left};
                            spqr_compat::node pole1Blk{sub.right};
                            spqr_compat::node pole0G = blockNodeToOrig(blk, pole0Blk);
                            spqr_compat::node pole1G = blockNodeToOrig(blk, pole1Blk);

                            EdgeDPState& state = out.states[sub_id];
                            reset_state_for_poles(state, pole0Blk, pole1Blk);

                            for (uint32_t k = 0; k < parent.children_count; ++k) {
                                if (k == child_idx) continue;
                                uint32_t sibling_ref =
                                    m_view.children_ptr[parent.children_offset + k];
                                merge_child_ref_into_state(state, pole0Blk, pole1Blk,
                                                           pole0G, pole1G, sibling_ref,
                                                           macro_states, m_view, blk);
                            }

                            if (parent_has_down) {
                                merge_substate_into_state(state, pole0Blk, pole1Blk,
                                                          out.states[parent_id]);
                            }
                            finish_child(sub_id);
                        }
                    }
                }

                return out;
            }

            inline uint32_t extend_root_parallel_series_contexts(
                const std::vector<EdgeDPState>& macro_states,
                const SpCompressTreeView& m_view,
                BlockData& blk,
                const std::vector<uint8_t>& target_filter,
                MacroDownContext& context)
            {
                const uint32_t M = m_view.macros_len;
                if (context.states.size() < M || context.has_state.size() < M) {
                    return 0;
                }

                std::vector<uint8_t> has_macro_parent(M, 0);
                for (uint32_t m_id = 0; m_id < M; ++m_id) {
                    const SpCompressNode& m = m_view.macros_ptr[m_id];
                    for (uint32_t i = 0; i < m.children_count; ++i) {
                        uint32_t cref = m_view.children_ptr[m.children_offset + i];
                        if (!SP_COMPRESS_CHILD_IS_MACRO(cref)) continue;
                        uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                        if (sub_id < M) has_macro_parent[sub_id] = 1;
                    }
                }

                uint32_t filled = 0;
                for (uint32_t parent_id = 0; parent_id < M; ++parent_id) {
                    if (has_macro_parent[parent_id]) continue;
                    if (!context.has_state[parent_id]) continue;

                    const SpCompressNode& parent = m_view.macros_ptr[parent_id];
                    if (parent.kind != SP_COMPRESS_KIND_PARALLEL) continue;

                    spqr_compat::node parent0Blk{parent.left};
                    spqr_compat::node parent1Blk{parent.right};
                    spqr_compat::node parent0G = blockNodeToOrig(blk, parent0Blk);
                    spqr_compat::node parent1G = blockNodeToOrig(blk, parent1Blk);

                    EdgeDPState total;
                    reset_state_for_poles(total, parent0Blk, parent1Blk);
                    for (uint32_t k = 0; k < parent.children_count; ++k) {
                        uint32_t cref = m_view.children_ptr[parent.children_offset + k];
                        merge_child_ref_into_state(total, parent0Blk, parent1Blk,
                                                   parent0G, parent1G, cref,
                                                   macro_states, m_view, blk);
                    }
                    merge_substate_into_state(total, parent0Blk, parent1Blk,
                                              context.states[parent_id]);

                    for (uint32_t child_idx = 0; child_idx < parent.children_count; ++child_idx) {
                        uint32_t child_ref =
                            m_view.children_ptr[parent.children_offset + child_idx];
                        if (!SP_COMPRESS_CHILD_IS_MACRO(child_ref)) continue;

                        uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(child_ref);
                        if (sub_id >= M) continue;
                        if (sub_id >= target_filter.size() || !target_filter[sub_id]) continue;

                        const SpCompressNode& sub = m_view.macros_ptr[sub_id];
                        if (sub.kind != SP_COMPRESS_KIND_SERIES) continue;

                        spqr_compat::node pole0Blk{sub.left};
                        spqr_compat::node pole1Blk{sub.right};
                        spqr_compat::node pole0G = blockNodeToOrig(blk, pole0Blk);
                        spqr_compat::node pole1G = blockNodeToOrig(blk, pole1Blk);

                        EdgeDPState& state = context.states[sub_id];
                        reset_state_for_poles(state, pole0Blk, pole1Blk);
                        merge_substate_into_state(state, pole0Blk, pole1Blk, total);
                        subtract_child_ref_from_state(state, pole0Blk, pole1Blk,
                                                      pole0G, pole1G, child_ref,
                                                      macro_states, blk);
                        if (!context.has_state[sub_id]) {
                            ++filled;
                        }
                        context.has_state[sub_id] = 1;
                    }
                }

                return filled;
            }

            inline uint32_t emit_T_core_psnarls(
                const std::vector<EdgeDPState> &macro_states,
                const std::vector<EdgeDPState> &up_states,
                const std::vector<EdgeDPState> &down_states,
                const std::vector<uint8_t>& absorbed_by_tcore,
                const TCoreContext &tctx,
                const SpCompressTreeView &m_view,
                BlockData &blk,
	                const CcData &cc,
	                std::vector<SnarlEndpointPair> *out_snarls,
	                MacroSeriesGccCutsCache *macro_gcc_cuts = nullptr,
	                const std::vector<std::vector<spqr_compat::node>> *tcore_s_gcc_cuts = nullptr)
            {
                auto &C = ctx();
                if (tctx.T_len == 0) return 0;

                uint32_t emitted = 0;

                struct ChildOri {
                    EdgePartType lSign;
                    EdgePartType rSign;
                    uint32_t gcc_ref;
                    uint8_t gcc_kind; // 0 = none, 1 = macro Series, 2 = T_core S-node
                };
                std::vector<ChildOri> oris;
                std::vector<uint32_t> leftPart, rightPart;

                for (uint32_t tn = 0; tn < tctx.T_len; ++tn) {
                    if (tctx.node_types[tn] != SPQR_NODE_TYPE_P) continue;

                    // Poles of this P-node.
                    uint32_t pole0_blk_id = tcore_local_to_block_id(tctx, tn, 0);
                    uint32_t pole1_blk_id = tcore_local_to_block_id(tctx, tn, 1);
                    spqr_compat::node pole0Blk{pole0_blk_id};
                    spqr_compat::node pole1Blk{pole1_blk_id};
                    spqr_compat::node pole0G = blockNodeToOrig(blk, pole0Blk);
                    spqr_compat::node pole1G = blockNodeToOrig(blk, pole1Blk);
                    spqr_compat::node pole0Gcc = blockNodeToCc(blk, pole0Blk);
                    spqr_compat::node pole1Gcc = blockNodeToCc(blk, pole1Blk);

                    auto hasDanglingOutside = [&](spqr_compat::node vGcc) {
                        if (!vGcc) return false;
                        if (!cc.isCutNode[vGcc]) return false;
                        if (cc.badCutCount[vGcc] >= 2) return true;
                        if (cc.badCutCount[vGcc] == 1 && cc.lastBad[vGcc] != blk.bNode) return true;
                        return false;
                    };
                    if (hasDanglingOutside(pole0Gcc) || hasDanglingOutside(pole1Gcc)) continue;

                    // capture poles
                    auto resolve_real_edge = [&](uint32_t block_edge_id, ChildOri& out) {
                        spqr_compat::edge eBlk{block_edge_id};
                        spqr_compat::edge eG = blockEdgeToOrig(blk, eBlk);
                        out.lSign = getNodeEdgeType(pole0G, eG);
                        out.rSign = getNodeEdgeType(pole1G, eG);
                        out.gcc_kind = 0;
                        out.gcc_ref = SPQR_INVALID;
                    };

                    auto resolve_substate = [&](const EdgeDPState& sub, ChildOri& out, bool& ambig) {
                        out.lSign = EdgePartType::NONE;
                        out.rSign = EdgePartType::NONE;
                        out.gcc_kind = 0;
                        out.gcc_ref = SPQR_INVALID;
                        out.lSign = state_sign_at_block_node(sub, pole0Blk);
                        out.rSign = state_sign_at_block_node(sub, pole1Blk);
                        if (state_has_ambiguous_sign_at_block_node(sub, pole0Blk) ||
                            state_has_ambiguous_sign_at_block_node(sub, pole1Blk)) {
                            ambig = true;
                        }
                    };

                    oris.clear();
                    bool ambiguous = false;

                    uint32_t e_start = tctx.skel_offsets[tn];
                    uint32_t e_end = tctx.skel_offsets[tn + 1];
                    oris.reserve(e_end - e_start);

                    uint32_t parent_tn = (tn == tctx.T_root) ? SPQR_INVALID : tctx.node_parents[tn];
                    const bool have_tcore_filter = !absorbed_by_tcore.empty();

                    for (uint32_t i = e_start; i < e_end && !ambiguous; ++i) {
                        const SkeletonEdge& se = tctx.skel_edges[i];

                        if (se.real_edge != SPQR_INVALID) {
                            uint32_t cr = m_view.core_edges_ptr[se.real_edge].child;

                            if (SP_COMPRESS_CHILD_IS_EDGE(cr)) {
                                ChildOri co{EdgePartType::NONE, EdgePartType::NONE, SPQR_INVALID, 0};
                                resolve_real_edge(SP_COMPRESS_CHILD_AS_EDGE(cr), co);
                                oris.push_back(co);
                            } else {
                                uint32_t macro_id = SP_COMPRESS_CHILD_AS_MACRO(cr);
                                if (have_tcore_filter && absorbed_by_tcore[macro_id]) {
                                    // INLINE: iterate this absorbed macro's children.
                                    const SpCompressNode& m = m_view.macros_ptr[macro_id];
                                    for (uint32_t k = 0; k < m.children_count && !ambiguous; ++k) {
                                        uint32_t cref = m_view.children_ptr[m.children_offset + k];
                                        if (SP_COMPRESS_CHILD_IS_EDGE(cref)) {
                                            ChildOri co{EdgePartType::NONE, EdgePartType::NONE, SPQR_INVALID, 0};
                                            resolve_real_edge(SP_COMPRESS_CHILD_AS_EDGE(cref), co);
                                            oris.push_back(co);
                                        } else {
                                            uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                                            ChildOri co{EdgePartType::NONE, EdgePartType::NONE, SPQR_INVALID, 0};
                                            resolve_substate(macro_states[sub_id], co, ambiguous);
                                            if (m_view.macros_ptr[sub_id].kind == SP_COMPRESS_KIND_SERIES) {
                                                co.gcc_kind = 1;
                                                co.gcc_ref = sub_id;
                                            }
                                            if (!ambiguous) oris.push_back(co);
                                        }
                                    }
                                } else {
                                    // Normal macro: use its DP state directly.
                                    ChildOri co{EdgePartType::NONE, EdgePartType::NONE, SPQR_INVALID, 0};
                                    resolve_substate(macro_states[macro_id], co, ambiguous);
                                    if (m_view.macros_ptr[macro_id].kind == SP_COMPRESS_KIND_SERIES) {
                                        co.gcc_kind = 1;
                                        co.gcc_ref = macro_id;
                                    }
                                    if (!ambiguous) oris.push_back(co);
                                }
                            }
                        } else {
                            // Virtual edge.
                            uint32_t v = se.twin_tree_node;
                            const EdgeDPState* sub = nullptr;
                            if (parent_tn != SPQR_INVALID && v == parent_tn) {
                                sub = &down_states[tn];
                            } else {
                                sub = &up_states[v];
                            }
                            ChildOri co{EdgePartType::NONE, EdgePartType::NONE, SPQR_INVALID, 0};
                            resolve_substate(*sub, co, ambiguous);
                            if (tctx.node_types[v] == SPQR_NODE_TYPE_S) {
                                co.gcc_kind = 2;
                                co.gcc_ref = v;
                            }
                            if (!ambiguous) oris.push_back(co);
                        }
                    }

                    if (ambiguous) continue;
                    if (oris.empty()) continue;

                    const uint32_t n_eff = static_cast<uint32_t>(oris.size());
                    for (auto leftSign : {EdgePartType::PLUS, EdgePartType::MINUS}) {
                        for (auto rightSign : {EdgePartType::PLUS, EdgePartType::MINUS}) {
                            leftPart.clear();
                            rightPart.clear();
                            for (uint32_t i = 0; i < n_eff; ++i) {
                                if (oris[i].lSign == leftSign) leftPart.push_back(i);
                                if (oris[i].rSign == rightSign) rightPart.push_back(i);
                            }
                            if (leftPart.empty() || leftPart != rightPart) continue;

                            bool ok = true;
                            if (leftPart.size() == 1) {
                                const ChildOri& only = oris[leftPart[0]];
                                const std::vector<spqr_compat::node>* cuts = nullptr;
                                if (only.gcc_kind == 1 && macro_gcc_cuts) {
                                    cuts = &macro_gcc_cuts->get(only.gcc_ref);
                                } else if (only.gcc_kind == 2 &&
                                           tcore_s_gcc_cuts &&
                                           only.gcc_ref < tcore_s_gcc_cuts->size()) {
                                    cuts = &(*tcore_s_gcc_cuts)[only.gcc_ref];
                                }
                                if (cuts) {
                                    for (spqr_compat::node gccCut : *cuts) {
                                        if (gccCut != pole0Gcc && gccCut != pole1Gcc) {
                                            ok = false;
                                            break;
                                        }
                                    }
                                }
                            }
                            if (!ok) continue;

                            if (out_snarls) {
                                out_snarls->push_back(
                                    {SnarlEndpoint{pole0G, leftSign},
                                     SnarlEndpoint{pole1G, rightSign}});
	                            }
                            emitted++;
                        }
                    }
                }

                return emitted;
            }

            inline uint32_t emit_T_core_s_snarls(
                const std::vector<EdgeDPState> &macro_states,
                const std::vector<EdgeDPState> &up_states,
                const std::vector<EdgeDPState> &down_states,
                const TCoreContext &tctx,
                const SpCompressTreeView &m_view,
                BlockData &blk,
                const CcData &cc,
                std::vector<std::vector<spqr_compat::node>> &tcore_s_gcc_cuts,
                bool count_only)
            {
                auto &C = ctx();
                if (tctx.T_len == 0) return 0;
                if (tcore_s_gcc_cuts.size() != tctx.T_len) {
                    tcore_s_gcc_cuts.assign(tctx.T_len, {});
                }

                struct Incident {
                    uint32_t edge_idx{SPQR_INVALID};
                    uint32_t other{SPQR_INVALID};
                };
                struct CycleSegment {
                    uint32_t src_blk{SPQR_INVALID};
                    uint32_t dst_blk{SPQR_INVALID};
                    uint32_t ref{SPQR_INVALID};
                    uint8_t kind{0}; // 0 = child_ref, 1 = T_core virtual skeleton edge
                };

                std::vector<std::array<Incident, 2>> adj;
                std::vector<uint8_t> deg;
                std::vector<uint32_t> nodes_in_order;
                std::vector<uint32_t> edge_order;
                std::vector<uint32_t> expanded_vertices;
                std::vector<CycleSegment> expanded_segments;
                std::vector<CutSides> cut_sides;

                uint32_t emitted = 0;

                auto child_ref_endpoints = [&](uint32_t cref,
                                               uint32_t& a,
                                               uint32_t& b) {
                    if (SP_COMPRESS_CHILD_IS_EDGE(cref)) {
                        uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cref);
                        spqr_compat::edge eBlk{bid};
                        a = blockEdgeSource(blk, eBlk).idx;
                        b = blockEdgeTarget(blk, eBlk).idx;
                    } else {
                        uint32_t macro_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                        const SpCompressNode& m = m_view.macros_ptr[macro_id];
                        a = m.left;
                        b = m.right;
                    }
                };

                auto append_series_segments = [&](uint32_t macro_id,
                                                  uint32_t from_blk,
                                                  uint32_t to_blk,
                                                  std::vector<CycleSegment>& out) -> bool {
                    const SpCompressNode& m = m_view.macros_ptr[macro_id];
                    if (m.kind != SP_COMPRESS_KIND_SERIES) return false;
                    const bool forward = (from_blk == m.left && to_blk == m.right);
                    const bool backward = (from_blk == m.right && to_blk == m.left);
                    if (!forward && !backward) return false;

                    uint32_t current = from_blk;
                    const uint32_t k = m.children_count;
                    for (uint32_t step = 0; step < k; ++step) {
                        uint32_t child_idx = forward ? step : (k - 1 - step);
                        uint32_t cref = m_view.children_ptr[m.children_offset + child_idx];
                        uint32_t a = SPQR_INVALID, b = SPQR_INVALID;
                        child_ref_endpoints(cref, a, b);
                        uint32_t next = SPQR_INVALID;
                        if (a == current) {
                            next = b;
                        } else if (b == current) {
                            next = a;
                        } else {
                            return false;
                        }
                        out.push_back(CycleSegment{current, next, cref, 0});
                        current = next;
                    }
                    return current == to_blk;
                };

                for (uint32_t tn = 0; tn < tctx.T_len; ++tn) {
                    if (tctx.node_types[tn] != SPQR_NODE_TYPE_S) continue;

                    const uint32_t local_n =
                        tctx.node_mapping_offsets[tn + 1] -
                        tctx.node_mapping_offsets[tn];
                    const uint32_t e_start = tctx.skel_offsets[tn];
                    const uint32_t e_end = tctx.skel_offsets[tn + 1];
                    const uint32_t edge_n = e_end - e_start;
                    if (local_n < 3 || edge_n < 3) continue;

                    std::array<Incident, 2> empty_inc{Incident{}, Incident{}};
                    adj.assign(local_n, empty_inc);
                    deg.assign(local_n, 0);

                    bool valid_cycle = true;
                    auto add_incident = [&](uint32_t local, uint32_t edge_idx, uint32_t other) {
                        if (local >= local_n || other >= local_n || deg[local] >= 2) {
                            valid_cycle = false;
                            return;
                        }
                        adj[local][deg[local]++] = Incident{edge_idx, other};
                    };

                    for (uint32_t i = e_start; i < e_end; ++i) {
                        const SkeletonEdge& se = tctx.skel_edges[i];
                        add_incident(se.src, i, se.dst);
                        add_incident(se.dst, i, se.src);
                    }
                    if (!valid_cycle) continue;
                    for (uint32_t local = 0; local < local_n; ++local) {
                        if (deg[local] != 2) {
                            valid_cycle = false;
                            break;
                        }
                    }
                    if (!valid_cycle) continue;

                    nodes_in_order.clear();
                    edge_order.clear();
                    nodes_in_order.reserve(local_n);
                    edge_order.reserve(local_n);

                    uint32_t prev = adj[0][0].other;
                    uint32_t cur = 0;
                    for (uint32_t step = 0; step < local_n; ++step) {
                        nodes_in_order.push_back(cur);
                        Incident chosen;
                        if (adj[cur][0].other == prev) {
                            chosen = adj[cur][1];
                        } else {
                            chosen = adj[cur][0];
                        }
                        if (chosen.edge_idx == SPQR_INVALID || chosen.other == SPQR_INVALID) {
                            valid_cycle = false;
                            break;
                        }
                        edge_order.push_back(chosen.edge_idx);
                        prev = cur;
                        cur = chosen.other;
                        if (cur == 0) break;
                    }
                    if (!valid_cycle || cur != 0 ||
                        nodes_in_order.size() != edge_order.size() ||
                        nodes_in_order.size() < 3) {
                        continue;
                    }

                    const uint32_t parent_tn =
                        (tn == tctx.T_root) ? SPQR_INVALID : tctx.node_parents[tn];

                    expanded_vertices.clear();
                    expanded_segments.clear();
                    expanded_vertices.reserve(nodes_in_order.size() + 1);
                    expanded_segments.reserve(edge_order.size());

                    const size_t n_base = nodes_in_order.size();
                    expanded_vertices.push_back(
                        tcore_local_to_block_id(tctx, tn, nodes_in_order[0]));

                    for (size_t i = 0; i < n_base; ++i) {
                        uint32_t from_blk =
                            tcore_local_to_block_id(tctx, tn, nodes_in_order[i]);
                        uint32_t to_blk =
                            tcore_local_to_block_id(tctx, tn, nodes_in_order[(i + 1) % n_base]);
                        const SkeletonEdge& se = tctx.skel_edges[edge_order[i]];

                        bool appended = false;
                        if (se.real_edge != SPQR_INVALID) {
                            uint32_t cr = m_view.core_edges_ptr[se.real_edge].child;
                            if (SP_COMPRESS_CHILD_IS_MACRO(cr)) {
                                uint32_t macro_id = SP_COMPRESS_CHILD_AS_MACRO(cr);
                                if (m_view.macros_ptr[macro_id].kind == SP_COMPRESS_KIND_SERIES) {
                                    const size_t before = expanded_segments.size();
                                    appended = append_series_segments(
                                        macro_id, from_blk, to_blk, expanded_segments);
                                    if (!appended) {
                                        expanded_segments.resize(before);
                                    }
                                }
                            }
                            if (!appended) {
                                expanded_segments.push_back(CycleSegment{from_blk, to_blk, cr, 0});
                            }
                        } else {
                            expanded_segments.push_back(
                                CycleSegment{from_blk, to_blk, edge_order[i], 1});
                        }

                        while (expanded_vertices.size() <= expanded_segments.size()) {
                            expanded_vertices.push_back(
                                expanded_segments[expanded_vertices.size() - 1].dst_blk);
                        }
                    }

                    if (expanded_segments.empty() ||
                        expanded_vertices.empty() ||
                        expanded_vertices.back() != expanded_vertices.front()) {
                        continue;
                    }
                    expanded_vertices.pop_back();
                    if (expanded_vertices.size() != expanded_segments.size() ||
                        expanded_vertices.size() < 3) {
                        continue;
                    }

                    auto segment_sign_at = [&](const CycleSegment& seg,
                                               uint32_t blk_id) -> EdgePartType {
                        spqr_compat::node vBlk{blk_id};
                        spqr_compat::node vG = blockNodeToOrig(blk, vBlk);

                        if (seg.kind == 0) {
                            if (SP_COMPRESS_CHILD_IS_EDGE(seg.ref)) {
                                uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(seg.ref);
                                spqr_compat::edge eBlk{bid};
                                spqr_compat::edge eG = blockEdgeToOrig(blk, eBlk);
                                return getNodeEdgeType(vG, eG);
                            }
                            uint32_t macro_id = SP_COMPRESS_CHILD_AS_MACRO(seg.ref);
                            return state_sign_at_block_node(macro_states[macro_id], vBlk);
                        }

                        const SkeletonEdge& se = tctx.skel_edges[seg.ref];
                        uint32_t other_tn = se.twin_tree_node;
                        const EdgeDPState* sub = nullptr;
                        if (parent_tn != SPQR_INVALID && other_tn == parent_tn) {
                            sub = &down_states[tn];
                        } else if (other_tn < up_states.size()) {
                            sub = &up_states[other_tn];
                        }
                        return sub ? state_sign_at_block_node(*sub, vBlk) : EdgePartType::NONE;
                    };

                    auto& cuts = tcore_s_gcc_cuts[tn];
                    cuts.clear();
                    if (!count_only) {
                        cut_sides.clear();
                        cut_sides.reserve(8);
                    }
                    uint32_t cut_count = 0;
                    SnarlEndpoint first_prev_side;
                    SnarlEndpoint prev_next_side;

                    const size_t n = expanded_vertices.size();
                    for (size_t i = 0; i < n; ++i) {
                        uint32_t u_blk_id = expanded_vertices[i];
                        spqr_compat::node uBlk{u_blk_id};
                        spqr_compat::node uGcc = blockNodeToCc(blk, uBlk);
                        if (!uGcc) continue;

                        bool nodeIsCut =
                            (cc.isCutNode[uGcc] && cc.badCutCount[uGcc] == 1) ||
                            (!cc.isCutNode[uGcc]);
                        if (!nodeIsCut) continue;

                        EdgePartType t0 =
                            segment_sign_at(expanded_segments[(i + n - 1) % n], u_blk_id);
                        EdgePartType t1 =
                            segment_sign_at(expanded_segments[i], u_blk_id);

                        nodeIsCut = (t0 != EdgePartType::NONE &&
                                     t1 != EdgePartType::NONE &&
                                     t0 != t1);
                        if (!nodeIsCut) continue;

                        if (cuts.size() < 3) {
                            cuts.push_back(uGcc);
                        }

                        ++cut_count;
                        if (!count_only) {
                            spqr_compat::node uG = cc.nodeToOrig[uGcc];
                            SnarlEndpoint prev_side{uG, t0};
                            SnarlEndpoint next_side{uG, t1};
                            cut_sides.push_back(CutSides{prev_side, next_side});
                            if (cut_count == 1) {
                                first_prev_side = prev_side;
                                prev_next_side = next_side;
                            } else {
                                addSnarlTaggedPairNodes(
                                    "S",
                                    prev_next_side.node, prev_next_side.sign,
                                    prev_side.node, prev_side.sign);
                                prev_next_side = next_side;
                            }
                        }
                    }

                    if (cut_count > 1) {
                        emitted += cut_count;
                        if (!count_only) {
                            addSnarlTaggedPairNodes(
                                "S",
                                prev_next_side.node, prev_next_side.sign,
                                first_prev_side.node, first_prev_side.sign);
                            emitted += emitMixedSelfLoopComposedS(
                                cut_sides, true,
                                [&](const SnarlEndpoint& first, const SnarlEndpoint& second) {
                                    addSnarlTaggedPairNodes(
                                        "S",
                                        first.node, first.sign,
                                        second.node, second.sign);
                                });
                        }
                    }
                }

                return emitted;
            }

            inline uint32_t emit_T_core_rr_snarls(
                const std::vector<EdgeDPState> &up_states,
                const std::vector<EdgeDPState> &down_states,
                const TCoreContext &tctx,
                BlockData &blk,
                const CcData &cc,
                bool count_only)
            {
                auto &C = ctx();
                if (tctx.T_len == 0) return 0;

                uint32_t emitted = 0;

                auto hasDanglingOutside = [&](spqr_compat::node vGcc) {
                    if (!vGcc) return false;
                    if (!cc.isCutNode[vGcc]) return false;
                    if (cc.badCutCount[vGcc] >= 2) return true;
                    if (cc.badCutCount[vGcc] == 1 && cc.lastBad[vGcc] != blk.bNode) return true;
                    return false;
                };

                auto type_at_pole = [](const EdgeDPState& state,
                                       spqr_compat::node poleBlk) -> EdgePartType {
                    if (state.s == poleBlk) {
                        return state.localPlusS > 0 ? EdgePartType::PLUS : EdgePartType::MINUS;
                    }
                    return state.localPlusT > 0 ? EdgePartType::PLUS : EdgePartType::MINUS;
                };

                for (uint32_t tn = 0; tn < tctx.T_len; ++tn) {
                    if (tn == tctx.T_root) continue;
                    uint32_t parent = tctx.node_parents[tn];
                    if (parent == SPQR_INVALID || parent >= tctx.T_len) continue;
                    if (tctx.node_types[tn] != SPQR_NODE_TYPE_R ||
                        tctx.node_types[parent] != SPQR_NODE_TYPE_R) {
                        continue;
                    }

                    const EdgeDPState& down = up_states[tn];
                    const EdgeDPState& up = down_states[tn];

                    if (!down.s || !down.t || !up.s || !up.t) continue;

                    spqr_compat::node pole0Blk = down.s;
                    spqr_compat::node pole1Blk = down.t;
                    if (!state_has_block_node(up, pole0Blk) ||
                        !state_has_block_node(up, pole1Blk)) {
                        continue;
                    }

                    spqr_compat::node pole0Gcc = blockNodeToCc(blk, pole0Blk);
                    spqr_compat::node pole1Gcc = blockNodeToCc(blk, pole1Blk);
                    if (!pole0Gcc || !pole1Gcc) continue;
                    if (hasDanglingOutside(pole0Gcc) ||
                        hasDanglingOutside(pole1Gcc)) {
                        continue;
                    }

                    if ((up.localMinusS > 0 && up.localPlusS > 0) ||
                        (up.localMinusT > 0 && up.localPlusT > 0) ||
                        (down.localMinusS > 0 && down.localPlusS > 0) ||
                        (down.localMinusT > 0 && down.localPlusT > 0)) {
                        continue;
                    }

                    EdgePartType pole0DownType = type_at_pole(down, pole0Blk);
                    EdgePartType pole0UpType = type_at_pole(up, pole0Blk);
                    EdgePartType pole1DownType = type_at_pole(down, pole1Blk);
                    EdgePartType pole1UpType = type_at_pole(up, pole1Blk);

                    if (pole0DownType == pole0UpType ||
                        pole1DownType == pole1UpType) {
                        continue;
                    }

                    if (!count_only) {
                        addSnarlTaggedPairNodes(
                            "RR",
                            cc.nodeToOrig[pole0Gcc], pole0DownType,
                            cc.nodeToOrig[pole1Gcc], pole1DownType);
                    }
                    ++emitted;

                    if (!count_only) {
                        addSnarlTaggedPairNodes(
                            "RR",
                            cc.nodeToOrig[pole0Gcc], pole0UpType,
                            cc.nodeToOrig[pole1Gcc], pole1UpType);
                    }
                    ++emitted;
                }

                return emitted;
            }

            inline std::vector<uint8_t> compute_macro_absorption_by_tcore(
                const SpCompressTreeView& m_view,
                const TCoreContext& tctx)
            {
                std::vector<uint8_t> absorbed(m_view.macros_len, 0);

                if (tctx.T_len == 0) return absorbed;

                for (uint32_t tn = 0; tn < tctx.T_len; ++tn) {
                    if (tctx.node_types[tn] != SPQR_NODE_TYPE_P) continue;

                    uint32_t pole0_blk_id = tcore_local_to_block_id(tctx, tn, 0);
                    uint32_t pole1_blk_id = tcore_local_to_block_id(tctx, tn, 1);

                    uint32_t e_start = tctx.skel_offsets[tn];
                    uint32_t e_end = tctx.skel_offsets[tn + 1];

                    for (uint32_t i = e_start; i < e_end; ++i) {
                        const SkeletonEdge& se = tctx.skel_edges[i];
                        if (se.real_edge == SPQR_INVALID) continue;

                        uint32_t cr = m_view.core_edges_ptr[se.real_edge].child;
                        if (!SP_COMPRESS_CHILD_IS_MACRO(cr)) continue;

                        uint32_t macro_id = SP_COMPRESS_CHILD_AS_MACRO(cr);
                        const SpCompressNode& m = m_view.macros_ptr[macro_id];
                        if (m.kind != SP_COMPRESS_KIND_PARALLEL) continue;

                        // Same poles in either orientation?
                        bool same_poles =
                            (m.left == pole0_blk_id && m.right == pole1_blk_id) ||
                            (m.left == pole1_blk_id && m.right == pole0_blk_id);
                        if (same_poles) {
                            absorbed[macro_id] = 1;
                        }
                    }
                }
                return absorbed;
            }

            inline std::vector<uint8_t> compute_series_inlined_in_tcore_s(
                const SpCompressTreeView& m_view,
                const TCoreContext& tctx)
            {
                std::vector<uint8_t> inlined(m_view.macros_len, 0);
                if (tctx.T_len == 0) return inlined;

                for (uint32_t tn = 0; tn < tctx.T_len; ++tn) {
                    if (tctx.node_types[tn] != SPQR_NODE_TYPE_S) continue;

                    uint32_t e_start = tctx.skel_offsets[tn];
                    uint32_t e_end = tctx.skel_offsets[tn + 1];

                    for (uint32_t i = e_start; i < e_end; ++i) {
                        const SkeletonEdge& se = tctx.skel_edges[i];
                        if (se.real_edge == SPQR_INVALID) continue;

                        uint32_t cr = m_view.core_edges_ptr[se.real_edge].child;
                        if (!SP_COMPRESS_CHILD_IS_MACRO(cr)) continue;

                        uint32_t macro_id = SP_COMPRESS_CHILD_AS_MACRO(cr);
                        if (macro_id >= m_view.macros_len) continue;
                        const SpCompressNode& m = m_view.macros_ptr[macro_id];
                        if (m.kind != SP_COMPRESS_KIND_SERIES) continue;
                        inlined[macro_id] = 1;
                    }
                }
                return inlined;
            }

            inline std::vector<uint8_t> compute_macro_series_s_targets(
                const std::vector<EdgeDPState>& macro_states,
                const SpCompressTreeView& m_view,
                BlockData& blk,
                const CcData& cc,
                uint32_t& target_count)
            {
                const uint32_t M = m_view.macros_len;
                std::vector<uint8_t> targets(M, 0);
                target_count = 0;

                auto child_ref_endpoints = [&](uint32_t cref, uint32_t& a, uint32_t& b) {
                    if (SP_COMPRESS_CHILD_IS_EDGE(cref)) {
                        uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cref);
                        spqr_compat::edge eBlk{bid};
                        a = blockEdgeSource(blk, eBlk).idx;
                        b = blockEdgeTarget(blk, eBlk).idx;
                    } else {
                        uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                        const SpCompressNode& sub = m_view.macros_ptr[sub_id];
                        a = sub.left;
                        b = sub.right;
                    }
                };

                auto orient_at = [&](uint32_t cref, spqr_compat::node uBlk) -> EdgePartType {
                    spqr_compat::node uG = blockNodeToOrig(blk, uBlk);
                    if (SP_COMPRESS_CHILD_IS_EDGE(cref)) {
                        uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cref);
                        spqr_compat::edge eBlk{bid};
                        spqr_compat::edge eG = blockEdgeToOrig(blk, eBlk);
                        return getNodeEdgeType(uG, eG);
                    }
                    uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                    if (sub_id >= macro_states.size()) return EdgePartType::NONE;
                    const EdgeDPState& sub = macro_states[sub_id];
                    if (state_has_ambiguous_sign_at_block_node(sub, uBlk)) {
                        return EdgePartType::NONE;
                    }
                    return state_sign_at_block_node(sub, uBlk);
                };

                std::vector<uint32_t> vertices;
                std::vector<uint32_t> child_order;

                for (uint32_t m_id = 0; m_id < M; ++m_id) {
                    const SpCompressNode& m = m_view.macros_ptr[m_id];
                    if (m.kind != SP_COMPRESS_KIND_SERIES) continue;
                    if (m.children_count < 2) continue;

                    vertices.clear();
                    child_order.clear();
                    vertices.reserve(static_cast<size_t>(m.children_count) + 1);
                    child_order.reserve(m.children_count);

                    uint32_t current = m.left;
                    vertices.push_back(current);
                    bool valid_chain = true;
                    for (uint32_t i = 0; i < m.children_count; ++i) {
                        uint32_t cref = m_view.children_ptr[m.children_offset + i];
                        uint32_t a = SPQR_INVALID, b = SPQR_INVALID;
                        child_ref_endpoints(cref, a, b);
                        uint32_t next = SPQR_INVALID;
                        if (a == current) {
                            next = b;
                        } else if (b == current) {
                            next = a;
                        } else {
                            valid_chain = false;
                            break;
                        }
                        child_order.push_back(i);
                        vertices.push_back(next);
                        current = next;
                    }
                    if (!valid_chain || current != m.right ||
                        vertices.size() != static_cast<size_t>(m.children_count) + 1) {
                        continue;
                    }

                    uint32_t possible_cuts = 0;

                    for (uint32_t vi = 1; vi + 1 < vertices.size(); ++vi) {
                        uint32_t v_blk_id = vertices[vi];
                        spqr_compat::node uBlk{v_blk_id};
                        spqr_compat::node uGcc = blockNodeToCc(blk, uBlk);
                        if (!uGcc) continue;

                        bool node_is_cut =
                            (cc.isCutNode[uGcc] && cc.badCutCount[uGcc] == 1) ||
                            (!cc.isCutNode[uGcc]);
                        if (!node_is_cut) continue;

                        uint32_t prev_child_idx = child_order[vi - 1];
                        uint32_t next_child_idx = child_order[vi];
                        uint32_t prev_ref =
                            m_view.children_ptr[m.children_offset + prev_child_idx];
                        uint32_t next_ref =
                            m_view.children_ptr[m.children_offset + next_child_idx];
                        EdgePartType prev_sign = orient_at(prev_ref, uBlk);
                        EdgePartType next_sign = orient_at(next_ref, uBlk);

                        if (prev_sign == EdgePartType::NONE ||
                            next_sign == EdgePartType::NONE ||
                            prev_sign == next_sign) {
                            continue;
                        }

                        ++possible_cuts;
                    }

                    auto pole_possible = [&](uint32_t vi, uint32_t child_idx) -> bool {
                        uint32_t v_blk_id = vertices[vi];
                        spqr_compat::node uBlk{v_blk_id};
                        spqr_compat::node uGcc = blockNodeToCc(blk, uBlk);
                        if (!uGcc) return false;
                        bool node_is_cut =
                            (cc.isCutNode[uGcc] && cc.badCutCount[uGcc] == 1) ||
                            (!cc.isCutNode[uGcc]);
                        if (!node_is_cut) return false;
                        uint32_t ref =
                            m_view.children_ptr[m.children_offset + child_idx];
                        return orient_at(ref, uBlk) != EdgePartType::NONE;
                    };

                    if (pole_possible(0, child_order.front())) ++possible_cuts;
                    if (pole_possible(vertices.size() - 1, child_order.back())) ++possible_cuts;

                    if (possible_cuts >= 2) {
                        targets[m_id] = 1;
                        ++target_count;
                    }
                }

                return targets;
            }

            inline uint32_t emit_macro_series_s_snarls_range(
                const std::vector<EdgeDPState>& macro_states,
                const SpCompressTreeView& m_view,
                BlockData& blk,
                const CcData& cc,
                const std::vector<uint8_t>& targets,
                const std::vector<EdgeDPState>& macro_down_states,
                const std::vector<uint8_t>& macro_down_has_state,
                bool count_only,
                const std::vector<uint8_t>* inlined_in_tcore_s,
                uint32_t begin_m_id,
                uint32_t end_m_id,
                std::vector<SnarlEndpointPair>* out_snarls)
            {
                const uint32_t M = m_view.macros_len;
                end_m_id = std::min(end_m_id, M);

                struct SeriesChild {
                    uint32_t ref{SPQR_INVALID};
                    uint32_t a{SPQR_INVALID};
                    uint32_t b{SPQR_INVALID};
                };

                auto child_ref_endpoints = [&](uint32_t cref, uint32_t& a, uint32_t& b) {
                    if (SP_COMPRESS_CHILD_IS_EDGE(cref)) {
                        uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cref);
                        spqr_compat::edge eBlk{bid};
                        a = blockEdgeSource(blk, eBlk).idx;
                        b = blockEdgeTarget(blk, eBlk).idx;
                    } else {
                        uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                        const SpCompressNode& sub = m_view.macros_ptr[sub_id];
                        a = sub.left;
                        b = sub.right;
                    }
                };

                auto orient_at = [&](uint32_t cref, spqr_compat::node uBlk) -> EdgePartType {
                    spqr_compat::node uG = blockNodeToOrig(blk, uBlk);
                    if (SP_COMPRESS_CHILD_IS_EDGE(cref)) {
                        uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cref);
                        spqr_compat::edge eBlk{bid};
                        spqr_compat::edge eG = blockEdgeToOrig(blk, eBlk);
                        return getNodeEdgeType(uG, eG);
                    }
                    uint32_t sub_id = SP_COMPRESS_CHILD_AS_MACRO(cref);
                    if (sub_id >= macro_states.size()) return EdgePartType::NONE;
                    const EdgeDPState& sub = macro_states[sub_id];
                    if (state_has_ambiguous_sign_at_block_node(sub, uBlk)) {
                        return EdgePartType::NONE;
                    }
                    return state_sign_at_block_node(sub, uBlk);
                };

                uint32_t emitted = 0;
                std::vector<SeriesChild> children;
                std::vector<std::pair<uint32_t, uint32_t>> incidence;
                std::vector<uint32_t> vertices;
                std::vector<uint32_t> child_order;
                std::vector<uint8_t> visited;
                std::vector<CutSides> cuts;

                auto build_series_path = [&](const SpCompressNode& m) -> bool {
                    const uint32_t k = m.children_count;
                    children.clear();
                    incidence.clear();
                    vertices.clear();
                    child_order.clear();
                    visited.clear();

                    children.reserve(k);
                    incidence.reserve(static_cast<size_t>(k) * 2);
                    for (uint32_t i = 0; i < k; ++i) {
                        uint32_t cref = m_view.children_ptr[m.children_offset + i];
                        uint32_t a = SPQR_INVALID, b = SPQR_INVALID;
                        child_ref_endpoints(cref, a, b);
                        if (a == SPQR_INVALID || b == SPQR_INVALID) return false;
                        children.push_back(SeriesChild{cref, a, b});
                        incidence.emplace_back(a, i);
                        incidence.emplace_back(b, i);
                    }

                    if (k == 2) {
                        auto try_order = [&](uint32_t first,
                                             uint32_t second) -> bool {
                            const SeriesChild& c0 = children[first];
                            uint32_t mid = SPQR_INVALID;
                            if (c0.a == m.left) mid = c0.b;
                            else if (c0.b == m.left) mid = c0.a;
                            else return false;

                            const SeriesChild& c1 = children[second];
                            uint32_t end = SPQR_INVALID;
                            if (c1.a == mid) end = c1.b;
                            else if (c1.b == mid) end = c1.a;
                            else return false;

                            if (end != m.right) return false;
                            vertices.push_back(m.left);
                            vertices.push_back(mid);
                            vertices.push_back(m.right);
                            child_order.push_back(first);
                            child_order.push_back(second);
                            return true;
                        };

                        if (try_order(0, 1)) return true;
                        vertices.clear();
                        child_order.clear();
                        if (try_order(1, 0)) return true;
                        vertices.clear();
                        child_order.clear();
                        return false;
                    }

                    std::sort(incidence.begin(), incidence.end());
                    visited.assign(k, 0);
                    vertices.reserve(static_cast<size_t>(k) + 1);
                    child_order.reserve(k);

                    auto incident_children = [&](uint32_t node_id,
                                                 std::array<uint32_t, 3>& out,
                                                 uint32_t& n_out) -> bool {
                        n_out = 0;
                        auto it = std::lower_bound(
                            incidence.begin(), incidence.end(), node_id,
                            [](const std::pair<uint32_t, uint32_t>& p, uint32_t v) {
                                return p.first < v;
                            });
                        for (; it != incidence.end() && it->first == node_id; ++it) {
                            if (n_out >= out.size()) return false;
                            out[n_out++] = it->second;
                        }
                        return n_out > 0;
                    };

                    uint32_t current = m.left;
                    uint32_t prev_child = SPQR_INVALID;
                    vertices.push_back(current);

                    for (uint32_t step = 0; step < k; ++step) {
                        std::array<uint32_t, 3> inc{{SPQR_INVALID, SPQR_INVALID, SPQR_INVALID}};
                        uint32_t n_inc = 0;
                        if (!incident_children(current, inc, n_inc)) return false;

                        uint32_t chosen = SPQR_INVALID;
                        for (uint32_t j = 0; j < n_inc; ++j) {
                            uint32_t child_idx = inc[j];
                            if (child_idx == prev_child) continue;
                            if (child_idx >= k || visited[child_idx]) continue;
                            chosen = child_idx;
                            break;
                        }
                        if (chosen == SPQR_INVALID) return false;

                        const SeriesChild& ch = children[chosen];
                        uint32_t next = SPQR_INVALID;
                        if (ch.a == current) next = ch.b;
                        else if (ch.b == current) next = ch.a;
                        else return false;

                        visited[chosen] = 1;
                        child_order.push_back(chosen);
                        vertices.push_back(next);
                        prev_child = chosen;
                        current = next;
                    }

                    if (current != m.right) return false;
                    for (uint8_t v : visited) {
                        if (!v) return false;
                    }
                    return vertices.size() == static_cast<size_t>(k) + 1 &&
                           child_order.size() == k;
                };

                auto emit_s_macro_pair = [&](const SnarlEndpoint& first,
                                             const SnarlEndpoint& second) {
                    if (out_snarls) {
                        out_snarls->push_back({first, second});
                    } else {
                        addSnarlTaggedPairNodes(
                            "S:macro",
                            first.node,
                            first.sign,
                            second.node,
                            second.sign);
                    }
                };

                for (uint32_t m_id = begin_m_id; m_id < end_m_id; ++m_id) {
                    if (!targets.empty() &&
                        (m_id >= targets.size() || !targets[m_id])) {
                        continue;
                    }
                    if (inlined_in_tcore_s &&
                        m_id < inlined_in_tcore_s->size() &&
                        (*inlined_in_tcore_s)[m_id]) {
                        continue;
                    }
                    const SpCompressNode& m = m_view.macros_ptr[m_id];
                    if (m.kind != SP_COMPRESS_KIND_SERIES) continue;
                    if (m.children_count < 2) continue;

                    if (!build_series_path(m)) {
                        continue;
                    }

                    const bool is_closed_cycle = (m.left == m.right);

                    const bool have_down =
                        !is_closed_cycle &&
                        m_id < macro_down_states.size() &&
                        m_id < macro_down_has_state.size() &&
                        macro_down_has_state[m_id];
                    const EdgeDPState* down =
                        have_down ? &macro_down_states[m_id] : nullptr;

                    if (!count_only) {
                        cuts.clear();
                        cuts.reserve(8);
                    }
                    uint32_t cut_count = 0;

                    if (is_closed_cycle) {
                        const size_t k = child_order.size();
                        for (size_t vi = 0; vi < k; ++vi) {
                            uint32_t v_blk_id = vertices[vi];
                            spqr_compat::node uBlk{v_blk_id};
                            spqr_compat::node uGcc = blockNodeToCc(blk, uBlk);
                            if (!uGcc) continue;

                            bool node_is_cut =
                                (cc.isCutNode[uGcc] && cc.badCutCount[uGcc] == 1) ||
                                (!cc.isCutNode[uGcc]);
                            if (!node_is_cut) continue;

                            const size_t prev_idx = (vi + k - 1) % k;
                            const size_t next_idx = vi % k;
                            EdgePartType prev_sign =
                                orient_at(children[child_order[prev_idx]].ref, uBlk);
                            EdgePartType next_sign =
                                orient_at(children[child_order[next_idx]].ref, uBlk);

                            if (prev_sign == EdgePartType::NONE ||
                                next_sign == EdgePartType::NONE ||
                                prev_sign == next_sign) {
                                continue;
                            }

                            ++cut_count;
                            if (!count_only) {
                                spqr_compat::node uG = cc.nodeToOrig[uGcc];
                                cuts.push_back(CutSides{
                                    SnarlEndpoint{uG, prev_sign},
                                    SnarlEndpoint{uG, next_sign}});
                            }
                        }

                        if (cut_count <= 1) continue;
                        emitted += cut_count;
                        if (!count_only) {
                            for (size_t i = 1; i < cuts.size(); ++i) {
                                emit_s_macro_pair(
                                    cuts[i - 1].next_side,
                                    cuts[i].prev_side);
                            }
                            emit_s_macro_pair(
                                cuts.back().next_side,
                                cuts.front().prev_side);
                            emitted += emitMixedSelfLoopComposedS(
                                cuts, true, emit_s_macro_pair);
                        }
                        continue;
                    }

                    for (uint32_t vi = 0; vi < vertices.size(); ++vi) {
                        uint32_t v_blk_id = vertices[vi];
                        spqr_compat::node uBlk{v_blk_id};
                        spqr_compat::node uGcc = blockNodeToCc(blk, uBlk);
                        if (!uGcc) continue;

                        bool node_is_cut =
                            (cc.isCutNode[uGcc] && cc.badCutCount[uGcc] == 1) ||
                            (!cc.isCutNode[uGcc]);
                        if (!node_is_cut) continue;

                        const bool at_left_pole = (vi == 0);
                        const bool at_right_pole = (vi + 1 == vertices.size());
                        EdgePartType prev_sign = EdgePartType::NONE;
                        EdgePartType next_sign = EdgePartType::NONE;

                        if (at_left_pole) {
                            if (down && !state_has_ambiguous_sign_at_block_node(*down, uBlk)) {
                                prev_sign = state_sign_at_block_node(*down, uBlk);
                            }
                        } else {
                            uint32_t prev_child_idx = child_order[vi - 1];
                            prev_sign = orient_at(children[prev_child_idx].ref, uBlk);
                        }

                        if (at_right_pole) {
                            if (down && !state_has_ambiguous_sign_at_block_node(*down, uBlk)) {
                                next_sign = state_sign_at_block_node(*down, uBlk);
                            }
                        } else {
                            uint32_t next_child_idx = child_order[vi];
                            next_sign = orient_at(children[next_child_idx].ref, uBlk);
                        }

                        if (prev_sign == EdgePartType::NONE ||
                            next_sign == EdgePartType::NONE ||
                            prev_sign == next_sign) {
                            continue;
                        }

                            ++cut_count;
                            if (!count_only) {
                            spqr_compat::node uG = cc.nodeToOrig[uGcc];
                            cuts.push_back(CutSides{
                                SnarlEndpoint{uG, prev_sign},
                                SnarlEndpoint{uG, next_sign}});
                        }
                    }

                    if (cut_count <= 1) continue;
                    const uint32_t linear_pairs = cut_count - 1;
                    const uint32_t wrap_pairs = have_down ? 1u : 0u;
                    emitted += linear_pairs + wrap_pairs;

                    if (!count_only) {
                        for (size_t i = 1; i < cuts.size(); ++i) {
                            emit_s_macro_pair(
                                cuts[i - 1].next_side,
                                cuts[i].prev_side);
                        }
                        if (have_down) {
                            emit_s_macro_pair(
                                cuts.back().next_side,
                                cuts.front().prev_side);
                        }
                        emitted += emitMixedSelfLoopComposedS(
                            cuts, have_down, emit_s_macro_pair);
                    }
                }

                return emitted;
            }

            inline uint32_t emit_macro_series_s_snarls(
                const std::vector<EdgeDPState>& macro_states,
                const SpCompressTreeView& m_view,
                BlockData& blk,
                const CcData& cc,
                const std::vector<uint8_t>& targets,
                const std::vector<EdgeDPState>& macro_down_states,
                const std::vector<uint8_t>& macro_down_has_state,
                bool count_only,
                const std::vector<uint8_t>* inlined_in_tcore_s = nullptr,
                int num_threads = 1)
            {
                const uint32_t M = m_view.macros_len;
                const size_t workers =
                    std::min(static_cast<size_t>(M),
                             static_cast<size_t>(std::max(1, num_threads)));
                const bool parallel =
                    workers > 1 && M >= 1000000u;

                if (!parallel) {
                    return emit_macro_series_s_snarls_range(
                        macro_states, m_view, blk, cc, targets,
                        macro_down_states, macro_down_has_state,
                        count_only, inlined_in_tcore_s,
                        0, M, nullptr);
                }

                std::vector<uint32_t> local_counts(workers, 0);
                std::vector<std::vector<SnarlEndpointPair>> local_snarls;
                if (!count_only) {
                    local_snarls.resize(workers);
                }

                std::vector<std::thread> threads;
                threads.reserve(workers);
                for (size_t tid = 0; tid < workers; ++tid) {
                    const uint32_t begin =
                        static_cast<uint32_t>((static_cast<uint64_t>(M) * tid) / workers);
                    const uint32_t end =
                        static_cast<uint32_t>((static_cast<uint64_t>(M) * (tid + 1)) / workers);
                    threads.emplace_back([&, tid, begin, end]() {
                        std::vector<SnarlEndpointPair>* out =
                            count_only ? nullptr : &local_snarls[tid];
                        local_counts[tid] = emit_macro_series_s_snarls_range(
                            macro_states, m_view, blk, cc, targets,
                            macro_down_states, macro_down_has_state,
                            count_only, inlined_in_tcore_s,
                            begin, end, out);
                    });
                }
                for (auto& t : threads) {
                    t.join();
                }

                uint32_t emitted = 0;
                for (size_t tid = 0; tid < workers; ++tid) {
                    emitted += local_counts[tid];
                    if (!count_only) {
                        for (const auto& p : local_snarls[tid]) {
                            addSnarlTaggedPairNodes(
                                "S:macro",
                                p.first.node,
                                p.first.sign,
                                p.second.node,
                                p.second.sign);
                        }
                    }
                }
                return emitted;
            }

            void printAllStates(const spqr_compat::NodeArray<NodeDPState> &node_dp,
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
                std::vector<spqr_compat::edge> &edge_order,
                std::vector<spqr_compat::node> &node_order,
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

            void processEdge(spqr_compat::edge curr_edge,
                             spqr_compat::EdgeArray<EdgeDP> &dp,
                             const CcData &cc,
                             BlockData &blk)
            {
                BF_INSTR(
                profiling_patch::pe_total_calls.fetch_add(1, std::memory_order_relaxed);
                auto __pe_tA_start = std::chrono::high_resolution_clock::now();
                )

                auto &C = ctx();

                EdgeDPState &state = dp[curr_edge].down;
                EdgeDPState &back_state = dp[curr_edge].up;

                const StaticSPQRTree &spqr = *blk.spqr;
                const auto &T = spqr.tree();

                spqr_compat::node u = T.source(curr_edge);
                spqr_compat::node v = T.target(curr_edge);

                spqr_compat::node A = nullptr; 
                spqr_compat::node B = nullptr; 

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
                    SPQR_RUST_ASSERT(false);
                    return;
                }

                state.localPlusS = 0;
                state.localPlusT = 0;
                state.localMinusS = 0;
                state.localMinusT = 0;

                const Skeleton &skel = spqr.skeleton(B); 
                const auto &skelGraph = skel.getGraph();

                auto mapSkeletonToGlobal = [&](spqr_compat::node vSkel) -> spqr_compat::node
                {
                    if (!vSkel)
                        return nullptr;
                    spqr_compat::node vBlk = skel.original(vSkel);
                    if (!vBlk)
                        return nullptr;
                    spqr_compat::node vCc = blockNodeToCc(blk, vBlk);
                    if (!vCc)
                        return nullptr;
                    return cc.nodeToOrig[vCc];
                };

                for (spqr_compat::edge e : skelGraph.edges)
                {
                    spqr_compat::node uSk = skelGraph.source(e);
                    spqr_compat::node vSk = skelGraph.target(e);

                    spqr_compat::node D = skel.twinTreeNode(e);

                    if (D == A)
                    {
                        spqr_compat::node vBlk = skel.original(vSk);
                        spqr_compat::node uBlk = skel.original(uSk);

                        state.s = back_state.s = vBlk;
                        state.t = back_state.t = uBlk;
                        break;
                    }
                }

                BF_INSTR(
                auto __pe_tB_start = std::chrono::high_resolution_clock::now();
                profiling_patch::pe_A_setup_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__pe_tB_start - __pe_tA_start).count(),
                    std::memory_order_relaxed);
                )

                for (spqr_compat::edge e : skelGraph.edges)
                {
                    spqr_compat::node uSk = skelGraph.source(e);
                    spqr_compat::node vSk = skelGraph.target(e);

                    spqr_compat::node uBlk = skel.original(uSk);
                    spqr_compat::node vBlk = skel.original(vSk);

                    if (!skel.isVirtual(e))
                    {
                        spqr_compat::edge eG = blockEdgeToOrig(blk, skel.realEdge(e));

                        spqr_compat::node uG = C.G.source(eG);
                        spqr_compat::node vG = C.G.target(eG);

                        if (uG == blockNodeToOrig(blk, state.s))
                        {
                            auto t = getNodeEdgeType(uG, eG);
                            if (t == EdgePartType::PLUS)
                                state.localPlusS++;
                            else if (t == EdgePartType::MINUS)
                                state.localMinusS++;
                        }

                        if (vG == blockNodeToOrig(blk, state.s))
                        {
                            auto t = getNodeEdgeType(vG, eG);
                            if (t == EdgePartType::PLUS)
                                state.localPlusS++;
                            else if (t == EdgePartType::MINUS)
                                state.localMinusS++;
                        }

                        if (uG == blockNodeToOrig(blk, state.t))
                        {
                            auto t = getNodeEdgeType(uG, eG);
                            if (t == EdgePartType::PLUS)
                                state.localPlusT++;
                            else if (t == EdgePartType::MINUS)
                                state.localMinusT++;
                        }

                        if (vG == blockNodeToOrig(blk, state.t))
                        {
                            auto t = getNodeEdgeType(vG, eG);
                            if (t == EdgePartType::PLUS)
                                state.localPlusT++;
                            else if (t == EdgePartType::MINUS)
                                state.localMinusT++;
                        }

                        continue;
                    }

                    spqr_compat::node D = skel.twinTreeNode(e);

                    if (D == A)
                    {
                        continue;
                    }

                    spqr_compat::edge treeE = blk.skel2tree.at(e);
                    SPQR_RUST_ASSERT(treeE != nullptr);

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
            
                BF_INSTR(
                auto __pe_tB_end = std::chrono::high_resolution_clock::now();
                profiling_patch::pe_B_build_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__pe_tB_end - __pe_tB_start).count(),
                    std::memory_order_relaxed);
                )
            }

            void processNode(spqr_compat::node curr_node,
                             spqr_compat::EdgeArray<EdgeDP> &edge_dp,
                             const CcData & /*cc*/,
                             BlockData &blk)
            {
                BF_INSTR(
                profiling_patch::pn_total_calls.fetch_add(1, std::memory_order_relaxed);
                auto __pn_tA_start = std::chrono::high_resolution_clock::now();
                )

                auto& C = ctx();
                spqr_compat::node A = curr_node;
                const StaticSPQRTree &spqr = *blk.spqr;
                const Skeleton &skel = spqr.skeleton(A);
                const auto &skelG = skel.getGraph();

                struct VirtEdgeRef { EdgeDPState *toUpdate; EdgeDPState *opposite; };
                thread_local std::vector<int> tls_localPlusDeg;
                thread_local std::vector<int> tls_localMinusDeg;
                thread_local std::vector<VirtEdgeRef> tls_virtualEdges;

                uint32_t nSkelMax = 0;
                for (spqr_compat::node h : skelG.nodes) {
                    if (h.idx + 1 > nSkelMax) nSkelMax = h.idx + 1;
                }

                if (tls_localPlusDeg.size() < nSkelMax) {
                    tls_localPlusDeg.resize(nSkelMax);
                    tls_localMinusDeg.resize(nSkelMax);
                }
                tls_virtualEdges.clear();

                for (spqr_compat::node h : skelG.nodes) {
                    tls_localPlusDeg [h.idx] = 0;
                    tls_localMinusDeg[h.idx] = 0;
                    spqr_compat::node vB = skel.original(h);
                    blk.blkToSkel[vB] = h;
                }

                BF_INSTR(
                auto __pn_tB_start = std::chrono::high_resolution_clock::now();
                profiling_patch::pn_A_setup_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__pn_tB_start - __pn_tA_start).count(),
                    std::memory_order_relaxed);
                )

                const spqr_compat::node parent_of_A = blk.parent(A);

                skel.forEachEdge([&](spqr_compat::edge e, spqr_compat::node u, spqr_compat::node v) {
                    if (!skel.isVirtual(e)) {
                        spqr_compat::edge eG = blockEdgeToOrig(blk, skel.realEdge(e));
                        spqr_compat::node uG = C.G.source(eG);
                        spqr_compat::node vG = C.G.target(eG);

                        spqr_compat::node uOrig = blockNodeToOrig(blk, skel.original(u));
                        if (uOrig == uG) {
                            tls_localPlusDeg [u.idx] += (getNodeEdgeType(uG, eG) == EdgePartType::PLUS);
                            tls_localMinusDeg[u.idx] += (getNodeEdgeType(uG, eG) == EdgePartType::MINUS);
                            tls_localPlusDeg [v.idx] += (getNodeEdgeType(vG, eG) == EdgePartType::PLUS);
                            tls_localMinusDeg[v.idx] += (getNodeEdgeType(vG, eG) == EdgePartType::MINUS);
                        } else {
                            tls_localPlusDeg [u.idx] += (getNodeEdgeType(vG, eG) == EdgePartType::PLUS);
                            tls_localMinusDeg[u.idx] += (getNodeEdgeType(vG, eG) == EdgePartType::MINUS);
                            tls_localPlusDeg [v.idx] += (getNodeEdgeType(uG, eG) == EdgePartType::PLUS);
                            tls_localMinusDeg[v.idx] += (getNodeEdgeType(uG, eG) == EdgePartType::MINUS);
                        }
                        return;
                    }

                    // Virtual edge
                    auto B = skel.twinTreeNode(e);
                    spqr_compat::edge treeE = blk.skel2tree.at(e);
                    SPQR_RUST_ASSERT(treeE != nullptr);

                    EdgeDPState *child = (B == parent_of_A ? &edge_dp[treeE].up : &edge_dp[treeE].down);
                    EdgeDPState *toUpdate = (B == parent_of_A ? &edge_dp[treeE].down : &edge_dp[treeE].up);


                    spqr_compat::node nS = blk.blkToSkel[child->s];
                    spqr_compat::node nT = blk.blkToSkel[child->t];

                    tls_virtualEdges.push_back({toUpdate, child});

                    if (nS == u && nT == v) {
                        tls_localMinusDeg[nS.idx] += child->localMinusT;
                        tls_localPlusDeg [nS.idx] += child->localPlusT;
                        tls_localMinusDeg[nT.idx] += child->localMinusS;
                        tls_localPlusDeg [nT.idx] += child->localPlusS;
                    } else {
                        tls_localMinusDeg[nS.idx] += child->localMinusS;
                        tls_localPlusDeg [nS.idx] += child->localPlusS;
                        tls_localMinusDeg[nT.idx] += child->localMinusT;
                        tls_localPlusDeg [nT.idx] += child->localPlusT;
                    }
                });

                BF_INSTR(
                auto __pn_tC_start = std::chrono::high_resolution_clock::now();
                profiling_patch::pn_B_build_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__pn_tC_start - __pn_tB_start).count(),
                    std::memory_order_relaxed);
                )

                for (const auto &ve : tls_virtualEdges) {
                    EdgeDPState *BA = ve.toUpdate;
                    EdgeDPState *AB = ve.opposite;

                    spqr_compat::node sH = blk.blkToSkel[BA->s];
                    spqr_compat::node tH = blk.blkToSkel[BA->t];

                    BA->localPlusS = tls_localPlusDeg[sH.idx] - AB->localPlusS;
                    BA->localPlusT = tls_localPlusDeg[tH.idx] - AB->localPlusT;
                    BA->localMinusS = tls_localMinusDeg[sH.idx] - AB->localMinusS;
                    BA->localMinusT = tls_localMinusDeg[tH.idx] - AB->localMinusT;
                }

                BF_INSTR(
                auto __pn_tC_end = std::chrono::high_resolution_clock::now();
                profiling_patch::pn_C_propagate_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(__pn_tC_end - __pn_tC_start).count(),
                    std::memory_order_relaxed);
                )
            }

            void solveS(spqr_compat::node sNode,
                        NodeArray<NodeDPState> &node_dp,
                        spqr_compat::EdgeArray<EdgeDP> &dp,
                        BlockData &blk,
                        const CcData &cc)
            {
                const Skeleton &skel = blk.spqr->skeleton(sNode);
                const auto &skelG = skel.getGraph();
                const auto &T = blk.spqr->tree();

                std::vector<spqr_compat::node> nodesInOrderGcc;
                std::vector<spqr_compat::node> nodesInOrderSkel;

                std::unordered_map<uint32_t, EdgeDPState *> skelToState;
                skelToState.reserve(8);

                std::vector<spqr_compat::edge> adjEdgesG;
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
                        spqr_compat::node u = firstNode;
                        spqr_compat::node prev = secondNode;

                        while (true)
                        {
                            nodesInOrderGcc.push_back(blockNodeToCc(blk, skel.original(u)));
                            nodesInOrderSkel.push_back(u);

                            spqr_compat::node nextU = nullptr;
                            spqr_compat::edge nextE = nullptr;
                            bool closing = false;
                            spqr_compat::edge closingEdge = nullptr;

                            skelG.forEachAdj(u, [&](spqr_compat::node neighbor, spqr_compat::edge e) {
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
                                    adjEdgesG.push_back(blockEdgeToOrig(blk, skel.realEdge(closingEdge)));
                                else
                                    adjEdgesG.push_back(nullptr);
                                adjEntriesSkel.push_back(adjEntry{firstNode, closingEdge});
                                break;
                            }

                            if (!nextU)
                                break;

                            if (skel.realEdge(nextE))
                                adjEdgesG.push_back(blockEdgeToOrig(blk, skel.realEdge(nextE)));
                            else
                                adjEdgesG.push_back(nullptr);
                            adjEntriesSkel.push_back(adjEntry{nextU, nextE});

                            prev = u;
                            u = nextU;
                        }
                    }
                }

                std::vector<CutSides> cutSides;

                for (size_t i = 0; i < nodesInOrderGcc.size(); ++i)
                {
                    auto uGcc = nodesInOrderGcc[i];

                    std::vector<edge> adjEdgesSkelLoc = {
                        adjEntriesSkel[(i + adjEntriesSkel.size() - 1) % adjEntriesSkel.size()].theEdge(),
                        adjEntriesSkel[i].theEdge()};
                    std::vector<spqr_compat::edge> adjEdgesGLoc = {
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
                        if (blockNodeToCc(blk, state0->s) == uGcc)
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
                        if (blockNodeToCc(blk, state1->s) == uGcc)
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

                        cutSides.push_back(CutSides{
                            SnarlEndpoint{cc.nodeToOrig[uGcc], t0},
                            SnarlEndpoint{cc.nodeToOrig[uGcc], t1}});

                    }
                }

                if (cutSides.size() > 1)
                {
                    for (size_t i = 0; i < cutSides.size(); ++i)
                    {
                        const CutSides &left = cutSides[i];
                        const CutSides &right = cutSides[(i + 1) % cutSides.size()];
                        addSnarlTaggedPairNodes(
                            "S",
                            left.next_side.node, left.next_side.sign,
                            right.prev_side.node, right.prev_side.sign);
                    }
                    emitMixedSelfLoopComposedS(
                        cutSides, true,
                        [&](const SnarlEndpoint& first, const SnarlEndpoint& second) {
                            addSnarlTaggedPairNodes(
                                "S",
                                first.node, first.sign,
                                second.node, second.sign);
                        });
                }
            }

            void solveP(spqr_compat::node pNode,
                        spqr_compat::NodeArray<SPQRsolve::NodeDPState> &node_dp,
                        spqr_compat::EdgeArray<EdgeDP> &edge_dp,
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

                spqr_compat::node pole0Skel = nullptr, pole1Skel = nullptr;
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

                spqr_compat::node pole0Blk = skel.original(pole0Skel);
                spqr_compat::node pole1Blk = skel.original(pole1Skel);
                if (!pole0Blk || !pole1Blk)
                {
                    VLOG << "[DEBUG][solveP]  skel.original() returned nullptr pole, skip\n";
                    return;
                }

                spqr_compat::node pole0Gcc = blockNodeToCc(blk, pole0Blk);
                spqr_compat::node pole1Gcc = blockNodeToCc(blk, pole1Blk);

                VLOG << "[DEBUG][solveP]  poles: "
                     << C.node2name[cc.nodeToOrig[pole0Gcc]] << " (Gcc idx=" << pole0Gcc.index() << "), "
                     << C.node2name[cc.nodeToOrig[pole1Gcc]] << " (Gcc idx=" << pole1Gcc.index() << ")\n";

                static const char* dbg_poles_env_walker = std::getenv("BF_DEBUG_WALKER_POLES");
                bool dbg_walker = false;
                if (dbg_poles_env_walker) {
                    std::string p0 = C.node2name[cc.nodeToOrig[pole0Gcc]];
                    std::string p1 = C.node2name[cc.nodeToOrig[pole1Gcc]];
                    std::string env(dbg_poles_env_walker);
                    auto comma = env.find(',');
                    if (comma != std::string::npos) {
                        std::string a = env.substr(0, comma);
                        std::string b = env.substr(comma + 1);
                        if ((p0 == a && p1 == b) || (p0 == b && p1 == a)) {
                            dbg_walker = true;
                        }
                    }
                }

                if (dbg_walker) {
                    std::fprintf(stderr,
                        "\n[WALKER_DBG] === solveP visited for pNode=%d, poles=(%s, %s) skel|V|=%d |E|=%d ===\n",
                        pNode.index(),
                        C.node2name[cc.nodeToOrig[pole0Gcc]].c_str(),
                        C.node2name[cc.nodeToOrig[pole1Gcc]].c_str(),
                        skelGraph.numberOfNodes(), skelGraph.numberOfEdges());
                    std::fprintf(stderr,
                        "[WALKER_DBG]   pole0: isCutNode=%d badCutCount=%d  pole1: isCutNode=%d badCutCount=%d\n",
                        cc.isCutNode[pole0Gcc] ? 1 : 0, cc.badCutCount[pole0Gcc],
                        cc.isCutNode[pole1Gcc] ? 1 : 0, cc.badCutCount[pole1Gcc]);
                }

                auto hasDanglingOutside = [&](spqr_compat::node vGcc)
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
                    if (dbg_walker) std::fprintf(stderr, "[WALKER_DBG]   ABORT: dangling outside\n");
                    return;
                }

                std::vector<spqr_compat::adjEntry> edgeOrdering;
                skelGraph.forEachAdj(pole0Skel, [&](node neighbor, edge e) {
                    edgeOrdering.push_back(adjEntry{neighbor, e});
                });

                if (dbg_walker) {
                    std::fprintf(stderr, "[WALKER_DBG]   skeleton edges (%zu):\n", edgeOrdering.size());
                    int ei = 0;
                    for (spqr_compat::adjEntry adj : edgeOrdering) {
                        spqr_compat::edge eSkel = adj.theEdge();
                        if (skel.isVirtual(eSkel)) {
                            auto it = blk.skel2tree.find(eSkel);
                            if (it != blk.skel2tree.end()) {
                                spqr_compat::edge treeE = it->second;
                                spqr_compat::node B = (T.source(treeE) == pNode ? T.target(treeE) : T.source(treeE));
                                std::string typeStr = "?";
                                auto t_ = blk.spqr->typeOf(B);
                                if (t_ == StaticSPQRTree::NodeType::SNode) typeStr = "S";
                                else if (t_ == StaticSPQRTree::NodeType::PNode) typeStr = "P";
                                else if (t_ == StaticSPQRTree::NodeType::RNode) typeStr = "R";
                                EdgeDP &dpVal = edge_dp[treeE];
                                EdgeDPState &st = (blk.parent[pNode] == B ? dpVal.up : dpVal.down);
                                std::fprintf(stderr,
                                    "[WALKER_DBG]     edge[%d] = VIRTUAL -> %s-node idx=%d state.s=%d state.t=%d +S=%d -S=%d +T=%d -T=%d\n",
                                    ei, typeStr.c_str(), B.index(), st.s.index(), st.t.index(),
                                    st.localPlusS, st.localMinusS, st.localPlusT, st.localMinusT);
                            }
                        } else {
                            spqr_compat::edge eB = skel.realEdge(eSkel);
                            spqr_compat::edge eG = blockEdgeToOrig(blk, eB);
                            spqr_compat::node uG = C.G.source(eG);
                            spqr_compat::node vG = C.G.target(eG);
                            std::fprintf(stderr,
                                "[WALKER_DBG]     edge[%d] = REAL %s -> %s\n",
                                ei, C.node2name[uG].c_str(), C.node2name[vG].c_str());
                        }
                        ei++;
                    }
                }

                for (spqr_compat::adjEntry adj : edgeOrdering)
                {
                    spqr_compat::edge eSkel = adj.theEdge();
                    if (!skel.isVirtual(eSkel))
                        continue;

                    auto itMap = blk.skel2tree.find(eSkel);
                    if (itMap == blk.skel2tree.end())
                        continue;
                    spqr_compat::edge treeE = itMap->second;
                    spqr_compat::node B = (T.source(treeE) == pNode ? T.target(treeE) : T.source(treeE));

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

                        std::vector<spqr_compat::edge> leftPart, rightPart;

                        for (spqr_compat::adjEntry adj : edgeOrdering)
                        {
                            spqr_compat::edge eSkel = adj.theEdge();

                            EdgePartType lSign = EdgePartType::NONE;
                            EdgePartType rSign = EdgePartType::NONE;

                            if (!skel.isVirtual(eSkel))
                            {
                                spqr_compat::edge eB = skel.realEdge(eSkel);
                                spqr_compat::edge eG = blockEdgeToOrig(blk, eB);

                                spqr_compat::node pole0G = cc.nodeToOrig[pole0Gcc];
                                spqr_compat::node pole1G = cc.nodeToOrig[pole1Gcc];

                                lSign = getNodeEdgeType(pole0G, eG);
                                rSign = getNodeEdgeType(pole1G, eG);
                            }
                            else
                            {
                                auto itMap = blk.skel2tree.find(eSkel);
                                if (itMap == blk.skel2tree.end())
                                    continue;
                                spqr_compat::edge treeE = itMap->second;
                                spqr_compat::node B = (T.source(treeE) == pNode ? T.target(treeE) : T.source(treeE));

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
                            spqr_compat::edge eSkel = leftPart[0];
                            if (skel.isVirtual(eSkel))
                            {
                                spqr_compat::node B = skel.twinTreeNode(eSkel);
                                if (blk.spqr->typeOf(B) == StaticSPQRTree::NodeType::SNode)
                                {
                                    for (spqr_compat::node gccCut : node_dp[B].GccCuts_last3)
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
                        VLOG << "[DEBUG][solveP]  addSnarlTagged P: "
                             << C.node2name[cc.nodeToOrig[pole0Gcc]]
                             << (left == EdgePartType::PLUS ? "+" : "-")
                             << " "
                             << C.node2name[cc.nodeToOrig[pole1Gcc]]
                             << (right == EdgePartType::PLUS ? "+" : "-")
                             << "\n";
                        addSnarlTaggedPairNodes(
                            "P",
                            cc.nodeToOrig[pole0Gcc], left,
                            cc.nodeToOrig[pole1Gcc], right);
                    }
                }
            }

            void solveRR(spqr_compat::edge rrEdge,
                         spqr_compat::NodeArray<SPQRsolve::NodeDPState> &node_dp,
                         spqr_compat::EdgeArray<EdgeDP> &edge_dp,
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

                spqr_compat::node pole0Blk = down.s;
                spqr_compat::node pole1Blk = down.t;

                spqr_compat::node pole0Gcc = blockNodeToCc(blk, pole0Blk);
                spqr_compat::node pole1Gcc = blockNodeToCc(blk, pole1Blk);
                if (!pole0Gcc || !pole1Gcc)
                {
                    return;
                }

                auto hasDanglingOutside = [&](spqr_compat::node vGcc)
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

                addSnarlTaggedPairNodes(
                    "RR",
                    cc.nodeToOrig[pole0Gcc], pole0DownType,
                    cc.nodeToOrig[pole1Gcc], pole1DownType);
                addSnarlTaggedPairNodes(
                    "RR",
                    cc.nodeToOrig[pole0Gcc], pole0UpType,
                    cc.nodeToOrig[pole1Gcc], pole1UpType);
            }

            static inline uint64_t bf_choose_num_tasks(uint64_t n_iters,
                                                       uint64_t loop_weight,
                                                       const IntraPlan &plan)
            {

                const uint64_t raw = bf_ceil_div(loop_weight, plan.quantum);
                if (raw < 2 || n_iters < 2 || plan.numThreads <= 1)
                    return 0; 

                uint64_t target = std::max<uint64_t>(raw, 2ULL * static_cast<uint64_t>(plan.numThreads));
                target = std::min<uint64_t>(target, n_iters);
                return target;
            }

            void solveNodes(NodeArray<SPQRsolve::NodeDPState> &node_dp,
                            spqr_compat::EdgeArray<EdgeDP> &edge_dp,
                            BlockData &blk,
                            const CcData &cc,
                            IntraPlan &plan)
            {
                PROFILE_FUNCTION();
                if (!blk.spqr)
                    return;

                const auto &T = blk.spqr->tree();

                VLOG << "[DEBUG][solveNodes] start, |T.nodes|=" << T.numberOfNodes()
                     << " |T.edges|=" << T.numberOfEdges() << "\n";

                if (!plan.critical)
                {
                    BF_INSTR(profiling_patch::BlockTiming *bt_nc = profiling_patch::try_get_block_timing(plan.bid);)

                    {
                    BF_INSTR(auto __sn_t0 = std::chrono::high_resolution_clock::now();)
                    for (node tNode : T.nodes) {
                        if (blk.spqr->typeOf(tNode) == StaticSPQRTree::NodeType::SNode) {
                            BF_INSTR(profiling_patch::p3_solveS_calls.fetch_add(1, std::memory_order_relaxed);)
                            solveS(tNode, node_dp, edge_dp, blk, cc);
                        }
                    }
                    BF_INSTR(
                    auto __sn_t1 = std::chrono::high_resolution_clock::now();
                    uint64_t __dt_S = std::chrono::duration_cast<std::chrono::nanoseconds>(__sn_t1 - __sn_t0).count();
                    profiling_patch::p3_solveS_ns.fetch_add(__dt_S, std::memory_order_relaxed);
                    if (bt_nc) bt_nc->phase3_S_ns = __dt_S;
                    profiling_patch::taskloops_S_skipped.fetch_add(1, std::memory_order_relaxed);
                    )
                    }

                    {
                    BF_INSTR(auto __sn_t0 = std::chrono::high_resolution_clock::now();)
                    for (node tNode : T.nodes) {
                        if (blk.spqr->typeOf(tNode) == StaticSPQRTree::NodeType::PNode) {
                            BF_INSTR(profiling_patch::p3_solveP_calls.fetch_add(1, std::memory_order_relaxed);)
                            solveP(tNode, node_dp, edge_dp, blk, cc);
                        }
                    }
                    BF_INSTR(
                    auto __sn_t1 = std::chrono::high_resolution_clock::now();
                    uint64_t __dt_P = std::chrono::duration_cast<std::chrono::nanoseconds>(__sn_t1 - __sn_t0).count();
                    profiling_patch::p3_solveP_ns.fetch_add(__dt_P, std::memory_order_relaxed);
                    if (bt_nc) bt_nc->phase3_P_ns = __dt_P;
                    profiling_patch::taskloops_P_skipped.fetch_add(1, std::memory_order_relaxed);
                    )
                    }

                    {
                    BF_INSTR(auto __sn_t0 = std::chrono::high_resolution_clock::now();)
                    for (edge e : T.edges) {
                        auto srcT = blk.spqr->typeOf(T.source(e));
                        auto dstT = blk.spqr->typeOf(T.target(e));
                        if (srcT == SPQRTree::NodeType::RNode &&
                            dstT == SPQRTree::NodeType::RNode)
                        {
                            BF_INSTR(profiling_patch::p3_solveRR_calls.fetch_add(1, std::memory_order_relaxed);)
                            solveRR(e, node_dp, edge_dp, blk, cc);
                        }
                    }
                    BF_INSTR(
                    auto __sn_t1 = std::chrono::high_resolution_clock::now();
                    uint64_t __dt_RR = std::chrono::duration_cast<std::chrono::nanoseconds>(__sn_t1 - __sn_t0).count();
                    profiling_patch::p3_solveRR_ns.fetch_add(__dt_RR, std::memory_order_relaxed);
                    if (bt_nc) bt_nc->phase3_RR_ns = __dt_RR;
                    )
                    }

                    VLOG << "[DEBUG][solveNodes] end (serial path)\n";
                    return;
                }

                std::vector<node> sNodes;
                std::vector<node> pNodes;
                sNodes.reserve(T.numberOfNodes());
                pNodes.reserve(T.numberOfNodes() / 4 + 1);

                BF_INSTR(auto __collect_t0 = std::chrono::high_resolution_clock::now();)
                for (node tNode : T.nodes) {
                    auto ty = blk.spqr->typeOf(tNode);
                    if (ty == StaticSPQRTree::NodeType::SNode) sNodes.push_back(tNode);
                    else if (ty == StaticSPQRTree::NodeType::PNode) pNodes.push_back(tNode);
                }
                BF_INSTR(
                auto __collect_t1 = std::chrono::high_resolution_clock::now();
                uint64_t __dt_collect = std::chrono::duration_cast<std::chrono::nanoseconds>(__collect_t1 - __collect_t0).count();
                profiling_patch::sub_solveS_collect_ns.fetch_add(__dt_collect, std::memory_order_relaxed);
                profiling_patch::BlockTiming *bt_sn = profiling_patch::try_get_block_timing(plan.bid);
                if (bt_sn) bt_sn->phase3_S_collect_ns = __dt_collect;
                )

                BF_INSTR(profiling_patch::p3_solveS_calls.fetch_add(sNodes.size(), std::memory_order_relaxed);)
                BF_INSTR(profiling_patch::p3_solveP_calls.fetch_add(pNodes.size(), std::memory_order_relaxed);)

                {
                BF_INSTR(auto __sn_t0 = std::chrono::high_resolution_clock::now();)

                const uint64_t n = sNodes.size();
                const uint64_t W = T.numberOfNodes();
                const uint64_t nT = bf_choose_num_tasks(n, W, plan);

                if (nT == 0) {
                    BF_INSTR(
                    profiling_patch::taskloops_S_skipped.fetch_add(1, std::memory_order_relaxed);
                    )
                    for (size_t i = 0; i < sNodes.size(); ++i)
                        solveS(sNodes[i], node_dp, edge_dp, blk, cc);
                } else {
                    BF_INSTR(
                    profiling_patch::taskloops_S_created.fetch_add(1, std::memory_order_relaxed);
                    profiling_patch::tasks_requested_S_total.fetch_add(nT, std::memory_order_relaxed);
                    if (bt_sn) {
                        bt_sn->taskloops_S_created += 1;
                        bt_sn->tasks_requested_S += nT;
                    }
                    )
                    if (plan.activeIntraTaskloops)
                        plan.activeIntraTaskloops->fetch_add(1, std::memory_order_relaxed);

                    BF_OMP_PRAGMA(omp taskloop num_tasks(nT) default(shared))
                    for (long long i = 0; i < static_cast<long long>(sNodes.size()); ++i) {
                        solveS(sNodes[i], node_dp, edge_dp, blk, cc);
                    }

                    if (plan.activeIntraTaskloops)
                        plan.activeIntraTaskloops->fetch_sub(1, std::memory_order_relaxed);
                }

                BF_INSTR(
                auto __sn_t1 = std::chrono::high_resolution_clock::now();
                uint64_t __dt_S = std::chrono::duration_cast<std::chrono::nanoseconds>(__sn_t1 - __sn_t0).count();
                profiling_patch::p3_solveS_ns.fetch_add(__dt_S, std::memory_order_relaxed);
                if (bt_sn) bt_sn->phase3_S_ns = __dt_S;
                )
                }

                {
                BF_INSTR(auto __sn_t0 = std::chrono::high_resolution_clock::now();)

                const uint64_t n = pNodes.size();

                const uint64_t W = T.numberOfNodes();
                const uint64_t nT = bf_choose_num_tasks(n, W, plan);

                if (nT == 0) {
                    BF_INSTR(
                    profiling_patch::taskloops_P_skipped.fetch_add(1, std::memory_order_relaxed);
                    )
                    for (size_t i = 0; i < pNodes.size(); ++i)
                        solveP(pNodes[i], node_dp, edge_dp, blk, cc);
                } else {
                    BF_INSTR(
                    profiling_patch::taskloops_P_created.fetch_add(1, std::memory_order_relaxed);
                    profiling_patch::tasks_requested_P_total.fetch_add(nT, std::memory_order_relaxed);
                    if (bt_sn) {
                        bt_sn->taskloops_P_created += 1;
                        bt_sn->tasks_requested_P += nT;
                    }
                    )
                    if (plan.activeIntraTaskloops)
                        plan.activeIntraTaskloops->fetch_add(1, std::memory_order_relaxed);

                    BF_OMP_PRAGMA(omp taskloop num_tasks(nT) default(shared))
                    for (long long i = 0; i < static_cast<long long>(pNodes.size()); ++i) {
                        solveP(pNodes[i], node_dp, edge_dp, blk, cc);
                    }

                    if (plan.activeIntraTaskloops)
                        plan.activeIntraTaskloops->fetch_sub(1, std::memory_order_relaxed);
                }

                BF_INSTR(
                auto __sn_t1 = std::chrono::high_resolution_clock::now();
                uint64_t __dt_P = std::chrono::duration_cast<std::chrono::nanoseconds>(__sn_t1 - __sn_t0).count();
                profiling_patch::p3_solveP_ns.fetch_add(__dt_P, std::memory_order_relaxed);
                if (bt_sn) bt_sn->phase3_P_ns = __dt_P;
                )
                }

                {
                BF_INSTR(auto __sn_t0 = std::chrono::high_resolution_clock::now();)
                for (edge e : T.edges) {
                    auto srcT = blk.spqr->typeOf(T.source(e));
                    auto dstT = blk.spqr->typeOf(T.target(e));
                    if (srcT == SPQRTree::NodeType::RNode &&
                        dstT == SPQRTree::NodeType::RNode)
                    {
                        BF_INSTR(profiling_patch::p3_solveRR_calls.fetch_add(1, std::memory_order_relaxed);)
                        solveRR(e, node_dp, edge_dp, blk, cc);
                    }
                }
                BF_INSTR(
                auto __sn_t1 = std::chrono::high_resolution_clock::now();
                uint64_t __dt_RR = std::chrono::duration_cast<std::chrono::nanoseconds>(__sn_t1 - __sn_t0).count();
                profiling_patch::p3_solveRR_ns.fetch_add(__dt_RR, std::memory_order_relaxed);
                if (bt_sn) bt_sn->phase3_RR_ns = __dt_RR;
                )
                }

                VLOG << "[DEBUG][solveNodes] end (parallel path)\n";
            }
            void solveSPQR(BlockData &blk, const CcData &cc, IntraPlan &plan)
            {
                PROFILE_FUNCTION();

                if (ctx().spCompressMode == Context::SpCompressMode::MacroDirectDebug &&
                    blk.spCompressHandle)
                {
                    auto &C = ctx();
                    if (!blk.Gblk || blockNodeCount(blk) < 3)
                        return;

                    const SpCompressTreeView &m_view = blk.macroTreeView;
                    if (m_view.macros_len == 0)
                        return;

                    const bool with_macro_down =
                        (std::getenv("BF_MACRO_DIRECT_WITH_DOWN") != nullptr);
                    const bool with_gcc_cuts =
                        (std::getenv("BF_MACRO_DIRECT_WITH_GCC_CUTS") != nullptr);
                    const bool with_macro_series_s =
                        (std::getenv("BF_MACRO_DIRECT_EMIT_MACRO_SERIES_S") != nullptr);
                    const bool count_only =
                        (std::getenv("BF_MACRO_DIRECT_COUNT_ONLY") != nullptr);
                    const bool stats_enabled =
                        (std::getenv("BF_MACRO_DIRECT_STATS") != nullptr);

                    uint64_t t_states_us = 0;
                    uint64_t t_gcc_cuts_us = 0;
                    uint64_t t_build_tcore_us = 0;
                    uint64_t t_tcore_up_us = 0;
                    uint64_t t_tcore_down_us = 0;
                    uint64_t t_absorb_us = 0;
                    uint64_t t_macro_s_ctx_us = 0;
                    uint64_t t_emit_macro_s_us = 0;
                    uint64_t t_macro_down_us = 0;
                    uint64_t t_emit_macro_us = 0;
                    uint64_t t_emit_tcore_s_us = 0;
                    uint64_t t_emit_tcore_us = 0;
                    uint64_t t_emit_tcore_rr_us = 0;
                    uint64_t t_emit_e_us = 0;
                    uint64_t t_commit_us = 0;

                    auto phase_t0 = std::chrono::steady_clock::now();
                    auto lap_us = [&]() -> uint64_t {
                        auto t1 = std::chrono::steady_clock::now();
                        uint64_t us = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                t1 - phase_t0).count());
                        phase_t0 = t1;
                        return us;
                    };
                    auto log_big_phase = [&](const char *phase) {
                        if (stats_enabled && m_view.macros_len >= 1000000u) {
                            std::fprintf(stderr,
                                "[macro_direct_phase] block=%zu macros=%u phase=%s\n",
                                static_cast<size_t>(plan.bid),
                                m_view.macros_len,
                                phase);
                            std::fflush(stderr);
                        }
                    };

                    log_big_phase("states:start");
                    auto states = compute_all_macro_dp_states(m_view, blk);
                    t_states_us = lap_us();
                    log_big_phase("states:done");
                    std::unique_ptr<MacroSeriesGccCutsCache> macro_gcc_cuts_storage;
                    MacroSeriesGccCutsCache *macro_gcc_cuts = nullptr;
                    if (with_gcc_cuts)
                    {
                        log_big_phase("gcc_cuts:start");
                        macro_gcc_cuts_storage =
                            std::make_unique<MacroSeriesGccCutsCache>(states, m_view, blk, cc);
                        macro_gcc_cuts = macro_gcc_cuts_storage.get();
                        log_big_phase("gcc_cuts:done");
                    }
                    t_gcc_cuts_us = lap_us();

                    TCoreContext tctx;
                    uint32_t emitted_macro = 0;
                    uint32_t emitted_macro_s = 0;
                    uint32_t emitted_tcore_s = 0;
                    uint32_t emitted_tcore = 0;
                    uint32_t emitted_tcore_rr = 0;
                    uint32_t emitted_e = 0;
                    uint32_t macro_down_seeded = 0;
                    uint32_t macro_down_nested = 0;
                    uint32_t macro_s_targets = 0;
                    uint32_t macro_s_down_seeded = 0;
	                    uint32_t macro_s_down_nested = 0;
	                    std::vector<SnarlEndpointPair> psnarls;
	                    size_t macro_psnarls_end = 0;
	                    auto *psnarls_out = count_only ? nullptr : &psnarls;

                    if (build_T_core_context(blk, tctx))
                    {
                        t_build_tcore_us = lap_us();
                        log_big_phase("tcore_up:start");
                        auto up_states =
                            compute_T_core_up_states(states, m_view, blk, tctx);
                        t_tcore_up_us = lap_us();
                        log_big_phase("tcore_up:done");
                        log_big_phase("tcore_down:start");
                        auto down_states =
                            compute_T_core_down_states(up_states, states, m_view, blk, tctx);
                        t_tcore_down_us = lap_us();
                        log_big_phase("tcore_down:done");

                        log_big_phase("absorb:start");
                        auto absorbed_by_tcore =
                            compute_macro_absorption_by_tcore(m_view, tctx);

                        auto series_inlined_by_tcore_s =
                            compute_series_inlined_in_tcore_s(m_view, tctx);
                        t_absorb_us = lap_us();
                        log_big_phase("absorb:done");

                        std::vector<std::vector<spqr_compat::node>> tcore_s_gcc_cuts;
                        log_big_phase("emit_tcore_s:start");
                        emitted_tcore_s = emit_T_core_s_snarls(
                            states, up_states, down_states,
                            tctx, m_view, blk, cc,
                            tcore_s_gcc_cuts, count_only);
                        t_emit_tcore_s_us = lap_us();
                        log_big_phase("emit_tcore_s:done");

                        std::vector<uint8_t> macro_s_target_filter;
                        if (with_macro_series_s)
                        {
                            macro_s_target_filter.assign(m_view.macros_len, 0);
                            for (uint32_t mi = 0; mi < m_view.macros_len; ++mi) {
                                const SpCompressNode& mm = m_view.macros_ptr[mi];
                                if (mm.kind != SP_COMPRESS_KIND_SERIES) continue;
                                if (mm.children_count < 2) continue;
                                if (mi < series_inlined_by_tcore_s.size() &&
                                    series_inlined_by_tcore_s[mi]) continue;
                                macro_s_target_filter[mi] = 1;
                                ++macro_s_targets;
                            }
                            t_macro_s_ctx_us = lap_us();
                        }

                        MacroDownContext root_macro_context;
                        bool have_root_macro_context = false;
                        if (!with_macro_down)
                        {
                            log_big_phase("macro_root_ctx:start");
                            root_macro_context = compute_root_macro_tcore_contexts(
                                states, m_view, blk, tctx, up_states, down_states,
                                &tcore_s_gcc_cuts);
                            macro_down_seeded = root_macro_context.seeded_from_tcore;
                            if (with_macro_series_s)
                            {
                                macro_s_down_seeded = root_macro_context.seeded_from_tcore;
                                macro_s_down_nested = extend_root_parallel_series_contexts(
                                    states, m_view, blk, macro_s_target_filter,
                                    root_macro_context);
                            }
                            t_macro_down_us = lap_us();
                            have_root_macro_context = true;
                            log_big_phase("macro_root_ctx:done");
                        }

                        if (with_macro_series_s)
                        {
                            log_big_phase("emit_macro_s:start");
                            if (have_root_macro_context)
                            {
                                emitted_macro_s = emit_macro_series_s_snarls(
                                    states, m_view, blk, cc,
                                    macro_s_target_filter,
                                    root_macro_context.states,
                                    root_macro_context.has_state,
                                    count_only,
                                    &series_inlined_by_tcore_s,
                                    plan.numThreads);
                            }
                            t_emit_macro_s_us = lap_us();
                            log_big_phase("emit_macro_s:done");
                        }

                        if (with_macro_down)
                        {
                            log_big_phase("macro_down:start");
                            auto macro_down = compute_macro_down_states(
                                states, m_view, blk, &tctx, &up_states, &down_states);
                            macro_down_seeded = macro_down.seeded_from_tcore;
                            macro_down_nested = macro_down.nested_states;
                            t_macro_down_us = lap_us();
                            log_big_phase("macro_down:done");

	                            log_big_phase("emit_macro:start");
	                            emitted_macro = emit_all_parallel_macro_snarls(
	                                states, m_view, blk, cc, psnarls_out, absorbed_by_tcore,
	                                macro_gcc_cuts, &macro_down.states,
	                                &macro_down.has_state,
	                                nullptr, nullptr);
                        }
                        else
                        {
	                            log_big_phase("emit_macro:start");
	                            emitted_macro = emit_all_parallel_macro_snarls(
	                                states, m_view, blk, cc, psnarls_out, absorbed_by_tcore,
	                                macro_gcc_cuts,
	                                &root_macro_context.states,
	                                &root_macro_context.has_state,
	                                &root_macro_context.gcc_cut_count,
	                                &root_macro_context.gcc_cut_idx);
                        }
                        t_emit_macro_us = lap_us();
                        macro_psnarls_end = psnarls.size();
                        log_big_phase("emit_macro:done");

	                        log_big_phase("emit_tcore:start");
	                        emitted_tcore = emit_T_core_psnarls(
	                            states, up_states, down_states, absorbed_by_tcore,
	                            tctx, m_view, blk, cc, psnarls_out,
	                            macro_gcc_cuts, &tcore_s_gcc_cuts);
                        t_emit_tcore_us = lap_us();
                        log_big_phase("emit_tcore:done");

                        log_big_phase("emit_tcore_rr:start");
                        emitted_tcore_rr = emit_T_core_rr_snarls(
                            up_states, down_states, tctx, blk, cc, count_only);
                        t_emit_tcore_rr_us = lap_us();
                        log_big_phase("emit_tcore_rr:done");
                    }
                    else
                    {
                        t_build_tcore_us = lap_us();

                        std::vector<uint8_t> series_inlined_no_tcore; 
                        std::vector<uint8_t> macro_s_target_filter_no_tcore;
                        if (with_macro_series_s) {
                            macro_s_target_filter_no_tcore.assign(m_view.macros_len, 0);
                            for (uint32_t mi = 0; mi < m_view.macros_len; ++mi) {
                                const SpCompressNode& mm = m_view.macros_ptr[mi];
                                if (mm.kind != SP_COMPRESS_KIND_SERIES) continue;
                                if (mm.children_count < 2) continue;
                                macro_s_target_filter_no_tcore[mi] = 1;
                                ++macro_s_targets;
                            }
                        }

                        if (with_macro_down)
                        {
                            log_big_phase("macro_down:start");
                            auto macro_down = compute_macro_down_states(states, m_view, blk);
                            macro_down_nested = macro_down.nested_states;
                            t_macro_down_us = lap_us();
                            log_big_phase("macro_down:done");

                            if (with_macro_series_s) {
                                log_big_phase("emit_macro_s:start");
                                emitted_macro_s = emit_macro_series_s_snarls(
                                    states, m_view, blk, cc,
                                    macro_s_target_filter_no_tcore,
                                    macro_down.states,
                                    macro_down.has_state,
                                    count_only,
                                    &series_inlined_no_tcore,
                                    plan.numThreads);
                                t_emit_macro_s_us = lap_us();
                                log_big_phase("emit_macro_s:done");
                            }

	                            log_big_phase("emit_macro:start");
	                            emitted_macro = emit_all_parallel_macro_snarls(
	                                states, m_view, blk, cc, psnarls_out, {},
	                                macro_gcc_cuts, &macro_down.states,
	                                &macro_down.has_state,
	                                nullptr, nullptr);
                        }
                        else
                        {
                            if (with_macro_series_s) {
                                log_big_phase("macro_down_targeted:start");
                                auto macro_down_targeted =
                                    compute_macro_down_states(
                                        states, m_view, blk,
                                        nullptr, nullptr, nullptr,
                                        &macro_s_target_filter_no_tcore);
                                macro_s_down_seeded = macro_down_targeted.seeded_from_tcore;
                                macro_s_down_nested = macro_down_targeted.nested_states;
                                t_macro_down_us = lap_us();
                                log_big_phase("macro_down_targeted:done");

                                log_big_phase("emit_macro_s:start");
                                emitted_macro_s = emit_macro_series_s_snarls(
                                    states, m_view, blk, cc,
                                    macro_s_target_filter_no_tcore,
                                    macro_down_targeted.states,
                                    macro_down_targeted.has_state,
                                    count_only,
                                    &series_inlined_no_tcore,
                                    plan.numThreads);
                                t_emit_macro_s_us = lap_us();
                                log_big_phase("emit_macro_s:done");
                            }

	                            log_big_phase("emit_macro:start");
	                            emitted_macro = emit_all_parallel_macro_snarls(
	                                states, m_view, blk, cc, psnarls_out, {}, macro_gcc_cuts,
	                                nullptr, nullptr, nullptr, nullptr);
                        }
                        t_emit_macro_us = lap_us();
                        macro_psnarls_end = psnarls.size();
                        log_big_phase("emit_macro:done");
                    }

	                    log_big_phase("commit:start");
	                    for (size_t i = 0; i < psnarls.size(); ++i)
	                    {
	                        auto &p = psnarls[i];
	                        const char *tag = (i < macro_psnarls_end) ? "P:macro" : "P:tcore";
	                        addSnarlTaggedPairNodes(
	                            tag,
	                            p.first.node,
	                            p.first.sign,
	                            p.second.node,
	                            p.second.sign);
	                    }
                    t_commit_us = lap_us();
                    log_big_phase("commit:done");

                    if (macro_gcc_cuts)
                    {
                        t_gcc_cuts_us = macro_gcc_cuts->compute_us;
                    }

                    if (with_macro_series_s)
                    {
                        log_big_phase("emit_e:start");
                        auto& C = ctx();
                        const size_t nBlkNodes = blockNodeCount(blk);
                        auto hasDanglingOutside_prescan_md = [&](spqr_compat::node vGcc) -> bool {
                            if (!vGcc) return false;
                            if (!cc.isCutNode[vGcc]) return false;
                            if (cc.badCutCount[vGcc] >= 2) return true;
                            if (cc.badCutCount[vGcc] == 1 &&
                                cc.lastBad[vGcc] != blk.bNode) return true;
                            return false;
                        };
                        auto flipSign_prescan_md = [](EdgePartType t) {
                            return (t == EdgePartType::PLUS)
                                ? EdgePartType::MINUS
                                : EdgePartType::PLUS;
                        };
                        auto check_one_vertex_prescan_md = [&](
                            spqr_compat::node vNode,
                            EdgePartType sign,
                            EdgePartType eSign) -> bool {
                            int totPlus = blockDegPlus(blk, vNode);
                            int totMinus = blockDegMinus(blk, vNode);
                            if (sign == EdgePartType::PLUS) {
                                if (eSign == EdgePartType::PLUS) {
                                    int oP = totPlus - 1;
                                    int oM = totMinus;
                                    return (oP == 0 && oM > 0);
                                }
                                int oP = totPlus;
                                int oM = totMinus - 1;
                                return (oM == 0 && oP > 0);
                            }
                            if (eSign == EdgePartType::MINUS) {
                                int oM = totMinus - 1;
                                int oP = totPlus;
                                return (oM == 0 && oP > 0);
                            }
                            int oM = totMinus;
                            int oP = totPlus - 1;
                            return (oP == 0 && oM > 0);
                        };

	                        std::vector<spqr_compat::edge> edge_prescan_md =
	                            collectBlockEdges(blk);

                        const size_t prescan_workers_md =
                            std::min(edge_prescan_md.size(),
                                     static_cast<size_t>(std::max(1, plan.numThreads)));
                        struct ECandidateMd {
                            spqr_compat::edge eB;
                            EdgePartType edgeSignU;
                            EdgePartType edgeSignV;
                            bool original;
                            bool flipped;
                        };
                        std::vector<std::vector<uint32_t>> local_interesting_vertices(
                            prescan_workers_md);
                        std::vector<std::vector<ECandidateMd>> local_e_candidates(
                            prescan_workers_md);
                        auto prescan_edge_md = [&](spqr_compat::edge eB,
                                                   std::vector<uint32_t>& local_vertices,
                                                   std::vector<ECandidateMd>& local_candidates) {
                            spqr_compat::edge eG = blockEdgeToOrig(blk, eB);
                            spqr_compat::node uB = blockEdgeSource(blk, eB);
                            spqr_compat::node vB = blockEdgeTarget(blk, eB);
                            spqr_compat::node uGcc = blockNodeToCc(blk, uB);
                            spqr_compat::node vGcc = blockNodeToCc(blk, vB);
                            if (!uGcc || !vGcc) return;
                            spqr_compat::node uG = cc.nodeToOrig[uGcc];
                            spqr_compat::node vG = cc.nodeToOrig[vGcc];
                            if (cc.isTip[uGcc] || cc.isTip[vGcc]) return;
                            if (hasDanglingOutside_prescan_md(uGcc) ||
                                hasDanglingOutside_prescan_md(vGcc)) {
                                return;
                            }
                            EdgePartType edgeSignU = getNodeEdgeType(uG, eG);
                            EdgePartType edgeSignV = getNodeEdgeType(vG, eG);
                            EdgePartType flipU = flipSign_prescan_md(edgeSignU);
                            EdgePartType flipV = flipSign_prescan_md(edgeSignV);
                            bool original =
                                check_one_vertex_prescan_md(uB, edgeSignU, edgeSignU) &&
                                check_one_vertex_prescan_md(vB, edgeSignV, edgeSignV);
                            bool flipped =
                                check_one_vertex_prescan_md(uB, flipU, edgeSignU) &&
                                check_one_vertex_prescan_md(vB, flipV, edgeSignV);
                            if (!original && !flipped) return;
                            if (isTrashOriginalNode(uG) || isTrashOriginalNode(vG)) return;
                            if (flipped) {
                                local_vertices.push_back(uB.idx);
                                local_vertices.push_back(vB.idx);
                            }
                            local_candidates.push_back(
                                ECandidateMd{eB, edgeSignU, edgeSignV, original, flipped});
                        };
                        if (prescan_workers_md <= 1 || edge_prescan_md.size() < 200000) {
                            for (spqr_compat::edge eB : edge_prescan_md) {
                                prescan_edge_md(
                                    eB,
                                    local_interesting_vertices[0],
                                    local_e_candidates[0]);
                            }
                        } else {
                            std::vector<std::thread> prescan_workers;
                            prescan_workers.reserve(prescan_workers_md);
                            for (size_t tid = 0; tid < prescan_workers_md; ++tid) {
                                const size_t begin =
                                    (edge_prescan_md.size() * tid) / prescan_workers_md;
                                const size_t end =
                                    (edge_prescan_md.size() * (tid + 1)) / prescan_workers_md;
                                prescan_workers.emplace_back([&, tid, begin, end]() {
                                    auto& local = local_interesting_vertices[tid];
                                    auto& local_candidates = local_e_candidates[tid];
                                    for (size_t i = begin; i < end; ++i) {
                                        prescan_edge_md(
                                            edge_prescan_md[i],
                                            local,
                                            local_candidates);
                                    }
                                });
                            }
                            for (auto& th : prescan_workers) {
                                th.join();
                            }
                        }

                        size_t e_candidate_count_md = 0;
                        for (const auto& local : local_e_candidates) {
                            e_candidate_count_md += local.size();
                        }
                        std::vector<ECandidateMd> e_candidates_md;
                        e_candidates_md.reserve(e_candidate_count_md);
                        for (auto& local : local_e_candidates) {
                            e_candidates_md.insert(
                                e_candidates_md.end(),
                                std::make_move_iterator(local.begin()),
                                std::make_move_iterator(local.end()));
                        }

                        std::vector<uint32_t> vertexSlot_md(nBlkNodes, SPQR_INVALID);
                        std::vector<std::vector<uint32_t>> vertexInSnodes_md;
                        for (const auto& local : local_interesting_vertices) {
                            for (uint32_t v : local) {
                                if (v < nBlkNodes && vertexSlot_md[v] == SPQR_INVALID) {
                                    vertexSlot_md[v] = static_cast<uint32_t>(vertexInSnodes_md.size());
                                    vertexInSnodes_md.emplace_back();
                                }
                            }
                        }

                        auto child_ref_endpoints_e = [&](uint32_t cref,
                                                         uint32_t& a,
                                                         uint32_t& b) -> bool {
                            if (SP_COMPRESS_CHILD_IS_EDGE(cref)) {
                                uint32_t bid = SP_COMPRESS_CHILD_AS_EDGE(cref);
                                spqr_compat::edge eBlk{bid};
                                a = blockEdgeSource(blk, eBlk).idx;
                                b = blockEdgeTarget(blk, eBlk).idx;
                                return true;
                            }

                            uint32_t mid = SP_COMPRESS_CHILD_AS_MACRO(cref);
                            if (mid >= m_view.macros_len) return false;
                            a = m_view.macros_ptr[mid].left;
                            b = m_view.macros_ptr[mid].right;
                            return (a != SPQR_INVALID && b != SPQR_INVALID);
                        };

                        auto add_vertex_snode_md = [&](uint32_t blk_id,
                                                       uint32_t tag) {
                            if (blk_id >= nBlkNodes) return;
                            uint32_t slot = vertexSlot_md[blk_id];
                            if (slot == SPQR_INVALID) return;
                            auto& list = vertexInSnodes_md[slot];
                            if (list.empty() || list.back() != tag) {
                                list.push_back(tag);
                            }
                        };

                        for (uint32_t m_id = 0; m_id < m_view.macros_len; ++m_id) {
                            const SpCompressNode& sm = m_view.macros_ptr[m_id];
                            if (sm.kind != SP_COMPRESS_KIND_SERIES) continue;
                            if (sm.children_count == 0) continue;

                            const uint32_t k = sm.children_count;
                            for (uint32_t i = 0; i < k; ++i) {
                                uint32_t cref = m_view.children_ptr[sm.children_offset + i];
                                uint32_t a = SPQR_INVALID;
                                uint32_t b = SPQR_INVALID;
                                if (!child_ref_endpoints_e(cref, a, b)) {
                                    break;
                                }
                                add_vertex_snode_md(a, m_id);
                                if (b != a) {
                                    add_vertex_snode_md(b, m_id);
                                }
                            }
                        }

                        if (tctx.T_len > 0) {
                            for (uint32_t tn = 0; tn < tctx.T_len; ++tn) {
                                if (tctx.node_types[tn] != SPQR_NODE_TYPE_S) continue;
                                uint32_t map_start = tctx.node_mapping_offsets[tn];
                                uint32_t map_end = tctx.node_mapping_offsets[tn + 1];
                                uint32_t tag = m_view.macros_len + tn;
                                for (uint32_t k_loc = 0; k_loc < map_end - map_start; ++k_loc) {
                                    uint32_t blk_id = tcore_local_to_block_id(tctx, tn, k_loc);
                                    add_vertex_snode_md(blk_id, tag);
                                }
                            }
                        }

                        auto shareSnode_md = [&](spqr_compat::node aB, spqr_compat::node bB) -> bool {
                            if (aB.idx >= vertexSlot_md.size() ||
                                bB.idx >= vertexSlot_md.size()) {
                                return false;
                            }
                            uint32_t aSlot = vertexSlot_md[aB.idx];
                            uint32_t bSlot = vertexSlot_md[bB.idx];
                            if (aSlot == SPQR_INVALID || bSlot == SPQR_INVALID) return false;
                            const auto& La = vertexInSnodes_md[aSlot];
                            const auto& Lb = vertexInSnodes_md[bSlot];
                            if (La.empty() || Lb.empty()) return false;
                            size_t i = 0;
                            size_t j = 0;
                            while (i < La.size() && j < Lb.size()) {
                                if (La[i] == Lb[j]) return true;
                                if (La[i] < Lb[j]) {
                                    ++i;
                                } else {
                                    ++j;
                                }
                            }
                            return false;
                        };

                        auto flipSign_md = [](EdgePartType t) {
                            return (t == EdgePartType::PLUS)
                                ? EdgePartType::MINUS
                                : EdgePartType::PLUS;
                        };

                        auto process_edge_e_md = [&](const ECandidateMd& candidate,
                                                     std::vector<SnarlEndpointPair>* local_snarls)
                            -> uint32_t {
                            spqr_compat::edge eB = candidate.eB;
                            spqr_compat::node uB = blockEdgeSource(blk, eB);
                            spqr_compat::node vB = blockEdgeTarget(blk, eB);
                            spqr_compat::node uGcc = blockNodeToCc(blk, uB);
                            spqr_compat::node vGcc = blockNodeToCc(blk, vB);
                            if (!uGcc || !vGcc) return 0;
                            spqr_compat::node uG = cc.nodeToOrig[uGcc];
                            spqr_compat::node vG = cc.nodeToOrig[vGcc];

                            uint32_t local_emitted = 0;
                            auto testCandidate_md = [&](
                                EdgePartType signU,
                                EdgePartType signV,
                                bool isFlipCase) {
                                if (isFlipCase && shareSnode_md(uB, vB)) return;

                                if (!count_only) {
                                    if (local_snarls) {
                                        local_snarls->push_back(
                                            {SnarlEndpoint{uG, signU},
                                             SnarlEndpoint{vG, signV}});
                                    } else {
                                        addSnarlTaggedPairNodes("E", uG, signU, vG, signV);
                                    }
                                }
                                ++local_emitted;
                            };

                            if (candidate.original) {
                                testCandidate_md(
                                    candidate.edgeSignU,
                                    candidate.edgeSignV,
                                    false);
                            }
                            if (candidate.flipped) {
                                testCandidate_md(
                                    flipSign_md(candidate.edgeSignU),
                                    flipSign_md(candidate.edgeSignV),
                                    true);
                            }
                            return local_emitted;
                        };

                        const size_t edge_count_md = e_candidates_md.size();
                        const size_t workers_md =
                            std::min(edge_count_md,
                                     static_cast<size_t>(std::max(1, plan.numThreads)));
                        const bool parallel_e_md =
                            workers_md > 1 && edge_count_md >= 200000;

                        if (!parallel_e_md) {
                            for (const ECandidateMd& candidate : e_candidates_md) {
                                emitted_e += process_edge_e_md(candidate, nullptr);
                            }
                        } else {
                            std::vector<uint32_t> local_counts(workers_md, 0);
                            std::vector<std::vector<SnarlEndpointPair>> local_e_snarls;
                            if (!count_only) {
                                local_e_snarls.resize(workers_md);
                            }

                            std::vector<std::thread> workers;
                            workers.reserve(workers_md);
                            for (size_t tid = 0; tid < workers_md; ++tid) {
                                const size_t begin =
                                    (e_candidates_md.size() * tid) / workers_md;
                                const size_t end =
                                    (e_candidates_md.size() * (tid + 1)) / workers_md;
                                workers.emplace_back([&, tid, begin, end]() {
                                    uint32_t local = 0;
                                    std::vector<SnarlEndpointPair>* local_out =
                                        count_only ? nullptr : &local_e_snarls[tid];
                                    for (size_t i = begin; i < end; ++i) {
                                        local += process_edge_e_md(
                                            e_candidates_md[i], local_out);
                                    }
                                    local_counts[tid] = local;
                                });
                            }
                            for (auto& th : workers) {
                                th.join();
                            }

                            for (size_t tid = 0; tid < workers_md; ++tid) {
                                emitted_e += local_counts[tid];
                                if (!count_only) {
                                    for (const auto& p : local_e_snarls[tid]) {
                                        addSnarlTaggedPairNodes(
                                            "E",
                                            p.first.node,
                                            p.first.sign,
                                            p.second.node,
                                            p.second.sign);
                                    }
                                }
                            }
                        }
                        t_emit_e_us = lap_us();
                        log_big_phase("emit_e:done");
                    }

                    if (stats_enabled)
                    {
                        const uint64_t emitted_total =
                            static_cast<uint64_t>(emitted_macro) +
                            static_cast<uint64_t>(emitted_macro_s) +
                            static_cast<uint64_t>(emitted_tcore_s) +
                            static_cast<uint64_t>(emitted_tcore) +
                            static_cast<uint64_t>(emitted_tcore_rr) +
                            static_cast<uint64_t>(emitted_e);
                        const uint64_t stored_total =
                            count_only ? emitted_total :
                            static_cast<uint64_t>(psnarls.size()) +
                            static_cast<uint64_t>(emitted_macro_s) +
                            static_cast<uint64_t>(emitted_tcore_s) +
                            static_cast<uint64_t>(emitted_tcore_rr) +
                            static_cast<uint64_t>(emitted_e);
                        std::fprintf(stderr,
                            "[macro_direct] block=%zu macros=%u core_tree=%u "
                            "S_from_tcore=%u S_from_macros=%u "
                            "P_from_macros=%u P_from_tcore=%u "
                            "RR_from_tcore=%u E_from_macro=%u emitted=%llu "
                            "count_only=%d macro_down=%d seeded=%u nested=%u "
                            "series_targets=%u series_seeded=%u series_nested=%u "
                            "time_us={states:%llu,gcc:%llu,tcore_ctx:%llu,"
                            "tcore_up:%llu,tcore_down:%llu,absorb:%llu,"
                            "emit_tcore_s:%llu,macro_s_ctx:%llu,emit_macro_s:%llu,"
                            "macro_down:%llu,"
                            "emit_macro:%llu,emit_tcore:%llu,"
                            "emit_tcore_rr:%llu,emit_e:%llu,commit:%llu}\n",
                            static_cast<size_t>(plan.bid),
                            m_view.macros_len,
                            tctx.T_len,
                            emitted_tcore_s,
                            emitted_macro_s,
                            emitted_macro,
                            emitted_tcore,
                            emitted_tcore_rr,
                            emitted_e,
                            static_cast<unsigned long long>(stored_total),
                            count_only ? 1 : 0,
                            with_macro_down ? 1 : 0,
                            macro_down_seeded,
                            macro_down_nested,
                            macro_s_targets,
                            macro_s_down_seeded,
                            macro_s_down_nested,
                            static_cast<unsigned long long>(t_states_us),
                            static_cast<unsigned long long>(t_gcc_cuts_us),
                            static_cast<unsigned long long>(t_build_tcore_us),
                            static_cast<unsigned long long>(t_tcore_up_us),
                            static_cast<unsigned long long>(t_tcore_down_us),
                            static_cast<unsigned long long>(t_absorb_us),
                            static_cast<unsigned long long>(t_emit_tcore_s_us),
                            static_cast<unsigned long long>(t_macro_s_ctx_us),
                            static_cast<unsigned long long>(t_emit_macro_s_us),
                            static_cast<unsigned long long>(t_macro_down_us),
                            static_cast<unsigned long long>(t_emit_macro_us),
                            static_cast<unsigned long long>(t_emit_tcore_us),
                            static_cast<unsigned long long>(t_emit_tcore_rr_us),
                            static_cast<unsigned long long>(t_emit_e_us),
                            static_cast<unsigned long long>(t_commit_us));
                        std::fflush(stderr);
                    }

                    return;
                }

                if (!blk.spqr)
                    return;
                if (!blk.Gblk || blockNodeCount(blk) < 3)
                    return;

                auto &C = ctx();
                const auto &T = blk.spqr->tree();

                BF_INSTR(
                profiling_patch::blocks_with_spqr.fetch_add(1, std::memory_order_relaxed);
                profiling_patch::total_tree_nodes.fetch_add(T.numberOfNodes(), std::memory_order_relaxed);
                profiling_patch::BlockTiming *bt = profiling_patch::try_get_block_timing(plan.bid);
                if (bt) {
                    bt->bid = plan.bid;
                    bt->critical = plan.critical;
                    bt->blockNodes = blockNodeCount(blk);
                    bt->blockEdges = blockEdgeCount(blk);
                    bt->spqrTreeNodes = T.numberOfNodes();
                    bt->logicWeight = T.numberOfNodes() + blockEdgeCount(blk);
                    uint32_t cS = 0, cP = 0, cR = 0;
                    for (spqr_compat::node tn : T.nodes) {
                        auto ty = blk.spqr->typeOf(tn);
                        if (ty == spqr_compat::StaticSPQRTree::NodeType::SNode) ++cS;
                        else if (ty == spqr_compat::StaticSPQRTree::NodeType::PNode) ++cP;
                        else if (ty == spqr_compat::StaticSPQRTree::NodeType::RNode) ++cR;
                    }
                    bt->spqrSCount = cS;
                    bt->spqrPCount = cP;
                    bt->spqrRCount = cR;
                }
                )

                BF_INSTR(auto __sub_t0 = std::chrono::high_resolution_clock::now();)
                spqr_compat::EdgeArray<EdgeDP> edge_dp(T);
                spqr_compat::NodeArray<NodeDPState> node_dp(T);
                BF_INSTR(
                auto __sub_t1 = std::chrono::high_resolution_clock::now();
                uint64_t __dt_alloc = std::chrono::duration_cast<std::chrono::nanoseconds>(__sub_t1 - __sub_t0).count();
                profiling_patch::sub_alloc_dp_ns.fetch_add(__dt_alloc, std::memory_order_relaxed);
                if (bt) bt->sub_alloc_dp_ns = __dt_alloc;
                )

                std::vector<spqr_compat::node> nodeOrder;
                std::vector<spqr_compat::edge> edgeOrder;

                BF_INSTR(__sub_t0 = std::chrono::high_resolution_clock::now();)
                dfsSPQR_order(*blk.spqr, edgeOrder, nodeOrder);
                BF_INSTR(
                __sub_t1 = std::chrono::high_resolution_clock::now();
                uint64_t __dt_dfs = std::chrono::duration_cast<std::chrono::nanoseconds>(__sub_t1 - __sub_t0).count();
                profiling_patch::sub_dfs_order_ns.fetch_add(__dt_dfs, std::memory_order_relaxed);
                if (bt) bt->sub_dfs_order_ns = __dt_dfs;
                )

                BF_INSTR(__sub_t0 = std::chrono::high_resolution_clock::now();)
                blk.blkToSkel.init(*blk.Gblk, nullptr);
                BF_INSTR(
                __sub_t1 = std::chrono::high_resolution_clock::now();
                uint64_t __dt_bts = std::chrono::duration_cast<std::chrono::nanoseconds>(__sub_t1 - __sub_t0).count();
                profiling_patch::sub_blktoskel_init_ns.fetch_add(__dt_bts, std::memory_order_relaxed);
                if (bt) bt->sub_blktoskel_init_ns = __dt_bts;
                )

                BF_INSTR(
                if (bt) {
                    auto __levels_t0 = std::chrono::high_resolution_clock::now();
                    spqr_compat::NodeArray<uint32_t> depthArr(T, 0);
                    spqr_compat::NodeArray<uint32_t> heightArr(T, 0);
                    uint32_t maxDepth_loc = 0;
                    for (spqr_compat::node v : nodeOrder) {
                        spqr_compat::node p = blk.parent[v];
                        uint32_t d = (p == v) ? 0 : (depthArr[p] + 1);
                        depthArr[v] = d;
                        if (d > maxDepth_loc) maxDepth_loc = d;
                    }
                    uint32_t maxHeight_loc = 0;
                    for (size_t i = nodeOrder.size(); i-- > 0; ) {
                        spqr_compat::node v = nodeOrder[i];
                        uint32_t h = 0;
                        T.forEachAdj(v, [&](spqr_compat::node nb, spqr_compat::edge) {
                            if (nb == blk.parent[v]) return;
                            uint32_t hc = heightArr[nb] + 1;
                            if (hc > h) h = hc;
                        });
                        heightArr[v] = h;
                        if (h > maxHeight_loc) maxHeight_loc = h;
                    }
                    bt->maxDepth = maxDepth_loc;
                    bt->maxHeight = maxHeight_loc;
                    bt->widthByDepth.assign(static_cast<size_t>(maxDepth_loc) + 1, 0);
                    bt->widthByHeight.assign(static_cast<size_t>(maxHeight_loc) + 1, 0);
                    for (spqr_compat::node v : nodeOrder) {
                        bt->widthByDepth[depthArr[v]] += 1;
                        bt->widthByHeight[heightArr[v]] += 1;
                    }
                    auto __levels_t1 = std::chrono::high_resolution_clock::now();
                    uint64_t __dt_lvl = std::chrono::duration_cast<std::chrono::nanoseconds>(__levels_t1 - __levels_t0).count();
                    profiling_patch::sub_levels_ns.fetch_add(__dt_lvl, std::memory_order_relaxed);
                    bt->sub_levels_ns = __dt_lvl;
                }
                )

                {
                BF_INSTR(auto __sp_t0 = std::chrono::high_resolution_clock::now();)
                for (spqr_compat::edge e : edgeOrder)
                {
                    processEdge(e, edge_dp, cc, blk);
                }
                BF_INSTR(
                auto __sp_t1 = std::chrono::high_resolution_clock::now();
                uint64_t __dt_p1 = std::chrono::duration_cast<std::chrono::nanoseconds>(__sp_t1 - __sp_t0).count();
                profiling_patch::phase1_edge_dp_time_ns.fetch_add(__dt_p1, std::memory_order_relaxed);
                profiling_patch::phase1_calls.fetch_add(1, std::memory_order_relaxed);
                if (bt) bt->phase1_ns = __dt_p1;
                )
                }

                {
                BF_INSTR(auto __sp_t0 = std::chrono::high_resolution_clock::now();)
                for (spqr_compat::node v : nodeOrder)
                {
                    processNode(v, edge_dp, cc, blk);
                }
                BF_INSTR(
                auto __sp_t1 = std::chrono::high_resolution_clock::now();
                uint64_t __dt_p2 = std::chrono::duration_cast<std::chrono::nanoseconds>(__sp_t1 - __sp_t0).count();
                profiling_patch::phase2_node_dp_time_ns.fetch_add(__dt_p2, std::memory_order_relaxed);
                profiling_patch::phase2_calls.fetch_add(1, std::memory_order_relaxed);
                if (bt) bt->phase2_ns = __dt_p2;
                )
                }

                {
                BF_INSTR(auto __sp_t0 = std::chrono::high_resolution_clock::now();)
                solveNodes(node_dp, edge_dp, blk, cc, plan);
                BF_INSTR(
                auto __sp_t1 = std::chrono::high_resolution_clock::now();
                uint64_t __dt_p3 = std::chrono::duration_cast<std::chrono::nanoseconds>(__sp_t1 - __sp_t0).count();
                profiling_patch::phase3_solve_time_ns.fetch_add(__dt_p3, std::memory_order_relaxed);
                profiling_patch::phase3_calls.fetch_add(1, std::memory_order_relaxed);
                if (bt) bt->phase3_ns = __dt_p3;
                )
                }

                BF_INSTR(auto __sp_t0_p4 = std::chrono::high_resolution_clock::now();)
                BF_INSTR(auto __p4_setup_t0 = std::chrono::high_resolution_clock::now();)

                std::vector<std::vector<spqr_compat::node>> block_vertexInSnodes; 
                std::vector<std::vector<spqr_compat::node>> *vertexInSnodes_ptr = nullptr;

                thread_local std::vector<std::vector<spqr_compat::node>> tls_vertexInSnodes;
                thread_local std::vector<uint32_t> tls_vertexInSnodes_dirty;

                const size_t nBlkNodes = blockNodeCount(blk);

                if (plan.critical) {
                    block_vertexInSnodes.resize(nBlkNodes);
                    vertexInSnodes_ptr = &block_vertexInSnodes;
                } else {
                    for (uint32_t idx : tls_vertexInSnodes_dirty) {
                        tls_vertexInSnodes[idx].clear();
                    }
                    tls_vertexInSnodes_dirty.clear();
                    if (tls_vertexInSnodes.size() < nBlkNodes) {
                        tls_vertexInSnodes.resize(nBlkNodes);
                    }
                    vertexInSnodes_ptr = &tls_vertexInSnodes;
                }

                std::vector<std::vector<spqr_compat::node>> &vertexInSnodes = *vertexInSnodes_ptr;

                for (spqr_compat::node mu : T.nodes) {
                    if (blk.spqr->typeOf(mu) != spqr_compat::StaticSPQRTree::NodeType::SNode)
                        continue;
                    const spqr_compat::Skeleton &skel_p4 = blk.spqr->skeleton(mu);
                    const auto &skelG_p4 = skel_p4.getGraph();
                    for (spqr_compat::node vSk : skelG_p4.nodes) {
                        spqr_compat::node vBlk = skel_p4.original(vSk);
                        if (!plan.critical && vertexInSnodes[vBlk.idx].empty()) {
                            tls_vertexInSnodes_dirty.push_back(vBlk.idx);
                        }
                        vertexInSnodes[vBlk.idx].push_back(mu);
                    }
                }

                BF_INSTR(
                auto __p4_setup_t1 = std::chrono::high_resolution_clock::now();
                uint64_t __dt_p4_setup = std::chrono::duration_cast<std::chrono::nanoseconds>(__p4_setup_t1 - __p4_setup_t0).count();
                profiling_patch::sub_phase4_setup_ns.fetch_add(__dt_p4_setup, std::memory_order_relaxed);
                if (bt) bt->phase4_setup_ns = __dt_p4_setup;
                )

                auto shareSnode = [&](spqr_compat::node aB, spqr_compat::node bB) -> bool {
                    const auto &La = vertexInSnodes[aB.idx];
                    const auto &Lb = vertexInSnodes[bB.idx];
                    if (La.empty() || Lb.empty()) return false;

                    const auto &Small = (La.size() <= Lb.size()) ? La : Lb;
                    const auto &Big = (La.size() <= Lb.size()) ? Lb : La;

                    if (Small.size() <= 4) {
                        for (spqr_compat::node x : Small) {
                            for (spqr_compat::node y : Big) {
                                if (x == y) return true;
                            }
                        }
                        return false;
                    }

                    std::unordered_set<int> share_set;
                    share_set.reserve(Big.size());
                    for (spqr_compat::node y : Big) share_set.insert(y.idx);
                    for (spqr_compat::node x : Small) {
                        if (share_set.count(x.idx)) return true;
                    }
                    return false;
                };

                auto hasDanglingOutside = [&](spqr_compat::node vGcc) {
                    if (!cc.isCutNode[vGcc]) return false;
                    if (cc.badCutCount[vGcc] >= 2) return true;
                    if (cc.badCutCount[vGcc] == 1 && cc.lastBad[vGcc] != blk.bNode)
                        return true;
                    return false;
                };

                auto flipSign = [](EdgePartType t) {
                    return (t == EdgePartType::PLUS ? EdgePartType::MINUS : EdgePartType::PLUS);
                };

                auto caseE_body = [&](spqr_compat::edge eB) {
                    spqr_compat::edge eG = blockEdgeToOrig(blk, eB);

                    spqr_compat::node uB = blockEdgeSource(blk, eB);
                    spqr_compat::node vB = blockEdgeTarget(blk, eB);

                    spqr_compat::node uGcc = blockNodeToCc(blk, uB);
                    spqr_compat::node vGcc = blockNodeToCc(blk, vB);

                    spqr_compat::node uG = cc.nodeToOrig[uGcc];
                    spqr_compat::node vG = cc.nodeToOrig[vGcc];

                    if (isTrashOriginalNode(uG) || isTrashOriginalNode(vG))
                        return;

                    if (cc.isTip[uGcc] || cc.isTip[vGcc])
                        return;

                    if (hasDanglingOutside(uGcc) || hasDanglingOutside(vGcc))
                        return;

                    EdgePartType edgeSignU = getNodeEdgeType(uG, eG);
                    EdgePartType edgeSignV = getNodeEdgeType(vG, eG);

                    auto check_one_vertex = [&](spqr_compat::node vNode,
                                                EdgePartType sign,
                                                EdgePartType eSign) {
                        int totPlus = blockDegPlus(blk, vNode);
                        int totMinus = blockDegMinus(blk, vNode);
                        if (sign == EdgePartType::PLUS) {
                            if (eSign == EdgePartType::PLUS) {
                                int othersPlus = totPlus - 1;
                                int othersMinus = totMinus;
                                return (othersPlus == 0 && othersMinus > 0);
                            } else {
                                int othersPlus = totPlus;
                                int othersMinus = totMinus - 1;
                                return (othersMinus == 0 && othersPlus > 0);
                            }
                        } else {
                            if (eSign == EdgePartType::MINUS) {
                                int othersMinus = totMinus - 1;
                                int othersPlus = totPlus;
                                return (othersMinus == 0 && othersPlus > 0);
                            } else {
                                int othersMinus = totMinus;
                                int othersPlus = totPlus - 1;
                                return (othersPlus == 0 && othersMinus > 0);
                            }
                        }
                    };

                    auto testCandidate = [&](EdgePartType signU,
                                             EdgePartType signV,
                                             bool isFlipCase) {
                        if (isFlipCase && shareSnode(uB, vB))
                            return;
                        if (!check_one_vertex(uB, signU, edgeSignU))
                            return;
                        if (!check_one_vertex(vB, signV, edgeSignV))
                            return;
                        addSnarlTaggedPairNodes("E", uG, signU, vG, signV);
                    };

                    testCandidate(edgeSignU, edgeSignV, false);
                    testCandidate(flipSign(edgeSignU), flipSign(edgeSignV), true);
                };

                if (!plan.critical) {
                    BF_INSTR(auto __p4_body_t0 = std::chrono::high_resolution_clock::now();)
	                    for (spqr_compat::edge eB : collectBlockEdges(blk)) {
	                        caseE_body(eB);
	                    }
                    BF_INSTR(
                    auto __p4_body_t1 = std::chrono::high_resolution_clock::now();
                    uint64_t __dt_p4_body = std::chrono::duration_cast<std::chrono::nanoseconds>(__p4_body_t1 - __p4_body_t0).count();
                    profiling_patch::sub_phase4_caseE_body_ns.fetch_add(__dt_p4_body, std::memory_order_relaxed);
                    if (bt) bt->phase4_caseE_body_ns = __dt_p4_body;
                    profiling_patch::taskloops_caseE_skipped.fetch_add(1, std::memory_order_relaxed);
                    )
                } else {
                    BF_INSTR(auto __p4_collect_t0 = std::chrono::high_resolution_clock::now();)
	                    std::vector<spqr_compat::edge> blockEdges = collectBlockEdges(blk);
                    BF_INSTR(
                    auto __p4_collect_t1 = std::chrono::high_resolution_clock::now();
                    uint64_t __dt_p4_collect = std::chrono::duration_cast<std::chrono::nanoseconds>(__p4_collect_t1 - __p4_collect_t0).count();
                    profiling_patch::sub_caseE_collect_ns.fetch_add(__dt_p4_collect, std::memory_order_relaxed);
                    if (bt) bt->phase4_caseE_collect_ns = __dt_p4_collect;
                    )

                    const uint64_t n = blockEdges.size();
                    const uint64_t W = blockEdgeCount(blk);
                    const uint64_t nT = bf_choose_num_tasks(n, W, plan);

                    BF_INSTR(auto __p4_body_t0 = std::chrono::high_resolution_clock::now();)
                    if (nT == 0) {
                        BF_INSTR(
                        profiling_patch::taskloops_caseE_skipped.fetch_add(1, std::memory_order_relaxed);
                        )
                        for (size_t i = 0; i < blockEdges.size(); ++i)
                            caseE_body(blockEdges[i]);
                    } else {
                        BF_INSTR(
                        profiling_patch::taskloops_caseE_created.fetch_add(1, std::memory_order_relaxed);
                        profiling_patch::tasks_requested_caseE_total.fetch_add(nT, std::memory_order_relaxed);
                        if (bt) {
                            bt->taskloops_caseE_created += 1;
                            bt->tasks_requested_caseE += nT;
                        }
                        )
                        if (plan.activeIntraTaskloops)
                            plan.activeIntraTaskloops->fetch_add(1, std::memory_order_relaxed);

                        BF_OMP_PRAGMA(omp taskloop num_tasks(nT) default(shared))
                        for (long long i = 0; i < static_cast<long long>(blockEdges.size()); ++i) {
                            caseE_body(blockEdges[i]);
                        }

                        if (plan.activeIntraTaskloops)
                            plan.activeIntraTaskloops->fetch_sub(1, std::memory_order_relaxed);
                    }
                    BF_INSTR(
                    auto __p4_body_t1 = std::chrono::high_resolution_clock::now();
                    uint64_t __dt_p4_body = std::chrono::duration_cast<std::chrono::nanoseconds>(__p4_body_t1 - __p4_body_t0).count();
                    profiling_patch::sub_phase4_caseE_body_ns.fetch_add(__dt_p4_body, std::memory_order_relaxed);
                    if (bt) bt->phase4_caseE_body_ns = __dt_p4_body;
                    )
                }

                BF_INSTR(
                auto __sp_t1_p4 = std::chrono::high_resolution_clock::now();
                uint64_t __dt_p4 = std::chrono::duration_cast<std::chrono::nanoseconds>(__sp_t1_p4 - __sp_t0_p4).count();
                profiling_patch::phase4_caseE_time_ns.fetch_add(__dt_p4, std::memory_order_relaxed);
                profiling_patch::phase4_calls.fetch_add(1, std::memory_order_relaxed);
                )
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

                cc.Gcc->forEachAdj(v, [&](node /*neighbor*/, edge eAdj) {
                    spqr_compat::edge e = cc.edgeToOrig[eAdj];
                    const auto edgeTypes = C._edge2types(e);
                    if (cc.Gcc->source(eAdj) == v && cc.Gcc->target(eAdj) == v) {
                        addIncidentTypeCount(plusCnt, minusCnt, edgeTypes.first);
                        addIncidentTypeCount(plusCnt, minusCnt, edgeTypes.second);
                        return;
                    }
                    EdgePartType eType =
                        (cc.Gcc->source(eAdj) == v) ? edgeTypes.first : edgeTypes.second;
                    addIncidentTypeCount(plusCnt, minusCnt, eType);
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

                if (VERBOSE) {
                    const std::string &name = C.node2name[vG];
                    std::cerr << "[DEBUG][findTips] node " << name
                              << "(Gcc idx=" << v.index() << ")"
                              << "plusCnt=" << plusCnt
                              << "minusCnt=" << minusCnt
                              << "isTip=" << (cc.isTip[v] ? "true" : "false")
                              << "\n";
                }
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
            spqr_compat::EdgeArray<uint32_t> edge2block(*cc.Gcc, UINT32_MAX);
            for (uint32_t bIdx = 0; bIdx < numB; ++bIdx)
            {
                spqr_compat::node bNode{bIdx};
                for (spqr_compat::edge eCc : cc.bc->hEdges(bNode))
                {
                    edge2block[eCc] = bIdx;
                }
            }

            for (node v : cc.Gcc->nodes)
            {
                node vG = cc.nodeToOrig[v];
                if (cc.bc->typeOfGNode(v) == BCTree::GNodeType::CutVertex)
                {
                    cc.isCutNode[v] = true;

                    struct BlockFlags { uint32_t bIdx; bool hasPlus; bool hasMinus; };
                    std::vector<BlockFlags> blocks;
                    blocks.reserve(8);

                    cc.Gcc->forEachAdj(v, [&](spqr_compat::node /*nb*/, spqr_compat::edge eCc) {
                        spqr_compat::edge eG = cc.edgeToOrig[eCc];
                        if (!eG) return;
                        const auto edgeTypes = C._edge2types(eG);
                        if (cc.Gcc->source(eCc) == v && cc.Gcc->target(eCc) == v) {
                            if (edgeTypes.first != edgeTypes.second) {
                                BlockFlags *slot = nullptr;
                                uint32_t bIdx = edge2block[eCc];
                                for (auto &bf : blocks) {
                                    if (bf.bIdx == bIdx) { slot = &bf; break; }
                                }
                                if (!slot) {
                                    blocks.push_back({bIdx, false, false});
                                    slot = &blocks.back();
                                }
                                slot->hasPlus = true;
                                slot->hasMinus = true;
                            }
                            return;
                        }
                        uint32_t bIdx = edge2block[eCc];
                        if (bIdx == UINT32_MAX) return;  // edge not in any block (shouldn't happen)
                        EdgePartType outType =
                            (cc.Gcc->source(eCc) == v) ? edgeTypes.first : edgeTypes.second;

                        BlockFlags *slot = nullptr;
                        for (auto &bf : blocks) {
                            if (bf.bIdx == bIdx) { slot = &bf; break; }
                        }
                        if (!slot) {
                            blocks.push_back({bIdx, false, false});
                            slot = &blocks.back();
                        }
                        if (outType == EdgePartType::PLUS) slot->hasPlus = true;
                        if (outType == EdgePartType::MINUS) slot->hasMinus = true;
                    });

                    bool isGood = true;
                    for (auto &bf : blocks) {
                        if (bf.hasPlus && bf.hasMinus) {
                            isGood = false;
                            cc.lastBad[v] = spqr_compat::node{bf.bIdx};
                            cc.badCutCount[v]++;
                        }
                    }
                    cc.isGoodCutNode[v] = isGood;
                }
                if (VERBOSE) {
                    const std::string &name = C.node2name[vG];
                    std::cerr << "[DEBUG][processCutNodes] node " << name
                              << "(Gcc idx=" << v.index() << ")"
                              << "isCutNode=" << (cc.isCutNode[v] ? "true" : "false")
                              << "badCutCount=" << cc.badCutCount[v]
                              << "isGoodCutNode=" << (cc.isGoodCutNode[v] ? "true" : "false");
                    if (cc.lastBad[v] != nullptr)
                    {
                        std::cerr << "lastBad(B-node idx)=" << cc.lastBad[v].index();
                    }
                    std::cerr << "\n";
                }
            }
        }

        void findCutSnarl(CcData &cc)
        {
            MARK_SCOPE_MEM("sn/findCutSnarl");
            auto &C = ctx();

            spqr_compat::NodeArray<std::pair<bool, bool>> visited(
                *cc.Gcc, {false, false}); 

            auto isStateVisited = [&](spqr_compat::node v, EdgePartType t) -> bool {
                return (t == EdgePartType::PLUS) ? visited[v].second : visited[v].first;
            };

            auto flipSign = [](EdgePartType t) {
                return (t == EdgePartType::PLUS) ? EdgePartType::MINUS : EdgePartType::PLUS;
            };

            for (spqr_compat::node start : cc.Gcc->nodes)
            {
                for (auto t : {EdgePartType::PLUS, EdgePartType::MINUS})
                {
                    if (isStateVisited(start, t))
                        continue;

                    std::vector<SnarlEndpoint> goodNodes;

                    struct Frame
                    {
                        spqr_compat::node v;
                        EdgePartType edgeType;
                    };
                    std::vector<Frame> st;
                    st.reserve(1024);
                    st.push_back({start, t});

                    while (!st.empty())
                    {
                        auto [v, edgeType] = st.back();
                        st.pop_back();

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
                            !isTrashOriginalNode(cc.nodeToOrig[v]))
                        {

                            goodNodes.push_back({cc.nodeToOrig[v], edgeType});
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

                        bool canGoOther = !cc.isGoodCutNode[v] && !cc.isTip[v];
                        const EdgePartType otherEdgeType = flipSign(edgeType);

                        cc.Gcc->forEachAdj(v, [&](spqr_compat::node otherNode, spqr_compat::edge eCc) {
                            spqr_compat::edge eOrig = cc.edgeToOrig[eCc];
                            if (!eOrig) return;

                            const auto edgeTypes = C._edge2types(eOrig);
                            auto pushIfReachable = [&](spqr_compat::node to,
                                                       EdgePartType outType,
                                                       EdgePartType inType) {
                                if (outType != edgeType &&
                                    (!canGoOther || outType != otherEdgeType)) {
                                    return;
                                }
                                if (!isStateVisited(to, inType)) {
                                    st.push_back({to, inType});
                                }
                            };

                            if (cc.Gcc->source(eCc) == v && cc.Gcc->target(eCc) == v) {
                                pushIfReachable(v, edgeTypes.first, edgeTypes.second);
                                pushIfReachable(v, edgeTypes.second, edgeTypes.first);
                            } else if (cc.Gcc->source(eCc) == v) {
                                pushIfReachable(otherNode, edgeTypes.first, edgeTypes.second);
                            } else {
                                pushIfReachable(otherNode, edgeTypes.second, edgeTypes.first);
                            }
                        });
                    }

                    if (goodNodes.size() >= 2)
                    {
                        addSnarlTaggedEndpoints("CUT", goodNodes);
                    }
                }
            }
        }

        inline bool usesCompressedSpqr(Context::SpCompressMode mode)
        {
            return mode == Context::SpCompressMode::On ||
                   mode == Context::SpCompressMode::Instrument ||
                   mode == Context::SpCompressMode::MacroDirectDebug;
        }

        inline void materializeFlatEdges(BlockData &blk)
        {
            if (!blk.flatEdges) return;
            if (blk.flatNodes)
            {
                blk.Gblk = std::make_unique<spqr_compat::Graph>();
                for (size_t i = 0; i < blk.flatNodeToOrig.size(); ++i)
                    blk.Gblk->newNode();
                blk.nodeToOrig.init(*blk.Gblk, nullptr);
                blk.toCc.init(*blk.Gblk, nullptr);
                blk.edgeToOrig.init(*blk.Gblk, nullptr);
                blk.blkDegPlus.init(*blk.Gblk, 0);
                blk.blkDegMinus.init(*blk.Gblk, 0);
                for (spqr_compat::node v : blk.Gblk->nodes)
                {
                    blk.nodeToOrig[v] = blk.flatNodeToOrig[v.idx];
                    blk.toCc[v] = blk.flatNodeToCc[v.idx];
                    blk.blkDegPlus[v] = blk.flatDegPlus[v.idx];
                    blk.blkDegMinus[v] = blk.flatDegMinus[v.idx];
                }
                std::vector<spqr_compat::node>().swap(blk.flatNodeToOrig);
                std::vector<spqr_compat::node>().swap(blk.flatNodeToCc);
                blk.flatNodes = false;
            }
            for (uint32_t i = 0; i < blk.flatEdgeCount; ++i)
            {
                const SpCompressInputEdge &ie = blk.flatFfiEdges[i];
                spqr_compat::edge e = blk.Gblk->newEdge(
                    spqr_compat::node{ie.u},
                    spqr_compat::node{ie.v});
                blk.edgeToOrig[e] = blk.flatEdgeToOrig[i];
            }
            blk.flatEdgeToOrig.clear();
            blk.flatFfiEdges.clear();
            blk.flatEdgeSrc.clear();
            blk.flatEdgeDst.clear();
            blk.flatDegPlus.clear();
            blk.flatDegMinus.clear();
            blk.flatEdges = false;
        }

        inline std::vector<uint8_t> buildContractibleMask(const BlockData &blk)
        {
            std::vector<uint8_t> contractible(blockNodeCount(blk), 0);
            if (blk.flatEdges)
            {
                for (uint32_t i = 0; i < blk.flatDegPlus.size(); ++i)
                {
                    if (blk.flatDegPlus[i] + blk.flatDegMinus[i] == 2)
                        contractible[i] = 1;
                }
                return contractible;
            }
            for (spqr_compat::node v : blk.Gblk->nodes)
            {
                if (blk.Gblk->degree(v) == 2u)
                    contractible[v.idx] = 1;
            }
            return contractible;
        }

        inline std::vector<SpCompressInputEdge> buildSpCompressEdges(const BlockData &blk)
        {
            std::vector<SpCompressInputEdge> ffi_edges;
            ffi_edges.reserve(blockEdgeCount(blk));
            uint32_t fid = 0;
            if (blk.flatEdges)
            {
                for (uint32_t i = 0; i < blk.flatFfiEdges.size(); ++i)
                {
                    SpCompressInputEdge ie = blk.flatFfiEdges[i];
                    ie.original_edge_id = fid++;
                    ffi_edges.push_back(ie);
                }
                return ffi_edges;
            }
            for (spqr_compat::edge e : blk.Gblk->edges)
            {
                ffi_edges.push_back(SpCompressInputEdge{
                    static_cast<uint32_t>(blk.Gblk->source(e).idx),
                    static_cast<uint32_t>(blk.Gblk->target(e).idx),
                    fid++});
            }
            return ffi_edges;
        }

        inline void emitMacroDirectBuildStats(
            uint32_t n_blk,
            uint32_t n_edges,
            const BlockData &blk,
            const SpCompressTimings &timings,
            uint64_t ffi_total_us)
        {
            std::fprintf(stderr,
                "[macro_direct_build] block nodes=%u edges=%u "
                "macros=%u core_edges=%u core_tree=%s "
                "time_us={ffi_total:%llu,total:%llu,compress:%llu,"
                "core_total:%llu,core_remap:%llu,core_graph:%llu,"
                "core_spqr:%llu,handle:%llu}\n",
                n_blk,
                n_edges,
                blk.macroTreeView.macros_len,
                blk.macroTreeView.core_edges_len,
                blk.coreSpqrTree ? "yes" : "no",
                static_cast<unsigned long long>(ffi_total_us),
                static_cast<unsigned long long>(timings.t_total_us),
                static_cast<unsigned long long>(timings.t_compress_us),
                static_cast<unsigned long long>(timings.t_build_spqr_core_us),
                static_cast<unsigned long long>(timings.t_core_remap_us),
                static_cast<unsigned long long>(timings.t_core_graph_build_us),
                static_cast<unsigned long long>(timings.t_core_spqr_raw_us),
                static_cast<unsigned long long>(timings.t_handle_wrap_us));
            std::fprintf(stderr,
                "[macro_direct_build_detail] block nodes=%u edges=%u "
                "macros=%u "
                "compress_us={input_edges:%llu,init_work:%llu,"
                "init_dirty:%llu,reduce_series:%llu,"
                "reduce_parallel:%llu,materialize:%llu,"
                "cleanup:%llu,canon_series:%llu,sort_core:%llu,"
                "collect_core_nodes:%llu,stats_shrink:%llu} "
                "spqr_us={self_loop_scan:%llu,tree_total:%llu,"
                "precheck:%llu,split_multi:%llu,work_graph:%llu,"
                "triconn:%llu,relabel:%llu,combine:%llu,"
                "merge:%llu,assemble:%llu} "
                "split_counts={multi:%llu,triconn:%llu,precombine:%llu,"
                "combined:%llu,merged:%llu,merged_real:%llu,"
                "merged_virtual_inc:%llu,virtual_id_span:%llu,"
                "tree_nodes:%llu,tree_edges:%llu,tree_skel_edges:%llu,"
                "tree_virtual_inc:%llu}\n",
                n_blk,
                n_edges,
                blk.macroTreeView.macros_len,
                static_cast<unsigned long long>(timings.t_compress_input_edges_us),
                static_cast<unsigned long long>(timings.t_compress_init_work_us),
                static_cast<unsigned long long>(timings.t_compress_init_dirty_us),
                static_cast<unsigned long long>(timings.t_compress_reduce_series_us),
                static_cast<unsigned long long>(timings.t_compress_reduce_parallel_us),
                static_cast<unsigned long long>(timings.t_compress_materialize_us),
                static_cast<unsigned long long>(timings.t_compress_cleanup_us),
                static_cast<unsigned long long>(timings.t_compress_canon_series_us),
                static_cast<unsigned long long>(timings.t_compress_sort_core_edges_us),
                static_cast<unsigned long long>(timings.t_compress_collect_core_nodes_us),
                static_cast<unsigned long long>(timings.t_compress_stats_shrink_us),
                static_cast<unsigned long long>(timings.t_spqr_self_loop_scan_us),
                static_cast<unsigned long long>(timings.t_spqr_tree_total_us),
                static_cast<unsigned long long>(timings.t_spqr_precheck_us),
                static_cast<unsigned long long>(timings.t_spqr_split_multi_edges_us),
                static_cast<unsigned long long>(timings.t_spqr_work_graph_us),
                static_cast<unsigned long long>(timings.t_spqr_triconn_us),
                static_cast<unsigned long long>(timings.t_spqr_relabel_us),
                static_cast<unsigned long long>(timings.t_spqr_combine_us),
                static_cast<unsigned long long>(timings.t_spqr_merge_us),
                static_cast<unsigned long long>(timings.t_spqr_assemble_us),
                static_cast<unsigned long long>(timings.c_spqr_multi_components),
                static_cast<unsigned long long>(timings.c_spqr_triconn_components),
                static_cast<unsigned long long>(timings.c_spqr_precombine_components),
                static_cast<unsigned long long>(timings.c_spqr_combined_components),
                static_cast<unsigned long long>(timings.c_spqr_merged_components),
                static_cast<unsigned long long>(timings.c_spqr_merged_real_edges),
                static_cast<unsigned long long>(timings.c_spqr_merged_virtual_incidences),
                static_cast<unsigned long long>(timings.c_spqr_virtual_id_span),
                static_cast<unsigned long long>(timings.c_spqr_tree_nodes),
                static_cast<unsigned long long>(timings.c_spqr_tree_edges),
                static_cast<unsigned long long>(timings.c_spqr_tree_skeleton_edges),
                static_cast<unsigned long long>(timings.c_spqr_tree_virtual_incidences));
        }

        inline bool buildBlockSpqr(BlockData &blk)
        {
            SPQR_RUST_ASSERT(blk.Gblk != nullptr);
            SPQR_RUST_ASSERT(blockNodeCount(blk) > 0);

            const auto mode = ctx().spCompressMode;
            if (!usesCompressedSpqr(mode))
            {
                blk.spqr = std::make_unique<spqr_compat::StaticSPQRTree>(*blk.Gblk);
                return true;
            }

            const uint32_t n_blk = blockNodeCount(blk);
            std::vector<uint8_t> contractible = buildContractibleMask(blk);

            if (mode == Context::SpCompressMode::MacroDirectDebug)
            {
                std::vector<SpCompressInputEdge> ffi_edges_storage;
                const SpCompressInputEdge *ffi_edges_ptr = nullptr;
                uint32_t ffi_edges_len = 0;
                if (blk.flatEdges)
                {
                    ffi_edges_ptr = blk.flatFfiEdges.empty() ? nullptr : blk.flatFfiEdges.data();
                    ffi_edges_len = static_cast<uint32_t>(blk.flatFfiEdges.size());
                }
                else
                {
                    ffi_edges_storage = buildSpCompressEdges(blk);
                    ffi_edges_ptr = ffi_edges_storage.empty() ? nullptr : ffi_edges_storage.data();
                    ffi_edges_len = static_cast<uint32_t>(ffi_edges_storage.size());
                }
                const bool stats_enabled = (std::getenv("BF_MACRO_DIRECT_STATS") != nullptr);
                SpCompressTimings timings{};
                const auto t0 = std::chrono::steady_clock::now();
                SpCompressHandle *raw_handle =
                    stats_enabled
                        ? sp_compress_timed_ffi(
                              n_blk,
                              ffi_edges_ptr,
                              ffi_edges_len,
                              contractible.data(),
                              static_cast<uint32_t>(contractible.size()),
                              /*build_core_spqr=*/1,
                              &timings)
                        : sp_compress_ffi(
                              n_blk,
                              ffi_edges_ptr,
                              ffi_edges_len,
                              contractible.data(),
                              static_cast<uint32_t>(contractible.size()),
                              /*build_core_spqr=*/1);
                const auto t1 = std::chrono::steady_clock::now();
                const uint64_t ffi_total_us =
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

                if (!raw_handle || sp_compress_success(raw_handle) == 0)
                {
                    if (raw_handle) sp_compress_free(raw_handle);
                    throw std::runtime_error("sp_compress_ffi failed in macro-direct mode");
                }

                blk.spCompressHandle.reset(raw_handle);
                blk.macroTreeView = sp_compress_get_tree(raw_handle);
                blk.coreSpqrTree = sp_compress_get_core_spqr(raw_handle);
                blk.coreNodeInv = sp_compress_core_node_inv(raw_handle, &blk.coreNodeInvLen);

                if (stats_enabled)
                {
                    emitMacroDirectBuildStats(
                        n_blk,
                        ffi_edges_len,
                        blk,
                        timings,
                        ffi_total_us);
                }
                if (blk.macroTreeView.macros_len == 0)
                {
                    blk.spCompressHandle.reset();
                    blk.macroTreeView = SpCompressTreeView{};
                    blk.coreSpqrTree = nullptr;
                    blk.coreNodeInv = nullptr;
                    blk.coreNodeInvLen = 0;
                    materializeFlatEdges(blk);
                    blk.spqr = std::make_unique<spqr_compat::StaticSPQRTree>(*blk.Gblk);
                    return true;
                }
                if (blk.flatEdges)
                {
                    std::vector<SpCompressInputEdge>().swap(blk.flatFfiEdges);
                    std::vector<uint32_t>().swap(blk.flatEdgeSrc);
                    std::vector<uint32_t>().swap(blk.flatEdgeDst);
                }
                return false;
            }

            if (mode == Context::SpCompressMode::Instrument)
            {
                std::vector<SpCompressInputEdge> ffi_edges = buildSpCompressEdges(blk);
                const uint32_t n_edges = static_cast<uint32_t>(ffi_edges.size());

                auto t0 = std::chrono::steady_clock::now();
                auto baseline_tree = std::make_unique<spqr_compat::StaticSPQRTree>(*blk.Gblk);
                auto t1 = std::chrono::steady_clock::now();

                SpCompressStats stats{};
                SpCompressTimings timings{};
                auto t2 = std::chrono::steady_clock::now();
                SpqrResult* compress_raw = sp_compress_reconstruct_with_timings_ffi(
                    n_blk,
                    ffi_edges.data(), n_edges,
                    contractible.data(),
                    static_cast<uint32_t>(contractible.size()),
                    &stats,
                    &timings);
                auto t3 = std::chrono::steady_clock::now();

                if (compress_raw) spqr_result_free(compress_raw);

                const auto t_base_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                const auto t_comp_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
                const long long gain_us = t_base_us - t_comp_us;
                const double gain_pct =
                    t_base_us > 0 ? (100.0 * static_cast<double>(gain_us) / static_cast<double>(t_base_us)) : 0.0;

                static std::ofstream csv_stream;
                static bool csv_init = false;
                static int block_idx = 0;
                if (!csv_init) {
                    const std::string& path = ctx().spCompressInstrumentCsv;
                    if (!path.empty()) {
                        csv_stream.open(path);
                    }
                    std::ostream& os = csv_stream.is_open() ? static_cast<std::ostream&>(csv_stream) : std::cerr;
                    os << "block_idx,n_nodes,n_edges,core_nodes,core_edges,"
                          "macro_count,macro_series,macro_parallel,"
                          "fully_reducible,t_baseline_us,t_spcompress_us,"
                          "gain_us,gain_pct,"
                          "t_compress_us,t_build_spqr_core_us,"
                          "t_reconstruct_us,t_normalize_us,t_canonicalize_us,"
                          "t_canon_root_us,t_canon_node_order_us,"
                          "t_canon_edge_orient_us,t_canon_move_root_us,"
                          "t_reconstruct_build_builder_us,"
                          "t_reconstruct_normalize_in_place_us,"
                          "t_reconstruct_finalize_us,"
                          "t_reconstruct_defensive_normalize_us\n";
                    csv_init = true;
                }
                std::ostream& os = csv_stream.is_open() ? static_cast<std::ostream&>(csv_stream) : std::cerr;
                os << block_idx++ << "," << n_blk << "," << n_edges << ","
                   << stats.core_nodes << "," << stats.core_edges_count << ","
                   << stats.macro_count << "," << stats.macro_series << ","
                   << stats.macro_parallel << ","
                   << static_cast<int>(stats.fully_sp_reducible) << ","
                   << t_base_us << "," << t_comp_us << "," << gain_us << ","
                   << gain_pct << ","
                   << timings.t_compress_us << ","
                   << timings.t_build_spqr_core_us << ","
                   << timings.t_reconstruct_us << ","
                   << timings.t_normalize_us << ","
                   << timings.t_canonicalize_us << ","
                   << timings.t_canon_root_us << ","
                   << timings.t_canon_node_order_us << ","
                   << timings.t_canon_edge_orient_us << ","
                   << timings.t_canon_move_root_us << ","
                   << timings.t_reconstruct_build_builder_us << ","
                   << timings.t_reconstruct_normalize_in_place_us << ","
                   << timings.t_reconstruct_finalize_us << ","
                   << timings.t_reconstruct_defensive_normalize_us << "\n";
                if (block_idx % 100 == 0) os.flush();

                blk.spqr = std::move(baseline_tree);
                return true;
            }

            blk.spqr = std::make_unique<spqr_compat::StaticSPQRTree>(
                *blk.Gblk,
                contractible.data(),
                static_cast<uint32_t>(contractible.size()));
            return true;
        }

        void buildBlockData(BlockData &blk, CcData &cc)
        {
            PROFILE_FUNCTION();

            VLOG << "[DEBUG][buildBlockData] --------\n";
            VLOG << "[DEBUG][buildBlockData] B-node index=" << blk.bNode.index() << "\n";

            blk.Gblk = std::make_unique<spqr_compat::Graph>();

            blk.nodeToOrig.init(*blk.Gblk, nullptr);
            blk.edgeToOrig.init(*blk.Gblk, nullptr);
            blk.toCc.init(*blk.Gblk, nullptr);

            blk.blkDegPlus.init(*blk.Gblk, 0);
            blk.blkDegMinus.init(*blk.Gblk, 0);

            size_t hEdgeCount = 0;
            for (spqr_compat::edge hE : cc.bc->hEdges(blk.bNode))
            {
                (void)hE;
                ++hEdgeCount;
                if (hEdgeCount >= 1000000) break;
            }

            auto &C = ctx();
            const bool useFlatEdges =
                C.spCompressMode == Context::SpCompressMode::MacroDirectDebug &&
                hEdgeCount >= 1000000 &&
                std::getenv("BF_DISABLE_FLAT_BLOCK_EDGES") == nullptr;
            const bool useFlatNodes =
                useFlatEdges &&
                std::getenv("BF_DISABLE_FLAT_BLOCK_NODES") == nullptr;
            blk.flatEdges = useFlatEdges;
            blk.flatNodes = useFlatNodes;
            if (useFlatEdges)
            {
                blk.flatEdgeToOrig.reserve(hEdgeCount);
                blk.flatFfiEdges.reserve(hEdgeCount);
            }
            auto addBlockEdge = [&](spqr_compat::edge eCc, spqr_compat::node uB, spqr_compat::node vB) {
                spqr_compat::edge eOrig = cc.edgeToOrig[eCc];
                if (blk.flatEdges)
                {
                    uint32_t fid = blk.flatEdgeCount++;
                    blk.flatEdgeToOrig.push_back(eOrig);
                    blk.flatFfiEdges.push_back(SpCompressInputEdge{
                        static_cast<uint32_t>(uB.idx),
                        static_cast<uint32_t>(vB.idx),
                        fid});
                }
                else
                {
                    spqr_compat::edge eB = blk.Gblk->newEdge(uB, vB);
                    blk.edgeToOrig[eB] = eOrig;
                }

                const auto edgeTypes = C._edge2types(eOrig);
                EdgePartType tU = edgeTypes.first;
                EdgePartType tV = edgeTypes.second;

                if (tU == EdgePartType::PLUS)
                {
                    if (blk.flatEdges) blk.flatDegPlus[uB.idx]++;
                    else blk.blkDegPlus[uB]++;
                }
                else
                {
                    if (blk.flatEdges) blk.flatDegMinus[uB.idx]++;
                    else blk.blkDegMinus[uB]++;
                }

                if (tV == EdgePartType::PLUS)
                {
                    if (blk.flatEdges) blk.flatDegPlus[vB.idx]++;
                    else blk.blkDegPlus[vB]++;
                }
                else
                {
                    if (blk.flatEdges) blk.flatDegMinus[vB.idx]++;
                    else blk.blkDegMinus[vB]++;
                }
            };

            if (hEdgeCount >= 1000000)
            {
                spqr_compat::NodeArray<spqr_compat::node> cc_to_blk(*cc.Gcc, nullptr);
                auto getBlockNode = [&](spqr_compat::node vCc) -> spqr_compat::node {
                    spqr_compat::node vB = cc_to_blk[vCc];
                    if (!vB)
                    {
                        if (blk.flatNodes)
                        {
                            uint32_t idx = static_cast<uint32_t>(blk.flatNodeToOrig.size());
                            vB = spqr_compat::node{idx};
                            blk.flatNodeToCc.push_back(vCc);
                            blk.flatNodeToOrig.push_back(cc.nodeToOrig[vCc]);
                        }
                        else
                        {
                            vB = blk.Gblk->newNode();
                            blk.toCc[vB] = vCc;
                            blk.nodeToOrig[vB] = cc.nodeToOrig[vCc];
                        }
                        cc_to_blk[vCc] = vB;
                        if (blk.flatEdges)
                        {
                            blk.flatDegPlus.resize(static_cast<size_t>(vB.idx) + 1, 0);
                            blk.flatDegMinus.resize(static_cast<size_t>(vB.idx) + 1, 0);
                        }
                    }
                    return vB;
                };

                for (spqr_compat::edge hE : cc.bc->hEdges(blk.bNode))
                {
                    spqr_compat::edge eCc = cc.bc->original(hE);
                    addBlockEdge(eCc,
                                 getBlockNode(cc.Gcc->source(eCc)),
                                 getBlockNode(cc.Gcc->target(eCc)));
                }
            }
            else
            {
            struct EdgeRec { spqr_compat::edge eCc; spqr_compat::node uC; spqr_compat::node vC; };
            std::vector<EdgeRec> edges_vec;
            for (spqr_compat::edge hE : cc.bc->hEdges(blk.bNode))
            {
                spqr_compat::edge eCc = cc.bc->original(hE);
                edges_vec.push_back({eCc, cc.Gcc->source(eCc), cc.Gcc->target(eCc)});
            }

            std::vector<spqr_compat::node> verts_vec;
            verts_vec.reserve(2 * edges_vec.size());
            for (const auto &er : edges_vec)
            {
                verts_vec.push_back(er.uC);
                verts_vec.push_back(er.vC);
            }
            std::sort(verts_vec.begin(), verts_vec.end(),
                      [](spqr_compat::node a, spqr_compat::node b)
                      { return a.index() < b.index(); });
            verts_vec.erase(std::unique(verts_vec.begin(), verts_vec.end()), verts_vec.end());

            std::unordered_map<spqr_compat::node, spqr_compat::node> cc_to_blk;
            cc_to_blk.reserve(verts_vec.size());

            for (spqr_compat::node vCc : verts_vec)
            {
                spqr_compat::node vB;
                if (blk.flatNodes)
                {
                    uint32_t idx = static_cast<uint32_t>(blk.flatNodeToOrig.size());
                    vB = spqr_compat::node{idx};
                    blk.flatNodeToCc.push_back(vCc);
                    blk.flatNodeToOrig.push_back(cc.nodeToOrig[vCc]);
                }
                else
                {
                    vB = blk.Gblk->newNode();
                    blk.toCc[vB] = vCc;
                    blk.nodeToOrig[vB] = cc.nodeToOrig[vCc];
                }
                cc_to_blk[vCc] = vB;
                if (blk.flatEdges)
                {
                    blk.flatDegPlus.resize(static_cast<size_t>(vB.idx) + 1, 0);
                    blk.flatDegMinus.resize(static_cast<size_t>(vB.idx) + 1, 0);
                }
            }

            for (const auto &er : edges_vec)
            {
                auto srcIt = cc_to_blk.find(er.uC);
                auto tgtIt = cc_to_blk.find(er.vC);
                if (srcIt != cc_to_blk.end() && tgtIt != cc_to_blk.end())
                {
                    addBlockEdge(er.eCc, srcIt->second, tgtIt->second);
                }
            }
            }

            VLOG << "[DEBUG][buildBlockData]  |V(Gblk)|=" << blockNodeCount(blk)
                 << " |E(Gblk)|=" << blockEdgeCount(blk) << "\n";

            if (blockNodeCount(blk) >= 3)
            {
                if (!buildBlockSpqr(blk))
                {
                    return;
                }

                SPQR_RUST_ASSERT(blk.spqr != nullptr);

                const auto &T = blk.spqr->tree();

                blk.skel2tree.clear();
                blk.skel2tree.reserve(2 * T.numberOfEdges());
                for (spqr_compat::edge te : T.edges)
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
                spqr_compat::node root = blk.spqr->rootNode();
                blk.parent[root] = root;

                std::stack<spqr_compat::node> st;
                st.push(root);

                while (!st.empty())
                {
                    spqr_compat::node u = st.top();
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
            spqr_compat::node bNode;
            size_t ccIndex = 0;

            std::unique_ptr<BlockData> blk;

            uint64_t treeWeight = 0;
            uint64_t edgeWeight = 0;
            uint64_t logicWeight = 0;
            bool critical = false;

            BlockPrep(CcData *cc_, spqr_compat::node b, size_t ccIndex_) : cc(cc_), bNode(b), ccIndex(ccIndex_), blk(nullptr) {}

            BlockPrep() = default;
            BlockPrep(const BlockPrep &) = delete;
            BlockPrep &operator=(const BlockPrep &) = delete;
            BlockPrep(BlockPrep &&) = default;
            BlockPrep &operator=(BlockPrep &&) = default;

            spqr_compat::NodeArray<int> blkDegPlus, blkDegMinus;
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

                        auto& G = ctx().G;
                        auto addCcEdge = [&](edge e, node uC, node vC) {
                            auto eC = (*components)[cid]->Gcc->newEdge(uC, vC);
                            (*components)[cid]->edgeToOrig[eC] = e;

                            const auto edgeTypes = ctx()._edge2types(e);
                            if (G.source(e) == G.target(e)) {
                                addIncidentTypeCount((*components)[cid]->degPlus,
                                                     (*components)[cid]->degMinus,
                                                     vC, edgeTypes.first);
                                addIncidentTypeCount((*components)[cid]->degPlus,
                                                     (*components)[cid]->degMinus,
                                                     vC, edgeTypes.second);
                            } else {
                                addIncidentTypeCount((*components)[cid]->degPlus,
                                                     (*components)[cid]->degMinus,
                                                     uC,
                                                     edgeTypes.first);
                                addIncidentTypeCount((*components)[cid]->degPlus,
                                                     (*components)[cid]->degMinus,
                                                     vC,
                                                     edgeTypes.second);
                            }
                        };

                        if ((*bucket)[cid].size() >= 1000000)
                        {
                            spqr_compat::NodeArray<node> orig_to_cc(G, nullptr);

                            for (node vG : (*bucket)[cid])
                            {
                                node vC = (*components)[cid]->Gcc->newNode();
                                (*components)[cid]->nodeToOrig[vC] = vG;
                                orig_to_cc[vG] = vC;
                            }

                            for (edge e : (*edgeBuckets)[cid])
                            {
                                addCcEdge(e, orig_to_cc[G.source(e)], orig_to_cc[G.target(e)]);
                            }
                        }
                        else
                        {
                            std::unordered_map<node, node> orig_to_cc;
                            orig_to_cc.reserve((*bucket)[cid].size());

                            for (node vG : (*bucket)[cid])
                            {
                                node vC = (*components)[cid]->Gcc->newNode();
                                (*components)[cid]->nodeToOrig[vC] = vG;
                                orig_to_cc[vG] = vC;
                            }

                            for (edge e : (*edgeBuckets)[cid])
                            {
                                addCcEdge(e, orig_to_cc[G.source(e)], orig_to_cc[G.target(e)]);
                            }
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

                        SPQR_RUST_ASSERT(cc->Gcc->numberOfNodes() > 0);

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

                        for (spqr_compat::node v : cc->bc->bcTree().nodes)
                        {
                            if (cc->bc->typeOfBNode(v) == BCTree::BNodeType::BComp)
                            {
                                VLOG << "  [DEBUG][worker_bcTree]  B-node "
                                     << v.index() << " (block)\n";
                                localPreps.emplace_back(cc, v, cid);
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

                    BF_INSTR(auto __build_t0 = std::chrono::high_resolution_clock::now();)
                    {
                        buildBlockData(blk, *(*blockPreps)[bid].cc);
                    }
                    BF_INSTR(
                    auto __build_t1 = std::chrono::high_resolution_clock::now();
                    uint64_t __dt_build = std::chrono::duration_cast<std::chrono::nanoseconds>(__build_t1 - __build_t0).count();
                    profiling_patch::BlockTiming *bt_b = profiling_patch::try_get_block_timing(bid);
                    if (bt_b) {
                        bt_b->bid = bid;
                        bt_b->build_ns = __dt_build;
                    }
                    )

                    BlockPrep &prepRef = (*blockPreps)[bid];
                    if (blk.spqr) {
                        prepRef.treeWeight = blk.spqr->tree().numberOfNodes();
                    } else if (blk.spCompressHandle) {
                        prepRef.treeWeight =
                            static_cast<uint64_t>(blk.macroTreeView.macros_len) +
                            static_cast<uint64_t>(blk.macroTreeView.core_edges_len);
                    } else {
                        prepRef.treeWeight = 0;
                    }
                    prepRef.edgeWeight = (blk.Gblk ? blockEdgeCount(blk) : 0);
                    prepRef.logicWeight = prepRef.treeWeight + prepRef.edgeWeight;

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

        #ifdef BUBBLEFINDER_INSTRUMENT
        static void printSnarlsProfiling()
        {
            namespace pp = profiling_patch;

            constexpr double NS_TO_MS = 1e-6;
            const auto pct = [](double v, double total) {
                return total > 0.0 ? (100.0 * v / total) : 0.0;
            };

            const uint64_t n_blocks = pp::blocks_with_spqr.load();
            const uint64_t n_tnodes = pp::total_tree_nodes.load();
            const uint64_t n_pe = pp::pe_total_calls.load();
            const uint64_t n_pn = pp::pn_total_calls.load();

            const double t_p1 = pp::phase1_edge_dp_time_ns.load() * NS_TO_MS;
            const double t_p2 = pp::phase2_node_dp_time_ns.load() * NS_TO_MS;
            const double t_p3 = pp::phase3_solve_time_ns.load() * NS_TO_MS;
            const double t_p4 = pp::phase4_caseE_time_ns.load() * NS_TO_MS;
            const double t_phases = t_p1 + t_p2 + t_p3 + t_p4;

            auto &os = std::cout;
            auto old_flags = os.flags();
            auto old_precision = os.precision();

            auto p3 = [&]() -> std::ostream& { os << std::fixed << std::setprecision(3); return os; };
            auto p1 = [&]() -> std::ostream& { os << std::fixed << std::setprecision(1); return os; };

            p3() << "\nSnarls SPQR solver profile\n"
                 << n_blocks << " blocks, " << n_tnodes << " tree nodes (avg ";
            p1() << (n_blocks ? (double)n_tnodes / (double)n_blocks : 0.0) << "/block).\n";

            p3() << "\nPhases:\n"
                 << "  1  edge DP   (processEdge)            " << t_p1 << " ms (";
            p1() << pct(t_p1, t_phases) << "%) / " << n_pe << " calls\n";
            p3() << "  2  node DP   (processNode)            " << t_p2 << " ms (";
            p1() << pct(t_p2, t_phases) << "%) / " << n_pn << " calls\n";
            p3() << "  3  solve     (solveS/solveP/solveRR)  " << t_p3 << " ms (";
            p1() << pct(t_p3, t_phases) << "%)\n";
            p3() << "  4  case E    (single-edge snarls)     " << t_p4 << " ms (";
            p1() << pct(t_p4, t_phases) << "%)\n";

            const double t_alloc = pp::sub_alloc_dp_ns.load() * NS_TO_MS;
            const double t_dfs = pp::sub_dfs_order_ns.load() * NS_TO_MS;
            const double t_btsi = pp::sub_blktoskel_init_ns.load() * NS_TO_MS;
            const double t_lvl = pp::sub_levels_ns.load() * NS_TO_MS;
            const double t_p4_setup = pp::sub_phase4_setup_ns.load() * NS_TO_MS;
            const double t_p4_collect = pp::sub_caseE_collect_ns.load() * NS_TO_MS;
            const double t_p4_body = pp::sub_phase4_caseE_body_ns.load() * NS_TO_MS;
            const double t_p3_collect = pp::sub_solveS_collect_ns.load() * NS_TO_MS;
            const double t_destruct = pp::sub_destruct_ns.load() * NS_TO_MS;

            os << "\nSub-phases (cumulative across all blocks):\n";
            p3() << "  alloc DP arrays                     " << t_alloc << " ms (";
            p1() << pct(t_alloc, t_phases) << "% of phases)\n";
            p3() << "  dfsSPQR_order                       " << t_dfs << " ms (";
            p1() << pct(t_dfs, t_phases) << "% of phases)\n";
            p3() << "  blkToSkel.init                      " << t_btsi << " ms (";
            p1() << pct(t_btsi, t_phases) << "% of phases)\n";
            p3() << "  levels analysis   " << t_lvl << " ms (";
            p1() << pct(t_lvl, t_phases) << "% of phases)\n";
            p3() << "  phase 3 sNodes/pNodes collect       " << t_p3_collect << " ms\n";
            p3() << "  phase 4 vertexInSnodes setup        " << t_p4_setup << " ms (";
            p1() << pct(t_p4_setup, t_p4) << "% of phase 4)\n";
            p3() << "  phase 4 caseE edges collect         " << t_p4_collect << " ms\n";
            p3() << "  phase 4 caseE body                  " << t_p4_body << " ms (";
            p1() << pct(t_p4_body, t_p4) << "% of phase 4)\n";
            p3() << "  BlockData destruction               " << t_destruct << " ms\n";

            auto row = [&](const char *label, double t, double sum) {
                os << "  " << label;
                p3() << t << " ms (";
                p1() << pct(t, sum) << "%)\n";
            };

            if (n_pe > 0) {
                const double tA = pp::pe_A_setup_ns.load() * NS_TO_MS;
                const double tB = pp::pe_B_build_ns.load() * NS_TO_MS;
                const double sum = tA + tB;
                os << "\nprocessEdge: " << n_pe << " calls, total ";
                p3() << sum << " ms, ";
                p1() << ((sum * 1000.0) / (double)n_pe) << " us avg\n";
                row("setup            ", tA, sum);
                row("build + child DP ", tB, sum);
            }

            if (n_pn > 0) {
                const double tA = pp::pn_A_setup_ns.load() * NS_TO_MS;
                const double tB = pp::pn_B_build_ns.load() * NS_TO_MS;
                const double tC = pp::pn_C_propagate_ns.load() * NS_TO_MS;
                const double sum = tA + tB + tC;
                os << "\nprocessNode: " << n_pn << " calls, total ";
                p3() << sum << " ms, ";
                p1() << ((sum * 1000.0) / (double)n_pn) << " us avg\n";
                row("setup            ", tA, sum);
                row("build            ", tB, sum);
                row("propagate degrees", tC, sum);
            }

            {
                const uint64_t s_calls = pp::p3_solveS_calls.load();
                const uint64_t p_calls = pp::p3_solveP_calls.load();
                const uint64_t rr_calls = pp::p3_solveRR_calls.load();
                const double s_ms = pp::p3_solveS_ns.load() * NS_TO_MS;
                const double p_ms = pp::p3_solveP_ns.load() * NS_TO_MS;
                const double rr_ms = pp::p3_solveRR_ns.load() * NS_TO_MS;
                const double sum = s_ms + p_ms + rr_ms;
                if (sum > 0.0) {
                    os << "\nPhase 3 breakdown: total ";
                    p3() << sum << " ms\n";
                    os << "  solveS  " << s_calls  << " calls (";
                    p3() << s_ms  << " ms, "; p1() << pct(s_ms,  sum) << "%)\n";
                    os << "  solveP  " << p_calls  << " calls (";
                    p3() << p_ms  << " ms, "; p1() << pct(p_ms,  sum) << "%)\n";
                    os << "  solveRR " << rr_calls << " calls (";
                    p3() << rr_ms << " ms, "; p1() << pct(rr_ms, sum) << "%)\n";
                }
            }

            {
                const uint64_t tlS_c = pp::taskloops_S_created.load();
                const uint64_t tlS_s = pp::taskloops_S_skipped.load();
                const uint64_t tlP_c = pp::taskloops_P_created.load();
                const uint64_t tlP_s = pp::taskloops_P_skipped.load();
                const uint64_t tlE_c = pp::taskloops_caseE_created.load();
                const uint64_t tlE_s = pp::taskloops_caseE_skipped.load();
                const uint64_t taskS = pp::tasks_requested_S_total.load();
                const uint64_t taskP = pp::tasks_requested_P_total.load();
                const uint64_t taskE = pp::tasks_requested_caseE_total.load();
                os << "\nTaskloops created (= entered parallel path):\n";
                os << "  S-nodes :   created " << tlS_c << ", skipped " << tlS_s
                   << ", tasks requested total " << taskS;
                if (tlS_c > 0) os << " (avg " << (taskS / tlS_c) << "/taskloop)";
                os << "\n";
                os << "  P-nodes :   created " << tlP_c << ", skipped " << tlP_s
                   << ", tasks requested total " << taskP;
                if (tlP_c > 0) os << " (avg " << (taskP / tlP_c) << "/taskloop)";
                os << "\n";
                os << "  caseE   :   created " << tlE_c << ", skipped " << tlE_s
                   << ", tasks requested total " << taskE;
                if (tlE_c > 0) os << " (avg " << (taskE / tlE_c) << "/taskloop)";
                os << "\n";
            }

            {
                auto &tv = pp::g_block_timings();
                if (!tv.empty()) {
                    std::vector<size_t> idx;
                    idx.reserve(tv.size());
                    for (size_t i = 0; i < tv.size(); ++i) {
                        if (tv[i].solve_total_ns > 0 || tv[i].build_ns > 0)
                            idx.push_back(i);
                    }
                    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
                        uint64_t ka = tv[a].build_ns + tv[a].solve_total_ns;
                        uint64_t kb = tv[b].build_ns + tv[b].solve_total_ns;
                        return ka > kb;
                    });
                    const size_t topN = std::min<size_t>(idx.size(), 10);
                    if (topN > 0) {
                        os << "\nTop " << topN << " blocks (by build+solve time):\n";
                        for (size_t k = 0; k < topN; ++k) {
                            const auto &b = tv[idx[k]];
                            const double ms_build = b.build_ns * NS_TO_MS;
                            const double ms_solve = b.solve_total_ns * NS_TO_MS;
                            const double ms_alloc = b.sub_alloc_dp_ns * NS_TO_MS;
                            const double ms_dfs = b.sub_dfs_order_ns* NS_TO_MS;
                            const double ms_btsi = b.sub_blktoskel_init_ns* NS_TO_MS;
                            const double ms_lvl = b.sub_levels_ns * NS_TO_MS;
                            const double ms_p1 = b.phase1_ns * NS_TO_MS;
                            const double ms_p2 = b.phase2_ns * NS_TO_MS;
                            const double ms_p3 = b.phase3_ns * NS_TO_MS;
                            const double ms_p3_S = b.phase3_S_ns * NS_TO_MS;
                            const double ms_p3_P = b.phase3_P_ns * NS_TO_MS;
                            const double ms_p3_RR = b.phase3_RR_ns * NS_TO_MS;
                            const double ms_p3_col = b.phase3_S_collect_ns * NS_TO_MS;
                            const double ms_p4_set = b.phase4_setup_ns * NS_TO_MS;
                            const double ms_p4_col = b.phase4_caseE_collect_ns * NS_TO_MS;
                            const double ms_p4_body= b.phase4_caseE_body_ns    * NS_TO_MS;
                            const double ms_destr = b.destruct_ns     * NS_TO_MS;
                            const double sub_sum = ms_alloc + ms_dfs + ms_btsi + ms_lvl
                                                    + ms_p1 + ms_p2 + ms_p3
                                                    + ms_p4_set + ms_p4_col + ms_p4_body;
                            const double unaccounted = ms_solve - sub_sum;

                            os << "\n  block #" << b.bid
                               << " (rank " << (k + 1) << ")"
                               << "  critical=" << (b.critical ? "Y" : "N")
                               << "  block(n+e)=" << b.blockNodes << "+" << b.blockEdges
                               << "  spqr=" << b.spqrTreeNodes
                               << " [S=" << b.spqrSCount << " P=" << b.spqrPCount << " R=" << b.spqrRCount << "]"
                               << "  height=" << b.maxHeight
                               << "  depth=" << b.maxDepth << "\n";
                            p3() << "    build = " << ms_build << " ms\n";
                            p3() << "    solve = " << ms_solve << " ms\n";
                            p3() << "      alloc DP        " << ms_alloc << " ms\n";
                            p3() << "      dfsSPQR_order   " << ms_dfs   << " ms\n";
                            p3() << "      blkToSkel.init  " << ms_btsi  << " ms\n";
                            p3() << "      levels analysis " << ms_lvl   << " ms (instrumentation overhead)\n";
                            p3() << "      phase 1         " << ms_p1    << " ms\n";
                            p3() << "      phase 2         " << ms_p2    << " ms\n";
                            p3() << "      phase 3 total   " << ms_p3    << " ms\n";
                            p3() << "        sNodes/pNodes collect " << ms_p3_col << " ms\n";
                            p3() << "        solveS                 " << ms_p3_S << " ms\n";
                            p3() << "        solveP                 " << ms_p3_P << " ms\n";
                            p3() << "        solveRR                " << ms_p3_RR << " ms\n";
                            p3() << "      phase 4 setup           " << ms_p4_set  << " ms\n";
                            p3() << "      phase 4 caseE collect   " << ms_p4_col  << " ms\n";
                            p3() << "      phase 4 caseE body      " << ms_p4_body << " ms\n";
                            p3() << "      unaccounted in solve    " << unaccounted << " ms (";
                            p1() << pct(unaccounted, ms_solve) << "%)\n";
                            p3() << "    destruct   = " << ms_destr << " ms\n";
                            os << "    taskloops S/P/caseE created = "
                               << b.taskloops_S_created << "/"
                               << b.taskloops_P_created << "/"
                               << b.taskloops_caseE_created
                               << "  tasks requested = "
                               << b.tasks_requested_S << "/"
                               << b.tasks_requested_P << "/"
                               << b.tasks_requested_caseE << "\n";
                        }
                    }

                    if (!idx.empty()) {
                        const auto &b0 = tv[idx[0]];
                        if (!b0.widthByDepth.empty()) {
                            os << "\nDominant block #" << b0.bid << " - SPQR width by depth (top-down for phase 2 wavefront):\n";
                            os << "  maxDepth = " << b0.maxDepth
                               << ", total nodes = " << b0.spqrTreeNodes << "\n";
                            for (size_t d = 0; d < b0.widthByDepth.size(); ++d) {
                                if (b0.widthByDepth[d] == 0) continue;
                                os << "  depth " << std::setw(4) << d << " : "
                                   << b0.widthByDepth[d] << "\n";
                            }
                        }
                        if (!b0.widthByHeight.empty()) {
                            os << "\nDominant block #" << b0.bid << " - SPQR width by height (bottom-up for phase 1 wavefront):\n";
                            os << "  maxHeight = " << b0.maxHeight
                               << ", total nodes = " << b0.spqrTreeNodes << "\n";
                            for (size_t h = 0; h < b0.widthByHeight.size(); ++h) {
                                if (b0.widthByHeight[h] == 0) continue;
                                os << "  height " << std::setw(4) << h << " : "
                                   << b0.widthByHeight[h] << "\n";
                            }
                        }
                    }
                }
            }

            os << std::endl;

            os.precision(old_precision);
            os.flags(old_flags);
        }
        #endif


        void solve()
        {
            std::cout << "Finding snarls...\n";
            PROFILE_FUNCTION();
            auto &C = ctx();

            ::spqr_set_canonicalize_root_enabled(C.skipCanonicalizeRoot ? 0 : 1);

            C.fastSnarlPairsEnabled =
                (C.spCompressMode == Context::SpCompressMode::MacroDirectDebug &&
                 std::getenv("BF_DISABLE_FAST_SNARL_OUTPUT") == nullptr);
            if (std::getenv("BF_DEBUG_FAST_SNARL_OUTPUT")) {
                std::cerr << "[fast_snarl_solve] enabled="
                          << (C.fastSnarlPairsEnabled ? 1 : 0)
                          << " include_trivial=" << (C.includeTrivial ? 1 : 0)
                          << " sp_compress_mode=" << static_cast<int>(C.spCompressMode)
                          << " disable_env="
                          << (std::getenv("BF_DISABLE_FAST_SNARL_OUTPUT") ? 1 : 0)
                          << "\n";
            }
            C.fastSnarlPairs.clear();
            C.fastSnarlCliques.clear();
            if (C.spCompressMode == Context::SpCompressMode::MacroDirectDebug &&
                std::getenv("BF_SPQR_THREADS") == nullptr)
            {
                const std::string spqrThreads =
                    std::to_string(std::max<unsigned>(1u, C.threads));
                setenv("BF_SPQR_THREADS", spqrThreads.c_str(), /*overwrite=*/0);
            }
            const bool streamBlockBuildSolve =
                (std::getenv("BF_DISABLE_STREAMING_BLOCK_SOLVE") == nullptr &&
                 (C.threads <= 1 ||
                  C.spCompressMode == Context::SpCompressMode::MacroDirectDebug));

            Graph &G = C.G;
            int nCC = 0;
            std::vector<std::vector<node>> bucket;
            std::vector<std::vector<edge>> edgeBuckets;

            {
                PhaseSampler io_sampler(g_stats_io);
                NodeArray<int> compIdx(G);

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
            std::vector<std::atomic<size_t>> ccBlocksRemaining(nCC);
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

                if (C.threads > 1)
                {
                    decltype(bucket)().swap(bucket);
                    decltype(edgeBuckets)().swap(edgeBuckets);
                }

                {
                    for (auto &cc_ptr : components) {
                        if (!cc_ptr || !cc_ptr->Gcc) continue;
                        const size_t n = cc_ptr->Gcc->numberOfNodes();
                        if (n == 0) continue;
                        const spqr_compat::node last{static_cast<uint32_t>(n - 1)};
                        (void)cc_ptr->isTip[last];
                        (void)cc_ptr->isCutNode[last];
                        (void)cc_ptr->isGoodCutNode[last];
                        (void)cc_ptr->lastBad[last];
                        (void)cc_ptr->badCutCount[last];
                        (void)cc_ptr->nodeToOrig[last];
                        (void)cc_ptr->degPlus[last];
                        (void)cc_ptr->degMinus[last];
                    }

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
                    for (const auto &prep : blockPreps)
                    {
                        ccBlocksRemaining[prep.ccIndex].fetch_add(1, std::memory_order_relaxed);
                    }
                }

                if (!streamBlockBuildSolve)
                {
                    MARK_SCOPE_MEM("sn/phase/block_SPQR_build");

                    BF_INSTR(profiling_patch::reset_block_timings(blockPreps.size());)

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

                auto releaseBlockComponent = [&](BlockPrep &prep) {
                    if (C.threads <= 1) return;
                    if (prep.ccIndex < ccBlocksRemaining.size() &&
                        ccBlocksRemaining[prep.ccIndex].fetch_sub(1, std::memory_order_acq_rel) == 1)
                    {
                        auto &component = components[prep.ccIndex];
                        if (component && component->Gcc &&
                            component->Gcc->numberOfNodes() <= 10000000)
                        {
                            component.reset();
                        }
                    }
                };

                {
                    MARK_SCOPE_MEM("sn/phase/block_SPQR_solve");

                    size_t numThreads = std::thread::hardware_concurrency();
                    numThreads = std::min({(size_t)C.threads, blockPreps.size(), numThreads});
                    if (numThreads == 0) numThreads = 1;

                    const int P = static_cast<int>(numThreads);

                    if (streamBlockBuildSolve)
                    {
                        BF_INSTR(profiling_patch::reset_block_timings(blockPreps.size());)

                        const bool bufferingStringSnarls =
                            !C.fastSnarlPairsEnabled || debug_tagged_snarls_enabled();
                        const uint64_t Q = 1;
                        std::atomic<size_t> nextStreamBlock{0};
                        std::atomic<int> activeIntraTaskloops{0};

                        auto runStreamWorker = [&]() {
                            std::vector<std::vector<std::string>> localSnarls;
                            std::vector<uint64_t> localFastSnarlPairs;
                            std::vector<std::vector<uint64_t>> localFastSnarlCliques;
                            tls_snarl_buffer = &localSnarls;
                            tls_fast_snarl_pair_buffer =
                                C.fastSnarlPairsEnabled ? &localFastSnarlPairs : nullptr;
                            tls_fast_snarl_clique_buffer =
                                C.fastSnarlPairsEnabled ? &localFastSnarlCliques : nullptr;
                            tls_spqr_seen_endpoint_pairs.clear();

                            auto flushLocalStringsIfNeeded = [&]() {
                                if (localSnarls.size() >= (1u << 12)) {
                                    flushThreadLocalSnarls(localSnarls);
                                }
                            };

                            const int planThreads = (omp_get_num_threads() > 1) ? 1 : P;
                            size_t chunkSize = 1;

                            while (true)
                            {
                                const size_t startIndex =
                                    nextStreamBlock.fetch_add(chunkSize, std::memory_order_relaxed);
                                if (startIndex >= blockPreps.size())
                                    break;
                                const size_t endIndex =
                                    std::min(startIndex + chunkSize, blockPreps.size());

                                auto chunkStart = std::chrono::high_resolution_clock::now();

                                for (size_t bid = startIndex; bid < endIndex; ++bid)
                                {
                                    BlockPrep &prep = blockPreps[bid];
                                    prep.blk = std::make_unique<BlockData>();
                                    BlockData &blk = *prep.blk;
                                    blk.bNode = prep.bNode;

                                    BF_INSTR(auto __build_t0 = std::chrono::high_resolution_clock::now();)
                                    buildBlockData(blk, *prep.cc);
                                    BF_INSTR(
                                    auto __build_t1 = std::chrono::high_resolution_clock::now();
                                    uint64_t __dt_build = std::chrono::duration_cast<std::chrono::nanoseconds>(__build_t1 - __build_t0).count();
                                    profiling_patch::BlockTiming *bt_b = profiling_patch::try_get_block_timing(bid);
                                    if (bt_b) {
                                        bt_b->bid = bid;
                                        bt_b->build_ns = __dt_build;
                                    }
                                    )

                                    if (blk.spqr) {
                                        prep.treeWeight = blk.spqr->tree().numberOfNodes();
                                    } else if (blk.spCompressHandle) {
                                        prep.treeWeight =
                                            static_cast<uint64_t>(blk.macroTreeView.macros_len) +
                                            static_cast<uint64_t>(blk.macroTreeView.core_edges_len);
                                    } else {
                                        prep.treeWeight = 0;
                                    }
                                    prep.edgeWeight = (blk.Gblk ? blockEdgeCount(blk) : 0);
                                    prep.logicWeight = prep.treeWeight + prep.edgeWeight;

                                    if (blk.Gblk && blockNodeCount(blk) >= 3)
                                    {
                                        const int blockPlanThreads =
                                            (blk.spCompressHandle &&
                                             blk.macroTreeView.macros_len >= 1000000u)
                                                ? P
                                                : planThreads;
                                        IntraPlan plan;
                                        plan.critical = blockPlanThreads > 1;
                                        plan.quantum = Q;
                                        plan.numThreads = blockPlanThreads;
                                        plan.activeIntraTaskloops = &activeIntraTaskloops;
                                        plan.bid = bid;

                                        BF_INSTR(auto __solve_t0 = std::chrono::high_resolution_clock::now();)
                                        SPQRsolve::solveSPQR(blk, *prep.cc, plan);
                                        BF_INSTR(
                                        auto __solve_t1 = std::chrono::high_resolution_clock::now();
                                        uint64_t __dt_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(__solve_t1 - __solve_t0).count();
                                        profiling_patch::BlockTiming *bt_disp = profiling_patch::try_get_block_timing(bid);
                                        if (bt_disp) bt_disp->solve_total_ns = __dt_solve;
                                        )
                                    }

                                    BF_INSTR(auto __destr_t0 = std::chrono::high_resolution_clock::now();)
                                    prep.blk.reset();
                                    releaseBlockComponent(prep);
                                    BF_INSTR(
                                    auto __destr_t1 = std::chrono::high_resolution_clock::now();
                                    uint64_t __dt_destr = std::chrono::duration_cast<std::chrono::nanoseconds>(__destr_t1 - __destr_t0).count();
                                    profiling_patch::sub_destruct_ns.fetch_add(__dt_destr, std::memory_order_relaxed);
                                    profiling_patch::BlockTiming *bt_d = profiling_patch::try_get_block_timing(bid);
                                    if (bt_d) bt_d->destruct_ns = __dt_destr;
                                    )

                                    if (bufferingStringSnarls ||
                                        (bid & ((1u << 13) - 1)) == ((1u << 13) - 1)) {
                                        flushLocalStringsIfNeeded();
                                    }
                                }

                                auto chunkEnd = std::chrono::high_resolution_clock::now();
                                auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(
                                                         chunkEnd - chunkStart);
                                if (chunkDuration.count() < 1000) {
                                    chunkSize = std::min(chunkSize * 2, std::max<size_t>(1, blockPreps.size() / P));
                                } else if (chunkDuration.count() > 5000) {
                                    chunkSize = std::max<size_t>(1, chunkSize / 2);
                                }
                            }

                            tls_snarl_buffer = nullptr;
                            tls_fast_snarl_pair_buffer = nullptr;
                            tls_fast_snarl_clique_buffer = nullptr;
                            flushThreadLocalFastSnarlPairs(localFastSnarlPairs);
                            flushThreadLocalFastSnarlCliques(localFastSnarlCliques);
                            flushThreadLocalSnarls(localSnarls);
                        };

                        if (P <= 1)
                        {
                            runStreamWorker();
                        }
                        else
                        {
                            BF_OMP_PRAGMA(omp parallel num_threads(P))
                            {
                                runStreamWorker();
                            }
                        }
                    }
                    else
                    {
                    uint64_t W_total = 0;
                    for (const auto &prep : blockPreps) {
                        W_total += prep.logicWeight;
                    }

                    const uint64_t Q = std::max<uint64_t>(
                        bf_ceil_div(W_total, static_cast<uint64_t>(P) * static_cast<uint64_t>(P)),
                        1ULL);

                    std::vector<size_t> hotBlocks;
                    for (size_t bid = 0; bid < blockPreps.size(); ++bid) {
                        const uint64_t W = blockPreps[bid].logicWeight;
                        const __int128 lhs = static_cast<__int128>(W) * static_cast<__int128>(P);
                        const __int128 rhs = static_cast<__int128>(W_total);
                        const bool critical = (lhs > rhs);

                        blockPreps[bid].critical = critical;
                        if (critical) hotBlocks.push_back(bid);
                    }

                    std::sort(hotBlocks.begin(), hotBlocks.end(),
                              [&](size_t a, size_t b) {
                                  return blockPreps[a].logicWeight > blockPreps[b].logicWeight;
                              });

                    BF_INSTR(std::cout << "[snarls] inter/intra plan: "
                              << "W_total=" << W_total
                              << " P=" << P
                              << " Q=" << Q
                              << " hotBlocks=" << hotBlocks.size()
                              << " (max W=" << (hotBlocks.empty() ? 0 : blockPreps[hotBlocks[0]].logicWeight)
                              << ")\n";)

                    auto isHotBlock = [&](size_t bid) -> bool {
                        for (size_t h : hotBlocks)
                            if (h == bid) return true;
                        return false;
                    };

                    std::atomic<size_t> nextHot{0};
                    std::atomic<size_t> nextNormal{0};
                    std::atomic<int> activeIntraTaskloops{0};

                    const size_t nBlocks = blockPreps.size();

                    BF_OMP_PRAGMA(omp parallel num_threads(P))
                    {
                        std::vector<std::vector<std::string>> localSnarls;
                        std::vector<uint64_t> localFastSnarlPairs;
                        std::vector<std::vector<uint64_t>> localFastSnarlCliques;
                        tls_snarl_buffer = &localSnarls;
                        tls_fast_snarl_pair_buffer =
                            C.fastSnarlPairsEnabled ? &localFastSnarlPairs : nullptr;
                        tls_fast_snarl_clique_buffer =
                            C.fastSnarlPairsEnabled ? &localFastSnarlCliques : nullptr;
                        tls_spqr_seen_endpoint_pairs.clear();

                        size_t chunkSize = 1;
                        size_t processed = 0;
                        bool hotDone = hotBlocks.empty();

                        while (true) {
                            if (!hotDone) {
                                size_t h = nextHot.fetch_add(1, std::memory_order_relaxed);
                                if (h < hotBlocks.size()) {
                                    size_t bid = hotBlocks[h];
                                    BlockPrep &prep = blockPreps[bid];
                                    if (prep.blk) {
                                        BlockData &blk = *prep.blk;
                                        if (blk.Gblk && blockNodeCount(blk) >= 3) {
                                            IntraPlan plan;
                                            plan.critical = true;
                                            plan.quantum = Q;
                                            plan.numThreads = P;
                                            plan.activeIntraTaskloops = &activeIntraTaskloops;
                                            plan.bid = bid;
                                            BF_INSTR(auto __solve_t0 = std::chrono::high_resolution_clock::now();)
                                            SPQRsolve::solveSPQR(blk, *prep.cc, plan);
                                            BF_INSTR(
                                            auto __solve_t1 = std::chrono::high_resolution_clock::now();
                                            uint64_t __dt_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(__solve_t1 - __solve_t0).count();
                                            profiling_patch::BlockTiming *bt_disp = profiling_patch::try_get_block_timing(bid);
                                            if (bt_disp) bt_disp->solve_total_ns = __dt_solve;
                                            )
                                        }
                                        BF_INSTR(auto __destr_t0 = std::chrono::high_resolution_clock::now();)
                                        prep.blk.reset();
                                        releaseBlockComponent(prep);
                                        BF_INSTR(
                                        auto __destr_t1 = std::chrono::high_resolution_clock::now();
                                        uint64_t __dt_destr = std::chrono::duration_cast<std::chrono::nanoseconds>(__destr_t1 - __destr_t0).count();
                                        profiling_patch::sub_destruct_ns.fetch_add(__dt_destr, std::memory_order_relaxed);
                                        profiling_patch::BlockTiming *bt_d = profiling_patch::try_get_block_timing(bid);
                                        if (bt_d) bt_d->destruct_ns = __dt_destr;
                                        )
                                        ++processed;
                                    }
                                    continue;
                                }
                                hotDone = true;
                            }

                            size_t startIndex = nextNormal.fetch_add(chunkSize, std::memory_order_relaxed);
                            if (startIndex >= nBlocks) break;
                            size_t endIndex = std::min(startIndex + chunkSize, nBlocks);

                            auto chunkStart = std::chrono::high_resolution_clock::now();

                            for (size_t bid = startIndex; bid < endIndex; ++bid) {
                                if (!hotBlocks.empty() && isHotBlock(bid))
                                    continue;
                                BlockPrep &prep = blockPreps[bid];
                                if (!prep.blk) continue;
                                BlockData &blk = *prep.blk;
                                if (blk.Gblk && blockNodeCount(blk) >= 3) {
                                    IntraPlan plan;
                                    plan.critical = false; 
                                    plan.quantum = Q;
                                    plan.numThreads = P;
                                    plan.activeIntraTaskloops = &activeIntraTaskloops;
                                    plan.bid = bid;
                                    BF_INSTR(auto __solve_t0 = std::chrono::high_resolution_clock::now();)
                                    SPQRsolve::solveSPQR(blk, *prep.cc, plan);
                                    BF_INSTR(
                                    auto __solve_t1 = std::chrono::high_resolution_clock::now();
                                    uint64_t __dt_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(__solve_t1 - __solve_t0).count();
                                    profiling_patch::BlockTiming *bt_disp = profiling_patch::try_get_block_timing(bid);
                                    if (bt_disp) bt_disp->solve_total_ns = __dt_solve;
                                    )
                                }
                                BF_INSTR(auto __destr_t0 = std::chrono::high_resolution_clock::now();)
                                prep.blk.reset();
                                releaseBlockComponent(prep);
                                BF_INSTR(
                                auto __destr_t1 = std::chrono::high_resolution_clock::now();
                                uint64_t __dt_destr = std::chrono::duration_cast<std::chrono::nanoseconds>(__destr_t1 - __destr_t0).count();
                                profiling_patch::sub_destruct_ns.fetch_add(__dt_destr, std::memory_order_relaxed);
                                profiling_patch::BlockTiming *bt_d = profiling_patch::try_get_block_timing(bid);
                                if (bt_d) bt_d->destruct_ns = __dt_destr;
                                )
                                ++processed;
                            }

                            auto chunkEnd = std::chrono::high_resolution_clock::now();
                            auto chunkDuration = std::chrono::duration_cast<std::chrono::microseconds>(
                                                     chunkEnd - chunkStart);
                            if (chunkDuration.count() < 1000) {
                                chunkSize = std::min(chunkSize * 2, std::max<size_t>(1, nBlocks / numThreads));
                            } else if (chunkDuration.count() > 5000) {
                                chunkSize = std::max<size_t>(1, chunkSize / 2);
                            }

                            if (activeIntraTaskloops.load(std::memory_order_relaxed) > 0) {
                                BF_OMP_PRAGMA(omp taskyield)
                            }
                        }

                        BF_OMP_PRAGMA(omp barrier)

                        tls_snarl_buffer = nullptr;
                        tls_fast_snarl_pair_buffer = nullptr;
                        tls_fast_snarl_clique_buffer = nullptr;
                        flushThreadLocalFastSnarlPairs(localFastSnarlPairs);
                        flushThreadLocalFastSnarlCliques(localFastSnarlCliques);
                        flushThreadLocalSnarls(localSnarls);
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
        
            BF_INSTR(printSnarlsProfiling();)
        }

        void output_spqr_tree_only()
        {
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
                for (spqr_compat::node v : C.G.nodes)
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

            spqr_compat::NodeArray<int> component(C.G, -1);
            int numCC = spqr_compat::connectedComponents(C.G, component);
            std::cerr << "[spqr-tree] Graph has " << numCC << " connected components.\n";

            // Group original nodes by component
            std::vector<std::vector<spqr_compat::node>> ccNodes(numCC);
            for (spqr_compat::node v : C.G.nodes)
            {
                ccNodes[component[v]].push_back(v);
            }
            
            // Track which edges were output already to avoid duplicating self-loops
            std::unordered_set<spqr_compat::edge> self_loops_output;

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
                for (spqr_compat::node v : ccNodes[ccIdx])
                {
                    out << " " << C.node2name[v];
                }
                out << "\n";

                spqr_compat::Graph ccGraph;
                spqr_compat::NodeArray<spqr_compat::node> ccToOrig(ccGraph);
                spqr_compat::NodeArray<spqr_compat::node> origToCc(C.G, nullptr);

                for (spqr_compat::node vOrig : ccNodes[ccIdx])
                {
                    spqr_compat::node vCc = ccGraph.newNode();
                    ccToOrig[vCc] = vOrig;
                    origToCc[vOrig] = vCc;
                }

                for (spqr_compat::node vOrig : ccNodes[ccIdx])
                {
                    self_loops_output.clear();
                    C.G.forEachAdj(vOrig, [&](node /*neighbor*/, edge e) {
                        if (C.G.source(e) != vOrig) // Only process from source side
                            return;

                        spqr_compat::node src = C.G.source(e);
                        spqr_compat::node tgt = C.G.target(e);
                        
                        // Self-loops cannot be deduplicated by the above condition about the source side.
                        // Hence, we deduplicate them via a set.
                        if (src == tgt) {
                            if (self_loops_output.count(e)) {
                                return;
                            } else {
                                self_loops_output.insert(e);
                            }
                        }

                        spqr_compat::node srcCc = origToCc[src];
                        spqr_compat::node tgtCc = origToCc[tgt];
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
                    for (spqr_compat::edge eCc : ccGraph.edges)
                    {
                        spqr_compat::node uOrig = ccToOrig[ccGraph.source(eCc)];
                        spqr_compat::node vOrig = ccToOrig[ccGraph.target(eCc)];
                        (void)localE;

                        std::string eName = makeId("E", edgeCtr++);
                        writeE(eName, compName, C.node2name[uOrig], C.node2name[vOrig]);
                    }

                    std::cerr << "[spqr-tree]   CC done (single-node component)\n";
                    continue;
                }

                std::cerr << "[spqr-tree]   computing BC-tree...\n";
                spqr_compat::BCTree bc(ccGraph);

                std::map<spqr_compat::node, std::string> bcNodeToBlockName;
                std::unordered_map<uint32_t, std::vector<spqr_compat::node>> cutVertexBlocks;
                cutVertexBlocks.reserve(bc.numberOfCComps());

                spqr_compat::NodeArray<int> markCc(ccGraph, 0);
                int stampCc = 1;
                std::vector<spqr_compat::node> tmpNodes;
                tmpNodes.reserve(1024);

                // We use this to track which self-loops have been output.
                // We want to output each self-loop exactly once, assigned to an arbitrary block or SPQR node.
                self_loops_output.clear();

                for (spqr_compat::node bNode : bc.bcTree().nodes)
                {
                    if (bc.typeOfBNode(bNode) != spqr_compat::BCTree::BNodeType::BComp)
                        continue;

                    std::string blockName = makeId("B", blockCtr++);
                    bcNodeToBlockName[bNode] = blockName;

                    out << "B " << blockName << " " << compName;

                    tmpNodes.clear();
                    ++stampCc;

                    for (spqr_compat::edge hEdge : bc.hEdges(bNode))
                    {
                        spqr_compat::edge eCc = bc.original(hEdge);
                        if (!eCc)
                            continue;

                        spqr_compat::node a = ccGraph.source(eCc);
                        spqr_compat::node b = ccGraph.target(eCc);

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

                    for (spqr_compat::node vCc : tmpNodes)
                    {
                        spqr_compat::node vOrig = ccToOrig[vCc];
                        out << " " << C.node2name[vOrig];

                        if (bc.typeOfGNode(vCc) == spqr_compat::BCTree::GNodeType::CutVertex)
                        {
                            cutVertexBlocks[vCc.idx].push_back(bNode);
                        }
                    }
                    out << "\n";
                }

                std::cerr << "[spqr-tree]   blocks: " << bcNodeToBlockName.size() << "\n";

                // Write C-lines (cut nodes)
                for (spqr_compat::node vCc : ccGraph.nodes)
                {
                    if (bc.typeOfGNode(vCc) == spqr_compat::BCTree::GNodeType::CutVertex)
                    {
                        spqr_compat::node vOrig = ccToOrig[vCc];
                        out << "C " << C.node2name[vOrig];

                        auto blocksIt = cutVertexBlocks.find(vCc.idx);
                        if (blocksIt != cutVertexBlocks.end())
                        {
                            for (spqr_compat::node bNode : blocksIt->second)
                            {
                                auto blockNameIt = bcNodeToBlockName.find(bNode);
                                assert(blockNameIt != bcNodeToBlockName.end());
                                out << " " << blockNameIt->second;
                            }
                        }
                        out << "\n";
                    }
                }

                const int totalBlocks = (int)bcNodeToBlockName.size();
                int processedBlocks = 0;

                for (spqr_compat::node bNode : bc.bcTree().nodes)
                {
                    if (bc.typeOfBNode(bNode) != spqr_compat::BCTree::BNodeType::BComp)
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
                    for (spqr_compat::edge hEdge : bc.hEdges(bNode))
                    {
                        ++edgeCountApprox;
                        spqr_compat::edge eCc = bc.original(hEdge);
                        if (!eCc)
                            continue;

                        spqr_compat::node a = ccGraph.source(eCc);
                        spqr_compat::node b = ccGraph.target(eCc);

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
                        for (spqr_compat::edge hEdge : bc.hEdges(bNode))
                        {
                            spqr_compat::edge eCc = bc.original(hEdge);
                            if (!eCc)
                                continue;

                            spqr_compat::node uOrig = ccToOrig[ccGraph.source(eCc)];
                            spqr_compat::node vOrig = ccToOrig[ccGraph.target(eCc)];

                            std::string eName = makeId("E", edgeCtr++);
                            writeE(eName, blockName, C.node2name[uOrig], C.node2name[vOrig]);
                        }

                        // The BC tree contains no self-loops, hence we need to output self-loops manually.
                        for (spqr_compat::node vCc : tmpNodes) {
                            spqr_compat::node vOrig = ccToOrig[vCc];
                            
                            C.G.forEachAdj(vOrig, [&](node /*neighbor*/, edge e) {
                                spqr_compat::node src = C.G.source(e);
                                spqr_compat::node tgt = C.G.target(e);
                                if (src != tgt) // Only process self loops
                                    return;

                                // Self-loops are yielded twice by the iterator and may occur in multiple blocks and SPQR nodes,
                                // hence we need to deduplicate them manually.
                                if (self_loops_output.count(e)) {
                                    return;
                                } else {
                                    self_loops_output.insert(e);
                                }

                                std::string eName = makeId("E", edgeCtr++);
                                writeE(eName, blockName, C.node2name[vOrig], C.node2name[vOrig]);
                            });
                        }

                        continue;
                    }

                    // Build block graph
                    spqr_compat::Graph blockGraph;
                    spqr_compat::NodeArray<spqr_compat::node> blockToCC(blockGraph);
                    spqr_compat::EdgeArray<spqr_compat::edge> blockEdgeToCC(blockGraph);

                    // Map cc node -> block node using an unordered_map sized to block
                    // (ccGraph is huge; we cannot use NodeArray per block)
                    std::unordered_map<spqr_compat::node, spqr_compat::node> ccToBlock;
                    ccToBlock.reserve(tmpNodes.size() * 2);

                    for (spqr_compat::node vCc : tmpNodes)
                    {
                        spqr_compat::node vB = blockGraph.newNode();
                        blockToCC[vB] = vCc;
                        ccToBlock.emplace(vCc, vB);
                    }

                    // Second pass: add edges
                    for (spqr_compat::edge hEdge : bc.hEdges(bNode))
                    {
                        spqr_compat::edge eCc = bc.original(hEdge);
                        if (!eCc)
                            continue;

                        auto itS = ccToBlock.find(ccGraph.source(eCc));
                        auto itT = ccToBlock.find(ccGraph.target(eCc));
                        if (itS == ccToBlock.end() || itT == ccToBlock.end())
                            continue;

                        spqr_compat::edge eB = blockGraph.newEdge(itS->second, itT->second);
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

                        spqr_compat::StaticSPQRTree spqr(blockGraph);

                        std::map<spqr_compat::node, std::string> spqrNodeNames;

                        spqr_compat::NodeArray<int> markBlk(blockGraph, 0);
                        int stampBlk = 1;

                        // Write S/P/R-lines
                        for (spqr_compat::node treeNode : spqr.tree().nodes)
                        {
                            char typeChar;
                            switch (spqr.typeOf(treeNode))
                            {
                            case spqr_compat::SPQRTree::NodeType::SNode:
                                typeChar = 'S';
                                break;
                            case spqr_compat::SPQRTree::NodeType::PNode:
                                typeChar = 'P';
                                break;
                            case spqr_compat::SPQRTree::NodeType::RNode:
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
                            const spqr_compat::Skeleton &skel = spqr.skeleton(treeNode);

                            for (spqr_compat::node h : skelG.nodes)
                            {
                                spqr_compat::node vB = skel.original(h);
                                if (!vB)
                                    continue;

                                if (markBlk[vB] == stampBlk)
                                    continue;
                                markBlk[vB] = stampBlk;

                                spqr_compat::node vCc = blockToCC[vB];
                                spqr_compat::node vOrig = ccToOrig[vCc];
                                out << " " << C.node2name[vOrig];
                            }
                            out << "\n";
                        }

                        // Write V-lines (virtual edges in SPQR tree)
                        const auto& spqrTree = spqr.tree();
                        for (spqr_compat::edge te : spqrTree.edges)
                        {
                            spqr_compat::node src = spqrTree.source(te);
                            spqr_compat::node tgt = spqrTree.target(te);

                            std::string vName = makeId("V", virtEdgeCtr++);

                            spqr_compat::edge eSrc = spqr.skeletonEdgeSrc(te);

                            if (eSrc)
                            {
                                const spqr_compat::Skeleton &skelSrc = spqr.skeleton(src);
                                const auto& skelGraph = skelSrc.getGraph();

                                spqr_compat::node uSk = skelGraph.source(eSrc);
                                spqr_compat::node vSk = skelGraph.target(eSrc);

                                spqr_compat::node uB = skelSrc.original(uSk);
                                spqr_compat::node vB = skelSrc.original(vSk);

                                if (uB && vB)
                                {
                                    spqr_compat::node uCc = blockToCC[uB];
                                    spqr_compat::node vCc = blockToCC[vB];
                                    spqr_compat::node uOrig = ccToOrig[uCc];
                                    spqr_compat::node vOrig = ccToOrig[vCc];

                                    writeV(vName,
                                        spqrNodeNames[src],
                                        spqrNodeNames[tgt],
                                        C.node2name[uOrig],
                                        C.node2name[vOrig]);
                                }
                                continue;
                            }

                            // Fallback (should be rare): scan src skeleton for the virtual edge to tgt
                            const spqr_compat::Skeleton &skelSrc = spqr.skeleton(src);
                            const auto& skelGraphSrc = skelSrc.getGraph();
                            spqr_compat::edge virtualEdge = nullptr;
                            for (spqr_compat::edge e : skelGraphSrc.edges)
                            {
                                if (skelSrc.isVirtual(e) && skelSrc.twinTreeNode(e) == tgt)
                                {
                                    virtualEdge = e;
                                    break;
                                }
                            }
                            if (!virtualEdge)
                                continue;

                            spqr_compat::node uB = skelSrc.original(skelGraphSrc.source(virtualEdge));
                            spqr_compat::node vB = skelSrc.original(skelGraphSrc.target(virtualEdge));
                            if (!uB || !vB)
                                continue;

                            spqr_compat::node uCc = blockToCC[uB];
                            spqr_compat::node vCc = blockToCC[vB];
                            spqr_compat::node uOrig = ccToOrig[uCc];
                            spqr_compat::node vOrig = ccToOrig[vCc];

                            writeV(vName,
                                spqrNodeNames[src],
                                spqrNodeNames[tgt],
                                C.node2name[uOrig],
                                C.node2name[vOrig]);
                        }

                        // Write E-lines (real edges), assigned to their containing SPQR node
                        for (spqr_compat::node treeNode : spqr.tree().nodes)
                        {
                            const spqr_compat::Skeleton &skel = spqr.skeleton(treeNode);
                            for (spqr_compat::edge skelEdge : skel.getGraph().edges)
                            {
                                if (skel.isVirtual(skelEdge))
                                    continue;

                                spqr_compat::edge eB = skel.realEdge(skelEdge);
                                if (!eB)
                                    continue;

                                spqr_compat::edge eCc = blockEdgeToCC[eB];
                                if (!eCc)
                                    continue;

                                spqr_compat::node uOrig = ccToOrig[ccGraph.source(eCc)];
                                spqr_compat::node vOrig = ccToOrig[ccGraph.target(eCc)];

                                std::string eName = makeId("E", edgeCtr++);
                                writeE(eName,
                                    spqrNodeNames[treeNode],
                                    C.node2name[uOrig],
                                    C.node2name[vOrig]);
                            }

                            // The SPQR tree contains no self-loops, hence we need to output self-loops manually.
                            for (spqr_compat::node skelNode : skel.getGraph().nodes) {
                                spqr_compat::node vB = skel.original(skelNode);
                                spqr_compat::node vCc = blockToCC[vB];
                                spqr_compat::node vOrig = ccToOrig[vCc];

                                C.G.forEachAdj(vOrig, [&](node /*neighbor*/, edge e) {
                                    spqr_compat::node src = C.G.source(e);
                                    spqr_compat::node tgt = C.G.target(e);
                                    if (src != tgt) // Only process self loops
                                        return;

                                    // Self-loops are yielded twice by the iterator and may occur in multiple blocks and SPQR nodes,
                                    // hence we need to deduplicate them manually.
                                    if (self_loops_output.count(e)) {
                                        return;
                                    } else {
                                        self_loops_output.insert(e);
                                    }

                                    std::string eName = makeId("E", edgeCtr++);
                                    writeE(eName, spqrNodeNames[treeNode], C.node2name[vOrig], C.node2name[vOrig]);
                                });
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
            std::vector<uint8_t> plus_seen(n_comps, 0);
            std::vector<uint8_t> minus_seen(n_comps, 0);

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

            const uint32_t N = C.ubNumNodes;
            const auto &names = C.ubNodeNames;

            std::vector<bool> is_tip(N, false);
            {
                size_t tip_count = 0;
                for (uint32_t v = 0; v < N; v++) {
                    bool saw_plus = false, saw_minus = false;
                    for (const UBEdge *it = C.adjBegin(v), *end = C.adjEnd(v);
                         it != end; ++it) {
                        if (it->type_self == UB_PLUS) saw_plus = true;
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

    namespace ultrabubble_spqr {
        using PackedInc = std::pair<std::uint32_t, std::uint32_t>;

        static void add_packed_raw(std::vector<PackedInc> &out,
                                   std::uint32_t xpack,
                                   std::uint32_t ypack)
        {
            uint32_t x = xpack >> 1, y = ypack >> 1;
            std::uint32_t a1, b1;
            if (x <= y) { a1 = xpack; b1 = ypack; }
            else { a1 = ypack; b1 = xpack; }

            std::uint32_t xflip = xpack ^ 1u, yflip = ypack ^ 1u;
            std::uint32_t a2, b2;
            if (x <= y) { a2 = xflip; b2 = yflip; }
            else { a2 = yflip; b2 = xflip; }

            if (std::tie(a2, b2) < std::tie(a1, b1)) {
                out.emplace_back(a2, b2);
            } else {
                out.emplace_back(a1, b1);
            }
        }

        static bool parse_packed_endpoint(
            const std::string &s,
            const std::unordered_map<std::string, uint32_t> &ids,
            bool invert,
            std::uint32_t &packed)
        {
            if (s.size() < 2)
                return false;
            char sign = s.back();
            if (sign != '+' && sign != '-')
                return false;
            auto it = ids.find(s.substr(0, s.size() - 1));
            if (it == ids.end())
                return false;
            bool plus = (sign == '+');
            if (invert)
                plus = !plus;
            packed = (it->second << 1) | (plus ? 1u : 0u);
            return true;
        }

        static void solve_doubled_spqr()
        {
            auto &C = ctx();
            std::cout << "Finding ultrabubbles (SPQR weak superbubbles)...\n";

            const bool oldWeak = C.weakSuperbubbles;
            C.weakSuperbubbles = true;
            superbubble::solve();
            C.weakSuperbubbles = oldWeak;

            std::unordered_map<std::string, uint32_t> ids;
            ids.reserve(C.ubNodeNames.size());
            for (uint32_t i = 0; i < C.ubNodeNames.size(); ++i)
                ids.emplace(C.ubNodeNames[i], i);

            C.ultrabubbleIncPacked.clear();
            C.ultrabubbleIncPacked.reserve(C.superbubbles.size());

            for (const auto &sb : C.superbubbles) {
                std::uint32_t a, b;
                const std::string &s = C.node2name[sb.first];
                const std::string &t = C.node2name[sb.second];
                if (!parse_packed_endpoint(s, ids, true, a))
                    continue;
                if (!parse_packed_endpoint(t, ids, false, b))
                    continue;
                add_packed_raw(C.ultrabubbleIncPacked, a, b);
            }

            std::sort(C.ultrabubbleIncPacked.begin(), C.ultrabubbleIncPacked.end());
            C.ultrabubbleIncPacked.erase(
                std::unique(C.ultrabubbleIncPacked.begin(), C.ultrabubbleIncPacked.end()),
                C.ultrabubbleIncPacked.end());

            std::cout << "ULTRABUBBLES found: "
                      << C.ultrabubbleIncPacked.size()
                      << (C.includeTrivial ? " (trivial included)" : " (trivial excluded)")
                      << "\n";
        }

        void solve()
        {
            auto &C = ctx();
            if (C.G.numberOfNodes() != 0)
            {
                solve_doubled_spqr();
                return;
            }

            std::cout << "Finding ultrabubbles (SPQR weak superbubbles)...\n";
            PROFILE_FUNCTION();


            const uint32_t N = C.ubNumNodes;
            const auto &names = C.ubNodeNames;

            std::vector<bool> is_tip(N, false);
            for (uint32_t v = 0; v < N; v++) {
                bool saw_plus = false, saw_minus = false;
                for (const UBEdge *it = C.adjBegin(v), *end = C.adjEnd(v); it != end; ++it) {
                    if (it->type_self == ultrabubble::UB_PLUS) saw_plus = true;
                    if (it->type_self == ultrabubble::UB_MINUS) saw_minus = true;
                    if (saw_plus && saw_minus) break;
                }
                is_tip[v] = !(saw_plus && saw_minus);
            }

            std::vector<bool> is_tip_saved(is_tip);
            std::vector<bool> &can_start = is_tip;
            std::vector<bool> is_cut;
            ultrabubble::computeCutVertices(C, N, is_cut);
            for (uint32_t v = 0; v < N; v++)
                can_start[v] = is_tip_saved[v] || is_cut[v];
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
                        uint32_t v = stk.back();
                        stk.pop_back();
                        localId[v] = (int)cc.size();
                        cc.push_back(v);
                        for (const UBEdge *it = C.adjBegin(v), *end = C.adjEnd(v); it != end; ++it) {
                            if (!seen[it->neighbor]) {
                                seen[it->neighbor] = true;
                                stk.push_back(it->neighbor);
                            }
                        }
                    }
                }
            }

            struct DirectedCC {
                std::vector<std::pair<int, int>> edges;
                int nextId = 0;
            };

            std::vector<int> plus_dir(N, 0);
            std::vector<DirectedCC> directedByCC(comps.size());
            std::atomic<size_t> next{0};
            std::atomic<bool> abort_flag{false};
            std::exception_ptr eptr = nullptr;
            std::mutex ep_mtx;

            int T = std::min<int>(C.threads, (int)comps.size());
            if (T <= 0) T = 1;
            const size_t orientChunk =
                (comps.size() >= static_cast<size_t>(T) * 64u) ? 8u : 1u;

            std::vector<std::thread> threads;
            threads.reserve(T);
            for (int t = 0; t < T; ++t) {
                threads.emplace_back([&]() {
                    std::vector<int> cut_comp_scratch;
                    std::vector<uint32_t> cut_q_scratch;
                    try {
                        while (!abort_flag.load(std::memory_order_relaxed)) {
                            const size_t begin = next.fetch_add(orientChunk, std::memory_order_relaxed);
                            if (begin >= comps.size()) break;
                            const size_t end = std::min(begin + orientChunk, comps.size());
                            for (size_t ci = begin; ci < end; ++ci) {
                                if (abort_flag.load(std::memory_order_relaxed))
                                    break;

                                auto &cc = comps[ci];
                                size_t reserve_edges = 0;
                                if (cc.size() >= 16)
                                {
                                    for (uint32_t v : cc)
                                        reserve_edges += static_cast<size_t>(C.ubOffset[v + 1] - C.ubOffset[v]);
                                }
                                ultrabubble::DirectedEdgeBuilder out((int)cc.size(), reserve_edges);

                                for (uint32_t v : cc) {
                                    if (!plus_dir[v] && can_start[v]) {
                                        bool pe = is_tip_saved[v] ? true :
                                            ultrabubble::choose_cut_vertex_start(v, cc, localId, C,
                                                                                 cut_comp_scratch, cut_q_scratch);
                                        plus_dir[v] = pe ? 1 : 2;
                                        ultrabubble::orient(v, pe, plus_dir, localId, C, out);
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

                            auto &directed_edges = out.edges;
                            std::sort(directed_edges.begin(), directed_edges.end());
                            directed_edges.erase(
                                std::unique(directed_edges.begin(), directed_edges.end()),
                                directed_edges.end());

                                directedByCC[ci].edges = std::move(directed_edges);
                                directedByCC[ci].nextId = out.nextId;
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

            std::vector<uint32_t> conflict_base(comps.size(), N);
            uint32_t directed_node_count = N;
            for (size_t ci = 0; ci < comps.size(); ++ci) {
                conflict_base[ci] = directed_node_count;
                directed_node_count += (uint32_t)(directedByCC[ci].nextId - (int)comps[ci].size());
            }

            std::vector<superbubble::PrecomputedCcInput> precomputedComponents(comps.size());
            for (size_t ci = 0; ci < comps.size(); ++ci) {
                const auto &cc = comps[ci];
                const int k = (int)cc.size();
                const int nextId = directedByCC[ci].nextId;
                const uint32_t base = conflict_base[ci];
                auto &pc = precomputedComponents[ci];
                pc.localToGlobal.reserve(nextId);
                for (uint32_t v : cc)
                    pc.localToGlobal.push_back(v);
                for (int v = k; v < nextId; ++v)
                    pc.localToGlobal.push_back(base + (uint32_t)(v - k));
                pc.edges = std::move(directedByCC[ci].edges);
            }

            auto forEachGlobalDirectedEdge = [&](auto &&fn) {
                for (const auto &pc : precomputedComponents) {
                    const auto &map = pc.localToGlobal;
                    for (const auto &e : pc.edges) {
                        const uint32_t s = map[(size_t)e.first];
                        const uint32_t t = map[(size_t)e.second];
                        fn(s, t);
                    }
                }
	            };
            const bool needGlobalGraph = !(C.weakSuperbubbles && !C.includeTrivial);

            std::vector<int> odeg(directed_node_count, 0);
            std::vector<uint32_t> sole_out(directed_node_count, UINT32_MAX);
            std::vector<std::pair<uint32_t, uint32_t>> directed_edges;
            size_t edge_total = 0;
            for (auto &d : precomputedComponents) edge_total += d.edges.size();
            const bool materializeDirectedEdges =
                needGlobalGraph && C.includeTrivial;
            if (materializeDirectedEdges)
                directed_edges.reserve(edge_total);

            forEachGlobalDirectedEdge([&](uint32_t s, uint32_t t) {
                if (materializeDirectedEdges)
                    directed_edges.emplace_back(s, t);
                if (odeg[s] == 0)
                    sole_out[s] = t;
                odeg[s]++;
            });

            C.node2name.clear();
            C.name2node.clear();
            C.superbubbles.clear();
            C.isTrashNodeByIndex.assign(directed_node_count, 1);
            for (uint32_t i = 0; i < N && i < directed_node_count; ++i)
                C.isTrashNodeByIndex[i] = (names[i] == "_trash") ? 1 : 0;

            spqr_compat::node firstNode = nullptr;
            if (needGlobalGraph && directed_node_count != 0) {
                firstNode = C.G.newNodes(directed_node_count);
            }
            if (needGlobalGraph) {
                C.isEntry = NodeArray<bool>(C.G, false);
                C.isExit = NodeArray<bool>(C.G, false);
                C.inDeg = NodeArray<int>(C.G, 0);
                C.outDeg = NodeArray<int>(C.G, 0);
            }
            if (needGlobalGraph && edge_total != 0) {
                static_assert(sizeof(std::pair<uint32_t, uint32_t>) == 2 * sizeof(uint32_t),
                              "directed edge pair layout must be compact");
                if (materializeDirectedEdges) {
                    for (const auto &e : directed_edges) {
                        C.outDeg[node{e.first}]++;
                        C.inDeg[node{e.second}]++;
                    }
                } else {
                    forEachGlobalDirectedEdge([&](uint32_t s, uint32_t t) {
                        C.outDeg[node{s}]++;
                        C.inDeg[node{t}]++;
                    });
                }
                if (C.includeTrivial) {
                    C.G.newEdgesBatchFlat(
                        reinterpret_cast<const uint32_t *>(directed_edges.data()),
                        (uint32_t)directed_edges.size());
                }
            }

            const bool oldWeak = C.weakSuperbubbles;
            C.weakSuperbubbles = true;
            superbubble::clearRiskyWeakSuperbubbles();
            superbubble::solveStreamingPrecomputed(precomputedComponents);
            C.weakSuperbubbles = oldWeak;

            std::vector<std::pair<uint32_t, uint32_t>> riskyPairs;
            {
                auto risky = superbubble::getRiskyWeakSuperbubbles();
                riskyPairs.reserve(risky.size());
                for (const auto &p : risky)
                {
                    uint32_t s = p.first.index();
                    uint32_t t = p.second.index();
                    if (s < directed_node_count && t < directed_node_count)
                        riskyPairs.emplace_back(s, t);
                }
                std::sort(riskyPairs.begin(), riskyPairs.end());
                riskyPairs.erase(std::unique(riskyPairs.begin(), riskyPairs.end()),
                                 riskyPairs.end());
            }

            bool validationBuilt = false;
            std::vector<uint64_t> outOff, inOff;
            std::vector<uint32_t> outTo, inFrom;
            struct ValidationScratch
            {
                std::vector<uint8_t> fwd;
                std::vector<uint8_t> canReachT;
                std::vector<int> indeg;
                std::vector<uint32_t> q;
                std::vector<uint32_t> seenF;
                std::vector<uint32_t> seenB;
                std::vector<uint32_t> region;

                void ensure(uint32_t n)
                {
                    if (fwd.size() != n)
                    {
                        fwd.assign(n, 0);
                        canReachT.assign(n, 0);
                        indeg.assign(n, 0);
                    }
                }
            };
            ValidationScratch validationScratch;

            auto ensureValidationIndex = [&]() {
                if (validationBuilt)
                    return;
                const size_t nDir = static_cast<size_t>(directed_node_count);
                outOff.assign(nDir + 1, 0);
                inOff.assign(nDir + 1, 0);
                if (materializeDirectedEdges) {
                    for (const auto &e : directed_edges)
                    {
                        outOff[static_cast<size_t>(e.first) + 1]++;
                        inOff[static_cast<size_t>(e.second) + 1]++;
                    }
                } else {
                    forEachGlobalDirectedEdge([&](uint32_t s, uint32_t t) {
                        outOff[static_cast<size_t>(s) + 1]++;
                        inOff[static_cast<size_t>(t) + 1]++;
                    });
                }
                for (uint32_t i = 1; i <= directed_node_count; ++i)
                {
                    outOff[i] += outOff[i - 1];
                    inOff[i] += inOff[i - 1];
                }

                outTo.resize(edge_total);
                inFrom.resize(edge_total);
                std::vector<uint64_t> outCursor = outOff;
                std::vector<uint64_t> inCursor = inOff;
                if (materializeDirectedEdges) {
                    for (const auto &e : directed_edges)
                    {
                        outTo[outCursor[e.first]++] = e.second;
                        inFrom[inCursor[e.second]++] = e.first;
                    }
                } else {
                    forEachGlobalDirectedEdge([&](uint32_t s, uint32_t t) {
                        outTo[outCursor[s]++] = t;
                        inFrom[inCursor[t]++] = s;
                    });
                }

		                validationBuilt = true;
		            };
	            auto hasDirectValidationEdge = [&](uint32_t s, uint32_t t) {
	                for (uint64_t ei = outOff[s]; ei < outOff[s + 1]; ++ei)
	                    if (outTo[ei] == t)
	                        return true;
	                return false;
	            };

	            auto rejectsClosedCyclicRegion = [&](uint32_t s,
	                                                 uint32_t t,
	                                                 ValidationScratch &scratch) -> bool {
	                if (s >= directed_node_count || t >= directed_node_count)
	                    return false;
	                ensureValidationIndex();
	                if (hasDirectValidationEdge(t, s))
	                    return false;
	                scratch.ensure(directed_node_count);

                auto &fwd = scratch.fwd;
                auto &canReachT = scratch.canReachT;
                auto &indeg = scratch.indeg;
                auto &q = scratch.q;
                auto &seenF = scratch.seenF;
                auto &seenB = scratch.seenB;
                auto &region = scratch.region;

                seenF.clear();
                seenB.clear();
                region.clear();
                q.clear();

                fwd[s] = 1;
                q.push_back(s);
                seenF.push_back(s);
                for (size_t qi = 0; qi < q.size(); ++qi)
                {
                    uint32_t u = q[qi];
                    if (u == t)
                        continue;
                    for (uint64_t ei = outOff[u]; ei < outOff[u + 1]; ++ei)
                    {
                        uint32_t v = outTo[ei];
                        if (!fwd[v])
                        {
                            fwd[v] = 1;
                            q.push_back(v);
                            seenF.push_back(v);
                        }
                    }
                }

                q.clear();
                canReachT[t] = 1;
                q.push_back(t);
                seenB.push_back(t);
                for (size_t qi = 0; qi < q.size(); ++qi)
                {
                    uint32_t u = q[qi];
                    if (u == s)
                        continue;
                    for (uint64_t ei = inOff[u]; ei < inOff[u + 1]; ++ei)
                    {
                        uint32_t v = inFrom[ei];
                        if (!canReachT[v])
                        {
                            canReachT[v] = 1;
                            q.push_back(v);
                            seenB.push_back(v);
                        }
                    }
                }

                for (uint32_t u : seenF)
                {
                    if (canReachT[u])
                        region.push_back(u);
                }

	                uint64_t badOut = 0, badIn = 0, weakBackEdges = 0;
	                bool decided = false;
	                bool reject = false;
	                for (uint32_t u : region)
	                {
	                    for (uint64_t ei = outOff[u]; ei < outOff[u + 1]; ++ei)
	                    {
	                        uint32_t v = outTo[ei];
	                        if (u == t && v == s)
	                            weakBackEdges++;
	                        if (!(fwd[v] && canReachT[v]) && u != t)
	                            badOut++;
	                        if (badOut != 0 || weakBackEdges != 0)
	                        {
	                            decided = true;
	                            reject = false;
	                            break;
	                        }
	                    }
	                    if (decided)
	                        break;
	                    for (uint64_t ei = inOff[u]; ei < inOff[u + 1]; ++ei)
	                    {
	                        uint32_t v = inFrom[ei];
	                        if (!(fwd[v] && canReachT[v]) && u != s)
	                            badIn++;
	                        if (badIn != 0)
	                        {
	                            decided = true;
	                            reject = false;
	                            break;
	                        }
	                    }
	                    if (decided)
	                        break;
	                }

	                if (!decided)
	                {
	                    for (uint32_t u : region)
	                        indeg[u] = 0;
	                    for (uint32_t u : region)
	                    {
	                        for (uint64_t ei = outOff[u]; ei < outOff[u + 1]; ++ei)
	                        {
	                            uint32_t v = outTo[ei];
	                            if (!(fwd[v] && canReachT[v]))
	                                continue;
	                            if (u == t && v == s)
	                                continue;
	                            indeg[v]++;
	                        }
	                    }

	                    q.clear();
	                    for (uint32_t u : region)
	                    {
	                        if (indeg[u] == 0)
	                            q.push_back(u);
	                    }
	                    uint64_t seenTopo = 0;
	                    for (size_t qi = 0; qi < q.size(); ++qi)
	                    {
	                        uint32_t u = q[qi];
	                        seenTopo++;
	                        for (uint64_t ei = outOff[u]; ei < outOff[u + 1]; ++ei)
	                        {
	                            uint32_t v = outTo[ei];
	                            if (!(fwd[v] && canReachT[v]))
	                                continue;
	                            if (u == t && v == s)
	                                continue;
	                            if (--indeg[v] == 0)
	                                q.push_back(v);
	                        }
	                    }

	                    reject = (badOut == 0 && badIn == 0 &&
	                              weakBackEdges == 0 && seenTopo != region.size());
	                }

	                for (uint32_t u : seenF)
	                    fwd[u] = 0;
	                for (uint32_t u : seenB)
	                    canReachT[u] = 0;
	                if (!decided)
	                    for (uint32_t u : region)
	                        indeg[u] = 0;

                return reject;
            };

	            const bool checkRiskyRegions = !riskyPairs.empty();
	            auto packRaw = [&](uint32_t x, uint32_t y) {
	                const bool xplus = (plus_dir[x] == 1);
	                const bool yplus = (plus_dir[y] != 1);
	                uint32_t a = (x << 1) | (xplus ? 1u : 0u);
	                uint32_t b = (y << 1) | (yplus ? 1u : 0u);
	                uint32_t ar = a >> 1;
	                uint32_t br = b >> 1;
	                PackedInc p1 = (ar <= br) ? PackedInc{a, b} : PackedInc{b, a};
	                PackedInc p2 = (ar <= br) ? PackedInc{a ^ 1u, b ^ 1u} : PackedInc{b ^ 1u, a ^ 1u};
	                return (p2.first < p1.first ||
	                        (p2.first == p1.first && p2.second < p1.second)) ? p2 : p1;
	            };
	            std::vector<PackedInc> rejectedRiskyPacked;
	            if (checkRiskyRegions)
	            {
	                const int riskyThreadCap = (directed_node_count >= 10000000u) ? 2 : 4;
	                const int riskyThreads =
		                    (C.threads > 1 && riskyPairs.size() >= 1024)
	                        ? std::min({static_cast<int>(C.threads),
	                                    riskyThreadCap,
	                                    static_cast<int>((riskyPairs.size() + 1023) / 1024)})
	                        : 1;
	                if (riskyThreads <= 1)
	                {
	                    for (const auto &p : riskyPairs)
	                    {
	                        if (rejectsClosedCyclicRegion(p.first, p.second, validationScratch))
	                        {
	                            PackedInc packed = packRaw(p.first, p.second);
	                            rejectedRiskyPacked.push_back(packed);
	                        }
	                    }
	                }
	                else
	                {
	                    ensureValidationIndex();
	                    std::atomic<size_t> nextRisky{0};
	                    std::vector<std::vector<PackedInc>> rejectedByThread(riskyThreads);
	                    std::vector<std::thread> riskyWorkers;
	                    riskyWorkers.reserve(riskyThreads);
	                    for (int tid = 0; tid < riskyThreads; ++tid)
	                    {
	                        riskyWorkers.emplace_back([&, tid]() {
	                            ValidationScratch localScratch;
	                            auto &localRejected = rejectedByThread[tid];
	                            while (true)
	                            {
	                                const size_t i = nextRisky.fetch_add(1, std::memory_order_relaxed);
	                                if (i >= riskyPairs.size())
	                                    break;
	                                const auto &p = riskyPairs[i];
	                                if (rejectsClosedCyclicRegion(p.first, p.second, localScratch))
	                                    localRejected.push_back(packRaw(p.first, p.second));
	                            }
	                        });
	                    }
	                    for (auto &th : riskyWorkers)
	                        th.join();
	                    size_t rejectedTotal = 0;
	                    for (const auto &local : rejectedByThread)
	                        rejectedTotal += local.size();
	                    rejectedRiskyPacked.reserve(rejectedTotal);
	                    for (auto &local : rejectedByThread)
	                        rejectedRiskyPacked.insert(rejectedRiskyPacked.end(),
	                                                   std::make_move_iterator(local.begin()),
	                                                   std::make_move_iterator(local.end()));
	                }
	                std::sort(rejectedRiskyPacked.begin(), rejectedRiskyPacked.end());
	                rejectedRiskyPacked.erase(std::unique(rejectedRiskyPacked.begin(), rejectedRiskyPacked.end()),
	                                          rejectedRiskyPacked.end());
	            }
            auto packOne = [&](const std::pair<spqr_compat::node, spqr_compat::node> &sb,
                               std::vector<PackedInc> &out) {
                uint32_t x = sb.first.index();
                uint32_t y = sb.second.index();
                if (x >= N || y >= N) return;
                if (x < C.isTrashNodeByIndex.size() && C.isTrashNodeByIndex[x]) return;
                if (y < C.isTrashNodeByIndex.size() && C.isTrashNodeByIndex[y]) return;
                if (!C.includeTrivial &&
                    x < odeg.size() &&
                    odeg[x] == 1 &&
                    x < sole_out.size() &&
                    sole_out[x] == y) {
                    return;
                }
		                out.emplace_back(packRaw(x, y));
	            };

	            C.ultrabubbleIncPacked.clear();
	            const size_t nSuperbubbles = C.superbubbles.size();
	            int packThreads = (C.threads > 1 && nSuperbubbles >= 100000)
	                ? std::min<int>(C.threads, static_cast<int>((nSuperbubbles + 99999) / 100000))
	                : 1;
	            if (packThreads <= 1) {
	                C.ultrabubbleIncPacked.reserve(nSuperbubbles);
	                for (const auto &sb : C.superbubbles)
	                    packOne(sb, C.ultrabubbleIncPacked);
	            } else {
                std::vector<std::vector<PackedInc>> packedByThread(packThreads);
                std::vector<std::thread> packWorkers;
                packWorkers.reserve(packThreads);
                for (int tid = 0; tid < packThreads; ++tid) {
                    const size_t begin = (nSuperbubbles * static_cast<size_t>(tid)) / packThreads;
                    const size_t end = (nSuperbubbles * static_cast<size_t>(tid + 1)) / packThreads;
                    packWorkers.emplace_back([&, tid, begin, end]() {
                        auto &local = packedByThread[tid];
                        local.reserve(end - begin);
                        for (size_t i = begin; i < end; ++i)
                            packOne(C.superbubbles[i], local);
                    });
                }
                for (auto &th : packWorkers)
                    th.join();
                size_t packedTotal = 0;
                for (const auto &local : packedByThread)
                    packedTotal += local.size();
                C.ultrabubbleIncPacked.reserve(packedTotal);
                for (auto &local : packedByThread) {
                    C.ultrabubbleIncPacked.insert(C.ultrabubbleIncPacked.end(),
                                                  std::make_move_iterator(local.begin()),
                                                  std::make_move_iterator(local.end()));
                }
            }

            std::sort(C.ultrabubbleIncPacked.begin(), C.ultrabubbleIncPacked.end());
            C.ultrabubbleIncPacked.erase(
                std::unique(C.ultrabubbleIncPacked.begin(), C.ultrabubbleIncPacked.end()),
                C.ultrabubbleIncPacked.end());
            if (!rejectedRiskyPacked.empty())
            {
                size_t write = 0;
                size_t rejectedIdx = 0;
                for (size_t i = 0; i < C.ultrabubbleIncPacked.size(); ++i)
                {
                    const PackedInc &p = C.ultrabubbleIncPacked[i];
                    while (rejectedIdx < rejectedRiskyPacked.size() &&
                           rejectedRiskyPacked[rejectedIdx] < p)
                    {
                        rejectedIdx++;
                    }
                    if (rejectedIdx < rejectedRiskyPacked.size() &&
                        rejectedRiskyPacked[rejectedIdx] == p)
                    {
                        continue;
                    }
                    C.ultrabubbleIncPacked[write++] = p;
                }
                C.ultrabubbleIncPacked.resize(write);
            }

            std::cout << "ULTRABUBBLES found: "
                      << C.ultrabubbleIncPacked.size()
                      << (C.includeTrivial ? " (trivial included)" : " (trivial excluded)")
                      << "\n";
        }
    }

    namespace ultrabubble_doubled {

        static bool tryCommitSuperbubble(spqr_compat::node source, spqr_compat::node sink)
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
            std::vector<spqr_compat::node> nodes;
            std::vector<spqr_compat::edge> edges;
        };

        struct ThreadArgs {
            size_t tid;
            size_t numThreads;
            size_t nItems;
            std::atomic<size_t> *nextIndex;
            std::vector<CcWork> *work;
            std::vector<std::vector<std::pair<spqr_compat::node, spqr_compat::node>>> *results;
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

                std::unordered_map<spqr_compat::node, int> nodeToId;
                nodeToId.reserve(nNodes);
                std::vector<spqr_compat::node> idToNode(nNodes);
                for (int j = 0; j < nNodes; j++)
                {
                    nodeToId[cc.nodes[j]] = j;
                    idToNode[j] = cc.nodes[j];
                }

                std::vector<std::pair<int,int>> directed_edges;
                directed_edges.reserve(cc.edges.size());
                auto& G = ctx().G;
                for (spqr_compat::edge e : cc.edges)
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

                    spqr_compat::node xg = idToNode[xid];
                    spqr_compat::node yg = idToNode[yid];

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

                        spqr_compat::node xg = idToNode[xid];
                        spqr_compat::node yg = idToNode[yid];

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
                for (spqr_compat::node v : G.nodes)
                    work[compIdx[v]].nodes.push_back(v);
                for (spqr_compat::edge e : G.edges)
                    work[compIdx[G.source(e)]].edges.push_back(e);
            }

            logger::info("Doubled ultrabubbles: {} CCs", nCC);

            std::vector<std::vector<std::pair<spqr_compat::node, spqr_compat::node>>> results(nCC);
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

    {
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
    }

    {
        MARK_SCOPE_MEM("io/read_graph");
        PROFILE_BLOCK("Graph reading");
        ::GraphIO::readGraph();
    }

    {
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
            else if (ctx().spqrWeakUltrabubbles)
            {
                solver::ultrabubble_spqr::solve();
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
        std::cout << "Snarls found: "
                  << snarlsFound.load(std::memory_order_relaxed)
                  << std::endl;
    }

    {
        PROFILING_REPORT();

    logger::info("Process PeakRSS: {:.2f} GiB",
                 memtime::peakRSSBytes() / (1024.0 * 1024.0 * 1024.0));

        mark::report();
        if (!g_report_json_path.empty())
        {
            mark::report_to_json(g_report_json_path);
        }
    }

    return 0;
}
