#include "graph_io.hpp"
#include "util/context.hpp"
#include "util/timer.hpp"
#include "util/logger.hpp"
#include "gfa_parser.hpp"

#ifdef BUBBLEFINDER_HAS_GBZ
#include "gbz_parser.hpp"
#endif

#include <fstream>
#include <regex>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <unistd.h>  
#include <stdexcept> 
#include <sstream>   

using namespace ogdf;

namespace GraphIO {



void readStandard()
{
    auto &C = ctx();

    if (C.bubbleType == Context::BubbleType::SNARL) {
        throw std::runtime_error("Standard graph input not supported for snarls, use GFA input");
    }
    if (C.bubbleType == Context::BubbleType::SPQR_TREE_ONLY) {
        throw std::runtime_error("Standard graph input not supported for spqr-tree-only, use GFA input");
    }

    int n, m;

    std::istream* input = nullptr;
    std::ifstream infile;

    if (!C.graphPath.empty()) {
        infile.open(C.graphPath);
        if (!infile) {
            throw std::runtime_error("Cannot open " + C.graphPath);
        }
        input = &infile;
    } else {
        input = &std::cin;
    }

    const char* srcName = C.graphPath.empty() ? "<stdin>" : C.graphPath.c_str();

    if (!(*input >> n >> m)) {
        throw std::runtime_error(
            std::string("Invalid .graph header in ") + srcName +
            ": expected 'n m' on first line.");
    }

    if (n < 0 || m < 0) {
        throw std::runtime_error(
            std::string("Invalid .graph header in ") + srcName +
            ": n and m must be non-negative.");
    }

    C.node2name.reserve(static_cast<size_t>(n));

    struct EdgeKey {
        std::string u, v;
        bool operator==(const EdgeKey& o) const { return u == o.u && v == o.v; }
    };
    struct EdgeKeyHash {
        std::size_t operator()(EdgeKey const& k) const {
            std::size_t h1 = std::hash<std::string>{}(k.u);
            std::size_t h2 = std::hash<std::string>{}(k.v);
            std::size_t h1_h = h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
            return h1_h;
        }
    };

    std::unordered_set<EdgeKey, EdgeKeyHash> edges;
    for (int i = 0; i < m; ++i) {
        std::string u, v;
        if (!(*input >> u >> v)) {
            std::ostringstream oss;
            oss << "Unexpected end of file while reading edge " << (i + 1)
                << " of " << m << " in " << srcName
                << " (expected 'u v' on each line).";
            throw std::runtime_error(oss.str());
        }
        edges.insert({u, v});
    }

    std::unordered_set<EdgeKey, EdgeKeyHash> processed;

    for (auto const& e : edges) {
        if (processed.count(e)) continue;

        EdgeKey rev{e.v, e.u};

        if (edges.count(rev)) {
            processed.insert(e);
            processed.insert(rev);

            if (!C.name2node.count(e.u)) {
                C.name2node[e.u] = C.G.newNode();
                C.node2name[C.name2node[e.u]] = e.u;
            }
            if (!C.name2node.count(e.v)) {
                C.name2node[e.v] = C.G.newNode();
                C.node2name[C.name2node[e.v]] = e.v;
            }

            node t1 = C.G.newNode();
            node t2 = C.G.newNode();

            C.node2name[t1] = "_trash";
            C.node2name[t2] = "_trash";

            C.G.newEdge(C.name2node[e.u], t1);
            C.G.newEdge(t1, C.name2node[e.v]);
            C.G.newEdge(C.name2node[e.v], t2);
            C.G.newEdge(t2, C.name2node[e.u]);
        } else {
            processed.insert(e);

            if (!C.name2node.count(e.u)) {
                C.name2node[e.u] = C.G.newNode();
                C.node2name[C.name2node[e.u]] = e.u;
            }
            if (!C.name2node.count(e.v)) {
                C.name2node[e.v] = C.G.newNode();
                C.node2name[C.name2node[e.v]] = e.v;
            }

            C.G.newEdge(C.name2node[e.u], C.name2node[e.v]);
        }
    }
}

namespace {

inline char flipSign(char c) { return c == '+' ? '-' : '+'; }
inline EdgePartType charToType(char c) { return c == '+' ? EdgePartType::PLUS : EdgePartType::MINUS; }

std::vector<ogdf::node> createNodes(BiGraph& bg) {
    auto &C = ctx();
    std::vector<ogdf::node> id2node(bg.n_nodes);
    C.name2node.reserve(bg.n_nodes);
    for (uint32_t i = 0; i < bg.n_nodes; ++i) {
        ogdf::node v = C.G.newNode();
        id2node[i] = v;
        C.node2name[v] = bg.node_names[i];
        C.name2node[bg.node_names[i]] = v;
    }
    C.gfaSegmentIds = std::move(bg.node_names);
    return id2node;
}

void buildSnarlGraph(BiGraph& bg) {
    auto &C = ctx();
    auto id2node = createNodes(bg);
    C._edge2types.init(C.G, std::make_pair(EdgePartType::NONE, EdgePartType::NONE));

    struct PairKey {
        uint32_t u, v;
        bool operator<(const PairKey& o) const { return u!=o.u ? u<o.u : v<o.v; }
        bool operator==(const PairKey& o) const { return u==o.u && v==o.v; }
    };
    std::vector<PairKey> pkeys;
    pkeys.reserve(bg.links.size());
    for (auto& lk : bg.links) {
        uint32_t a = std::min(lk.src, lk.dst), b = std::max(lk.src, lk.dst);
        pkeys.push_back({a, b});
    }
    std::sort(pkeys.begin(), pkeys.end());

    auto is_multi = [&](uint32_t a, uint32_t b) -> bool {
        PairKey key{std::min(a,b), std::max(a,b)};
        auto lo = std::lower_bound(pkeys.begin(), pkeys.end(), key);
        auto hi = std::upper_bound(lo, pkeys.end(), key);
        return (hi - lo) > 1;
    };

    for (auto& lk : bg.links) {
        EdgePartType t1 = charToType(lk.orient_src);
        EdgePartType t2 = charToType(flipSign(lk.orient_dst));
        uint32_t u = lk.src, v = lk.dst;
        if (u > v) { std::swap(u, v); std::swap(t1, t2); }

        if (!is_multi(u, v)) {
            ogdf::edge e = C.G.newEdge(id2node[u], id2node[v]);
            C._edge2types[e] = {t1, t2};
        } else {
            ogdf::node mid = C.G.newNode();
            C.node2name[mid] = "_trash";
            ogdf::edge e1 = C.G.newEdge(id2node[u], mid);
            C._edge2types[e1] = {t1, EdgePartType::PLUS};
            ogdf::edge e2 = C.G.newEdge(mid, id2node[v]);
            C._edge2types[e2] = {EdgePartType::PLUS, t2};
        }
    }
}

void buildUltrabubbleLightGraph(BiGraph& bg) {
    auto &C = ctx();
    const uint32_t N = bg.n_nodes;
    C.ubNumNodes  = N;
    C.ubNodeNames = std::move(bg.node_names);

    struct CanonEdge {
        uint32_t u, v;
        uint8_t  tu, tv; 

        bool operator<(const CanonEdge &o) const {
            if (u != o.u) return u < o.u;
            if (v != o.v) return v < o.v;
            if (tu != o.tu) return tu < o.tu;
            return tv < o.tv;
        }
        bool operator==(const CanonEdge &o) const {
            return u == o.u && v == o.v && tu == o.tu && tv == o.tv;
        }
    };

    std::vector<CanonEdge> edges;
    edges.reserve(bg.links.size());

    for (auto& lk : bg.links) {
        uint8_t t1 = (uint8_t)charToType(lk.orient_src);
        uint8_t t2 = (uint8_t)charToType(flipSign(lk.orient_dst));
        uint32_t u = lk.src, v = lk.dst;
        if (u > v) { std::swap(u, v); std::swap(t1, t2); }
        edges.push_back({u, v, t1, t2});
    }

    { std::vector<BiLink>().swap(bg.links); }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    const size_t E = edges.size();

    C.ubOffset.assign(N + 1, 0);
    for (const auto &e : edges) {
        C.ubOffset[e.u + 1]++;
        C.ubOffset[e.v + 1]++;
    }
    for (uint32_t i = 1; i <= N; i++) {
        C.ubOffset[i] += C.ubOffset[i - 1];
    }

    C.ubEdges.resize(C.ubOffset[N]);  // = 2 * E

    std::vector<uint32_t> cursor(C.ubOffset.begin(), C.ubOffset.end());

    for (const auto &e : edges) {
        C.ubEdges[cursor[e.u]++] = {e.v, e.tu, e.tv};
        C.ubEdges[cursor[e.v]++] = {e.u, e.tv, e.tu};
    }

    logger::info("graph built: {} nodes, {} edges (CSR: {} adj entries)",
                 N, E, C.ubEdges.size());
}

void buildSuperbubbleGraph(BiGraph& bg, bool directed_only) {
    auto &C = ctx();
    C.name2node.reserve(bg.n_nodes * 2);
    std::vector<ogdf::node> id2plus(bg.n_nodes), id2minus(bg.n_nodes);

    for (uint32_t i = 0; i < bg.n_nodes; ++i) {
        std::string pn = bg.node_names[i] + "+", mn = bg.node_names[i] + "-";
        ogdf::node vp = C.G.newNode(), vm = C.G.newNode();
        id2plus[i] = vp; id2minus[i] = vm;
        C.node2name[vp] = pn; C.node2name[vm] = mn;
        C.name2node[pn] = vp; C.name2node[mn] = vm;
    }
    C.gfaSegmentIds = std::move(bg.node_names);

    auto getNode = [&](uint32_t id, char o) -> ogdf::node {
        return (o == '+') ? id2plus[id] : id2minus[id];
    };

    struct DE { int u, v; bool operator<(const DE& o) const { return u!=o.u ? u<o.u : v<o.v; }
                          bool operator==(const DE& o) const { return u==o.u && v==o.v; } };
    std::vector<DE> des;
    des.reserve(directed_only ? bg.links.size() : bg.links.size() * 2);

    for (auto& lk : bg.links) {
        des.push_back({getNode(lk.src, lk.orient_src)->index(),
                       getNode(lk.dst, lk.orient_dst)->index()});
        if (!directed_only)
            des.push_back({getNode(lk.dst, flipSign(lk.orient_dst))->index(),
                           getNode(lk.src, flipSign(lk.orient_src))->index()});
    }
    std::sort(des.begin(), des.end());
    des.erase(std::unique(des.begin(), des.end()), des.end());

    std::unordered_map<int, ogdf::node> idx2n;
    for (ogdf::node v : C.G.nodes) idx2n[v->index()] = v;
    for (auto& d : des) C.G.newEdge(idx2n[d.u], idx2n[d.v]);
}

void buildSpqrGraph(BiGraph& bg) {
    auto &C = ctx();
    auto id2node = createNodes(bg);
    for (auto& lk : bg.links) C.G.newEdge(id2node[lk.src], id2node[lk.dst]);
}

} 



namespace {

inline bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

BiGraph parse_graph_input(const std::string& path, int threads) {
    if (ends_with(path, ".gbz")) {
#ifdef BUBBLEFINDER_HAS_GBZ
        logger::info("GBZ parser: reading '{}'", path);
        auto bg = GBZParser::parse_file(path);
        logger::info("GBZ parser: {} segments, {} links", bg.n_nodes, bg.links.size());
        return bg;
#else
        throw std::runtime_error(
            "GBZ support not compiled. Rebuild with -DBUBBLEFINDER_HAS_GBZ=ON "
            "or convert: vg convert -f -H " + path + " | gzip > output.gfa.gz");
#endif
    }

    logger::info("GFA parser: reading '{}'", path);
    auto bg = GFAParser::parse_file(path, threads);
    logger::info("GFA parser: {} segments, {} links", bg.n_nodes, bg.links.size());
    return bg;
}

} 

void readGFA()
{
    auto &C = ctx();
    if (C.graphPath.empty())
        throw std::runtime_error("GFA input needs -g <file>");

    auto bg = parse_graph_input(C.graphPath, (int)C.threads);
    if (bg.n_nodes == 0) { logger::info("Empty graph"); return; }

    switch (C.bubbleType) {
        case Context::BubbleType::ULTRABUBBLE:
            buildUltrabubbleLightGraph(bg);
            return; 
        case Context::BubbleType::SNARL:
            C._edge2types.init(C.G, std::make_pair(EdgePartType::NONE, EdgePartType::NONE));
            buildSnarlGraph(bg);
            break;
        case Context::BubbleType::SUPERBUBBLE:
            buildSuperbubbleGraph(bg, C.inputFormat == Context::InputFormat::GfaDirected);
            break;
        case Context::BubbleType::SPQR_TREE_ONLY:
            buildSpqrGraph(bg);
            break;
        default:
            break;
    }

    logger::info("OGDF graph built: {} nodes, {} edges", C.G.numberOfNodes(), C.G.numberOfEdges());
}

namespace {

    std::string shellEscape(const std::string &s) {
        std::string r;
        r.reserve(s.size() + 2);
        r.push_back('\'');
        for (char c : s) {
            if (c == '\'') {
                r += "'\\''";
            } else {
                r.push_back(c);
            }
        }
        r.push_back('\'');
        return r;
    }

    std::string decompressToTempFile(const std::string &path,
                                     Context::Compression comp)
    {
        char tmpl[] = "/tmp/bubblefinder_XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd == -1) {
            throw std::runtime_error("mkstemp failed when creating temp file for decompression");
        }
        ::close(fd);

        std::string tmpPath = tmpl;

        std::string prog;
        switch (comp) {
            case Context::Compression::Gzip:
                prog = "gzip -dc ";
                break;
            case Context::Compression::Bzip2:
                prog = "bzip2 -dc ";
                break;
            case Context::Compression::Xz:
                prog = "xz -dc ";
                break;
            case Context::Compression::None:
            default:
                std::remove(tmpPath.c_str());
                throw std::runtime_error("decompressToTempFile called with Compression::None");
        }

        std::string cmd = prog + shellEscape(path);

        FILE *pipe = ::popen(cmd.c_str(), "r");
        if (!pipe) {
            std::remove(tmpPath.c_str());
            throw std::runtime_error("Failed to run decompression command: " + prog);
        }

        std::ofstream out(tmpPath, std::ios::binary);
        if (!out) {
            ::pclose(pipe);
            std::remove(tmpPath.c_str());
            throw std::runtime_error("Failed to open temp file for decompression: " + tmpPath);
        }

        char buffer[1 << 16];
        while (true) {
            std::size_t n = std::fread(buffer, 1, sizeof(buffer), pipe);
            if (n > 0) {
                out.write(buffer, static_cast<std::streamsize>(n));
            }
            if (std::ferror(pipe)) {
                ::pclose(pipe);
                out.close();
                std::remove(tmpPath.c_str());
                throw std::runtime_error("Error reading from decompression pipe");
            }
            if (n == 0) {
                break;
            }
        }

        int status = ::pclose(pipe);
        out.close();
        if (status != 0) {
            std::remove(tmpPath.c_str());
            throw std::runtime_error("Decompression command failed: " + cmd);
        }

        return tmpPath;
    }

}


void readGraph() {
    auto &C = ctx();
    TIME_BLOCK("Graph read");

    logger::info("Starting to read graph");

    if (C.inputFormat == Context::InputFormat::Gfa ||
        C.inputFormat == Context::InputFormat::GfaDirected)
    {
        readGFA();

        if (C.bubbleType == Context::BubbleType::ULTRABUBBLE) {
            logger::info("Graph read");
            return;
        }

        C.isEntry = NodeArray<bool>(C.G, false);
        C.isExit  = NodeArray<bool>(C.G, false);
        C.inDeg   = NodeArray<int>(C.G, 0);
        C.outDeg  = NodeArray<int>(C.G, 0);
        for (edge e : C.G.edges) {
            C.outDeg(e->source())++;
            C.inDeg (e->target())++;
        }
        logger::info("Graph read");
        return;
    }

    std::string originalPath = C.graphPath;
    std::string tempPath;
    bool usingTempFile = false;

    if (C.compression != Context::Compression::None) {
        logger::info("Detected compressed input; starting decompression");
        tempPath = decompressToTempFile(C.graphPath, C.compression);
        usingTempFile = true;
        C.graphPath = tempPath;
        logger::info("Decompressed '{}' to temporary file '{}'",
                     originalPath, tempPath);
    }

    try {
        if (C.bubbleType == Context::BubbleType::SNARL) {
            throw std::runtime_error("Standard .graph input is not supported for snarls, use GFA");
        }
        if (C.bubbleType == Context::BubbleType::SPQR_TREE_ONLY) {
            throw std::runtime_error("Standard .graph input is not supported for spqr-tree, use GFA");
        }
        readStandard();
    } catch (...) {
        if (usingTempFile) { C.graphPath = originalPath; std::remove(tempPath.c_str()); }
        throw;
    }

    if (usingTempFile) { C.graphPath = originalPath; std::remove(tempPath.c_str()); }

    C.isEntry = NodeArray<bool>(C.G, false);
    C.isExit  = NodeArray<bool>(C.G, false);
    C.inDeg   = NodeArray<int>(C.G, 0);
    C.outDeg  = NodeArray<int>(C.G, 0);
    for (edge e : C.G.edges) {
        C.outDeg(e->source())++;
        C.inDeg (e->target())++;
    }
    logger::info("Graph read");
}


void drawGraph(const ogdf::Graph &G, const std::string &file)
{
    return;
    using namespace ogdf;

    auto &C = ctx();
    TIME_BLOCK("Drawing graph");

    GraphAttributes GA(G,
        GraphAttributes::nodeGraphics | GraphAttributes::edgeGraphics |
        GraphAttributes::nodeLabel    | GraphAttributes::edgeLabel    |
        GraphAttributes::nodeStyle    | GraphAttributes::edgeStyle);
    GA.directed() = true;

    for (node v : G.nodes) {
        GA.label(v) = C.node2name.count(v) ? C.node2name[v]
                                           : std::to_string(v->index());
        GA.shape(v) = Shape::Ellipse;
        GA.width(v) = GA.height(v) = 20.0;
    }

    FMMMLayout().call(GA);

    constexpr double GAP = 12.0;

    struct PairHash {
        size_t operator()(const std::pair<int,int>& p) const noexcept {
            return (static_cast<size_t>(p.first) << 32) ^ p.second;
        }
    };

    std::unordered_map<std::pair<int,int>, std::vector<edge>, PairHash> bundle;

    for (edge e : G.edges) {
        int u = e->source()->index();
        int v = e->target()->index();
        bundle[{u, v}].push_back(e);
    }

    for (auto &entry : bundle) {
        auto &vec = entry.second;
        if (vec.size() <= 1) continue;

        edge e0 = vec[0];
        node a  = e0->source();
        node b  = e0->target();

        double ax = GA.x(a), ay = GA.y(a);
        double bx = GA.x(b), by = GA.y(b);

        double dx = bx - ax, dy = by - ay;
        double len = std::hypot(dx, dy);
        if (len == 0) len = 1;
        double px = -dy / len, py = dx / len;

        const int sign = 1;
        const int k = static_cast<int>(vec.size());

        for (int i = 0; i < k; ++i) {
            edge e = vec[i];
            double shift = (i - (k - 1) / 2.0) * GAP;

            double mx = (ax + bx) * 0.5 + sign * px * shift;
            double my = (ay + by) * 0.5 + sign * py * shift;

            GA.bends(e).clear();
            GA.bends(e).pushBack(DPoint(mx, my));
        }
    }

    const std::string tmp = file + ".svg.tmp";
    ogdf::GraphIO::drawSVG(GA, tmp);

    std::ifstream in(tmp);
    std::ofstream out(file + ".svg");

    std::string header;
    std::getline(in, header);
    std::string openTag;
    std::getline(in, openTag);
    out << header << '\n' << openTag << '\n';

    double x0 = 0, y0 = 0, w = 0, h = 0;
    std::smatch m;
    if (std::regex_search(openTag, m,
        std::regex(R"(viewBox\s*=\s*\"([\-0-9\.eE]+)\s+([\-0-9\.eE]+)\s+([\-0-9\.eE]+)\s+([\-0-9\.eE]+))"))) {
        x0 = std::stod(m[1]); y0 = std::stod(m[2]);
        w  = std::stod(m[3]); h  = std::stod(m[4]);
    } else if (std::regex_search(openTag, m,
        std::regex(R"(width=\"([0-9\.]+)\".*height=\"([0-9\.]+))"))) {
        w = std::stod(m[1]); h = std::stod(m[2]);
    }
    out << "  <rect x=\"" << x0 << "\" y=\"" << y0
        << "\" width=\"" << w  << "\" height=\"" << h
        << "\" fill=\"#ffffff\"/>\n";

    out << in.rdbuf();
    in.close();
    out.close();
    std::remove(tmp.c_str());
}


std::vector<std::pair<std::string, std::string>>
project_bubblegun_pairs_from_doubled() {
    auto& sb    = ctx().superbubbles; 
    auto& names = ctx().node2name;    

    auto is_oriented = [](const std::string& s) -> bool {
        return !s.empty() && (s.back() == '+' || s.back() == '-');
    };
    auto strip = [](std::string s) -> std::string {
        if (!s.empty() && (s.back() == '+' || s.back() == '-')) s.pop_back();
        return s;
    };
    auto pair_hash = [](const std::pair<std::string,std::string>& p) -> std::size_t {
        return std::hash<std::string>{}(p.first) ^
               (std::hash<std::string>{}(p.second) << 1);
    };

    std::vector<std::pair<std::string, std::string>> out;
    out.reserve(sb.size());

    std::unordered_set<std::pair<std::string,std::string>, decltype(pair_hash)> seen(0, pair_hash);
    std::unordered_set<std::pair<std::string,std::string>, decltype(pair_hash)> seen_oriented(0, pair_hash);

    for (auto const& e : sb) {
        const std::string& sa = names[e.first];
        const std::string& sbn = names[e.second];

        if (!seen_oriented.insert({sa, sbn}).second) continue;

        if (is_oriented(sa) && sa.back() == '-') continue;

        std::string a = strip(sa);
        std::string b = strip(sbn);
        if (a == b) continue;

        if (seen.insert({a, b}).second) {
            out.emplace_back(std::move(a), std::move(b));
        }
    }

    return out;
}


void writeSuperbubbles()
{
    auto &C = ctx();

    if (C.bubbleType == Context::BubbleType::SPQR_TREE_ONLY)
    {
        throw std::runtime_error("Cannot write superbubbles when bubbleType is SPQR_TREE_ONLY");
    }

    if (C.bubbleType == Context::BubbleType::SNARL)
    {
        if (C.outputPath.empty())
        {
            std::cout << C.snarls.size() << "\n";
            for (auto &s : C.snarls)
            {
                for (auto &v : s)
                {
                    std::cout << v << " ";
                }
                std::cout << std::endl;
            }
            if (!std::cout)
            {
                throw std::runtime_error("Error while writing snarls to standard output");
            }
        }
        else
        {
            std::ofstream out(C.outputPath);
            if (!out)
            {
                throw std::runtime_error("Failed to open output file '" +
                                         C.outputPath + "' for writing");
            }
            out << C.snarls.size() << "\n";
            for (auto &s : C.snarls)
            {
                for (auto &v : s)
                {
                    out << v << " ";
                }
                out << "\n";
            }
            if (!out)
            {
                throw std::runtime_error("Error while writing snarls to output file '" +
                                         C.outputPath + "'");
            }
        }
        return;
    }

    if (C.bubbleType == Context::BubbleType::ULTRABUBBLE)
    {
        auto unpack = [](std::uint32_t p) -> std::pair<std::uint32_t, bool>
        {
            return {(p >> 1), (p & 1u) != 0u};
        };

        auto write_one = [&](std::ostream &os, std::uint32_t packed)
        {
            auto [gid, plus] = unpack(packed);
            const std::string &name = C.ubNodeNames.at((size_t)gid);
            os << name << (plus ? '+' : '-');
        };

        if (C.outputPath.empty())
        {
            std::cout << C.ultrabubbleIncPacked.size() << "\n";
            for (auto &p : C.ultrabubbleIncPacked)
            {
                write_one(std::cout, p.first);
                std::cout << " ";
                write_one(std::cout, p.second);
                std::cout << "\n";
            }
            if (!std::cout)
            {
                throw std::runtime_error("Error while writing ultrabubbles to standard output");
            }
        }
        else
        {
            std::ofstream out(C.outputPath);
            if (!out)
            {
                throw std::runtime_error("Failed to open output file '" +
                                         C.outputPath + "' for writing");
            }
            out << C.ultrabubbleIncPacked.size() << "\n";
            for (auto &p : C.ultrabubbleIncPacked)
            {
                write_one(out, p.first);
                out << " ";
                write_one(out, p.second);
                out << "\n";
            }
            if (!out)
            {
                throw std::runtime_error("Error while writing ultrabubbles to output file '" +
                                         C.outputPath + "'");
            }
        }
        return;
    }

    std::vector<std::pair<std::string, std::string>> res;

    if (C.inputFormat == Context::InputFormat::Gfa &&
        !C.directedSuperbubbles)
    {
        auto has_orient = [](const std::string &s)
        {
            return !s.empty() && (s.back() == '+' || s.back() == '-');
        };
        auto flip_char = [](char c)
        { return c == '+' ? '-' : (c == '-') ? '+' : c; };
        auto invert = [&](std::string s)
        {
            if (has_orient(s))
                s.back() = flip_char(s.back());
            return s;
        };
        auto strip = [&](std::string s)
        {
            if (has_orient(s))
                s.pop_back();
            return s;
        };

        auto canonical_mirror_rep = [&](const std::string &x, const std::string &y)
        {
            std::string xA = x, yA = y;
            std::string xB = invert(y), yB = invert(x);
            if (std::tie(xB, yB) < std::tie(xA, yA))
                return std::pair<std::string, std::string>{xB, yB};
            return std::pair<std::string, std::string>{xA, yA};
        };

        auto transform_and_unorder = [&](const std::pair<std::string, std::string> &p)
        {
            std::string a = invert(p.first);
            std::string b = p.second;
            if (b < a)
                std::swap(a, b);
            return std::pair<std::string, std::string>{std::move(a), std::move(b)};
        };

        auto pair_hash2 = [](const std::pair<std::string, std::string> &pr) -> std::size_t
        {
            std::size_t h1 = std::hash<std::string>{}(pr.first);
            std::size_t h2 = std::hash<std::string>{}(pr.second);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        };
        std::unordered_set<std::pair<std::string, std::string>, decltype(pair_hash2)>
            seen2(0, pair_hash2);

        for (auto &w : C.superbubbles)
        {
            const std::string s = C.node2name[w.first];
            const std::string t = C.node2name[w.second];

            auto rep = canonical_mirror_rep(s, t);
            auto fin = transform_and_unorder(rep);

            fin.first = strip(fin.first);
            fin.second = strip(fin.second);

            if (fin.first != fin.second)
            {
                if (seen2.insert(fin).second)
                {
                    res.emplace_back(std::move(fin));
                }
            }
        }
    }
    else
    {
        for (auto &w : C.superbubbles)
        {
            res.push_back({C.node2name[w.first], C.node2name[w.second]});
        }
    }

    if (C.outputPath.empty())
    {
        std::cout << res.size() << "\n";
        for (auto &p : res)
        {
            std::cout << p.first << " " << p.second << "\n";
        }
        if (!std::cout)
        {
            throw std::runtime_error("Error while writing superbubbles to standard output");
        }
    }
    else
    {
        std::ofstream out(C.outputPath);
        if (!out)
        {
            throw std::runtime_error("Failed to open output file '" +
                                     C.outputPath + "' for writing");
        }
        out << res.size() << "\n";
        for (auto &p : res)
        {
            out << p.first << " " << p.second << "\n";
        }
        if (!out)
        {
            throw std::runtime_error("Error while writing superbubbles to output file '" +
                                     C.outputPath + "'");
        }
    }
}

}