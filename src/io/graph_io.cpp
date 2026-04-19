#include "graph_io.hpp"
#include "util/context.hpp"
#include "util/timer.hpp"
#include "util/logger.hpp"
#include "gfa_parser.hpp"

#include "gbz_parser.hpp"

#include <fstream>
#include <regex>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <unistd.h>

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

    std::vector<char> buf;
    const char *srcName = C.graphPath.empty() ? "<stdin>" : C.graphPath.c_str();

    if (!C.graphPath.empty()) {
        std::FILE *fp = std::fopen(C.graphPath.c_str(), "rb");
        if (!fp) throw std::runtime_error(std::string("Cannot open ") + srcName);
        std::fseek(fp, 0, SEEK_END);
        long sz = std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        if (sz < 0) {
            std::fclose(fp);
            throw std::runtime_error(std::string("ftell failed on ") + srcName);
        }
        buf.resize(static_cast<size_t>(sz));
        size_t got = std::fread(buf.data(), 1, buf.size(), fp);
        int rd_err = std::ferror(fp);
        std::fclose(fp);
        if (rd_err || got != buf.size()) {
            throw std::runtime_error(std::string("Short read on ") + srcName);
        }
    } else {
        char chunk[1 << 16];
        while (true) {
            size_t got = std::fread(chunk, 1, sizeof(chunk), stdin);
            if (got == 0) break;
            buf.insert(buf.end(), chunk, chunk + got);
        }
    }
    buf.push_back('\n');

    const char *p   = buf.data();
    const char *end = buf.data() + buf.size();

    auto skip_ws = [&]() {
        while (p < end) {
            char c = *p;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++p;
            else break;
        }
    };

    auto parse_uint = [&](uint64_t &out) -> bool {
        skip_ws();
        if (p >= end || *p < '0' || *p > '9') return false;
        uint64_t v = 0;
        while (p < end && *p >= '0' && *p <= '9') {
            v = v * 10u + static_cast<uint64_t>(*p - '0');
            ++p;
        }
        out = v;
        return true;
    };

    // --- Header: "n m" ----------------------------------------------------
    uint64_t n64 = 0, m64 = 0;
    if (!parse_uint(n64) || !parse_uint(m64)) {
        throw std::runtime_error(
            std::string("Invalid .graph header in ") + srcName +
            ": expected 'n m' (non-negative integers) on the first line.");
    }
    if (n64 > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(std::string("n too large in ") + srcName +
                                 " (.graph reader uses 32-bit node IDs).");
    }
    const uint32_t n = static_cast<uint32_t>(n64);
    const size_t   m = static_cast<size_t>(m64);

    // --- Parse all m edges as (uint32_t, uint32_t) -----------------------
    std::vector<std::pair<uint32_t, uint32_t>> edges_raw;
    edges_raw.reserve(m);
    for (size_t i = 0; i < m; ++i) {
        uint64_t u, v;
        if (!parse_uint(u) || !parse_uint(v)) {
            std::ostringstream oss;
            oss << "Failed to parse edge " << (i + 1) << " of " << m
                << " in " << srcName
                << " (expected two non-negative integers per line; "
                << ".graph reader requires integer node IDs).";
            throw std::runtime_error(oss.str());
        }
        if (u >= n || v >= n) {
            std::ostringstream oss;
            oss << "Edge " << (i + 1) << " in " << srcName
                << " references node id (" << u << " or " << v
                << ") out of range [0, " << n << ").";
            throw std::runtime_error(oss.str());
        }
        edges_raw.push_back({static_cast<uint32_t>(u), static_cast<uint32_t>(v)});
    }
    std::vector<char>().swap(buf);
    auto encode = [](uint32_t u, uint32_t v) -> uint64_t {
        return (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(v);
    };

    std::unordered_set<uint64_t> edge_set;
    edge_set.reserve(edges_raw.size() * 2);
    std::vector<std::pair<uint32_t, uint32_t>> edges_ordered;
    edges_ordered.reserve(edges_raw.size());
    for (const auto &e : edges_raw) {
        if (edge_set.insert(encode(e.first, e.second)).second) {
            edges_ordered.push_back(e);
        }
    }
    std::vector<std::pair<uint32_t, uint32_t>>().swap(edges_raw);

    std::vector<ogdf::node> id2node(n, nullptr);
    C.node2name.reserve(n);
    C.name2node.reserve(n);
    for (const auto &e : edges_ordered) {
        if (!id2node[e.first]) {
            ogdf::node v = C.G.newNode();
            id2node[e.first] = v;
            std::string name = std::to_string(e.first);
            C.node2name[v] = name;
            C.name2node[std::move(name)] = v;
        }
        if (!id2node[e.second]) {
            ogdf::node v = C.G.newNode();
            id2node[e.second] = v;
            std::string name = std::to_string(e.second);
            C.node2name[v] = name;
            C.name2node[std::move(name)] = v;
        }
    }

    std::unordered_set<uint64_t> processed;
    processed.reserve(edges_ordered.size() * 2);
    for (const auto &e : edges_ordered) {
        uint64_t key = encode(e.first, e.second);
        if (!processed.insert(key).second) continue;

        uint64_t revkey = encode(e.second, e.first);
        bool has_rev = edge_set.count(revkey) > 0;

        if (has_rev) {
            processed.insert(revkey);

            ogdf::node t1 = C.G.newNode();
            ogdf::node t2 = C.G.newNode();
            C.node2name[t1] = "_trash";
            C.node2name[t2] = "_trash";

            C.G.newEdge(id2node[e.first],  t1);
            C.G.newEdge(t1,                id2node[e.second]);
            C.G.newEdge(id2node[e.second], t2);
            C.G.newEdge(t2,                id2node[e.first]);
        } else {
            C.G.newEdge(id2node[e.first], id2node[e.second]);
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
    C.ubNumNodes = N;
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

    std::vector<bool> saw_plus(N, false), saw_minus(N, false);

    C.ubOffset.assign(N + 1, 0);
    for (const auto &e : edges) {
        C.ubOffset[e.u + 1]++;
        C.ubOffset[e.v + 1]++;

        if (e.tu == (uint8_t)EdgePartType::PLUS) saw_plus[e.u] = true;
        else                                      saw_minus[e.u] = true;
        if (e.tv == (uint8_t)EdgePartType::PLUS) saw_plus[e.v] = true;
        else                                      saw_minus[e.v] = true;
    }

    for (uint32_t i = 1; i <= N; i++) {
        C.ubOffset[i] += C.ubOffset[i - 1];
    }

    C.ubEdges.resize(C.ubOffset[N]);

    std::vector<uint32_t> cursor(C.ubOffset.begin(), C.ubOffset.end());

    for (const auto &e : edges) {
        C.ubEdges[cursor[e.u]++] = {e.v, e.tu, e.tv};
        C.ubEdges[cursor[e.v]++] = {e.u, e.tv, e.tu};
    }

    C.ubIsTip.resize(N);
    size_t tip_count = 0;
    for (uint32_t i = 0; i < N; i++) {
        C.ubIsTip[i] = !(saw_plus[i] && saw_minus[i]);
        if (C.ubIsTip[i]) tip_count++;
    }

    logger::info("graph built: {} nodes, {} edges (CSR: {} adj entries), {} tips",
                 N, E, C.ubEdges.size(), tip_count);
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
        ogdf::node nSrc = getNode(lk.src, lk.orient_src);
        ogdf::node nDst = getNode(lk.dst, lk.orient_dst);
        des.push_back({(int)nSrc.index(), (int)nDst.index()});
        if (!directed_only) {
            ogdf::node nRevSrc = getNode(lk.dst, flipSign(lk.orient_dst));
            ogdf::node nRevDst = getNode(lk.src, flipSign(lk.orient_src));
            des.push_back({(int)nRevSrc.index(), (int)nRevDst.index()});
        }
    }

    std::sort(des.begin(), des.end());
    des.erase(std::unique(des.begin(), des.end()), des.end());

    std::unordered_map<int, ogdf::node> idx2n;
    for (ogdf::node v : C.G.nodes) idx2n[v.index()] = v;
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
        logger::info("GBZ parser: reading '{}'", path);
        auto bg = GBZParser::parse_file(path);
        logger::info("GBZ parser: {} segments, {} links", bg.n_nodes, bg.links.size());
        return bg;
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
            if (C.doubledUltrabubbles) {
                buildSuperbubbleGraph(bg, false);
            } else {
                buildUltrabubbleLightGraph(bg);
                return;
            }
            break;
        case Context::BubbleType::SNARL:
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

        if (C.bubbleType == Context::BubbleType::ULTRABUBBLE && !C.doubledUltrabubbles) {
            logger::info("Graph read");
            return;
        }

        C.isEntry = NodeArray<bool>(C.G, false);
        C.isExit= NodeArray<bool>(C.G, false);
        C.inDeg = NodeArray<int>(C.G, 0);
        C.outDeg= NodeArray<int>(C.G, 0);
        for (edge e : C.G.edges) {
            C.outDeg[C.G.source(e)]++;
            C.inDeg [C.G.target(e)]++;
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
    C.isExit= NodeArray<bool>(C.G, false);
    C.inDeg = NodeArray<int>(C.G, 0);
    C.outDeg= NodeArray<int>(C.G, 0);
    for (edge e : C.G.edges) {
        C.outDeg[C.G.source(e)]++;
        C.inDeg [C.G.target(e)]++;
    }
    logger::info("Graph read");
}


void drawGraph(const ogdf::Graph &G, const std::string &file)
{
    // Drawing functionality disabled - requires OGDF GraphAttributes/FMMMLayout
    (void)G; (void)file;
    return;
}


std::vector<std::pair<std::string, std::string>>
project_bubblegun_pairs_from_doubled() {
    auto& sb= ctx().superbubbles; 
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
        if (C.includeTrivial)
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
        }
        else
        {
            auto get_real_neighbors = [&](ogdf::node u, EdgePartType t)
                -> std::unordered_set<ogdf::node>
            {
                std::unordered_set<ogdf::node> result;
                C.G.forEachAdj(u, [&](node other, edge e) {
                    EdgePartType typeAtU;
                    if (C.G.source(e) == u)
                        typeAtU = C._edge2types[e].first;
                    else
                        typeAtU = C._edge2types[e].second;

                    if (typeAtU != t) return;

                    if (C.node2name.count(other) && C.node2name[other] == "_trash")
                    {
                        C.G.forEachAdj(other, [&](node real, edge) {
                            if (real != u) result.insert(real);
                        });
                    }
                    else
                    {
                        result.insert(other);
                    }
                });
                return result;
            };

            auto is_trivial_pair = [&](const std::string &s1,
                                       const std::string &s2) -> bool
            {
                if (s1.size() < 2 || s2.size() < 2) return false;

                std::string name1 = s1.substr(0, s1.size() - 1);
                std::string name2 = s2.substr(0, s2.size() - 1);
                char c1 = s1.back(), c2 = s2.back();

                if (c1 != '+' && c1 != '-') return false;
                if (c2 != '+' && c2 != '-') return false;
                if (name1 == "_trash" || name2 == "_trash") return false;

                auto it1 = C.name2node.find(name1);
                auto it2 = C.name2node.find(name2);
                if (it1 == C.name2node.end() || it2 == C.name2node.end()) return false;

                ogdf::node u = it1->second;
                ogdf::node v = it2->second;

                EdgePartType t1 = (c1 == '+') ? EdgePartType::PLUS : EdgePartType::MINUS;
                EdgePartType t2 = (c2 == '+') ? EdgePartType::PLUS : EdgePartType::MINUS;

                auto nbrs1 = get_real_neighbors(u, t1);
                auto nbrs2 = get_real_neighbors(v, t2);

                return nbrs1.size() == 1 && nbrs1.count(v) &&
                       nbrs2.size() == 1 && nbrs2.count(u);
            };

            struct PairHash
            {
                size_t operator()(const std::pair<std::string, std::string> &p) const
                {
                    size_t h1 = std::hash<std::string>{}(p.first);
                    size_t h2 = std::hash<std::string>{}(p.second);
                    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
                }
            };

            std::vector<std::pair<std::string, std::string>> filtered;
            std::unordered_set<std::pair<std::string, std::string>, PairHash> seen;

            for (auto &s : C.snarls)
            {
                for (size_t i = 0; i < s.size(); i++)
                {
                    for (size_t j = i + 1; j < s.size(); j++)
                    {
                        std::string a = s[i], b = s[j];
                        if (a > b) std::swap(a, b);

                        if (!seen.insert({a, b}).second) continue;
                        if (is_trivial_pair(a, b)) continue;

                        filtered.emplace_back(a, b);
                    }
                }
            }

            if (C.outputPath.empty())
            {
                std::cout << filtered.size() << "\n";
                for (auto &p : filtered)
                {
                    std::cout << p.first << " " << p.second << "\n";
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
                out << filtered.size() << "\n";
                for (auto &p : filtered)
                {
                    out << p.first << " " << p.second << "\n";
                }
                if (!out)
                {
                    throw std::runtime_error("Error while writing snarls to output file '" +
                                             C.outputPath + "'");
                }
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