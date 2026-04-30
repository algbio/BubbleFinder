#include "graph_io.hpp"
#include "util/context.hpp"
#include "util/timer.hpp"
#include "util/logger.hpp"
#include "util/profiling_macros.hpp"
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
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

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

    auto encodePair = [](uint32_t a, uint32_t b) -> uint64_t {
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    };

    std::unordered_set<uint64_t> seen;
    std::unordered_set<uint64_t> multis;
    seen.reserve(bg.links.size());
    for (auto& lk : bg.links) {
        uint32_t a = std::min(lk.src, lk.dst), b = std::max(lk.src, lk.dst);
        uint64_t key = encodePair(a, b);
        if (!seen.insert(key).second) {
            multis.insert(key);
        }
    }
    { std::unordered_set<uint64_t>().swap(seen); }

    auto is_multi = [&](uint32_t a, uint32_t b) -> bool {
        uint32_t lo = std::min(a, b), hi = std::max(a, b);
        return multis.count(encodePair(lo, hi)) > 0;
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


#ifdef BUBBLEFINDER_INSTRUMENT
namespace {

struct ChainAuditResult {
    uint64_t totalNodes = 0;
    uint64_t totalEdges = 0;
    uint64_t trashNodes = 0;
    uint64_t contractibleNodes = 0;
    uint64_t entryNodes = 0;
    uint64_t branchingNodes = 0;
    uint64_t isolatedNodes = 0;
    uint64_t selfLoopOnContract = 0;
    uint64_t parallelEdgeOnContract = 0;
    uint64_t numChains = 0;
    uint64_t maxChainLen = 0;
    uint64_t sumChainLen = 0;
    uint64_t chainLenHist[16] = {0};
    uint64_t chainLenP50 = 0;
    uint64_t chainLenP90 = 0;
    uint64_t chainLenP99 = 0;
    uint64_t edgesAfterContract = 0;
    uint64_t auditTimeMs = 0;
};

inline EdgePartType edgeTypeAtNode(const ogdf::Graph &G,
                                   const ogdf::EdgeArray<std::pair<EdgePartType, EdgePartType>> &e2t,
                                   ogdf::edge e,
                                   ogdf::node v)
{
    if (G.source(e) == v) return e2t[e].first;
    if (G.target(e) == v) return e2t[e].second;
    return EdgePartType::NONE;
}

inline bool isContractible(const ogdf::Graph &G,
                           const ogdf::EdgeArray<std::pair<EdgePartType, EdgePartType>> &e2t,
                           ogdf::node v)
{
    int dPlus = 0, dMinus = 0, dOther = 0;
    int total = 0;
    bool hasSelfLoop = false;
    G.forEachAdj(v, [&](ogdf::node nbr, ogdf::edge e) {
        if (nbr == v) hasSelfLoop = true;
        ++total;
        EdgePartType t = edgeTypeAtNode(G, e2t, e, v);
        if (t == EdgePartType::PLUS) ++dPlus;
        else if (t == EdgePartType::MINUS) ++dMinus;
        else ++dOther;
    });
    if (hasSelfLoop) return false;
    if (dOther != 0) return false;
    if (total != 2) return false;
    return dPlus == 1 && dMinus == 1;
}

inline ogdf::node otherEndOnSide(const ogdf::Graph &G,
                                 const ogdf::EdgeArray<std::pair<EdgePartType, EdgePartType>> &e2t,
                                 ogdf::node v,
                                 EdgePartType wantedSide,
                                 ogdf::edge &outEdge)
{
    ogdf::node res = nullptr;
    outEdge = nullptr;
    G.forEachAdj(v, [&](ogdf::node nbr, ogdf::edge e) {
        if (res != nullptr) return;
        if (edgeTypeAtNode(G, e2t, e, v) == wantedSide) {
            res = nbr;
            outEdge = e;
        }
    });
    return res;
}

void auditDeg2ContractionPotential(ChainAuditResult &R)
{
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();
    auto &C = ctx();
    const ogdf::Graph &G = C.G;
    const auto &e2t = C._edge2types;

    R.totalNodes = G.numberOfNodes();
    R.totalEdges = G.numberOfEdges();

    std::vector<bool> trash(G.numberOfNodes() + 1, false);
    for (const auto &kv : C.node2name) {
        if (kv.second == "_trash") {
            if (kv.first.idx >= 0 && static_cast<size_t>(kv.first.idx) < trash.size())
                trash[kv.first.idx] = true;
        }
    }
    for (size_t i = 0; i < trash.size(); ++i) if (trash[i]) ++R.trashNodes;

    std::vector<bool> isCtr(G.numberOfNodes() + 1, false);
    for (ogdf::node v : G.nodes) {
        size_t idx = static_cast<size_t>(v.idx);
        if (idx >= isCtr.size()) continue;
        if (idx < trash.size() && trash[idx]) continue;
        int total = 0;
        G.forEachAdj(v, [&](ogdf::node, ogdf::edge) { ++total; });
        if (total == 0) { ++R.isolatedNodes; continue; }
        if (isContractible(G, e2t, v)) {
            isCtr[idx] = true;
            ++R.contractibleNodes;
        } else {
            if (total >= 3) ++R.branchingNodes;
            else ++R.entryNodes;
        }
    }

    std::vector<bool> visited(G.numberOfNodes() + 1, false);
    std::vector<uint64_t> chainLens;
    chainLens.reserve(R.contractibleNodes / 2 + 1);

    for (ogdf::node start : G.nodes) {
        size_t sIdx = static_cast<size_t>(start.idx);
        if (sIdx >= isCtr.size()) continue;
        if (!isCtr[sIdx]) continue;
        if (visited[sIdx]) continue;

        std::vector<ogdf::node> chain;
        chain.push_back(start);
        visited[sIdx] = true;

        ogdf::edge ePlus = nullptr;
        ogdf::node curPlus = otherEndOnSide(G, e2t, start, EdgePartType::PLUS, ePlus);
        while (curPlus != nullptr) {
            size_t cIdx = static_cast<size_t>(curPlus.idx);
            if (cIdx >= isCtr.size() || !isCtr[cIdx] || visited[cIdx]) break;
            visited[cIdx] = true;
            chain.push_back(curPlus);
            ogdf::edge nextE = nullptr;
            ogdf::node nx = nullptr;
            G.forEachAdj(curPlus, [&](ogdf::node nbr, ogdf::edge e) {
                if (e == ePlus) return;
                if (nx != nullptr) return;
                nx = nbr;
                nextE = e;
            });
            if (nx == nullptr) break;
            curPlus = nx;
            ePlus = nextE;
        }

        ogdf::edge eMinus = nullptr;
        ogdf::node curMinus = otherEndOnSide(G, e2t, start, EdgePartType::MINUS, eMinus);
        while (curMinus != nullptr) {
            size_t cIdx = static_cast<size_t>(curMinus.idx);
            if (cIdx >= isCtr.size() || !isCtr[cIdx] || visited[cIdx]) break;
            visited[cIdx] = true;
            chain.push_back(curMinus);
            ogdf::edge nextE = nullptr;
            ogdf::node nx = nullptr;
            G.forEachAdj(curMinus, [&](ogdf::node nbr, ogdf::edge e) {
                if (e == eMinus) return;
                if (nx != nullptr) return;
                nx = nbr;
                nextE = e;
            });
            if (nx == nullptr) break;
            curMinus = nx;
            eMinus = nextE;
        }

        ogdf::node leftAnchor = nullptr;
        ogdf::node rightAnchor = nullptr;
        ogdf::edge eL = nullptr, eR = nullptr;
        ogdf::node leftEnd = chain.back();
        ogdf::node rightEnd = chain.front();
        if (chain.size() == 1) {
            leftEnd = start; rightEnd = start;
        } else {
            leftEnd = chain.back();
            rightEnd = chain.front();
        }
        G.forEachAdj(leftEnd, [&](ogdf::node nbr, ogdf::edge e) {
            size_t nIdx = static_cast<size_t>(nbr.idx);
            if (nIdx < isCtr.size() && isCtr[nIdx] && visited[nIdx] && nbr != leftEnd) return;
            if (nIdx >= isCtr.size() || !isCtr[nIdx]) {
                leftAnchor = nbr;
                eL = e;
            }
        });
        G.forEachAdj(rightEnd, [&](ogdf::node nbr, ogdf::edge e) {
            size_t nIdx = static_cast<size_t>(nbr.idx);
            if (nIdx < isCtr.size() && isCtr[nIdx] && visited[nIdx] && nbr != rightEnd) return;
            if (nIdx >= isCtr.size() || !isCtr[nIdx]) {
                rightAnchor = nbr;
                eR = e;
            }
        });

        if (leftAnchor != nullptr && leftAnchor == rightAnchor) {
            ++R.selfLoopOnContract;
        }

        uint64_t L = chain.size();
        chainLens.push_back(L);
        ++R.numChains;
        R.sumChainLen += L;
        if (L > R.maxChainLen) R.maxChainLen = L;
        size_t bucket = (L >= 16) ? 15 : static_cast<size_t>(L);
        R.chainLenHist[bucket] += 1;
    }

    if (!chainLens.empty()) {
        std::sort(chainLens.begin(), chainLens.end());
        auto pick = [&](double q) -> uint64_t {
            if (chainLens.empty()) return 0;
            size_t i = static_cast<size_t>(q * (chainLens.size() - 1));
            return chainLens[i];
        };
        R.chainLenP50 = pick(0.50);
        R.chainLenP90 = pick(0.90);
        R.chainLenP99 = pick(0.99);
    }

    R.edgesAfterContract = R.totalEdges;
    if (R.contractibleNodes > 0 && R.numChains > 0) {
        R.edgesAfterContract = (R.totalEdges > R.contractibleNodes)
            ? (R.totalEdges - R.contractibleNodes)
            : R.totalEdges;
    }

    auto t1 = high_resolution_clock::now();
    R.auditTimeMs = static_cast<uint64_t>(duration_cast<milliseconds>(t1 - t0).count());
}

void printAuditReport(const ChainAuditResult &R)
{
    std::ostream &os = std::cout;
    auto old_flags = os.flags();
    auto old_prec = os.precision();

    auto pct = [](double n, double d) -> double {
        return d > 0.0 ? (100.0 * n / d) : 0.0;
    };

    os << "\n => Deg 2 information\n";
    os << "  time" << R.auditTimeMs << " ms\n";
    os << "\n  Input graph\n";
    os << " nodes total: " << R.totalNodes << "\n";
    os << " edges total: " << R.totalEdges << "\n";
    os << " trash nodes: " << R.trashNodes
       << " (" << std::fixed << std::setprecision(2) << pct(R.trashNodes, R.totalNodes) << "%)\n";

    os << "\n  classification\n";
    os << " contractible (deg+=1, deg-=1): " << R.contractibleNodes
       << " (" << std::fixed << std::setprecision(2) << pct(R.contractibleNodes, R.totalNodes) << "%)\n";
    os << " branching (>=3 edges): " << R.branchingNodes
       << " (" << std::fixed << std::setprecision(2) << pct(R.branchingNodes, R.totalNodes) << "%)\n";
    os << " end / single-side: " << R.entryNodes
       << " (" << std::fixed << std::setprecision(2) << pct(R.entryNodes, R.totalNodes) << "%)\n";
    os << " isolated: " << R.isolatedNodes << "\n";

    os << "\n  Chain extraction\n";
    os << " chains found: " << R.numChains << "\n";
    if (R.numChains > 0) {
        double avg = static_cast<double>(R.sumChainLen) / static_cast<double>(R.numChains);
        os << " avg chain length: " << std::fixed << std::setprecision(2) << avg << "\n";
        os << " max chain length: " << R.maxChainLen << "\n";
        os << " p50 / p90 / p99: " << R.chainLenP50 << " / "
           << R.chainLenP90 << " / " << R.chainLenP99 << "\n";
        os << " histogram (chain len -> count):\n";
        for (size_t i = 1; i < 16; ++i) {
            if (R.chainLenHist[i] == 0) continue;
            const char *prefix = (i == 15) ? ">=15" : "    ";
            os << "      len " << prefix << " " << std::setw(4) << i << " : "
               << R.chainLenHist[i] << "\n";
        }
        os << " self loops created: " << R.selfLoopOnContract << "\n";
    }

    os << "\n  Reduction if contraction is applied \n";
    uint64_t nodesAfter = (R.totalNodes > R.contractibleNodes)
        ? (R.totalNodes - R.contractibleNodes) : R.totalNodes;
    os << " nodes:" << R.totalNodes << " -> " << nodesAfter
       << "  (-" << std::fixed << std::setprecision(2) << pct(R.contractibleNodes, R.totalNodes) << "%)\n";
    os << " edges:" << R.totalEdges << " -> " << R.edgesAfterContract
       << "  (-" << std::fixed << std::setprecision(2) << pct(R.totalEdges - R.edgesAfterContract, R.totalEdges) << "%)\n";

    if (R.totalNodes > 0 && nodesAfter > 0) {
        double speedup = static_cast<double>(R.totalNodes) / static_cast<double>(nodesAfter);
        os << " raw size ratio: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    }

    os.precision(old_prec);
    os.flags(old_flags);
}

}
#endif

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

        BF_INSTR(
        if (C.bubbleType == Context::BubbleType::SNARL) {
            ChainAuditResult R;
            auditDeg2ContractionPotential(R);
            printAuditReport(R);
        }
        )

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


namespace {

constexpr size_t kIoChunkHighWater = 64ull * 1024ull * 1024ull;  // 64 MiB

inline void flushStringBuf(std::ostream &out, std::string &buf) {
    if (!buf.empty()) {
        out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        buf.clear();
    }
}

template <typename SnarlSet>
void writeAllSnarls_buffered(std::ostream &out, const SnarlSet &snarls)
{
    std::string buf;
    buf.reserve(kIoChunkHighWater + 4096);

    buf.append(std::to_string(snarls.size()));
    buf.push_back('\n');

    for (const auto &s : snarls) {
        for (const auto &v : s) {
            buf.append(v);
            buf.push_back(' ');
        }
        buf.push_back('\n');
        if (buf.size() >= kIoChunkHighWater) {
            flushStringBuf(out, buf);
        }
    }
    flushStringBuf(out, buf);
}

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
                writeAllSnarls_buffered(std::cout, C.snarls);
                if (!std::cout) {
                    throw std::runtime_error("Error while writing snarls to standard output");
                }
            }
            else
            {
                std::ofstream out(C.outputPath, std::ios::out | std::ios::binary);
                if (!out) {
                    throw std::runtime_error("Failed to open output file '" +
                                             C.outputPath + "' for writing");
                }
                writeAllSnarls_buffered(out, C.snarls);
                if (!out) {
                    throw std::runtime_error("Error while writing snarls to output file '" +
                                             C.outputPath + "'");
                }
            }
        }
        else
        {
            struct RealNbrKey {
                uint32_t node_idx;
                uint8_t sign;  // 0 = PLUS, 1 = MINUS
                bool operator==(const RealNbrKey &o) const noexcept {
                    return node_idx == o.node_idx && sign == o.sign;
                }
            };
            struct RealNbrKeyHash {
                size_t operator()(const RealNbrKey &k) const noexcept {
                    return (static_cast<size_t>(k.node_idx) << 1) ^ k.sign;
                }
            };
            std::unordered_map<RealNbrKey, ogdf::node, RealNbrKeyHash> uniqueRealNbr;
            uniqueRealNbr.reserve(C.G.numberOfNodes() * 2);

            size_t max_idx = 0;
            for (ogdf::node n : C.G.nodes) {
                if (static_cast<size_t>(n.idx) > max_idx) max_idx = static_cast<size_t>(n.idx);
            }
            std::vector<bool> is_trash(max_idx + 1, false);
            for (const auto &kv : C.node2name) {
                if (kv.second == "_trash") {
                    is_trash[kv.first.idx] = true;
                }
            }

            for (ogdf::node u : C.G.nodes) {
                ogdf::node nbrPlus{nullptr};
                ogdf::node nbrMinus{nullptr};
                int countPlus = 0, countMinus = 0;
                C.G.forEachAdj(u, [&](ogdf::node other, ogdf::edge e) {
                    EdgePartType typeAtU = (C.G.source(e) == u)
                        ? C._edge2types[e].first
                        : C._edge2types[e].second;
                    int *cnt;
                    ogdf::node *slot;
                    if (typeAtU == EdgePartType::PLUS) { cnt = &countPlus;  slot = &nbrPlus;  }
                    else if (typeAtU == EdgePartType::MINUS) { cnt = &countMinus; slot = &nbrMinus; }
                    else return;

                    if (*cnt > 1) return;  // already disqualified

                    if (is_trash[other.idx]) {
                        C.G.forEachAdj(other, [&](ogdf::node real, ogdf::edge) {
                            if (real == u) return;
                            if (*cnt > 1) return;
                            if (*cnt == 0 || *slot == real) {
                                *slot = real;
                                if (*cnt == 0) (*cnt)++;
                            } else {
                                (*cnt)++;  // becomes 2 -> disqualified
                            }
                        });
                    } else {
                        if (*cnt == 0 || *slot == other) {
                            *slot = other;
                            if (*cnt == 0) (*cnt)++;
                        } else {
                            (*cnt)++;
                        }
                    }
                });
                if (countPlus  == 1) uniqueRealNbr[{static_cast<uint32_t>(u.idx), 0}] = nbrPlus;
                if (countMinus == 1) uniqueRealNbr[{static_cast<uint32_t>(u.idx), 1}] = nbrMinus;
            }


            std::vector<std::pair<uint32_t, uint8_t>> snarl_nodes;

            std::unordered_set<uint64_t> seen_num;
            seen_num.reserve(20'000'000);

            struct PairHash {
                size_t operator()(const std::pair<std::string, std::string> &p) const noexcept {
                    size_t h1 = std::hash<std::string>{}(p.first);
                    size_t h2 = std::hash<std::string>{}(p.second);
                    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
                }
            };
            std::unordered_set<std::pair<std::string, std::string>, PairHash> seen_str;

            std::string out_buf;
            out_buf.reserve(400ull * 1024ull * 1024ull);
            size_t pair_count = 0;

            auto pack_key = [](uint32_t a_idx, uint8_t a_sign,
                                uint32_t b_idx, uint8_t b_sign) -> uint64_t {
                uint64_t ka = (static_cast<uint64_t>(a_idx) << 1) | a_sign;
                uint64_t kb = (static_cast<uint64_t>(b_idx) << 1) | b_sign;
                if (ka > kb) std::swap(ka, kb);
                return (ka << 32) | kb;
            };

            for (const auto &s : C.snarls)
            {
                snarl_nodes.clear();
                snarl_nodes.reserve(s.size());
                for (const auto &str : s) {
                    uint8_t sign = 255;
                    uint32_t idx = 0;
                    if (str.size() >= 2) {
                        char c = str.back();
                        if (c == '+' || c == '-') {
                            std::string name(str.data(), str.size() - 1);
                            if (name != "_trash") {
                                auto it = C.name2node.find(name);
                                if (it != C.name2node.end()) {
                                    idx = static_cast<uint32_t>(it->second.idx);
                                    sign = (c == '+') ? 0 : 1;
                                }
                            }
                        }
                    }
                    snarl_nodes.push_back({idx, sign});
                }

                const size_t n = s.size();
                for (size_t i = 0; i < n; i++)
                {
                    auto [iu, su] = snarl_nodes[i];
                    for (size_t j = i + 1; j < n; j++)
                    {
                        auto [iv, sv] = snarl_nodes[j];

                        const std::string *pa = &s[i];
                        const std::string *pb = &s[j];
                        if (*pa > *pb) std::swap(pa, pb);

                        bool is_trivial = false;

                        if (su != 255 && sv != 255) {
                            uint64_t key = pack_key(iu, su, iv, sv);
                            if (!seen_num.insert(key).second) continue;

                            auto it_u = uniqueRealNbr.find({iu, su});
                            if (it_u != uniqueRealNbr.end() &&
                                static_cast<uint32_t>(it_u->second.idx) == iv) {
                                auto it_v = uniqueRealNbr.find({iv, sv});
                                if (it_v != uniqueRealNbr.end() &&
                                    static_cast<uint32_t>(it_v->second.idx) == iu) {
                                    is_trivial = true;
                                }
                            }
                        } else {
                            if (!seen_str.insert({*pa, *pb}).second) continue;
                            // is_trivial is false for non-resolvable names (matches original)
                        }

                        if (is_trivial) continue;

                        out_buf.append(*pa);
                        out_buf.push_back(' ');
                        out_buf.append(*pb);
                        out_buf.push_back('\n');
                        pair_count++;
                    }
                }
            }

            // Free dedup memory before writing (out_buf alone is ~340 MB).
            std::unordered_set<uint64_t>().swap(seen_num);
            std::unordered_set<std::pair<std::string, std::string>, PairHash>().swap(seen_str);
            std::vector<std::pair<uint32_t, uint8_t>>().swap(snarl_nodes);

            auto writeStreamedOutput = [&](std::ostream &os) {
                std::string header = std::to_string(pair_count) + "\n";
                os.write(header.data(), static_cast<std::streamsize>(header.size()));
                os.write(out_buf.data(), static_cast<std::streamsize>(out_buf.size()));
            };

            if (C.outputPath.empty())
            {
                writeStreamedOutput(std::cout);
                if (!std::cout) {
                    throw std::runtime_error("Error while writing snarls to standard output");
                }
            }
            else
            {
                std::ofstream out(C.outputPath, std::ios::out | std::ios::binary);
                if (!out) {
                    throw std::runtime_error("Failed to open output file '" +
                                             C.outputPath + "' for writing");
                }
                writeStreamedOutput(out);
                if (!out) {
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