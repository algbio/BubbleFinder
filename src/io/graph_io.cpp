#include "graph_io.hpp"
#include "util/context.hpp"
#include "util/timer.hpp"
#include "util/logger.hpp"
#include "util/profiling_macros.hpp"
#include "gfa_parser.hpp"

#include "gbz_parser.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <regex>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <charconv>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>
#if defined(_OPENMP) && defined(__GLIBCXX__)
#include <parallel/algorithm>
#endif

using namespace spqr_compat;

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

    std::vector<spqr_compat::node> id2node(n, nullptr);
    C.node2name.reserve(n);
    C.name2node.reserve(n);
    for (const auto &e : edges_ordered) {
        if (!id2node[e.first]) {
            spqr_compat::node v = C.G.newNode();
            id2node[e.first] = v;
            std::string name = std::to_string(e.first);
            C.node2name[v] = name;
            C.name2node[std::move(name)] = v;
        }
        if (!id2node[e.second]) {
            spqr_compat::node v = C.G.newNode();
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

            spqr_compat::node t1 = C.G.newNode();
            spqr_compat::node t2 = C.G.newNode();
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

void ensureStringNodeNames(BiGraph& bg)
{
    if (bg.node_names.size() == bg.n_nodes) return;
    if (bg.numeric_node_names.size() != bg.n_nodes ||
        bg.numeric_node_name_valid.size() != bg.n_nodes) return;
    bg.node_names.resize(bg.n_nodes);
    size_t sparse_i = 0;
    for (uint32_t i = 0; i < bg.n_nodes; ++i) {
        if (bg.numeric_node_name_valid[i]) {
            bg.node_names[i] = std::to_string(bg.numeric_node_names[i]);
        } else {
            while (sparse_i < bg.string_node_names.size() &&
                   bg.string_node_names[sparse_i].first < i) {
                ++sparse_i;
            }
            if (sparse_i < bg.string_node_names.size() &&
                bg.string_node_names[sparse_i].first == i) {
                bg.node_names[i] = bg.string_node_names[sparse_i].second;
            }
        }
    }
    std::vector<uint64_t>().swap(bg.numeric_node_names);
    std::vector<uint8_t>().swap(bg.numeric_node_name_valid);
    std::vector<std::pair<uint32_t, std::string>>().swap(bg.string_node_names);
}

std::string biGraphNodeName(const BiGraph& bg, uint32_t i)
{
    if (bg.node_names.size() == bg.n_nodes) {
        return bg.node_names[i];
    }
    if (bg.numeric_node_names.size() == bg.n_nodes &&
        bg.numeric_node_name_valid.size() == bg.n_nodes &&
        bg.numeric_node_name_valid[i]) {
        return std::to_string(bg.numeric_node_names[i]);
    }
    auto it = std::lower_bound(
        bg.string_node_names.begin(), bg.string_node_names.end(), i,
        [](const std::pair<uint32_t, std::string>& item, uint32_t value) {
            return item.first < value;
        });
    if (it != bg.string_node_names.end() && it->first == i) {
        return it->second;
    }
    return std::to_string(i);
}

bool useCompactSnarlNameTables(const Context &C)
{
    return C.bubbleType == Context::BubbleType::SNARL &&
           C.includeTrivial &&
           C.spCompressMode == Context::SpCompressMode::MacroDirectDebug &&
           std::getenv("BF_DISABLE_FAST_SNARL_OUTPUT") == nullptr &&
           std::getenv("BF_DEBUG_TAGGED_SNARLS") == nullptr &&
           std::getenv("BF_FORCE_SNARL_NAME_INDEX") == nullptr &&
           std::getenv("BF_FORCE_SNARL_NODE2NAME") == nullptr;
}

void ensureCompactNodeSlot(Context &C, spqr_compat::node v)
{
    const size_t idx = static_cast<size_t>(v.idx);
    if (C.nodeNumericNamesByIndex.empty() && idx >= C.nodeNamesByIndex.size()) {
        C.nodeNamesByIndex.resize(idx + 1);
    }
    if (!C.nodeNumericNamesByIndex.empty() && idx >= C.nodeNumericNamesByIndex.size()) {
        C.nodeNumericNamesByIndex.resize(idx + 1);
    }
    if (!C.nodeNumericNameValidByIndex.empty() && idx >= C.nodeNumericNameValidByIndex.size()) {
        C.nodeNumericNameValidByIndex.resize(idx + 1, 0);
    }
    if (idx >= C.isTrashNodeByIndex.size()) {
        C.isTrashNodeByIndex.resize(idx + 1, 0);
    }
}

void setCompactNodeName(Context &C, spqr_compat::node v, std::string name)
{
    ensureCompactNodeSlot(C, v);
    const size_t idx = static_cast<size_t>(v.idx);
    if (idx >= C.nodeNamesByIndex.size()) {
        C.nodeNamesByIndex.resize(idx + 1);
    }
    C.isTrashNodeByIndex[idx] = (name == "_trash") ? 1 : 0;
    if (!C.nodeNumericNamesByIndex.empty() && idx < C.nodeNumericNamesByIndex.size()) {
        C.nodeNumericNamesByIndex[idx] = 0;
    }
    if (!C.nodeNumericNameValidByIndex.empty() && idx < C.nodeNumericNameValidByIndex.size()) {
        C.nodeNumericNameValidByIndex[idx] = 0;
    }
    if (C.isTrashNodeByIndex[idx]) {
        C.nodeNamesByIndex[idx].clear();
    } else {
        C.nodeNamesByIndex[idx] = std::move(name);
    }
}

void setCompactNumericNodeName(Context &C, spqr_compat::node v, uint64_t name)
{
    ensureCompactNodeSlot(C, v);
    const size_t idx = static_cast<size_t>(v.idx);
    if (idx < C.nodeNamesByIndex.size()) {
        C.nodeNamesByIndex[idx].clear();
    }
    C.nodeNumericNamesByIndex[idx] = name;
    if (!C.nodeNumericNameValidByIndex.empty()) {
        C.nodeNumericNameValidByIndex[idx] = 1;
    }
    C.isTrashNodeByIndex[idx] = 0;
}

void setCompactTrashNode(Context &C, spqr_compat::node v)
{
    ensureCompactNodeSlot(C, v);
    const size_t idx = static_cast<size_t>(v.idx);
    if (idx < C.nodeNamesByIndex.size()) {
        C.nodeNamesByIndex[idx].clear();
    }
    if (!C.nodeNumericNamesByIndex.empty() && idx < C.nodeNumericNamesByIndex.size()) {
        C.nodeNumericNamesByIndex[idx] = 0;
    }
    if (!C.nodeNumericNameValidByIndex.empty() && idx < C.nodeNumericNameValidByIndex.size()) {
        C.nodeNumericNameValidByIndex[idx] = 0;
    }
    C.isTrashNodeByIndex[idx] = 1;
}

std::vector<spqr_compat::node> createNodes(BiGraph& bg, size_t extra_nodes = 0) {
    auto &C = ctx();
    std::vector<spqr_compat::node> id2node(bg.n_nodes);
    const bool compact_names = useCompactSnarlNameTables(C);
    const bool bg_has_numeric_names =
        bg.numeric_node_names.size() == bg.n_nodes &&
        bg.numeric_node_name_valid.size() == bg.n_nodes;
    const bool build_name_index =
        !(C.bubbleType == Context::BubbleType::SNARL &&
          C.includeTrivial &&
          std::getenv("BF_FORCE_SNARL_NAME_INDEX") == nullptr);
    if (build_name_index) {
        C.name2node.reserve(bg.n_nodes);
    }
    if (compact_names) {
        C.nodeNamesByIndex.clear();
        C.nodeNumericNamesByIndex.clear();
        C.nodeNumericNameValidByIndex.clear();
        C.sparseNodeNamesByIndex.clear();
        C.isTrashNodeByIndex.clear();
        if (bg_has_numeric_names) {
            C.nodeNumericNamesByIndex.resize(static_cast<size_t>(bg.n_nodes) + extra_nodes);
            C.nodeNumericNameValidByIndex.resize(static_cast<size_t>(bg.n_nodes) + extra_nodes, 0);
            C.sparseNodeNamesByIndex.reserve(bg.string_node_names.size());
        } else {
            C.nodeNamesByIndex.resize(static_cast<size_t>(bg.n_nodes) + extra_nodes);
        }
        C.isTrashNodeByIndex.resize(static_cast<size_t>(bg.n_nodes) + extra_nodes, 0);
    } else {
        C.node2name.reserve(static_cast<size_t>(bg.n_nodes) + extra_nodes);
    }
    spqr_compat::node first = C.G.newNodes(bg.n_nodes);
    for (uint32_t i = 0; i < bg.n_nodes; ++i) {
        spqr_compat::node v(first.index() + i);
        id2node[i] = v;
        if (compact_names) {
            if (bg_has_numeric_names) {
                if (bg.numeric_node_name_valid[i]) {
                    setCompactNumericNodeName(C, v, bg.numeric_node_names[i]);
                }
            } else {
                setCompactNodeName(C, v, std::move(bg.node_names[i]));
            }
        } else {
            std::string name = bg_has_numeric_names
                ? biGraphNodeName(bg, i)
                : std::move(bg.node_names[i]);
            auto it = C.node2name.emplace(v, std::move(name)).first;
            if (build_name_index) {
                C.name2node.emplace(it->second, v);
            }
        }
    }
    if (compact_names && bg_has_numeric_names && !bg.string_node_names.empty()) {
        for (auto &item : bg.string_node_names) {
            spqr_compat::node v(first.index() + item.first);
            C.sparseNodeNamesByIndex.emplace(static_cast<uint32_t>(v.idx),
                                             std::move(item.second));
        }
    }
    std::vector<std::string>().swap(bg.node_names);
    std::vector<uint64_t>().swap(bg.numeric_node_names);
    std::vector<uint8_t>().swap(bg.numeric_node_name_valid);
    std::vector<std::pair<uint32_t, std::string>>().swap(bg.string_node_names);
    return id2node;
}


void buildSnarlGraph(BiGraph& bg) {
    auto &C = ctx();
    auto encodePair = [](uint32_t a, uint32_t b) -> uint64_t {
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    };
    const size_t link_count = bg.links.size();

    std::vector<uint64_t> pair_keys;
    pair_keys.resize(link_count);
    #pragma omp parallel for schedule(static) if(pair_keys.size() > 100000)
    for (int64_t i = 0; i < static_cast<int64_t>(link_count); ++i) {
        const size_t idx = static_cast<size_t>(i);
        const auto& lk = bg.links[idx];
        const uint32_t src = lk.src;
        const uint32_t dst = lk.dst;
        uint32_t a = std::min(src, dst), b = std::max(src, dst);
        pair_keys[static_cast<size_t>(i)] = encodePair(a, b);
    }

#if defined(_OPENMP) && defined(__GLIBCXX__)
    __gnu_parallel::sort(pair_keys.begin(), pair_keys.end());
#else
    std::sort(pair_keys.begin(), pair_keys.end());
#endif
    std::vector<uint64_t> multis;
    for (size_t i = 1; i < pair_keys.size(); ++i) {
        if (pair_keys[i] == pair_keys[i - 1] &&
            (multis.empty() || multis.back() != pair_keys[i])) {
            multis.push_back(pair_keys[i]);
        }
    }
    std::vector<uint64_t>().swap(pair_keys);

    auto is_multi = [&](uint32_t a, uint32_t b) -> bool {
        uint32_t lo = std::min(a, b), hi = std::max(a, b);
        return std::binary_search(multis.begin(), multis.end(), encodePair(lo, hi));
    };

    const size_t worker_count =
        (C.threads > 1 && link_count > 100000)
            ? std::min<size_t>(static_cast<size_t>(C.threads), link_count)
            : 1;

    std::vector<uint8_t> link_is_multi;
    std::vector<size_t> chunk_multi(worker_count, 0);
    std::vector<size_t> chunk_out_base(worker_count, 0);
    std::vector<size_t> chunk_mid_base(worker_count, 0);

    if (!multis.empty()) {
        link_is_multi.resize(link_count, 0);
        #pragma omp parallel for schedule(static) if(worker_count > 1)
        for (int64_t tid_i = 0; tid_i < static_cast<int64_t>(worker_count); ++tid_i) {
            const size_t tid = static_cast<size_t>(tid_i);
            const size_t begin = (link_count * tid) / worker_count;
            const size_t end = (link_count * (tid + 1)) / worker_count;
            size_t local_multi = 0;
            for (size_t i = begin; i < end; ++i) {
                const auto& lk = bg.links[i];
                const uint32_t src = lk.src;
                const uint32_t dst = lk.dst;
                const bool multi = is_multi(src, dst);
                link_is_multi[i] = static_cast<uint8_t>(multi);
                local_multi += multi ? 1u : 0u;
            }
            chunk_multi[tid] = local_multi;
        }
    }

    size_t multi_links = 0;
    size_t out_edges = 0;
    for (size_t tid = 0; tid < worker_count; ++tid) {
        const size_t begin = (link_count * tid) / worker_count;
        const size_t end = (link_count * (tid + 1)) / worker_count;
        chunk_out_base[tid] = out_edges;
        chunk_mid_base[tid] = multi_links;
        out_edges += (end - begin) + chunk_multi[tid];
        multi_links += chunk_multi[tid];
    }

    auto id2node = createNodes(bg, multi_links);
    spqr_compat::node first_mid = C.G.newNodes(static_cast<uint32_t>(multi_links));
    const bool compact_names = useCompactSnarlNameTables(C);
    for (size_t i = 0; i < multi_links; ++i) {
        spqr_compat::node mid(first_mid.index() + static_cast<uint32_t>(i));
        if (compact_names) {
            setCompactTrashNode(C, mid);
        } else {
            C.node2name[mid] = "_trash";
        }
    }

    std::vector<uint32_t> endpoints(out_edges * 2);
    std::vector<uint8_t> edge_types(out_edges);

    auto packType = [](EdgePartType a, EdgePartType b) -> uint8_t {
        return (static_cast<uint8_t>(a) << 2) | static_cast<uint8_t>(b);
    };

    #pragma omp parallel for schedule(static) if(worker_count > 1)
    for (int64_t tid_i = 0; tid_i < static_cast<int64_t>(worker_count); ++tid_i) {
        const size_t tid = static_cast<size_t>(tid_i);
        const size_t begin = (link_count * tid) / worker_count;
        const size_t end = (link_count * (tid + 1)) / worker_count;
        size_t out_i = chunk_out_base[tid];
        size_t mid_i = chunk_mid_base[tid];

        for (size_t i = begin; i < end; ++i) {
            const auto& lk = bg.links[i];
            EdgePartType t1 = charToType(lk.orient_src);
            EdgePartType t2 = charToType(flipSign(lk.orient_dst));
            uint32_t u = lk.src;
            uint32_t v = lk.dst;
            if (u > v) { std::swap(u, v); std::swap(t1, t2); }

            const bool multi = !link_is_multi.empty() && link_is_multi[i] != 0;
            if (!multi) {
                endpoints[2 * out_i] = id2node[u].index();
                endpoints[2 * out_i + 1] = id2node[v].index();
                edge_types[out_i] = packType(t1, t2);
                ++out_i;
            } else {
                spqr_compat::node mid(first_mid.index() + static_cast<uint32_t>(mid_i++));
                endpoints[2 * out_i] = id2node[u].index();
                endpoints[2 * out_i + 1] = mid.index();
                edge_types[out_i] = packType(t1, EdgePartType::PLUS);
                ++out_i;

                endpoints[2 * out_i] = mid.index();
                endpoints[2 * out_i + 1] = id2node[v].index();
                edge_types[out_i] = packType(EdgePartType::PLUS, t2);
                ++out_i;
            }
        }
    }

    spqr_compat::edge first_edge = edge_types.empty()
                                ? spqr_compat::edge(C.G.numberOfEdges())
                                : C.G.newEdgesBatchFlat(endpoints.data(), static_cast<uint32_t>(edge_types.size()));
    C._edge2types.init(C.G, std::make_pair(EdgePartType::NONE, EdgePartType::NONE));
    #pragma omp parallel for schedule(static) if(C.threads > 1 && edge_types.size() > 100000)
    for (int64_t i_i = 0; i_i < static_cast<int64_t>(edge_types.size()); ++i_i) {
        const size_t i = static_cast<size_t>(i_i);
        uint8_t t = edge_types[i];
        C._edge2types[spqr_compat::edge(first_edge.index() + static_cast<uint32_t>(i))] = {
            static_cast<EdgePartType>(t >> 2),
            static_cast<EdgePartType>(t & 3)
        };
    }
    std::vector<BiLink>().swap(bg.links);
}

void buildUltrabubbleLightGraph(BiGraph& bg) {
    auto &C = ctx();
    ensureStringNodeNames(bg);
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
    ensureStringNodeNames(bg);
    C.name2node.reserve(bg.n_nodes * 2);
    std::vector<spqr_compat::node> id2plus(bg.n_nodes), id2minus(bg.n_nodes);

    for (uint32_t i = 0; i < bg.n_nodes; ++i) {
        std::string pn = bg.node_names[i] + "+", mn = bg.node_names[i] + "-";
        spqr_compat::node vp = C.G.newNode(), vm = C.G.newNode();
        id2plus[i] = vp; id2minus[i] = vm;
        C.node2name[vp] = pn; C.node2name[vm] = mn;
        C.name2node[pn] = vp; C.name2node[mn] = vm;
    }

    auto getNode = [&](uint32_t id, char o) -> spqr_compat::node {
        return (o == '+') ? id2plus[id] : id2minus[id];
    };

    struct DE { int u, v; bool operator<(const DE& o) const { return u!=o.u ? u<o.u : v<o.v; }
                          bool operator==(const DE& o) const { return u==o.u && v==o.v; } };
    std::vector<DE> des;
    des.reserve(directed_only ? bg.links.size() : bg.links.size() * 2);

    for (auto& lk : bg.links) {
        spqr_compat::node nSrc = getNode(lk.src, lk.orient_src);
        spqr_compat::node nDst = getNode(lk.dst, lk.orient_dst);
        des.push_back({(int)nSrc.index(), (int)nDst.index()});
        if (!directed_only) {
            spqr_compat::node nRevSrc = getNode(lk.dst, flipSign(lk.orient_dst));
            spqr_compat::node nRevDst = getNode(lk.src, flipSign(lk.orient_src));
            des.push_back({(int)nRevSrc.index(), (int)nRevDst.index()});
        }
    }

    std::sort(des.begin(), des.end());
    des.erase(std::unique(des.begin(), des.end()), des.end());

    std::unordered_map<int, spqr_compat::node> idx2n;
    for (spqr_compat::node v : C.G.nodes) idx2n[v.index()] = v;
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
        auto bg = GBZParser::parse_file(path, threads);
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

    const bool gfaReadStats = (std::getenv("BF_GFA_READ_STATS") != nullptr);
    const auto parseStart = std::chrono::steady_clock::now();
    auto bg = parse_graph_input(C.graphPath, (int)C.threads);
    const auto parseEnd = std::chrono::steady_clock::now();
    if (bg.n_nodes == 0) { logger::info("Empty graph"); return; }

    const auto buildStart = std::chrono::steady_clock::now();
    switch (C.bubbleType) {
        case Context::BubbleType::ULTRABUBBLE:
            if (C.doubledUltrabubbles) {
                buildSuperbubbleGraph(bg, false);
            } else {
                buildUltrabubbleLightGraph(bg);
                if (gfaReadStats) {
                    const auto buildEnd = std::chrono::steady_clock::now();
                    auto parseMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                        parseEnd - parseStart).count();
                    auto buildMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                        buildEnd - buildStart).count();
                    std::cerr << "[gfa_read] parse_ms=" << parseMs
                              << " build_ms=" << buildMs << "\n";
                }
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
    const auto buildEnd = std::chrono::steady_clock::now();
    if (gfaReadStats) {
        auto parseMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            parseEnd - parseStart).count();
        auto buildMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            buildEnd - buildStart).count();
        std::cerr << "[gfa_read] parse_ms=" << parseMs
                  << " build_ms=" << buildMs << "\n";
    }

    logger::info("spqr-rust graph built: {} nodes, {} edges", C.G.numberOfNodes(), C.G.numberOfEdges());
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

inline EdgePartType edgeTypeAtNode(const spqr_compat::Graph &G,
                                   const spqr_compat::EdgeArray<std::pair<EdgePartType, EdgePartType>> &e2t,
                                   spqr_compat::edge e,
                                   spqr_compat::node v)
{
    if (G.source(e) == v) return e2t[e].first;
    if (G.target(e) == v) return e2t[e].second;
    return EdgePartType::NONE;
}

inline bool isContractible(const spqr_compat::Graph &G,
                           const spqr_compat::EdgeArray<std::pair<EdgePartType, EdgePartType>> &e2t,
                           spqr_compat::node v)
{
    int dPlus = 0, dMinus = 0, dOther = 0;
    int total = 0;
    bool hasSelfLoop = false;
    G.forEachAdj(v, [&](spqr_compat::node nbr, spqr_compat::edge e) {
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

inline spqr_compat::node otherEndOnSide(const spqr_compat::Graph &G,
                                 const spqr_compat::EdgeArray<std::pair<EdgePartType, EdgePartType>> &e2t,
                                 spqr_compat::node v,
                                 EdgePartType wantedSide,
                                 spqr_compat::edge &outEdge)
{
    spqr_compat::node res = nullptr;
    outEdge = nullptr;
    G.forEachAdj(v, [&](spqr_compat::node nbr, spqr_compat::edge e) {
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
    const spqr_compat::Graph &G = C.G;
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
    for (spqr_compat::node v : G.nodes) {
        size_t idx = static_cast<size_t>(v.idx);
        if (idx >= isCtr.size()) continue;
        if (idx < trash.size() && trash[idx]) continue;
        int total = 0;
        G.forEachAdj(v, [&](spqr_compat::node, spqr_compat::edge) { ++total; });
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

    for (spqr_compat::node start : G.nodes) {
        size_t sIdx = static_cast<size_t>(start.idx);
        if (sIdx >= isCtr.size()) continue;
        if (!isCtr[sIdx]) continue;
        if (visited[sIdx]) continue;

        std::vector<spqr_compat::node> chain;
        chain.push_back(start);
        visited[sIdx] = true;

        spqr_compat::edge ePlus = nullptr;
        spqr_compat::node curPlus = otherEndOnSide(G, e2t, start, EdgePartType::PLUS, ePlus);
        while (curPlus != nullptr) {
            size_t cIdx = static_cast<size_t>(curPlus.idx);
            if (cIdx >= isCtr.size() || !isCtr[cIdx] || visited[cIdx]) break;
            visited[cIdx] = true;
            chain.push_back(curPlus);
            spqr_compat::edge nextE = nullptr;
            spqr_compat::node nx = nullptr;
            G.forEachAdj(curPlus, [&](spqr_compat::node nbr, spqr_compat::edge e) {
                if (e == ePlus) return;
                if (nx != nullptr) return;
                nx = nbr;
                nextE = e;
            });
            if (nx == nullptr) break;
            curPlus = nx;
            ePlus = nextE;
        }

        spqr_compat::edge eMinus = nullptr;
        spqr_compat::node curMinus = otherEndOnSide(G, e2t, start, EdgePartType::MINUS, eMinus);
        while (curMinus != nullptr) {
            size_t cIdx = static_cast<size_t>(curMinus.idx);
            if (cIdx >= isCtr.size() || !isCtr[cIdx] || visited[cIdx]) break;
            visited[cIdx] = true;
            chain.push_back(curMinus);
            spqr_compat::edge nextE = nullptr;
            spqr_compat::node nx = nullptr;
            G.forEachAdj(curMinus, [&](spqr_compat::node nbr, spqr_compat::edge e) {
                if (e == eMinus) return;
                if (nx != nullptr) return;
                nx = nbr;
                nextE = e;
            });
            if (nx == nullptr) break;
            curMinus = nx;
            eMinus = nextE;
        }

        spqr_compat::node leftAnchor = nullptr;
        spqr_compat::node rightAnchor = nullptr;
        spqr_compat::edge eL = nullptr, eR = nullptr;
        spqr_compat::node leftEnd = chain.back();
        spqr_compat::node rightEnd = chain.front();
        if (chain.size() == 1) {
            leftEnd = start; rightEnd = start;
        } else {
            leftEnd = chain.back();
            rightEnd = chain.front();
        }
        G.forEachAdj(leftEnd, [&](spqr_compat::node nbr, spqr_compat::edge e) {
            size_t nIdx = static_cast<size_t>(nbr.idx);
            if (nIdx < isCtr.size() && isCtr[nIdx] && visited[nIdx] && nbr != leftEnd) return;
            if (nIdx >= isCtr.size() || !isCtr[nIdx]) {
                leftAnchor = nbr;
                eL = e;
            }
        });
        G.forEachAdj(rightEnd, [&](spqr_compat::node nbr, spqr_compat::edge e) {
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


void drawGraph(const spqr_compat::Graph &G, const std::string &file)
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

struct ChainEdge {
    uint64_t endpoint[2];
    size_t next[2];
};

struct ChainOcc {
    uint64_t key;
    size_t edge;
    uint8_t side;
};

void compactEndpointPairChains(std::vector<std::pair<uint64_t, uint64_t>> &pairs)
{
    if (pairs.size() < 2) return;

    const size_t none = std::numeric_limits<size_t>::max();
    std::vector<ChainEdge> edges;
    edges.reserve(pairs.size());
    std::vector<ChainOcc> occ;
    occ.reserve(pairs.size() * 2);

    for (const auto &p : pairs) {
        size_t idx = edges.size();
        edges.push_back({{p.first, p.second}, {none, none}});
        occ.push_back({p.first, idx, 0});
        occ.push_back({p.second, idx, 1});
    }

    std::sort(occ.begin(), occ.end(), [](const ChainOcc &a, const ChainOcc &b) {
        if (a.key != b.key) return a.key < b.key;
        if (a.edge != b.edge) return a.edge < b.edge;
        return a.side < b.side;
    });

    auto uniqueOcc = [&](uint64_t key) -> std::pair<size_t, uint8_t> {
        auto it = std::lower_bound(occ.begin(), occ.end(), key,
                                   [](const ChainOcc &a, uint64_t b) {
                                       return a.key < b;
                                   });
        if (it == occ.end() || it->key != key) return {none, 0};
        auto next = it;
        ++next;
        if (next != occ.end() && next->key == key) return {none, 0};
        return {it->edge, it->side};
    };

    for (size_t i = 0; i < edges.size(); ++i) {
        for (uint8_t side = 0; side < 2; ++side) {
            auto [other, other_side] = uniqueOcc(edges[i].endpoint[side] ^ 1ull);
            if (other != none && other != i && edges[other].endpoint[other_side] == (edges[i].endpoint[side] ^ 1ull)) {
                edges[i].next[side] = other;
            }
        }
    }

    auto degree = [&](size_t i) {
        return (edges[i].next[0] != none ? 1 : 0) +
               (edges[i].next[1] != none ? 1 : 0);
    };

    std::vector<uint8_t> seen(edges.size(), 0);
    std::vector<std::pair<uint64_t, uint64_t>> out;
    out.reserve(pairs.size());

    for (size_t i = 0; i < edges.size(); ++i) {
        if (seen[i] || degree(i) != 1) continue;

        size_t cur = i;
        uint8_t entry = edges[cur].next[0] == none ? 0 : 1;
        uint64_t first = edges[cur].endpoint[entry];
        uint64_t last = edges[cur].endpoint[entry ^ 1u];

        while (true) {
            seen[cur] = 1;
            uint8_t exit_side = entry ^ 1u;
            size_t nxt = edges[cur].next[exit_side];
            if (nxt == none || seen[nxt]) {
                last = edges[cur].endpoint[exit_side];
                break;
            }
            int next_entry = edges[nxt].next[0] == cur ? 0 :
                             edges[nxt].next[1] == cur ? 1 : -1;
            if (next_entry < 0) {
                last = edges[cur].endpoint[exit_side];
                break;
            }
            cur = nxt;
            entry = static_cast<uint8_t>(next_entry);
        }

        out.emplace_back(first, last);
    }

    for (size_t i = 0; i < edges.size(); ++i) {
        if (!seen[i]) out.emplace_back(edges[i].endpoint[0], edges[i].endpoint[1]);
    }

    pairs.swap(out);
}

std::vector<std::pair<std::string, std::string>>
compactStringPairChains(std::vector<std::pair<std::string, std::string>> pairs)
{
    if (pairs.size() < 2) return pairs;

    std::unordered_map<std::string, uint64_t> ids;
    std::vector<std::string> names;
    ids.reserve(pairs.size() * 2);
    names.reserve(pairs.size() * 2);

    auto keyOf = [&](const std::string &s, uint64_t &key) -> bool {
        if (s.size() < 2) return false;
        char c = s.back();
        if (c != '+' && c != '-') return false;
        std::string name(s.data(), s.size() - 1);
        uint64_t id = ids.size();
        auto [it, inserted] = ids.emplace(name, id);
        if (inserted) names.push_back(std::move(name));
        key = (it->second << 1) | (c == '-' ? 1ull : 0ull);
        return true;
    };

    std::vector<std::pair<uint64_t, uint64_t>> packed;
    packed.reserve(pairs.size());
    for (const auto &p : pairs) {
        uint64_t a = 0, b = 0;
        if (!keyOf(p.first, a) || !keyOf(p.second, b)) return pairs;
        packed.emplace_back(a, b);
    }

    compactEndpointPairChains(packed);

    std::vector<std::pair<std::string, std::string>> out;
    out.reserve(packed.size());
    for (const auto &p : packed) {
        auto endpoint = [&](uint64_t key) {
            std::string s = names[key >> 1];
            s.push_back((key & 1ull) ? '-' : '+');
            return s;
        };
        out.emplace_back(endpoint(p.first), endpoint(p.second));
    }
    return out;
}

template <typename SnarlSet>
void writeAllSnarls_buffered(std::ostream &out, const SnarlSet &snarls)
{
    if (ctx().compactOutputChains) {
        std::vector<std::pair<std::string, std::string>> pairs;
        std::vector<std::vector<std::string>> other;
        pairs.reserve(snarls.size());
        for (const auto &s : snarls) {
            if (s.size() == 2) {
                pairs.emplace_back(s[0], s[1]);
            } else {
                other.emplace_back(s.begin(), s.end());
            }
        }
        pairs = compactStringPairChains(std::move(pairs));

        std::string buf;
        buf.reserve(kIoChunkHighWater + 4096);
        buf.append(std::to_string(pairs.size() + other.size()));
        buf.push_back('\n');

        for (const auto &s : other) {
            for (const auto &v : s) {
                buf.append(v);
                buf.push_back(' ');
            }
            buf.push_back('\n');
            if (buf.size() >= kIoChunkHighWater) flushStringBuf(out, buf);
        }
        for (const auto &p : pairs) {
            buf.append(p.first);
            buf.push_back(' ');
            buf.append(p.second);
            buf.push_back('\n');
            if (buf.size() >= kIoChunkHighWater) flushStringBuf(out, buf);
        }
        flushStringBuf(out, buf);
        return;
    }

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

struct FastSnarlOutputTables {
    const std::vector<std::string>* compact_node_names = nullptr;
    const std::vector<uint64_t>* compact_numeric_node_names = nullptr;
    const std::vector<uint8_t>* compact_numeric_name_valid = nullptr;
    const std::unordered_map<uint32_t, std::string>* sparse_node_names = nullptr;
    const std::vector<uint8_t>* compact_is_trash = nullptr;
    std::vector<const std::string*> node_names;
    std::vector<uint8_t> is_trash;
    std::vector<int32_t> unique_plus;
    std::vector<int32_t> unique_minus;
    size_t max_idx = 0;
};

inline bool tableIsTrash(const FastSnarlOutputTables &t, uint32_t idx)
{
    if (idx < t.is_trash.size() && t.is_trash[idx]) return true;
    return t.compact_is_trash != nullptr &&
           idx < t.compact_is_trash->size() &&
           (*(t.compact_is_trash))[idx] != 0;
}

inline const std::string* tableStringName(const FastSnarlOutputTables &t, uint32_t idx)
{
    if (idx < t.node_names.size() && t.node_names[idx] != nullptr) {
        return t.node_names[idx];
    }
    if (t.compact_node_names != nullptr &&
        idx < t.compact_node_names->size() &&
        !(*(t.compact_node_names))[idx].empty()) {
        return &(*(t.compact_node_names))[idx];
    }
    if (t.sparse_node_names != nullptr) {
        auto it = t.sparse_node_names->find(idx);
        if (it != t.sparse_node_names->end()) {
            return &it->second;
        }
    }
    return nullptr;
}

inline bool tableNumericName(const FastSnarlOutputTables &t, uint32_t idx, uint64_t &name)
{
    if (tableIsTrash(t, idx)) return false;
    if (t.compact_numeric_node_names != nullptr &&
        idx < t.compact_numeric_node_names->size() &&
        t.compact_numeric_name_valid != nullptr &&
        idx < t.compact_numeric_name_valid->size() &&
        (*(t.compact_numeric_name_valid))[idx]) {
        name = (*(t.compact_numeric_node_names))[idx];
        return true;
    }
    return false;
}

inline uint64_t decimalDivisor(uint64_t value)
{
    uint64_t divisor = 1;
    while (value >= 10) {
        value /= 10;
        divisor *= 10;
    }
    return divisor;
}

inline int compareDecimalNames(uint64_t a, uint64_t b)
{
    uint64_t div_a = decimalDivisor(a);
    uint64_t div_b = decimalDivisor(b);
    while (div_a != 0 && div_b != 0) {
        const uint64_t da = a / div_a;
        const uint64_t db = b / div_b;
        if (da != db) return da < db ? -1 : 1;
        a %= div_a;
        b %= div_b;
        div_a /= 10;
        div_b /= 10;
    }
    if (div_a == div_b) return 0;
    return div_a == 0 ? -1 : 1;
}

inline int compareDecimalNameToString(uint64_t value, const std::string &name)
{
    char tmp[32];
    auto [ptr, ec] = std::to_chars(std::begin(tmp), std::end(tmp), value);
    if (ec != std::errc()) {
        return -1;
    }
    const std::string_view numeric(tmp, static_cast<size_t>(ptr - tmp));
    const std::string_view other(name.data(), name.size());
    const int cmp = numeric.compare(other);
    return cmp < 0 ? -1 : (cmp > 0 ? 1 : 0);
}

inline void appendU64(uint64_t value, std::string &buf)
{
    char tmp[32];
    auto [ptr, ec] = std::to_chars(std::begin(tmp), std::end(tmp), value);
    if (ec == std::errc()) {
        buf.append(tmp, ptr);
    } else {
        buf.append(std::to_string(value));
    }
}

inline uint64_t packFastPairKey(uint32_t a_idx, uint8_t a_sign,
                                uint32_t b_idx, uint8_t b_sign)
{
    uint64_t a = (static_cast<uint64_t>(a_idx) << 1) |
                 static_cast<uint64_t>(a_sign);
    uint64_t b = (static_cast<uint64_t>(b_idx) << 1) |
                 static_cast<uint64_t>(b_sign);
    if (a > b) std::swap(a, b);
    return (a << 32) | b;
}

inline uint64_t packFastPairEndpointKeys(uint64_t a, uint64_t b)
{
    if (a > b) std::swap(a, b);
    return (a << 32) | b;
}

inline uint32_t fastEndpointIdx(uint64_t key)
{
    return static_cast<uint32_t>(key >> 1);
}

inline uint8_t fastEndpointSign(uint64_t key)
{
    return static_cast<uint8_t>(key & 1u);
}

inline bool parseFastEndpoint(const Context &C,
                              const std::string &s,
                              uint32_t &idx,
                              uint8_t &sign)
{
    if (s.size() < 2) return false;
    char c = s.back();
    if (c == '+') {
        sign = 0;
    } else if (c == '-') {
        sign = 1;
    } else {
        return false;
    }

    std::string name(s.data(), s.size() - 1);
    if (name == "_trash") return false;
    auto it = C.name2node.find(name);
    if (it == C.name2node.end()) return false;
    idx = static_cast<uint32_t>(it->second.idx);
    return true;
}

FastSnarlOutputTables buildFastSnarlOutputTables(Context &C, bool need_trivial_filter)
{
    size_t max_idx = 0;
    if (!C.nodeNamesByIndex.empty()) {
        max_idx = C.nodeNamesByIndex.size() - 1;
    } else if (!C.nodeNumericNamesByIndex.empty()) {
        max_idx = C.nodeNumericNamesByIndex.size() - 1;
    } else {
        for (spqr_compat::node n : C.G.nodes) {
            max_idx = std::max(max_idx, static_cast<size_t>(n.idx));
        }
    }

    FastSnarlOutputTables t;
    t.max_idx = max_idx;
    if (!C.nodeNamesByIndex.empty()) {
        t.compact_node_names = &C.nodeNamesByIndex;
    }
    if (!C.nodeNumericNamesByIndex.empty()) {
        t.compact_numeric_node_names = &C.nodeNumericNamesByIndex;
    }
    if (!C.nodeNumericNameValidByIndex.empty()) {
        t.compact_numeric_name_valid = &C.nodeNumericNameValidByIndex;
    }
    if (!C.sparseNodeNamesByIndex.empty()) {
        t.sparse_node_names = &C.sparseNodeNamesByIndex;
    }
    if (!C.isTrashNodeByIndex.empty()) {
        t.compact_is_trash = &C.isTrashNodeByIndex;
    }
    if (need_trivial_filter) {
        t.unique_plus.assign(max_idx + 1, -1);
        t.unique_minus.assign(max_idx + 1, -1);
    }

    if (!C.node2name.empty()) {
        t.node_names.assign(max_idx + 1, nullptr);
        t.is_trash.assign(max_idx + 1, 0);
        for (const auto &kv : C.node2name) {
            const uint32_t idx = static_cast<uint32_t>(kv.first.idx);
            if (idx >= t.node_names.size()) continue;
            t.node_names[idx] = &kv.second;
            if (kv.second == "_trash") {
                t.is_trash[idx] = 1;
            }
        }
    }

    if (!need_trivial_filter) {
        return t;
    }

    for (spqr_compat::node u : C.G.nodes) {
        spqr_compat::node nbrPlus{nullptr};
        spqr_compat::node nbrMinus{nullptr};
        int countPlus = 0;
        int countMinus = 0;

        C.G.forEachAdj(u, [&](spqr_compat::node other, spqr_compat::edge e) {
            EdgePartType typeAtU = (C.G.source(e) == u)
                ? C._edge2types[e].first
                : C._edge2types[e].second;
            int *cnt = nullptr;
            spqr_compat::node *slot = nullptr;
            if (typeAtU == EdgePartType::PLUS) {
                cnt = &countPlus;
                slot = &nbrPlus;
            } else if (typeAtU == EdgePartType::MINUS) {
                cnt = &countMinus;
                slot = &nbrMinus;
            } else {
                return;
            }

            if (*cnt > 1) return;

            if (tableIsTrash(t, static_cast<uint32_t>(other.idx))) {
                C.G.forEachAdj(other, [&](spqr_compat::node real, spqr_compat::edge) {
                    if (real == u) return;
                    if (*cnt > 1) return;
                    if (*cnt == 0 || *slot == real) {
                        *slot = real;
                        if (*cnt == 0) (*cnt)++;
                    } else {
                        (*cnt)++;
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

        const uint32_t uidx = static_cast<uint32_t>(u.idx);
        if (uidx >= t.unique_plus.size()) continue;
        if (countPlus == 1 && nbrPlus) {
            t.unique_plus[uidx] = static_cast<int32_t>(nbrPlus.idx);
        }
        if (countMinus == 1 && nbrMinus) {
            t.unique_minus[uidx] = static_cast<int32_t>(nbrMinus.idx);
        }
    }

    return t;
}

inline bool fastEndpointLess(const FastSnarlOutputTables &t,
                             uint32_t a_idx,
                             uint8_t a_sign,
                             uint32_t b_idx,
                             uint8_t b_sign)
{
    const bool a_trash = tableIsTrash(t, a_idx);
    const bool b_trash = tableIsTrash(t, b_idx);
    uint64_t a_numeric = 0, b_numeric = 0;
    const bool a_has_numeric = !a_trash && tableNumericName(t, a_idx, a_numeric);
    const bool b_has_numeric = !b_trash && tableNumericName(t, b_idx, b_numeric);
    const std::string *a_name = (a_trash || a_has_numeric) ? nullptr : tableStringName(t, a_idx);
    const std::string *b_name = (b_trash || b_has_numeric) ? nullptr : tableStringName(t, b_idx);

    int cmp = 0;
    if (a_trash || b_trash) {
        static const std::string kTrashName = "_trash";
        const std::string *as = a_trash ? &kTrashName : a_name;
        const std::string *bs = b_trash ? &kTrashName : b_name;
        if (a_has_numeric && bs) {
            cmp = compareDecimalNameToString(a_numeric, *bs);
        } else if (b_has_numeric && as) {
            cmp = -compareDecimalNameToString(b_numeric, *as);
        } else if (as && bs) {
            cmp = as->compare(*bs);
        } else {
            if (a_idx != b_idx) return a_idx < b_idx;
            return a_sign < b_sign;
        }
    } else if (a_has_numeric && b_has_numeric) {
        cmp = compareDecimalNames(a_numeric, b_numeric);
    } else if (a_has_numeric && b_name) {
        cmp = compareDecimalNameToString(a_numeric, *b_name);
    } else if (b_has_numeric && a_name) {
        cmp = -compareDecimalNameToString(b_numeric, *a_name);
    } else if (a_name && b_name) {
        cmp = a_name->compare(*b_name);
    } else {
        if (a_idx != b_idx) return a_idx < b_idx;
        return a_sign < b_sign;
    }

    if (cmp != 0) return cmp < 0;
    return (a_sign == 0 ? '+' : '-') < (b_sign == 0 ? '+' : '-');
}

inline bool fastEndpointKeyLess(const FastSnarlOutputTables &t,
                                uint64_t a,
                                uint64_t b)
{
    return fastEndpointLess(t,
                            fastEndpointIdx(a), fastEndpointSign(a),
                            fastEndpointIdx(b), fastEndpointSign(b));
}

inline bool fastPairIsTrivial(const FastSnarlOutputTables &t, uint64_t key)
{
    const uint32_t a = static_cast<uint32_t>(key >> 32);
    const uint32_t b = static_cast<uint32_t>(key & 0xffffffffu);
    const uint32_t a_idx = a >> 1;
    const uint8_t a_sign = static_cast<uint8_t>(a & 1u);
    const uint32_t b_idx = b >> 1;
    const uint8_t b_sign = static_cast<uint8_t>(b & 1u);

    if (a_idx >= t.unique_plus.size() || b_idx >= t.unique_plus.size()) {
        return false;
    }
    const int32_t a_nbr = (a_sign == 0) ? t.unique_plus[a_idx] : t.unique_minus[a_idx];
    const int32_t b_nbr = (b_sign == 0) ? t.unique_plus[b_idx] : t.unique_minus[b_idx];
    return a_nbr == static_cast<int32_t>(b_idx) &&
           b_nbr == static_cast<int32_t>(a_idx);
}

inline bool fastEndpointPairIsTrivial(const FastSnarlOutputTables &t,
                                      uint64_t a,
                                      uint64_t b)
{
    return fastPairIsTrivial(t, packFastPairEndpointKeys(a, b));
}

inline void appendFastEndpoint(const FastSnarlOutputTables &t,
                               uint32_t idx,
                               uint8_t sign,
                               std::string &buf)
{
    uint64_t numeric_name = 0;
    if (tableIsTrash(t, idx)) {
        buf.append("_trash");
    } else if (tableNumericName(t, idx, numeric_name)) {
        appendU64(numeric_name, buf);
    } else {
        const std::string *name = tableStringName(t, idx);
        if (name) {
            buf.append(*name);
        } else {
            appendU64(idx, buf);
        }
    }
    buf.push_back(sign == 0 ? '+' : '-');
}

inline void appendFastEndpointKey(const FastSnarlOutputTables &t,
                                  uint64_t key,
                                  std::string &buf)
{
    appendFastEndpoint(t, fastEndpointIdx(key), fastEndpointSign(key), buf);
}

void appendFallbackStringSnarlsToFastPairs(Context &C)
{
    if (C.snarls.empty()) return;

    thread_local std::vector<std::pair<uint32_t, uint8_t>> endpoints;
    for (const auto &s : C.snarls) {
        endpoints.clear();
        endpoints.reserve(s.size());
        for (const std::string &endpoint : s) {
            uint32_t idx = 0;
            uint8_t sign = 255;
            if (!parseFastEndpoint(C, endpoint, idx, sign)) {
                throw std::runtime_error(
                    "fast snarl output encountered a non-packable endpoint");
            }
            endpoints.emplace_back(idx, sign);
        }
        if (endpoints.size() == 2) {
            C.fastSnarlPairs.push_back(packFastPairKey(
                endpoints[0].first, endpoints[0].second,
                endpoints[1].first, endpoints[1].second));
        } else if (endpoints.size() > 2) {
            std::vector<uint64_t> clique;
            clique.reserve(endpoints.size());
            for (const auto &endpoint : endpoints) {
                clique.push_back((static_cast<uint64_t>(endpoint.first) << 1) |
                                 static_cast<uint64_t>(endpoint.second));
            }
            C.fastSnarlCliques.push_back(std::move(clique));
        }
    }
    C.snarls.clear();
}

void writeFastSnarlPairs(std::ostream &out, Context &C)
{
    appendFallbackStringSnarlsToFastPairs(C);

    const bool filter_trivial = !C.includeTrivial;
    auto tables = buildFastSnarlOutputTables(C, filter_trivial);
    auto &pairs = C.fastSnarlPairs;
    auto &cliques = C.fastSnarlCliques;

    auto endpointLess = [&](uint64_t a, uint64_t b) {
        return fastEndpointKeyLess(tables, a, b);
    };
    auto cliqueLess = [&](const std::vector<uint64_t> &a,
                          const std::vector<uint64_t> &b) {
        return std::lexicographical_compare(a.begin(), a.end(),
                                            b.begin(), b.end(),
                                            endpointLess);
    };

    for (auto &clique : cliques) {
        std::sort(clique.begin(), clique.end(), endpointLess);
        clique.erase(std::unique(clique.begin(), clique.end()), clique.end());
    }
    cliques.erase(std::remove_if(cliques.begin(), cliques.end(),
                                 [](const std::vector<uint64_t> &clique) {
                                     return clique.size() < 2;
                                 }),
                  cliques.end());
    std::sort(cliques.begin(), cliques.end(), cliqueLess);
    cliques.erase(std::unique(cliques.begin(), cliques.end()), cliques.end());

    std::vector<uint64_t> coveredPairs;
    std::vector<std::vector<uint64_t>> outputCliques;
    std::vector<uint64_t> cliquePairs;

    for (auto &clique : cliques) {
        cliquePairs.clear();
        cliquePairs.reserve(clique.size() * (clique.size() - 1) / 2);
        bool canCompact = true;

        for (size_t i = 0; i < clique.size(); ++i) {
            for (size_t j = i + 1; j < clique.size(); ++j) {
                uint64_t key = packFastPairEndpointKeys(clique[i], clique[j]);
                if (filter_trivial && fastPairIsTrivial(tables, key)) {
                    canCompact = false;
                } else {
                    cliquePairs.push_back(key);
                }
            }
        }

        std::sort(cliquePairs.begin(), cliquePairs.end());
        cliquePairs.erase(std::unique(cliquePairs.begin(), cliquePairs.end()),
                          cliquePairs.end());

        if (canCompact) {
            bool overlaps = false;
            for (uint64_t key : cliquePairs) {
                if (std::binary_search(coveredPairs.begin(), coveredPairs.end(), key)) {
                    overlaps = true;
                    break;
                }
            }

            if (!overlaps) {
                outputCliques.push_back(std::move(clique));

                std::vector<uint64_t> merged;
                merged.reserve(coveredPairs.size() + cliquePairs.size());
                std::merge(coveredPairs.begin(), coveredPairs.end(),
                           cliquePairs.begin(), cliquePairs.end(),
                           std::back_inserter(merged));
                merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
                coveredPairs.swap(merged);
                continue;
            }
        }

        pairs.insert(pairs.end(), cliquePairs.begin(), cliquePairs.end());
    }

    if (C.threads > 1 && pairs.size() > 100000) {
#if defined(_OPENMP) && defined(__GLIBCXX__)
        __gnu_parallel::sort(pairs.begin(), pairs.end());
#else
        std::sort(pairs.begin(), pairs.end());
#endif
    } else {
        std::sort(pairs.begin(), pairs.end());
    }
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    size_t pair_count = 0;
    std::vector<std::pair<uint64_t, uint64_t>> outputPairs;
    const bool write_all_pairs =
        !filter_trivial && !C.compactOutputChains && coveredPairs.empty();
    if (C.compactOutputChains) {
        outputPairs.reserve(pairs.size());
        for (uint64_t key : pairs) {
            if ((!filter_trivial || !fastPairIsTrivial(tables, key)) &&
                !std::binary_search(coveredPairs.begin(), coveredPairs.end(), key)) {
                outputPairs.emplace_back(static_cast<uint32_t>(key >> 32),
                                         static_cast<uint32_t>(key & 0xffffffffu));
            }
        }
        compactEndpointPairChains(outputPairs);
        pair_count = outputPairs.size();
    } else if (write_all_pairs) {
        pair_count = pairs.size();
    } else {
        size_t covered_i = 0;
        for (uint64_t key : pairs) {
            if (filter_trivial && fastPairIsTrivial(tables, key)) continue;
            while (covered_i < coveredPairs.size() && coveredPairs[covered_i] < key) {
                ++covered_i;
            }
            if (covered_i < coveredPairs.size() && coveredPairs[covered_i] == key) continue;
            ++pair_count;
        }
    }

    std::string buf;
    buf.reserve(kIoChunkHighWater + 4096);
    buf.append(std::to_string(pair_count + outputCliques.size()));
    buf.push_back('\n');

    for (const auto &clique : outputCliques) {
        for (size_t i = 0; i < clique.size(); ++i) {
            if (i) buf.push_back(' ');
            appendFastEndpointKey(tables, clique[i], buf);
        }
        buf.push_back('\n');

        if (buf.size() >= kIoChunkHighWater) {
            flushStringBuf(out, buf);
        }
    }

    auto appendPairLine = [&](uint64_t a, uint64_t b, std::string &dst) {
        uint32_t a_idx = static_cast<uint32_t>(a >> 1);
        uint8_t a_sign = static_cast<uint8_t>(a & 1u);
        uint32_t b_idx = static_cast<uint32_t>(b >> 1);
        uint8_t b_sign = static_cast<uint8_t>(b & 1u);

        if (!fastEndpointLess(tables, a_idx, a_sign, b_idx, b_sign)) {
            std::swap(a_idx, b_idx);
            std::swap(a_sign, b_sign);
        }

        appendFastEndpoint(tables, a_idx, a_sign, dst);
        dst.push_back(' ');
        appendFastEndpoint(tables, b_idx, b_sign, dst);
        dst.push_back('\n');
    };

    auto writePair = [&](uint64_t a, uint64_t b) {
        appendPairLine(a, b, buf);

        if (buf.size() >= kIoChunkHighWater) flushStringBuf(out, buf);
    };

    auto appendPairKeyLine = [&](uint64_t key, std::string &dst) {
        appendPairLine(static_cast<uint32_t>(key >> 32),
                       static_cast<uint32_t>(key & 0xffffffffu),
                       dst);
    };

    if (C.compactOutputChains) {
        for (auto [a, b] : outputPairs) writePair(a, b);
    } else if (write_all_pairs) {
        const bool use_parallel_format =
            C.threads > 1 && pairs.size() > 100000;
#if defined(_OPENMP)
        if (use_parallel_format) {
            const int workers = std::max<int>(1, static_cast<int>(C.threads));
            const size_t block_pairs = 1ull << 19;
            const size_t block_count = (pairs.size() + block_pairs - 1) / block_pairs;
            std::vector<std::string> buffers;
            for (size_t group_begin = 0; group_begin < block_count;
                 group_begin += static_cast<size_t>(workers)) {
                const size_t group_end = std::min(
                    block_count, group_begin + static_cast<size_t>(workers));
                const size_t group_size = group_end - group_begin;
                buffers.clear();
                buffers.resize(group_size);

                #pragma omp parallel for schedule(static) num_threads(workers)
                for (int64_t bi_i = 0; bi_i < static_cast<int64_t>(group_size); ++bi_i) {
                    const size_t bi = static_cast<size_t>(bi_i);
                    const size_t block = group_begin + bi;
                    const size_t begin = block * block_pairs;
                    const size_t end = std::min(pairs.size(), begin + block_pairs);
                    std::string local;
                    local.reserve((end - begin) * 32);
                    for (size_t i = begin; i < end; ++i) {
                        appendPairKeyLine(pairs[i], local);
                    }
                    buffers[bi].swap(local);
                }

                for (std::string &local : buffers) {
                    flushStringBuf(out, local);
                }
            }
        } else
#endif
        {
            for (uint64_t key : pairs) {
                appendPairKeyLine(key, buf);
                if (buf.size() >= kIoChunkHighWater) flushStringBuf(out, buf);
            }
        }
    } else {
        const bool use_parallel_format =
            C.threads > 1 && pairs.size() > 100000 && !C.compactOutputChains;
#if defined(_OPENMP)
        if (use_parallel_format) {
            const int workers = std::max<int>(1, static_cast<int>(C.threads));
            const size_t block_pairs = 1ull << 19;
            const size_t block_count = (pairs.size() + block_pairs - 1) / block_pairs;
            std::vector<std::string> buffers;
            for (size_t group_begin = 0; group_begin < block_count;
                 group_begin += static_cast<size_t>(workers)) {
                const size_t group_end = std::min(
                    block_count, group_begin + static_cast<size_t>(workers));
                const size_t group_size = group_end - group_begin;
                buffers.clear();
                buffers.resize(group_size);

                #pragma omp parallel for schedule(static) num_threads(workers)
                for (int64_t bi_i = 0; bi_i < static_cast<int64_t>(group_size); ++bi_i) {
                    const size_t bi = static_cast<size_t>(bi_i);
                    const size_t block = group_begin + bi;
                    const size_t begin = block * block_pairs;
                    const size_t end = std::min(pairs.size(), begin + block_pairs);
                    std::string local;
                    local.reserve((end - begin) * 32);
                    size_t covered_i = coveredPairs.empty()
                        ? 0
                        : static_cast<size_t>(std::lower_bound(
                              coveredPairs.begin(), coveredPairs.end(), pairs[begin]) -
                                              coveredPairs.begin());
                    for (size_t i = begin; i < end; ++i) {
                        const uint64_t key = pairs[i];
                        if (filter_trivial && fastPairIsTrivial(tables, key)) continue;
                        while (covered_i < coveredPairs.size() && coveredPairs[covered_i] < key) {
                            ++covered_i;
                        }
                        if (covered_i < coveredPairs.size() && coveredPairs[covered_i] == key) continue;
                        appendPairKeyLine(key, local);
                    }
                    buffers[bi].swap(local);
                }

                for (std::string &local : buffers) {
                    flushStringBuf(out, local);
                }
            }
        } else
#endif
        {
            size_t covered_i = 0;
            for (uint64_t key : pairs) {
                if (filter_trivial && fastPairIsTrivial(tables, key)) continue;
                while (covered_i < coveredPairs.size() && coveredPairs[covered_i] < key) {
                    ++covered_i;
                }
                if (covered_i < coveredPairs.size() && coveredPairs[covered_i] == key) continue;
                writePair(static_cast<uint32_t>(key >> 32),
                          static_cast<uint32_t>(key & 0xffffffffu));
            }
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
        if (std::getenv("BF_DEBUG_FAST_SNARL_OUTPUT")) {
            std::cerr << "[fast_snarl_output] enabled="
                      << (C.fastSnarlPairsEnabled ? 1 : 0)
                      << " include_trivial=" << (C.includeTrivial ? 1 : 0)
                      << " sp_compress_mode=" << static_cast<int>(C.spCompressMode)
                      << " disable_env=" << (std::getenv("BF_DISABLE_FAST_SNARL_OUTPUT") ? 1 : 0)
                      << " pairs=" << C.fastSnarlPairs.size()
                      << " cliques=" << C.fastSnarlCliques.size()
                      << " string_snarls=" << C.snarls.size()
                      << "\n";
        }
        if (C.fastSnarlPairsEnabled)
        {
            if (C.outputPath.empty())
            {
                writeFastSnarlPairs(std::cout, C);
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
                writeFastSnarlPairs(out, C);
                if (!out) {
                    throw std::runtime_error("Error while writing snarls to output file '" +
                                             C.outputPath + "'");
                }
            }
            return;
        }

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
            std::unordered_map<RealNbrKey, spqr_compat::node, RealNbrKeyHash> uniqueRealNbr;
            uniqueRealNbr.reserve(C.G.numberOfNodes() * 2);

            size_t max_idx = 0;
            for (spqr_compat::node n : C.G.nodes) {
                if (static_cast<size_t>(n.idx) > max_idx) max_idx = static_cast<size_t>(n.idx);
            }
            std::vector<bool> is_trash(max_idx + 1, false);
            for (const auto &kv : C.node2name) {
                if (kv.second == "_trash") {
                    is_trash[kv.first.idx] = true;
                }
            }

            for (spqr_compat::node u : C.G.nodes) {
                spqr_compat::node nbrPlus{nullptr};
                spqr_compat::node nbrMinus{nullptr};
                int countPlus = 0, countMinus = 0;
                C.G.forEachAdj(u, [&](spqr_compat::node other, spqr_compat::edge e) {
                    EdgePartType typeAtU = (C.G.source(e) == u)
                        ? C._edge2types[e].first
                        : C._edge2types[e].second;
                    int *cnt;
                    spqr_compat::node *slot;
                    if (typeAtU == EdgePartType::PLUS) { cnt = &countPlus;  slot = &nbrPlus;  }
                    else if (typeAtU == EdgePartType::MINUS) { cnt = &countMinus; slot = &nbrMinus; }
                    else return;

                    if (*cnt > 1) return;  // already disqualified

                    if (is_trash[other.idx]) {
                        C.G.forEachAdj(other, [&](spqr_compat::node real, spqr_compat::edge) {
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
            if (!C.compactOutputChains) out_buf.reserve(400ull * 1024ull * 1024ull);
            std::vector<std::pair<std::string, std::string>> compact_pairs;
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

                        if (C.compactOutputChains) {
                            compact_pairs.emplace_back(*pa, *pb);
                        } else {
                            out_buf.append(*pa);
                            out_buf.push_back(' ');
                            out_buf.append(*pb);
                            out_buf.push_back('\n');
                        }
                        pair_count++;
                    }
                }
            }

            // Free dedup memory before writing (out_buf alone is ~340 MB).
            std::unordered_set<uint64_t>().swap(seen_num);
            std::unordered_set<std::pair<std::string, std::string>, PairHash>().swap(seen_str);
            std::vector<std::pair<uint32_t, uint8_t>>().swap(snarl_nodes);
            if (C.compactOutputChains) compact_pairs = compactStringPairChains(std::move(compact_pairs));

            auto writeStreamedOutput = [&](std::ostream &os) {
                std::string header = std::to_string(C.compactOutputChains ? compact_pairs.size() : pair_count) + "\n";
                os.write(header.data(), static_cast<std::streamsize>(header.size()));
                if (C.compactOutputChains) {
                    std::string buf;
                    buf.reserve(kIoChunkHighWater + 4096);
                    for (const auto &p : compact_pairs) {
                        buf.append(p.first);
                        buf.push_back(' ');
                        buf.append(p.second);
                        buf.push_back('\n');
                        if (buf.size() >= kIoChunkHighWater) flushStringBuf(os, buf);
                    }
                    flushStringBuf(os, buf);
                } else {
                    os.write(out_buf.data(), static_cast<std::streamsize>(out_buf.size()));
                }
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

        std::vector<std::pair<uint64_t, uint64_t>> compact_pairs;
        if (C.compactOutputChains) {
            compact_pairs.reserve(C.ultrabubbleIncPacked.size());
            for (const auto &p : C.ultrabubbleIncPacked) {
                compact_pairs.emplace_back(p.first, p.second);
            }
            compactEndpointPairChains(compact_pairs);
        }

        if (C.outputPath.empty())
        {
            std::cout << (C.compactOutputChains ? compact_pairs.size() : C.ultrabubbleIncPacked.size()) << "\n";
            if (C.compactOutputChains) {
                for (auto &p : compact_pairs)
                {
                    write_one(std::cout, static_cast<std::uint32_t>(p.first));
                    std::cout << " ";
                    write_one(std::cout, static_cast<std::uint32_t>(p.second));
                    std::cout << "\n";
                }
            } else {
                for (auto &p : C.ultrabubbleIncPacked)
                {
                    write_one(std::cout, p.first);
                    std::cout << " ";
                    write_one(std::cout, p.second);
                    std::cout << "\n";
                }
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
            out << (C.compactOutputChains ? compact_pairs.size() : C.ultrabubbleIncPacked.size()) << "\n";
            if (C.compactOutputChains) {
                for (auto &p : compact_pairs)
                {
                    write_one(out, static_cast<std::uint32_t>(p.first));
                    out << " ";
                    write_one(out, static_cast<std::uint32_t>(p.second));
                    out << "\n";
                }
            } else {
                for (auto &p : C.ultrabubbleIncPacked)
                {
                    write_one(out, p.first);
                    out << " ";
                    write_one(out, p.second);
                    out << "\n";
                }
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

    if (C.compactOutputChains) res = compactStringPairChains(std::move(res));

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
