#include "graph_io.hpp"
#include "util/context.hpp"
#include "util/graph_index.hpp"
#include "util/timer.hpp"
#include "util/logger.hpp"
#include "util/profiling_macros.hpp"
#include "gfa_parser.hpp"

#include "gbz_parser.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <charconv>
#include <unistd.h>
#include <atomic>
#include <iostream>
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
            const uint64_t digit = static_cast<uint64_t>(*p - '0');
            if (v > (std::numeric_limits<uint64_t>::max() - digit) / 10u) {
                return false;
            }
            v = v * 10u + digit;
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
    const uint32_t n = graph_index::require_spqr_count(
        n64, std::string(srcName) + " .graph declared node count");
    const uint32_t m_u32 = graph_index::require_spqr_count(
        m64, std::string(srcName) + " .graph declared edge count");
    const size_t m = static_cast<size_t>(m_u32);

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
    edge_set.reserve(graph_index::checked_mul_size(
        edges_raw.size(), 2u, ".graph raw edge hash table"));
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
    processed.reserve(graph_index::checked_mul_size(
        edges_ordered.size(), 2u, ".graph processed edge hash table"));
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

void clearCompactNodeNameTables(Context &C)
{
    C.nodeNamesByIndex.clear();
    C.nodeNumericNamesByIndex.clear();
    C.nodeNumericNameValidByIndex.clear();
    C.sparseNodeNamesByIndex.clear();
    C.isTrashNodeByIndex.clear();
}

void clearNodeNameTables(Context &C)
{
    C.node2name.clear();
    C.name2node.clear();
    clearCompactNodeNameTables(C);
}

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

bool isGeneratedSpqrTrashName(const std::string &name, uint32_t graphOrdinal);
std::string spqrIndexGraphNodeName(const spqr_index::IndexedSpqrTree &index,
                                   std::uint32_t ordinal);
bool spqrIndexHasIdentityGraphOrdinals(const spqr_index::IndexedSpqrTree &index);
bool hasCompleteGraphEdgeTypesForDirectBuild(
    const spqr_index::IndexedSpqrTree &index,
    std::size_t edge_count,
    unsigned threads);
bool hasCompleteRealEdgeTypesForDirectBuild(
    const spqr_index::IndexedSpqrTree &index,
    std::size_t edge_count,
    unsigned threads);

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
        if (isGeneratedSpqrTrashName(it->second, i)) return "_trash";
        return it->second;
    }
    return std::to_string(i);
}

bool isGeneratedSpqrTrashName(const std::string &name, uint32_t graphOrdinal)
{
    if (name.rfind("__BF__", 0) != 0) return false;
    const std::string suffix = "N" + std::to_string(graphOrdinal);
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return true;
    }

    std::size_t n_pos = std::string::npos;
    for (std::size_t i = 6; i < name.size(); ++i) {
        if (name[i] == 'N') {
            n_pos = i;
            break;
        }
        if (name[i] != '_') return false;
    }
    if (n_pos == std::string::npos || n_pos + 1 == name.size()) return false;
    for (std::size_t i = n_pos + 1; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') return false;
    }
    return true;
}

bool useCompactSnarlNameTables(const Context &C)
{
    return C.bubbleType == Context::BubbleType::SNARL &&
           C.includeTrivial &&
           C.spCompressMode == Context::SpCompressMode::MacroDirect;
}

bool useCompactSuperbubbleNameTables(const Context &C,
                                     const BiGraph &bg,
                                     bool directed_only)
{
    if (C.bubbleType != Context::BubbleType::SPQR_TREE_ONLY ||
        C.outputPath.empty() ||
        !spqr_index::detail::ends_with(C.outputPath, ".spqri")) {
        return false;
    }
    if (!spqr_index::graph_profile_matches_oriented_double(C.spqrTreeView,
                                                           directed_only) ||
        bg.numeric_node_names.size() != bg.n_nodes ||
        !bg.string_node_names.empty()) {
        return false;
    }
    if (!bg.numeric_node_name_valid.empty() &&
        !std::all_of(bg.numeric_node_name_valid.begin(),
                     bg.numeric_node_name_valid.end(),
                     [](std::uint8_t v) { return v != 0; })) {
        return false;
    }
    for (std::uint64_t name : bg.numeric_node_names) {
        if (name > (std::numeric_limits<std::uint64_t>::max() >> 1u)) {
            return false;
        }
    }
    return true;
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
    const size_t node_storage_size = graph_index::checked_add_size(
        static_cast<size_t>(bg.n_nodes), extra_nodes, "input graph node storage");
    graph_index::require_spqr_count(bg.n_nodes, "input graph node count");
    graph_index::require_spqr_count_size(
        graph_index::checked_add_size(
            static_cast<size_t>(C.G.numberOfNodes()), node_storage_size,
            "SPQR graph node count"),
        "SPQR graph node count");
    std::vector<spqr_compat::node> id2node(bg.n_nodes);
    const bool compact_names = useCompactSnarlNameTables(C);
    const bool bg_has_numeric_names =
        bg.numeric_node_names.size() == bg.n_nodes &&
        bg.numeric_node_name_valid.size() == bg.n_nodes;
    const bool build_name_index =
        !(C.bubbleType == Context::BubbleType::SNARL && C.includeTrivial);
    if (build_name_index) {
        C.name2node.reserve(bg.n_nodes);
    }
    if (compact_names) {
        clearCompactNodeNameTables(C);
        if (bg_has_numeric_names) {
            C.nodeNumericNamesByIndex.resize(node_storage_size);
            C.nodeNumericNameValidByIndex.resize(node_storage_size, 0);
            C.sparseNodeNamesByIndex.reserve(bg.string_node_names.size());
        } else {
            C.nodeNamesByIndex.resize(node_storage_size);
        }
        C.isTrashNodeByIndex.resize(node_storage_size, 0);
    } else {
        C.node2name.reserve(node_storage_size);
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
            if (item.second == "_trash" ||
                isGeneratedSpqrTrashName(item.second, item.first)) {
                setCompactTrashNode(C, v);
            } else {
                C.sparseNodeNamesByIndex.emplace(static_cast<uint32_t>(v.idx),
                                                 std::move(item.second));
            }
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
        out_edges = graph_index::checked_add_size(
            out_edges,
            graph_index::checked_add_size(
                end - begin, chunk_multi[tid], "snarl graph edge count"),
            "snarl graph edge count");
        multi_links = graph_index::checked_add_size(
            multi_links, chunk_multi[tid], "snarl multi-link node count");
    }

    const uint32_t multi_links_u32 =
        graph_index::require_spqr_count_size(multi_links, "snarl multi-link node count");
    auto id2node = createNodes(bg, multi_links);
    spqr_compat::node first_mid = C.G.newNodes(multi_links_u32);
    const bool compact_names = useCompactSnarlNameTables(C);
    for (size_t i = 0; i < multi_links; ++i) {
        spqr_compat::node mid(first_mid.index() + static_cast<uint32_t>(i));
        if (compact_names) {
            setCompactTrashNode(C, mid);
        } else {
            C.node2name[mid] = "_trash";
        }
    }

    const uint32_t out_edges_u32 =
        graph_index::require_spqr_count_size(out_edges, "snarl graph edge count");
    std::vector<uint32_t> endpoints(graph_index::checked_mul_size(
        out_edges, 2u, "snarl graph endpoint array"));
    std::vector<uint8_t> edge_types(out_edges);

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
                edge_types[out_i] = packEdgePartTypes(t1, t2);
                ++out_i;
            } else {
                spqr_compat::node mid(first_mid.index() + static_cast<uint32_t>(mid_i++));
                endpoints[2 * out_i] = id2node[u].index();
                endpoints[2 * out_i + 1] = mid.index();
                edge_types[out_i] = packEdgePartTypes(t1, EdgePartType::PLUS);
                ++out_i;

                endpoints[2 * out_i] = mid.index();
                endpoints[2 * out_i + 1] = id2node[v].index();
                edge_types[out_i] = packEdgePartTypes(EdgePartType::PLUS, t2);
                ++out_i;
            }
        }
    }

    spqr_compat::edge first_edge = out_edges == 0
                                ? spqr_compat::edge(C.G.numberOfEdges())
                                : C.G.newEdgesBatchFlat(endpoints.data(), out_edges_u32);
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

bool tryBuildUltrabubbleLightGraphDirectlyFromSpqrIndex(
    const spqr_index::IndexedSpqrTree &index)
{
    auto &C = ctx();
    const std::size_t edge_count = index.graph_edges.size();
    if (C.bubbleType != Context::BubbleType::ULTRABUBBLE ||
        C.doubledUltrabubbles ||
        edge_count == 0 ||
        !hasCompleteGraphEdgeTypesForDirectBuild(index, edge_count, C.threads) ||
        !spqrIndexHasIdentityGraphOrdinals(index)) {
        return false;
    }

    const uint32_t N = index.graph_node_count();
    if (N != 0 &&
        !graph_index::fits_packed_endpoint_id(static_cast<std::uint64_t>(N - 1u))) {
        throw std::runtime_error(
            "SPQR-index ultrabubble direct graph has too many nodes for packed ultrabubble endpoints");
    }

    C.ubNumNodes = N;
    C.ubNodeNames.clear();
    C.ubNodeNames.resize(N);
    #pragma omp parallel for schedule(static) if(C.threads > 1 && N > 100000)
    for (int64_t i_i = 0; i_i < static_cast<int64_t>(N); ++i_i) {
        const uint32_t i = static_cast<uint32_t>(i_i);
        C.ubNodeNames[i] = spqrIndexGraphNodeName(index, i);
    }

    const std::vector<spqr_index::GraphEdgeRecord> &edge_records =
        index.graph_edges;
    std::vector<std::uint64_t> edge_keys(edge_count);
    std::atomic<bool> invalid_edge{false};

    auto packKey = [](uint32_t u, uint32_t v, uint8_t tu, uint8_t tv) -> std::uint64_t {
        return (static_cast<std::uint64_t>(u) << 33u) |
               (static_cast<std::uint64_t>(v) << 2u) |
               (static_cast<std::uint64_t>(tu) << 1u) |
               static_cast<std::uint64_t>(tv);
    };

    #pragma omp parallel for schedule(static) if(C.threads > 1 && edge_count > 100000)
    for (int64_t i_i = 0; i_i < static_cast<int64_t>(edge_count); ++i_i) {
        const std::size_t i = static_cast<std::size_t>(i_i);
        const auto &edge = edge_records[i];
        if (edge.src >= N || edge.dst >= N) {
            invalid_edge.store(true, std::memory_order_relaxed);
            continue;
        }

        const uint8_t packed_type = index.graph_edge_type_pairs[i];
        uint8_t tu = static_cast<uint8_t>(
            storedEndpointTypeToPart((packed_type >> 4u) & 0x0fu));
        uint8_t tv = static_cast<uint8_t>(
            storedEndpointTypeToPart(packed_type & 0x0fu));
        uint32_t u = edge.src;
        uint32_t v = edge.dst;
        if (u > v) {
            std::swap(u, v);
            std::swap(tu, tv);
        }
        edge_keys[i] = packKey(u, v, tu, tv);
    }

    if (invalid_edge.load(std::memory_order_relaxed)) {
        throw std::runtime_error("SPQR index graph edge references a missing graph node");
    }

#if defined(_OPENMP) && defined(__GLIBCXX__)
    __gnu_parallel::sort(edge_keys.begin(), edge_keys.end());
#else
    std::sort(edge_keys.begin(), edge_keys.end());
#endif
    edge_keys.erase(std::unique(edge_keys.begin(), edge_keys.end()), edge_keys.end());

    auto keyU = [](std::uint64_t key) -> uint32_t {
        return static_cast<uint32_t>(key >> 33u);
    };
    auto keyV = [](std::uint64_t key) -> uint32_t {
        return static_cast<uint32_t>((key >> 2u) & graph_index::packed_endpoint_id_max);
    };
    auto keyTu = [](std::uint64_t key) -> uint8_t {
        return static_cast<uint8_t>((key >> 1u) & 1u);
    };
    auto keyTv = [](std::uint64_t key) -> uint8_t {
        return static_cast<uint8_t>(key & 1u);
    };

    const std::size_t adj_entries = graph_index::checked_mul_size(
        edge_keys.size(), 2u, "SPQR-index ultrabubble CSR adjacency entries");
    graph_index::require_spqr_count_size(
        adj_entries, "SPQR-index ultrabubble CSR adjacency entries");

    C.ubOffset.assign(static_cast<std::size_t>(N) + 1u, 0);
    std::vector<uint8_t> endpoint_mask(N, 0);
    for (std::uint64_t key : edge_keys) {
        const uint32_t u = keyU(key);
        const uint32_t v = keyV(key);
        const uint8_t tu = keyTu(key);
        const uint8_t tv = keyTv(key);
        C.ubOffset[static_cast<std::size_t>(u) + 1u]++;
        C.ubOffset[static_cast<std::size_t>(v) + 1u]++;
        endpoint_mask[u] |= static_cast<uint8_t>(1u << tu);
        endpoint_mask[v] |= static_cast<uint8_t>(1u << tv);
    }

    for (uint32_t i = 1; i <= N; ++i) {
        C.ubOffset[i] += C.ubOffset[i - 1];
    }

    C.ubEdges.resize(C.ubOffset[N]);
    std::vector<uint32_t> cursor(C.ubOffset.begin(), C.ubOffset.end());
    for (std::uint64_t key : edge_keys) {
        const uint32_t u = keyU(key);
        const uint32_t v = keyV(key);
        const uint8_t tu = keyTu(key);
        const uint8_t tv = keyTv(key);
        C.ubEdges[cursor[u]++] = {v, tu, tv};
        C.ubEdges[cursor[v]++] = {u, tv, tu};
    }

    C.ubIsTip.resize(N);
    for (uint32_t i = 0; i < N; ++i) {
        C.ubIsTip[i] = endpoint_mask[i] != 3u;
    }

    return true;
}

void buildSuperbubbleGraph(BiGraph& bg, bool directed_only) {
    auto &C = ctx();
    const bool compact_names = useCompactSuperbubbleNameTables(C, bg, directed_only);
    if (!compact_names) {
        ensureStringNodeNames(bg);
    }
    const size_t oriented_node_count = graph_index::checked_mul_size(
        static_cast<size_t>(bg.n_nodes), 2u, "oriented superbubble graph node count");
    graph_index::require_spqr_count_size(oriented_node_count,
                                         "oriented superbubble graph node count");
    clearNodeNameTables(C);

    std::vector<spqr_compat::node> id2plus(bg.n_nodes), id2minus(bg.n_nodes);
    const spqr_compat::node first_node = C.G.newNodes(
        graph_index::require_spqr_count_size(
            oriented_node_count, "oriented superbubble graph node count"));
    const std::size_t first_idx = static_cast<std::size_t>(first_node.index());

    if (compact_names) {
        const std::size_t storage_size = graph_index::checked_add_size(
            first_idx, oriented_node_count, "oriented superbubble compact name table");
        C.nodeNumericNamesByIndex.assign(storage_size, 0);
        C.nodeNumericNameValidByIndex.assign(storage_size, 0);
    } else {
        C.name2node.reserve(oriented_node_count);
        C.node2name.reserve(oriented_node_count);
    }

    for (uint32_t i = 0; i < bg.n_nodes; ++i) {
        spqr_compat::node vp(first_node.index() + 2u * i);
        spqr_compat::node vm(first_node.index() + 2u * i + 1u);
        id2plus[i] = vp; id2minus[i] = vm;
        if (compact_names) {
            const std::uint64_t base = bg.numeric_node_names[i] << 1u;
            C.nodeNumericNamesByIndex[vp.idx] = base;
            C.nodeNumericNamesByIndex[vm.idx] = base | 1u;
            C.nodeNumericNameValidByIndex[vp.idx] = 1u;
            C.nodeNumericNameValidByIndex[vm.idx] = 1u;
        } else {
            std::string pn = bg.node_names[i] + "+", mn = bg.node_names[i] + "-";
            C.node2name[vp] = pn; C.node2name[vm] = mn;
            C.name2node[pn] = vp; C.name2node[mn] = vm;
        }
    }

    auto getNode = [&](uint32_t id, char o) -> spqr_compat::node {
        return (o == '+') ? id2plus[id] : id2minus[id];
    };

    struct DE { uint32_t u, v; bool operator<(const DE& o) const { return u!=o.u ? u<o.u : v<o.v; }
                          bool operator==(const DE& o) const { return u==o.u && v==o.v; } };
    std::vector<DE> des;
    const size_t oriented_edge_capacity = directed_only
        ? bg.links.size()
        : graph_index::checked_mul_size(
              bg.links.size(), 2u, "oriented superbubble graph raw edge capacity");
    graph_index::require_spqr_count_size(oriented_edge_capacity,
                                         "oriented superbubble graph raw edge count");
    des.reserve(oriented_edge_capacity);

    for (auto& lk : bg.links) {
        spqr_compat::node nSrc = getNode(lk.src, lk.orient_src);
        spqr_compat::node nDst = getNode(lk.dst, lk.orient_dst);
        des.push_back({nSrc.index(), nDst.index()});
        if (!directed_only) {
            spqr_compat::node nRevSrc = getNode(lk.dst, flipSign(lk.orient_dst));
            spqr_compat::node nRevDst = getNode(lk.src, flipSign(lk.orient_src));
            des.push_back({nRevSrc.index(), nRevDst.index()});
        }
    }

    std::sort(des.begin(), des.end());
    des.erase(std::unique(des.begin(), des.end()), des.end());
    graph_index::require_spqr_count_size(des.size(), "oriented superbubble graph edge count");

    if (!des.empty()) {
        std::vector<std::uint32_t> endpoints(graph_index::checked_mul_size(
            des.size(), 2u, "oriented superbubble graph endpoint array"));
        for (std::size_t i = 0; i < des.size(); ++i) {
            endpoints[2 * i] = des[i].u;
            endpoints[2 * i + 1] = des[i].v;
        }
        C.G.newEdgesBatchFlat(
            endpoints.data(),
            graph_index::require_spqr_count_size(
                des.size(), "oriented superbubble graph edge count"));
    }

    std::vector<std::string>().swap(bg.node_names);
    std::vector<std::uint64_t>().swap(bg.numeric_node_names);
    std::vector<std::uint8_t>().swap(bg.numeric_node_name_valid);
    std::vector<std::pair<std::uint32_t, std::string>>().swap(bg.string_node_names);
}

void buildSpqrGraph(BiGraph& bg) {
    auto &C = ctx();
    graph_index::require_spqr_graph(bg.n_nodes, bg.links.size(), "SPQR input graph");
    auto id2node = createNodes(bg);
    std::vector<uint32_t> endpoints(graph_index::checked_mul_size(
        bg.links.size(), 2u, "SPQR graph endpoint array"));
    std::vector<uint8_t> edge_types(bg.links.size());

    for (size_t i = 0; i < bg.links.size(); ++i) {
        const auto &lk = bg.links[i];
        endpoints[2 * i] = id2node[lk.src].index();
        endpoints[2 * i + 1] = id2node[lk.dst].index();
        edge_types[i] = packEdgePartTypes(charToType(lk.orient_src),
                                          charToType(flipSign(lk.orient_dst)));
    }

    spqr_compat::edge first_edge = edge_types.empty()
                                ? spqr_compat::edge(C.G.numberOfEdges())
                                : C.G.newEdgesBatchFlat(endpoints.data(),
                                                        static_cast<uint32_t>(edge_types.size()));
    C._edge2types.init(C.G, std::make_pair(EdgePartType::NONE, EdgePartType::NONE));
    for (size_t i = 0; i < edge_types.size(); ++i) {
        const uint8_t t = edge_types[i];
        C._edge2types[spqr_compat::edge(first_edge.index() + static_cast<uint32_t>(i))] = {
            static_cast<EdgePartType>(t >> 2),
            static_cast<EdgePartType>(t & 3)
        };
    }
}

char edgeEndpointTypeToOrient(std::uint8_t type)
{
    if (type == 1u) return '+';
    if (type == 2u) return '-';
    throw std::runtime_error("SPQR index contains an invalid BubbleFinder edge endpoint type");
}

bool spqrIndexHasIdentityGraphOrdinals(const spqr_index::IndexedSpqrTree &index)
{
    if (index.node_names.empty() && index.has_compact_numeric_graph_node_names()) {
        return true;
    }
    if (index.node_names.size() != index.graph_node_count()) {
        return false;
    }
    for (std::uint32_t i = 0; i < index.node_names.size(); ++i) {
        if (index.node_names[i] != i) return false;
    }
    return true;
}

spqr_compat::node createSnarlNodesFromIdentitySpqrIndex(
    const spqr_index::IndexedSpqrTree &index,
    std::size_t extra_nodes)
{
    auto &C = ctx();
    const std::uint32_t n = index.graph_node_count();
    const std::size_t node_storage_size = graph_index::checked_add_size(
        static_cast<std::size_t>(n), extra_nodes, "SPQR-index snarl graph node storage");
    graph_index::require_spqr_count_size(
        graph_index::checked_add_size(
            static_cast<std::size_t>(C.G.numberOfNodes()), node_storage_size,
            "SPQR-index snarl graph node count"),
        "SPQR-index snarl graph node count");

    clearCompactNodeNameTables(C);
    C.nodeNumericNamesByIndex.assign(node_storage_size, 0);
    C.nodeNumericNameValidByIndex.assign(node_storage_size, 0);
    C.sparseNodeNamesByIndex.reserve(index.graph_node_string_names.size());
    C.isTrashNodeByIndex.assign(node_storage_size, 0);

    spqr_compat::node first = C.G.newNodes(n);
    const std::uint32_t first_idx = first.index();
    const std::uint32_t graph_node_count = index.graph_node_count();
    const bool dense_numeric_names = index.graph_node_numeric_names_dense;
    const bool u64_numeric_names =
        index.graph_node_numeric_names.size() == graph_node_count;
    const bool u32_numeric_names =
        index.graph_node_numeric_names32.size() == graph_node_count;
    if (!dense_numeric_names && !u64_numeric_names && !u32_numeric_names) {
        throw std::runtime_error("SPQR index compact graph node numeric name table is incomplete");
    }
    if (dense_numeric_names && graph_node_count > 0 &&
        index.graph_node_numeric_name_base >
            std::numeric_limits<std::uint64_t>::max() -
                static_cast<std::uint64_t>(graph_node_count - 1u)) {
        throw std::overflow_error("SPQR index dense numeric graph node name overflow");
    }
    const bool has_numeric_name_validity =
        !index.graph_node_numeric_name_valid.empty();
    if (has_numeric_name_validity &&
        index.graph_node_numeric_name_valid.size() < graph_node_count) {
        throw std::runtime_error(
            "SPQR index numeric graph node validity table is truncated");
    }

    #pragma omp parallel for schedule(static) if(C.threads > 1 && graph_node_count > 100000)
    for (int64_t i_i = 0; i_i < static_cast<int64_t>(graph_node_count); ++i_i) {
        const std::uint32_t i = static_cast<std::uint32_t>(i_i);
        const std::size_t dst = static_cast<std::size_t>(first_idx) + i;
        C.nodeNumericNamesByIndex[dst] =
            dense_numeric_names
                ? index.graph_node_numeric_name_base + static_cast<std::uint64_t>(i)
                : (u64_numeric_names
                       ? index.graph_node_numeric_names[i]
                       : static_cast<std::uint64_t>(index.graph_node_numeric_names32[i]));
        C.nodeNumericNameValidByIndex[dst] =
            has_numeric_name_validity
                ? (index.graph_node_numeric_name_valid[i] != 0 ? 1u : 0u)
                : 1u;
    }

    for (const auto &item : index.graph_node_string_names) {
        const std::size_t dst = static_cast<std::size_t>(first_idx) + item.first;
        if (dst >= C.nodeNumericNameValidByIndex.size()) {
            throw std::runtime_error("SPQR index graph string name is out of range");
        }
        C.nodeNumericNameValidByIndex[dst] = 0;
        if (item.second == "_trash" ||
            isGeneratedSpqrTrashName(item.second, item.first)) {
            C.isTrashNodeByIndex[dst] = 1;
        } else {
            C.sparseNodeNamesByIndex.emplace(static_cast<std::uint32_t>(dst),
                                             item.second);
        }
    }

    return first;
}

bool hasCompleteEndpointTypesForDirectBuild(
    const std::vector<std::uint8_t> &endpoint_types,
    std::size_t edge_count,
    unsigned threads,
    bool require_nonempty)
{
    if ((require_nonempty && edge_count == 0) ||
        endpoint_types.size() != edge_count) {
        return false;
    }
    std::atomic<bool> invalid_type{false};
#if !defined(_OPENMP)
    (void)threads;
#endif
    #pragma omp parallel for schedule(static) if(threads > 1 && edge_count > 100000)
    for (int64_t i_i = 0; i_i < static_cast<int64_t>(edge_count); ++i_i) {
        const std::uint8_t packed =
            endpoint_types[static_cast<std::size_t>(i_i)];
        const std::uint8_t src = static_cast<std::uint8_t>((packed >> 4u) & 0x0fu);
        const std::uint8_t dst = static_cast<std::uint8_t>(packed & 0x0fu);
        if ((src != 1u && src != 2u) || (dst != 1u && dst != 2u)) {
            invalid_type.store(true, std::memory_order_relaxed);
        }
    }
    return !invalid_type.load(std::memory_order_relaxed);
}

bool hasCompleteGraphEdgeTypesForDirectBuild(
    const spqr_index::IndexedSpqrTree &index,
    std::size_t edge_count,
    unsigned threads)
{
    return hasCompleteEndpointTypesForDirectBuild(
        index.graph_edge_type_pairs, edge_count, threads, false);
}

bool hasCompleteRealEdgeTypesForDirectBuild(
    const spqr_index::IndexedSpqrTree &index,
    std::size_t edge_count,
    unsigned threads)
{
    return hasCompleteEndpointTypesForDirectBuild(
        index.real_edge_type_pairs, edge_count, threads, true);
}

bool addExactGraphEdgesFromSpqrIndex(
    spqr_index::IndexedSpqrTree &index,
    const std::vector<spqr_index::GraphEdgeRecord> &edge_records,
    std::uint32_t graph_node_count,
    spqr_compat::node first_node,
    bool allow_packed_direct_edge_types)
{
    auto &C = ctx();
    const std::size_t edge_count = edge_records.size();
    const std::uint32_t edge_count_u32 =
        graph_index::require_spqr_count_size(
            edge_count, "SPQR-index exact graph edge count");
    const std::uint32_t first_node_index = first_node.index();

    static_assert(std::is_standard_layout<spqr_index::GraphEdgeRecord>::value,
                  "GraphEdgeRecord must be a standard-layout src/dst pair");
    static_assert(sizeof(spqr_index::GraphEdgeRecord) == 2 * sizeof(std::uint32_t),
                  "GraphEdgeRecord must stay compatible with flat endpoint batches");
    std::vector<std::uint32_t> endpoints;
    std::uint32_t *endpoint_data = nullptr;
    if (first_node_index != 0u) {
        endpoints.resize(graph_index::checked_mul_size(
            edge_count, 2u, "SPQR-index exact graph endpoint array"));
        endpoint_data = endpoints.empty() ? nullptr : endpoints.data();
    }
    std::atomic<bool> invalid_edge{false};

    #pragma omp parallel for schedule(static) if(C.threads > 1 && edge_count > 100000)
    for (int64_t i_i = 0; i_i < static_cast<int64_t>(edge_count); ++i_i) {
        const std::size_t i = static_cast<std::size_t>(i_i);
        const auto &edge = edge_records[i];
        if (edge.src >= graph_node_count || edge.dst >= graph_node_count) {
            invalid_edge.store(true, std::memory_order_relaxed);
            continue;
        }
        if (first_node_index != 0u) {
            endpoint_data[2 * i] = first_node_index + edge.src;
            endpoint_data[2 * i + 1] = first_node_index + edge.dst;
        }
    }
    if (invalid_edge.load(std::memory_order_relaxed)) {
        throw std::runtime_error("SPQR index graph edge references a missing graph node");
    }

    const std::uint32_t *batch_endpoints = first_node_index == 0u
        ? reinterpret_cast<const std::uint32_t *>(edge_records.data())
        : endpoints.data();
    spqr_compat::edge first_edge = edge_count == 0
                                ? spqr_compat::edge(C.G.numberOfEdges())
                                : C.G.newEdgesBatchFlat(
                                      batch_endpoints, edge_count_u32);
    const std::uint32_t first_edge_index = first_edge.index();
    if (allow_packed_direct_edge_types &&
        edge_count != 0 &&
        first_edge_index == 0u) {
        C.directSpqrGraphEdgeTypePairs =
            std::move(index.graph_edge_type_pairs);
        return true;
    }

    C._edge2types.init(C.G, std::make_pair(EdgePartType::NONE, EdgePartType::NONE));
    auto edge_type_begin = C._edge2types.begin();
    auto *edge_type_data =
        C._edge2types.size() == 0 ? nullptr : &*edge_type_begin;
    #pragma omp parallel for schedule(static) if(C.threads > 1 && edge_count > 100000)
    for (int64_t i_i = 0; i_i < static_cast<int64_t>(edge_count); ++i_i) {
        const std::size_t i = static_cast<std::size_t>(i_i);
        const std::uint8_t t = index.graph_edge_type_pairs[i];
        edge_type_data[first_edge_index + static_cast<std::uint32_t>(i)] =
            storedEndpointTypesToParts(t);
    }

    return true;
}

bool tryBuildExactGraphDirectlyFromSpqrIndex(spqr_index::IndexedSpqrTree &index)
{
    auto &C = ctx();
    const std::uint32_t graph_node_count = index.graph_node_count();
    const std::vector<spqr_index::GraphEdgeRecord> &edge_records =
        index.graph_edges;
    const std::size_t edge_count = edge_records.size();
    if (C.bubbleType != Context::BubbleType::SNARL ||
        C.spCompressMode != Context::SpCompressMode::MacroDirect ||
        (graph_node_count != 0 &&
         !graph_index::fits_packed_endpoint_id(
             static_cast<std::uint64_t>(graph_node_count - 1))) ||
        !index.has_compact_numeric_graph_node_names() ||
        !spqrIndexHasIdentityGraphOrdinals(index) ||
        !hasCompleteGraphEdgeTypesForDirectBuild(
            index, edge_count, C.threads)) {
        return false;
    }

    spqr_compat::node first_node = createSnarlNodesFromIdentitySpqrIndex(index, 0);
    return addExactGraphEdgesFromSpqrIndex(
        index, edge_records, graph_node_count, first_node, true);
}

std::string spqrIndexGraphNodeName(const spqr_index::IndexedSpqrTree &index,
                                   std::uint32_t ordinal)
{
    const std::uint32_t n = index.graph_node_count();
    if (ordinal >= n) {
        throw std::runtime_error("SPQR index graph node ordinal is out of range");
    }

    if (index.node_names.size() == n) {
        std::string name = index.name(index.node_names[ordinal]);
        if (isGeneratedSpqrTrashName(name, ordinal)) return "_trash";
        return name;
    }

    if (index.has_compact_numeric_graph_node_names()) {
        const auto sparse_string = std::lower_bound(
            index.graph_node_string_names.begin(),
            index.graph_node_string_names.end(),
            ordinal,
            [](const auto &item, std::uint32_t value) {
                return item.first < value;
            });
        const bool has_sparse_string =
            sparse_string != index.graph_node_string_names.end() &&
            sparse_string->first == ordinal;
        bool numeric_valid = !has_sparse_string;
        if (numeric_valid && !index.graph_node_numeric_name_valid.empty()) {
            if (ordinal >= index.graph_node_numeric_name_valid.size()) {
                throw std::runtime_error(
                    "SPQR index numeric graph node validity table is truncated");
            }
            numeric_valid = index.graph_node_numeric_name_valid[ordinal] != 0;
        }
        if (numeric_valid) {
            const std::uint64_t numeric = index.graph_node_numeric_name_at(ordinal);
            if (index.uses_oriented_numeric_graph_node_names()) {
                return spqr_index::IndexedSpqrTree::oriented_numeric_graph_node_name(numeric);
            }
            return std::to_string(numeric);
        }
        if (has_sparse_string) {
            if (isGeneratedSpqrTrashName(sparse_string->second, ordinal)) return "_trash";
            return sparse_string->second;
        }
    }

    return std::to_string(ordinal);
}

bool tryBuildNamedExactGraphDirectlyFromSpqrIndex(
    spqr_index::IndexedSpqrTree &index)
{
    auto &C = ctx();

    const std::uint32_t graph_node_count = index.graph_node_count();
    const std::vector<spqr_index::GraphEdgeRecord> &edge_records =
        index.graph_edges;
    const std::size_t edge_count = edge_records.size();
    graph_index::require_spqr_count_size(edge_count,
                                         "SPQR-index exact graph edge count");

    const bool indexBackedNames =
        C.inputFormat == Context::InputFormat::SpqrIndex &&
        C.spqrIndex == nullptr &&
        spqr_index::graph_profile_is_oriented_double(C.spqrIndexInputGraphView);

    C.directSpqrInputGraphUsesIndexNames = indexBackedNames;
    C.directSpqrInputGraphEdgesMaterialized = !indexBackedNames;
    clearNodeNameTables(C);

    if (!indexBackedNames) {
        C.node2name.reserve(graph_node_count);
        C.name2node.reserve(graph_node_count);
    }

    if (!indexBackedNames && edge_records.empty() && index.graph_edge_count() != 0) {
        return false;
    }
    if (!indexBackedNames && edge_count != 0 &&
        !hasCompleteGraphEdgeTypesForDirectBuild(
            index, edge_count, C.threads)) {
        return false;
    }
    spqr_compat::node first_node = C.G.newNodes(graph_node_count);

    if (first_node.index() != 0u && indexBackedNames) {
        throw std::runtime_error(
            "SPQR-index node-only graph reconstruction requires an empty target graph");
    }

    if (!indexBackedNames) {
        for (std::uint32_t i = 0; i < graph_node_count; ++i) {
            spqr_compat::node v(first_node.index() + i);
            std::string name = spqrIndexGraphNodeName(index, i);
            auto it = C.node2name.emplace(v, std::move(name)).first;
            if (it->second != "_trash") {
                C.name2node.emplace(it->second, v);
            }
        }
    }

    if (indexBackedNames) {
        C._edge2types.init(C.G, std::make_pair(EdgePartType::NONE, EdgePartType::NONE));
        return true;
    }

    return addExactGraphEdgesFromSpqrIndex(
        index,
        edge_records,
        graph_node_count,
        first_node,
        C.bubbleType == Context::BubbleType::SNARL &&
            C.spCompressMode == Context::SpCompressMode::MacroDirect);
}

bool tryBuildSnarlGraphDirectlyFromSpqrIndex(const spqr_index::IndexedSpqrTree &index)
{
    auto &C = ctx();
    const bool use_graph_edges = !index.graph_edges.empty();
    const std::vector<spqr_index::GraphEdgeRecord> &graph_edge_records =
        index.graph_edges;
    const std::size_t link_count = use_graph_edges
        ? graph_edge_records.size()
        : index.real_edge_count();
    if (C.bubbleType != Context::BubbleType::SNARL ||
        !useCompactSnarlNameTables(C) ||
        !index.has_compact_numeric_graph_node_names() ||
        !spqrIndexHasIdentityGraphOrdinals(index)) {
        return false;
    }
    if (use_graph_edges) {
        if (!hasCompleteGraphEdgeTypesForDirectBuild(index, link_count, C.threads))
            return false;
    } else if (!hasCompleteRealEdgeTypesForDirectBuild(index, link_count, C.threads)) {
        return false;
    }

    const std::uint32_t graph_node_count = index.graph_node_count();
    const spqr_index::GraphEdgeRecord *edge_records =
        use_graph_edges
            ? graph_edge_records.data()
            : (index.has_compact_real_edge_endpoints()
                   ? index.real_edge_endpoints.data()
                   : nullptr);
    auto edge_at = [&](std::size_t i) -> spqr_index::GraphEdgeRecord {
        return edge_records != nullptr
            ? edge_records[i]
            : index.real_edge_endpoint_at(static_cast<std::uint32_t>(i));
    };
    auto link_endpoint_types = [&](std::size_t i) -> std::pair<EdgePartType, EdgePartType> {
        const std::uint8_t packed = use_graph_edges
            ? index.graph_edge_type_pairs[i]
            : index.real_edge_type_pairs[i];
        return storedEndpointTypesToParts(packed);
    };
    auto encodePair = [](std::uint32_t a, std::uint32_t b) -> std::uint64_t {
        return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
    };

    std::atomic<bool> invalid_edge{false};
    const bool use_multi_flags =
        use_graph_edges &&
        index.graph_edge_multi_flags.size() == link_count;
    std::vector<std::uint64_t> pair_keys;
    if (!use_multi_flags) pair_keys.resize(link_count);
    #pragma omp parallel for schedule(static) if(C.threads > 1 && link_count > 100000)
    for (int64_t i_i = 0; i_i < static_cast<int64_t>(link_count); ++i_i) {
        const std::size_t i = static_cast<std::size_t>(i_i);
        const auto edge = edge_at(i);
        const std::uint32_t src = edge.src;
        const std::uint32_t dst = edge.dst;
        if (src >= graph_node_count || dst >= graph_node_count) {
            invalid_edge.store(true, std::memory_order_relaxed);
            continue;
        }
        const std::uint32_t a = std::min(src, dst);
        const std::uint32_t b = std::max(src, dst);
        if (!use_multi_flags) pair_keys[i] = encodePair(a, b);
    }
    if (invalid_edge.load(std::memory_order_relaxed)) {
        throw std::runtime_error("SPQR index real edge references a missing graph node");
    }

    std::vector<std::uint64_t> multis;
    if (!use_multi_flags) {
#if defined(_OPENMP) && defined(__GLIBCXX__)
        __gnu_parallel::sort(pair_keys.begin(), pair_keys.end());
#else
        std::sort(pair_keys.begin(), pair_keys.end());
#endif

        for (std::size_t i = 1; i < pair_keys.size(); ++i) {
            if (pair_keys[i] == pair_keys[i - 1] &&
                (multis.empty() || multis.back() != pair_keys[i])) {
                multis.push_back(pair_keys[i]);
            }
        }
        std::vector<std::uint64_t>().swap(pair_keys);
    }

    auto is_multi = [&](std::uint32_t a, std::uint32_t b) -> bool {
        const std::uint32_t lo = std::min(a, b);
        const std::uint32_t hi = std::max(a, b);
        return std::binary_search(multis.begin(), multis.end(), encodePair(lo, hi));
    };

    const std::size_t worker_count =
        (C.threads > 1 && link_count > 100000)
            ? std::min<std::size_t>(static_cast<std::size_t>(C.threads), link_count)
            : 1;

    std::vector<std::uint8_t> link_is_multi;
    std::vector<std::size_t> chunk_multi(worker_count, 0);
    std::vector<std::size_t> chunk_out_base(worker_count, 0);
    std::vector<std::size_t> chunk_mid_base(worker_count, 0);

    if (use_multi_flags) {
        #pragma omp parallel for schedule(static) if(worker_count > 1)
        for (int64_t tid_i = 0; tid_i < static_cast<int64_t>(worker_count); ++tid_i) {
            const std::size_t tid = static_cast<std::size_t>(tid_i);
            const std::size_t begin = (link_count * tid) / worker_count;
            const std::size_t end = (link_count * (tid + 1)) / worker_count;
            std::size_t local_multi = 0;
            for (std::size_t i = begin; i < end; ++i) {
                local_multi += index.graph_edge_multi_flags[i] != 0 ? 1u : 0u;
            }
            chunk_multi[tid] = local_multi;
        }
    } else if (!multis.empty()) {
        link_is_multi.resize(link_count, 0);
        #pragma omp parallel for schedule(static) if(worker_count > 1)
        for (int64_t tid_i = 0; tid_i < static_cast<int64_t>(worker_count); ++tid_i) {
            const std::size_t tid = static_cast<std::size_t>(tid_i);
            const std::size_t begin = (link_count * tid) / worker_count;
            const std::size_t end = (link_count * (tid + 1)) / worker_count;
            std::size_t local_multi = 0;
            for (std::size_t i = begin; i < end; ++i) {
                const auto edge = edge_at(i);
                const bool multi = is_multi(edge.src, edge.dst);
                link_is_multi[i] = static_cast<std::uint8_t>(multi);
                local_multi += multi ? 1u : 0u;
            }
            chunk_multi[tid] = local_multi;
        }
    }

    std::size_t multi_links = 0;
    std::size_t out_edges = 0;
    for (std::size_t tid = 0; tid < worker_count; ++tid) {
        const std::size_t begin = (link_count * tid) / worker_count;
        const std::size_t end = (link_count * (tid + 1)) / worker_count;
        chunk_out_base[tid] = out_edges;
        chunk_mid_base[tid] = multi_links;
        out_edges += (end - begin) + chunk_multi[tid];
        multi_links += chunk_multi[tid];
    }

    const std::uint32_t multi_links_u32 =
        graph_index::require_spqr_count_size(multi_links,
                                             "SPQR-index snarl multi-link node count");
    spqr_compat::node first_node =
        createSnarlNodesFromIdentitySpqrIndex(index, multi_links);
    spqr_compat::node first_mid = C.G.newNodes(multi_links_u32);
    for (std::size_t i = 0; i < multi_links; ++i) {
        spqr_compat::node mid(first_mid.index() + static_cast<std::uint32_t>(i));
        setCompactTrashNode(C, mid);
    }

    const std::uint32_t out_edges_u32 =
        graph_index::require_spqr_count_size(out_edges,
                                             "SPQR-index snarl graph edge count");
    std::vector<std::uint32_t> endpoints(graph_index::checked_mul_size(
        out_edges, 2u, "SPQR-index snarl graph endpoint array"));
    std::vector<std::uint8_t> edge_types(out_edges);

    #pragma omp parallel for schedule(static) if(worker_count > 1)
    for (int64_t tid_i = 0; tid_i < static_cast<int64_t>(worker_count); ++tid_i) {
        const std::size_t tid = static_cast<std::size_t>(tid_i);
        const std::size_t begin = (link_count * tid) / worker_count;
        const std::size_t end = (link_count * (tid + 1)) / worker_count;
        std::size_t out_i = chunk_out_base[tid];
        std::size_t mid_i = chunk_mid_base[tid];

        for (std::size_t i = begin; i < end; ++i) {
            auto types = link_endpoint_types(i);
            EdgePartType t1 = types.first;
            EdgePartType t2 = types.second;
            const auto edge = edge_at(i);
            std::uint32_t u = edge.src;
            std::uint32_t v = edge.dst;
            if (u > v) {
                std::swap(u, v);
                std::swap(t1, t2);
            }

            const bool multi = use_multi_flags
                ? index.graph_edge_multi_flags[i] != 0
                : (!link_is_multi.empty() && link_is_multi[i] != 0);
            if (!multi) {
                endpoints[2 * out_i] = first_node.index() + u;
                endpoints[2 * out_i + 1] = first_node.index() + v;
                edge_types[out_i] = packEdgePartTypes(t1, t2);
                ++out_i;
            } else {
                spqr_compat::node mid(first_mid.index() + static_cast<std::uint32_t>(mid_i++));
                endpoints[2 * out_i] = first_node.index() + u;
                endpoints[2 * out_i + 1] = mid.index();
                edge_types[out_i] = packEdgePartTypes(t1, EdgePartType::PLUS);
                ++out_i;

                endpoints[2 * out_i] = mid.index();
                endpoints[2 * out_i + 1] = first_node.index() + v;
                edge_types[out_i] = packEdgePartTypes(EdgePartType::PLUS, t2);
                ++out_i;
            }
        }
    }

    spqr_compat::edge first_edge = edge_types.empty()
                                ? spqr_compat::edge(C.G.numberOfEdges())
                                : C.G.newEdgesBatchFlat(endpoints.data(), out_edges_u32);
    C._edge2types.init(C.G, std::make_pair(EdgePartType::NONE, EdgePartType::NONE));
    #pragma omp parallel for schedule(static) if(C.threads > 1 && edge_types.size() > 100000)
    for (int64_t i_i = 0; i_i < static_cast<int64_t>(edge_types.size()); ++i_i) {
        const std::size_t i = static_cast<std::size_t>(i_i);
        std::uint8_t t = edge_types[i];
        C._edge2types[spqr_compat::edge(first_edge.index() + static_cast<std::uint32_t>(i))] = {
            static_cast<EdgePartType>(t >> 2),
            static_cast<EdgePartType>(t & 3)
        };
    }

    return true;
}

BiGraph buildBiGraphFromSpqrIndex(const spqr_index::IndexedSpqrTree &index)
{
    const bool use_graph_edges = !index.graph_edges.empty();
    const std::vector<spqr_index::GraphEdgeRecord> &graph_edge_records =
        index.graph_edges;
    const std::size_t graph_edge_count = use_graph_edges
        ? graph_edge_records.size()
        : index.real_edge_count();
    const spqr_index::GraphEdgeRecord *edge_records =
        use_graph_edges
            ? graph_edge_records.data()
            : (index.has_compact_real_edge_endpoints()
                   ? index.real_edge_endpoints.data()
                   : nullptr);
    auto edge_at = [&](std::size_t i) -> spqr_index::GraphEdgeRecord {
        return edge_records != nullptr
            ? edge_records[i]
            : index.real_edge_endpoint_at(static_cast<std::uint32_t>(i));
    };
    if (use_graph_edges &&
        !hasCompleteGraphEdgeTypesForDirectBuild(index, graph_edge_count, ctx().threads)) {
        throw std::runtime_error(
            ".spqr/.spqri input contains raw graph edges without complete BF_EDGE_TYPE metadata; "
            "recreate it with the updated 'spqr-tree' command");
    }
    if (!use_graph_edges &&
        !hasCompleteRealEdgeTypesForDirectBuild(index, graph_edge_count, ctx().threads)) {
        throw std::runtime_error(
            ".spqr/.spqri input does not contain complete BF_EDGE_TYPE metadata; "
            "recreate it with the updated 'spqr-tree' command");
    }

    BiGraph bg;
    bg.n_nodes = graph_index::require_spqr_count_size(
        index.graph_node_count(), "SPQR-index reconstructed graph node count");
    const bool compact_numeric_names = index.has_compact_numeric_graph_node_names();
    if (compact_numeric_names) {
        index.copy_graph_node_numeric_names(bg.numeric_node_names);
        if (index.graph_node_numeric_name_valid.size() == index.graph_node_count()) {
            bg.numeric_node_name_valid = index.graph_node_numeric_name_valid;
        } else {
            bg.numeric_node_name_valid.assign(index.graph_node_count(), 1u);
            for (const auto &item : index.graph_node_string_names) {
                if (item.first < bg.numeric_node_name_valid.size()) {
                    bg.numeric_node_name_valid[item.first] = 0;
                }
            }
        }
        bg.string_node_names = index.graph_node_string_names;
    } else {
        bg.node_names.reserve(index.node_names.size());
    }

    bool identity_node_names = true;
    if (index.node_names.empty() && compact_numeric_names) {
        identity_node_names = true;
    } else {
        for (std::uint32_t i = 0; i < index.node_names.size(); ++i) {
            const std::uint32_t name_id = index.node_names[i];
            if (name_id != i) identity_node_names = false;
            if (!compact_numeric_names) {
                bg.node_names.push_back(index.name(name_id));
            }
        }
    }
    if (!compact_numeric_names && bg.node_names.size() != bg.n_nodes) {
        throw std::runtime_error("SPQR index graph node table is incomplete");
    }

    std::unordered_map<std::uint32_t, std::uint32_t> ordinal_by_name;
    if (!identity_node_names) {
        ordinal_by_name.reserve(index.node_names.size());
        for (std::uint32_t i = 0; i < index.node_names.size(); ++i) {
            ordinal_by_name[index.node_names[i]] = i;
        }
    }

    bg.links.reserve(graph_edge_count);
    for (std::uint32_t i = 0; i < graph_edge_count; ++i) {
        const auto edge = edge_at(i);
        std::uint32_t src = spqr_index::invalid_id;
        std::uint32_t dst = spqr_index::invalid_id;
        if (identity_node_names) {
            if (edge.src >= bg.n_nodes || edge.dst >= bg.n_nodes) {
                throw std::runtime_error("SPQR index real edge references a missing graph node");
            }
            src = edge.src;
            dst = edge.dst;
        } else {
            const auto src_it = ordinal_by_name.find(edge.src);
            const auto dst_it = ordinal_by_name.find(edge.dst);
            if (src_it == ordinal_by_name.end() || dst_it == ordinal_by_name.end()) {
                throw std::runtime_error("SPQR index real edge references a missing graph node");
            }
            src = src_it->second;
            dst = dst_it->second;
        }

        char src_orient = '+';
        char stored_dst_type = '+';
        const auto types = use_graph_edges
            ? std::make_pair(
                  static_cast<std::uint8_t>((index.graph_edge_type_pairs[i] >> 4) & 0x0fu),
                  static_cast<std::uint8_t>(index.graph_edge_type_pairs[i] & 0x0fu))
            : index.real_edge_endpoint_types(i);
        src_orient = edgeEndpointTypeToOrient(types.first);
        stored_dst_type = edgeEndpointTypeToOrient(types.second);
        bg.links.push_back(BiLink{
            src,
            dst,
            src_orient,
            flipSign(stored_dst_type)
        });
    }

    return bg;
}

void readSpqrIndexGraph()
{
    auto &C = ctx();
    if (C.graphPath.empty())
        throw std::runtime_error("SPQR-index input needs -g <file>");
    C.directSpqrGraphComponents.clear();
    C.directSpqrGraphComponentNodes.clear();
    C.directSpqrGraphComponentEdges.clear();
    C.directSpqrGraphEdgeTypePairs.clear();
    C.directSpqrGraphComponentNodesIdentity = false;
    C.directSpqrGraphComponentEdgesIdentity = false;

    spqr_index::LoadOptions loadOptions;
    const bool keepDirectInputIndex =
        C.spqrHaplotypes ||
        (C.bubbleType == Context::BubbleType::SNARL &&
         C.spCompressMode == Context::SpCompressMode::Off);
    loadOptions.skip_graph_edge_tables_for_oriented_views =
        C.bubbleType == Context::BubbleType::SUPERBUBBLE;
    loadOptions.skip_macro_tree_cache = true;
    loadOptions.graph_only =
        C.bubbleType == Context::BubbleType::SNARL &&
        !keepDirectInputIndex;
    loadOptions.load_graph_components =
        C.bubbleType == Context::BubbleType::SNARL;
    loadOptions.load_haplotypes = C.spqrHaplotypes;
    if (C.bubbleType == Context::BubbleType::SUPERBUBBLE)
    {
        loadOptions.eager_block_hash_lookup = true;
        loadOptions.eager_lookup_view_filter =
            C.directedSuperbubbles
                ? spqr_index::EagerLookupViewFilter::OrientedDirected
                : spqr_index::EagerLookupViewFilter::OrientedBidirected;
    }

    auto loaded = std::make_unique<spqr_index::IndexedSpqrTree>(
        spqr_index::IndexedSpqrTree::load(C.graphPath, loadOptions));
    C.spqrIndexInputLoaded = true;
    C.spqrIndexInputGraphView =
        spqr_index::canonical_graph_profile(loaded->graph_view);

    const std::string view = C.spqrIndexInputGraphView;
    const std::uint64_t indexedGraphEdges =
        !loaded->graph_edges.empty()
            ? static_cast<std::uint64_t>(loaded->graph_edges.size())
            : loaded->graph_edge_count_hint;
    const bool viewMatchesSuperbubble =
        C.bubbleType == Context::BubbleType::SUPERBUBBLE &&
        spqr_index::graph_profile_matches_oriented_double(
            view, C.directedSuperbubbles);
    const bool hasDirectGraphEdges =
        !loaded->graph_edges.empty();
    if (!hasDirectGraphEdges &&
        (!viewMatchesSuperbubble ||
         loaded->real_edge_count() != indexedGraphEdges)) {
        throw std::runtime_error(
            ".spqr/.spqri direct graph input requires exact graph-edge tables; "
            "rebuild the cache with 'spqr-tree --spqr-profile " + view +
            "' command.");
    }
    if (!spqr_index::graph_profile_is_raw(view) &&
        !(spqr_index::graph_profile_is_parallel_subdivided(view) &&
          C.bubbleType == Context::BubbleType::SNARL) &&
        !viewMatchesSuperbubble) {
        throw std::runtime_error(
            ".spqr/.spqri direct graph input supports raw graph profiles, "
            "endpoint-typed parallel-subdivided profiles for the snarls command, "
            "and oriented-double profiles for the "
            "matching superbubble command; "
            "this index uses profile '" + view +
            "'. Use the original graph with --spqr-index, or rebuild the cache with a compatible --spqr-profile.");
    }
    if (spqr_index::graph_profile_is_parallel_subdivided(view) &&
        C.bubbleType == Context::BubbleType::SNARL &&
        C.spCompressMode != Context::SpCompressMode::MacroDirect &&
        C.spCompressMode != Context::SpCompressMode::Off) {
        throw std::runtime_error(
            ".spqr/.spqri endpoint-typed parallel-subdivided direct input is currently supported only by "
            "the default snarls macro-direct mode or --sp-compress off.");
    }

    const bool dropInputIndex =
        C.spqrIndex == nullptr &&
        loadOptions.graph_only;

    auto dropLoadedGraphEdgeTablesIfUsed = [&]() {
        if (!C.directSpqrInputGraphEdgesMaterialized ||
            loaded->graph_edges.empty()) {
            return;
        }
        loaded->drop_graph_edge_tables();
    };

    auto finishLoadedInput = [&]() {
        dropLoadedGraphEdgeTablesIfUsed();
        if (!C.spqrIndex && !dropInputIndex) {
            C.spqrIndexHasCompleteBubbleEdgeTypes =
                loaded->has_complete_bubble_edge_types();
            C.spqrIndexBubbleEdgeTypesChecked = true;
            C.spqrIndex = std::move(loaded);
        }
    };

    auto retainDirectGraphComponents = [&](spqr_index::IndexedSpqrTree &index,
                                           bool exact_edge_order) {
        if (!exact_edge_order ||
            C.bubbleType != Context::BubbleType::SNARL ||
            !index.has_graph_components() ||
            C.G.numberOfNodes() != index.graph_node_count() ||
            C.G.numberOfEdges() != index.graph_edge_count()) {
            return;
        }
        C.directSpqrGraphComponents = std::move(index.graph_components);
        C.directSpqrGraphComponentNodes = std::move(index.graph_component_nodes);
        C.directSpqrGraphComponentEdges = std::move(index.graph_component_edges);
        C.directSpqrGraphComponentNodesIdentity =
            index.graph_component_nodes_identity;
        C.directSpqrGraphComponentEdgesIdentity =
            index.graph_component_edges_identity;
    };

    if (spqr_index::graph_profile_is_parallel_subdivided(view)) {
        bool exact_graph = tryBuildExactGraphDirectlyFromSpqrIndex(*loaded);
        if (!exact_graph) {
            exact_graph = tryBuildNamedExactGraphDirectlyFromSpqrIndex(*loaded);
        }
        if (!exact_graph) {
            BiGraph bg = buildBiGraphFromSpqrIndex(*loaded);
            buildSpqrGraph(bg);
        }
        retainDirectGraphComponents(*loaded, exact_graph);
        finishLoadedInput();
        return;
    }

    if (viewMatchesSuperbubble) {
        if (!tryBuildNamedExactGraphDirectlyFromSpqrIndex(*loaded)) {
            throw std::runtime_error(
                ".spqr/.spqri oriented-double profile direct input cannot be reconstructed "
                "from this cache; rebuild it with 'spqr-tree --spqr-profile " +
                view + "' command.");
        }
        finishLoadedInput();
        return;
    }

    if (tryBuildSnarlGraphDirectlyFromSpqrIndex(*loaded)) {
        finishLoadedInput();
        return;
    }

    if (spqr_index::graph_profile_is_raw(view) &&
        C.bubbleType == Context::BubbleType::ULTRABUBBLE &&
        !C.doubledUltrabubbles &&
        tryBuildUltrabubbleLightGraphDirectlyFromSpqrIndex(*loaded)) {
        return;
    }

    BiGraph bg = buildBiGraphFromSpqrIndex(*loaded);
    finishLoadedInput();

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
            buildSuperbubbleGraph(bg, C.directedSuperbubbles);
            break;
        default:
            throw std::runtime_error(".spqr/.spqri graph input is only supported for bubble commands");
    }
}

}

namespace {

BiGraph parse_graph_input(const std::string& path, int threads) {
    if (spqr_index::detail::ends_with(path, ".gbz")) {
        const auto &C = ctx();
        const bool load_haplotypes =
            C.spqrHaplotypes &&
            C.bubbleType == Context::BubbleType::SPQR_TREE_ONLY &&
            spqr_index::detail::ends_with(C.outputPath, ".spqri");
        logger::info("GBZ parser: reading '{}'", path);
        auto bg = GBZParser::parse_file(path, threads, load_haplotypes);
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
    C.inputHaplotypePaths.clear();
    C.inputHaplotypeSteps.clear();
    if (bg.n_nodes == 0) { logger::info("Empty graph"); return; }
    if (!bg.haplotype_paths.empty()) {
        C.inputHaplotypePaths.reserve(bg.haplotype_paths.size());
        for (auto &path_rec : bg.haplotype_paths) {
            InputHaplotypePath rec;
            rec.name = std::move(path_rec.name);
            rec.sample = std::move(path_rec.sample);
            rec.locus = std::move(path_rec.locus);
            rec.haplotype = path_rec.haplotype;
            rec.phase_block = path_rec.phase_block;
            rec.sense = path_rec.sense;
            rec.step_begin = path_rec.step_begin;
            rec.step_end = path_rec.step_end;
            C.inputHaplotypePaths.push_back(std::move(rec));
        }
        C.inputHaplotypeSteps.reserve(bg.haplotype_steps.size());
        for (const auto &step : bg.haplotype_steps) {
            C.inputHaplotypeSteps.push_back({step.node, step.is_reverse});
        }
        std::vector<BiHaplotypePath>().swap(bg.haplotype_paths);
        std::vector<BiHaplotypeStep>().swap(bg.haplotype_steps);
    }

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
            if (spqr_index::graph_profile_is_parallel_subdivided(C.spqrTreeView)) {
                buildSnarlGraph(bg);
            } else if (spqr_index::graph_profile_is_oriented_bidirected(C.spqrTreeView)) {
                buildSuperbubbleGraph(bg, false);
            } else if (spqr_index::graph_profile_is_oriented_directed(C.spqrTreeView)) {
                buildSuperbubbleGraph(bg, true);
            } else {
                buildSpqrGraph(bg);
            }
            break;
        default:
            break;
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

static bool tryLoadGraphDegreesFromSpqrIndex(
    Context &C,
    const spqr_index::IndexedSpqrTree &index)
{
    const std::uint32_t graph_node_count = index.graph_node_count();
    if (C.G.numberOfNodes() != graph_node_count ||
        C.inDeg.size() < graph_node_count ||
        C.outDeg.size() < graph_node_count) {
        return false;
    }

    int *in_data = C.inDeg.size() == 0 ? nullptr : &*C.inDeg.begin();
    int *out_data = C.outDeg.size() == 0 ? nullptr : &*C.outDeg.begin();
    auto has_u8_degree_table = [graph_node_count](
        const std::vector<std::uint8_t> &base,
        const std::vector<std::uint32_t> &overflow_nodes,
        const std::vector<std::uint32_t> &overflow_values) {
        return base.size() == graph_node_count &&
               overflow_nodes.size() == overflow_values.size();
    };
    auto has_degree_fragment = [](
        const std::vector<std::uint8_t> &base,
        const std::vector<std::uint32_t> &overflow_nodes,
        const std::vector<std::uint32_t> &overflow_values,
        const std::vector<std::uint32_t> &raw) {
        return !base.empty() || !overflow_nodes.empty() ||
               !overflow_values.empty() || !raw.empty();
    };

    const bool has_raw_in_degrees =
        index.graph_node_in_degrees.size() == graph_node_count;
    const bool has_raw_out_degrees =
        index.graph_node_out_degrees.size() == graph_node_count;
    const bool has_u8_in_degrees =
        has_u8_degree_table(
            index.graph_node_in_degrees8,
            index.graph_node_in_degree_overflow_nodes,
            index.graph_node_in_degree_overflow_values);
    const bool has_u8_out_degrees =
        has_u8_degree_table(
            index.graph_node_out_degrees8,
            index.graph_node_out_degree_overflow_nodes,
            index.graph_node_out_degree_overflow_values);
    const bool has_stored_in_degrees =
        has_raw_in_degrees || has_u8_in_degrees;
    const bool has_stored_out_degrees =
        has_raw_out_degrees || has_u8_out_degrees;

    if (has_stored_in_degrees != has_stored_out_degrees ||
        (!has_stored_in_degrees &&
         (has_degree_fragment(
              index.graph_node_in_degrees8,
              index.graph_node_in_degree_overflow_nodes,
              index.graph_node_in_degree_overflow_values,
              index.graph_node_in_degrees) ||
          has_degree_fragment(
              index.graph_node_out_degrees8,
              index.graph_node_out_degree_overflow_nodes,
              index.graph_node_out_degree_overflow_values,
              index.graph_node_out_degrees)))) {
        throw std::runtime_error(
            "SPQR index graph node degree table does not match graph node count");
    }

    if (!has_stored_in_degrees) {
        return false;
    }

    constexpr int max_degree = std::numeric_limits<int>::max();
    auto copy_degrees = [graph_node_count, max_degree, threads = C.threads](
        const std::vector<std::uint8_t> &base,
        const std::vector<std::uint32_t> &overflow_nodes,
        const std::vector<std::uint32_t> &overflow_values,
        const std::vector<std::uint32_t> &raw,
        int *dst) {
        if (raw.size() == graph_node_count) {
            std::atomic<bool> invalid_degree{false};
            #pragma omp parallel for schedule(static) if(threads > 1 && graph_node_count > 1000000u)
            for (int64_t i_i = 0; i_i < static_cast<int64_t>(graph_node_count); ++i_i) {
                const std::uint32_t i = static_cast<std::uint32_t>(i_i);
                const std::uint32_t degree = raw[i];
                if (degree > static_cast<std::uint32_t>(max_degree)) {
                    invalid_degree.store(true, std::memory_order_relaxed);
                    continue;
                }
                dst[i] = static_cast<int>(degree);
            }
            if (invalid_degree.load(std::memory_order_relaxed)) {
                throw std::runtime_error(
                    "SPQR index graph node degree exceeds BubbleFinder degree storage");
            }
            return;
        }
        if (base.size() != graph_node_count ||
            overflow_nodes.size() != overflow_values.size()) {
            throw std::runtime_error(
                "SPQR index graph node degree table does not match graph node count");
        }
        #pragma omp parallel for schedule(static) if(threads > 1 && graph_node_count > 1000000u)
        for (int64_t i_i = 0; i_i < static_cast<int64_t>(graph_node_count); ++i_i) {
            const std::uint32_t i = static_cast<std::uint32_t>(i_i);
            dst[i] = static_cast<int>(base[i]);
        }
        for (std::uint32_t i = 0; i < overflow_nodes.size(); ++i) {
            const std::uint32_t node = overflow_nodes[i];
            const std::uint32_t degree = overflow_values[i];
            if (node >= graph_node_count ||
                degree > static_cast<std::uint32_t>(max_degree)) {
                throw std::runtime_error(
                    "SPQR index graph node degree table is invalid");
            }
            dst[node] = static_cast<int>(degree);
        }
    };

    copy_degrees(index.graph_node_in_degrees8,
                 index.graph_node_in_degree_overflow_nodes,
                 index.graph_node_in_degree_overflow_values,
                 index.graph_node_in_degrees,
                 in_data);
    copy_degrees(index.graph_node_out_degrees8,
                 index.graph_node_out_degree_overflow_nodes,
                 index.graph_node_out_degree_overflow_values,
                 index.graph_node_out_degrees,
                 out_data);
    return true;
}

void readGraph() {
    auto &C = ctx();
    TIME_BLOCK("Graph read");

    logger::info("Starting to read graph");

    if (C.inputFormat == Context::InputFormat::SpqrIndex)
    {
        readSpqrIndexGraph();

        if (C.bubbleType == Context::BubbleType::ULTRABUBBLE && !C.doubledUltrabubbles) return;

        C.isEntry = NodeArray<bool>(C.G, false);
        C.isExit= NodeArray<bool>(C.G, false);
        C.inDeg = NodeArray<int>(C.G, 0);
        C.outDeg= NodeArray<int>(C.G, 0);

        bool loaded_spqr_degrees = false;
        if (C.spqrIndex) {
            loaded_spqr_degrees = tryLoadGraphDegreesFromSpqrIndex(C, *C.spqrIndex);
            if (loaded_spqr_degrees) {
                auto &index = *C.spqrIndex;
                std::vector<std::uint32_t>().swap(index.graph_node_in_degrees);
                std::vector<std::uint32_t>().swap(index.graph_node_out_degrees);
                std::vector<std::uint8_t>().swap(index.graph_node_in_degrees8);
                std::vector<std::uint8_t>().swap(index.graph_node_out_degrees8);
                std::vector<std::uint32_t>().swap(index.graph_node_in_degree_overflow_nodes);
                std::vector<std::uint32_t>().swap(index.graph_node_in_degree_overflow_values);
                std::vector<std::uint32_t>().swap(index.graph_node_out_degree_overflow_nodes);
                std::vector<std::uint32_t>().swap(index.graph_node_out_degree_overflow_values);
            }
        }

        if (!C.directSpqrInputGraphEdgesMaterialized) {
            if (!C.spqrIndex) {
                throw std::runtime_error(
                    "SPQR-index node-only graph needs the loaded index to compute degrees");
            }
            const auto &index = *C.spqrIndex;
            const std::uint32_t graph_node_count = index.graph_node_count();
            if (C.G.numberOfNodes() != graph_node_count) {
                throw std::runtime_error(
                    "SPQR-index node-only graph node count does not match its index");
            }
            int *in_data = C.inDeg.size() == 0 ? nullptr : &*C.inDeg.begin();
            int *out_data = C.outDeg.size() == 0 ? nullptr : &*C.outDeg.begin();
            if (!loaded_spqr_degrees) {
                if (!index.graph_edges.empty()) {
                    const std::size_t edge_count = index.graph_edges.size();
                    if (C.threads > 1 && edge_count > 1000000) {
                        std::atomic<bool> invalid_edge{false};
                        #pragma omp parallel for schedule(static)
                        for (int64_t i_i = 0; i_i < static_cast<int64_t>(edge_count); ++i_i) {
                            const auto &ge = index.graph_edges[static_cast<std::size_t>(i_i)];
                            if (ge.src >= graph_node_count || ge.dst >= graph_node_count) {
                                invalid_edge.store(true, std::memory_order_relaxed);
                                continue;
                            }
                            #pragma omp atomic update
                            out_data[ge.src] += 1;
                            #pragma omp atomic update
                            in_data[ge.dst] += 1;
                        }
                        if (invalid_edge.load(std::memory_order_relaxed)) {
                            throw std::runtime_error(
                                "SPQR index graph edge references a missing graph node");
                        }
                    } else {
                        for (const auto &ge : index.graph_edges) {
                            if (ge.src >= graph_node_count || ge.dst >= graph_node_count) {
                                throw std::runtime_error(
                                    "SPQR index graph edge references a missing graph node");
                            }
                            ++out_data[ge.src];
                            ++in_data[ge.dst];
                        }
                    }
                } else if (spqr_index::graph_profile_is_oriented_double(
                               C.spqrIndexInputGraphView) &&
                           index.real_edge_count() == index.graph_edge_count()) {
                    const std::uint32_t real_edge_count =
                        graph_index::require_spqr_count(
                            index.real_edge_count(),
                            "SPQR-index direct graph real-edge count");
                    const spqr_index::GraphEdgeRecord *real_edge_records =
                        index.has_compact_real_edge_endpoints()
                            ? index.real_edge_endpoints.data()
                            : nullptr;
                    if (C.threads > 1 && real_edge_count > 1000000u) {
                        std::atomic<bool> invalid_edge{false};
                        #pragma omp parallel for schedule(static)
                        for (int64_t i_i = 0; i_i < static_cast<int64_t>(real_edge_count); ++i_i) {
                            const auto re = real_edge_records != nullptr
                                ? real_edge_records[static_cast<std::size_t>(i_i)]
                                : index.real_edge_endpoint_at(static_cast<std::uint32_t>(i_i));
                            if (re.src >= graph_node_count || re.dst >= graph_node_count) {
                                invalid_edge.store(true, std::memory_order_relaxed);
                                continue;
                            }
                            #pragma omp atomic update
                            out_data[re.src] += 1;
                            #pragma omp atomic update
                            in_data[re.dst] += 1;
                        }
                        if (invalid_edge.load(std::memory_order_relaxed)) {
                            throw std::runtime_error(
                                "SPQR index real edge references a missing graph node");
                        }
                    } else {
                        for (std::uint32_t i = 0; i < real_edge_count; ++i) {
                            const auto re = real_edge_records != nullptr
                                ? real_edge_records[static_cast<std::size_t>(i)]
                                : index.real_edge_endpoint_at(i);
                            if (re.src >= graph_node_count || re.dst >= graph_node_count) {
                                throw std::runtime_error(
                                    "SPQR index real edge references a missing graph node");
                            }
                            ++out_data[re.src];
                            ++in_data[re.dst];
                        }
                    }
                } else {
                    throw std::runtime_error(
                        "SPQR-index node-only graph needs graph-edge tables or matching real-edge endpoints to compute degrees");
                }
            }
        } else {
            if (!loaded_spqr_degrees) {
                for (edge e : C.G.edges) {
                    C.outDeg[C.G.source(e)]++;
                    C.inDeg [C.G.target(e)]++;
                }
            }
        }

        if (C.spqrIndex &&
            C.spqrIndexPath.empty() &&
            C.bubbleType == Context::BubbleType::SNARL &&
            !C.directSpqrInputGraphUsesIndexNames) {
            const bool keep_for_stored_spqr =
                C.spCompressMode == Context::SpCompressMode::Off &&
                spqr_index::graph_profile_is_raw_or_parallel_subdivided(
                    C.spqrIndexInputGraphView);
            if (!keep_for_stored_spqr) {
                C.spqrIndex.reset();
            }
        }

        return;
    }

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

constexpr size_t kIoChunkHighWater = 64ull * 1024ull * 1024ull;

inline void flushStringBuf(std::ostream &out, std::string &buf) {
    if (!buf.empty()) {
        out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        buf.clear();
    }
}

template <typename EndpointT>
struct ChainEdge {
    EndpointT endpoint[2];
    size_t next[2];
};

template <typename EndpointT>
struct ChainOcc {
    EndpointT key;
    size_t edge;
    uint8_t side;
};

template <typename EndpointT>
void compactEndpointPairChains(std::vector<std::pair<EndpointT, EndpointT>> &pairs)
{
    static_assert(std::is_unsigned<EndpointT>::value,
                  "packed endpoint keys must be unsigned");
    if (pairs.size() < 2) return;

    const size_t none = std::numeric_limits<size_t>::max();
    constexpr EndpointT strand_mask = static_cast<EndpointT>(1);
    std::vector<ChainEdge<EndpointT>> edges;
    edges.reserve(pairs.size());
    std::vector<ChainOcc<EndpointT>> occ;
    occ.reserve(pairs.size() * 2);

    for (const auto &p : pairs) {
        size_t idx = edges.size();
        edges.push_back({{p.first, p.second}, {none, none}});
        occ.push_back({p.first, idx, 0});
        occ.push_back({p.second, idx, 1});
    }

    std::sort(occ.begin(), occ.end(), [](const ChainOcc<EndpointT> &a,
                                          const ChainOcc<EndpointT> &b) {
        if (a.key != b.key) return a.key < b.key;
        if (a.edge != b.edge) return a.edge < b.edge;
        return a.side < b.side;
    });

    auto uniqueOcc = [&](EndpointT key) -> std::pair<size_t, uint8_t> {
        auto it = std::lower_bound(occ.begin(), occ.end(), key,
                                   [](const ChainOcc<EndpointT> &a, EndpointT b) {
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
            const EndpointT mate =
                static_cast<EndpointT>(edges[i].endpoint[side] ^ strand_mask);
            auto [other, other_side] = uniqueOcc(mate);
            if (other != none && other != i && edges[other].endpoint[other_side] == mate) {
                edges[i].next[side] = other;
            }
        }
    }

    auto degree = [&](size_t i) {
        return (edges[i].next[0] != none ? 1 : 0) +
               (edges[i].next[1] != none ? 1 : 0);
    };

    std::vector<uint8_t> seen(edges.size(), 0);
    std::vector<std::pair<EndpointT, EndpointT>> out;
    out.reserve(pairs.size());

    for (size_t i = 0; i < edges.size(); ++i) {
        if (seen[i] || degree(i) != 1) continue;

        size_t cur = i;
        uint8_t entry = edges[cur].next[0] == none ? 0 : 1;
        EndpointT first = edges[cur].endpoint[entry];
        EndpointT last = edges[cur].endpoint[entry ^ 1u];

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

using FastEndpointKey = std::uint64_t;
using FastPairKey = std::uint64_t;

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

inline FastEndpointKey packFastEndpointKey(uint32_t idx, uint8_t sign)
{
    return static_cast<FastEndpointKey>((static_cast<uint64_t>(idx) << 1) |
                                        static_cast<uint64_t>(sign));
}

inline FastPairKey packFastPairKey(uint32_t a_idx, uint8_t a_sign,
                                   uint32_t b_idx, uint8_t b_sign)
{
    FastEndpointKey a = packFastEndpointKey(a_idx, a_sign);
    FastEndpointKey b = packFastEndpointKey(b_idx, b_sign);
    if (a > b) std::swap(a, b);
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

inline uint32_t fastEndpointIdx(FastEndpointKey key)
{
    return static_cast<uint32_t>(key >> 1);
}

inline uint8_t fastEndpointSign(FastEndpointKey key)
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
    if (!graph_index::fits_packed_endpoint_id(static_cast<uint64_t>(it->second.idx)))
        return false;
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
            const auto types = edgePartTypes(C, e);
            EdgePartType typeAtU = (C.G.source(e) == u)
                ? types.first
                : types.second;
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
                                FastEndpointKey a,
                                FastEndpointKey b)
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
                                  FastEndpointKey key,
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
            std::vector<FastEndpointKey> clique;
            clique.reserve(endpoints.size());
            for (const auto &endpoint : endpoints) {
                clique.push_back(packFastEndpointKey(endpoint.first,
                                                     endpoint.second));
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

    auto endpointLess = [&](FastEndpointKey a, FastEndpointKey b) {
        return fastEndpointKeyLess(tables, a, b);
    };
    auto cliqueLess = [&](const std::vector<FastEndpointKey> &a,
                          const std::vector<FastEndpointKey> &b) {
        return std::lexicographical_compare(a.begin(), a.end(),
                                            b.begin(), b.end(),
                                            endpointLess);
    };

    for (auto &clique : cliques) {
        std::sort(clique.begin(), clique.end(), endpointLess);
        clique.erase(std::unique(clique.begin(), clique.end()), clique.end());
    }
    cliques.erase(std::remove_if(cliques.begin(), cliques.end(),
                                 [](const std::vector<FastEndpointKey> &clique) {
                                     return clique.size() < 2;
                                 }),
                  cliques.end());
    std::sort(cliques.begin(), cliques.end(), cliqueLess);
    cliques.erase(std::unique(cliques.begin(), cliques.end()), cliques.end());

    std::vector<FastPairKey> coveredPairs;
    std::vector<std::vector<FastEndpointKey>> outputCliques;
    std::vector<FastPairKey> cliquePairs;

    for (auto &clique : cliques) {
        cliquePairs.clear();
        cliquePairs.reserve(clique.size() * (clique.size() - 1) / 2);
        bool canCompact = true;

        for (size_t i = 0; i < clique.size(); ++i) {
            for (size_t j = i + 1; j < clique.size(); ++j) {
                FastEndpointKey a = clique[i];
                FastEndpointKey b = clique[j];
                if (a > b) std::swap(a, b);
                FastPairKey key =
                    (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
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
            for (FastPairKey key : cliquePairs) {
                if (std::binary_search(coveredPairs.begin(), coveredPairs.end(), key)) {
                    overlaps = true;
                    break;
                }
            }

            if (!overlaps) {
                outputCliques.push_back(std::move(clique));

                std::vector<FastPairKey> merged;
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
    std::vector<std::pair<FastEndpointKey, FastEndpointKey>> outputPairs;
    const bool write_all_pairs =
        !filter_trivial && !C.compactOutputChains && coveredPairs.empty();
    if (C.compactOutputChains) {
        outputPairs.reserve(pairs.size());
        for (FastPairKey key : pairs) {
            if ((!filter_trivial || !fastPairIsTrivial(tables, key)) &&
                !std::binary_search(coveredPairs.begin(), coveredPairs.end(), key)) {
                outputPairs.emplace_back(static_cast<FastEndpointKey>(key >> 32),
                                         static_cast<FastEndpointKey>(key & 0xffffffffu));
            }
        }
        compactEndpointPairChains(outputPairs);
        pair_count = outputPairs.size();
    } else if (write_all_pairs) {
        pair_count = pairs.size();
    } else {
        size_t covered_i = 0;
        for (FastPairKey key : pairs) {
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

    auto appendPairLine = [&](FastEndpointKey a, FastEndpointKey b, std::string &dst) {
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

    auto writePair = [&](FastEndpointKey a, FastEndpointKey b) {
        appendPairLine(a, b, buf);

        if (buf.size() >= kIoChunkHighWater) flushStringBuf(out, buf);
    };

    auto appendPairKeyLine = [&](FastPairKey key, std::string &dst) {
        appendPairLine(static_cast<FastEndpointKey>(key >> 32),
                       static_cast<FastEndpointKey>(key & 0xffffffffu),
                       dst);
    };

    if (C.compactOutputChains) {
        for (auto [a, b] : outputPairs) writePair(a, b);
    } else if (write_all_pairs) {
#if defined(_OPENMP)
        const bool use_parallel_format =
            C.threads > 1 && pairs.size() > 100000;
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
            for (FastPairKey key : pairs) {
                appendPairKeyLine(key, buf);
                if (buf.size() >= kIoChunkHighWater) flushStringBuf(out, buf);
            }
        }
    } else {
#if defined(_OPENMP)
        const bool use_parallel_format =
            C.threads > 1 && pairs.size() > 100000 && !C.compactOutputChains;
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
                        const FastPairKey key = pairs[i];
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
            for (FastPairKey key : pairs) {
                if (filter_trivial && fastPairIsTrivial(tables, key)) continue;
                while (covered_i < coveredPairs.size() && coveredPairs[covered_i] < key) {
                    ++covered_i;
                }
                if (covered_i < coveredPairs.size() && coveredPairs[covered_i] == key) continue;
                writePair(static_cast<FastEndpointKey>(key >> 32),
                          static_cast<FastEndpointKey>(key & 0xffffffffu));
            }
        }
    }
    flushStringBuf(out, buf);
}

}

std::string contextGraphNodeName(const Context &C, spqr_compat::node v)
{
    auto direct = C.node2name.find(v);
    if (direct != C.node2name.end()) {
        return direct->second;
    }

    const std::uint32_t idx = static_cast<std::uint32_t>(v.idx);
    if (C.directSpqrInputGraphUsesIndexNames &&
        C.spqrIndex &&
        idx < C.spqrIndex->graph_node_count()) {
        return spqrIndexGraphNodeName(*C.spqrIndex, idx);
    }

    if (idx < C.nodeNamesByIndex.size() && !C.nodeNamesByIndex[idx].empty()) {
        return C.nodeNamesByIndex[idx];
    }

    auto sparse = C.sparseNodeNamesByIndex.find(idx);
    const bool hasSparse = sparse != C.sparseNodeNamesByIndex.end();
    const bool numericAvailable =
        idx < C.nodeNumericNamesByIndex.size() &&
        (C.nodeNumericNameValidByIndex.empty() ||
         (idx < C.nodeNumericNameValidByIndex.size() &&
          C.nodeNumericNameValidByIndex[idx] != 0));
    if (numericAvailable && !hasSparse) {
        const std::uint64_t numeric = C.nodeNumericNamesByIndex[idx];
        if (spqr_index::graph_profile_is_oriented_double(C.spqrIndexInputGraphView)) {
            return spqr_index::IndexedSpqrTree::oriented_numeric_graph_node_name(numeric);
        }
        return std::to_string(numeric);
    }

    if (hasSparse) {
        return sparse->second;
    }

    return std::to_string(idx);
}

void appendUInt64Decimal(std::string &out, std::uint64_t value)
{
    char buf[32];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), value);
    if (ec != std::errc()) {
        throw std::runtime_error("failed to format integer output");
    }
    out.append(buf, ptr);
}

bool decimalLexLess(std::uint64_t a, std::uint64_t b)
{
    char abuf[32];
    char bbuf[32];
    auto [aptr, aec] = std::to_chars(abuf, abuf + sizeof(abuf), a);
    auto [bptr, bec] = std::to_chars(bbuf, bbuf + sizeof(bbuf), b);
    if (aec != std::errc() || bec != std::errc()) {
        throw std::runtime_error("failed to compare integer output names");
    }
    const std::size_t asz = static_cast<std::size_t>(aptr - abuf);
    const std::size_t bsz = static_cast<std::size_t>(bptr - bbuf);
    const int cmp = std::memcmp(abuf, bbuf, std::min(asz, bsz));
    return cmp < 0 || (cmp == 0 && asz < bsz);
}

bool tryWriteDirectIndexedNumericSuperbubbles(std::ostream &out, const Context &C)
{
    if (C.compactOutputChains ||
        C.bubbleType != Context::BubbleType::SUPERBUBBLE ||
        C.directedSuperbubbles ||
        C.inputFormat != Context::InputFormat::SpqrIndex ||
        !spqr_index::graph_profile_is_oriented_bidirected(C.spqrIndexInputGraphView) ||
        !C.directSpqrInputGraphUsesIndexNames ||
        !C.spqrIndex ||
        !C.spqrIndex->has_compact_numeric_graph_node_names() ||
        !C.spqrIndex->graph_node_string_names.empty()) {
        return false;
    }

    const auto &index = *C.spqrIndex;
    const std::uint32_t graphNodeCount = index.graph_node_count();
    if (!index.uses_oriented_numeric_graph_node_names()) {
        return false;
    }

    auto encodedAt = [&](std::uint32_t idx) -> std::uint64_t {
        if (idx >= graphNodeCount) {
            throw std::runtime_error("superbubble endpoint is outside SPQR index graph node table");
        }
        return index.graph_node_numeric_name_at(idx);
    };

    if (index.graph_node_numeric_names32.size() == graphNodeCount) {
        std::vector<std::uint64_t> keys;
        keys.reserve(C.superbubbles.size());
        for (const auto &sb : C.superbubbles) {
            std::uint64_t a = encodedAt(static_cast<std::uint32_t>(sb.first.idx)) >> 1u;
            std::uint64_t b = encodedAt(static_cast<std::uint32_t>(sb.second.idx)) >> 1u;
            if (a == b) continue;
            if (decimalLexLess(b, a)) {
                std::swap(a, b);
            }
            keys.push_back((a << 32u) | b);
        }

#if defined(_OPENMP) && defined(__GLIBCXX__)
        __gnu_parallel::sort(keys.begin(), keys.end());
#else
        std::sort(keys.begin(), keys.end());
#endif
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

        std::string header;
        header.reserve(32);
        appendUInt64Decimal(header, static_cast<std::uint64_t>(keys.size()));
        header.push_back('\n');
        flushStringBuf(out, header);

#if defined(_OPENMP)
        const bool use_parallel_format = C.threads > 1 && keys.size() > 100000;
        if (use_parallel_format) {
            const int workers = std::max<int>(1, static_cast<int>(C.threads));
            const std::size_t blockPairs = 1ull << 19;
            const std::size_t blockCount = (keys.size() + blockPairs - 1u) / blockPairs;
            std::vector<std::string> buffers;
            for (std::size_t groupBegin = 0; groupBegin < blockCount;
                 groupBegin += static_cast<std::size_t>(workers)) {
                const std::size_t groupEnd = std::min(
                    blockCount, groupBegin + static_cast<std::size_t>(workers));
                const std::size_t groupSize = groupEnd - groupBegin;
                buffers.clear();
                buffers.resize(groupSize);

                #pragma omp parallel for schedule(static) num_threads(workers)
                for (int64_t bi_i = 0; bi_i < static_cast<int64_t>(groupSize); ++bi_i) {
                    const std::size_t bi = static_cast<std::size_t>(bi_i);
                    const std::size_t block = groupBegin + bi;
                    const std::size_t begin = block * blockPairs;
                    const std::size_t end = std::min(keys.size(), begin + blockPairs);
                    std::string local;
                    local.reserve((end - begin) * 24u);
                    for (std::size_t i = begin; i < end; ++i) {
                        appendUInt64Decimal(local, keys[i] >> 32u);
                        local.push_back(' ');
                        appendUInt64Decimal(local, keys[i] & 0xffffffffull);
                        local.push_back('\n');
                    }
                    buffers[bi].swap(local);
                }

                for (std::string &local : buffers) {
                    flushStringBuf(out, local);
                }
            }
            return true;
        }
#endif
        std::string buf;
        buf.reserve(kIoChunkHighWater + 4096);
        for (std::uint64_t key : keys) {
            appendUInt64Decimal(buf, key >> 32u);
            buf.push_back(' ');
            appendUInt64Decimal(buf, key & 0xffffffffull);
            buf.push_back('\n');
            if (buf.size() >= kIoChunkHighWater) flushStringBuf(out, buf);
        }
        flushStringBuf(out, buf);
        return true;
    }

    using NumPair = std::pair<std::uint64_t, std::uint64_t>;
    std::vector<NumPair> pairs;
    pairs.reserve(C.superbubbles.size());
    for (const auto &sb : C.superbubbles) {
        const std::uint64_t a = encodedAt(static_cast<std::uint32_t>(sb.first.idx)) >> 1u;
        const std::uint64_t b = encodedAt(static_cast<std::uint32_t>(sb.second.idx)) >> 1u;
        if (a == b) continue;
        if (decimalLexLess(b, a)) {
            pairs.emplace_back(b, a);
        } else {
            pairs.emplace_back(a, b);
        }
    }

    auto pairLess = [](const NumPair &x, const NumPair &y) {
        if (x.first != y.first) return x.first < y.first;
        return x.second < y.second;
    };
#if defined(_OPENMP) && defined(__GLIBCXX__)
    __gnu_parallel::sort(pairs.begin(), pairs.end(), pairLess);
#else
    std::sort(pairs.begin(), pairs.end(), pairLess);
#endif
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    std::string header;
    header.reserve(32);
    appendUInt64Decimal(header, static_cast<std::uint64_t>(pairs.size()));
    header.push_back('\n');
    flushStringBuf(out, header);

#if defined(_OPENMP)
    const bool use_parallel_format = C.threads > 1 && pairs.size() > 100000;
    if (use_parallel_format) {
        const int workers = std::max<int>(1, static_cast<int>(C.threads));
        const std::size_t blockPairs = 1ull << 19;
        const std::size_t blockCount = (pairs.size() + blockPairs - 1u) / blockPairs;
        std::vector<std::string> buffers;
        for (std::size_t groupBegin = 0; groupBegin < blockCount;
             groupBegin += static_cast<std::size_t>(workers)) {
            const std::size_t groupEnd = std::min(
                blockCount, groupBegin + static_cast<std::size_t>(workers));
            const std::size_t groupSize = groupEnd - groupBegin;
            buffers.clear();
            buffers.resize(groupSize);

            #pragma omp parallel for schedule(static) num_threads(workers)
            for (int64_t bi_i = 0; bi_i < static_cast<int64_t>(groupSize); ++bi_i) {
                const std::size_t bi = static_cast<std::size_t>(bi_i);
                const std::size_t block = groupBegin + bi;
                const std::size_t begin = block * blockPairs;
                const std::size_t end = std::min(pairs.size(), begin + blockPairs);
                std::string local;
                local.reserve((end - begin) * 24u);
                for (std::size_t i = begin; i < end; ++i) {
                    appendUInt64Decimal(local, pairs[i].first);
                    local.push_back(' ');
                    appendUInt64Decimal(local, pairs[i].second);
                    local.push_back('\n');
                }
                buffers[bi].swap(local);
            }

            for (std::string &local : buffers) {
                flushStringBuf(out, local);
            }
        }
        return true;
    }
#endif

    std::string buf;
    buf.reserve(kIoChunkHighWater + 4096);
    for (const auto &p : pairs) {
        appendUInt64Decimal(buf, p.first);
        buf.push_back(' ');
        appendUInt64Decimal(buf, p.second);
        buf.push_back('\n');
        if (buf.size() >= kIoChunkHighWater) flushStringBuf(out, buf);
    }
    flushStringBuf(out, buf);

    return true;
}

void appendUltrabubbleEndpoint(std::string &out,
                               const Context &C,
                               std::uint32_t packed)
{
    const std::uint32_t gid = packed >> 1u;
    if (gid >= C.ubNodeNames.size()) {
        throw std::runtime_error("ultrabubble output endpoint is out of range");
    }
    out.append(C.ubNodeNames[gid]);
    out.push_back((packed & 1u) ? '+' : '-');
}

void writeUltrabubblesBuffered(
    std::ostream &out,
    const Context &C,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> &pairs)
{
    std::string header;
    header.reserve(32);
    appendUInt64Decimal(header, static_cast<std::uint64_t>(pairs.size()));
    header.push_back('\n');
    flushStringBuf(out, header);

#if defined(_OPENMP)
    const bool use_parallel_format = C.threads > 1 && pairs.size() > 100000;
    if (use_parallel_format) {
        const int workers = std::max<int>(1, static_cast<int>(C.threads));
        const std::size_t blockPairs = 1ull << 19;
        const std::size_t blockCount = (pairs.size() + blockPairs - 1u) / blockPairs;
        std::vector<std::string> buffers;
        for (std::size_t groupBegin = 0; groupBegin < blockCount;
             groupBegin += static_cast<std::size_t>(workers)) {
            const std::size_t groupEnd = std::min(
                blockCount, groupBegin + static_cast<std::size_t>(workers));
            const std::size_t groupSize = groupEnd - groupBegin;
            buffers.clear();
            buffers.resize(groupSize);

            #pragma omp parallel for schedule(static) num_threads(workers)
            for (int64_t bi_i = 0; bi_i < static_cast<int64_t>(groupSize); ++bi_i) {
                const std::size_t bi = static_cast<std::size_t>(bi_i);
                const std::size_t block = groupBegin + bi;
                const std::size_t begin = block * blockPairs;
                const std::size_t end = std::min(pairs.size(), begin + blockPairs);
                std::string local;
                local.reserve((end - begin) * 24u);
                for (std::size_t i = begin; i < end; ++i) {
                    appendUltrabubbleEndpoint(local, C, pairs[i].first);
                    local.push_back(' ');
                    appendUltrabubbleEndpoint(local, C, pairs[i].second);
                    local.push_back('\n');
                }
                buffers[bi].swap(local);
            }

            for (std::string &local : buffers) {
                flushStringBuf(out, local);
            }
        }
        return;
    }
#endif

    std::string buf;
    buf.reserve(kIoChunkHighWater + 4096);
    for (const auto &p : pairs) {
        appendUltrabubbleEndpoint(buf, C, p.first);
        buf.push_back(' ');
        appendUltrabubbleEndpoint(buf, C, p.second);
        buf.push_back('\n');
        if (buf.size() >= kIoChunkHighWater) flushStringBuf(out, buf);
    }
    flushStringBuf(out, buf);
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
                    const auto types = edgePartTypes(C, e);
                    EdgePartType typeAtU = (C.G.source(e) == u)
                        ? types.first
                        : types.second;
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
        std::vector<std::pair<std::uint32_t, std::uint32_t>> compact_pairs;
        if (C.compactOutputChains) {
            compact_pairs = C.ultrabubbleIncPacked;
            compactEndpointPairChains(compact_pairs);
        }
        const auto &output_pairs =
            C.compactOutputChains ? compact_pairs : C.ultrabubbleIncPacked;

        if (C.outputPath.empty())
        {
            writeUltrabubblesBuffered(std::cout, C, output_pairs);
            if (!std::cout)
            {
                throw std::runtime_error("Error while writing ultrabubbles to standard output");
            }
        }
        else
        {
            std::ofstream out(C.outputPath, std::ios::out | std::ios::binary);
            if (!out)
            {
                throw std::runtime_error("Failed to open output file '" +
                                         C.outputPath + "' for writing");
            }
            writeUltrabubblesBuffered(out, C, output_pairs);
            if (!out)
            {
                throw std::runtime_error("Error while writing ultrabubbles to output file '" +
                                         C.outputPath + "'");
            }
        }
        return;
    }

    if (C.outputPath.empty())
    {
        if (tryWriteDirectIndexedNumericSuperbubbles(std::cout, C))
        {
            if (!std::cout)
            {
                throw std::runtime_error("Error while writing superbubbles to standard output");
            }
            return;
        }
    }
    else
    {
        std::ofstream out(C.outputPath, std::ios::out | std::ios::binary);
        if (!out)
        {
            throw std::runtime_error("Failed to open output file '" +
                                     C.outputPath + "' for writing");
        }
        if (tryWriteDirectIndexedNumericSuperbubbles(out, C))
        {
            if (!out)
            {
                throw std::runtime_error("Error while writing superbubbles to output file '" +
                                         C.outputPath + "'");
            }
            return;
        }
    }

    std::vector<std::pair<std::string, std::string>> res;

    const bool rawSpqrInput =
        C.inputFormat == Context::InputFormat::SpqrIndex &&
        C.spqrIndexInputLoaded &&
        spqr_index::graph_profile_is_raw(C.spqrIndexInputGraphView);
    const bool orientedSuperbubbleSpqrInput =
        C.inputFormat == Context::InputFormat::SpqrIndex &&
        C.spqrIndexInputLoaded &&
        spqr_index::graph_profile_is_oriented_bidirected(C.spqrIndexInputGraphView);
    if ((C.inputFormat == Context::InputFormat::Gfa || rawSpqrInput ||
         orientedSuperbubbleSpqrInput) &&
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
            const std::string s = contextGraphNodeName(C, w.first);
            const std::string t = contextGraphNodeName(C, w.second);

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
            res.push_back({contextGraphNodeName(C, w.first),
                           contextGraphNodeName(C, w.second)});
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
