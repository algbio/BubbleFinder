#pragma once
// spqr-rust C++ interface
//
// Two namespaces:
//   spqr             -> with G.source(e) syntax
//   spqr::ogdf_compat -> with e->source() syntax, like OGDF
// Pick spqr unless you're porting OGDF code and don't want to change every call site

#include "spqr_rust_wrapper.hpp"
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>

namespace spqr {

struct node {
    uint32_t idx;
    
    constexpr node() : idx(UINT32_MAX) {}
    constexpr node(uint32_t i) : idx(i) {}
    constexpr node(std::nullptr_t) : idx(UINT32_MAX) {}  
    
    constexpr operator uint32_t() const { return idx; }
    constexpr uint32_t index() const { return idx; }
    
    constexpr bool operator!() const { return idx == UINT32_MAX; }
    constexpr explicit operator bool() const { return idx != UINT32_MAX; }
    constexpr bool operator==(std::nullptr_t) const { return idx == UINT32_MAX; }
    constexpr bool operator!=(std::nullptr_t) const { return idx != UINT32_MAX; }
    constexpr bool operator==(node o) const { return idx == o.idx; }
    constexpr bool operator!=(node o) const { return idx != o.idx; }
    constexpr bool operator<(node o) const { return idx < o.idx; }
};

struct edge {
    uint32_t idx;
    
    constexpr edge() : idx(UINT32_MAX) {}
    constexpr edge(uint32_t i) : idx(i) {}
    constexpr edge(std::nullptr_t) : idx(UINT32_MAX) {}
    
    constexpr operator uint32_t() const { return idx; }
    constexpr uint32_t index() const { return idx; }
    
    constexpr bool operator!() const { return idx == UINT32_MAX; }
    constexpr explicit operator bool() const { return idx != UINT32_MAX; }
    constexpr bool operator==(std::nullptr_t) const { return idx == UINT32_MAX; }
    constexpr bool operator!=(std::nullptr_t) const { return idx != UINT32_MAX; }
    constexpr bool operator==(edge o) const { return idx == o.idx; }
    constexpr bool operator!=(edge o) const { return idx != o.idx; }
    constexpr bool operator<(edge o) const { return idx < o.idx; }
};

constexpr node INVALID_NODE{UINT32_MAX};
constexpr edge INVALID_EDGE{UINT32_MAX};

// adjEntry for OGDF compatibility
struct adjEntry {
    node neighbor;
    edge e;
    node twinNode() const { return neighbor; }
    edge theEdge() const { return e; }
};


template<typename T>
class NodeArray {
    std::vector<T> data_;
    T default_{};
public:
    NodeArray() = default;
    template<typename G> NodeArray(const G& g, const T& def = T())
        : data_(g.numberOfNodes(), def), default_(def) {}
    NodeArray(size_t n, const T& def = T()) : data_(n, def), default_(def) {}

    template<typename G> void init(const G& g, const T& def = T()) {
        data_.assign(g.numberOfNodes(), def);
        default_ = def;
    }

    T& operator[](size_t idx) {
        if (idx >= data_.size()) data_.resize(idx + 1, default_);
        return data_[idx];
    }
    const T& operator[](size_t idx) const { return data_[idx]; }

    T& operator()(node v) { return (*this)[v.idx]; }
    const T& operator()(node v) const { return data_[v.idx]; }

    size_t size() const { return data_.size(); }
    void resize(size_t n, const T& val = T()) { data_.resize(n, val); }
    void clear() { data_.clear(); }
    template<typename Iter> void assign(Iter first, Iter last) { data_.assign(first, last); }
    void assign(size_t n, const T& val) { data_.assign(n, val); }
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
};

// Specialization for bool to avoid std::vector<bool> proxy issues
template<>
class NodeArray<bool> {
    std::vector<char> data_;
    char default_ = 0;
public:
    NodeArray() = default;
    template<typename G> NodeArray(const G& g, bool def = false)
        : data_(g.numberOfNodes(), def ? 1 : 0), default_(def ? 1 : 0) {}
    NodeArray(size_t n, bool def = false) : data_(n, def ? 1 : 0), default_(def ? 1 : 0) {}

    template<typename G> void init(const G& g, bool def = false) {
        default_ = def ? 1 : 0;
        data_.assign(g.numberOfNodes(), default_);
    }

    // Proxy class for bool access
    class Ref {
        char& c_;
    public:
        Ref(char& c) : c_(c) {}
        operator bool() const { return c_ != 0; }
        Ref& operator=(bool b) { c_ = b ? 1 : 0; return *this; }
    };

    Ref operator[](size_t idx) {
        if (idx >= data_.size()) data_.resize(idx + 1, default_);
        return Ref(data_[idx]);
    }
    bool operator[](size_t idx) const { return data_[idx] != 0; }

    Ref operator()(node v) { return (*this)[v.idx]; }
    bool operator()(node v) const { return data_[v.idx] != 0; }

    size_t size() const { return data_.size(); }
    void resize(size_t n, bool val = false) { data_.resize(n, val ? 1 : 0); }
    void clear() { data_.clear(); }
};

template<typename T>
class EdgeArray {
    std::vector<T> data_;
    T default_{};
public:
    EdgeArray() = default;
    template<typename G> EdgeArray(const G& g, const T& def = T())
        : data_(g.numberOfEdges(), def), default_(def) {}
    EdgeArray(size_t n, const T& def = T()) : data_(n, def), default_(def) {}

    template<typename G> void init(const G& g, const T& def = T()) {
        data_.assign(g.numberOfEdges(), def);
        default_ = def;
    }

    T& operator[](size_t idx) {
        if (idx >= data_.size()) data_.resize(idx + 1, default_);
        return data_[idx];
    }
    const T& operator[](size_t idx) const { return data_[idx]; }

    T& operator()(edge e) { return (*this)[e.idx]; }
    const T& operator()(edge e) const { return data_[e.idx]; }

    size_t size() const { return data_.size(); }
    void resize(size_t n, const T& val = T()) { data_.resize(n, val); }
    void clear() { data_.clear(); }
    template<typename Iter> void assign(Iter first, Iter last) { data_.assign(first, last); }
    void assign(size_t n, const T& val) { data_.assign(n, val); }
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
};

// Specialization for bool to avoid std::vector<bool> proxy issues
template<>
class EdgeArray<bool> {
    std::vector<char> data_;
    char default_ = 0;
public:
    EdgeArray() = default;
    template<typename G> EdgeArray(const G& g, bool def = false)
        : data_(g.numberOfEdges(), def ? 1 : 0), default_(def ? 1 : 0) {}
    EdgeArray(size_t n, bool def = false) : data_(n, def ? 1 : 0), default_(def ? 1 : 0) {}

    template<typename G> void init(const G& g, bool def = false) {
        default_ = def ? 1 : 0;
        data_.assign(g.numberOfEdges(), default_);
    }

    class Ref {
        char& c_;
    public:
        Ref(char& c) : c_(c) {}
        operator bool() const { return c_ != 0; }
        Ref& operator=(bool b) { c_ = b ? 1 : 0; return *this; }
    };

    Ref operator[](size_t idx) {
        if (idx >= data_.size()) data_.resize(idx + 1, default_);
        return Ref(data_[idx]);
    }
    bool operator[](size_t idx) const { return data_[idx] != 0; }

    Ref operator()(edge e) { return (*this)[e.idx]; }
    bool operator()(edge e) const { return data_[e.idx] != 0; }

    size_t size() const { return data_.size(); }
    void resize(size_t n, bool val = false) { data_.resize(n, val ? 1 : 0); }
    void clear() { data_.clear(); }
};

// Graph

class Graph {
    std::unique_ptr<spqr_rust::RustGraph> g_;
    std::vector<char> deletedEdge_;
    uint32_t deletedCount_ = 0;

    bool isDeleted_(uint32_t eidx) const {
        return !deletedEdge_.empty() && eidx < deletedEdge_.size() && deletedEdge_[eidx];
    }

public:
    struct NodesRange {
        const Graph* g;
        struct It { 
            uint32_t i; 
            constexpr node operator*() const { return node{i}; } 
            constexpr It& operator++() { ++i; return *this; } 
            constexpr It operator++(int) { It tmp = *this; ++i; return tmp; }
            constexpr bool operator!=(It o) const { return i != o.i; }
            constexpr bool operator==(It o) const { return i == o.i; }
        };
        It begin() const { return {0}; }
        It end() const { return {g->numberOfNodes()}; }
        uint32_t size() const { return g->numberOfNodes(); }
    };
    struct EdgesRange {
        const Graph* g;
        struct It {
            const Graph* g;
            uint32_t i;
            uint32_t endi;
            void advance_to_live() { while (i < endi && g->isDeleted_(i)) ++i; }
            edge operator*() const { return edge{i}; }
            It& operator++() { ++i; advance_to_live(); return *this; }
            It operator++(int) { It tmp = *this; ++(*this); return tmp; }
            bool operator!=(It o) const { return i != o.i; }
            bool operator==(It o) const { return i == o.i; }
        };
        It begin() const {
            It it{g, 0, g->g_->numEdges()};
            it.advance_to_live();
            return it;
        }
        It end() const {
            uint32_t raw = g->g_->numEdges();
            return {g, raw, raw};
        }
        uint32_t size() const { return g->numberOfEdges(); }
    };

    Graph() : g_(std::make_unique<spqr_rust::RustGraph>()), nodes{this}, edges{this} {}
    
    node newNode() { return node{g_->addNode()}; }
    node newNodes(uint32_t count) { return node{g_->addNodes(count)}; }
    edge newEdge(node u, node v) { return edge{g_->addEdge(u.idx, v.idx)}; }
    edge newEdgesBatchFlat(const uint32_t* endpoints, uint32_t count) {
        uint32_t first = g_->numEdges();
        g_->addEdgesBatchFlat(endpoints, count);
        return edge{first};
    }

    void delEdge(edge e) {
        const uint32_t rawN = g_->numEdges();
        if (e.idx >= rawN) return;
        if (deletedEdge_.size() < rawN) deletedEdge_.resize(rawN, 0);
        if (!deletedEdge_[e.idx]) {
            deletedEdge_[e.idx] = 1;
            ++deletedCount_;
        }
    }
    
    uint32_t numberOfNodes() const { return g_->numNodes(); }
    uint32_t numberOfEdges() const { return g_->numEdges() - deletedCount_; }
    
    node firstNode() const { return numberOfNodes() > 0 ? node{0u} : INVALID_NODE; }
    
    node source(edge e) const { return node{g_->edgeSrc(e.idx)}; }
    node target(edge e) const { return node{g_->edgeDst(e.idx)}; }
    
    NodesRange nodes;
    EdgesRange edges;
    
    template<typename F>
    void forEachAdj(node v, F&& f) const {
        if (deletedCount_ == 0) {
            g_->forEachNeighbor(v.idx, [&](uint32_t n, uint32_t e) { f(node{n}, edge{e}); });
        } else {
            g_->forEachNeighbor(v.idx, [&](uint32_t n, uint32_t e) {
                if (!isDeleted_(e)) f(node{n}, edge{e});
            });
        }
    }
    
    uint32_t degree(node v) const {
        if (deletedCount_ == 0) return g_->degree(v.idx);
        uint32_t c = 0;
        g_->forEachNeighbor(v.idx, [&](uint32_t, uint32_t e) { if (!isDeleted_(e)) ++c; });
        return c;
    }
    uint32_t outdeg(node v) const {
        if (deletedCount_ == 0) return g_->outdeg(v.idx);
        uint32_t c = 0;
        g_->forEachNeighbor(v.idx, [&](uint32_t, uint32_t e) {
            if (!isDeleted_(e) && g_->edgeSrc(e) == v.idx) ++c;
        });
        return c;
    }
    uint32_t indeg(node v) const {
        if (deletedCount_ == 0) return g_->indeg(v.idx);
        uint32_t c = 0;
        g_->forEachNeighbor(v.idx, [&](uint32_t, uint32_t e) {
            if (!isDeleted_(e) && g_->edgeDst(e) == v.idx) ++c;
        });
        return c;
    }
    
    spqr_rust::RustGraph& raw() { return *g_; }
    const spqr_rust::RustGraph& raw() const { return *g_; }
};

// BCTree

class BCTree {
    std::unique_ptr<spqr_rust::RustBCTree> bc_;
    std::vector<bool> isCut_;

public:
    enum class BNodeType { BComp, CComp };
    enum class GNodeType { Normal, CutVertex };
    
    explicit BCTree(const Graph& g) : bc_(std::make_unique<spqr_rust::RustBCTree>(g.raw())) {
        isCut_.assign(g.numberOfNodes(), false);
        for (uint32_t v : bc_->cutVertices()) isCut_[v] = true;
    }
    
    uint32_t numberOfBComps() const { return bc_->numBlocks(); }
    uint32_t numberOfCComps() const { return bc_->numCutVertices(); }
    
    GNodeType typeOfGNode(node v) const { return isCut_[v.idx] ? GNodeType::CutVertex : GNodeType::Normal; }
    BNodeType typeOfBNode(node v) const { return v.idx < bc_->numBlocks() ? BNodeType::BComp : BNodeType::CComp; }
    
    std::vector<edge> hEdges(node bNode) const {
        if (bNode.idx >= bc_->numBlocks()) return {};
        auto raw = bc_->blockEdges(bNode.idx);
        std::vector<edge> r; r.reserve(raw.size());
        for (auto e : raw) r.push_back(edge{e});
        return r;
    }
    
    edge original(edge e) const { return e; }
    node repVertex(node v, node) const { return v; }
    node bcproper(node v) const { return v; }
    
    struct BCTreeGraph {
        uint32_t n_;
        struct NodesRange { 
            uint32_t n; 
            struct It { 
                uint32_t i; 
                node operator*() const { return node{i}; } 
                It& operator++() { ++i; return *this; }
                It operator++(int) { It tmp = *this; ++i; return tmp; }
                bool operator!=(It o) const { return i != o.i; }
                bool operator==(It o) const { return i == o.i; }
            }; 
            It begin() const { return {0}; } 
            It end() const { return {n}; }
            uint32_t size() const { return n; }
        };
        NodesRange nodes{0};
        BCTreeGraph(uint32_t n) : n_(n), nodes{n} {}
        uint32_t numberOfNodes() const { return n_; }
    };
    BCTreeGraph bcTree() const { return BCTreeGraph{bc_->numBlocks() + bc_->numCutVertices()}; }
};

// StaticSPQRTree

using tree_node = node;

class TreeGraph {
    uint32_t n_ = 0;
    std::vector<uint32_t> parents_, src_, tgt_;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> adj_;  // adj_[v] = [(neighbor, edge_idx), ...]
public:
    void build(uint32_t n, const std::vector<uint32_t>& parents) {
        n_ = n; parents_ = parents;
        src_.clear(); tgt_.clear();
        adj_.assign(n, {});
        for (uint32_t i = 0; i < n; ++i) {
            if (parents[i] != UINT32_MAX && parents[i] != i) {
                uint32_t eIdx = src_.size();
                src_.push_back(parents[i]);
                tgt_.push_back(i);
                adj_[parents[i]].push_back({i, eIdx});
                adj_[i].push_back({parents[i], eIdx});
            }
        }
    }
    uint32_t numberOfNodes() const { return n_; }
    uint32_t numberOfEdges() const { return src_.size(); }
    node source(edge e) const { return node{src_[e.idx]}; }
    node target(edge e) const { return node{tgt_[e.idx]}; }
    
    template<typename F>
    void forEachAdj(node v, F&& f) const {
        for (auto& [neighbor, eIdx] : adj_[v.idx]) {
            f(node{neighbor}, edge{eIdx});
        }
    }
    
    // Zero-overhead ranges - size computed on access
    struct NodesRange {
        const TreeGraph* g;
        struct It { 
            uint32_t i; 
            node operator*() const { return node{i}; } 
            It& operator++() { ++i; return *this; } 
            It operator++(int) { It tmp = *this; ++i; return tmp; }
            bool operator!=(It o) const { return i != o.i; }
            bool operator==(It o) const { return i == o.i; }
        };
        It begin() const { return {0}; }
        It end() const { return {g->n_}; }
        uint32_t size() const { return g->n_; }
    };
    struct EdgesRange {
        const TreeGraph* g;
        struct It { 
            uint32_t i; 
            edge operator*() const { return edge{i}; } 
            It& operator++() { ++i; return *this; }
            It operator++(int) { It tmp = *this; ++i; return tmp; }
            bool operator!=(It o) const { return i != o.i; }
            bool operator==(It o) const { return i == o.i; }
        };
        It begin() const { return {0}; }
        It end() const { return {uint32_t(g->src_.size())}; }
        uint32_t size() const { return g->src_.size(); }
    };
    
    // OGDF-style member access (lazy evaluation - zero sync overhead)
    NodesRange nodes{this};
    EdgesRange edges{this};

    node firstNode() const { return n_ > 0 ? node{0u} : node{}; }
};

class StaticSPQRTree {
public:
    struct FlatData {
        uint32_t numNodes = 0;
        uint32_t root = UINT32_MAX;
        std::vector<uint8_t> nodeTypes;
        std::vector<uint32_t> nodeParents;
        std::vector<uint32_t> childrenOffsets;
        std::vector<uint32_t> children;
        std::vector<uint32_t> skeletonOffsets;
        std::vector<SkeletonEdge> skeletonEdges;
        std::vector<uint32_t> skeletonNumNodes;
        std::vector<uint32_t> nodeMappingOffsets;
        std::vector<uint32_t> nodeMapping;
        std::vector<uint32_t> edgeToTreeNode;

        spqr_rust::SpqrTreeFlatView view() const {
            return spqr_rust::SpqrTreeFlatView(
                numNodes,
                root,
                nodeTypes.empty() ? nullptr : nodeTypes.data(),
                nodeParents.empty() ? nullptr : nodeParents.data(),
                childrenOffsets.empty() ? nullptr : childrenOffsets.data(),
                children.empty() ? nullptr : children.data(),
                static_cast<uint32_t>(children.size()),
                skeletonOffsets.empty() ? nullptr : skeletonOffsets.data(),
                skeletonEdges.empty() ? nullptr : skeletonEdges.data(),
                static_cast<uint32_t>(skeletonEdges.size()),
                skeletonNumNodes.empty() ? nullptr : skeletonNumNodes.data(),
                nodeMappingOffsets.empty() ? nullptr : nodeMappingOffsets.data(),
                nodeMapping.empty() ? nullptr : nodeMapping.data(),
                static_cast<uint32_t>(nodeMapping.size()),
                edgeToTreeNode.empty() ? nullptr : edgeToTreeNode.data(),
                static_cast<uint32_t>(edgeToTreeNode.size()));
        }
    };

private:
    std::unique_ptr<spqr_rust::RustSPQRResult> result_;
    std::unique_ptr<FlatData> ownedFlat_;
    spqr_rust::SpqrTreeFlatView view_;
    std::vector<uint32_t> parents_;
    TreeGraph tree_;
    const Graph* gccGraph_ = nullptr;

    mutable std::unordered_map<uint64_t, uint32_t> virtualIndex_;
    mutable bool virtualIndexBuilt_ = false;

    static uint64_t packTreePair_(uint32_t from, uint32_t to) {
        return (static_cast<uint64_t>(from) << 32) | static_cast<uint64_t>(to);
    }

    void buildVirtualIndex_() const {
        if (virtualIndexBuilt_) return;
        // Walk all tree nodes, scan their skeleton edges once, index virtual ones.
        virtualIndex_.reserve(view_.numNodes * 2);
        for (uint32_t tn = 0; tn < view_.numNodes; ++tn) {
            uint32_t s = view_.skeletonOffsets[tn];
            uint32_t e = view_.skeletonOffsets[tn + 1];
            for (uint32_t i = s; i < e; ++i) {
                const auto& se = view_.skeletonEdges[i];
                if (se.real_edge == UINT32_MAX) {
                    // Virtual edge: connects tn to se.twin_tree_node
                    virtualIndex_.emplace(packTreePair_(tn, se.twin_tree_node), i);
                }
            }
        }
        virtualIndexBuilt_ = true;
    }

    void buildTree() {
        parents_.resize(view_.numNodes);
        for (uint32_t i = 0; i < view_.numNodes; ++i) parents_[i] = view_.nodeParents[i];
        tree_.build(view_.numNodes, parents_);
    }
    edge findVirtual(tree_node from, tree_node to) const {
        buildVirtualIndex_();
        auto it = virtualIndex_.find(packTreePair_(from.idx, to.idx));
        if (it == virtualIndex_.end()) return INVALID_EDGE;
        // Return LOCAL index within the from-skeleton
        return edge{it->second - view_.skeletonOffsets[from.idx]};
    }
    // Return GLOBAL edge index (unique across all skeletons) for use as map key
    edge findVirtualGlobal(tree_node from, tree_node to) const {
        buildVirtualIndex_();
        auto it = virtualIndex_.find(packTreePair_(from.idx, to.idx));
        if (it == virtualIndex_.end()) return INVALID_EDGE;
        return edge{it->second};
    }

public:
    enum class NodeType { SNode, PNode, RNode };
    using SkeletonEdge = ::SkeletonEdge;
    
    explicit StaticSPQRTree(const Graph& g)
        : result_(std::make_unique<spqr_rust::RustSPQRResult>(g.raw())),
          view_(*result_),
          gccGraph_(&g) { buildTree(); }

    /**
     * Build via SP-Compress + Reconstruct
     *
     * contractible[v] != 0 iff vertex v is eligible for Series compression
     * (for example any non-pole interior vertex of the biconnected block)
     */
    StaticSPQRTree(const Graph& g, const uint8_t* contractible, uint32_t contractible_len)
        : result_(buildViaSpCompress_(g, contractible, contractible_len)),
          view_(*result_),
          gccGraph_(&g) {
        buildTree();
    }

    StaticSPQRTree(FlatData data, const Graph* gccGraph)
        : ownedFlat_(std::make_unique<FlatData>(std::move(data))),
          view_(ownedFlat_->view()),
          gccGraph_(gccGraph) {
        buildTree();
    }

private:
    static std::unique_ptr<spqr_rust::RustSPQRResult> buildViaSpCompress_(
        const Graph& g, const uint8_t* contractible, uint32_t contractible_len)
    {
        const uint32_t n_nodes = static_cast<uint32_t>(g.numberOfNodes());
        const uint32_t n_edges = static_cast<uint32_t>(g.numberOfEdges());
        std::vector<SpCompressInputEdge> in_edges;
        in_edges.reserve(n_edges);
        for (uint32_t i = 0; i < n_edges; ++i) {
            uint32_t u = spqr_graph_edge_src(g.raw().raw(), i);
            uint32_t v = spqr_graph_edge_dst(g.raw().raw(), i);
            in_edges.push_back(SpCompressInputEdge{ u, v, i });
        }
        return std::make_unique<spqr_rust::RustSPQRResult>(
            n_nodes,
            in_edges.empty() ? nullptr : in_edges.data(),
            n_edges,
            n_edges == 0 ? 0u : n_edges - 1u,
            contractible,
            contractible_len);
    }

public:
    tree_node rootNode() const { return node{view_.root}; }
    uint32_t numberOfNodes() const { return view_.numNodes; }
    NodeType typeOf(tree_node tn) const { return view_.nodeTypes[tn.idx] == 0 ? NodeType::SNode : view_.nodeTypes[tn.idx] == 1 ? NodeType::PNode : NodeType::RNode; }
    const TreeGraph& tree() const { return tree_; }
    tree_node parent(tree_node tn) const { return node{parents_[tn.idx]}; }


    const spqr_rust::SpqrTreeFlatView& flatView() const { return view_; }
    const Graph* gccGraph() const { return gccGraph_; }

    class SkeletonGraph {
        const spqr_rust::SpqrTreeFlatView& view_;
        const Graph* gccGraph_;
        tree_node tn_;
        uint32_t nNodes_, edgeOff_, edgeEnd_;
        uint32_t mapOff_;

        mutable std::vector<uint32_t> adjStart_;
        mutable std::vector<uint32_t> adjEdges_;
        mutable std::vector<uint32_t> adjNeighbors_;
        mutable bool adjBuilt_ = false;

        std::pair<uint32_t, uint32_t> orientedEndpoints_(uint32_t skelIdx) const {
            const auto& se = view_.skeletonEdges[skelIdx];
            if (se.real_edge == UINT32_MAX || gccGraph_ == nullptr)
                return {se.src, se.dst};
            auto gSrc = gccGraph_->source(::spqr::edge{se.real_edge});
            if (view_.nodeMapping[mapOff_ + se.src] == gSrc.idx)
                return {se.src, se.dst};
            return {se.dst, se.src};
        }

        void buildAdj_() const {
            if (adjBuilt_) return;
            const uint32_t n = nNodes_;
            // 1) degree count
            adjStart_.assign(n + 1, 0);
            for (uint32_t i = edgeOff_; i < edgeEnd_; ++i) {
                auto [s, d] = orientedEndpoints_(i);
                ++adjStart_[s + 1];
                if (d != s) ++adjStart_[d + 1];
            }
            // 2) prefix sum offsets
            for (uint32_t v = 0; v < n; ++v)
                adjStart_[v + 1] += adjStart_[v];
            // 3) bucket fill
            const uint32_t total = adjStart_[n];
            adjEdges_.assign(total, 0);
            adjNeighbors_.assign(total, 0);
            std::vector<uint32_t> cursor(adjStart_.begin(), adjStart_.begin() + n);
            for (uint32_t i = edgeOff_; i < edgeEnd_; ++i) {
                auto [s, d] = orientedEndpoints_(i);
                uint32_t ps = cursor[s]++;
                adjEdges_[ps] = i;
                adjNeighbors_[ps] = d;
                if (d != s) {
                    uint32_t pd = cursor[d]++;
                    adjEdges_[pd] = i;
                    adjNeighbors_[pd] = s;
                }
            }
            adjBuilt_ = true;
        }

    public:
        SkeletonGraph(const spqr_rust::SpqrTreeFlatView& view, tree_node tn, const Graph* gccGraph)
            : view_(view), gccGraph_(gccGraph), tn_(tn),
              nNodes_(view.skeletonNumNodes[tn.idx]),
              edgeOff_(view.skeletonOffsets[tn.idx]),
              edgeEnd_(view.skeletonOffsets[tn.idx + 1]),
              mapOff_(view.nodeMappingOffsets[tn.idx]) {}

        uint32_t numberOfNodes() const { return nNodes_; }
        uint32_t numberOfEdges() const { return edgeEnd_ - edgeOff_; }

        // source/target accept GLOBAL edge indices
        node source(edge e) const { return node{orientedEndpoints_(e.idx).first}; }
        node target(edge e) const { return node{orientedEndpoints_(e.idx).second}; }

        node firstNode() const { return nNodes_ > 0 ? node{0u} : node{}; }

        template<typename F>
        void forEachAdj(node v, F&& f) const {
            buildAdj_();
            const uint32_t begin = adjStart_[v.idx];
            const uint32_t end   = adjStart_[v.idx + 1];
            for (uint32_t k = begin; k < end; ++k)
                f(node{adjNeighbors_[k]}, edge{adjEdges_[k]});
        }
        
        struct NodesRange {
            uint32_t n;
            struct It { 
                uint32_t i; 
                node operator*() const { return node{i}; } 
                It& operator++() { ++i; return *this; } 
                It operator++(int) { It tmp = *this; ++i; return tmp; }
                bool operator!=(It o) const { return i != o.i; } 
            };
            It begin() const { return {0}; }
            It end() const { return {n}; }
            uint32_t size() const { return n; }
        };
        NodesRange nodes{nNodes_};
        
        struct EdgesRange {
            uint32_t off, end_;
            struct It { 
                uint32_t i; 
                edge operator*() const { return edge{i}; } 
                It& operator++() { ++i; return *this; } 
                It operator++(int) { It tmp = *this; ++i; return tmp; }
                bool operator!=(It o) const { return i != o.i; } 
            };
            It begin() const { return {off}; }
            It end() const { return {end_}; }
            uint32_t size() const { return end_ - off; }
        };
        EdgesRange edges{edgeOff_, edgeEnd_};
    };

    class Skeleton {
        const StaticSPQRTree& t_; tree_node tn_; mutable std::unique_ptr<SkeletonGraph> g_;

        const SkeletonEdge* edgeAt(edge e) const { return &t_.view_.skeletonEdges[e.idx]; }
    public:
        Skeleton(const StaticSPQRTree& t, tree_node tn) : t_(t), tn_(tn) {}
        const SkeletonGraph& getGraph() const { if (!g_) g_ = std::make_unique<SkeletonGraph>(t_.view_, tn_, t_.gccGraph_); return *g_; }
        node original(node local) const { return node{t_.view_.nodeMapping[t_.view_.nodeMappingOffsets[tn_.idx] + local.idx]}; }
        // All edge methods accept GLOBAL edge indices
        bool isVirtual(edge e) const { return edgeAt(e)->real_edge == UINT32_MAX; }
        tree_node twinTreeNode(edge e) const { auto* se = edgeAt(e); return se->real_edge == UINT32_MAX ? node{se->twin_tree_node} : INVALID_NODE; }
        edge realEdge(edge e) const { auto* se = edgeAt(e); return se->real_edge != UINT32_MAX ? edge{se->real_edge} : INVALID_EDGE; }

        uint32_t numberOfNodes() const {
            return t_.view_.skeletonNumNodes[tn_.idx];
        }
        uint32_t numberOfEdges() const {
            return t_.view_.skeletonOffsets[tn_.idx + 1] - t_.view_.skeletonOffsets[tn_.idx];
        }

        template<typename F>
        void forEachEdge(F&& f) const {
            const auto& view = t_.view_;
            const uint32_t off = view.skeletonOffsets[tn_.idx];
            const uint32_t end = view.skeletonOffsets[tn_.idx + 1];
            const uint32_t mapOff = view.nodeMappingOffsets[tn_.idx];
            const Graph* gccGraph = t_.gccGraph_;
            for (uint32_t i = off; i < end; ++i) {
                const auto& se = view.skeletonEdges[i];
                uint32_t s = se.src, d = se.dst;
                if (se.real_edge != UINT32_MAX && gccGraph != nullptr) {
                    auto gSrc = gccGraph->source(edge{se.real_edge});
                    if (view.nodeMapping[mapOff + s] != gSrc.idx) {
                        uint32_t tmp = s; s = d; d = tmp;
                    }
                }
                f(edge{i}, node{s}, node{d});
            }
        }
    };
    
    Skeleton skeleton(tree_node tn) const { return Skeleton(*this, tn); }
    edge skeletonEdgeSrc(edge te) const { return findVirtualGlobal(tree_.source(te), tree_.target(te)); }
    edge skeletonEdgeTgt(edge te) const { return findVirtualGlobal(tree_.target(te), tree_.source(te)); }
};

using SPQRTree = StaticSPQRTree;
using Skeleton = StaticSPQRTree::Skeleton;

template<typename NA>
inline uint32_t connectedComponents(const Graph& g, NA& comp) {
    spqr_rust::RustConnectedComponents cc(g.raw());
    auto [data, len] = cc.componentsRaw();
    comp.assign(data, data + len);
    return cc.count();
}


namespace ogdf_compat {

using node = spqr::node;
constexpr node INVALID_NODE = spqr::INVALID_NODE;

class Graph;

struct edge {
    uint32_t idx;
    const Graph* g;  
    
    constexpr edge() : idx(UINT32_MAX), g(nullptr) {}
    constexpr edge(uint32_t i) : idx(i), g(nullptr) {}
    constexpr edge(uint32_t i, const Graph* gr) : idx(i), g(gr) {}
    constexpr edge(std::nullptr_t) : idx(UINT32_MAX), g(nullptr) {}
    
    constexpr uint32_t index() const { return idx; }
    
    constexpr bool operator!() const { return idx == UINT32_MAX; }
    constexpr explicit operator bool() const { return idx != UINT32_MAX; }
    constexpr bool operator==(std::nullptr_t) const { return idx == UINT32_MAX; }
    constexpr bool operator!=(std::nullptr_t) const { return idx != UINT32_MAX; }
    constexpr bool operator==(edge o) const { return idx == o.idx; }
    constexpr bool operator!=(edge o) const { return idx != o.idx; }
    constexpr bool operator<(edge o) const { return idx < o.idx; }
    
    inline node source() const;
    inline node target() const;
    
    const edge* operator->() const { return this; }
};

const edge INVALID_EDGE{UINT32_MAX, nullptr};

struct adjEntry {
    node neighbor;
    edge e;
    node twinNode() const { return neighbor; }
    edge theEdge() const { return e; }
};

template<typename T>
class NodeArray {
    mutable std::vector<T> data_;
    T default_{};
    const void* graph_ = nullptr;
    uint32_t (*size_fn_)(const void*) = nullptr;

    template<typename G> static uint32_t node_count_of(const void* g) {
        return static_cast<const G*>(g)->numberOfNodes();
    }

    void sync_() const {
        if (size_fn_) {
            uint32_t n = size_fn_(graph_);
            if (data_.size() < n) data_.resize(n, default_);
        }
    }
public:
    NodeArray() = default;
    template<typename G> NodeArray(const G& g, const T& def = T())
        : data_(g.numberOfNodes(), def), default_(def),
          graph_(&g), size_fn_(&node_count_of<G>) {}
    NodeArray(size_t n, const T& def = T()) : data_(n, def), default_(def) {}

    template<typename G> void init(const G& g, const T& def = T()) {
        default_ = def;
        graph_ = &g;
        size_fn_ = &node_count_of<G>;
        data_.assign(g.numberOfNodes(), def);
    }

    T& operator[](size_t idx) {
        sync_();
        if (idx >= data_.size()) data_.resize(idx + 1, default_);
        return data_[idx];
    }
    const T& operator[](size_t idx) const {
        sync_();
        if (idx >= data_.size()) data_.resize(idx + 1, default_);
        return data_[idx];
    }

    T& operator()(node v) { return (*this)[v.idx]; }
    const T& operator()(node v) const { return (*this)[v.idx]; }

    size_t size() const { sync_(); return data_.size(); }
    void resize(size_t n, const T& val = T()) { data_.resize(n, val); }
    void clear() { data_.clear(); }
    template<typename Iter> void assign(Iter first, Iter last) { data_.assign(first, last); }
    void assign(size_t n, const T& val) { data_.assign(n, val); }
    auto begin() { sync_(); return data_.begin(); }
    auto end() { sync_(); return data_.end(); }
    auto begin() const { sync_(); return data_.begin(); }
    auto end() const { sync_(); return data_.end(); }
};

// Specialization for bool to avoid std::vector<bool> proxy issues
template<>
class NodeArray<bool> {
    mutable std::vector<char> data_;
    char default_ = 0;
    const void* graph_ = nullptr;
    uint32_t (*size_fn_)(const void*) = nullptr;

    template<typename G> static uint32_t node_count_of(const void* g) {
        return static_cast<const G*>(g)->numberOfNodes();
    }

    void sync_() const {
        if (size_fn_) {
            uint32_t n = size_fn_(graph_);
            if (data_.size() < n) data_.resize(n, default_);
        }
    }
public:
    NodeArray() = default;
    template<typename G> NodeArray(const G& g, bool def = false)
        : data_(g.numberOfNodes(), def ? 1 : 0), default_(def ? 1 : 0),
          graph_(&g), size_fn_(&node_count_of<G>) {}
    NodeArray(size_t n, bool def = false) : data_(n, def ? 1 : 0), default_(def ? 1 : 0) {}

    template<typename G> void init(const G& g, bool def = false) {
        default_ = def ? 1 : 0;
        graph_ = &g;
        size_fn_ = &node_count_of<G>;
        data_.assign(g.numberOfNodes(), default_);
    }

    class Ref {
        char& c_;
    public:
        Ref(char& c) : c_(c) {}
        operator bool() const { return c_ != 0; }
        Ref& operator=(bool b) { c_ = b ? 1 : 0; return *this; }
    };

    Ref operator[](size_t idx) {
        sync_();
        if (idx >= data_.size()) data_.resize(idx + 1, default_);
        return Ref(data_[idx]);
    }
    bool operator[](size_t idx) const {
        sync_();
        if (idx >= data_.size()) data_.resize(idx + 1, default_);
        return data_[idx] != 0;
    }

    Ref operator()(node v) { return (*this)[v.idx]; }
    bool operator()(node v) const { return (*this)[v.idx]; }

    size_t size() const { sync_(); return data_.size(); }
    void resize(size_t n, bool val = false) { data_.resize(n, val ? 1 : 0); }
    void clear() { data_.clear(); }
};

template<typename T>
class EdgeArray {
    mutable std::vector<T> data_;
    T default_{};
    const void* graph_ = nullptr;
    uint32_t (*size_fn_)(const void*) = nullptr;

    template<typename G> static uint32_t edge_count_of(const void* g) {
        return static_cast<const G*>(g)->numberOfEdges();
    }

    void sync_() const {
        if (size_fn_) {
            uint32_t n = size_fn_(graph_);
            if (data_.size() < n) data_.resize(n, default_);
        }
    }
public:
    EdgeArray() = default;
    template<typename G> EdgeArray(const G& g, const T& def = T())
        : data_(g.numberOfEdges(), def), default_(def),
          graph_(&g), size_fn_(&edge_count_of<G>) {}
    EdgeArray(size_t n, const T& def = T()) : data_(n, def), default_(def) {}

    template<typename G> void init(const G& g, const T& def = T()) {
        default_ = def;
        graph_ = &g;
        size_fn_ = &edge_count_of<G>;
        data_.assign(g.numberOfEdges(), def);
    }

    T& operator[](size_t idx) {
        sync_();
        if (idx >= data_.size()) data_.resize(idx + 1, default_);
        return data_[idx];
    }
    const T& operator[](size_t idx) const {
        sync_();
        if (idx >= data_.size()) data_.resize(idx + 1, default_);
        return data_[idx];
    }

    T& operator()(edge e) { return (*this)[e.idx]; }
    const T& operator()(edge e) const { return (*this)[e.idx]; }
    T& operator()(::spqr::edge e) { return (*this)[e.idx]; }
    const T& operator()(::spqr::edge e) const { return (*this)[e.idx]; }

    size_t size() const { sync_(); return data_.size(); }
    void resize(size_t n, const T& val = T()) { data_.resize(n, val); }
    void clear() { data_.clear(); }
    template<typename Iter> void assign(Iter first, Iter last) { data_.assign(first, last); }
    void assign(size_t n, const T& val) { data_.assign(n, val); }
    auto begin() { sync_(); return data_.begin(); }
    auto end() { sync_(); return data_.end(); }
    auto begin() const { sync_(); return data_.begin(); }
    auto end() const { sync_(); return data_.end(); }
};

// Specialization for bool to avoid std::vector<bool> proxy issues
template<>
class EdgeArray<bool> {
    mutable std::vector<char> data_;
    char default_ = 0;
    const void* graph_ = nullptr;
    uint32_t (*size_fn_)(const void*) = nullptr;

    template<typename G> static uint32_t edge_count_of(const void* g) {
        return static_cast<const G*>(g)->numberOfEdges();
    }

    void sync_() const {
        if (size_fn_) {
            uint32_t n = size_fn_(graph_);
            if (data_.size() < n) data_.resize(n, default_);
        }
    }
public:
    EdgeArray() = default;
    template<typename G> EdgeArray(const G& g, bool def = false)
        : data_(g.numberOfEdges(), def ? 1 : 0), default_(def ? 1 : 0),
          graph_(&g), size_fn_(&edge_count_of<G>) {}
    EdgeArray(size_t n, bool def = false) : data_(n, def ? 1 : 0), default_(def ? 1 : 0) {}

    template<typename G> void init(const G& g, bool def = false) {
        default_ = def ? 1 : 0;
        graph_ = &g;
        size_fn_ = &edge_count_of<G>;
        data_.assign(g.numberOfEdges(), default_);
    }

    class Ref {
        char& c_;
    public:
        Ref(char& c) : c_(c) {}
        operator bool() const { return c_ != 0; }
        Ref& operator=(bool b) { c_ = b ? 1 : 0; return *this; }
    };

    Ref operator[](size_t idx) {
        sync_();
        if (idx >= data_.size()) data_.resize(idx + 1, default_);
        return Ref(data_[idx]);
    }
    bool operator[](size_t idx) const {
        sync_();
        if (idx >= data_.size()) data_.resize(idx + 1, default_);
        return data_[idx] != 0;
    }

    Ref operator()(edge e) { return (*this)[e.idx]; }
    bool operator()(edge e) const { return (*this)[e.idx]; }
    Ref operator()(::spqr::edge e) { return (*this)[e.idx]; }
    bool operator()(::spqr::edge e) const { return (*this)[e.idx]; }

    size_t size() const { sync_(); return data_.size(); }
    void resize(size_t n, bool val = false) { data_.resize(n, val ? 1 : 0); }
    void clear() { data_.clear(); }
};

class Graph {
    std::unique_ptr<spqr_rust::RustGraph> g_;
    mutable std::vector<adjEntry> adjCache_;

public:
    struct NodesRange {
        const Graph* g;
        struct It { uint32_t i; node operator*() const { return node{i}; } It& operator++() { ++i; return *this; } bool operator!=(It o) const { return i != o.i; } };
        It begin() const { return {0}; }
        It end() const { return {g->numberOfNodes()}; }
    };
    struct EdgesRange {
        const Graph* g;
        struct It { uint32_t i; const Graph* g; edge operator*() const { return edge{i, g}; } It& operator++() { ++i; return *this; } bool operator!=(It o) const { return i != o.i; } };
        It begin() const { return {0, g}; }
        It end() const { return {g->numberOfEdges(), g}; }
    };

    Graph() : g_(std::make_unique<spqr_rust::RustGraph>()), nodes{this}, edges{this} {}
    
    node newNode() { return node{g_->addNode()}; }
    edge newEdge(node u, node v) { return edge{g_->addEdge(u.idx, v.idx), this}; }
    
    uint32_t numberOfNodes() const { return g_->numNodes(); }
    uint32_t numberOfEdges() const { return g_->numEdges(); }
    
    node firstNode() const { return numberOfNodes() > 0 ? node{0u} : INVALID_NODE; }
    
    node source(edge e) const { return node{g_->edgeSrc(e.idx)}; }
    node target(edge e) const { return node{g_->edgeDst(e.idx)}; }
    
    NodesRange nodes;
    EdgesRange edges;
    
    template<typename F>
    void forEachAdj(node v, F&& f) const {
        g_->forEachNeighbor(v.idx, [&](uint32_t n, uint32_t e) { f(node{n}, edge{e, this}); });
    }
    
    const std::vector<adjEntry>& getAdjEntries(node v) const {
        adjCache_.clear();
        g_->forEachNeighbor(v.idx, [this](uint32_t n, uint32_t e) {
            adjCache_.push_back({node{n}, edge{e, this}});
        });
        return adjCache_;
    }
    
    uint32_t degree(node v) const { return g_->degree(v.idx); }
    
    spqr_rust::RustGraph& raw() { return *g_; }
    const spqr_rust::RustGraph& raw() const { return *g_; }
};

inline node edge::source() const { return g->source(*this); }
inline node edge::target() const { return g->target(*this); }

// BCTree / StaticSPQRTree for ogdf_compat
// (mostly copy-paste from above, but using ogdf_compat::edge)

class BCTree {
    std::unique_ptr<spqr_rust::RustBCTree> bc_;
    std::vector<bool> isCut_;
public:
    enum class BNodeType { BComp, CComp };
    enum class GNodeType { Normal, CutVertex };
    
    explicit BCTree(const Graph& g) : bc_(std::make_unique<spqr_rust::RustBCTree>(g.raw())) {
        isCut_.assign(g.numberOfNodes(), false);
        for (uint32_t v : bc_->cutVertices()) isCut_[v] = true;
    }
    
    uint32_t numberOfBComps() const { return bc_->numBlocks(); }
    uint32_t numberOfCComps() const { return bc_->numCutVertices(); }
    GNodeType typeOfGNode(node v) const { return isCut_[v.idx] ? GNodeType::CutVertex : GNodeType::Normal; }
    BNodeType typeOfBNode(node v) const { return v.idx < bc_->numBlocks() ? BNodeType::BComp : BNodeType::CComp; }
    
    std::vector<edge> hEdges(node bNode) const {
        if (bNode.idx >= bc_->numBlocks()) return {};
        auto raw = bc_->blockEdges(bNode.idx);
        std::vector<edge> r; r.reserve(raw.size());
        for (auto e : raw) r.push_back(edge{e});
        return r;
    }
    edge original(edge e) const { return e; }
    node repVertex(node v, node) const { return v; }
    node bcproper(node v) const { return v; }
    
    struct BCTreeGraph {
        uint32_t n_;
        struct NodesRange { 
            uint32_t n; 
            struct It { 
                uint32_t i; 
                node operator*() const { return node{i}; } 
                It& operator++() { ++i; return *this; }
                It operator++(int) { It tmp = *this; ++i; return tmp; }
                bool operator!=(It o) const { return i != o.i; }
                bool operator==(It o) const { return i == o.i; }
            }; 
            It begin() const { return {0}; } 
            It end() const { return {n}; }
            uint32_t size() const { return n; }
        };
        NodesRange nodes{0};
        BCTreeGraph(uint32_t n) : n_(n), nodes{n} {}
        uint32_t numberOfNodes() const { return n_; }
    };
    BCTreeGraph bcTree() const { return BCTreeGraph{bc_->numBlocks() + bc_->numCutVertices()}; }
};

using tree_node = node;

class TreeGraph {
    uint32_t n_ = 0;
    std::vector<uint32_t> parents_, src_, tgt_;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> adj_;
public:
    void build(uint32_t n, const std::vector<uint32_t>& parents) {
        n_ = n; parents_ = parents; src_.clear(); tgt_.clear();
        adj_.assign(n, {});
        for (uint32_t i = 0; i < n; ++i) {
            if (parents[i] != UINT32_MAX && parents[i] != i) {
                uint32_t eIdx = src_.size();
                src_.push_back(parents[i]);
                tgt_.push_back(i);
                adj_[parents[i]].push_back({i, eIdx});
                adj_[i].push_back({parents[i], eIdx});
            }
        }
    }
    uint32_t numberOfNodes() const { return n_; }
    uint32_t numberOfEdges() const { return src_.size(); }
    node source(edge e) const { return node{src_[e.idx]}; }
    node target(edge e) const { return node{tgt_[e.idx]}; }
    template<typename F>
    void forEachAdj(node v, F&& f) const {
        for (auto& [neighbor, eIdx] : adj_[v.idx]) {
            f(node{neighbor}, edge{eIdx});
        }
    }
    
    // Zero-overhead ranges - size computed on access
    struct NodesRange {
        const TreeGraph* g;
        struct It { 
            uint32_t i; 
            node operator*() const { return node{i}; } 
            It& operator++() { ++i; return *this; }
            It operator++(int) { It tmp = *this; ++i; return tmp; }
            bool operator!=(It o) const { return i != o.i; }
            bool operator==(It o) const { return i == o.i; }
        };
        It begin() const { return {0}; }
        It end() const { return {g->n_}; }
        uint32_t size() const { return g->n_; }
    };
    struct EdgesRange {
        const TreeGraph* g;
        struct It { 
            uint32_t i; 
            edge operator*() const { return edge{i}; } 
            It& operator++() { ++i; return *this; }
            It operator++(int) { It tmp = *this; ++i; return tmp; }
            bool operator!=(It o) const { return i != o.i; }
            bool operator==(It o) const { return i == o.i; }
        };
        It begin() const { return {0}; }
        It end() const { return {uint32_t(g->src_.size())}; }
        uint32_t size() const { return g->src_.size(); }
    };
    
    NodesRange nodes{this};
    EdgesRange edges{this};
    node firstNode() const { return n_ > 0 ? node{0u} : node{}; }
};

class StaticSPQRTree {
    std::unique_ptr<spqr_rust::RustSPQRResult> result_;
    spqr_rust::SpqrTreeFlatView view_;
    std::vector<uint32_t> parents_;
    TreeGraph tree_;

    // Precomputed O(1) virtual-edge index (see spqr::StaticSPQRTree for rationale).
    mutable std::unordered_map<uint64_t, uint32_t> virtualIndex_;
    mutable bool virtualIndexBuilt_ = false;

    static uint64_t packTreePair_(uint32_t from, uint32_t to) {
        return (static_cast<uint64_t>(from) << 32) | static_cast<uint64_t>(to);
    }

    void buildVirtualIndex_() const {
        if (virtualIndexBuilt_) return;
        virtualIndex_.reserve(view_.numNodes * 2);
        for (uint32_t tn = 0; tn < view_.numNodes; ++tn) {
            uint32_t s = view_.skeletonOffsets[tn];
            uint32_t e = view_.skeletonOffsets[tn + 1];
            for (uint32_t i = s; i < e; ++i) {
                const auto& se = view_.skeletonEdges[i];
                if (se.real_edge == UINT32_MAX) {
                    virtualIndex_.emplace(packTreePair_(tn, se.twin_tree_node), i);
                }
            }
        }
        virtualIndexBuilt_ = true;
    }

    void buildTree() { parents_.resize(view_.numNodes); for (uint32_t i = 0; i < view_.numNodes; ++i) parents_[i] = view_.nodeParents[i]; tree_.build(view_.numNodes, parents_); }
    edge findVirtual(tree_node from, tree_node to) const {
        buildVirtualIndex_();
        auto it = virtualIndex_.find(packTreePair_(from.idx, to.idx));
        if (it == virtualIndex_.end()) return INVALID_EDGE;
        return edge{it->second - view_.skeletonOffsets[from.idx]};
    }
    edge findVirtualGlobal(tree_node from, tree_node to) const {
        buildVirtualIndex_();
        auto it = virtualIndex_.find(packTreePair_(from.idx, to.idx));
        if (it == virtualIndex_.end()) return INVALID_EDGE;
        return edge{it->second};
    }
public:
    enum class NodeType { SNode, PNode, RNode };
    using SkeletonEdge = ::SkeletonEdge;
    
    explicit StaticSPQRTree(const Graph& g) : result_(std::make_unique<spqr_rust::RustSPQRResult>(g.raw())), view_(*result_) { buildTree(); }

    /**
     * Build via SP-Compress + Reconstruct.
     *
     * contractible is a byte mask indexed by node ID: contractible[v] != 0
     * iff vertex v is eligible for Series compression. Typically, the caller
     * marks every interior vertex of the biconnected block (= all vertices
     * except the BC-tree poles, or equivalently every vertex whose degree in
     * the block equals its degree in the global graph minus 0... in practice,
     * just all vertices that are not BCcut vertices)
     *
     * The SPQR tree returned is isomorphic to that of the regular
     * constructor StaticSPQRTree(const Graph&) (modulo as permutation of skeleton 
     * edges and children).
     */
    StaticSPQRTree(const Graph& g, const uint8_t* contractible, uint32_t contractible_len)
        : result_(buildViaSpCompress_(g, contractible, contractible_len)),
          view_(*result_) {
        buildTree();
    }

private:
    static std::unique_ptr<spqr_rust::RustSPQRResult> buildViaSpCompress_(
        const Graph& g, const uint8_t* contractible, uint32_t contractible_len)
    {
        const uint32_t n_nodes = static_cast<uint32_t>(g.numberOfNodes());
        const uint32_t n_edges = static_cast<uint32_t>(g.numberOfEdges());

        // Materialize edges into the FFI representation.
        std::vector<SpCompressInputEdge> in_edges;
        in_edges.reserve(n_edges);
        for (uint32_t i = 0; i < n_edges; ++i) {
            edge e{i};
            // Use the raw graph to get src/dst.
            uint32_t u = spqr_graph_edge_src(g.raw().raw(), i);
            uint32_t v = spqr_graph_edge_dst(g.raw().raw(), i);
            in_edges.push_back(SpCompressInputEdge{ u, v, i });
        }

        return std::make_unique<spqr_rust::RustSPQRResult>(
            n_nodes,
            in_edges.empty() ? nullptr : in_edges.data(),
            n_edges,
            n_edges == 0 ? 0u : n_edges - 1u,
            contractible,
            contractible_len);
    }

public:
    tree_node rootNode() const { return node{view_.root}; }
    uint32_t numberOfNodes() const { return view_.numNodes; }
    NodeType typeOf(tree_node tn) const { return view_.nodeTypes[tn.idx] == 0 ? NodeType::SNode : view_.nodeTypes[tn.idx] == 1 ? NodeType::PNode : NodeType::RNode; }
    const TreeGraph& tree() const { return tree_; }
    tree_node parent(tree_node tn) const { return node{parents_[tn.idx]}; }

    const spqr_rust::SpqrTreeFlatView& flatView() const { return view_; }
    
    class SkeletonGraph {
        const spqr_rust::SpqrTreeFlatView& view_;
        tree_node tn_;
        uint32_t nNodes_, edgeOff_, edgeEnd_;
        mutable std::vector<uint32_t> adjStart_;
        mutable std::vector<uint32_t> adjEdges_;
        mutable std::vector<uint32_t> adjNeighbors_;
        mutable bool adjBuilt_ = false;

        void buildAdj_() const {
            if (adjBuilt_) return;
            const uint32_t n = nNodes_;
            adjStart_.assign(n + 1, 0);
            for (uint32_t i = edgeOff_; i < edgeEnd_; ++i) {
                const auto& se = view_.skeletonEdges[i];
                ++adjStart_[se.src + 1];
                if (se.dst != se.src) ++adjStart_[se.dst + 1];
            }
            for (uint32_t v = 0; v < n; ++v)
                adjStart_[v + 1] += adjStart_[v];
            const uint32_t total = adjStart_[n];
            adjEdges_.assign(total, 0);
            adjNeighbors_.assign(total, 0);
            std::vector<uint32_t> cursor(adjStart_.begin(), adjStart_.begin() + n);
            for (uint32_t i = edgeOff_; i < edgeEnd_; ++i) {
                const auto& se = view_.skeletonEdges[i];
                uint32_t ps = cursor[se.src]++;
                adjEdges_[ps] = i;
                adjNeighbors_[ps] = se.dst;
                if (se.dst != se.src) {
                    uint32_t pd = cursor[se.dst]++;
                    adjEdges_[pd] = i;
                    adjNeighbors_[pd] = se.src;
                }
            }
            adjBuilt_ = true;
        }

    public:
        SkeletonGraph(const spqr_rust::SpqrTreeFlatView& view, tree_node tn)
            : view_(view), tn_(tn), 
              nNodes_(view.skeletonNumNodes[tn.idx]),
              edgeOff_(view.skeletonOffsets[tn.idx]), 
              edgeEnd_(view.skeletonOffsets[tn.idx + 1]) {}
        
        uint32_t numberOfNodes() const { return nNodes_; }
        uint32_t numberOfEdges() const { return edgeEnd_ - edgeOff_; }
        node source(edge e) const { return node{view_.skeletonEdges[e.idx].src}; }
        node target(edge e) const { return node{view_.skeletonEdges[e.idx].dst}; }
        node firstNode() const { return nNodes_ > 0 ? node{0u} : node{}; }
        
        // O(deg(v)) after a single O(E_skel) build on first call.
        template<typename F>
        void forEachAdj(node v, F&& f) const {
            buildAdj_();
            const uint32_t begin = adjStart_[v.idx];
            const uint32_t end   = adjStart_[v.idx + 1];
            for (uint32_t k = begin; k < end; ++k)
                f(node{adjNeighbors_[k]}, edge{adjEdges_[k]});
        }
        
        struct NodesRange {
            uint32_t n;
            struct It { 
                uint32_t i; 
                node operator*() const { return node{i}; } 
                It& operator++() { ++i; return *this; } 
                It operator++(int) { It tmp = *this; ++i; return tmp; }
                bool operator!=(It o) const { return i != o.i; } 
            };
            It begin() const { return {0}; }
            It end() const { return {n}; }
            uint32_t size() const { return n; }
        };
        NodesRange nodes{nNodes_};
        
        struct EdgesRange {
            uint32_t off, end_;
            struct It { 
                uint32_t i; 
                edge operator*() const { return edge{i}; } 
                It& operator++() { ++i; return *this; } 
                It operator++(int) { It tmp = *this; ++i; return tmp; }
                bool operator!=(It o) const { return i != o.i; } 
            };
            It begin() const { return {off}; }
            It end() const { return {end_}; }
            uint32_t size() const { return end_ - off; }
        };
        EdgesRange edges{edgeOff_, edgeEnd_};
    };

    class Skeleton {
        const StaticSPQRTree& t_; tree_node tn_; mutable std::unique_ptr<SkeletonGraph> g_;
        const SkeletonEdge* edgeAt(edge e) const { return &t_.view_.skeletonEdges[e.idx]; }
    public:
        Skeleton(const StaticSPQRTree& t, tree_node tn) : t_(t), tn_(tn) {}
        const SkeletonGraph& getGraph() const { if (!g_) g_ = std::make_unique<SkeletonGraph>(t_.view_, tn_); return *g_; }
        node original(node local) const { return node{t_.view_.nodeMapping[t_.view_.nodeMappingOffsets[tn_.idx] + local.idx]}; }
        bool isVirtual(edge e) const { return edgeAt(e)->real_edge == UINT32_MAX; }
        tree_node twinTreeNode(edge e) const { auto* se = edgeAt(e); return se->real_edge == UINT32_MAX ? node{se->twin_tree_node} : INVALID_NODE; }
        edge realEdge(edge e) const { auto* se = edgeAt(e); return se->real_edge != UINT32_MAX ? edge{se->real_edge} : INVALID_EDGE; }

        // V2 zero-alloc accessors (added 2026-04-23). Same semantics as
        // this namespace's SkeletonGraph::source/target (no reorientation).
        uint32_t numberOfNodes() const {
            return t_.view_.skeletonNumNodes[tn_.idx];
        }
        uint32_t numberOfEdges() const {
            return t_.view_.skeletonOffsets[tn_.idx + 1] - t_.view_.skeletonOffsets[tn_.idx];
        }

        template<typename F>
        void forEachEdge(F&& f) const {
            const auto& view = t_.view_;
            const uint32_t off = view.skeletonOffsets[tn_.idx];
            const uint32_t end = view.skeletonOffsets[tn_.idx + 1];
            for (uint32_t i = off; i < end; ++i) {
                const auto& se = view.skeletonEdges[i];
                f(edge{i}, node{se.src}, node{se.dst});
            }
        }
    };
    
    Skeleton skeleton(tree_node tn) const { return Skeleton(*this, tn); }
    edge skeletonEdgeSrc(edge te) const { return findVirtualGlobal(tree_.source(te), tree_.target(te)); }
    edge skeletonEdgeTgt(edge te) const { return findVirtualGlobal(tree_.target(te), tree_.source(te)); }
};

using SPQRTree = StaticSPQRTree;
using Skeleton = StaticSPQRTree::Skeleton;

inline uint32_t connectedComponents(const Graph& g, NodeArray<int>& comp) {
    spqr_rust::RustConnectedComponents cc(g.raw());
    auto [data, len] = cc.componentsRaw();
    comp.assign(data, data + len);
    return cc.count();
}

template<typename NA>
inline uint32_t connectedComponents(const Graph& g, NA& comp) {
    spqr_rust::RustConnectedComponents cc(g.raw());
    auto [data, len] = cc.componentsRaw();
    comp.assign(data, data + len);
    return cc.count();
}

} 
} 

namespace spqr {

// Check if directed graph is acyclic 
inline bool isAcyclic(const Graph& G) {
    NodeArray<int> state(G, 0);  // 0=unvisited, 1=visiting, 2=done
    bool hasCycle = false;
    
    std::function<void(node)> dfs = [&](node u) {
        if (hasCycle) return;
        state[u] = 1;
        G.forEachAdj(u, [&](node v, edge e) {
            if (hasCycle) return;
            if (G.source(e) != u) return; 
            if (state[v] == 1) { hasCycle = true; return; }
            if (state[v] == 0) dfs(v);
        });
        state[u] = 2;
    };
    
    for (node v : G.nodes) {
        if (state[v] == 0) dfs(v);
        if (hasCycle) return false;
    }
    return true;
}

// Acyclicity check pretending edge skip is absent
inline bool isAcyclicWithoutEdge(const Graph& G, edge skip) {
    NodeArray<int> state(G, 0);
    bool hasCycle = false;

    std::function<void(node)> dfs = [&](node u) {
        if (hasCycle) return;
        state[u] = 1;
        G.forEachAdj(u, [&](node v, edge e) {
            if (hasCycle) return;
            if (e.idx == skip.idx) return; // pretend this edge is gone
            if (G.source(e) != u) return; // only outgoing
            if (state[v] == 1) { hasCycle = true; return; }
            if (state[v] == 0) dfs(v);
        });
        state[u] = 2;
    };

    for (node v : G.nodes) {
        if (state[v] == 0) dfs(v);
        if (hasCycle) return false;
    }
    return true;
}

// Compute strongly CC (with Kosaraju algorithm)
template<typename NA>
inline int strongComponents(const Graph& G, NA& comp) {
    const uint32_t n = G.numberOfNodes();
    comp.init(G, -1);
    
    // First DFS to get finish order
    std::vector<node> order;
    order.reserve(n);
    NodeArray<bool> vis(G, false);
    
    std::function<void(node)> dfs1 = [&](node u) {
        vis[u] = true;
        G.forEachAdj(u, [&](node v, edge e) {
            if (G.source(e) != u) return;  // outgoing only
            if (!vis[v]) dfs1(v);
        });
        order.push_back(u);
    };
    
    for (node v : G.nodes) {
        if (!vis[v]) dfs1(v);
    }
    
    // Build reverse adjacency
    std::vector<std::vector<node>> radj(n);
    for (edge e : G.edges) {
        radj[G.target(e).idx].push_back(G.source(e));
    }
    
    // Second DFS on reverse graph in reverse finish order
    int numSCC = 0;
    std::function<void(node, int)> dfs2 = [&](node u, int c) {
        comp[u] = c;
        for (node v : radj[u.idx]) {
            if (comp[v] == -1) dfs2(v, c);
        }
    };
    
    for (int i = n - 1; i >= 0; --i) {
        node u = order[i];
        if (comp[u] == -1) {
            dfs2(u, numSCC++);
        }
    }
    
    return numSCC;
}

}

namespace std {
    template<> struct hash<spqr::node> {
        size_t operator()(spqr::node n) const noexcept {
            return std::hash<uint32_t>{}(n.idx);
        }
    };
    template<> struct hash<spqr::edge> {
        size_t operator()(spqr::edge e) const noexcept {
            return std::hash<uint32_t>{}(e.idx);
        }
    };
}
