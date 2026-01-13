#include "graph_io.hpp"
#include "util/context.hpp"
#include "util/timer.hpp"
#include "util/logger.hpp"

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

void readGFA()
{
    auto &C = ctx();

    if (C.graphPath.empty())
        throw std::runtime_error("GFA input needs -g <file>");

    std::ifstream in(C.graphPath);
    if (!in) throw std::runtime_error("Cannot open " + C.graphPath);

    // [AJOUT] Nettoyage préalable pour le stockages des données brutes
    C.gfaSegmentIds.clear();
    C.gfaLinkLines.clear();

    // Initialisation de la map des types d'incidences pour Snarls/Ultrabubbles
    if (C.bubbleType == Context::BubbleType::SNARL ||
        C.bubbleType == Context::BubbleType::ULTRABUBBLE) {
        C._edge2types.init(C.G,
                           std::make_pair(EdgePartType::NONE, EdgePartType::NONE));
    }

    std::unordered_set<std::string> have_segment;
    std::vector<std::string> raw_edges;
    raw_edges.reserve(1 << 16);

    // Helper pour créer les nœuds lazy (Superbubble)
    auto ensure = [&](const std::string& name){
        if (!C.name2node.count(name)) {
            auto id = C.G.newNode();
            C.name2node[name] = id;
            C.node2name[id]   = name;
        }
    };

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        // Lecture des Segments (S)
        if (line[0] == 'S') {
            std::istringstream iss(line);
            std::string tok, id, seq;
            if (!(iss >> tok >> id >> seq)) {
                // On peut loguer une erreur ou continuer, ici on continue par sécurité
                continue;
            }
            if (tok != "S") continue;
            if (id.empty()) continue;

            // [FIX] Protection contre double S-line (évite de créer des nœuds fantômes)
            if (C.name2node.count(id)) continue;

            // [AJOUT] Stockage de l'ID brut pour l'aux-graph (recompute snarls)
            C.gfaSegmentIds.push_back(id);

            // Création des nœuds selon le mode
            if (C.bubbleType == Context::BubbleType::SNARL ||
                C.bubbleType == Context::BubbleType::ULTRABUBBLE) {
                // 1 segment = 1 nœud OGDF
                have_segment.insert(id);
                auto newNode = C.G.newNode();
                C.name2node[id]      = newNode;
                C.node2name[newNode] = id;

            } else {
                // SUPERBUBBLE : 1 segment = 2 nœuds (id+, id-)
                have_segment.insert(id);
                ensure(id + "+");
                ensure(id + "-");
            }
            continue;
        }

        // Lecture des Liens (L) -> Stockage temporaire
        if (line[0] == 'L') {
            raw_edges.push_back(line);
            continue;
        }
    }

    in.close();

    // [AJOUT] Copie des liens bruts pour l'aux-graph
    C.gfaLinkLines = raw_edges;


    // --- Définition des structures de hachage et helpers pour la 2ème passe ---

    auto flip = [](char c){ return c == '+' ? '-' : '+'; };

    // Pour SUPERBUBBLE
    struct EdgeKey {
        std::string u, v;
        bool operator==(const EdgeKey& o) const { return u == o.u && v == o.v; }
    };
    struct EdgeKeyHash {
        std::size_t operator()(EdgeKey const& k) const {
            std::size_t h1 = std::hash<std::string>{}(k.u);
            std::size_t h2 = std::hash<std::string>{}(k.v);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
        }
    };
    std::unordered_set<EdgeKey, EdgeKeyHash> seen;

    // Pour SNARL (comptage des multi-edges pour insérer _trash)
    struct PairHash {
        std::size_t operator()(const std::pair<std::string,std::string>& p) const noexcept {
            std::size_t h1 = std::hash<std::string>{}(p.first);
            std::size_t h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> pair_count_snarl;

    if (C.bubbleType == Context::BubbleType::SNARL) {
        for (const std::string& e : raw_edges) {
            std::istringstream iss(e);
            std::string tok, from, to, ovl_str;
            char o1 = 0, o2 = 0;
            if (!(iss >> tok >> from >> o1 >> to >> o2 >> ovl_str)) continue;
            if (tok != "L") continue;
            
            if (have_segment.count(from) && have_segment.count(to)) {
                auto key = (from < to) ? std::make_pair(from, to) : std::make_pair(to, from);
                pair_count_snarl[key]++;
            }
        }
    }

    // Pour ULTRABUBBLE (Déduplication bidirectionnelle stricte)
    struct BiEdgeKey {
        std::string u;
        EdgePartType su;
        std::string v;
        EdgePartType sv;
        bool operator==(BiEdgeKey const& o) const noexcept {
            return u == o.u && su == o.su && v == o.v && sv == o.sv;
        }
    };
    struct BiEdgeKeyHash {
        std::size_t operator()(BiEdgeKey const& k) const noexcept {
            auto hs = [](EdgePartType t) -> std::size_t {
                return t == EdgePartType::PLUS ? 0x9e3779b9u :
                       t == EdgePartType::MINUS ? 0x85ebca6bu : 0u;
            };
            std::size_t h1 = std::hash<std::string>{}(k.u) ^ (hs(k.su) + 0x9e3779b97f4a7c15ULL);
            std::size_t h2 = std::hash<std::string>{}(k.v) ^ (hs(k.sv) + 0x9e3779b97f4a7c15ULL);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
        }
    };
    std::unordered_set<BiEdgeKey, BiEdgeKeyHash> seen_bidir_ultrabubble;
    if (C.bubbleType == Context::BubbleType::ULTRABUBBLE) {
        seen_bidir_ultrabubble.reserve(raw_edges.size() * 2);
    }

    // --- Lambdas d'ajout d'arêtes ---

    auto add_edge_double = [&](const std::string& u, const std::string& v){
        EdgeKey key{u, v};
        if (seen.insert(key).second) {
            C.G.newEdge(C.name2node[u], C.name2node[v]);
        }
    };

    auto add_edge_snarl = [&](std::string& u, std::string& v, 
                              EdgePartType t1, EdgePartType t2) 
    {
        if (u > v) {
            std::swap(u, v);
            std::swap(t1, t2);
        }
        auto key = std::make_pair(u, v);
        bool multi = pair_count_snarl[key] > 1;

        if (!multi) {
            ogdf::edge e = C.G.newEdge(C.name2node[u], C.name2node[v]);
            C._edge2types[e] = std::make_pair(t1, t2);
            return;
        }

        std::string mid_name = "_trash";
        auto mid_node = C.G.newNode();
        C.node2name[mid_node] = mid_name;

        {
            ogdf::edge e1 = C.G.newEdge(C.name2node[u], mid_node);
            C._edge2types[e1] = std::make_pair(t1, EdgePartType::PLUS);
        }
        {
            ogdf::edge e2 = C.G.newEdge(mid_node, C.name2node[v]);
            C._edge2types[e2] = std::make_pair(EdgePartType::PLUS, t2);
        }
    };

    // [CORRECTION] Gestion propre de la canonicalisation pour Ultrabubbles
    auto add_edge_ultrabubble = [&](const std::string& u_in,
                                    const std::string& v_in,
                                    EdgePartType t1_in,
                                    EdgePartType t2_in)
    {
        ogdf::node u = C.name2node[u_in];
        ogdf::node v = C.name2node[v_in];
        EdgePartType su = t1_in;
        EdgePartType sv = t2_in;

        // Canonicalisation basée sur l'index OGDF
        // Si on swap les noeuds, on DOIT swapper les signes associés
        if (u->index() > v->index()) {
            std::swap(u, v);
            std::swap(su, sv); 
        }

        const std::string& u_name = C.node2name[u];
        const std::string& v_name = C.node2name[v];

        BiEdgeKey key{u_name, su, v_name, sv};
        if (!seen_bidir_ultrabubble.insert(key).second) {
            return; // doublon strict déjà traité
        }

        ogdf::edge e = C.G.newEdge(u, v);
        C._edge2types[e] = std::make_pair(su, sv);
    };

    // --- Deuxième passe : Création des arêtes ---

    for (const std::string& e : raw_edges) {
        std::istringstream iss(e);
        std::string tok, from, to, ovl_str;
        char o1 = 0, o2 = 0;

        if (!(iss >> tok >> from >> o1 >> to >> o2 >> ovl_str)) {
             throw std::runtime_error("Invalid GFA L-line: " + e);
        }
        if (tok != "L") continue;

        if (C.bubbleType == Context::BubbleType::SUPERBUBBLE) {
            ensure(from + "+"); ensure(from + "-");
            ensure(to   + "+"); ensure(to   + "-");

            if (C.inputFormat == Context::InputFormat::Gfa) {
                const std::string u1 = from + std::string(1, o1);
                const std::string v1 = to   + std::string(1, o2);
                add_edge_double(u1, v1);

                const std::string u2 = to   + std::string(1, flip(o2));
                const std::string v2 = from + std::string(1, flip(o1));
                add_edge_double(u2, v2);

            } else if (C.inputFormat == Context::InputFormat::GfaDirected) {
                const std::string u = from + std::string(1, o1);
                const std::string v = to   + std::string(1, o2);
                add_edge_double(u, v);
            } else {
                throw std::logic_error("Unexpected inputFormat in readGFA() for SUPERBUBBLE");
            }

        } else if (C.bubbleType == Context::BubbleType::SNARL) {
            auto t1 = (o1 == '+' ? EdgePartType::PLUS  : EdgePartType::MINUS);
            auto t2 = (o2 == '+' ? EdgePartType::MINUS : EdgePartType::PLUS);
            add_edge_snarl(from, to, t1, t2);

        } else if (C.bubbleType == Context::BubbleType::ULTRABUBBLE) {
            auto t1 = (o1 == '+' ? EdgePartType::PLUS  : EdgePartType::MINUS);
            auto t2 = (o2 == '+' ? EdgePartType::MINUS : EdgePartType::PLUS);
            // On appelle le helper qui gère le swap correctement
            add_edge_ultrabubble(from, to, t1, t2);
        }
    }
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
                // EOF
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
        switch (C.inputFormat) {
            case Context::InputFormat::Gfa:
            case Context::InputFormat::GfaDirected:
                readGFA();
                break;

            case Context::InputFormat::Graph:
                if (C.bubbleType == Context::BubbleType::SNARL) {
                    throw std::runtime_error(
                        "Standard .graph input is not supported for snarls, use GFA input");
                }
                readStandard();
                break;

            case Context::InputFormat::Auto:
                throw std::runtime_error(
                    "InputFormat::Auto should have been resolved in readArgs()");
        }
    } catch (...) {
        if (usingTempFile) {
            C.graphPath = originalPath;
            std::remove(tempPath.c_str());
        }
        throw;
    }

    if (usingTempFile) {
        C.graphPath = originalPath;
        std::remove(tempPath.c_str());
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
    auto& sb    = ctx().superbubbles; // vector<pair<NodeId, NodeId>> on the doubled graph
    auto& names = ctx().node2name;    // NodeId -> "SEGID+" or "SEGID-"

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


void writeSuperbubbles() {
    auto &C = ctx();

    // ------------------------------------------------------------------
    // 1. Cas SNARL : sortie déjà définie en termes de C.snarls (incidences)
    // ------------------------------------------------------------------
    if (C.bubbleType == Context::BubbleType::SNARL) {
        if (C.outputPath.empty()) {
            std::cout << C.snarls.size() << "\n";
            for (auto &s : C.snarls) {
                for (auto &v : s) {
                    std::cout << v << " ";
                }
                std::cout << std::endl;
            }
            if (!std::cout) {
                throw std::runtime_error("Error while writing snarls to standard output");
            }
        } else {
            std::ofstream out(C.outputPath);
            if (!out) {
                throw std::runtime_error("Failed to open output file '" +
                                         C.outputPath + "' for writing");
            }
            out << C.snarls.size() << "\n";
            for (auto &s : C.snarls) {
                for (auto &v : s) {
                    out << v << " ";
                }
                out << "\n";
            }
            if (!out) {
                throw std::runtime_error("Error while writing snarls to output file '" +
                                         C.outputPath + "'");
            }
        }
        return;
    }

    // ------------------------------------------------------------------
    // 2. Cas ULTRABUBBLE : utiliser C.ultrabubbleIncidences
    // ------------------------------------------------------------------
    if (C.bubbleType == Context::BubbleType::ULTRABUBBLE) {
        auto pair_hash = [](const std::pair<std::string,std::string> &p) -> std::size_t {
            std::size_t h1 = std::hash<std::string>{}(p.first);
            std::size_t h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
        };
        std::unordered_set<std::pair<std::string,std::string>, decltype(pair_hash)>
            seen(0, pair_hash);

        std::vector<std::pair<std::string,std::string>> res;
        res.reserve(C.ultrabubbleIncidences.size());

        for (auto &p : C.ultrabubbleIncidences) {
            std::string a = p.first;
            std::string b = p.second;
            if (b < a) std::swap(a, b);
            std::pair<std::string,std::string> key{a,b};
            if (seen.insert(key).second) {
                res.emplace_back(std::move(key));
            }
        }

        if (C.outputPath.empty()) {
            std::cout << res.size() << "\n";
            for (auto &p : res) {
                std::cout << p.first << " " << p.second << "\n";
            }
            if (!std::cout) {
                throw std::runtime_error("Error while writing ultrabubbles to standard output");
            }
        } else {
            std::ofstream out(C.outputPath);
            if (!out) {
                throw std::runtime_error("Failed to open output file '" +
                                         C.outputPath + "' for writing");
            }
            out << res.size() << "\n";
            for (auto &p : res) {
                out << p.first << " " << p.second << "\n";
            }
            if (!out) {
                throw std::runtime_error("Error while writing ultrabubbles to output file '" +
                                         C.outputPath + "'");
            }
        }
        return;
    }

    // ------------------------------------------------------------------
    // 3. Cas SUPERBUBBLE (comportement existant)
    // ------------------------------------------------------------------

    std::vector<std::pair<std::string, std::string>> res;

    if (C.inputFormat == Context::InputFormat::Gfa &&
        !C.directedSuperbubbles) {

        auto has_orient = [](const std::string& s) {
            return !s.empty() && (s.back()=='+' || s.back()=='-');
        };
        auto flip_char = [](char c){ return c=='+'?'-': (c=='-')?'+': c; };
        auto invert = [&](std::string s){
            if (has_orient(s)) s.back() = flip_char(s.back());
            return s;
        };
        auto strip = [&](std::string s){
            if (has_orient(s)) s.pop_back();
            return s;
        };

        auto canonical_mirror_rep = [&](const std::string& x, const std::string& y){
            std::string xA = x, yA = y;
            std::string xB = invert(y), yB = invert(x);
            if (std::tie(xB, yB) < std::tie(xA, yA))
                return std::pair<std::string,std::string>{xB, yB};
            return std::pair<std::string,std::string>{xA, yA};
        };

        auto transform_and_unorder = [&](const std::pair<std::string,std::string>& p){
            std::string a = invert(p.first);
            std::string b = p.second;
            if (b < a) std::swap(a, b);
            return std::pair<std::string,std::string>{std::move(a), std::move(b)};
        };

        auto pair_hash2 = [](const std::pair<std::string,std::string>& pr)->std::size_t {
            std::size_t h1 = std::hash<std::string>{}(pr.first);
            std::size_t h2 = std::hash<std::string>{}(pr.second);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
        };
        std::unordered_set<std::pair<std::string,std::string>, decltype(pair_hash2)>
            seen2(0, pair_hash2);

        for (auto& w : C.superbubbles) {
            const std::string s = C.node2name[w.first];
            const std::string t = C.node2name[w.second];

            auto rep = canonical_mirror_rep(s, t);
            auto fin = transform_and_unorder(rep);

            fin.first  = strip(fin.first);
            fin.second = strip(fin.second);

            if (fin.first != fin.second) {
                if (seen2.insert(fin).second) {
                    res.emplace_back(std::move(fin));
                }
            }
        }
    } else {
        for (auto &w : C.superbubbles) {
            res.push_back({C.node2name[w.first], C.node2name[w.second]});
        }
    }

    if (C.outputPath.empty()) {
        std::cout << res.size() << "\n";
        for (auto &p : res) {
            std::cout << p.first << " " << p.second << "\n";
        }
        if (!std::cout) {
            throw std::runtime_error("Error while writing superbubbles to standard output");
        }
    } else {
        std::ofstream out(C.outputPath);
        if (!out) {
            throw std::runtime_error("Failed to open output file '" +
                                     C.outputPath + "' for writing");
        }
        out << res.size() << "\n";
        for (auto &p : res) {
            out << p.first << " " << p.second << "\n";
        }
        if (!out) {
            throw std::runtime_error("Error while writing superbubbles to output file '" +
                                     C.outputPath + "'");
        }
    }
}

}