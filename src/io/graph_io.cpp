#include "graph_io.hpp"
#include "util/context.hpp"
#include "util/timer.hpp"
#include <ogdf/energybased/FMMMLayout.h> 
#include <ogdf/fileformats/GraphIO.h>
#include <fstream>
#include <regex>
#include <unordered_set>


#include "util/logger.hpp"

using namespace ogdf;

namespace GraphIO {
void readStandard()
{
    auto &C = ctx();

    int n, m;
    if (!C.graphPath.empty()) {
        std::ifstream in(C.graphPath);

        if (!in) throw std::runtime_error("Cannot open " + C.graphPath);

        if (!(in >> n >> m))
            throw std::runtime_error("expected: n m, then m lines u v");

        C.node2name.reserve(n);

        for (int i = 0; i < m; ++i) {
            std::string u, v;
            if (!(in >> u >> v)) throw std::runtime_error("edge line missing");

            if (!C.name2node.count(u)) {
                C.name2node[u] = C.G.newNode();
                C.node2name[C.name2node[u]] = u;
            }
            if (!C.name2node.count(v)) {
                C.name2node[v] = C.G.newNode();
                C.node2name[C.name2node[v]] = v;
            }
            C.G.newEdge(C.name2node[u], C.name2node[v]);
        }
    } else {
        if (!(std::cin >> n >> m))
            throw std::runtime_error("expected: n m, then m lines u v");

        C.node2name.reserve(n);

        for (int i = 0; i < m; ++i) {
            std::string u, v;
            if (!(std::cin >> u >> v)) throw std::runtime_error("edge line");

            if (!C.name2node.count(u)) {
                C.name2node[u] = C.G.newNode();
                C.node2name[C.name2node[u]] = u;
            }
            if (!C.name2node.count(v)) {
                C.name2node[v] = C.G.newNode();
                C.node2name[C.name2node[v]] = v;
            }
            C.G.newEdge(C.name2node[u], C.name2node[v]);
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

    // ---- pass 1: collect S, create oriented nodes, stash L lines ----
    std::unordered_set<std::string> have_segment;   // IDs that had an S line
    std::vector<std::string> raw_edges;             // store L lines for pass 2
    raw_edges.reserve(1 << 16);

    auto ensure = [&](const std::string& name){
        if (!C.name2node.count(name)) {
            auto id = C.G.newNode();
            C.name2node[name] = id;
            C.node2name[id] = name;
        }
    };

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        if (line[0] == 'S') {
            // GFA1: S <name> <sequence> [tags...]
            // We only need <name>. Keep parsing simple and robust.
            std::istringstream iss(line);
            std::string tok, id, seq;
            iss >> tok >> id >> seq;
            if (id.empty()) continue;                 // malformed
            have_segment.insert(id);
            // create both orientations now (like "one node" in Python but expanded here)
            ensure(id + "+");
            ensure(id + "-");
            continue;
        }
        if (line[0] == 'L') {
            raw_edges.push_back(line);
            continue;
        }
        // ignore other records (H, P, W, etc.)
    }

    in.close();

    // ---- pass 2: process L lines exactly like the Python logic ----

    auto flip = [](char c){ return c == '+' ? '-' : '+'; };

    // dedup per oriented endpoints AND overlap (mirrors Python's set-of-triples uniqueness)
    struct EdgeKey {
        std::string u, v;
        int ovl;
        bool operator==(const EdgeKey& o) const { return ovl==o.ovl && u==o.u && v==o.v; }
    };
    struct EdgeKeyHash {
        std::size_t operator()(EdgeKey const& k) const {
            std::size_t h1 = std::hash<std::string>{}(k.u);
            std::size_t h2 = std::hash<std::string>{}(k.v);
            std::size_t h3 = std::hash<int>{}(k.ovl);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2)) ^ (h3 + (h2<<1));
        }
    };
    std::unordered_set<EdgeKey, EdgeKeyHash> seen;

    auto parse_overlap_like_python = [](const std::string& ovl)->int {
        if (ovl == "*" || ovl.empty()) return 0;
        if (!ovl.empty() && ovl.back()=='M') {
            try { return std::stoi(ovl.substr(0, ovl.size()-1)); } catch (...) { return 0; }
        }
        try { return std::stoi(ovl); } catch (...) { return 0; }
    };

    auto add_edge = [&](const std::string& u, const std::string& v, int ovl){
        EdgeKey key{u, v, ovl};
        if (seen.insert(key).second) {
            C.G.newEdge(C.name2node[u], C.name2node[v]);
        }
    };

    for (const std::string& e : raw_edges) {
        // Python’s second pass uses .split() (whitespace). Do the same.
        std::istringstream iss(e);
        std::string tok, from, to, ovl_str;
        char o1 = 0, o2 = 0;

        // Expect: L <from> <o1> <to> <o2> <overlap> ...
        if (!(iss >> tok >> from >> o1 >> to >> o2 >> ovl_str)) continue;
        if (tok != "L") continue;

        // --- match Python: skip if either endpoint lacks an S record
        if (!have_segment.count(from) || !have_segment.count(to)) {
            // (Python logs a warning; keep quiet or print if you want)
            // std::cerr << "Warning: edge " << from << " - " << to << " but missing S; skipping\n";
            continue;
        }

        // ensure oriented endpoints exist (they should, from pass 1; keep defensive)
        ensure(from + "+"); ensure(from + "-");
        ensure(to   + "+"); ensure(to   + "-");

        int overlap = parse_overlap_like_python(ovl_str);

        // primary and mirror arcs (skew-symmetric), same as your original approach
        if (o1=='+' || o1=='-') {
            if (o2=='+' || o2=='-') {
                const std::string u1 = from + std::string(1, o1);
                const std::string v1 = to   + std::string(1, o2);
                add_edge(u1, v1, overlap);

                const std::string u2 = to   + std::string(1, flip(o2));
                const std::string v2 = from + std::string(1, flip(o1));
                add_edge(u2, v2, overlap);
            }
        }
    }
}


// void readGFA()
// {
//     auto &C = ctx();
//     if (C.graphPath.empty())
//         throw std::runtime_error("GFA input needs -g <file>");

//     std::ifstream in(C.graphPath);
//     if (!in) throw std::runtime_error("Cannot open " + C.graphPath);


    
//     std::string line;
//     while (std::getline(in, line)) {
//         if (line.empty() || line[0] == '#') continue;

//         std::istringstream iss(line);
//         std::string token; iss >> token;

//         if (token == "S") {
//             std::string id, seq; iss >> id >> seq;
//             // if (!C.name2node.count(id))
//             //     C.name2node[id] = C.G.newNode();
//         }
//         else if (token == "L") {
//                 std::string from, to, ovl;
//     char o1 = 0, o2 = 0;                 // parse orientation as single chars
//     iss >> from >> o1 >> to >> o2 >> ovl;

//     std::cout << from << " " << to << " " << ovl << std::endl;

//     auto flip = [](char c){ return c == '+' ? '-' : '+'; };

//     // ensure oriented nodes exist
//     auto ensure = [&](const std::string& name){
//         if (!C.name2node.count(name)) {
//             auto id = C.G.newNode();
//             C.name2node[name] = id;
//             C.node2name[id] = name;
//         }
//     };
//     ensure(from + "+"); ensure(from + "-");
//     ensure(to   + "+"); ensure(to   + "-");

//     // --- Dedup edges by oriented *names* (string pairs) ---
//     auto pair_hash = [](const std::pair<std::string,std::string>& p)->std::size_t {
//         std::size_t h1 = std::hash<std::string>{}(p.first);
//         std::size_t h2 = std::hash<std::string>{}(p.second);
//         // hash combine
//         return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
//     };
//     static std::unordered_set<std::pair<std::string,std::string>, decltype(pair_hash)>
//         seen_edges(0, pair_hash);

//     auto add_edge = [&](const std::string& u_name, const std::string& v_name){
//         auto key = std::make_pair(u_name, v_name);
//         if (seen_edges.insert(key).second) {
//             C.G.newEdge(C.name2node[u_name], C.name2node[v_name]);
//         }
//     };

//     // add primary and mirror (skew-symmetric) arcs, but only once each
//     add_edge(from + std::string(1, o1), to + std::string(1, o2));
//     add_edge(to   + std::string(1, flip(o2)),
//              from + std::string(1, flip(o1)));


//             // std::string from, o1, to, o2, ovl;
//             // iss >> from >> o1 >> to >> o2 >> ovl;
//             // // if (o1 != o2) continue;

//             // auto flip = [&](string c) {
//             //     return (c == "+" ? "-" : "+");
//             // };

//             // // bool from_start = (o1 == "-"), to_end = (o2 == "-"); 




//             // // if (o1 == "-") std::swap(from, to);


//             // // if(from_start && to_end) {
                
//             // // }

//             // if (!C.name2node.count(from + "-")) {
//             //     string name = from + "-";
//             //     C.name2node[name] = C.G.newNode();
//             //     C.node2name[C.name2node[name]] = name;
//             // }

//             // if (!C.name2node.count(from + "+")) {
//             //     string name = from + "+";
//             //     C.name2node[name] = C.G.newNode();
//             //     C.node2name[C.name2node[name]] = name;
//             // }

//             // if (!C.name2node.count(to + "-")) {
//             //     string name = to + "-";
//             //     C.name2node[name] = C.G.newNode();
//             //     C.node2name[C.name2node[name]] = name;
//             // }

//             // if (!C.name2node.count(to + "+")) {
//             //     string name = to + "+";
//             //     C.name2node[name] = C.G.newNode();
//             //     C.node2name[C.name2node[name]] = name;
//             // }



//             // // if (!C.name2node.count(to)) {
//             // //     C.name2node[to]  = C.G.newNode();
//             // //     C.node2name[C.name2node[to]]  = to;
//             // // }

//             // // if(from_start && to_end) {

//             // // }

//             // C.G.newEdge(C.name2node[from + o1], C.name2node[to + o2]);
//             // C.G.newEdge(C.name2node[to + flip(o2)], C.name2node[from + flip(o1)]);   
//         }
//     }
// }


void readGraph() {
    auto &C = ctx();
    TIME_BLOCK("Graph read");

    logger::info("Starting to read graph");

    if (C.gfaInput) readGFA();
    else readStandard();

    C.isEntry = NodeArray<bool>(C.G, false);
    C.isExit  = NodeArray<bool>(C.G, false);
    C.inDeg   = NodeArray<int >(C.G, 0);
    C.outDeg  = NodeArray<int >(C.G, 0);

    for (edge e : C.G.edges) {
        C.outDeg(e->source())++;
        C.inDeg (e->target())++;
    }

    logger::info("Graph read");
}

void drawGraph(const Graph &G, const std::string &file)
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
    // for (edge e : G.edges)
    //     GA.label(e) = C.node2name.count(e->source())
    //                 ? C.node2name[e->source()]
    //                 : std::to_string(e->index());

    FMMMLayout().call(GA);


    constexpr double GAP = 12.0;                  // gap between parallels

    struct PairHash {
        size_t operator()(const std::pair<int,int>& p) const noexcept {
            return (static_cast<size_t>(p.first) << 32) ^ p.second;
        }
    };


    std::unordered_map<std::pair<int,int>, std::vector<edge>, PairHash> bundle;

    for(edge e : G.edges) {
        int u = e->source()->index();
        int v = e->target()->index();
        bundle[{u,v}].push_back(e);              // **ordered** pair (u,v)
    }

    for(auto &entry : bundle) {
        auto &vec = entry.second;
        if(vec.size() <= 1) continue;            // no need to fan out

        // coordinates of first edge’s endpoints are enough
        edge e0 = vec[0];
        node a  = e0->source();
        node b  = e0->target();

        double ax = GA.x(a), ay = GA.y(a);
        double bx = GA.x(b), by = GA.y(b);

        // perpendicular unit vector
        double dx = bx - ax, dy = by - ay;
        double len = std::hypot(dx, dy);
        if(len == 0) len = 1;
        double px = -dy / len, py = dx / len;

        // put bends left of (u→v) and right of (v→u) to separate directions
        const int sign = 1;                      // bundle is (u,v) already
        const int k = static_cast<int>(vec.size());

        for(int i = 0; i < k; ++i) {
            edge e = vec[i];
            double shift = (i - (k - 1) / 2.0) * GAP;

            // mid-point of the straight segment plus perpendicular offset
            double mx = (ax + bx) * 0.5 + sign * px * shift;
            double my = (ay + by) * 0.5 + sign * py * shift;

            GA.bends(e).clear();
            GA.bends(e).pushBack(DPoint(mx, my));
        }
    }

    /* 4 ─ write SVG once ------------------------------------------- */
    const std::string tmp = file + ".svg.tmp";
    ogdf::GraphIO::drawSVG(GA, tmp);

    

    /* 5 ─ copy header, extract viewBox, inject white background ----- */
    std::ifstream in(tmp);
    std::ofstream out(file + ".svg");

    std::string header;  std::getline(in, header);
    std::string openTag; std::getline(in, openTag);
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
    in.close(); out.close();
    std::remove(tmp.c_str());
}

std::vector<std::pair<std::string, std::string>> project_bubblegun_pairs_from_doubled() {
    auto& sb = ctx().superbubbles;     // vector<pair<NodeId, NodeId>> on the doubled graph
    auto& names = ctx().node2name;     // NodeId -> "SEGID+" or "SEGID-"

    auto is_oriented = [](const std::string& s) -> bool {
        return !s.empty() && (s.back() == '+' || s.back() == '-');
    };
    auto strip = [](std::string s) -> std::string {
        if (!s.empty() && (s.back() == '+' || s.back() == '-')) s.pop_back();
        return s;
    };
    auto pair_hash = [](const std::pair<std::string,std::string>& p) -> std::size_t {
        return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
    };

    std::vector<std::pair<std::string, std::string>> out;
    out.reserve(sb.size());

    // Deduplicate exact directed (src, dst) after projection
    std::unordered_set<std::pair<std::string,std::string>, decltype(pair_hash)> seen(0, pair_hash);

    // Optional: dedupe oriented inputs first (in case your finder emitted duplicates)
    using Oriented = std::pair<std::string,std::string>;
    std::unordered_set<Oriented, decltype(pair_hash)> seen_oriented(0, pair_hash);

    for (auto const& e : sb) {
        const std::string& sa = names[e.first];   // e.g. "12345+"
        const std::string& sbn = names[e.second]; // e.g. "67890-"

        // Ignore exact duplicate oriented inputs
        if (!seen_oriented.insert({sa, sbn}).second) continue;

        // Canonicalize orientation to kill the doubled mirror: keep only source '+'
        if (is_oriented(sa) && sa.back() == '-') continue;

        // (If some inputs are not marked with +/- at all, we keep them as-is.)

        std::string a = strip(sa);   // segment id (no +/-)
        std::string b = strip(sbn);  // segment id (no +/-)
        if (a == b) continue;        // BubbleGun won’t report degenerate self-bubbles

        if (seen.insert({a, b}).second) {
            out.emplace_back(std::move(a), std::move(b));
        }
    }

    return out;
}


// void writeSuperbubbles() {
//     std::vector<std::pair<std::string, std::string>> res;

//     if (ctx().gfaInput) {
//         // res = project_bubblegun_pairs_from_doubled();

//         auto has_orient = [](const std::string& s) {
//             return !s.empty() && (s.back() == '+' || s.back() == '-');
//         };
//         auto strip = [](std::string s) {
//             if (!s.empty()) {
//                 char c = s.back();
//                 if (c == '+' || c == '-') s.pop_back();
//             }
//             return s;
//         };
//         auto invert = [](std::string s) {
//             if (!s.empty()) {
//                 char& c = s.back();
//                 if (c == '+') c = '-';
//                 else if (c == '-') c = '+';
//             }
//             return s;
//         };

//         for (auto& w : ctx().superbubbles) {
//             const std::string sa0 = ctx().node2name[w.first];   // oriented, e.g. "12345+"
//             const std::string sb0 = ctx().node2name[w.second];  // oriented, e.g. "67890-"

//             // Mirror suppression: keep exactly one representative.
//             // If source ends with '-', flip to its mirror (!sb0, !sa0) using ORIGINALS (temporaries!).
//             std::string sa = sa0, sb = sb0;
//             if (has_orient(sa0) && sa0.back() == '-') {
//                 const std::string sa_m = invert(sb0);  // !sb0
//                 const std::string sb_m = invert(sa0);  // !sa0
//                 sa = sa_m;
//                 sb = sb_m;
//             }

//             // Project to segment IDs (drop +/-).
//             std::string a = strip(sa);
//             std::string b = strip(sb);

//             if (a == b) continue; // safety: BG won't report self-bubbles

//             // If your output is UNORDERED, keep one canonical ordering:
//             if (b < a) std::swap(a, b);

//             // IMPORTANT: do NOT dedupe on (a,b); preserve multiplicity.
//             res.emplace_back(std::move(a), std::move(b));
//         }



//         // auto strip = [](std::string s) {
//         //     if (!s.empty() && (s.back() == '+' || s.back() == '-')) s.pop_back();
//         //     return s;
//         // };

//         // auto pair_hash = [](const std::pair<std::string,std::string>& p) -> std::size_t {
//         //     return std::hash<std::string>{}(p.first) ^ (std::hash<std::string>{}(p.second) << 1);
//         // };

//         // std::unordered_set<std::pair<std::string,std::string>, decltype(pair_hash)> seen(0, pair_hash);

//         // for (auto &w : ctx().superbubbles) {
//         //     std::string a = strip(ctx().node2name[w.first]);
//         //     std::string b = strip(ctx().node2name[w.second]);

//         //     // Canonicalize to unordered pair
//         //     if (b < a) std::swap(a, b);

//         //     if (seen.insert({a, b}).second) {
//         //         res.emplace_back(a, b);
//         //     }
//         // }
//     } else {
//         for(auto &w:ctx().superbubbles) {
//             res.push_back({ctx().node2name[w.first], ctx().node2name[w.second]});
//         }
//     }


//     // std::cout << ctx().superbubbles.size() << "\n";
//     // for(auto &p:ctx().superbubbles) {
//     //     std::cout << ctx().node2name[p.first] << " " << ctx().node2name[p.second] << "\n";
//     // }



//     if(ctx().outputPath == "") {
//         std::cout << res.size() << "\n";
//         for(auto &p:res) {
//             std::cout << p.first << " " << p.second << "\n";
//         }

//         // std::cout << ctx().superbubbles.size() << "\n";
//         // for(auto &p:ctx().superbubbles) {
//         //     std::cout << ctx().node2name[p.first] << " " << ctx().node2name[p.second] << "\n";
//         // }



//     } else {
//         std::ofstream out(ctx().outputPath);
//         out << res.size() << "\n";
//         for(auto &p:res) {
//             out << p.first << " " << p.second << "\n";
//         }
//         // std::ofstream out(ctx().outputPath);
//         // out << ctx().superbubbles.size() << "\n";
//         // for(auto &p:ctx().superbubbles) {
//         //     out << ctx().node2name[p.first] << " " << ctx().node2name[p.second] << "\n";
//         // }
//     }
// }

void writeSuperbubbles() {
    std::vector<std::pair<std::string, std::string>> res;

    if (ctx().gfaInput) {
        // auto has_orient = [](const std::string& s) {
        //     return !s.empty() && (s.back() == '+' || s.back() == '-');
        // };
        // auto flip_char = [](char c) {
        //     return c == '+' ? '-' : c == '-' ? '+' : c;
        // };
        // auto invert = [&](std::string s) {
        //     if (has_orient(s)) s.back() = flip_char(s.back());
        //     return s;
        // };
        // auto strip = [&](std::string s) {
        //     if (has_orient(s)) s.pop_back();
        //     return s;
        // };

        // auto pair_hash = [](const std::pair<std::string,std::string>& p) -> std::size_t {
        //     std::size_t h1 = std::hash<std::string>{}(p.first);
        //     std::size_t h2 = std::hash<std::string>{}(p.second);
        //     return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
        // };
        // std::unordered_set<std::pair<std::string,std::string>, decltype(pair_hash)> seen(0, pair_hash);

        // for (auto& w : ctx().superbubbles) {
        //     std::string sa = ctx().node2name[w.first];   // oriented, e.g. "12345+"
        //     std::string sb = ctx().node2name[w.second];  // oriented, e.g. "67890-"

        //     // Mirror normalize so the first endpoint is '+'
        //     if (has_orient(sa) && sa.back() == '-') {
        //         std::string sa_m = invert(sb);
        //         std::string sb_m = invert(sa);
        //         sa = sa_m;
        //         sb = sb_m;
        //     }

        //     // Strip orientation
        //     std::string a = strip(sa);
        //     std::string b = strip(sb);

        //     if (a == b) continue; // skip self-pairs

        //     // Canonicalize unordered pair
        //     if (b < a) std::swap(a, b);

        //     // Deduplicate
        //     if (seen.insert({a, b}).second) {
        //         res.emplace_back(std::move(a), std::move(b));
        //     }
        // }

                // --- helpers on oriented names like "12345+" / "67890-"
        auto has_orient = [](const std::string& s) {
            return !s.empty() && (s.back()=='+' || s.back()=='-');
        };
        auto flip_char = [](char c){ return c=='+'?'-': (c=='-')?'+': c; };
        auto invert = [&](std::string s){ if (has_orient(s)) s.back() = flip_char(s.back()); return s; };
        auto strip = [&](std::string s){ if (has_orient(s)) s.pop_back(); return s; };

        // Canonical representative under (x, y) ~ (invert(y), invert(x))
        auto canonical_mirror_rep = [&](const std::string& x, const std::string& y){
            std::string xA = x, yA = y;
            std::string xB = invert(y), yB = invert(x);
            if (std::tie(xB, yB) < std::tie(xA, yA))
                return std::pair<std::string,std::string>{xB, yB};
            return std::pair<std::string,std::string>{xA, yA};
        };

        // After mirror-canonicalization, transform by inverting the first
        // and then make the pair unordered (sort the two strings)
        auto transform_and_unorder = [&](const std::pair<std::string,std::string>& p){
            std::string a = invert(p.first);   // invert sign of first
            std::string b = p.second;          // keep second as-is
            if (b < a) std::swap(a, b);        // unordered set -> canonical order
            return std::pair<std::string,std::string>{std::move(a), std::move(b)};
        };

        // hash for unordered pair<string,string>
        auto pair_hash = [](const std::pair<std::string,std::string>& pr)->std::size_t {
            std::size_t h1 = std::hash<std::string>{}(pr.first);
            std::size_t h2 = std::hash<std::string>{}(pr.second);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
        };
        std::unordered_set<std::pair<std::string,std::string>, decltype(pair_hash)> seen(0, pair_hash);

        for (auto& w : ctx().superbubbles) {
            const std::string s = ctx().node2name[w.first];   // e.g. "a+"
            const std::string t = ctx().node2name[w.second];  // e.g. "b+"

            // 1) mirror-canonicalize
            auto rep = canonical_mirror_rep(s, t);

            // 2) invert first, unorder
            auto fin = transform_and_unorder(rep);

            // 3) strip orientations for final bidirected output
            fin.first  = strip(fin.first);
            fin.second = strip(fin.second);

            // 4) dedupe
            if (fin.first != fin.second) {
                if (seen.insert(fin).second) {
                    res.emplace_back(std::move(fin));
                }
            }
        }


    } else {
        for (auto &w : ctx().superbubbles) {
            res.push_back({ctx().node2name[w.first], ctx().node2name[w.second]});
        }
    }

    for(auto &p:ctx().superbubbles) {
        std::cout << ctx().node2name[p.first] << " " << ctx().node2name[p.second] << "\n";
    }


    if (ctx().outputPath.empty()) {
        std::cout << res.size() << "\n";
        for (auto &p : res) {
            std::cout << p.first << " " << p.second << "\n";
        }
    } else {
        std::ofstream out(ctx().outputPath);

        // for(auto &p:ctx().superbubbles) {
        //     out << ctx().node2name[p.first] << " " << ctx().node2name[p.second] << "\n";
        // }

        out << res.size() << "\n";
        for (auto &p : res) {
            out << p.first << " " << p.second << "\n";
        }
    }
}


}
