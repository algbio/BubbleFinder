// ─── src/io/graph_io.cpp ────────────────────────────────────────────────────
#include "graph_io.hpp"          // declarations
#include "util/context.hpp"         // ctx()
#include "util/timer.hpp"
#include <ogdf/energybased/FMMMLayout.h>   // drawGraph needs it
#include <ogdf/fileformats/GraphIO.h>
#include <fstream>
#include <regex>


#include "util/logger.hpp"

using namespace ogdf;               // exactly as in your old main.cpp

namespace GraphIO {

// ---------- readStandard ---------------------------------------------------
void readStandard()
{
    auto &C = ctx();

    int n, m;
    if (!C.graphPath.empty()) {
        std::ifstream in(C.graphPath);
        if (!in) throw std::runtime_error("Cannot open " + C.graphPath);

        if (!(in >> n >> m))
            throw std::runtime_error("expected: n m, then m lines u v");

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

// ---------- readGFA --------------------------------------------------------
void readGFA()
{
    auto &C = ctx();
    if (C.graphPath.empty())
        throw std::runtime_error("GFA input needs -g <file>");

    std::ifstream in(C.graphPath);
    if (!in) throw std::runtime_error("Cannot open " + C.graphPath);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token; iss >> token;

        if (token == "S") {
            std::string id, seq; iss >> id >> seq;
            if (!C.name2node.count(id))
                C.name2node[id] = C.G.newNode();
        }
        else if (token == "L") {
            std::string from, o1, to, o2, ovl;
            iss >> from >> o1 >> to >> o2 >> ovl;
            if (o1 != o2) continue;
            if (o1 == "-") std::swap(from, to);

            if (!C.name2node.count(from)) {
                C.name2node[from] = C.G.newNode();
                C.node2name[C.name2node[from]] = from;
            }
            if (!C.name2node.count(to)) {
                C.name2node[to]  = C.G.newNode();
                C.node2name[C.name2node[to]]  = to;
            }
            C.G.newEdge(C.name2node[from], C.name2node[to]);
        }
    }
}

// ---------- readGraph (wrapper) -------------------------------------------
void readGraph()
{
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

// ---------- drawGraph ------------------------------------------------------
void drawGraph(Graph &G, const std::string &file)
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
    for (edge e : G.edges)
        GA.label(e) = C.node2name.count(e->source())
                    ? C.node2name[e->source()]
                    : std::to_string(e->index());

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

    /*  … rest of your bending/bundling and SVG-writing
        code copied unchanged … */
}


void writeSuperbubbles() {
    if(ctx().outputPath == "") {
        std::cout << ctx().superbubbles.size() << std::endl;
        for(auto &p:ctx().superbubbles) {
            std::cout << ctx().node2name[p.first] << " " << ctx().node2name[p.second] << std::endl;
        }

    } else {
        std::ofstream out(ctx().outputPath);
        out << ctx().superbubbles.size() << std::endl;
        for(auto &p:ctx().superbubbles) {
            out << ctx().node2name[p.first] << " " << ctx().node2name[p.second] << std::endl;
        }
    }
}

} // namespace GraphIO
