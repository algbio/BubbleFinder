#include <bits/stdc++.h>
using namespace std;

/*
 * - Nodes: strings "id+" and "id-" (id = GFA segment identifier).
 * - For each link L from o1 to o2 overlap:
 *     u1 = from + o1,  v1 = to   + o2      => arc u1 -> v1
 *     u2 = to   + flip(o2), v2 = from + flip(o1) => arc u2 -> v2
 *
 *
 * Superbubbloid (s,t) if:
 *   (1) t reachable from s without using t as an internal vertex,
 *   (2) R1 = { v | s -> v without t internal } == R2 = { v | v -> t without s internal },
 *   (3) induced subgraph B_st = G[R1] is acyclic,
 *   (4) no edge t -> s.
 *
 * Superbubble (s,t) if superbubbloid AND minimal:
 *   no u in B_st \ {s,t} makes (s,u) a superbubbloid.
 *
 */

// ====================== Graph representation ======================

static int N = 0; // number of directed vertices

// adj[u]  : outgoing arcs u -> v
// radj[v] : arcs u -> v viewed reversed (v -> u)
static vector<vector<int>> adj;
static vector<vector<int>> radj;

// mapping name "id+" / "id-" -> index 0..N-1
static unordered_map<string,int> name2id;
static vector<string> id2name;     // id2name[v] = "id+" or "id-"

// List of existing vertices (0..N-1)
static vector<int> orientedNodes;

// ========================= Graph tools / build ======================

// Returns the integer index for a directed vertex name, creating it if needed
int ensureNode(const string &name) {
    auto it = name2id.find(name);
    if (it != name2id.end()) return it->second;
    int id = (int)id2name.size();
    name2id[name] = id;
    id2name.push_back(name);
    return id;
}

char flip_char(char c) {
    return (c == '+') ? '-' : (c == '-') ? '+' : c;
}

// Read GFA and build the directed graph
void buildGraphFromGFA(const string &path) {
    ifstream in(path);
    if (!in) {
        cerr << "Error: cannot open " << path << "\n";
        exit(1);
    }

    unordered_set<string> have_segment;
    vector<string> raw_edges;
    raw_edges.reserve(1 << 16);

    string line;

    // First pass: read S and L lines, collect segments and store L lines
    while (getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        if (line[0] == 'S') {
            istringstream iss(line);
            string tok, id, seq;
            iss >> tok >> id >> seq;
            if (id.empty()) continue;
            have_segment.insert(id);
            // Create both orientations id+, id-
            ensureNode(id + "+");
            ensureNode(id + "-");
            continue;
        }
        if (line[0] == 'L') {
            raw_edges.push_back(line);
            continue;
        }
    }
    in.close();

    // Graph initialization (final size N known after potential ensures)
    // Some segments may appear in L without S: create them while parsing for robustness.
    // Second pass: parse L lines and add arcs
    struct PairHash {
        size_t operator()(const pair<int,int> &p) const noexcept {
            return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
        }
    };
    unordered_set<pair<int,int>, PairHash> seenArcs;

    for (const string &e : raw_edges) {
        istringstream iss(e);
        string tok, from, to, ovl_str;
        char o1 = 0, o2 = 0;
        if (!(iss >> tok >> from >> o1 >> to >> o2 >> ovl_str)) continue;
        if (tok != "L") continue;

        ensureNode(from + "+");
        ensureNode(from + "-");
        ensureNode(to   + "+");
        ensureNode(to   + "-");
    }

    N = (int)id2name.size();
    adj.assign(N, {});
    radj.assign(N, {});
    orientedNodes.clear();
    orientedNodes.reserve(N);
    for (int v = 0; v < N; ++v) orientedNodes.push_back(v);

    auto add_arc = [&](int u, int v) {
        pair<int,int> key{u,v};
        if (seenArcs.insert(key).second) {
            adj[u].push_back(v);
            radj[v].push_back(u);
        }
    };

    for (const string &e : raw_edges) {
        istringstream iss(e);
        string tok, from, to, ovl_str;
        char o1 = 0, o2 = 0;
        if (!(iss >> tok >> from >> o1 >> to >> o2 >> ovl_str)) continue;
        if (tok != "L") continue;

        if ((o1 == '+' || o1 == '-') && (o2 == '+' || o2 == '-')) {
            string u1_name = from + string(1, o1);
            string v1_name = to   + string(1, o2);
            string u2_name = to   + string(1, flip_char(o2));
            string v2_name = from + string(1, flip_char(o1));

            int u1 = ensureNode(u1_name);
            int v1 = ensureNode(v1_name);
            int u2 = ensureNode(u2_name);
            int v2 = ensureNode(v2_name);

            add_arc(u1, v1);
            add_arc(u2, v2);
        }
    }

    N = (int)id2name.size();
    if ((int)adj.size() < N) {
        adj.resize(N);
        radj.resize(N);
        for (int v = adj.size(); v < N; ++v) {
            orientedNodes.push_back(v);
        }
    }

    cerr << "Directed graph built.\n";
    cerr << "Number of directed vertices present: " << orientedNodes.size() << "\n";
}

// ===================== Superbubble algorithm (Onodera) =================

static vector<char> reachS;   // reachable from s (without t as internal)
static vector<char> reachT;   // reaches t   (without s as internal)
static vector<char> inBubble; // belongs to B_st
static vector<int>  color;    // 0 = unvisited, 1 = on stack, 2 = finished (cycle check)

bool dfs_cycle(int u) {
    color[u] = 1;
    for (int v : adj[u]) {
        if (!inBubble[v]) continue;
        if (color[v] == 0) {
            if (dfs_cycle(v)) return true;
        } else if (color[v] == 1) {
            return true;
        }
    }
    color[u] = 2;
    return false;
}

/*
 * Test if (s,t) is a superbubbloid (without minimality) in the sense of Onodera/Gärtner:
 *   (1) t reachable from s without using t as internal vertex,
 *   (2) sets R1/R2 coincide,
 *   (3) B_st is acyclic,
 *   (4) no direct edge t->s.
 * If yes, bubbleVerts receives the vertices of B_st.
 */
bool is_superbubbloid_no_min(int s, int t, vector<int> &bubbleVerts) {
    if (s == t) return false;

    // Forward BFS from s, without using t as an internal vertex
    fill(reachS.begin(), reachS.end(), 0);
    queue<int> q;
    q.push(s);
    reachS[s] = 1;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            // do not explore from t (avoid using t as internal)
            if (u == t) continue;
            if (!reachS[v]) {
                reachS[v] = 1;
                q.push(v);
            }
        }
    }

    if (!reachS[t]) return false; // reachability

    // Reverse BFS from t, without using s as an internal vertex
    fill(reachT.begin(), reachT.end(), 0);
    q.push(t);
    reachT[t] = 1;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : radj[u]) {
            // do not explore from s in the reversed graph
            if (u == s) continue;
            if (!reachT[v]) {
                reachT[v] = 1;
                q.push(v);
            }
        }
    }

    // Matching: R1 == R2 on all vertices
    for (int v : orientedNodes) {
        if (reachS[v] != reachT[v]) {
            return false;
        }
    }

    // B_st = { v | reachS[v] == 1 }
    bubbleVerts.clear();
    fill(inBubble.begin(), inBubble.end(), 0);
    for (int v : orientedNodes) {
        if (reachS[v]) {
            inBubble[v] = 1;
            bubbleVerts.push_back(v);
        }
    }

    // Acyclicity of G[B_st]
    fill(color.begin(), color.end(), 0);
    for (int v : bubbleVerts) {
        if (color[v] == 0) {
            if (dfs_cycle(v)) {
                return false;
            }
        }
    }

    // No direct edge t->s
    for (int v : adj[t]) {
        if (v == s) return false;
    }

    return true;
}

/*
 * Full test of superbubble (s,t):
 *   - (s,t) is a superbubbloid as above,
 *   - minimality: for every u in B_st \ {s,t}, (s,u) is NOT a superbubbloid.
 */
bool is_superbubble(int s, int t) {
    vector<int> bubble;
    if (!is_superbubbloid_no_min(s, t, bubble)) return false;

    for (int u : bubble) {
        if (u == s || u == t) continue;
        vector<int> tmp;
        if (is_superbubbloid_no_min(s, u, tmp)) {
            // There exists a smaller superbubbloid (s,u) included in B_st
            return false;
        }
    }

    return true;
}

// =================== Canonicalization and output =======================

struct PairHash {
    size_t operator()(const pair<string,string>& p) const noexcept {
        size_t h1 = std::hash<string>{}(p.first);
        size_t h2 = std::hash<string>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
    }
};

inline bool has_orient(const std::string &s) {
    return !s.empty() && (s.back()=='+' || s.back()=='-');
}

inline std::string flip_last(const std::string &s) {
    if (!has_orient(s)) return s;
    std::string r = s;
    r.back() = (r.back() == '+' ? '-' : (r.back() == '-' ? '+' : r.back()));
    return r;
}

inline std::string strip_orient(std::string s) {
    if (has_orient(s)) s.pop_back();
    return s;
}

// Mirror canonicalization as in writeSuperbubbles():
// (x,y) ~ (invert(y), invert(x)), keep the lexicographically smallest.
pair<string,string> canonical_mirror_rep(const string &x, const string &y) {
    string xA = x, yA = y;
    string xB = flip_last(y), yB = flip_last(x);
    if (tie(xB, yB) < tie(xA, yA))
        return {xB, yB};
    return {xA, yA};
}

// After canonical_mirror_rep, invert the first element and return an unordered pair.
pair<string,string> transform_and_unorder(const pair<string,string> &p) {
    string a = flip_last(p.first);
    string b = p.second;
    if (b < a) std::swap(a, b);
    return {std::move(a), std::move(b)};
}

// =============================== main =================================

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " input.gfa\n";
        return 1;
    }

    string path = argv[1];
    cerr << "Reading GFA graph and building directed graph (DP-style)..\n";
    buildGraphFromGFA(path);

    // Allocate workspace structures
    reachS.assign(N, 0);
    reachT.assign(N, 0);
    inBubble.assign(N, 0);
    color.assign(N, 0);

    cerr << "Brute-force enumeration of superbubbles (double orientation)\n";

    // First collect all oriented superbubbles (s,t)
    vector<pair<string,string>> orientedSuperbubbles;

    for (int s : orientedNodes) {
        for (int t : orientedNodes) {
            if (s == t) continue;
            if (is_superbubble(s, t)) {
                orientedSuperbubbles.emplace_back(
                    id2name[s], 
                    id2name[t]  
                );
            }
        }
    }

    // Now apply the same projection/canonicalization pipeline
    // as GraphIO::writeSuperbubbles() and project_bubblegun_pairs_from_doubled().

    unordered_set<pair<string,string>, PairHash> seen_final;
    vector<pair<string,string>> finalPairs;

    for (auto &w : orientedSuperbubbles) {
        const string &s = w.first;  
        const string &t = w.second; 

        // 1) mirror canonicalization
        auto rep = canonical_mirror_rep(s, t);

        // 2) invert first + unordered
        auto fin = transform_and_unorder(rep);

        // 3) strip signs
        fin.first  = strip_orient(fin.first);
        fin.second = strip_orient(fin.second);

        if (fin.first == fin.second) continue;

        // 4) deduplicate
        if (seen_final.insert(fin).second) {
            finalPairs.emplace_back(std::move(fin));
        }
    }

    cout << finalPairs.size() << "\n";
    for (auto &p : finalPairs) {
        cout << p.first << " " << p.second << "\n";
    }

    return 0;
}
