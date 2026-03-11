#include <bits/stdc++.h>
#include <cassert>
#include <cstdint>
#ifdef __unix__
#include <unistd.h>
#include <limits.h>
#include <sys/wait.h>
#endif

using namespace std;

#ifndef DEBUG_BLOCKS
#define DEBUG_BLOCKS 1
#endif

static inline int sgnIdx(char c) { return (c == '-') ? 1 : 0; }

struct IncEdge {
    int v;
    char su; // side at u
    char sv; // side at v
};

struct SnarlKey {
    int a, b;
    char da, db;
    bool operator<(SnarlKey const& o) const {
        return std::tie(a, da, b, db) < std::tie(o.a, o.da, o.b, o.db);
    }
    bool operator==(SnarlKey const& o) const {
        return a==o.a && b==o.b && da==o.da && db==o.db;
    }
};

static inline SnarlKey canon_snarl(int x, char dx, int y, char dy) {
    if (x < y) return {x,y,dx,dy};
    return {y,x,dy,dx};
}

#ifdef __unix__
static std::string get_self_dir() {
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len == -1) return {};
    buf[len] = '\0';
    std::string abs_path(buf);
    auto pos = abs_path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    return abs_path.substr(0, pos);
}

static bool parse_side_token(const std::string& t,
                            const std::unordered_map<std::string,int>& id,
                            int &v, char &s) {
    if (t.size() < 2) return false;
    char c = t.back();
    if (c != '+' && c != '-') return false;
    std::string nm = t.substr(0, t.size()-1);
    auto it = id.find(nm);
    if (it == id.end()) return false;
    v = it->second;
    s = c;
    return true;
}

static bool run_snarls_bf(const std::string& snarls_bf_path,
                          const std::string& gfa_path,
                          const std::unordered_map<std::string,int>& id,
                          std::set<SnarlKey>& out) {
    out.clear();

    if (snarls_bf_path.empty()) return false;
    if (access(snarls_bf_path.c_str(), X_OK) != 0) return false;

    std::string cmd = "'" + snarls_bf_path + "' '" + gfa_path + "'";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;

    char buf[1<<15];
    while (fgets(buf, sizeof(buf), pipe)) {
        std::string line(buf);
        std::stringstream ss(line);
        std::string t1, t2;
        ss >> t1 >> t2;
        if (!ss) continue;

        int x,y; char dx,dy;
        if (!parse_side_token(t1, id, x, dx)) continue;
        if (!parse_side_token(t2, id, y, dy)) continue;

        out.insert(canon_snarl(x, dx, y, dy));
    }

    int status = pclose(pipe);
    if (status == -1) return false;
    if (!WIFEXITED(status)) return false;
    if (WEXITSTATUS(status) != 0) return false;
    return true;
}
#endif

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc != 2) {
        cerr << "Usage: ./ultrabubbles_bf input.gfa\n";
        return 1;
    }

#ifdef __unix__
    const std::string self_dir = get_self_dir();
    const std::string snarls_bf_path = self_dir.empty()
                                        ? ""
                                        : (self_dir + "/snarls_bf");
#else
    const std::string snarls_bf_path;
#endif

    ifstream fin(argv[1]);
    if (!fin) {
        cerr << "Cannot open input.\n";
        return 1;
    }

    unordered_map<string,int> id;
    vector<string> nameOf;
    auto getId = [&](const string& s)->int{
        auto it = id.find(s);
        if (it != id.end()) return it->second;
        int nid = (int)nameOf.size();
        id[s] = nid;
        nameOf.push_back(s);
        return nid;
    };

    vector<vector<IncEdge>> adj;
    vector<uint8_t> seenSegment;

    auto ensureN = [&](int n){
        if ((int)adj.size() < n) adj.resize(n);
        if ((int)seenSegment.size() < n) seenSegment.resize(n, 0);
    };

    string line;
    while (getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        string type;
        ss >> type;
        if (!ss) continue;

        if (type == "S") {
            string seg, seq;
            ss >> seg >> seq;
            if (!ss) continue;
            int u = getId(seg);
            ensureN(u+1);
            seenSegment[u] = 1;
        } else if (type == "L") {
            string a, b, ovl;
            char oa = 0, ob = 0;
            ss >> a >> oa >> b >> ob >> ovl;
            if (!ss) continue;

            int u = getId(a);
            int v = getId(b);
            ensureN(max(u,v)+1);
            seenSegment[u] = seenSegment[v] = 1;

            // bidirected incidence convention:
            // at u we store oa, at v we store flip(ob)
            char su = oa;
            char sv = (ob == '-') ? '+' : '-';
            adj[u].push_back({v, su, sv});
            adj[v].push_back({u, sv, su});
        }
    }

    int N = (int)nameOf.size();
    ensureN(N);

    vector<int> nodes;
    nodes.reserve(N);
    for (int v = 0; v < N; v++) if (seenSegment[v]) nodes.push_back(v);

    sort(nodes.begin(), nodes.end());
    assert(is_sorted(nodes.begin(), nodes.end()));

    cerr << "Vertices: " << nodes.size() << "\n";
    cerr << "Enumerating ultrabubbles (INCLUDING weak)\n";

    const int REACH_FLAG = 1;
    const int BAD_FLAG   = 2;

    vector<int> seen(N, 0);
    int dfs_it = 1;
    vector<int> snarl_component;

    // DFS used for separability + component construction (split terminals semantics)
    function<int(int,int,int,char,char,bool)> dfs =
        [&](int z, int x, int y, char dx, char dy, bool track)->int {

        if (seen[z] == dfs_it) return 0;
        seen[z] = dfs_it;
        if (track) snarl_component.push_back(z);

        int ans = 0;
        if (z == x) {
            for (auto &inc : adj[z]) {
                if (inc.su != dx) continue;
                if (inc.v == y && inc.sv == dy) ans |= REACH_FLAG;
                if (inc.v == y && inc.sv != dy) ans |= BAD_FLAG;
                ans |= dfs(inc.v, x, y, dx, dy, track);
            }
        } else if (z == y) {
            for (auto &inc : adj[z]) {
                if (inc.su != dy) continue;
                if (inc.v == x && inc.sv != dx) ans |= BAD_FLAG;
                ans |= dfs(inc.v, x, y, dx, dy, track);
            }
        } else {
            for (auto &inc : adj[z]) {
                if (inc.v == y && inc.sv == dy) ans |= REACH_FLAG;
                if (inc.v == y && inc.sv != dy) ans |= BAD_FLAG;
                if (inc.v == x && inc.sv != dx) ans |= BAD_FLAG;
                ans |= dfs(inc.v, x, y, dx, dy, track);
            }
        }
        return ans;
    };

    auto test_separability = [&](int x, int y, char dx, char dy)->bool{
        int out = dfs(x, x, y, dx, dy, false);
        dfs_it++;
        return (out & REACH_FLAG) && !(out & BAD_FLAG);
    };

    auto compute_component = [&](int x, int y, char dx, char dy)->void{
        snarl_component.clear();
        dfs(x, x, y, dx, dy, true);
        dfs_it++;
    };

    // separability memo: N*N*4 (dx,dy in {0,1})
    vector<int8_t> sep((size_t)N * (size_t)N * 4, (int8_t)-1);

    auto sepIndex = [&](int x, int y, int dx, int dy)->size_t{
        return ((size_t)x * (size_t)N + (size_t)y) * 4 + (size_t)(dx*2 + dy);
    };

    auto getSep = [&](int x, int y, int dx, int dy)->bool{
        int8_t &cell = sep[sepIndex(x,y,dx,dy)];
        if (cell == (int8_t)-1) {
            char sx = dx ? '-' : '+';
            char sy = dy ? '-' : '+';
            cell = test_separability(x, y, sx, sy) ? (int8_t)1 : (int8_t)0;
        }
        return cell == (int8_t)1;
    };

    // ---- split-terminals filter helpers ----
    int CX=-1, CY=-1;
    char CDX='?', CDY='?';

    auto edge_survives_split = [&](int u, int v, char su, char sv)->bool{
        if (u == CX && su != CDX) return false;
        if (v == CX && sv != CDX) return false;
        if (u == CY && su != CDY) return false;
        if (v == CY && sv != CDY) return false;
        return true;
    };

    vector<uint8_t> in_comp(N, 0);

    // tip-free check in the NET GRAPH (split-filtered)
    auto has_internal_tip_in_component = [&]()->bool{
        vector<array<uint8_t,2>> hasSign(N, {0,0});

        for (int z : snarl_component) {
            for (auto &inc : adj[z]) {
                int w = inc.v;
                if (!in_comp[w]) continue;
                if (!edge_survives_split(z, w, inc.su, inc.sv)) continue;
                hasSign[z][sgnIdx(inc.su)] = 1;
            }
        }

        for (int z : snarl_component) {
            if (z == CX || z == CY) continue;
            if (!hasSign[z][0] || !hasSign[z][1]) return true;
        }
        return false;
    };


    auto has_cycloid_in_component = [&]()->bool{
        auto sideId = [&](int v, char s)->int { return 2*v + sgnIdx(s); };
        auto oppSide = [&](int side)->int { return side ^ 1; }; // flip + <-> -

        const int S = 2 * N;     // number of sides
        const int ST = 2 * S;    // (side, used_exception)
        auto stId = [&](int side, int used)->int { return (side<<1) | used; };

        // collect UNIQUE bidirected edges of X after splitting terminals
        // store as (a,b) where a=sideId(u,α), b=sideId(v,β) with a<b
        vector<pair<int,int>> bedges;
        bedges.reserve(4 * snarl_component.size());

        for (int z : snarl_component) {
            for (auto &inc : adj[z]) {
                int w = inc.v;
                if (!in_comp[w]) continue;
                if (!edge_survives_split(z, w, inc.su, inc.sv)) continue;

                int a = sideId(z, inc.su);
                int b = sideId(w, inc.sv);
                if (a > b) std::swap(a,b);
                bedges.emplace_back(a,b);
            }
        }
        sort(bedges.begin(), bedges.end());
        bedges.erase(unique(bedges.begin(), bedges.end()), bedges.end());

        // build directed graph on states (side, used)
        vector<vector<int>> g(ST);
        g.reserve(ST);

        auto add_arc = [&](int fromSide, int fromUsed, int toSide, int toUsed){
            g[stId(fromSide, fromUsed)].push_back(stId(toSide, toUsed));
        };

        for (auto [a,b] : bedges) {

            int ah = oppSide(a);
            int bh = oppSide(b);

            for (int used = 0; used <= 1; ++used) {
                add_arc(a, used, bh, used);
                add_arc(b, used, ah, used);
            }

            add_arc(a, 0, b, 1);
            add_arc(b, 0, a, 1);
        }

        {
            vector<uint8_t> color(ST, 0); 
            function<bool(int)> dfsCyc0 = [&](int u)->bool{
                color[u] = 1;
                for (int v : g[u]) {
                    if ((v & 1) != 0) continue;
                    if (color[v] == 1) return true;
                    if (color[v] == 0 && dfsCyc0(v)) return true;
                }
                color[u] = 2;
                return false;
            };

            for (int side = 0; side < S; ++side) {
                int u = stId(side, 0);
                if (color[u] == 0) {
                    if (dfsCyc0(u)) return true;
                }
            }
        }

        {
            vector<int> vis(ST, 0);
            int it = 1;

            for (int side = 0; side < S; ++side) {
                int start = stId(side, 0);
                if (g[start].empty()) continue;

                int target = stId(side, 1);

                std::stack<int> st;
                st.push(start);
                vis[start] = it;

                while (!st.empty()) {
                    int u = st.top(); st.pop();
                    if (u == target) return true;
                    for (int v : g[u]) {
                        if (vis[v] == it) continue;
                        vis[v] = it;
                        st.push(v);
                    }
                }
                ++it;
            }
        }

        return false;
    };

    const size_t nV = nodes.size();
    const size_t expected_candidates = (nV * (nV - 1) / 2) * 4;
    size_t tested_candidates = 0;

    for (size_t ii = 0; ii < nodes.size(); ii++) {
        int x = nodes[ii];
        for (size_t jj = ii + 1; jj < nodes.size(); jj++) {
            int y = nodes[jj];

            for (int dx = 0; dx < 2; dx++) {
                for (int dy = 0; dy < 2; dy++) {
                    tested_candidates++;

                    bool separable = getSep(x, y, dx, dy);
#if DEBUG_BLOCKS
                    assert(separable == getSep(y, x, dy, dx));
#endif
                    if (!separable) continue;

                    CDX = dx ? '-' : '+';
                    CDY = dy ? '-' : '+';
                    compute_component(x, y, CDX, CDY);

                    fill(in_comp.begin(), in_comp.end(), 0);
                    for (int z : snarl_component) in_comp[z] = 1;

                    // minimality
                    bool not_minimal = false;
                    for (int z : snarl_component) {
                        if (z == x || z == y) continue;
                        if (getSep(x, z, dx, 0) && getSep(z, y, 1, dy)) { not_minimal = true; break; }
                        if (getSep(x, z, dx, 1) && getSep(z, y, 0, dy)) { not_minimal = true; break; }
                    }
                    if (not_minimal) continue;

                    // set split terminals for split-filtered checks
                    CX = x; CY = y;
                    CDX = dx ? '-' : '+';
                    CDY = dy ? '-' : '+';

                    bool internal_tip = has_internal_tip_in_component();
                    bool cyclic = has_cycloid_in_component();

                    if (!internal_tip && !cyclic) {
                        cout << nameOf[x] << (dx ? "-" : "+") << " "
                             << nameOf[y] << (dy ? "-" : "+") << "\n";
                    }
                }
            }
        }
    }

    assert(tested_candidates == expected_candidates);

#if DEBUG_BLOCKS
#ifdef __unix__
    const char* env = std::getenv("RUN_SNARLS_BF");
    const bool do_external = (env && std::string(env) == "1");

    if (do_external) {
        std::set<SnarlKey> snarls_ref;
        bool ok = run_snarls_bf(snarls_bf_path, argv[1], id, snarls_ref);
        if (!ok) {
            cerr << "Could not run ./snarls_bf from same directory or it failed. Skipping external snarl cross check.\n";
        }
    } else {
        cerr << "Skipping external ./snarls_bf cross check (set RUN_SNARLS_BF=1 to enable).\n";
    }
#endif
#endif

    return 0;
}