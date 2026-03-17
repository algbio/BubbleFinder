#include <bits/stdc++.h>
#include <cassert>
#include <cstdint>
#ifdef __unix__
#include <unistd.h>
#include <limits.h>
#include <sys/wait.h>
#endif

using namespace std;

static inline int sgnIdx(char c) { return (c == '-') ? 1 : 0; }

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
    if (x < y) return {x, y, dx, dy};
    if (x > y) return {y, x, dy, dx};
    if (dx <= dy) return {x, y, dx, dy};
    return {y, x, dy, dx};
}

static inline string fmt_key(const SnarlKey& k, const vector<string>& nameOf) {
    return nameOf[k.a] + string(1, k.da) + " " + nameOf[k.b] + string(1, k.db);
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

static bool run_external_tool(const std::string& cmd,
                              const std::unordered_map<std::string,int>& id,
                              std::set<SnarlKey>& out) {
    out.clear();
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;

    char buf[1<<15];
    while (fgets(buf, sizeof(buf), pipe)) {
        std::string line(buf);
        if (line.find_first_not_of("0123456789 \t\r\n") == std::string::npos)
            continue;

        std::stringstream ss(line);
        std::string t1, t2;
        ss >> t1 >> t2;
        if (!ss) continue;

        int x, y; char dx, dy;
        if (!parse_side_token(t1, id, x, dx)) continue;
        if (!parse_side_token(t2, id, y, dy)) continue;

        out.insert(canon_snarl(x, dx, y, dy));
    }

    int status = pclose(pipe);
    if (status == -1) return false;
    if (!WIFEXITED(status)) return false;
    return WEXITSTATUS(status) == 0;
}
#endif


static bool parse_bf_output(const std::string& path,
                            const std::unordered_map<std::string,int>& id,
                            std::set<SnarlKey>& out) {
    out.clear();
    ifstream f(path);
    if (!f) return false;

    string line;
    if (!getline(f, line)) return false;

    while (getline(f, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        vector<pair<int,char>> endpoints;
        string tok;
        while (ss >> tok) {
            int v; char s;
            if (parse_side_token(tok, id, v, s))
                endpoints.push_back({v, s});
        }
        for (size_t i = 0; i < endpoints.size(); i++)
            for (size_t j = i+1; j < endpoints.size(); j++)
                out.insert(canon_snarl(endpoints[i].first, endpoints[i].second,
                                       endpoints[j].first, endpoints[j].second));
    }
    return true;
}


struct IncEdge {
    int v;
    char su; 
    char sv; 
};

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string bf_ub_file, bf_snarls_file;
    string gfa_path;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--bf-ub" && i+1 < argc) {
            bf_ub_file = argv[++i];
        } else if (arg == "--bf-snarls" && i+1 < argc) {
            bf_snarls_file = argv[++i];
        } else if (gfa_path.empty()) {
            gfa_path = arg;
        } else {
            cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    if (gfa_path.empty()) {
        cerr << "Usage: ./ultrabubbles_bf_consolidated input.gfa "
                "[--bf-ub BF_UB_OUTPUT] [--bf-snarls BF_SNARLS_OUTPUT]\n";
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

    ifstream fin(gfa_path);
    if (!fin) { cerr << "Cannot open " << gfa_path << "\n"; return 1; }

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

            char su = oa;
            char sv = (ob == '-') ? '+' : '-';
            adj[u].push_back({v, su, sv});
            adj[v].push_back({u, sv, su});
        }
    }
    fin.close();

    int N = (int)nameOf.size();
    ensureN(N);

    vector<int> nodes;
    nodes.reserve(N);
    for (int v = 0; v < N; v++) if (seenSegment[v]) nodes.push_back(v);
    sort(nodes.begin(), nodes.end());

    cerr << "Vertices: " << nodes.size() << "\n";

    const int REACH_FLAG = 1;
    const int BAD_FLAG   = 2;

    vector<int> seen(N, 0);
    int dfs_it = 1;
    vector<int> snarl_component;

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
        auto oppSide = [&](int side)->int { return side ^ 1; };

        const int S = 2 * N;
        const int ST = 2 * S;
        auto stId = [&](int side, int used)->int { return (side<<1) | used; };

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

        vector<vector<int>> g(ST);

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
                if (color[u] == 0 && dfsCyc0(u)) return true;
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
                bool found = false;

                while (!st.empty()) {
                    int u = st.top(); st.pop();
                    if (u == target) { found = true; break; }
                    for (int v : g[u]) {
                        if (vis[v] == it) continue;
                        vis[v] = it;
                        st.push(v);
                    }
                }
                if (found) return true;
                ++it;
            }
        }

        return false;
    };


    set<SnarlKey> bf_snarls_set;    
    set<SnarlKey> bf_ub_set;        

    set<SnarlKey> snarls_with_tips;  
    set<SnarlKey> snarls_with_cycle; 

    const size_t nV = nodes.size();
    size_t tested = 0;

    cerr << "Enumerating all snarls and ultrabubbles...\n";

    for (size_t ii = 0; ii < nodes.size(); ii++) {
        int x = nodes[ii];
        for (size_t jj = ii + 1; jj < nodes.size(); jj++) {
            int y = nodes[jj];
            for (int dx = 0; dx < 2; dx++) {
                for (int dy = 0; dy < 2; dy++) {
                    tested++;

                    bool separable = getSep(x, y, dx, dy);
                    assert(separable == getSep(y, x, dy, dx));
                    if (!separable) continue;

                    CDX = dx ? '-' : '+';
                    CDY = dy ? '-' : '+';
                    compute_component(x, y, CDX, CDY);

                    fill(in_comp.begin(), in_comp.end(), 0);
                    for (int z : snarl_component) in_comp[z] = 1;

                    bool not_minimal = false;
                    for (int z : snarl_component) {
                        if (z == x || z == y) continue;
                        if (getSep(x, z, dx, 0) && getSep(z, y, 1, dy)) { not_minimal = true; break; }
                        if (getSep(x, z, dx, 1) && getSep(z, y, 0, dy)) { not_minimal = true; break; }
                    }
                    if (not_minimal) continue;

                    SnarlKey key = canon_snarl(x, dx ? '-' : '+', y, dy ? '-' : '+');
                    bf_snarls_set.insert(key);

                    CX = x; CY = y;
                    CDX = dx ? '-' : '+';
                    CDY = dy ? '-' : '+';

                    bool has_tip = has_internal_tip_in_component();
                    bool has_cycle = has_cycloid_in_component();

                    if (has_tip)   snarls_with_tips.insert(key);
                    if (has_cycle) snarls_with_cycle.insert(key);

                    if (!has_tip && !has_cycle) {
                        bf_ub_set.insert(key);
                        cout << nameOf[key.a] << key.da << " "
                             << nameOf[key.b] << key.db << "\n";
                    }
                }
            }
        }
    }

    assert(tested == (nV * (nV - 1) / 2) * 4);

    int internal_failures = 0;

    cerr << "\n";
    cerr << "=== Brute-force summary ===\n";
    cerr << "  Snarls found:        " << bf_snarls_set.size() << "\n";
    cerr << "  Ultrabubbles found:  " << bf_ub_set.size() << "\n";
    cerr << "  Snarls with tips:    " << snarls_with_tips.size() << "\n";
    cerr << "  Snarls with cycles:  " << snarls_with_cycle.size() << "\n";

    {
        int violations = 0;
        for (auto& k : bf_ub_set) {
            if (bf_snarls_set.find(k) == bf_snarls_set.end()) {
                cerr << "  BUG: ultrabubble " << fmt_key(k, nameOf)
                     << " is NOT in snarls set!\n";
                violations++;
            }
        }
        if (violations == 0)
            cerr << "  CHECK OK: all ultrabubbles are snarls\n";
        else {
            cerr << "  CHECK FAILED: " << violations
                 << " ultrabubbles missing from snarls!\n";
            internal_failures += violations;
        }
    }

    {
        int uncategorized = 0;
        for (auto& k : bf_snarls_set) {
            bool is_ub = bf_ub_set.count(k);
            bool has_t = snarls_with_tips.count(k);
            bool has_c = snarls_with_cycle.count(k);
            if (!is_ub && !has_t && !has_c) {
                cerr << "  BUG: snarl " << fmt_key(k, nameOf)
                     << " is neither UB nor has tip/cycle!\n";
                uncategorized++;
            }
        }
        if (uncategorized == 0)
            cerr << "  CHECK OK: all snarls categorized (UB / tip / cycle)\n";
        else {
            cerr << "  CHECK FAILED: " << uncategorized
                 << " snarls uncategorized!\n";
            internal_failures += uncategorized;
        }
    }

#ifdef __unix__
    if (!snarls_bf_path.empty() && access(snarls_bf_path.c_str(), X_OK) == 0) {
        cerr << "\n=== Cross-check with snarls_bf ===\n";

        set<SnarlKey> ext_snarls;
        string cmd = "'" + snarls_bf_path + "' '" + gfa_path + "'";
        bool ok = run_external_tool(cmd, id, ext_snarls);

        if (ok) {
            cerr << "  snarls_bf found: " << ext_snarls.size() << " snarls\n";

            int missing_from_ext = 0, extra_in_ext = 0;
            for (auto& k : bf_snarls_set) {
                if (!ext_snarls.count(k)) {
                    if (missing_from_ext < 10)
                        cerr << "    BF-only snarl: " << fmt_key(k, nameOf) << "\n";
                    missing_from_ext++;
                }
            }
            for (auto& k : ext_snarls) {
                if (!bf_snarls_set.count(k)) {
                    if (extra_in_ext < 10)
                        cerr << "    snarls_bf-only snarl: " << fmt_key(k, nameOf) << "\n";
                    extra_in_ext++;
                }
            }
            cerr << "  In brute-force but not snarls_bf: " << missing_from_ext << "\n";
            cerr << "  In snarls_bf but not brute-force:  " << extra_in_ext << "\n";

            if (missing_from_ext == 0 && extra_in_ext == 0)
                cerr << "  CHECK OK: snarl sets match perfectly\n";
            else {
                cerr << "  CHECK FAILED: snarl sets differ\n";
                internal_failures += missing_from_ext + extra_in_ext;
            }
        } else {
            cerr << "  snarls_bf not found or failed, skipping\n";
        }
    }
#endif


    if (!bf_ub_file.empty()) {
        cerr << "\n=== Cross-check with BubbleFinder ultrabubbles ===\n";

        set<SnarlKey> bf_tool_ub;
        if (parse_bf_output(bf_ub_file, id, bf_tool_ub)) {
            cerr << "  BubbleFinder UBs: " << bf_tool_ub.size() << "\n";
            cerr << "  Brute-force UBs:  " << bf_ub_set.size() << "\n";

            int fn = 0, fp = 0; 
            for (auto& k : bf_ub_set) {
                if (!bf_tool_ub.count(k)) {
                    if (fn < 20)
                        cerr << "    MISSED by BubbleFinder: " << fmt_key(k, nameOf) << "\n";
                    fn++;
                }
            }
            for (auto& k : bf_tool_ub) {
                if (!bf_ub_set.count(k)) {
                    if (fp < 20)
                        cerr << "    EXTRA in BubbleFinder:  " << fmt_key(k, nameOf) << "\n";
                    fp++;
                }
            }

            cerr << "  False negatives (BF missed): " << fn << "\n";
            cerr << "  False positives (BF extra):  " << fp << "\n";

            if (fn == 0 && fp == 0)
                cerr << "  CHECK OK: ultrabubble sets match perfectly\n";
            else
                cerr << "  CHECK FAILED: sets differ\n";
        } else {
            cerr << "  Could not parse BubbleFinder UB file: " << bf_ub_file << "\n";
        }
    }

    if (!bf_snarls_file.empty()) {
        cerr << "\n=== Cross-check with BubbleFinder snarls ===\n";

        set<SnarlKey> bf_tool_snarls;
        if (parse_bf_output(bf_snarls_file, id, bf_tool_snarls)) {
            cerr << "  BubbleFinder snarls: " << bf_tool_snarls.size() << "\n";
            cerr << "  Brute-force snarls:  " << bf_snarls_set.size() << "\n";

            int fn = 0, fp = 0;
            for (auto& k : bf_snarls_set) {
                if (!bf_tool_snarls.count(k)) {
                    if (fn < 20)
                        cerr << "    MISSED by BubbleFinder: " << fmt_key(k, nameOf) << "\n";
                    fn++;
                }
            }
            for (auto& k : bf_tool_snarls) {
                if (!bf_snarls_set.count(k)) {
                    if (fp < 20)
                        cerr << "    EXTRA in BubbleFinder:  " << fmt_key(k, nameOf) << "\n";
                    fp++;
                }
            }

            cerr << "  False negatives (BF missed): " << fn << "\n";
            cerr << "  False positives (BF extra):  " << fp << "\n";

            if (fn == 0 && fp == 0)
                cerr << "  CHECK OK: snarl sets match perfectly\n";
            else
                cerr << "  CHECK FAILED: sets differ\n";


            if (!bf_ub_file.empty()) {
                set<SnarlKey> bf_tool_ub;
                if (parse_bf_output(bf_ub_file, id, bf_tool_ub)) {
                    int missed_ub_in_snarls = 0;
                    for (auto& k : bf_tool_snarls) {
                        if (bf_ub_set.count(k) && !bf_tool_ub.count(k)) {
                            if (missed_ub_in_snarls < 20)
                                cerr << "    BF snarl is true UB but BF-UB missed it: "
                                     << fmt_key(k, nameOf) << "\n";
                            missed_ub_in_snarls++;
                        }
                    }
                    cerr << "  BF snarls that are true UBs but missed by BF-UB: "
                         << missed_ub_in_snarls << "\n";
                }
            }
        } else {
            cerr << "  Could not parse BubbleFinder snarls file: " << bf_snarls_file << "\n";
        }
    }

    cerr << "\nDone.\n";
    if (internal_failures > 0) {
        cerr << "INTERNAL CHECK FAILURES: " << internal_failures << "\n";
        return 2;
    }
    return 0;
}